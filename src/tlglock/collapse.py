"""
TLG collapse and merge -- step 4 of Fig. 3.

Two adjacent threshold gates can sometimes be replaced by a single one. The
paper cites Lee et al. [23] for the analytic treatment; the implementation
here is the constructive form of the same question. Given

    y = 1[b.z >= T2]        (the driver)
    f = 1[a.x + w_y*y >= T1] (the driven gate)

substitute y and ask whether the composed function over x union z is itself a
threshold function. That is precisely the test in separable.py, so collapse
reuses it rather than deriving merge conditions by hand.

Doing it constructively rather than analytically has one clear advantage: the
merged gate is verified against the composed truth table before it is
accepted, so a merge can never silently change the function. The cost is that
the support of the merged gate has to be small enough to tabulate, which caps
the merge at `max_support` inputs.

Why collapse matters for TLGLock specifically: merging reduces the gate count,
and the key count in Table I is a percentage of the gate count. Reproducing
that column therefore needs this pass, not just the mapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from .separable import identify, truth_bits
from .thfile import ThGate, ThNetwork

DEFAULT_MAX_SUPPORT = 8


@dataclass
class CollapseStats:
    attempts: int = 0
    merges: int = 0
    gates_before: int = 0
    gates_after: int = 0
    depth_before: int = 0
    depth_after: int = 0
    rejected_binate: int = 0
    rejected_too_wide: int = 0
    merged_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def gate_reduction(self) -> float:
        if not self.gates_before:
            return 0.0
        return 1 - self.gates_after / self.gates_before


def compose(
    driven: ThGate, driver: ThGate, max_support: int = DEFAULT_MAX_SUPPORT
) -> ThGate | None:
    """
    Try to merge `driver` into `driven`, eliminating the intermediate signal.

    Returns the merged gate, or None if the composed function is not linearly
    separable or its support is too wide to tabulate.

    The driver's output must be an input of the driven gate. The caller is
    responsible for checking that the intermediate signal has no other
    consumers -- see can_merge().
    """
    if driver.output not in driven.inputs:
        raise ValueError(
            f"'{driver.output}' is not an input of gate '{driven.output}'"
        )

    keep = [n for n in driven.inputs if n != driver.output]
    support = list(dict.fromkeys(keep + list(driver.inputs)))
    if len(support) > max_support:
        return None

    n = len(support)
    pos = {name: i for i, name in enumerate(support)}

    table = []
    for bits in truth_bits(n):
        values = dict(zip(support, bits))
        y = driver.eval(values)
        merged_values = dict(values)
        merged_values[driver.output] = y
        table.append(driven.eval(merged_values))

    real = identify(table, n)
    if real is None:
        return None

    gate = real.to_gate(support, driven.output)

    # Verify against the composed table rather than trusting identify().
    for bits, want in zip(truth_bits(n), table):
        if gate.eval(dict(zip(support, bits))) != want:
            raise AssertionError(
                f"merged gate '{gate.output}' disagrees with its own "
                f"composition at {bits}"
            )
    return gate


def can_merge(net: ThNetwork, driven: ThGate, driver: ThGate) -> bool:
    """
    Is it safe to absorb `driver` into `driven`?

    Requires the intermediate signal to have exactly one consumer and not to
    be a primary output. Merging a signal with other consumers would duplicate
    the driver's logic rather than remove it, which is the opposite of what
    this pass is for.
    """
    if driver.output in net.outputs:
        return False
    fanout = net.fanout_count()
    return fanout.get(driver.output, 0) == 1


def collapse(
    net: ThNetwork,
    max_support: int = DEFAULT_MAX_SUPPORT,
    max_passes: int = 10,
    max_weight: int | None = None,
) -> tuple[ThNetwork, CollapseStats]:
    """
    Repeatedly merge adjacent gates until no further merge is possible.

    Merges are attempted deepest-first, which tends to pull long chains up
    into single wide gates rather than making one merge that blocks two.

    `max_weight` optionally rejects merges whose weights exceed a bound. Wide
    weight ranges are what make a TLG expensive in silicon -- a CRTL gate's
    capacitor ratios scale with the weight spread -- so a mapper targeting
    real cells wants this set even though the function is realisable without
    it.
    """
    work = net.copy()
    stats = CollapseStats(
        gates_before=len(net.gates), depth_before=_depth(net)
    )

    for _ in range(max_passes):
        merged_any = False
        order = list(reversed(work.topological_order()))

        for driven in order:
            if driven not in work.gates:
                continue
            drivers = {g.output: g for g in work.gates}
            for name in list(driven.inputs):
                driver = drivers.get(name)
                if driver is None or driver is driven:
                    continue
                if not can_merge(work, driven, driver):
                    continue

                stats.attempts += 1
                candidate = compose(driven, driver, max_support=max_support)
                if candidate is None:
                    keep = [n for n in driven.inputs if n != driver.output]
                    width = len(set(keep) | set(driver.inputs))
                    if width > max_support:
                        stats.rejected_too_wide += 1
                    else:
                        stats.rejected_binate += 1
                    continue
                if max_weight is not None and any(
                    abs(w) > max_weight for w in candidate.weights
                ):
                    continue

                idx = work.gates.index(driven)
                work.gates[idx] = candidate
                work.gates.remove(driver)
                stats.merges += 1
                stats.merged_pairs.append((driver.output, driven.output))
                merged_any = True
                break

        if not merged_any:
            break

    work.validate()
    stats.gates_after = len(work.gates)
    stats.depth_after = _depth(work)
    return work, stats


def _depth(net: ThNetwork) -> int:
    level = {n: 0 for n in net.inputs}
    d = 0
    for g in net.topological_order():
        level[g.output] = 1 + max((level.get(i, 0) for i in g.inputs), default=0)
        d = max(d, level[g.output])
    return d


def equivalent(a: ThNetwork, b: ThNetwork, limit: int = 1 << 16) -> bool:
    """
    Exhaustive equivalence check over the shared primary inputs.

    Used to gate the collapse pass in tests. Refuses rather than sampling when
    the input space is too large, so a True is always conclusive.
    """
    if sorted(a.inputs) != sorted(b.inputs) or a.outputs != b.outputs:
        return False
    n = len(a.inputs)
    if (1 << n) > limit:
        raise ValueError(f"{n} inputs is too many for an exhaustive check")

    from .sim import outputs_of

    for bits in product((0, 1), repeat=n):
        assign = dict(zip(a.inputs, bits))
        if outputs_of(a, assign) != outputs_of(b, assign):
            return False
    return True
