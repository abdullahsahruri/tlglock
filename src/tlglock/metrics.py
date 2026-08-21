"""
Security and correctness metrics.

Corruption rate is defined in Section IV-B as "the fraction of primary-output
mismatches averaged over all incorrect keys". Concretely, for a locked network
L with correct key k*, original network C, input space X and wrong-key set
K' = {0,1}^m \\ {k*}:

    rho_c  =  1/(|K'| |X|)  *  sum_{k' in K'} sum_{x in X} HD(L(x,k'), C(x)) / |Y|

where HD is Hamming distance over the |Y| primary outputs. A scheme with
rho_c near 0.5 corrupts half the output bits on a typical wrong key, which is
the practical maximum -- rho_c = 1.0 would mean every wrong key inverts every
output, itself an exploitable signature.

Both spaces are exponential, so every routine here takes a sampling path and
switches to it automatically past a size threshold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .locking import LockReport, VALID_MODES, lock
from .sim import outputs_of
from .thfile import ThNetwork

# Above this many patterns, sample instead of enumerating.
EXHAUSTIVE_LIMIT = 1 << 14


def _sample_or_enumerate(
    names: Sequence[str],
    limit: int,
    rng: random.Random,
    exclude: Mapping[str, int] | None = None,
) -> list[dict[str, int]]:
    """Enumerate all assignments if small, else draw `limit` random ones."""
    n = len(names)
    if n == 0:
        return [{}]

    excl = tuple(exclude[k] for k in names) if exclude else None

    if n <= 30 and (1 << n) <= limit:
        rows = []
        for bits in product((0, 1), repeat=n):
            if excl is not None and bits == excl:
                continue
            rows.append(dict(zip(names, bits)))
        return rows

    rows = []
    seen = set()
    attempts = 0
    while len(rows) < limit and attempts < limit * 4:
        attempts += 1
        bits = tuple(rng.randint(0, 1) for _ in names)
        if excl is not None and bits == excl:
            continue
        if bits in seen:
            continue
        seen.add(bits)
        rows.append(dict(zip(names, bits)))
    return rows


def is_equivalent_under_correct_key(
    original: ThNetwork,
    report: LockReport,
    limit: int = EXHAUSTIVE_LIMIT,
    seed: int = 0,
) -> bool:
    """
    Check that the locked network reproduces the original under k*.

    This is the invariant the paper's Eq. (6) example violates. Exhaustive
    when the data-input space is small enough, sampled otherwise -- so a False
    is conclusive, while a True is conclusive only in the exhaustive regime.
    """
    rng = random.Random(seed)
    locked = report.locked_network
    data_inputs = [n for n in locked.inputs if n not in report.correct_key]

    if list(original.outputs) != list(locked.outputs):
        return False

    for assign in _sample_or_enumerate(data_inputs, limit, rng):
        ref = outputs_of(original, assign)
        got = outputs_of(locked, {**assign, **report.correct_key})
        if ref != got:
            return False
    return True


def corruption_rate(
    original: ThNetwork,
    report: LockReport,
    input_limit: int = 512,
    key_limit: int = 256,
    seed: int = 0,
) -> float:
    """
    Mean fraction of corrupted primary-output bits over incorrect keys.

    Returns a value in [0, 1]. 0.0 means wrong keys never change an output
    (a broken lock); ~0.5 is the practical target.
    """
    rng = random.Random(seed)
    locked = report.locked_network
    n_out = len(locked.outputs)
    if n_out == 0 or not report.key_names:
        return 0.0

    data_inputs = [n for n in locked.inputs if n not in report.correct_key]
    patterns = _sample_or_enumerate(data_inputs, input_limit, rng)
    wrong_keys = _sample_or_enumerate(
        report.key_names, key_limit, rng, exclude=report.correct_key
    )
    if not wrong_keys or not patterns:
        return 0.0

    golden = {
        tuple(sorted(a.items())): outputs_of(original, a) for a in patterns
    }

    total = 0
    for kv in wrong_keys:
        for assign in patterns:
            ref = golden[tuple(sorted(assign.items()))]
            got = outputs_of(locked, {**assign, **kv})
            total += sum(1 for a, b in zip(ref, got) if a != b)

    return total / (len(wrong_keys) * len(patterns) * n_out)


def output_hamming_profile(
    original: ThNetwork,
    report: LockReport,
    input_limit: int = 256,
    key_limit: int = 128,
    seed: int = 0,
) -> dict[int, float]:
    """
    Distribution of per-pattern Hamming distance, normalised to a density.

    A lock whose mass sits at 0 is weak; one whose mass sits at n_out is
    strong but structurally obvious. Healthy schemes peak near n_out/2.
    """
    rng = random.Random(seed)
    locked = report.locked_network
    n_out = len(locked.outputs)
    hist = {d: 0 for d in range(n_out + 1)}
    if n_out == 0 or not report.key_names:
        return {d: 0.0 for d in hist}

    data_inputs = [n for n in locked.inputs if n not in report.correct_key]
    patterns = _sample_or_enumerate(data_inputs, input_limit, rng)
    wrong_keys = _sample_or_enumerate(
        report.key_names, key_limit, rng, exclude=report.correct_key
    )
    if not wrong_keys or not patterns:
        return {d: 0.0 for d in hist}

    trials = 0
    for kv in wrong_keys:
        for assign in patterns:
            ref = outputs_of(original, assign)
            got = outputs_of(locked, {**assign, **kv})
            hist[sum(1 for a, b in zip(ref, got) if a != b)] += 1
            trials += 1

    return {d: c / trials for d, c in hist.items()}


def equivalent_key_count(
    original: ThNetwork,
    report: LockReport,
    key_limit: int = 4096,
    input_limit: int = 256,
    seed: int = 0,
) -> int:
    """
    How many keys (including k*) yield the original function.

    Threshold gates admit weight-vector degeneracy, so distinct key vectors
    can land on the same halfspace. Every extra equivalent key shrinks the
    effective key space by a factor, which is the security cost of the
    compression the "equal" weight mode buys.
    """
    rng = random.Random(seed)
    locked = report.locked_network
    data_inputs = [n for n in locked.inputs if n not in report.correct_key]
    patterns = _sample_or_enumerate(data_inputs, input_limit, rng)
    keys = _sample_or_enumerate(report.key_names, key_limit, rng)

    golden = {
        tuple(sorted(a.items())): outputs_of(original, a) for a in patterns
    }

    count = 0
    for kv in keys:
        if all(
            golden[tuple(sorted(a.items()))] == outputs_of(locked, {**a, **kv})
            for a in patterns
        ):
            count += 1
    return count


@dataclass
class SweepRow:
    mode: str
    keys_per_gate: int
    percent: float
    num_keys: int
    corruption: float
    equivalent_keys: int


def key_weight_sweep(
    net: ThNetwork,
    modes: Sequence[str] = VALID_MODES,
    keys_per_gate: Sequence[int] = (1, 2, 3, 4),
    percent: float = 50.0,
    seed: int = 0,
) -> list[SweepRow]:
    """
    Reproduce the Fig. 4 trade-off sweep: corruption vs key weight assignment.

    The paper reports corruption peaking at moderate total key weight rather
    than growing monotonically; this sweep is what tests that claim.
    """
    rows: list[SweepRow] = []
    for mode in modes:
        for kpg in keys_per_gate:
            report = lock(
                net, percent=percent, keys_per_gate=kpg, mode=mode, seed=seed
            )
            rows.append(
                SweepRow(
                    mode=mode,
                    keys_per_gate=kpg,
                    percent=percent,
                    num_keys=report.num_keys,
                    corruption=corruption_rate(net, report, seed=seed),
                    equivalent_keys=equivalent_key_count(net, report, seed=seed),
                )
            )
    return rows
