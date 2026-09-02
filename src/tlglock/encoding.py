"""
How many equations does a locked TLG network actually create?

Three different numbers answer that, and they disagree by orders of magnitude,
so a claim about "clause blowup" means nothing until it says which encoding it
is talking about. This module measures all three for the same network.

  PB (OPB)          Two constraints per gate, always. A threshold gate *is* a
                    linear constraint, so nothing has to be expanded. This is
                    what opb.py emits and what the attack in attack.py solves.

  CNF, no aux vars  Encoding y <-> [sum w_i x_i >= T] in clauses over x and y
                    alone requires one clause per prime implicant of f and one
                    per prime implicant of ~f. For a threshold function those
                    counts are the minimal true points and maximal false
                    points, and for k-of-n they are exactly C(n,k) and
                    C(n,k-1) -- exponential at k ~ n/2. This is the "not
                    expressible in polynomial-size CNF" claim.

  CNF, with aux     A sequential weighted counter (Hoelldobler & Manthey 2012,
                    and totalizer / BDD encodings likewise) encodes the same
                    constraint in O(n*T) clauses and O(n*T) auxiliary
                    variables. Polynomial. Every practical SAT front end uses
                    an encoding of this family.

The third row is the one that matters for a resistance argument, and it is
easy to miss: the exponential blowup is a property of *CNF over the original
variables*, not of threshold functions. An attacker chooses the encoding, so a
scheme cannot rest its security on the attacker picking the bad one. opb.py
already takes this position -- it hands the attacker the good representation
deliberately, so that a timeout means something.

Separately, `key_equations()` counts the equations the *locking* creates
rather than the ones the *encoding* creates: each distinct value of
sum_j v_j k_j selects a distinct effective threshold, so the key space
partitions into that many equivalence classes. That is the K_s partition, and
its size is the real key space as far as an attacker is concerned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .separable import _flip_table, gate_to_table
from .thfile import ThGate, ThNetwork


def _prime_implicant_counts(pos: Sequence[int], n: int) -> tuple[int, int]:
    """
    (minimal true points, maximal false points) of a *monotone* truth table.

    A true point of a monotone function is minimal exactly when clearing any
    one of its set bits makes it false; a false point is maximal exactly when
    setting any one of its clear bits makes it true. That is O(2^n * n).

    separable.py computes the same two sets by pairwise domination, which is
    O(|T|^2 * n). That is the right trade there -- it runs on instances already
    reduced to a handful of points, and the explicit point lists are what the
    LP needs. Here only the counts are wanted, at fan-ins where the quadratic
    form does not finish.
    """
    bits = [1 << j for j in range(n)]
    minimal = maximal = 0
    for idx in range(1 << n):
        if pos[idx]:
            if all(not pos[idx ^ b] for b in bits if idx & b):
                minimal += 1
        elif all(pos[idx | b] for b in bits if not idx & b):
            maximal += 1
    return minimal, maximal

# Above this fan-in, the direct-CNF count needs a 2^n truth table and is not
# computed. The count is exponential anyway; the point is already made.
DEFAULT_MAX_FANIN = 16


@dataclass
class GateEncoding:
    """Encoding size for one reified threshold gate."""

    output: str
    fanin: int
    threshold: int
    pb_constraints: int = 2
    cnf_direct: int | None = None       # None when fan-in exceeded the cap
    cnf_aux_clauses: int = 0
    cnf_aux_vars: int = 0

    @property
    def blowup(self) -> float | None:
        """Direct-CNF clauses per PB constraint."""
        if self.cnf_direct is None:
            return None
        return self.cnf_direct / self.pb_constraints


@dataclass
class NetworkEncoding:
    """Encoding size for a whole network, plus the attack formula it feeds."""

    gates: list[GateEncoding] = field(default_factory=list)
    pb_vars: int = 0
    pb_constraints: int = 0
    miter_pb_vars: int = 0
    miter_pb_constraints: int = 0
    pb_constraints_per_dip: int = 0
    truncated: list[str] = field(default_factory=list)

    @property
    def cnf_direct(self) -> int | None:
        """Total direct-CNF clauses, or None if any gate was too wide."""
        if self.truncated:
            return None
        return sum(g.cnf_direct or 0 for g in self.gates)

    @property
    def cnf_aux_clauses(self) -> int:
        return sum(g.cnf_aux_clauses for g in self.gates)

    @property
    def cnf_aux_vars(self) -> int:
        return sum(g.cnf_aux_vars for g in self.gates)

    @property
    def widest(self) -> int:
        return max((g.fanin for g in self.gates), default=0)


def gate_encoding(gate: ThGate, max_fanin: int = DEFAULT_MAX_FANIN) -> GateEncoding:
    """
    Measure one gate under all three encodings.

    The direct-CNF count is exact, computed by enumerating prime implicants.
    Because `sum w_i x_i >= T` is unate in every variable -- with polarity
    sign(w_i) -- flipping the negative-weight variables makes the function
    monotone, and then its prime implicants are exactly the minimal true
    points. That is the same reduction separable.py uses to shrink the LP.

    The auxiliary-variable figures are the standard sequential weighted
    counter's O(n*T) growth. Exact constants vary by variant; what is being
    reported is that this column is polynomial while the other is not.
    """
    n = gate.fanin
    result = GateEncoding(
        output=gate.output, fanin=n, threshold=gate.threshold
    )
    if n == 0:
        result.cnf_direct = 0
        return result

    # Normalise to non-negative weights: x_i -> 1 - x_i for w_i < 0 raises the
    # bound by |w_i|, which is the bound the counter encoding actually builds.
    negative = sum(w for w in gate.weights if w < 0)
    bound = gate.threshold - negative
    bound = max(0, min(bound, sum(abs(w) for w in gate.weights)))
    result.cnf_aux_vars = n * bound
    result.cnf_aux_clauses = 2 * n * bound + n

    if n > max_fanin:
        return result

    table = gate_to_table(gate)
    flips = [i for i, w in enumerate(gate.weights) if w < 0]
    pos = _flip_table(table, n, flips) if flips else list(table)
    minimal, maximal = _prime_implicant_counts(pos, n)
    result.cnf_direct = minimal + maximal
    return result


def network_encoding(
    net: ThNetwork,
    key_names: Sequence[str] = (),
    max_fanin: int = DEFAULT_MAX_FANIN,
) -> NetworkEncoding:
    """
    Measure a whole network, and the attack formula built from it.

    `miter_*` describe one iteration of the distinguishing-input formula, and
    `pb_constraints_per_dip` is how much it grows per recorded oracle query --
    the attack instance grows linearly in DIP count, so these two together
    give the size of iteration i.
    """
    from .attack import build_attack_formula
    from .opb import OpbEncoder

    report = NetworkEncoding()
    for gate in net.gates:
        ge = gate_encoding(gate, max_fanin=max_fanin)
        report.gates.append(ge)
        if ge.cnf_direct is None:
            report.truncated.append(gate.output)

    enc = OpbEncoder()
    for name in net.inputs:
        enc.var(name)
    enc.encode_network(net)
    report.pb_vars = enc.num_vars
    report.pb_constraints = len(enc.constraints)

    if key_names:
        base = build_attack_formula(net, key_names, [])
        report.miter_pb_vars = base.num_vars
        report.miter_pb_constraints = len(base.constraints)

        pattern = {n: 0 for n in net.inputs if n not in set(key_names)}
        one = build_attack_formula(net, key_names, [(pattern, tuple(
            0 for _ in net.outputs
        ))])
        report.pb_constraints_per_dip = (
            len(one.constraints) - len(base.constraints)
        )
    return report


@dataclass
class KeyEquations:
    """The K_s partition induced by one gate's key weights."""

    gate: str
    key_weights: list[int]
    key_space: int
    distinct_equations: int
    class_sizes: dict[int, int] = field(default_factory=dict)

    @property
    def compression(self) -> float:
        """Key space divided by the number of distinct functions it selects."""
        return self.key_space / self.distinct_equations


def key_equations(
    gate: ThGate, key_names: Sequence[str]
) -> KeyEquations:
    """
    How many distinct equations does the key actually select on this gate?

    Every key vector k contributes a shift sum_j v_j k_j, and the gate becomes
    `sum_i w_i x_i >= T - shift`. Two keys giving the same shift give the
    identical inequality, so the number of *distinct* shifts is the number of
    distinct gates the key space can select -- not 2^n_k.

    This is the K_s partition of the equivalence-class analysis, and the gap
    between key_space and distinct_equations is the compression. It bounds
    key-space compression from above per gate: keys that differ only in ways
    that cancel are indistinguishable to any attack, exact or approximate.
    """
    weights = [gate.weight_of(k) for k in key_names if k in gate.inputs]
    counts: dict[int, int] = {0: 1}
    for v in weights:
        nxt: dict[int, int] = {}
        for s, c in counts.items():
            nxt[s] = nxt.get(s, 0) + c
            nxt[s + v] = nxt.get(s + v, 0) + c
        counts = nxt
    return KeyEquations(
        gate=gate.output,
        key_weights=weights,
        key_space=1 << len(weights),
        distinct_equations=len(counts),
        class_sizes=dict(sorted(counts.items())),
    )


def format_report(
    net: ThNetwork,
    key_names: Sequence[str] = (),
    max_fanin: int = DEFAULT_MAX_FANIN,
) -> str:
    """Human-readable summary, for the CLI and for pasting into a discussion."""
    r = network_encoding(net, key_names=key_names, max_fanin=max_fanin)
    lines = [
        f"network {net.model}: {len(net.gates)} gates, "
        f"{len(net.inputs)} inputs, {len(net.outputs)} outputs, "
        f"widest fan-in {r.widest}",
        "",
        f"{'gate':<14}{'fanin':>6}{'T':>5}{'PB':>6}"
        f"{'CNF direct':>13}{'CNF aux cls':>13}{'aux vars':>10}",
    ]
    for g in r.gates:
        direct = "n/a" if g.cnf_direct is None else f"{g.cnf_direct:,}"
        lines.append(
            f"{g.output:<14}{g.fanin:>6}{g.threshold:>5}{g.pb_constraints:>6}"
            f"{direct:>13}{g.cnf_aux_clauses:>13,}{g.cnf_aux_vars:>10,}"
        )

    total_direct = "n/a" if r.cnf_direct is None else f"{r.cnf_direct:,}"
    lines += [
        "",
        f"PB encoding      {r.pb_constraints:,} constraints over {r.pb_vars:,} variables",
        f"CNF, no aux      {total_direct} clauses over {r.pb_vars:,} variables",
        f"CNF, with aux    {r.cnf_aux_clauses:,} clauses, "
        f"+{r.cnf_aux_vars:,} auxiliary variables",
    ]
    if r.truncated:
        lines.append(
            f"  ({len(r.truncated)} gate(s) over fan-in {max_fanin} not "
            f"expanded: {', '.join(r.truncated[:5])})"
        )

    if key_names:
        lines += [
            "",
            f"attack miter     {r.miter_pb_constraints:,} constraints over "
            f"{r.miter_pb_vars:,} variables",
            f"  per recorded DIP  +{r.pb_constraints_per_dip:,} constraints",
            "",
            f"{'gate':<14}{'key bits':>10}{'key space':>12}"
            f"{'distinct eqs':>14}{'compression':>13}",
        ]
        keys = set(key_names)
        for gate in net.gates:
            if not (keys & set(gate.inputs)):
                continue
            ke = key_equations(gate, key_names)
            lines.append(
                f"{ke.gate:<14}{len(ke.key_weights):>10}{ke.key_space:>12,}"
                f"{ke.distinct_equations:>14,}{ke.compression:>12.1f}x"
            )
    return "\n".join(lines)
