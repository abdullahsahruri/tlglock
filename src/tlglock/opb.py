"""
Pseudo-Boolean (OPB) encoding of threshold networks, for MiniSAT+.

Why pseudo-Boolean rather than CNF. A threshold gate is *natively* a linear
constraint over binary variables, so it maps to one PB constraint per
polarity. Forcing it into CNF requires enumerating the halfspace's minimal
true points, and a threshold function on n inputs can have exponentially many
of them -- this is the "cannot be expressed in polynomial-size CNF" claim in
the introduction, and it is the structural reason TLG locking resists
CNF-based SAT attacks. Encoding in OPB is the honest baseline: it gives the
attacker the best representation available, so a timeout under PB is a
stronger result than a timeout under a blown-up CNF.

OPB syntax used here (MiniSAT+ / PB competition dialect):

    * #variable= N #constraint= M
    +3 x1 +2 x2 -2 x3 >= 7;

All constraints are normalised to >=, since <= is just negation.
Variables are numbered x1..xN; `OpbEncoder.var_map` records the mapping back
to signal names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .thfile import ThGate, ThNetwork

Term = tuple[int, int]  # (coefficient, variable index)


@dataclass
class Constraint:
    """A single PB constraint: sum(coeff * var) >= rhs."""

    terms: list[Term]
    rhs: int
    comment: str = ""

    def to_line(self) -> str:
        body = " ".join(f"{c:+d} x{v}" for c, v in self.terms)
        return f"{body} >= {self.rhs};"


@dataclass
class OpbEncoder:
    """Builds an OPB instance incrementally."""

    var_map: dict[str, int] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    _next: int = 1

    # -- variables ----------------------------------------------------------

    def var(self, name: str) -> int:
        """Get (or create) the OPB index for a named signal."""
        if name not in self.var_map:
            self.var_map[name] = self._next
            self._next += 1
        return self.var_map[name]

    def fresh(self, hint: str = "t") -> int:
        """Allocate an auxiliary variable."""
        name = f"__{hint}{self._next}"
        return self.var(name)

    @property
    def num_vars(self) -> int:
        return self._next - 1

    # -- constraints --------------------------------------------------------

    def add(self, terms: Sequence[Term], rhs: int, comment: str = "") -> None:
        self.constraints.append(Constraint(list(terms), rhs, comment))

    def fix(self, name: str, value: int) -> None:
        """Pin a signal to a constant."""
        if value not in (0, 1):
            raise ValueError(f"cannot fix '{name}' to non-binary {value}")
        v = self.var(name)
        if value == 1:
            self.add([(1, v)], 1, f"{name} = 1")
        else:
            self.add([(-1, v)], 0, f"{name} = 0")

    # -- gate reification ---------------------------------------------------

    def encode_gate(self, gate: ThGate, rename: dict[str, str] | None = None) -> None:
        """
        Encode  y <-> (sum_i w_i x_i >= T)  as two PB constraints.

        Let S = sum_i w_i x_i, with
            Smin = sum_i min(0, w_i)      (all negative weights active)
            Smax = sum_i max(0, w_i)      (all positive weights active)

        Forward,  y = 1 -> S >= T:
            S - (T - Smin) * y  >=  Smin
          y=1 gives S >= T; y=0 gives S >= Smin, vacuous.

        Reverse,  y = 0 -> S <= T - 1, written as >= after negating:
            -S + (Smax - T + 1) * y  >=  1 - T
          y=0 gives S <= T-1; y=1 gives S <= Smax, vacuous.

        Note both slack coefficients must be computed from the true bounds,
        not from sum|w_i|: with mixed-sign weights (which "balanced" key
        assignment produces on every locked gate) those differ, and using
        the symmetric bound silently makes one direction vacuous. The
        brute-force check in tests/test_opb.py exists to catch exactly that.
        """
        rename = rename or {}

        def sig(n: str) -> str:
            return rename.get(n, n)

        y = self.var(sig(gate.output))
        s_terms = [(w, self.var(sig(n))) for n, w in zip(gate.inputs, gate.weights)]

        smin = sum(w for w in gate.weights if w < 0)
        smax = sum(w for w in gate.weights if w > 0)
        T = gate.threshold

        # Forward implication.
        self.add(
            s_terms + [(-(T - smin), y)],
            smin,
            f"{gate.output}=1 -> sum >= {T}",
        )
        # Reverse implication.
        self.add(
            [(-c, v) for c, v in s_terms] + [(smax - T + 1, y)],
            1 - T,
            f"{gate.output}=0 -> sum <= {T - 1}",
        )

    def encode_network(
        self,
        net: ThNetwork,
        suffix: str = "",
        share: Iterable[str] = (),
        rename: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Encode every gate of `net`, renaming signals for this copy.

        By default signals get `suffix` appended and anything in `share` keeps
        its original name, which is the miter construction: two copies reading
        the same primary inputs through independent keys.

        An explicit `rename` overrides both for the signals it names. The SAT
        attack needs that: each recorded distinguishing input adds a circuit
        copy whose internal signals are private but whose key variables must
        be the *same* variables as in the main miter, or the accumulated
        constraints would not actually constrain the key being solved for.
        """
        shared = set(share)
        mapping: dict[str, str] = {}
        if suffix:
            for s in net.signals:
                if s not in shared:
                    mapping[s] = f"{s}{suffix}"
        if rename:
            mapping.update(rename)

        for gate in net.topological_order():
            self.encode_gate(gate, rename=mapping)
        return mapping

    # -- XOR / difference ---------------------------------------------------

    def encode_xor(self, a: int, b: int, d: int) -> None:
        """
        Encode d <-> (a != b) for binary a, b, d.

        d >= a - b ;  d >= b - a ;  d <= a + b ;  d <= 2 - a - b
        """
        self.add([(1, d), (-1, a), (1, b)], 0, "d >= a - b")
        self.add([(1, d), (1, a), (-1, b)], 0, "d >= b - a")
        self.add([(-1, d), (1, a), (1, b)], 0, "d <= a + b")
        self.add([(-1, d), (-1, a), (-1, b)], -2, "d <= 2 - a - b")

    # -- output -------------------------------------------------------------

    def to_text(self, objective: Sequence[Term] | None = None) -> str:
        lines = [
            f"* #variable= {self.num_vars} #constraint= {len(self.constraints)}"
        ]
        if objective:
            body = " ".join(f"{c:+d} x{v}" for c, v in objective)
            lines.append(f"min: {body};")
        for c in self.constraints:
            if c.comment:
                lines.append(f"* {c.comment}")
            lines.append(c.to_line())
        return "\n".join(lines) + "\n"

    def write(self, path: str, objective: Sequence[Term] | None = None) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_text(objective))


# -- miter construction -----------------------------------------------------


def build_distinguishing_miter(
    locked: ThNetwork, key_names: Sequence[str]
) -> OpbEncoder:
    """
    Build the SAT-attack miter: find a data input x and two key vectors
    k_A, k_B such that the locked circuit disagrees on some output.

        exists x, k_A, k_B :  L(x, k_A) != L(x, k_B)

    A satisfying assignment is a distinguishing input pattern (DIP); the
    oracle-guided attack queries the activated chip on x and adds the
    response as a new constraint, iterating until the miter is UNSAT. At
    that point every surviving key is functionally equivalent to k*.

    This function builds one iteration's formula. The outer loop lives in
    attack.py (not yet implemented).
    """
    keys = set(key_names)
    data_inputs = [n for n in locked.inputs if n not in keys]

    enc = OpbEncoder()
    for n in data_inputs:
        enc.var(n)

    enc.encode_network(locked, suffix="_A", share=data_inputs)
    enc.encode_network(locked, suffix="_B", share=data_inputs)

    diffs = []
    for o in locked.outputs:
        a = enc.var(f"{o}_A")
        b = enc.var(f"{o}_B")
        d = enc.fresh(f"diff_{o}_")
        enc.encode_xor(a, b, d)
        diffs.append(d)

    enc.add([(1, d) for d in diffs], 1, "at least one output differs")
    return enc


# add_oracle_constraint() lived here. It linked each oracle copy's key
# variables to a set of "shared" key variables that the miter never actually
# used -- the miter names its keys k_A and k_B, so the constraints it added
# constrained nothing. The correct construction needs one oracle copy per key
# set per recorded input, which is more context than a single function should
# own; it now lives in attack.py as _encode_oracle_copy().
