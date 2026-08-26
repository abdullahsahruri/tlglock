"""
Threshold identification: is a Boolean function realisable as a single TLG?

This is the feasibility test that replaces "is the cut small enough?" in a
conventional LUT mapper. Where a k-LUT mapper accepts any cut with at most k
inputs, a TLG mapper must ask whether the cut's local function is linearly
separable -- and separability is neither monotone in cut size nor free to
test. That difference is why cut ranking for TLG mapping is an open problem
rather than a solved one; see CLAUDE.md.

The test runs in three stages, cheapest first:

  1. Constants and degenerate variables. Handled directly.
  2. Unateness. Every threshold function is unate in each variable, so a
     binate variable is an immediate reject. This filter kills the large
     majority of non-threshold functions at negligible cost.
  3. Exact LP. The function is made positively unate by flipping the
     negative-unate variables, which lets us restrict the constraint set to
     minimal true points and maximal false points -- for a positive-unate
     function with non-negative weights these dominate all the others. The
     resulting LP has a handful of constraints instead of 2^n.

A rational solution is then scaled by the LCM of its denominators to give
integer weights, which is what the .th format and the hardware both need.
Scaling preserves the constraints: if w.x - T >= 0 and T - w.x >= 1 hold over
the rationals, multiplying through by a positive integer k preserves the
first and strengthens the second to >= k >= 1.

Reference: Neutzling et al., "Effective Logic Synthesis for Threshold Logic
Circuit Design," IEEE TCAD 38(5), 2019 -- reference [24] of the TLGLock paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Sequence

from .lp import Constraint, solve_feasibility
from .thfile import ThGate


@dataclass(frozen=True)
class ThresholdRealisation:
    """Integer weights and threshold realising a function: 1[w.x >= T]."""

    weights: tuple[int, ...]
    threshold: int

    def eval_bits(self, bits: Sequence[int]) -> int:
        return int(sum(w * b for w, b in zip(self.weights, bits)) >= self.threshold)

    def to_gate(self, inputs: Sequence[str], output: str) -> ThGate:
        return ThGate(
            inputs=list(inputs),
            output=output,
            weights=list(self.weights),
            threshold=self.threshold,
        )

    @property
    def max_weight(self) -> int:
        return max((abs(w) for w in self.weights), default=0)


def truth_bits(n: int) -> list[tuple[int, ...]]:
    """All 2^n input patterns, in counting order (MSB = variable 0)."""
    return list(product((0, 1), repeat=n))


def is_unate_in(table: Sequence[int], n: int, var: int) -> int | None:
    """
    Unateness of `table` in `var`.

    Returns +1 positive unate, -1 negative unate, 0 if the variable is not
    used at all, or None if binate (which rules out a threshold function).
    """
    stride = 1 << (n - 1 - var)
    rising = falling = False
    for idx in range(1 << n):
        if idx & stride:
            continue
        lo, hi = table[idx], table[idx | stride]
        if hi > lo:
            rising = True
        elif hi < lo:
            falling = True
        if rising and falling:
            return None
    if rising:
        return 1
    if falling:
        return -1
    return 0


def _flip_table(table: Sequence[int], n: int, flips: Sequence[int]) -> list[int]:
    """Rewrite the truth table with the listed variables complemented."""
    mask = 0
    for v in flips:
        mask |= 1 << (n - 1 - v)
    return [table[idx ^ mask] for idx in range(1 << n)]


def _minimal_true_points(table: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """True points with no true point strictly below them."""
    trues = [bits for i, bits in enumerate(truth_bits(n)) if table[i]]
    out = []
    for x in trues:
        if not any(
            y != x and all(a <= b for a, b in zip(y, x)) for y in trues
        ):
            out.append(x)
    return out


def _maximal_false_points(table: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """False points with no false point strictly above them."""
    falses = [bits for i, bits in enumerate(truth_bits(n)) if not table[i]]
    out = []
    for x in falses:
        if not any(
            y != x and all(a >= b for a, b in zip(y, x)) for y in falses
        ):
            out.append(x)
    return out


def identify(table: Sequence[int], n: int) -> ThresholdRealisation | None:
    """
    Decide whether `table` is a threshold function and if so realise it.

    `table` is indexed so that bit (n-1-i) of the index carries variable i,
    matching truth_bits() ordering.
    """
    if len(table) != 1 << n:
        raise ValueError(f"table has {len(table)} entries, expected {1 << n}")
    if any(v not in (0, 1) for v in table):
        raise ValueError("table entries must be binary")

    # Stage 1: constants.
    if not any(table):
        return ThresholdRealisation(tuple([0] * n), 1)
    if all(table):
        return ThresholdRealisation(tuple([0] * n), 0)

    # Stage 2: unateness.
    polarity: list[int] = []
    for v in range(n):
        p = is_unate_in(table, n, v)
        if p is None:
            return None
        polarity.append(p)

    flips = [v for v, p in enumerate(polarity) if p < 0]
    pos = _flip_table(table, n, flips) if flips else list(table)

    # Stage 3: LP over the dominating constraints.
    min_true = _minimal_true_points(pos, n)
    max_false = _maximal_false_points(pos, n)

    # Variables: w_0..w_{n-1} >= 0, then T = tp - tq with tp, tq >= 0.
    n_vars = n + 2
    tp, tq = n, n + 1
    cons: list[Constraint] = []

    for x in min_true:
        row = [0] * n_vars
        for i, b in enumerate(x):
            row[i] = b
        row[tp], row[tq] = -1, 1
        cons.append(Constraint(row, ">=", 0))

    for x in max_false:
        row = [0] * n_vars
        for i, b in enumerate(x):
            row[i] = -b
        row[tp], row[tq] = 1, -1
        cons.append(Constraint(row, ">=", 1))

    # An unused variable must carry weight zero, so it cannot smuggle in
    # influence that the truth table does not have.
    for v, p in enumerate(polarity):
        if p == 0:
            row = [0] * n_vars
            row[v] = 1
            cons.append(Constraint(row, "==", 0))

    sol = solve_feasibility(n_vars, cons)
    if sol is None:
        return None

    w_frac = [sol[i] for i in range(n)]
    t_frac = sol[tp] - sol[tq]

    scale = 1
    for f in list(w_frac) + [t_frac]:
        scale = scale * f.denominator // gcd(scale, f.denominator)

    w_int = [int(f * scale) for f in w_frac]
    t_int = int(t_frac * scale)

    # Undo the polarity flips.
    shift = sum(w_int[v] for v in flips)
    for v in flips:
        w_int[v] = -w_int[v]
    t_int -= shift

    # Reduce by the common factor, keeping the >= boundary intact. Dividing
    # weights by g requires the threshold to round *up*, since w.x is a
    # multiple of g and w.x >= T iff w.x/g >= ceil(T/g).
    g = 0
    for w in w_int:
        g = gcd(g, abs(w))
    if g > 1:
        w_int = [w // g for w in w_int]
        t_int = -((-t_int) // g)  # ceiling division

    result = ThresholdRealisation(tuple(w_int), t_int)

    # The LP is exact, but the flip/scale/reduce arithmetic is fiddly enough
    # to be worth checking directly against the truth table.
    for idx, bits in enumerate(truth_bits(n)):
        if result.eval_bits(bits) != table[idx]:
            raise AssertionError(
                f"identify() produced a gate that does not match its table: "
                f"{result} at {bits}"
            )
    return result


def is_threshold(table: Sequence[int], n: int) -> bool:
    """Separability test only, discarding the realisation."""
    return identify(table, n) is not None


def gate_to_table(gate: ThGate) -> list[int]:
    """Truth table of an existing gate, in truth_bits() order."""
    return [
        int(sum(w * b for w, b in zip(gate.weights, bits)) >= gate.threshold)
        for bits in truth_bits(gate.fanin)
    ]
