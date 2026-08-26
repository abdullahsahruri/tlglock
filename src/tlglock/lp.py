"""
Exact rational LP feasibility, via phase-1 simplex over Fraction arithmetic.

Threshold identification asks whether a weight vector and threshold exist that
realise a given Boolean function. That is an LP feasibility question, and the
answer has to be exact: a floating-point solver that reports "feasible" with a
margin of 1e-12 will happily hand back a weight vector that does not actually
implement the function, and the error surfaces much later as a mismatched
netlist. So this uses Fractions throughout and is exact by construction.

The instances are tiny -- after the unateness reduction in separable.py, a
6-input cut yields on the order of ten constraints over eight variables -- so
the cost of exact arithmetic is irrelevant here.

Bland's rule is used for pivot selection. It is slower than Dantzig's, but it
guarantees termination, and a synthesis pass that hangs on one cut in ten
thousand is worse than one that is uniformly a little slower.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Sequence

Sense = Literal[">=", "<=", "=="]
Num = int | Fraction


class LpError(RuntimeError):
    """Raised when the solver hits an internal inconsistency."""


class Constraint:
    """One linear constraint: sum(coeffs * x) <sense> rhs."""

    __slots__ = ("coeffs", "sense", "rhs")

    def __init__(self, coeffs: Sequence[Num], sense: Sense, rhs: Num):
        if sense not in (">=", "<=", "=="):
            raise ValueError(f"bad sense '{sense}'")
        self.coeffs = [Fraction(c) for c in coeffs]
        self.sense = sense
        self.rhs = Fraction(rhs)

    def __repr__(self) -> str:
        return f"Constraint({self.coeffs}, '{self.sense}', {self.rhs})"


def solve_feasibility(
    n_vars: int,
    constraints: Sequence[Constraint],
    max_iters: int = 100_000,
) -> list[Fraction] | None:
    """
    Find x >= 0 satisfying every constraint, or None if none exists.

    All variables are constrained non-negative. Model a free variable as the
    difference of two non-negative variables; separable.py does this for the
    threshold, which genuinely can be negative.
    """
    if n_vars <= 0:
        raise ValueError("n_vars must be positive")
    if not constraints:
        return [Fraction(0)] * n_vars

    # -- build the phase-1 tableau ------------------------------------------
    # Each row: coeffs | surplus/slack | artificials | rhs, with rhs >= 0.
    rows: list[list[Fraction]] = []
    senses: list[Sense] = []

    for c in constraints:
        if len(c.coeffs) != n_vars:
            raise ValueError(
                f"constraint has {len(c.coeffs)} coefficients, expected {n_vars}"
            )
        coeffs, rhs, sense = list(c.coeffs), c.rhs, c.sense
        if rhs < 0:
            # Normalise to a non-negative right-hand side.
            coeffs = [-v for v in coeffs]
            rhs = -rhs
            if sense == ">=":
                sense = "<="
            elif sense == "<=":
                sense = ">="
        rows.append(coeffs)
        senses.append(sense)
        # store normalised rhs back
        c_rhs = rhs
        rows[-1] = coeffs + [c_rhs]  # rhs parked at the end for now

    n_extra = sum(1 for s in senses if s != "==")
    n_art = len(rows)
    width = n_vars + n_extra + n_art + 1
    rhs_col = width - 1

    tableau: list[list[Fraction]] = []
    extra_at = n_vars
    art_at = n_vars + n_extra
    basis: list[int] = []

    for i, (row, sense) in enumerate(zip(rows, senses)):
        new = [Fraction(0)] * width
        new[:n_vars] = row[:n_vars]
        new[rhs_col] = row[n_vars]
        if sense == ">=":
            new[extra_at] = Fraction(-1)   # surplus
            extra_at += 1
        elif sense == "<=":
            new[extra_at] = Fraction(1)    # slack
            extra_at += 1
        new[art_at + i] = Fraction(1)      # artificial
        basis.append(art_at + i)
        tableau.append(new)

    # Objective: minimise the sum of artificials, expressed in terms of
    # non-basic variables by subtracting every row.
    cost = [Fraction(0)] * width
    for row in tableau:
        for j in range(width):
            cost[j] -= row[j]
    for i in range(n_art):
        cost[art_at + i] = Fraction(0)

    # -- simplex ------------------------------------------------------------
    for _ in range(max_iters):
        # Bland: lowest index with negative reduced cost.
        pivot_col = -1
        for j in range(width - 1):
            if cost[j] < 0:
                pivot_col = j
                break
        if pivot_col < 0:
            break  # optimal

        # Ratio test, ties broken by lowest basis index (Bland).
        pivot_row = -1
        best: Fraction | None = None
        for i, row in enumerate(tableau):
            if row[pivot_col] > 0:
                ratio = row[rhs_col] / row[pivot_col]
                if best is None or ratio < best or (ratio == best and basis[i] < basis[pivot_row]):
                    best, pivot_row = ratio, i
        if pivot_row < 0:
            # Unbounded below. The phase-1 objective is bounded by zero, so
            # this cannot happen on a well-formed tableau.
            raise LpError("phase-1 objective unbounded")

        _pivot(tableau, cost, pivot_row, pivot_col, width)
        basis[pivot_row] = pivot_col
    else:
        raise LpError(f"simplex did not converge in {max_iters} iterations")

    # Objective value is -cost[rhs]; feasible iff it is zero.
    if -cost[rhs_col] != 0:
        return None

    # Drive any artificial still in the basis out at zero level, so the
    # extracted solution does not depend on a degenerate artificial.
    for i, b in enumerate(basis):
        if b >= art_at:
            for j in range(art_at):
                if tableau[i][j] != 0:
                    _pivot(tableau, cost, i, j, width)
                    basis[i] = j
                    break

    solution = [Fraction(0)] * n_vars
    for i, b in enumerate(basis):
        if b < n_vars:
            solution[b] = tableau[i][rhs_col]
    return solution


def _pivot(
    tableau: list[list[Fraction]],
    cost: list[Fraction],
    prow: int,
    pcol: int,
    width: int,
) -> None:
    piv = tableau[prow][pcol]
    if piv == 0:
        raise LpError("zero pivot")
    row = tableau[prow]
    if piv != 1:
        tableau[prow] = [v / piv for v in row]
        row = tableau[prow]

    for i, other in enumerate(tableau):
        if i == prow:
            continue
        factor = other[pcol]
        if factor != 0:
            tableau[i] = [a - factor * b for a, b in zip(other, row)]

    factor = cost[pcol]
    if factor != 0:
        for j in range(width):
            cost[j] -= factor * row[j]


def verify(
    n_vars: int, constraints: Sequence[Constraint], x: Sequence[Fraction]
) -> bool:
    """Independently check a candidate solution. Used by the tests."""
    if len(x) != n_vars or any(v < 0 for v in x):
        return False
    for c in constraints:
        total = sum(a * b for a, b in zip(c.coeffs, x))
        if c.sense == ">=" and total < c.rhs:
            return False
        if c.sense == "<=" and total > c.rhs:
            return False
        if c.sense == "==" and total != c.rhs:
            return False
    return True
