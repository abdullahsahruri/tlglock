from fractions import Fraction

import pytest

from tlglock.lp import Constraint, LpError, solve_feasibility, verify


def solve(n, cons):
    return solve_feasibility(n, cons)


def test_trivially_feasible():
    x = solve(2, [Constraint([1, 1], ">=", 0)])
    assert x is not None and verify(2, [Constraint([1, 1], ">=", 0)], x)


def test_simple_feasible_system():
    cons = [
        Constraint([1, 1], ">=", 2),
        Constraint([1, 0], "<=", 3),
        Constraint([0, 1], "<=", 3),
    ]
    x = solve(2, cons)
    assert x is not None and verify(2, cons, x)


def test_infeasible_system():
    cons = [Constraint([1, 1], ">=", 5), Constraint([1, 1], "<=", 2)]
    assert solve(2, cons) is None


def test_infeasible_by_nonnegativity():
    """x >= 0 is implicit, so demanding x <= -1 is unsatisfiable."""
    assert solve(1, [Constraint([1], "<=", -1)]) is None


def test_equality_constraint():
    cons = [Constraint([1, 1], "==", 4), Constraint([1, -1], "==", 0)]
    x = solve(2, cons)
    assert x is not None
    assert x[0] == 2 and x[1] == 2


def test_negative_rhs_is_normalised():
    cons = [Constraint([-1, -1], ">=", -3)]
    x = solve(2, cons)
    assert x is not None and verify(2, cons, x)


def test_exact_rational_solution():
    """The point of exact arithmetic: 1/3 must come back as 1/3."""
    cons = [Constraint([3], "==", 1)]
    x = solve(1, cons)
    assert x == [Fraction(1, 3)]


def test_tight_equality_chain():
    cons = [
        Constraint([2, 1], "==", 5),
        Constraint([1, 3], "==", 5),
    ]
    x = solve(2, cons)
    assert x is not None
    assert 2 * x[0] + x[1] == 5
    assert x[0] + 3 * x[1] == 5


def test_degenerate_system_terminates():
    """Bland's rule must prevent cycling on a degenerate vertex."""
    cons = [
        Constraint([1, 1, 0], ">=", 0),
        Constraint([1, 0, 1], ">=", 0),
        Constraint([0, 1, 1], ">=", 0),
        Constraint([1, 1, 1], "<=", 0),
    ]
    x = solve(3, cons)
    assert x == [0, 0, 0]


def test_many_constraints():
    n = 6
    cons = [Constraint([1] * n, ">=", 3)]
    for i in range(n):
        row = [0] * n
        row[i] = 1
        cons.append(Constraint(row, "<=", 2))
    x = solve(n, cons)
    assert x is not None and verify(n, cons, x)


def test_wrong_coefficient_count_rejected():
    with pytest.raises(ValueError, match="coefficients"):
        solve(3, [Constraint([1, 1], ">=", 1)])


def test_bad_sense_rejected():
    with pytest.raises(ValueError, match="sense"):
        Constraint([1], "<", 1)


def test_zero_vars_rejected():
    with pytest.raises(ValueError):
        solve(0, [])


def test_no_constraints_returns_origin():
    assert solve(3, []) == [0, 0, 0]


def test_iteration_cap_raises():
    cons = [Constraint([1, 1], ">=", 1)]
    with pytest.raises(LpError):
        solve_feasibility(2, cons, max_iters=0)


def test_verify_rejects_negative_values():
    assert not verify(1, [Constraint([1], ">=", -5)], [Fraction(-1)])


@pytest.mark.parametrize("rhs", range(0, 8))
def test_feasibility_boundary(rhs):
    """x1 + x2 >= rhs with both capped at 2 is feasible exactly up to 4."""
    cons = [
        Constraint([1, 1], ">=", rhs),
        Constraint([1, 0], "<=", 2),
        Constraint([0, 1], "<=", 2),
    ]
    x = solve(2, cons)
    assert (x is not None) == (rhs <= 4)
    if x is not None:
        assert verify(2, cons, x)
