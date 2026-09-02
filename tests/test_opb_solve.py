"""
Tests for the PySAT-backed OPB solver in tools/.

The risk with a PB-to-CNF translation is that it is silently *a* problem
rather than *the* problem: negative coefficients normalised the wrong way,
or a bound off by the sum of the negatives, still produce a well-formed
instance that solves cleanly and answers a different question. So the
translation is not inspected, it is differential-tested -- every instance is
solved twice, once by this path and once by the built-in PbSolver, and the
verdicts must agree. Where the answer is SAT, the returned model is checked
against the original PB constraints directly.

Skipped when pysat/pypblib are not installed; they are optional extras, not
runtime dependencies of the library.
"""

import pathlib
import sys
from itertools import product

import pytest

pytest.importorskip("pysat", reason="needs: pip install --user python-sat pypblib")
pytest.importorskip("pysat.pb", reason="needs: pip install --user pypblib")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tools"))

import opb_solve  # noqa: E402

from tlglock.abc import map_to_tlg, read_bench  # noqa: E402
from tlglock.attack import (  # noqa: E402
    ExternalSolver,
    PbSolver,
    Status,
    build_attack_formula,
    oracle_from,
    sat_attack,
    verify_recovered_key,
)
from tlglock.collapse import collapse  # noqa: E402
from tlglock.locking import lock  # noqa: E402
from tlglock.opb import OpbEncoder  # noqa: E402

BINARY = str(pathlib.Path(__file__).parent.parent / "tools" / "opb_solve.py")

C17 = """\
INPUT(1)
INPUT(2)
INPUT(3)
INPUT(6)
INPUT(7)
OUTPUT(22)
OUTPUT(23)
10 = NAND(1, 3)
11 = NAND(3, 6)
16 = NAND(2, 11)
19 = NAND(11, 7)
22 = NAND(10, 16)
23 = NAND(16, 19)
"""


def c17_tlg():
    net, _ = map_to_tlg(read_bench(C17, name="c17"))
    net, _ = collapse(net)
    return net


def satisfies(enc, model):
    return all(
        sum(c * model.get(v, 0) for c, v in con.terms) >= con.rhs
        for con in enc.constraints
    )


def brute_force(enc):
    n = enc.num_vars
    assert n <= 14
    for bits in product((0, 1), repeat=n):
        if satisfies(enc, {i + 1: b for i, b in enumerate(bits)}):
            return True
    return False


# -- parsing ----------------------------------------------------------------

def test_parses_what_opbencoder_writes(tmp_path):
    enc = OpbEncoder()
    a, b = enc.var("a"), enc.var("b")
    enc.add([(3, a), (-2, b)], 1, "a comment")
    path = tmp_path / "i.opb"
    enc.write(str(path))

    n_vars, cons = opb_solve.parse_opb(path.read_text())
    assert n_vars == 2
    assert cons == [([(3, a), (-2, b)], 1)]


def test_comments_and_objective_are_ignored():
    text = "* #variable= 2 #constraint= 1\n* note\nmin: +1 x1;\n+1 x1 +1 x2 >= 1;\n"
    n_vars, cons = opb_solve.parse_opb(text)
    assert n_vars == 2 and len(cons) == 1


def test_unsupported_relation_rejected():
    with pytest.raises(ValueError, match="expected >="):
        opb_solve.parse_opb("+1 x1 <= 1;\n")


# -- the translation --------------------------------------------------------

@pytest.mark.parametrize(
    "weights,rhs",
    [
        ([1, 1, 1], 2),
        ([-1, -1], -1),
        ([2, -1], 1),
        ([-3, 4, -2, 1], -1),
        ([3, 2, 1, -2], 3),
        ([1, 1], 0),      # trivially true
        ([1, 1], 3),      # unreachable
        ([-2, -2], 1),    # unreachable with negatives
    ],
)
def test_translation_preserves_the_solution_set(weights, rhs):
    """
    Exhaustive: the CNF must admit a model exactly when the PB constraint is
    satisfiable, for the same variable assignment. This is where a mishandled
    negative coefficient shows up.
    """
    n = len(weights)
    enc = OpbEncoder()
    vs = [enc.var(f"x{i}") for i in range(n)]
    enc.add(list(zip(weights, vs)), rhs)

    clauses, _ = opb_solve.to_cnf(n, [(list(zip(weights, vs)), rhs)])
    from pysat.solvers import Solver

    for bits in product((0, 1), repeat=n):
        assign = {i + 1: b for i, b in enumerate(bits)}
        want = satisfies(enc, assign)
        units = [(v if bits[v - 1] else -v) for v in vs]
        with Solver(name="cadical153", bootstrap_with=clauses) as s:
            got = s.solve(assumptions=units)
        assert got == want, f"{weights} >= {rhs} at {bits}"


# -- differential against the built-in solver -------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_agrees_with_builtin_on_random_instances(seed):
    import random

    rng = random.Random(seed)
    enc = OpbEncoder()
    vs = [enc.var(f"v{i}") for i in range(6)]
    for _ in range(5):
        terms = [
            (rng.choice([-3, -2, -1, 1, 2, 3]), v)
            for v in rng.sample(vs, rng.randint(1, 4))
        ]
        enc.add(terms, rng.randint(-2, 4))

    external = ExternalSolver(binary=BINARY).solve(enc, timeout=60)
    expected = brute_force(enc)
    assert (external.status is Status.SAT) == expected
    if external.status is Status.SAT:
        assert satisfies(enc, external.model)


def test_agrees_with_builtin_on_a_real_miter():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=1)
    enc = build_attack_formula(report.locked_network, report.key_names, [])

    builtin = PbSolver().solve(enc, timeout=60)
    external = ExternalSolver(binary=BINARY).solve(enc, timeout=60)
    assert builtin.status is external.status is Status.SAT
    assert satisfies(enc, external.model)


def test_detects_unsat():
    enc = OpbEncoder()
    a = enc.var("a")
    enc.add([(1, a)], 1)
    enc.add([(-1, a)], 0)
    assert ExternalSolver(binary=BINARY).solve(enc, timeout=60).status is Status.UNSAT


def test_reports_solver_statistics():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=3, seed=1)
    enc = build_attack_formula(report.locked_network, report.key_names, [])
    res = ExternalSolver(binary=BINARY).solve(enc, timeout=60)
    assert res.conflicts >= 0 and res.decisions >= 0


# -- end to end -------------------------------------------------------------

@pytest.mark.parametrize("keys_per_gate", [1, 2, 3])
def test_attack_recovers_key_through_the_external_solver(keys_per_gate):
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=keys_per_gate, seed=1)
    res = sat_attack(
        report.locked_network,
        report.key_names,
        oracle_from(net),
        solver=ExternalSolver(binary=BINARY),
        timeout=180,
    )
    assert res.status is Status.UNSAT
    assert verify_recovered_key(net, report.locked_network, res.key)


def test_both_solvers_reach_the_same_verdict_end_to_end():
    """
    Different search, same answer. Iteration counts may differ -- the solvers
    pick different distinguishing inputs -- but both must terminate UNSAT with
    a functionally correct key.
    """
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=1)
    args = (report.locked_network, report.key_names, oracle_from(net))

    a = sat_attack(*args, solver=PbSolver(), timeout=180)
    b = sat_attack(*args, solver=ExternalSolver(binary=BINARY), timeout=180)

    assert a.status is b.status is Status.UNSAT
    assert verify_recovered_key(net, report.locked_network, a.key)
    assert verify_recovered_key(net, report.locked_network, b.key)
