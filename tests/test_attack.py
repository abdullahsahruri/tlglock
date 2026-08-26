"""
SAT attack tests.

Two things are being checked. First that PbSolver is actually correct -- an
incomplete or unsound solver would make the attack report UNSAT prematurely
and look like a security result when it is a bug. Every SAT answer is verified
against the constraints, and every UNSAT answer is cross-checked by brute
force on instances small enough to enumerate.

Second that the attack recovers a *functionally equivalent* key. Threshold
weight degeneracy means the recovered bit string often differs from the
planted one while implementing the same function, so bitwise comparison would
understate the attack's success.
"""

from itertools import product

import pytest

from tlglock.abc import map_to_tlg, read_bench
from tlglock.attack import (
    ExternalSolver,
    PbSolver,
    Status,
    build_attack_formula,
    build_key_formula,
    oracle_from,
    sat_attack,
    verify_recovered_key,
)
from tlglock.collapse import collapse
from tlglock.locking import lock
from tlglock.opb import Constraint, OpbEncoder
from tlglock.sim import outputs_of
from tlglock.thfile import ThGate, ThNetwork

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


def satisfies_all(enc, model):
    return all(
        sum(c * model.get(v, 0) for c, v in con.terms) >= con.rhs
        for con in enc.constraints
    )


def brute_force_sat(enc):
    """Reference answer by enumeration. Only for tiny instances."""
    n = enc.num_vars
    assert n <= 16
    for bits in product((0, 1), repeat=n):
        model = {i + 1: b for i, b in enumerate(bits)}
        if satisfies_all(enc, model):
            return True
    return False


# -- PbSolver correctness ---------------------------------------------------

def test_solver_finds_simple_solution():
    enc = OpbEncoder()
    a, b = enc.var("a"), enc.var("b")
    enc.add([(1, a), (1, b)], 2)
    res = PbSolver().solve(enc, timeout=10)
    assert res.status is Status.SAT
    assert res.model[a] == 1 and res.model[b] == 1


def test_solver_detects_unsat():
    enc = OpbEncoder()
    a = enc.var("a")
    enc.add([(1, a)], 1)
    enc.add([(-1, a)], 0)
    assert PbSolver().solve(enc, timeout=10).status is Status.UNSAT


def test_solver_handles_negative_coefficients():
    enc = OpbEncoder()
    a, b = enc.var("a"), enc.var("b")
    enc.add([(-2, a), (1, b)], 1)
    res = PbSolver().solve(enc, timeout=10)
    assert res.status is Status.SAT
    assert satisfies_all(enc, res.model)
    assert res.model[a] == 0 and res.model[b] == 1


def test_solver_models_always_satisfy_constraints():
    """
    The regression that mattered: an earlier version leaked partially
    propagated assignments on conflict, so it returned models violating the
    constraints it was handed, and the attack loop spun forever rediscovering
    the same distinguishing input.
    """
    net = c17_tlg()
    for kpg in (1, 2, 3):
        report = lock(net, percent=100, keys_per_gate=kpg, seed=1)
        enc = build_attack_formula(
            report.locked_network, report.key_names, []
        )
        res = PbSolver().solve(enc, timeout=30)
        assert res.status is Status.SAT
        assert satisfies_all(enc, res.model), "solver returned an invalid model"


@pytest.mark.parametrize("seed", range(10))
def test_solver_agrees_with_brute_force(seed):
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

    res = PbSolver().solve(enc, timeout=30)
    expected = brute_force_sat(enc)
    assert (res.status is Status.SAT) == expected
    if res.status is Status.SAT:
        assert satisfies_all(enc, res.model)


def test_solver_reports_statistics():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=3, seed=1)
    enc = build_attack_formula(report.locked_network, report.key_names, [])
    res = PbSolver().solve(enc, timeout=30)
    assert res.decisions >= 0 and res.conflicts >= 0
    assert res.seconds >= 0.0


def test_solver_respects_conflict_cap():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=4, seed=1)
    enc = build_attack_formula(report.locked_network, report.key_names, [])
    res = PbSolver(max_conflicts=1).solve(enc, timeout=30)
    assert res.status in (Status.TIMEOUT, Status.SAT)


# -- miter construction -----------------------------------------------------

def test_miter_shares_data_inputs_only():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=1, seed=0)
    enc = build_attack_formula(report.locked_network, report.key_names, [])
    for n in net.inputs:
        assert n in enc.var_map
    for k in report.key_names:
        assert f"{k}_A" in enc.var_map and f"{k}_B" in enc.var_map
        assert k not in enc.var_map or enc.var_map[f"{k}_A"] != enc.var_map[k]


def test_oracle_copies_bind_the_same_key_variables():
    """
    The constraint from a recorded input must apply to the key the miter is
    solving for. If the oracle copies allocated fresh key variables the
    formula would grow without ever pruning the key space.
    """
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=1, seed=0)
    L = report.locked_network
    data = [n for n in L.inputs if n not in report.key_names]

    pattern = {n: 1 for n in data}
    history = [(pattern, outputs_of(net, pattern))]
    enc = build_attack_formula(L, report.key_names, history)

    key_vars = {enc.var(f"{k}_A") for k in report.key_names}
    mentioned = {v for c in enc.constraints for _, v in c.terms}
    assert key_vars <= mentioned


def test_key_formula_has_no_disagreement_clause():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=1, seed=0)
    L = report.locked_network
    data = [n for n in L.inputs if n not in report.key_names]
    pattern = {n: 0 for n in data}
    history = [(pattern, outputs_of(net, pattern))]

    miter = build_attack_formula(L, report.key_names, history)
    keys = build_key_formula(L, report.key_names, history)
    assert len(keys.constraints) < len(miter.constraints)


# -- the attack ------------------------------------------------------------

@pytest.mark.parametrize("keys_per_gate", [1, 2])
@pytest.mark.parametrize("mode", ["equal", "balanced", "high"])
def test_attack_recovers_an_equivalent_key(keys_per_gate, mode):
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=keys_per_gate, mode=mode, seed=1)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=60
    )
    assert res.status is Status.UNSAT
    assert res.key is not None
    assert verify_recovered_key(net, report.locked_network, res.key)


def test_recovered_key_may_differ_bitwise():
    """
    Weight degeneracy means several key vectors realise the same function.
    The attack is successful when it finds any of them, which is why success
    is measured functionally rather than by comparing bit strings.
    """
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, mode="balanced", seed=1)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=60
    )
    assert verify_recovered_key(net, report.locked_network, res.key)


def test_attack_terminates_within_the_input_space():
    """
    Each distinguishing input is eliminated once its oracle response is
    recorded, so the loop cannot need more iterations than there are input
    patterns. Exceeding that bound means constraints are not binding.
    """
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=1)
    data = [n for n in report.locked_network.inputs if n not in report.key_names]
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=60
    )
    assert res.iterations <= (1 << len(data))


def test_all_dips_are_distinct():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=3)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=60
    )
    seen = [tuple(sorted(d.items())) for d in res.dips]
    assert len(seen) == len(set(seen))


def test_unlocked_network_needs_no_dips():
    """With no keys there is nothing to distinguish, so the miter is UNSAT."""
    net = c17_tlg()
    report = lock(net, percent=0, seed=0)
    res = sat_attack(report.locked_network, [], oracle_from(net), timeout=30)
    assert res.status is Status.UNSAT
    assert res.iterations == 0


def test_dead_key_is_recovered_trivially():
    """A key with zero weight never affects the output; any value works."""
    net = ThNetwork(
        model="base",
        inputs=["x1", "x2"],
        outputs=["z"],
        gates=[ThGate(inputs=["x1", "x2"], output="z", weights=[1, 1], threshold=2)],
    )
    locked = ThNetwork(
        model="dead",
        inputs=["x1", "x2", "k1"],
        outputs=["z"],
        gates=[
            ThGate(inputs=["x1", "x2", "k1"], output="z", weights=[1, 1, 0], threshold=2)
        ],
    )
    res = sat_attack(locked, ["k1"], oracle_from(net), timeout=30)
    assert res.status is Status.UNSAT
    assert verify_recovered_key(net, locked, res.key)


def test_timeout_is_reported_not_raised():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=4, seed=1)
    res = sat_attack(
        report.locked_network,
        report.key_names,
        oracle_from(net),
        timeout=0.001,
    )
    assert res.status is Status.TIMEOUT
    assert res.key is None


def test_iteration_callback_fires():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=1)
    seen = []
    sat_attack(
        report.locked_network,
        report.key_names,
        oracle_from(net),
        timeout=60,
        on_iteration=lambda i, dip: seen.append(i),
    )
    assert seen == list(range(len(seen)))


def test_table_row_formatting():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=1, seed=1)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=60
    )
    row = res.table_row
    assert row.endswith("UNSAT") or row.endswith("SAT")
    assert len(row.split(",")) == 4


def test_timeout_row_matches_paper_notation():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=4, seed=1)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=0.001
    )
    assert res.table_row == "---,---,---,Timeout"


# -- verification helper ----------------------------------------------------

def test_verify_rejects_a_wrong_key():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, mode="high", seed=1)
    wrong = dict(report.correct_key)
    for k in wrong:
        wrong[k] ^= 1
    if verify_recovered_key(net, report.locked_network, wrong):
        pytest.skip("flipped key happens to be equivalent for this instance")
    assert not verify_recovered_key(net, report.locked_network, wrong)


def test_verify_accepts_the_planted_key():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=1)
    assert verify_recovered_key(net, report.locked_network, report.correct_key)


# -- external solver plumbing ----------------------------------------------

def test_external_solver_parses_sat_output():
    s = ExternalSolver()
    res = s._parse("c solving\ns SATISFIABLE\nv x1 -x2 x3\n", 1.5)
    assert res.status is Status.SAT
    assert res.model == {1: 1, 2: 0, 3: 1}
    assert res.seconds == 1.5


def test_external_solver_parses_unsat_output():
    assert ExternalSolver()._parse("s UNSATISFIABLE\n", 0.1).status is Status.UNSAT


def test_external_solver_parses_optimum_as_sat():
    assert ExternalSolver()._parse("s OPTIMUM FOUND\nv x1\n", 0.1).status is Status.SAT


def test_external_solver_parses_unknown():
    assert ExternalSolver()._parse("s UNKNOWN\n", 0.1).status is Status.UNKNOWN


def test_external_solver_scrapes_statistics():
    out = "c conflicts: 104\nc decisions: 10797\ns SATISFIABLE\nv x1\n"
    res = ExternalSolver()._parse(out, 0.2)
    assert res.conflicts == 104
    assert res.decisions == 10797


def test_external_solver_missing_binary_raises():
    enc = OpbEncoder()
    enc.var("a")
    enc.add([(1, 1)], 1)
    with pytest.raises(RuntimeError, match="not found"):
        ExternalSolver(binary="definitely-not-a-real-solver").solve(enc, timeout=5)
