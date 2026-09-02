"""
AppSAT tests.

Two things are being established. First that this really is AppSAT and not a
key-sampling loop wearing its name: the inner solve is the exact miter solve,
the stopping rule is an error estimate from oracle queries, and disabling the
stopping rule must reduce the whole thing to sat_attack() exactly.

Second that the two output-error definitions are kept apart. "Fraction of
patterns with any wrong output" and "fraction of wrong output bits" are
different numbers on a multi-output circuit, and a resistance claim that
quotes one while comparing against the other is comparing different
quantities. The tests pin the invariant between them and demonstrate that the
gap is real rather than theoretical.
"""

from itertools import product

import pytest

from tlglock.abc import map_to_tlg, read_bench
from tlglock.attack import (
    AppSatResult,
    PbSolver,
    Status,
    appsat_attack,
    estimate_key_error,
    oracle_from,
    sat_attack,
    verify_recovered_key,
)
from tlglock.collapse import collapse
from tlglock.locking import lock
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


def locked_c17(keys_per_gate=1, mode="balanced", seed=1):
    net = c17_tlg()
    return net, lock(net, percent=100, keys_per_gate=keys_per_gate, mode=mode, seed=seed)


def data_inputs_of(report):
    return [n for n in report.locked_network.inputs if n not in report.key_names]


# -- error estimation -------------------------------------------------------

def test_correct_key_has_zero_error():
    """The planted key must measure as error-free, by both definitions."""
    import random

    net, report = locked_c17()
    ep, eb, counter = estimate_key_error(
        report.locked_network,
        report.correct_key,
        oracle_from(net),
        data_inputs_of(report),
        samples=64,
        rng=random.Random(0),
    )
    assert ep == 0.0 and eb == 0.0
    assert counter == []


def test_wrong_key_has_positive_error():
    import random

    net, report = locked_c17(keys_per_gate=2, mode="high")
    wrong = {k: v ^ 1 for k, v in report.correct_key.items()}
    ep, eb, counter = estimate_key_error(
        report.locked_network,
        wrong,
        oracle_from(net),
        data_inputs_of(report),
        samples=64,
        rng=random.Random(0),
    )
    if ep == 0.0:
        pytest.skip("flipped key happens to be equivalent for this instance")
    assert ep > 0.0 and eb > 0.0
    assert counter, "a positive error rate must yield counterexamples"


def test_counterexamples_really_disagree():
    """Every returned counterexample must be a genuine oracle mismatch."""
    import random

    net, report = locked_c17(keys_per_gate=2, mode="high")
    wrong = {k: v ^ 1 for k, v in report.correct_key.items()}
    locked = report.locked_network
    _, _, counter = estimate_key_error(
        locked, wrong, oracle_from(net), data_inputs_of(report),
        samples=64, rng=random.Random(0),
    )
    assert counter, "expected at least one mismatch to check"
    for pattern, response in counter:
        assert outputs_of(net, pattern) == response
        assert outputs_of(locked, {**pattern, **wrong}) != response


def test_zero_samples_is_a_no_measurement():
    import random

    net, report = locked_c17()
    ep, eb, counter = estimate_key_error(
        report.locked_network, report.correct_key, oracle_from(net),
        data_inputs_of(report), samples=0, rng=random.Random(0),
    )
    assert (ep, eb, counter) == (0.0, 0.0, [])


# -- the two error definitions ----------------------------------------------

def test_bit_error_never_exceeds_pattern_error():
    """
    The invariant: a pattern counted as wrong contributes at most n_out wrong
    bits, so error_bits <= error_patterns always.
    """
    import random

    net, report = locked_c17(keys_per_gate=2, mode="high")
    locked = report.locked_network
    oracle = oracle_from(net)
    data = data_inputs_of(report)

    for trial, kbits in enumerate(product((0, 1), repeat=report.num_keys)):
        kv = dict(zip(report.key_names, kbits))
        ep, eb, _ = estimate_key_error(
            locked, kv, oracle, data, samples=64, rng=random.Random(trial)
        )
        assert eb <= ep + 1e-12, f"key {kbits}: bits {eb} > patterns {ep}"


def test_the_two_definitions_actually_differ():
    """
    Not a theoretical distinction. c17 has two outputs, so a key that corrupts
    one of them on some pattern separates the two measures -- which is exactly
    why a paper must say which one it is quoting.
    """
    import random

    net, report = locked_c17(keys_per_gate=2, mode="high")
    locked = report.locked_network
    oracle = oracle_from(net)
    data = data_inputs_of(report)

    gaps = []
    for trial, kbits in enumerate(product((0, 1), repeat=report.num_keys)):
        kv = dict(zip(report.key_names, kbits))
        ep, eb, _ = estimate_key_error(
            locked, kv, oracle, data, samples=64, rng=random.Random(trial)
        )
        if ep > 0:
            gaps.append(ep - eb)

    assert gaps, "expected at least one wrong key"
    assert max(gaps) > 0.0, "the two definitions never differed -- check n_out"


# -- exact termination ------------------------------------------------------

def test_recovers_an_exact_key_on_c17():
    """c17 is small, so the loop reaches UNSAT before the stopping rule fires."""
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=0.0, round_size=3, samples=32,
    )
    assert res.exact is True
    assert res.settled is False
    assert res.status is Status.UNSAT
    assert verify_recovered_key(net, report.locked_network, res.key)


def test_exact_termination_measures_zero_error():
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=3, samples=64,
    )
    assert res.exact is True
    assert res.error_patterns == 0.0 and res.error_bits == 0.0


@pytest.mark.parametrize("mode", ["equal", "balanced", "high"])
def test_exact_key_is_equivalent_across_modes(mode):
    net, report = locked_c17(keys_per_gate=2, mode=mode)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=4, samples=16,
    )
    assert res.exact is True
    assert verify_recovered_key(net, report.locked_network, res.key)


# -- reduction to sat_attack ------------------------------------------------

def test_disabling_the_stopping_rule_reproduces_sat_attack():
    """
    The load-bearing claim about what AppSAT is. With no random queries and a
    negative epsilon there is no approximate criterion left, so the trajectory
    must be sat_attack()'s exactly -- same DIP count, same iterations. If this
    diverges, the inner loop is not the standard attack and any comparison
    between the two is meaningless.
    """
    net, report = locked_c17(keys_per_gate=2)
    locked, names = report.locked_network, report.key_names

    exact = sat_attack(locked, names, oracle_from(net), timeout=120)
    approx = appsat_attack(
        locked, names, oracle_from(net), timeout=120,
        epsilon=-1.0, samples=0, learn_from_queries=False, round_size=3,
    )

    assert approx.iterations == exact.iterations
    assert [sorted(d.items()) for d in approx.dips] == [
        sorted(d.items()) for d in exact.dips
    ]
    assert verify_recovered_key(net, locked, approx.key)
    assert verify_recovered_key(net, locked, exact.key)


# -- early termination ------------------------------------------------------

def test_settles_early_when_epsilon_is_permissive():
    """
    epsilon = 1.0 accepts any key at all, so the first round must stop the
    attack -- the path a real approximate attack takes on a weak lock.
    """
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=1.0, round_size=1, settle_rounds=1, samples=16,
    )
    assert res.settled is True
    assert res.exact is False
    assert res.status is Status.SAT
    assert res.key is not None
    assert res.rounds == 1
    assert res.iterations == 1


def test_settled_key_meets_the_reported_bound():
    net, report = locked_c17(keys_per_gate=2)
    eps = 0.5
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=eps, round_size=1, settle_rounds=1, samples=32,
    )
    if not res.settled:
        pytest.skip("ran to exact termination before settling")
    assert res.error_patterns <= eps


def test_returns_a_genuinely_approximate_key():
    """
    The point of the attack, and the thing a resistance claim has to rule out:
    AppSAT can stop on a key that is *wrong* -- not equivalent to the planted
    key, failing verify_recovered_key() -- yet within the error bound, because
    it stopped before the DIP loop could eliminate it.

    Swept over lock seeds because whether a wrong key survives round one
    depends on which key the solver happens to extract. What is asserted is
    that such a case exists at all, and that every one of them respects the
    bound it claims.
    """
    net = c17_tlg()
    approximate = []

    for seed in range(12):
        report = lock(net, percent=100, keys_per_gate=2, mode="high", seed=seed)
        res = appsat_attack(
            report.locked_network, report.key_names, oracle_from(net),
            timeout=120, epsilon=0.5, round_size=1, settle_rounds=1,
            samples=32, seed=seed,
        )
        if not res.settled:
            continue
        assert res.error_patterns <= 0.5, "settled above its own bound"
        if not verify_recovered_key(net, report.locked_network, res.key):
            approximate.append((seed, res.error_patterns))

    assert approximate, (
        "AppSAT never returned a non-equivalent key -- the approximate path "
        "is not being exercised"
    )
    # An approximate key is wrong somewhere, or it would have verified.
    assert all(ep > 0.0 for _, ep in approximate)


def test_settle_rounds_requires_a_streak():
    """One lucky sample must not end the attack when settle_rounds > 1."""
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=1.0, round_size=1, settle_rounds=3, samples=8,
    )
    if res.settled:
        assert res.rounds >= 3
    else:
        assert res.exact is True


# -- bookkeeping ------------------------------------------------------------

def test_queries_are_counted():
    net, report = locked_c17(keys_per_gate=2)
    samples = 16
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=2, samples=samples,
    )
    assert res.queries == samples * (res.rounds + 1)  # rounds, plus the final check


def test_round_callback_fires_in_order():
    net, report = locked_c17(keys_per_gate=2)
    seen = []
    appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=1, samples=8,
        on_round=lambda r, ep, eb: seen.append((r, ep, eb)),
    )
    assert [r for r, _, _ in seen] == list(range(1, len(seen) + 1))
    for _, ep, eb in seen:
        assert 0.0 <= eb <= ep <= 1.0


def test_all_dips_are_distinct():
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=3, samples=8,
    )
    seen = [tuple(sorted(d.items())) for d in res.dips]
    assert len(seen) == len(set(seen))


def test_is_seed_reproducible():
    net, report = locked_c17(keys_per_gate=2)
    kw = dict(
        timeout=120, epsilon=0.05, round_size=2, samples=16, settle_rounds=2
    )
    a = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net), seed=5, **kw
    )
    b = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net), seed=5, **kw
    )
    assert (a.key, a.settled, a.exact) == (b.key, b.settled, b.exact)
    assert a.error_patterns == b.error_patterns


# -- degenerate cases -------------------------------------------------------

def test_unlocked_network_terminates_immediately():
    net = c17_tlg()
    report = lock(net, percent=0, seed=0)
    res = appsat_attack(report.locked_network, [], oracle_from(net), timeout=30)
    assert res.exact is True
    assert res.iterations == 0
    assert res.key == {}


def test_dead_key_is_recovered():
    """A zero-weight key never affects the output, so any value is exact."""
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
    res = appsat_attack(locked, ["k1"], oracle_from(net), timeout=30)
    assert res.exact is True
    assert verify_recovered_key(net, locked, res.key)
    assert res.error_patterns == 0.0


def test_timeout_is_reported_not_raised():
    net, report = locked_c17(keys_per_gate=4)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=0.001
    )
    assert res.status is Status.TIMEOUT
    assert res.key is None


def test_learning_disabled_still_terminates():
    net, report = locked_c17(keys_per_gate=2)
    res = appsat_attack(
        report.locked_network, report.key_names, oracle_from(net),
        timeout=120, epsilon=-1.0, round_size=2, samples=8,
        learn_from_queries=False,
    )
    assert res.exact is True
    assert verify_recovered_key(net, report.locked_network, res.key)
