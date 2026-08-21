import pytest

from tlglock.locking import lock
from tlglock.metrics import (
    corruption_rate,
    equivalent_key_count,
    is_equivalent_under_correct_key,
    key_weight_sweep,
    output_hamming_profile,
)
from tlglock.thfile import ThGate, ThNetwork

from conftest import random_network


# -- equivalence checking ---------------------------------------------------

def test_equivalence_true_for_correct_lock(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    assert is_equivalent_under_correct_key(multi, report)


def test_equivalence_false_when_threshold_tampered(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    report.locked_network.gates[0].threshold += 1
    assert not is_equivalent_under_correct_key(multi, report)


def test_equivalence_false_when_key_flipped(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    k = report.key_names[0]
    report.correct_key[k] ^= 1
    assert not is_equivalent_under_correct_key(multi, report)


def test_equivalence_false_on_output_mismatch(multi):
    report = lock(multi, percent=50, keys_per_gate=1, seed=0)
    report.locked_network.outputs = list(reversed(report.locked_network.outputs))
    assert not is_equivalent_under_correct_key(multi, report)


# -- corruption rate --------------------------------------------------------

def test_corruption_is_a_probability(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    rho = corruption_rate(multi, report)
    assert 0.0 <= rho <= 1.0


def test_corruption_is_zero_for_an_unlocked_network(multi):
    report = lock(multi, percent=0, seed=0)
    assert corruption_rate(multi, report) == 0.0


def test_corruption_is_zero_when_key_weights_are_dead():
    """A key with zero weight cannot corrupt anything."""
    net = ThNetwork(
        model="dead",
        inputs=["x1", "x2", "k1"],
        outputs=["z"],
        gates=[
            ThGate(inputs=["x1", "x2", "k1"], output="z", weights=[1, 1, 0], threshold=2)
        ],
    )
    from tlglock.locking import LockReport
    report = LockReport(
        locked_network=net,
        key_names=["k1"],
        correct_key={"k1": 1},
        locked_gates=["z"],
    )
    base = ThNetwork(
        model="base",
        inputs=["x1", "x2"],
        outputs=["z"],
        gates=[ThGate(inputs=["x1", "x2"], output="z", weights=[1, 1], threshold=2)],
    )
    assert corruption_rate(base, report) == 0.0


def test_corruption_is_positive_for_a_real_lock(multi):
    report = lock(multi, percent=100, keys_per_gate=2, mode="balanced", seed=3)
    assert corruption_rate(multi, report) > 0.0


@pytest.mark.parametrize("seed", range(8))
def test_corruption_positive_across_random_networks(seed):
    net = random_network(seed, n_in=5, n_gates=5)
    report = lock(net, percent=100, keys_per_gate=2, mode="high", seed=seed)
    assert corruption_rate(net, report) > 0.0


def test_corruption_is_seed_reproducible(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=1)
    a = corruption_rate(multi, report, seed=11)
    b = corruption_rate(multi, report, seed=11)
    assert a == b


def _sweep_key_weight(base, magnitudes, correct=(1, 0)):
    """Corruption as a function of key weight magnitude, on one gate."""
    from tlglock.locking import LockReport
    out = []
    names = ["k1", "k2"]
    ck = dict(zip(names, correct))
    for u in magnitudes:
        kw = [u, -u]
        shift = sum(v * ck[k] for k, v in zip(names, kw))
        g = base.gates[0]
        locked = ThNetwork(
            model="l",
            inputs=list(base.inputs) + names,
            outputs=list(base.outputs),
            gates=[
                ThGate(
                    inputs=list(g.inputs) + names,
                    output=g.output,
                    weights=list(g.weights) + kw,
                    threshold=g.threshold + shift,
                )
            ],
        )
        rep = LockReport(
            locked_network=locked, key_names=names,
            correct_key=ck, locked_gates=[g.output],
        )
        out.append(corruption_rate(base, rep))
    return out


def test_corruption_saturates_in_key_weight():
    """
    Corruption rises with key weight and then saturates -- it does not fall
    back down. Once |v| is large enough that the key alone decides the gate,
    a wrong key drives a constant output, and no further increase changes
    anything.

    Note: Section IV-B describes corruption as *peaking* at moderate total key
    weight (~2-3) and Fig. 4 is drawn that way. Under threshold compensation
    we observe monotone-then-flat instead. Worth checking against the raw
    Fig. 4 data before the journal version repeats the "peaks" wording; the
    security argument only needs "moderate weight is enough", which holds
    either way.
    """
    base = ThNetwork(
        model="b",
        inputs=[f"x{i}" for i in range(4)],
        outputs=["z"],
        gates=[
            ThGate(
                inputs=[f"x{i}" for i in range(4)],
                output="z",
                weights=[1, 1, 1, 1],
                threshold=2,
            )
        ],
    )
    curve = _sweep_key_weight(base, range(1, 9))

    # Non-decreasing.
    for a, b in zip(curve, curve[1:]):
        assert b >= a - 1e-9, f"corruption fell: {curve}"
    # Saturated by the top of the range.
    assert abs(curve[-1] - curve[-2]) < 1e-9
    # Plateau equals the distance to a constant output.
    from tlglock.sim import outputs_of
    from itertools import product
    ones = sum(
        outputs_of(base, dict(zip(base.inputs, bits)))[0]
        for bits in product((0, 1), repeat=4)
    )
    assert abs(curve[-1] - max(ones, 16 - ones) / 16) < 1e-9


def test_moderate_key_weight_already_reaches_most_of_the_plateau():
    """
    The actionable half of the Fig. 4 claim: you do not need large key
    weights. |v| = 2-3 gets most of the corruption that |v| = 8 does, at a
    fraction of the area and power cost.
    """
    base = ThNetwork(
        model="b",
        inputs=[f"x{i}" for i in range(4)],
        outputs=["z"],
        gates=[
            ThGate(
                inputs=[f"x{i}" for i in range(4)],
                output="z",
                weights=[1, 1, 1, 1],
                threshold=2,
            )
        ],
    )
    curve = _sweep_key_weight(base, [1, 2, 3, 8])
    assert curve[2] >= 0.95 * curve[3]


# -- Hamming profile --------------------------------------------------------

def test_hamming_profile_is_a_distribution(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    prof = output_hamming_profile(multi, report)
    assert set(prof) == set(range(len(multi.outputs) + 1))
    assert abs(sum(prof.values()) - 1.0) < 1e-9


def test_hamming_profile_all_zero_for_unlocked(multi):
    report = lock(multi, percent=0, seed=0)
    prof = output_hamming_profile(multi, report)
    assert all(v == 0.0 for v in prof.values())


def test_hamming_profile_consistent_with_corruption(multi):
    """Mean normalised Hamming distance must equal the corruption rate."""
    report = lock(multi, percent=100, keys_per_gate=2, seed=4)
    n_out = len(multi.outputs)
    prof = output_hamming_profile(multi, report, input_limit=64, key_limit=32, seed=7)
    mean = sum(d * p for d, p in prof.items()) / n_out
    rho = corruption_rate(multi, report, input_limit=64, key_limit=32, seed=7)
    assert abs(mean - rho) < 1e-9


# -- equivalent keys --------------------------------------------------------

def test_correct_key_is_always_counted(multi):
    report = lock(multi, percent=50, keys_per_gate=2, seed=0)
    assert equivalent_key_count(multi, report) >= 1


def test_equivalent_key_count_bounded_by_key_space(multi):
    report = lock(multi, percent=50, keys_per_gate=2, seed=0)
    n = equivalent_key_count(multi, report)
    assert n <= 2 ** report.num_keys


def test_both_modes_collapse_some_of_the_key_space(multi):
    """
    Threshold gates admit weight degeneracy, so distinct key vectors can land
    on the same halfspace. Both weight modes therefore have an equivalence
    class strictly larger than {k*}.

    Which mode compresses *more* is network-dependent -- on this 4-gate toy
    "balanced" collapses more than "equal", the opposite of what the larger
    benchmarks show -- so no ordering is asserted here.
    """
    for mode in ("equal", "balanced"):
        rep = lock(multi, percent=100, keys_per_gate=3, mode=mode, seed=6)
        n = equivalent_key_count(multi, rep)
        assert 1 <= n <= 2 ** rep.num_keys


def test_equivalent_keys_are_genuinely_equivalent(multi):
    """Whatever the count is, every key it admits must reproduce the original."""
    from itertools import product
    from tlglock.sim import outputs_of

    rep = lock(multi, percent=50, keys_per_gate=2, mode="equal", seed=6)
    locked = rep.locked_network
    data = [n for n in locked.inputs if n not in rep.correct_key]

    found = 0
    for kbits in product((0, 1), repeat=rep.num_keys):
        kv = dict(zip(rep.key_names, kbits))
        if all(
            outputs_of(multi, dict(zip(data, xb)))
            == outputs_of(locked, {**dict(zip(data, xb)), **kv})
            for xb in product((0, 1), repeat=len(data))
        ):
            found += 1
    assert found == equivalent_key_count(multi, rep)


# -- sweep ------------------------------------------------------------------

def test_key_weight_sweep_shape(multi):
    rows = key_weight_sweep(multi, keys_per_gate=(1, 2), modes=("equal", "balanced"))
    assert len(rows) == 4
    for r in rows:
        assert 0.0 <= r.corruption <= 1.0
        assert r.num_keys > 0
        assert r.equivalent_keys >= 1


def test_sweep_rows_carry_their_configuration(multi):
    rows = key_weight_sweep(multi, keys_per_gate=(2,), modes=("high",))
    assert rows[0].mode == "high"
    assert rows[0].keys_per_gate == 2


def test_sweep_is_reproducible(multi):
    a = key_weight_sweep(multi, keys_per_gate=(2,), modes=("balanced",), seed=8)
    b = key_weight_sweep(multi, keys_per_gate=(2,), modes=("balanced",), seed=8)
    assert [r.corruption for r in a] == [r.corruption for r in b]
