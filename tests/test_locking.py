import random

import pytest

from tlglock.locking import (
    assign_key_weights,
    generate_key,
    lock,
    select_lock_gates,
)
from tlglock.metrics import is_equivalent_under_correct_key
from tlglock.sim import outputs_of
from tlglock.thfile import ThGate

from conftest import random_network


# -- Algorithm 1: gate selection -------------------------------------------

@pytest.mark.parametrize(
    "percent,expected",
    [(0, 0), (25, 1), (50, 2), (75, 3), (100, 4)],
)
def test_selection_count_follows_percent(multi, percent, expected):
    """|G_lock| = |G_TLG| * P/100."""
    assert len(select_lock_gates(multi, percent)) == expected


def test_tiny_percent_still_locks_one_gate(multi):
    """Rounding to zero would silently produce an unlocked circuit."""
    assert len(select_lock_gates(multi, 1)) == 1


def test_zero_percent_locks_nothing(multi):
    assert select_lock_gates(multi, 0) == []


def test_percent_out_of_range_rejected(multi):
    with pytest.raises(ValueError):
        select_lock_gates(multi, 101)
    with pytest.raises(ValueError):
        select_lock_gates(multi, -1)


def test_selection_never_exceeds_gate_count(multi):
    assert len(select_lock_gates(multi, 100)) == len(multi.gates)


def test_fanin_strategy_prefers_wide_gates():
    net = random_network(1, n_in=5, n_gates=6)
    picked = select_lock_gates(net, 50, strategy="fanin")
    rest = [g for g in net.gates if g not in picked]
    assert min(g.fanin for g in picked) >= max(g.fanin for g in rest)


def test_fanout_strategy_prefers_driving_gates(multi):
    picked = select_lock_gates(multi, 25, strategy="fanout")
    assert picked[0].output == "n2"  # feeds both F and G


def test_first_strategy_is_deterministic(multi):
    a = select_lock_gates(multi, 50, strategy="first")
    b = select_lock_gates(multi, 50, strategy="first")
    assert [g.output for g in a] == [g.output for g in b] == ["n1", "n2"]


def test_random_strategy_respects_seed(multi):
    a = select_lock_gates(multi, 50, rng=random.Random(4), strategy="random")
    b = select_lock_gates(multi, 50, rng=random.Random(4), strategy="random")
    assert [g.output for g in a] == [g.output for g in b]


def test_unknown_strategy_rejected(multi):
    with pytest.raises(ValueError, match="unknown strategy"):
        select_lock_gates(multi, 50, strategy="vibes")


def test_empty_network_selects_nothing():
    from tlglock.thfile import ThNetwork
    assert select_lock_gates(ThNetwork(), 50) == []


# -- key generation ---------------------------------------------------------

def test_generate_key_length_and_domain():
    k = generate_key(16, rng=random.Random(0))
    assert len(k) == 16
    assert set(k) <= {0, 1}


def test_generate_key_is_seed_reproducible():
    assert generate_key(8, random.Random(9)) == generate_key(8, random.Random(9))


# -- key weight assignment --------------------------------------------------

@pytest.mark.parametrize("mode", ["equal", "balanced", "high", "random"])
def test_weight_modes_produce_nonzero_weights(mode):
    gate = ThGate(inputs=["a", "b", "c"], output="z", weights=[2, 2, 1], threshold=3)
    w = assign_key_weights(gate, 4, mode=mode, rng=random.Random(1))
    assert len(w) == 4
    assert all(v != 0 for v in w)


def test_equal_mode_is_uniform():
    gate = ThGate(inputs=["a", "b"], output="z", weights=[2, 2], threshold=2)
    w = assign_key_weights(gate, 4, mode="equal")
    assert len(set(w)) == 1


def test_balanced_mode_alternates_sign():
    gate = ThGate(inputs=["a", "b"], output="z", weights=[2, 2], threshold=2)
    w = assign_key_weights(gate, 4, mode="balanced")
    assert [v > 0 for v in w] == [True, False, True, False]


def test_high_mode_exceeds_equal_mode():
    gate = ThGate(inputs=["a", "b", "c"], output="z", weights=[3, 3, 3], threshold=5)
    eq = assign_key_weights(gate, 2, mode="equal")
    hi = assign_key_weights(gate, 2, mode="high")
    assert abs(hi[0]) > abs(eq[0])


def test_weights_scale_with_gate_weight_sum():
    """Flow step 5: key weights proportional to the input weight sum."""
    small = ThGate(inputs=["a", "b"], output="z", weights=[1, 1], threshold=1)
    large = ThGate(inputs=["a", "b"], output="z", weights=[8, 8], threshold=8)
    assert abs(assign_key_weights(large, 1, mode="equal")[0]) > abs(
        assign_key_weights(small, 1, mode="equal")[0]
    )


def test_unknown_mode_rejected():
    gate = ThGate(inputs=["a"], output="z", weights=[1], threshold=1)
    with pytest.raises(ValueError, match="mode must be"):
        assign_key_weights(gate, 2, mode="spicy")


# -- Algorithm 2: end-to-end locking ---------------------------------------

def test_lock_adds_key_inputs(multi):
    report = lock(multi, percent=50, keys_per_gate=2, seed=0)
    assert report.num_keys == 4
    for k in report.key_names:
        assert k in report.locked_network.inputs


def test_lock_preserves_original_inputs_and_outputs(multi):
    report = lock(multi, percent=100, keys_per_gate=1, seed=0)
    locked = report.locked_network
    for n in multi.inputs:
        assert n in locked.inputs
    assert locked.outputs == multi.outputs


def test_lock_does_not_mutate_the_original(multi):
    before = multi.to_text()
    lock(multi, percent=100, keys_per_gate=2, seed=0)
    assert multi.to_text() == before


def test_key_count_is_keys_per_gate_times_locked_gates(multi):
    report = lock(multi, percent=50, keys_per_gate=3, seed=0)
    assert report.num_keys == 3 * len(report.locked_gates)


def test_locked_network_validates(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    report.locked_network.validate()


def test_locked_network_roundtrips_through_th(multi):
    from tlglock.thfile import parse_th
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    text = report.locked_network.to_text()
    assert parse_th(text).to_text() == text


def test_key_string_matches_correct_key(multi):
    report = lock(multi, percent=50, keys_per_gate=2, seed=1)
    assert report.key_string == "".join(
        str(report.correct_key[k]) for k in report.key_names
    )


def test_threshold_shift_recorded(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    for gname, shift in report.threshold_shift.items():
        weights = report.key_weights[gname]
        expect = sum(v * report.correct_key[k] for k, v in weights.items())
        assert shift == expect


def test_locked_gate_thresholds_are_compensated(multi):
    report = lock(multi, percent=100, keys_per_gate=2, seed=0)
    orig = {g.output: g for g in multi.gates}
    for g in report.locked_network.gates:
        if g.output in report.threshold_shift:
            assert g.threshold == orig[g.output].threshold + report.threshold_shift[g.output]


def test_lock_is_seed_reproducible(multi):
    a = lock(multi, percent=50, keys_per_gate=2, seed=42)
    b = lock(multi, percent=50, keys_per_gate=2, seed=42)
    assert a.locked_network.to_text() == b.locked_network.to_text()
    assert a.correct_key == b.correct_key


def test_zero_percent_lock_is_a_noop(multi):
    report = lock(multi, percent=0, seed=0)
    assert report.num_keys == 0
    assert report.locked_network.gates[0].weights == multi.gates[0].weights


# -- the central invariant --------------------------------------------------

@pytest.mark.parametrize("mode", ["equal", "balanced", "high", "random"])
@pytest.mark.parametrize("keys_per_gate", [1, 2, 3])
def test_correct_key_restores_function_across_modes(multi, mode, keys_per_gate):
    report = lock(multi, percent=100, keys_per_gate=keys_per_gate, mode=mode, seed=5)
    assert is_equivalent_under_correct_key(multi, report)


@pytest.mark.parametrize("seed", range(15))
def test_correct_key_restores_function_on_random_networks(seed):
    net = random_network(seed, n_in=5, n_gates=5)
    report = lock(net, percent=100, keys_per_gate=2, mode="balanced", seed=seed)
    assert is_equivalent_under_correct_key(net, report)


@pytest.mark.parametrize("seed", range(10))
def test_exhaustive_equivalence_under_correct_key(seed):
    """Full truth-table comparison, not sampled."""
    from itertools import product
    net = random_network(seed, n_in=5, n_gates=4)
    report = lock(net, percent=100, keys_per_gate=2, seed=seed)
    locked = report.locked_network

    for bits in product((0, 1), repeat=len(net.inputs)):
        assign = dict(zip(net.inputs, bits))
        assert outputs_of(net, assign) == outputs_of(
            locked, {**assign, **report.correct_key}
        )


def test_verify_flag_catches_broken_compensation(multi, monkeypatch):
    """If compensation is disabled, lock() must refuse to return."""
    import tlglock.locking as L

    def broken(gate, key_names, key_weights, correct_bits):
        from tlglock.thfile import ThGate as G
        return (
            G(
                inputs=list(gate.inputs) + list(key_names),
                output=gate.output,
                weights=list(gate.weights) + list(key_weights),
                threshold=gate.threshold,  # no shift -- the Eq. 6 mistake
            ),
            0,
        )

    monkeypatch.setattr(L, "embed_keys", broken)
    with pytest.raises(AssertionError, match="not equivalent"):
        L.lock(multi, percent=100, keys_per_gate=2, seed=0)
