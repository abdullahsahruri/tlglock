"""
The Eq. (6) discrepancy, as an executable argument.

Section III of the paper introduces locking with a worked example. Eq. (5) is
an unlocked 3-input gate:

    Output = 1  iff  1*x1 + 1*x2 + 1*x3 >= 3                        (5)

Eq. (6) then adds key inputs k1, k2 with weights v1 = -2, v2 = 3, and states
that the correct key K = [1, 1] "neutralizes the effect of the added inputs,
ensuring circuit functionality":

    Output = 1  iff  1*x1 + 1*x2 + 1*x3 - 2*k1 + 3*k2 >= 2          (6)

Under K = [1, 1] the key terms contribute -2 + 3 = +1, which does not
neutralise: it shifts the effective threshold to 2 - 1 = 1. The locked gate
computes an OR where the original computed an AND.

These tests establish three things:
  1. Eq. (6) as printed is not equivalent to Eq. (5) under the stated key.
  2. Raising the threshold to 4 -- i.e. T' = T + sum_j v_j k*_j -- repairs it
     with the paper's own key weights, so 2 is most likely a typo for 4.
  3. Equivalence holds exactly when T - sum_j v_j k*_j equals the original
     threshold. That identity is the real content of the finding: the printed
     numbers violate it, and both "T should be 4" and "the weights should sum
     to -1" are single-digit repairs that satisfy it. The text alone does not
     determine which was intended.
"""

from itertools import product

import pytest

from tlglock.locking import embed_keys, lock
from tlglock.metrics import is_equivalent_under_correct_key
from tlglock.sim import outputs_of
from tlglock.thfile import ThGate, ThNetwork, parse_th

CORRECT_KEY = {"k1": 1, "k2": 1}
DATA = ["x1", "x2", "x3"]


def _eq5() -> ThNetwork:
    return ThNetwork(
        model="eq5",
        inputs=list(DATA),
        outputs=["z"],
        gates=[ThGate(inputs=list(DATA), output="z", weights=[1, 1, 1], threshold=3)],
    )


def _locked(threshold: int, v1: int = -2, v2: int = 3) -> ThNetwork:
    return ThNetwork(
        model="eq6",
        inputs=DATA + ["k1", "k2"],
        outputs=["z"],
        gates=[
            ThGate(
                inputs=DATA + ["k1", "k2"],
                output="z",
                weights=[1, 1, 1, v1, v2],
                threshold=threshold,
            )
        ],
    )


def _agrees(locked: ThNetwork, key: dict[str, int]) -> bool:
    orig = _eq5()
    for bits in product((0, 1), repeat=3):
        assign = dict(zip(DATA, bits))
        if outputs_of(orig, assign) != outputs_of(locked, {**assign, **key}):
            return False
    return True


def test_eq5_is_a_three_input_and():
    orig = _eq5()
    for bits in product((0, 1), repeat=3):
        assign = dict(zip(DATA, bits))
        assert outputs_of(orig, assign) == (int(all(bits)),)


def test_eq6_as_printed_is_not_equivalent_under_the_stated_key():
    """The headline finding: T = 2 does not preserve functionality."""
    assert not _agrees(_locked(threshold=2), CORRECT_KEY)


def test_eq6_as_printed_computes_or_not_and():
    """Specifically, the correct key turns the AND into an OR."""
    locked = _locked(threshold=2)
    for bits in product((0, 1), repeat=3):
        assign = dict(zip(DATA, bits))
        got = outputs_of(locked, {**assign, **CORRECT_KEY})
        assert got == (int(any(bits)),)


def test_compensated_threshold_repairs_the_example():
    """T' = T + sum_j v_j k*_j = 3 + (-2 + 3) = 4 works with the same weights."""
    assert _agrees(_locked(threshold=4), CORRECT_KEY)


def test_equivalence_holds_exactly_when_compensation_identity_holds():
    """
    The general condition, swept over both thresholds and key weights.

    Equivalence under k* holds iff the compensated threshold matches the
    original: T - sum_j v_j k*_j == 3. Nothing else about v1, v2 matters --
    only their weighted sum under the correct key.
    """
    for T, v1, v2 in product(range(-6, 10), range(-8, 9), range(-8, 9)):
        shift = v1 * CORRECT_KEY["k1"] + v2 * CORRECT_KEY["k2"]
        expected = (T - shift) == 3
        assert _agrees(_locked(T, v1, v2), CORRECT_KEY) is expected, (
            f"T={T}, v1={v1}, v2={v2}: identity says {expected}"
        )


def test_the_typo_could_be_either_the_threshold_or_the_weights():
    """
    Two minimal repairs exist, so the paper's intent is underdetermined.

    Keeping the printed weights (v1=-2, v2=3) forces T = 4. Keeping the
    printed T = 2 forces v1 + v2 = -1, e.g. (-2, 1) or (-3, 2). Both are
    single-digit edits, so the text alone cannot say which was meant --
    but every repair satisfies the same identity.
    """
    assert _agrees(_locked(4, -2, 3), CORRECT_KEY)      # threshold was the typo
    assert _agrees(_locked(2, -2, 1), CORRECT_KEY)      # v2 was the typo
    assert _agrees(_locked(2, -3, 2), CORRECT_KEY)      # v1 was the typo

    # ...and each satisfies T - sum(v_j k*_j) == 3.
    for T, v1, v2 in [(4, -2, 3), (2, -2, 1), (2, -3, 2)]:
        assert T - (v1 + v2) == 3


def test_no_threshold_rescues_the_printed_weights_except_four():
    """With v1=-2, v2=3 fixed as printed, T=4 is the unique repair."""
    working = [t for t in range(-6, 10) if _agrees(_locked(threshold=t), CORRECT_KEY)]
    assert working == [4]


def test_wrong_keys_corrupt_under_the_repaired_gate():
    """The repair must not neuter the lock -- wrong keys still misbehave."""
    locked = _locked(threshold=4)
    wrong = [k for k in product((0, 1), repeat=2) if k != (1, 1)]
    assert any(
        not _agrees(locked, dict(zip(("k1", "k2"), k))) for k in wrong
    )


def test_delta_matches_eq4():
    """Eq. 4: delta = sum_j v_j (k'_j - k_j)."""
    v = {"k1": -2, "k2": 3}
    for bits in product((0, 1), repeat=2):
        wrong = dict(zip(("k1", "k2"), bits))
        delta = sum(v[j] * (wrong[j] - CORRECT_KEY[j]) for j in v)
        locked = _locked(threshold=4)
        g = locked.gates[0]
        base = {"x1": 1, "x2": 1, "x3": 0}
        s_correct = g.weighted_sum({**base, **CORRECT_KEY})
        s_wrong = g.weighted_sum({**base, **wrong})
        assert s_wrong - s_correct == delta


# -- the implementation must not repeat the mistake -------------------------


def test_embed_keys_applies_compensation():
    gate = ThGate(inputs=list(DATA), output="z", weights=[1, 1, 1], threshold=3)
    locked, shift = embed_keys(gate, ["k1", "k2"], [-2, 3], [1, 1])
    assert shift == 1
    assert locked.threshold == 4
    assert locked.weights == [1, 1, 1, -2, 3]


def test_embed_keys_zero_shift_when_correct_key_is_all_zero():
    gate = ThGate(inputs=list(DATA), output="z", weights=[1, 1, 1], threshold=3)
    locked, shift = embed_keys(gate, ["k1", "k2"], [-2, 3], [0, 0])
    assert shift == 0
    assert locked.threshold == 3


def test_lock_verifies_equivalence_by_construction():
    net = _eq5()
    report = lock(net, percent=100, keys_per_gate=2, seed=7)
    assert is_equivalent_under_correct_key(net, report)


def test_key_name_collision_rejected():
    gate = ThGate(inputs=["x1", "k1"], output="z", weights=[1, 1], threshold=1)
    with pytest.raises(ValueError, match="collide"):
        embed_keys(gate, ["k1"], [2], [1])


def test_non_binary_correct_bits_rejected():
    gate = ThGate(inputs=list(DATA), output="z", weights=[1, 1, 1], threshold=3)
    with pytest.raises(ValueError, match="binary"):
        embed_keys(gate, ["k1"], [2], [3])
