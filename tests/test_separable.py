"""
Threshold identification tests.

The strongest check here is the exhaustive one: the number of threshold
functions of n variables is a known sequence (OEIS A000609) -- 2, 4, 14, 104,
1882 for n = 0..4. Counting them by brute force over every truth table and
matching the published value validates the classifier in both directions at
once. A test that only checked known-positive cases would not catch a
classifier that accepts everything.
"""

from itertools import product

import pytest

from tlglock.separable import (
    gate_to_table,
    identify,
    is_threshold,
    is_unate_in,
    truth_bits,
)
from tlglock.thfile import ThGate

# OEIS A000609: threshold functions of n variables.
KNOWN_COUNTS = {1: 4, 2: 14, 3: 104, 4: 1882}


def table_of(n, fn):
    return [int(fn(bits)) for bits in truth_bits(n)]


def check(n, fn):
    """Identify a function and confirm the realisation matches its table."""
    table = table_of(n, fn)
    real = identify(table, n)
    if real is None:
        return None
    for bits, want in zip(truth_bits(n), table):
        assert real.eval_bits(bits) == want
    return real


# -- known threshold functions ---------------------------------------------

def test_and():
    r = check(3, all)
    assert r is not None and r.threshold == 3


def test_or():
    r = check(3, any)
    assert r is not None and r.threshold == 1


def test_majority3():
    r = check(3, lambda b: sum(b) >= 2)
    assert r is not None and r.weights == (1, 1, 1) and r.threshold == 2


def test_majority5():
    r = check(5, lambda b: sum(b) >= 3)
    assert r is not None and r.threshold == 3


@pytest.mark.parametrize("k", range(1, 7))
def test_k_of_6(k):
    r = check(6, lambda b: sum(b) >= k)
    assert r is not None and r.weights == (1,) * 6 and r.threshold == k


def test_negated_input():
    r = check(2, lambda b: b[0] and not b[1])
    assert r is not None
    assert r.weights[0] > 0 and r.weights[1] < 0


def test_all_negative_unate():
    r = check(3, lambda b: not any(b))
    assert r is not None and all(w <= 0 for w in r.weights)


def test_weighted_function_needs_unequal_weights():
    r = check(3, lambda b: b[0] or (b[1] and b[2]))
    assert r is not None
    assert abs(r.weights[0]) > abs(r.weights[1])


def test_constants():
    zero = identify([0] * 8, 3)
    one = identify([1] * 8, 3)
    assert zero is not None and all(w == 0 for w in zero.weights)
    assert one is not None and all(w == 0 for w in one.weights)
    assert all(zero.eval_bits(b) == 0 for b in truth_bits(3))
    assert all(one.eval_bits(b) == 1 for b in truth_bits(3))


def test_unused_variable_gets_zero_weight():
    r = check(3, lambda b: b[0] and b[1])
    assert r is not None and r.weights[2] == 0


def test_single_variable_projection():
    r = check(3, lambda b: b[1])
    assert r is not None
    assert r.weights[0] == 0 and r.weights[2] == 0 and r.weights[1] != 0


# -- known non-threshold functions -----------------------------------------

def test_xor_is_not_threshold():
    assert check(2, lambda b: b[0] ^ b[1]) is None
    assert check(3, lambda b: b[0] ^ b[1] ^ b[2]) is None


def test_parity_is_not_threshold():
    for n in (2, 3, 4, 5):
        assert not is_threshold(table_of(n, lambda b: sum(b) % 2), n)


def test_two_term_and_or_is_not_threshold():
    """(a&b)|(c&d) is the canonical smallest non-threshold unate function."""
    assert check(4, lambda b: (b[0] and b[1]) or (b[2] and b[3])) is None


def test_binate_function_rejected_early():
    """Mux is binate in its select, so unateness alone rules it out."""
    assert check(3, lambda b: b[1] if b[0] else b[2]) is None


# -- unateness --------------------------------------------------------------

def test_unateness_polarities():
    table = table_of(2, lambda b: b[0] and not b[1])
    assert is_unate_in(table, 2, 0) == 1
    assert is_unate_in(table, 2, 1) == -1


def test_unused_variable_reports_zero():
    table = table_of(2, lambda b: b[0])
    assert is_unate_in(table, 2, 1) == 0


def test_binate_variable_reports_none():
    table = table_of(2, lambda b: b[0] ^ b[1])
    assert is_unate_in(table, 2, 0) is None


# -- exhaustive validation --------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_exhaustive_threshold_function_count(n):
    """Match OEIS A000609 exactly."""
    count = sum(
        1
        for bits in product((0, 1), repeat=1 << n)
        if identify(list(bits), n) is not None
    )
    assert count == KNOWN_COUNTS[n]


@pytest.mark.slow
def test_exhaustive_count_four_variables():
    count = sum(
        1
        for bits in product((0, 1), repeat=16)
        if identify(list(bits), 4) is not None
    )
    assert count == KNOWN_COUNTS[4]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_realisation_matches_its_table(n):
    """Whenever identify() returns a gate, that gate must be correct."""
    for bits in product((0, 1), repeat=1 << n):
        real = identify(list(bits), n)
        if real is None:
            continue
        for i, pattern in enumerate(truth_bits(n)):
            assert real.eval_bits(pattern) == bits[i]


# -- round trip through ThGate ---------------------------------------------

@pytest.mark.parametrize(
    "weights,threshold",
    [
        ([1, 1, 1], 3),
        ([1, 1, 1], 2),
        ([2, 1, 1], 2),
        ([3, 2, 1, -2], 3),
        ([1, 1, 1, -2, 3], 4),
        ([-1, -1], -1),
        ([5, 3, 2, 1, 1, 1], 6),
    ],
)
def test_gate_table_roundtrip(weights, threshold):
    """Any real gate's table must identify back to an equivalent gate."""
    n = len(weights)
    gate = ThGate(
        inputs=[f"x{i}" for i in range(n)],
        output="z",
        weights=list(weights),
        threshold=threshold,
    )
    table = gate_to_table(gate)
    real = identify(table, n)
    assert real is not None
    for bits, want in zip(truth_bits(n), table):
        assert real.eval_bits(bits) == want


def test_to_gate_produces_usable_thgate():
    real = identify(table_of(3, lambda b: sum(b) >= 2), 3)
    gate = real.to_gate(["a", "b", "c"], "z")
    assert gate.eval({"a": 1, "b": 1, "c": 0}) == 1
    assert gate.eval({"a": 1, "b": 0, "c": 0}) == 0


def test_weights_are_reduced_to_lowest_terms():
    """(2,2,2)/T=4 and (1,1,1)/T=2 are the same gate; prefer the small one."""
    real = identify(table_of(3, lambda b: sum(b) >= 2), 3)
    assert real.weights == (1, 1, 1)


# -- errors -----------------------------------------------------------------

def test_wrong_table_length_rejected():
    with pytest.raises(ValueError, match="expected"):
        identify([0, 1, 1], 3)


def test_non_binary_table_rejected():
    with pytest.raises(ValueError, match="binary"):
        identify([0, 2, 1, 0], 2)
