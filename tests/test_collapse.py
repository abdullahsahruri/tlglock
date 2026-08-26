"""
Collapse tests.

The invariant that matters is that collapse never changes the function. Every
test that performs a merge checks equivalence exhaustively against the
pre-collapse network, because a merge pass that quietly alters behaviour is
far worse than one that misses opportunities.
"""

from itertools import product

import pytest

from tlglock.abc import map_to_tlg, read_bench
from tlglock.collapse import (
    CollapseStats,
    can_merge,
    collapse,
    compose,
    equivalent,
)
from tlglock.sim import outputs_of
from tlglock.thfile import ThGate, ThNetwork

from conftest import random_network

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


def chain_net():
    """n1 = a AND b ; F = n1 AND c. Collapses to a single 3-input AND."""
    return ThNetwork(
        model="chain",
        inputs=["a", "b", "c"],
        outputs=["F"],
        gates=[
            ThGate(inputs=["a", "b"], output="n1", weights=[1, 1], threshold=2),
            ThGate(inputs=["n1", "c"], output="F", weights=[1, 1], threshold=2),
        ],
    )


def xor_net():
    """F = a XOR b, built from threshold gates. Must NOT collapse to one."""
    return ThNetwork(
        model="xor",
        inputs=["a", "b"],
        outputs=["F"],
        gates=[
            ThGate(inputs=["a", "b"], output="n1", weights=[1, 1], threshold=1),
            ThGate(inputs=["a", "b"], output="n2", weights=[1, 1], threshold=2),
            ThGate(inputs=["n1", "n2"], output="F", weights=[1, -1], threshold=1),
        ],
    )


# -- compose ----------------------------------------------------------------

def test_compose_merges_and_chain():
    net = chain_net()
    driven = next(g for g in net.gates if g.output == "F")
    driver = next(g for g in net.gates if g.output == "n1")
    merged = compose(driven, driver)
    assert merged is not None
    assert set(merged.inputs) == {"a", "b", "c"}
    for bits in product((0, 1), repeat=3):
        assert merged.eval(dict(zip("abc", bits))) == int(all(bits))


def test_compose_preserves_function_exactly():
    net = chain_net()
    driven = next(g for g in net.gates if g.output == "F")
    driver = next(g for g in net.gates if g.output == "n1")
    merged = compose(driven, driver)
    for bits in product((0, 1), repeat=3):
        assign = dict(zip("abc", bits))
        assert merged.eval(assign) == outputs_of(net, assign)[0]


def test_compose_can_merge_into_xor_partially():
    """
    XOR is not a threshold function, but that does not mean none of its
    internal merges are possible. Absorbing the OR term leaves

        F = 1[-2*n2 + a + b >= 1]     with n2 = a AND b

    which is separable: n2=0 gives a OR b, n2=1 forces 0. So the merge is
    legitimate even though the overall function is not a single TLG.
    """
    net = xor_net()
    driven = next(g for g in net.gates if g.output == "F")
    driver = next(g for g in net.gates if g.output == "n1")
    merged = compose(driven, driver)
    assert merged is not None
    assert set(merged.inputs) == {"n2", "a", "b"}

    # Still computes what the original pair computed.
    for bits in product((0, 1), repeat=3):
        vals = dict(zip(["n2", "a", "b"], bits))
        want = driven.eval({"n1": driver.eval({"a": vals["a"], "b": vals["b"]}),
                            "n2": vals["n2"]})
        assert merged.eval(vals) == want


def test_compose_refuses_when_result_would_be_xor():
    """
    The merge that would produce XOR itself must be refused. Absorbing the
    AND term into the already-merged gate leaves a function of a and b alone,
    and that function is XOR -- not linearly separable, so compose() returns
    None rather than emitting a gate that does not compute it.
    """
    net = xor_net()
    merged, _ = collapse(net)
    assert len(merged.gates) == 2

    driven = next(g for g in merged.gates if g.output == "F")
    driver = next(g for g in merged.gates if g.output != "F")
    assert driver.output in driven.inputs
    assert compose(driven, driver) is None


def test_compose_respects_max_support():
    net = chain_net()
    driven = next(g for g in net.gates if g.output == "F")
    driver = next(g for g in net.gates if g.output == "n1")
    assert compose(driven, driver, max_support=2) is None
    assert compose(driven, driver, max_support=3) is not None


def test_compose_rejects_unrelated_gates():
    net = chain_net()
    driven = next(g for g in net.gates if g.output == "n1")
    driver = next(g for g in net.gates if g.output == "F")
    with pytest.raises(ValueError, match="not an input"):
        compose(driven, driver)


def test_compose_with_shared_inputs():
    """Driver and driven sharing an input must not duplicate it."""
    net = ThNetwork(
        model="shared",
        inputs=["a", "b"],
        outputs=["F"],
        gates=[
            ThGate(inputs=["a", "b"], output="n1", weights=[1, 1], threshold=2),
            ThGate(inputs=["n1", "a"], output="F", weights=[1, 1], threshold=2),
        ],
    )
    merged = compose(net.gates[1], net.gates[0])
    assert merged is not None
    assert sorted(merged.inputs) == ["a", "b"]
    for bits in product((0, 1), repeat=2):
        assert merged.eval(dict(zip("ab", bits))) == outputs_of(net, dict(zip("ab", bits)))[0]


# -- can_merge --------------------------------------------------------------

def test_cannot_merge_multi_fanout():
    net = ThNetwork(
        model="fan",
        inputs=["a", "b", "c"],
        outputs=["F", "G"],
        gates=[
            ThGate(inputs=["a", "b"], output="n1", weights=[1, 1], threshold=2),
            ThGate(inputs=["n1", "c"], output="F", weights=[1, 1], threshold=2),
            ThGate(inputs=["n1", "c"], output="G", weights=[1, 1], threshold=1),
        ],
    )
    assert not can_merge(net, net.gates[1], net.gates[0])


def test_cannot_merge_primary_output():
    net = chain_net()
    net.outputs.append("n1")
    assert not can_merge(net, net.gates[1], net.gates[0])


def test_can_merge_single_fanout():
    net = chain_net()
    assert can_merge(net, net.gates[1], net.gates[0])


# -- collapse ---------------------------------------------------------------

def test_collapse_chain_to_single_gate():
    net = chain_net()
    out, stats = collapse(net)
    assert len(out.gates) == 1
    assert stats.merges == 1
    assert equivalent(net, out)


def test_collapse_stops_short_of_a_single_xor_gate():
    """
    XOR collapses from three gates to two and then stops, because the final
    merge would have to produce XOR as a single threshold function. This is
    the case that distinguishes a correct collapse pass from one that merges
    whenever the support fits.
    """
    net = xor_net()
    out, stats = collapse(net)
    assert len(out.gates) == 2
    assert stats.merges == 1
    assert stats.rejected_binate >= 1
    assert equivalent(net, out)

    for bits in product((0, 1), repeat=2):
        assign = dict(zip("ab", bits))
        assert outputs_of(out, assign) == (bits[0] ^ bits[1],)


def test_collapse_preserves_c17():
    net, _ = map_to_tlg(read_bench(C17, name="c17"))
    out, stats = collapse(net)
    assert stats.merges > 0
    assert equivalent(net, out)
    assert stats.gates_after < stats.gates_before


def test_collapse_reduces_depth_on_c17():
    net, _ = map_to_tlg(read_bench(C17))
    out, stats = collapse(net)
    assert stats.depth_after <= stats.depth_before


def test_collapse_does_not_mutate_input():
    net = chain_net()
    before = net.to_text()
    collapse(net)
    assert net.to_text() == before


def test_collapse_result_validates():
    net, _ = map_to_tlg(read_bench(C17))
    out, _ = collapse(net)
    out.validate()


def test_collapse_is_idempotent():
    net, _ = map_to_tlg(read_bench(C17))
    once, _ = collapse(net)
    twice, stats = collapse(once)
    assert stats.merges == 0
    assert once.to_text() == twice.to_text()


def test_max_weight_bound_is_respected():
    net, _ = map_to_tlg(read_bench(C17))
    out, _ = collapse(net, max_weight=2)
    for g in out.gates:
        assert all(abs(w) <= 2 for w in g.weights)
    assert equivalent(net, out)


def test_max_support_bound_is_respected():
    net, _ = map_to_tlg(read_bench(C17))
    out, _ = collapse(net, max_support=3)
    for g in out.gates:
        assert g.fanin <= 3
    assert equivalent(net, out)


def test_tighter_support_bound_merges_no_more():
    net, _ = map_to_tlg(read_bench(C17))
    _, tight = collapse(net, max_support=3)
    _, loose = collapse(net, max_support=8)
    assert tight.merges <= loose.merges


@pytest.mark.parametrize("seed", range(12))
def test_collapse_preserves_random_networks(seed):
    net = random_network(seed, n_in=5, n_gates=5)
    out, stats = collapse(net)
    assert equivalent(net, out)
    assert stats.gates_after <= stats.gates_before


def test_stats_accounting():
    net, _ = map_to_tlg(read_bench(C17))
    out, stats = collapse(net)
    assert stats.gates_before == len(net.gates)
    assert stats.gates_after == len(out.gates)
    assert stats.merges == len(stats.merged_pairs)
    assert stats.gates_before - stats.gates_after == stats.merges
    assert 0.0 <= stats.gate_reduction < 1.0


def test_merged_pairs_name_removed_gates():
    net, _ = map_to_tlg(read_bench(C17))
    out, stats = collapse(net)
    remaining = {g.output for g in out.gates}
    for driver, _driven in stats.merged_pairs:
        assert driver not in remaining


# -- equivalent() -----------------------------------------------------------

def test_equivalent_detects_difference():
    a = chain_net()
    b = a.copy()
    b.gates[0].threshold = 1
    assert not equivalent(a, b)


def test_equivalent_rejects_mismatched_interface():
    a = chain_net()
    b = a.copy()
    b.inputs.append("extra")
    assert not equivalent(a, b)


def test_equivalent_refuses_huge_input_space():
    net = ThNetwork(
        model="big",
        inputs=[f"x{i}" for i in range(20)],
        outputs=["z"],
        gates=[
            ThGate(
                inputs=[f"x{i}" for i in range(20)],
                output="z",
                weights=[1] * 20,
                threshold=10,
            )
        ],
    )
    with pytest.raises(ValueError, match="too many"):
        equivalent(net, net.copy(), limit=1024)


# -- interaction with locking ----------------------------------------------

def test_collapsed_network_still_locks_correctly():
    from tlglock.locking import lock
    from tlglock.metrics import is_equivalent_under_correct_key

    net, _ = map_to_tlg(read_bench(C17))
    out, _ = collapse(net)
    report = lock(out, percent=100, keys_per_gate=2, seed=0)
    assert is_equivalent_under_correct_key(out, report)


def test_collapse_changes_key_count():
    """
    Table I's #Keys column is a percentage of the gate count, so collapse
    changes it. This is why the column cannot be reproduced by the mapper
    alone.
    """
    from tlglock.locking import lock

    net, _ = map_to_tlg(read_bench(C17))
    out, _ = collapse(net)
    before = lock(net, percent=100, keys_per_gate=2, seed=0).num_keys
    after = lock(out, percent=100, keys_per_gate=2, seed=0).num_keys
    assert after < before
