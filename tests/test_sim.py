from itertools import product

import pytest

from tlglock.sim import outputs_of, simulate, truth_table
from tlglock.thfile import ThGate, ThNetwork, parse_th

from conftest import random_network


def _net(weights, threshold, n=None):
    n = n or len(weights)
    names = [f"x{i}" for i in range(n)]
    return ThNetwork(
        model="t",
        inputs=names,
        outputs=["z"],
        gates=[ThGate(inputs=names, output="z", weights=list(weights), threshold=threshold)],
    )


def test_and_gate_exhaustive():
    """w = (1,1,1), T = 3 is a 3-input AND (Eq. 5)."""
    net = _net([1, 1, 1], 3)
    for bits in product((0, 1), repeat=3):
        assign = dict(zip(net.inputs, bits))
        assert outputs_of(net, assign) == (int(all(bits)),)


def test_or_gate_exhaustive():
    net = _net([1, 1, 1], 1)
    for bits in product((0, 1), repeat=3):
        assign = dict(zip(net.inputs, bits))
        assert outputs_of(net, assign) == (int(any(bits)),)


def test_majority_gate_exhaustive():
    net = _net([1, 1, 1, 1, 1], 3)
    for bits in product((0, 1), repeat=5):
        assign = dict(zip(net.inputs, bits))
        assert outputs_of(net, assign) == (int(sum(bits) >= 3),)


def test_negative_weight_gate():
    """w = (2,-1), T = 1 fires iff x0 and not x1... plus x0 alone."""
    net = _net([2, -1], 1)
    expect = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 1}
    for bits, want in expect.items():
        assert outputs_of(net, dict(zip(net.inputs, bits))) == (want,)


def test_fig3_gate_matches_hand_computation(fig3):
    """3*K1 + 2*X1 + 1*X2 + 3*X3 + 2*Y2 + 1*Y3 - 2*K2 >= 7."""
    assign = {"K1": 1, "X1": 1, "X2": 0, "X3": 1, "Y2": 0, "Y3": 0, "K2": 0}
    assert 3 + 2 + 3 == 8 >= 7
    assert outputs_of(fig3, assign) == (1,)

    assign["K2"] = 1  # subtract 2 -> 6 < 7
    assert outputs_of(fig3, assign) == (0,)


def test_threshold_boundary_is_inclusive():
    """out = 1 iff sum >= T, so sum == T must fire."""
    net = _net([2, 3], 5)
    assert outputs_of(net, {"x0": 1, "x1": 1}) == (1,)
    assert outputs_of(net, {"x0": 0, "x1": 1}) == (0,)


def test_multilevel_network(multi):
    # a=1,b=1 -> n1=1 ; c=1,d=0 -> n2=1 ; F = n1+n2>=2 = 1
    vals = simulate(multi, {"a": 1, "b": 1, "c": 1, "d": 0})
    assert vals["n1"] == 1 and vals["n2"] == 1 and vals["F"] == 1
    # G = 2*a - n2 >= 1 -> 2 - 1 = 1 >= 1 -> 1
    assert vals["G"] == 1


def test_unspecified_inputs_default_to_zero(multi):
    vals = simulate(multi, {"a": 1})
    assert vals["b"] == 0 and vals["c"] == 0 and vals["d"] == 0


def test_unknown_signal_in_assignment_rejected(eq5):
    with pytest.raises(KeyError, match="unknown"):
        simulate(eq5, {"X1": 1, "TYPO": 1})


def test_non_binary_value_rejected(eq5):
    with pytest.raises(ValueError, match="not binary"):
        simulate(eq5, {"X1": 2})


def test_loop_detected_at_simulation():
    net = ThNetwork(
        model="loop",
        inputs=["a"],
        outputs=["p"],
        gates=[
            ThGate(inputs=["a", "q"], output="p", weights=[1, 1], threshold=1),
            ThGate(inputs=["p"], output="q", weights=[1], threshold=1),
        ],
    )
    with pytest.raises(Exception):
        simulate(net, {"a": 1})


def test_truth_table_shape(multi):
    rows = truth_table(multi)
    assert len(rows) == 16
    assert all(len(r[0]) == 4 and len(r[1]) == 2 for r in rows)


def test_truth_table_with_fixed_signals(eq5):
    rows = truth_table(eq5, over=["X1", "X2"], fixed={"X3": 1})
    assert len(rows) == 4
    # only X1=X2=1 (with X3 pinned high) reaches the threshold of 3
    assert dict(rows)[(1, 1)] == (1,)
    assert dict(rows)[(1, 0)] == (0,)


def test_weighted_sum_helper(fig3):
    g = fig3.gates[0]
    assign = {"K1": 1, "X1": 1, "X2": 1, "X3": 0, "Y2": 0, "Y3": 0, "K2": 1}
    assert g.weighted_sum(assign) == 3 + 2 + 1 - 2


@pytest.mark.parametrize("seed", range(20))
def test_random_networks_are_deterministic(seed):
    """Same assignment must give the same response every time."""
    net = random_network(seed)
    assign = {n: (i % 2) for i, n in enumerate(net.inputs)}
    first = outputs_of(net, assign)
    for _ in range(3):
        assert outputs_of(net, assign) == first


@pytest.mark.parametrize("seed", range(20))
def test_random_networks_match_direct_evaluation(seed):
    """Levelised simulation must agree with naive per-gate evaluation."""
    net = random_network(seed)
    for bits in product((0, 1), repeat=len(net.inputs)):
        assign = dict(zip(net.inputs, bits))
        vals = dict(assign)
        for g in net.topological_order():
            total = sum(w * vals[n] for n, w in zip(g.inputs, g.weights))
            vals[g.output] = 1 if total >= g.threshold else 0
        assert outputs_of(net, assign) == tuple(vals[o] for o in net.outputs)
