"""
Brute-force verification of the pseudo-Boolean encoding.

A PB encoding is only useful if its models correspond exactly to the
network's behaviour. The reification in encode_gate() has two implications
with slack coefficients derived from the gate's true weight bounds, and it is
easy to get a sign or a bound wrong in a way that makes one direction vacuous
-- the constraint still parses, the solver still runs, and the answer is
silently wrong.

So rather than eyeballing the algebra, these tests enumerate every assignment
over the encoded variables, check which ones satisfy all constraints, and
compare that model set against the simulator. Any sign error shows up as an
extra or missing model.
"""

from itertools import product

import pytest

from tlglock.locking import lock
from tlglock.opb import (
    Constraint,
    OpbEncoder,
    build_distinguishing_miter,
)
from tlglock.sim import outputs_of, simulate
from tlglock.thfile import ThGate, ThNetwork

from conftest import random_network


def satisfies(constraints: list[Constraint], assign: dict[int, int]) -> bool:
    """Does this variable assignment satisfy every constraint?"""
    for c in constraints:
        total = sum(coeff * assign.get(var, 0) for coeff, var in c.terms)
        if total < c.rhs:
            return False
    return True


def all_models(enc: OpbEncoder) -> list[dict[int, int]]:
    """Every satisfying assignment. Only for tiny instances."""
    n = enc.num_vars
    assert n <= 18, f"{n} variables is too many to brute force"
    out = []
    for bits in product((0, 1), repeat=n):
        assign = {i + 1: b for i, b in enumerate(bits)}
        if satisfies(enc.constraints, assign):
            out.append(assign)
    return out


def _single(weights, threshold):
    names = [f"x{i}" for i in range(len(weights))]
    return ThNetwork(
        model="t",
        inputs=names,
        outputs=["z"],
        gates=[ThGate(inputs=names, output="z", weights=list(weights), threshold=threshold)],
    )


# -- single-gate reification ------------------------------------------------

GATES = [
    ([1, 1, 1], 3),        # AND
    ([1, 1, 1], 1),        # OR
    ([1, 1, 1], 2),        # majority
    ([1, 1, 1, 1, 1], 3),  # 5-input majority
    ([2, -1], 1),          # mixed sign
    ([-1, -1], -1),        # all negative
    ([3, 2, 1, -2], 3),    # the Fig. 3 shape, truncated
    ([1, 1, 1, -2, 3], 4), # the repaired Eq. 6 gate
    ([5], 5),              # single input, tight
    ([1, 1], 0),           # trivially true
    ([1, 1], 3),           # trivially false
    ([-3, 4, -2, 1], -1),  # heavily mixed
]


@pytest.mark.parametrize("weights,threshold", GATES)
def test_gate_reification_models_match_simulation(weights, threshold):
    """
    The encoded models must be exactly the gate's truth table.

    This is the test that catches sign errors in either implication: a broken
    forward direction admits models where y=1 but the sum is below T; a broken
    reverse direction admits y=0 with the sum at or above T.
    """
    net = _single(weights, threshold)
    enc = OpbEncoder()
    for n in net.inputs:
        enc.var(n)
    enc.encode_network(net)

    models = all_models(enc)
    model_set = {
        tuple(m[enc.var(n)] for n in net.inputs) + (m[enc.var("z")],)
        for m in models
    }

    expected = set()
    for bits in product((0, 1), repeat=len(net.inputs)):
        assign = dict(zip(net.inputs, bits))
        expected.add(bits + (outputs_of(net, assign)[0],))

    assert model_set == expected
    # Exactly one model per input pattern: the encoding is functional.
    assert len(models) == 2 ** len(net.inputs)


@pytest.mark.parametrize("weights,threshold", GATES)
def test_no_model_has_wrong_output_polarity(weights, threshold):
    """Restate the invariant directly, so a failure names the direction."""
    net = _single(weights, threshold)
    enc = OpbEncoder()
    for n in net.inputs:
        enc.var(n)
    enc.encode_network(net)

    for m in all_models(enc):
        s = sum(w * m[enc.var(n)] for n, w in zip(net.inputs, weights))
        y = m[enc.var("z")]
        if y == 1:
            assert s >= threshold, f"forward broken: y=1 with sum {s} < T {threshold}"
        else:
            assert s < threshold, f"reverse broken: y=0 with sum {s} >= T {threshold}"


# -- multi-level networks ---------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_random_network_encoding_matches_simulation(seed):
    net = random_network(seed, n_in=4, n_gates=4)
    enc = OpbEncoder()
    for n in net.inputs:
        enc.var(n)
    enc.encode_network(net)

    models = all_models(enc)
    # One model per input pattern -- internal signals are fully determined.
    assert len(models) == 2 ** len(net.inputs)

    for m in models:
        assign = {n: m[enc.var(n)] for n in net.inputs}
        vals = simulate(net, assign)
        for sig, want in vals.items():
            assert m[enc.var(sig)] == want, f"signal {sig} disagrees"


def test_multilevel_fixture_encoding(multi):
    enc = OpbEncoder()
    for n in multi.inputs:
        enc.var(n)
    enc.encode_network(multi)
    assert len(all_models(enc)) == 16


# -- XOR reification --------------------------------------------------------

def test_xor_reification_exhaustive():
    enc = OpbEncoder()
    a, b, d = enc.var("a"), enc.var("b"), enc.var("d")
    enc.encode_xor(a, b, d)

    models = {(m[a], m[b], m[d]) for m in all_models(enc)}
    assert models == {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}


# -- fixing and constants ---------------------------------------------------

def test_fix_pins_value():
    enc = OpbEncoder()
    enc.var("a")
    enc.var("b")
    enc.fix("a", 1)
    enc.fix("b", 0)
    models = all_models(enc)
    assert len(models) == 1
    assert models[0] == {enc.var("a"): 1, enc.var("b"): 0}


def test_fix_rejects_non_binary():
    enc = OpbEncoder()
    with pytest.raises(ValueError):
        enc.fix("a", 7)


# -- miter ------------------------------------------------------------------

def test_miter_is_unsat_when_key_is_irrelevant():
    """
    A gate whose key weights are all zero has no distinguishing input:
    every key produces the same function, so the miter must be UNSAT.
    """
    net = ThNetwork(
        model="dead",
        inputs=["x1", "k1"],
        outputs=["z"],
        gates=[ThGate(inputs=["x1", "k1"], output="z", weights=[1, 0], threshold=1)],
    )
    enc = build_distinguishing_miter(net, ["k1"])
    assert all_models(enc) == []


def test_miter_is_sat_when_key_matters():
    """A real lock admits at least one distinguishing input pattern."""
    net = ThNetwork(
        model="live",
        inputs=["x1", "x2", "k1"],
        outputs=["z"],
        gates=[
            ThGate(inputs=["x1", "x2", "k1"], output="z", weights=[1, 1, 2], threshold=2)
        ],
    )
    enc = build_distinguishing_miter(net, ["k1"])
    models = all_models(enc)
    assert models, "expected a distinguishing input"

    # Every model must genuinely be a distinguishing input.
    for m in models:
        x = {n: m[enc.var(n)] for n in ("x1", "x2")}
        ka = m[enc.var("k1_A")]
        kb = m[enc.var("k1_B")]
        ra = outputs_of(net, {**x, "k1": ka})
        rb = outputs_of(net, {**x, "k1": kb})
        assert ra != rb, f"model is not distinguishing: {x}, {ka} vs {kb}"


def test_miter_shares_data_inputs():
    """Both copies must read the same x, or the miter is meaningless."""
    net = ThNetwork(
        model="s",
        inputs=["x1", "k1"],
        outputs=["z"],
        gates=[ThGate(inputs=["x1", "k1"], output="z", weights=[1, 2], threshold=2)],
    )
    enc = build_distinguishing_miter(net, ["k1"])
    assert "x1" in enc.var_map
    assert "x1_A" not in enc.var_map
    assert "x1_B" not in enc.var_map
    assert "k1_A" in enc.var_map and "k1_B" in enc.var_map


def test_miter_on_locked_network_finds_distinguishing_input(eq5):
    report = lock(eq5, percent=100, keys_per_gate=1, mode="balanced", seed=3)
    enc = build_distinguishing_miter(report.locked_network, report.key_names)
    assert enc.num_vars <= 18
    assert all_models(enc), "a working lock must admit a DIP"


# -- format -----------------------------------------------------------------

def test_opb_header_and_syntax(eq5):
    enc = OpbEncoder()
    for n in eq5.inputs:
        enc.var(n)
    enc.encode_network(eq5)
    text = enc.to_text()

    lines = [l for l in text.splitlines() if l and not l.startswith("*")]
    assert text.startswith(f"* #variable= {enc.num_vars} #constraint= {len(enc.constraints)}")
    for line in lines:
        assert line.endswith(";")
        assert ">=" in line


def test_opb_objective_line(eq5):
    enc = OpbEncoder()
    enc.encode_network(eq5)
    text = enc.to_text(objective=[(1, 1), (2, 2)])
    assert "min: +1 x1 +2 x2;" in text


def test_constraint_line_format():
    c = Constraint([(3, 1), (-2, 4)], 7)
    assert c.to_line() == "+3 x1 -2 x4 >= 7;"
