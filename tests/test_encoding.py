"""
Encoding-size tests.

The claim under test is comparative, so the tests are too: for the same gate,
PB is constant, CNF without auxiliary variables is exponential in fan-in, and
CNF with auxiliary variables is polynomial. A measurement that did not
separate those three would make "clause blowup" unfalsifiable.

The direct-CNF counts are checked against the closed form for k-of-n, where
the prime implicants of f and ~f number exactly C(n,k) and C(n,k-1). That
pins the counter to something independent of this implementation.
"""

from itertools import product
from math import comb

import pytest

from tlglock.abc import map_to_tlg, read_bench
from tlglock.collapse import collapse
from tlglock.encoding import (
    GateEncoding,
    format_report,
    gate_encoding,
    key_equations,
    network_encoding,
)
from tlglock.locking import lock
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


def k_of_n(n, k):
    return ThGate(
        inputs=[f"x{i}" for i in range(n)],
        output="z",
        weights=[1] * n,
        threshold=k,
    )


# -- direct CNF counts against the closed form ------------------------------

@pytest.mark.parametrize("n", range(2, 10))
def test_k_of_n_matches_binomial(n):
    """Prime implicants of f and ~f are C(n,k) and C(n,k-1) exactly."""
    for k in range(1, n + 1):
        got = gate_encoding(k_of_n(n, k)).cnf_direct
        assert got == comb(n, k) + comb(n, k - 1), f"n={n} k={k}"


def test_direct_cnf_counts_prime_implicants_of_a_mixed_sign_gate():
    """
    Negative weights must be handled by polarity flip, not ignored. The gate
    below is the repaired Eq. 6 gate; its count is verified against a direct
    enumeration of prime implicants from the truth table.
    """
    gate = ThGate(
        inputs=["x1", "x2", "x3", "k1", "k2"],
        output="z",
        weights=[1, 1, 1, -2, 3],
        threshold=4,
    )
    n = gate.fanin
    table = [
        gate.eval(dict(zip(gate.inputs, bits)))
        for bits in product((0, 1), repeat=n)
    ]

    def implicants(want):
        pts = [b for b, v in zip(product((0, 1), repeat=n), table) if v == want]
        # A point is minimal (for want=1) / maximal (for want=0) in the order
        # induced by each variable's polarity.
        pol = [1 if w >= 0 else -1 for w in gate.weights]
        def le(a, b):
            return all(
                (x <= y) if p > 0 else (x >= y) for x, y, p in zip(a, b, pol)
            )
        return [
            p for p in pts
            if not any(q != p and le(q, p) for q in pts)
        ] if want else [
            p for p in pts
            if not any(q != p and le(p, q) for q in pts)
        ]

    expected = len(implicants(1)) + len(implicants(0))
    assert gate_encoding(gate).cnf_direct == expected


def test_constant_gate_has_no_clauses():
    assert gate_encoding(ThGate(inputs=[], output="c", weights=[], threshold=1)).cnf_direct == 0


# -- the three encodings separate -------------------------------------------

def test_pb_is_constant_in_fanin():
    for n in (2, 6, 12, 20, 40):
        assert gate_encoding(k_of_n(n, (n + 1) // 2)).pb_constraints == 2


def test_direct_cnf_is_exponential_and_aux_cnf_is_not():
    """
    The load-bearing comparison. At fan-in 12 the direct encoding is already
    three orders of magnitude larger than the auxiliary-variable one, and the
    gap widens monotonically -- so "TLG resists SAT because of clause blowup"
    is a statement about one encoding, not about threshold functions.
    """
    ratios = []
    for n in (8, 10, 12, 14):
        e = gate_encoding(k_of_n(n, (n + 1) // 2))
        ratios.append(e.cnf_direct / e.cnf_aux_clauses)
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] > 10


def test_aux_encoding_is_polynomial_in_fanin():
    """O(n*T): doubling fan-in on a majority gate must stay well under 8x."""
    small = gate_encoding(k_of_n(8, 4)).cnf_aux_clauses
    large = gate_encoding(k_of_n(16, 8)).cnf_aux_clauses
    assert large / small < 8.0


def test_wide_gate_is_reported_not_computed():
    e = gate_encoding(k_of_n(24, 12), max_fanin=16)
    assert e.cnf_direct is None
    assert e.blowup is None
    assert e.cnf_aux_clauses > 0     # the polynomial column still works


# -- network level ----------------------------------------------------------

def test_network_pb_count_is_two_per_gate():
    net = c17_tlg()
    r = network_encoding(net)
    assert r.pb_constraints == 2 * len(net.gates)
    assert len(r.gates) == len(net.gates)


def test_network_totals_sum_the_gates():
    net = c17_tlg()
    r = network_encoding(net)
    assert r.cnf_direct == sum(g.cnf_direct for g in r.gates)
    assert r.cnf_aux_clauses == sum(g.cnf_aux_clauses for g in r.gates)


def test_truncation_makes_the_total_unavailable():
    """A partial total would understate the count and read as a real number."""
    net = ThNetwork(
        model="wide",
        inputs=[f"x{i}" for i in range(20)],
        outputs=["z"],
        gates=[k_of_n(20, 10)],
    )
    net.gates[0].output = "z"
    r = network_encoding(net, max_fanin=8)
    assert r.truncated == ["z"]
    assert r.cnf_direct is None


def test_miter_grows_linearly_per_dip():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=0)
    r = network_encoding(report.locked_network, key_names=report.key_names)
    assert r.miter_pb_constraints > r.pb_constraints
    assert r.pb_constraints_per_dip > 0


def test_locking_increases_encoding_size():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=3, seed=0)
    before = network_encoding(net)
    after = network_encoding(report.locked_network)
    assert after.cnf_direct > before.cnf_direct
    assert after.pb_constraints == before.pb_constraints  # still 2 per gate


# -- key equations ----------------------------------------------------------

def test_alternating_weights_collapse_the_key_space():
    """
    Balanced mode gives +u/-u pairs, so many key vectors share a shift and the
    key space collapses. This is the per-gate form of the equivalence-class
    compression, and it is why a wrong key string can still be functionally
    correct.
    """
    gate = ThGate(
        inputs=["a", "k1", "k2", "k3", "k4"],
        output="z",
        weights=[1, 2, -2, 2, -2],
        threshold=1,
    )
    ke = key_equations(gate, ["k1", "k2", "k3", "k4"])
    assert ke.key_space == 16
    assert ke.distinct_equations < 16
    assert ke.compression > 1.0
    assert sum(ke.class_sizes.values()) == 16


def test_distinct_weights_do_not_collapse():
    """Powers of two make every subset sum unique, so nothing compresses."""
    gate = ThGate(
        inputs=["a", "k1", "k2", "k3"],
        output="z",
        weights=[1, 1, 2, 4],
        threshold=1,
    )
    ke = key_equations(gate, ["k1", "k2", "k3"])
    assert ke.distinct_equations == 8
    assert ke.compression == 1.0


def test_class_sizes_partition_the_key_space():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=3, mode="balanced", seed=0)
    for gate in report.locked_network.gates:
        if not (set(report.key_names) & set(gate.inputs)):
            continue
        ke = key_equations(gate, report.key_names)
        assert sum(ke.class_sizes.values()) == ke.key_space
        assert len(ke.class_sizes) == ke.distinct_equations


def test_key_equations_ignores_keys_on_other_gates():
    net = c17_tlg()
    report = lock(net, percent=50, keys_per_gate=2, seed=0)
    locked_outputs = set(report.locked_gates)
    for gate in report.locked_network.gates:
        ke = key_equations(gate, report.key_names)
        if gate.output in locked_outputs:
            assert len(ke.key_weights) == 2
        else:
            assert ke.key_weights == []
            assert ke.distinct_equations == 1


# -- report -----------------------------------------------------------------

def test_report_mentions_all_three_encodings():
    net = c17_tlg()
    report = lock(net, percent=100, keys_per_gate=2, seed=0)
    text = format_report(report.locked_network, key_names=report.key_names)
    for marker in ("PB encoding", "CNF, no aux", "CNF, with aux", "attack miter"):
        assert marker in text, marker


def test_report_works_without_keys():
    text = format_report(c17_tlg())
    assert "PB encoding" in text
    assert "attack miter" not in text
