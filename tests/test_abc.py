"""
Synthesis front-end tests.

The binding correctness property throughout is that the mapped TLG network
computes the same function as the source netlist. Every mapping test checks
that exhaustively against an independent evaluation of the input file rather
than against the mapper's own intermediate state.
"""

from itertools import product

import pytest

from tlglock.abc import (
    Aig,
    SynthError,
    cut_function,
    enumerate_cuts,
    map_to_tlg,
    read_bench,
    read_blif,
)
from tlglock.separable import identify
from tlglock.sim import outputs_of

C17 = """\
# ISCAS'85 c17
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

MAJ_BLIF = """\
.model maj3
.inputs a b c
.outputs f
.names a b c f
11- 1
1-1 1
-11 1
.end
"""

XOR_BENCH = """\
INPUT(a)
INPUT(b)
OUTPUT(z)
z = XOR(a, b)
"""


def c17_golden(bits):
    nand = lambda p, q: 1 - (p & q)
    g1, g2, g3, g6, g7 = bits
    n10 = nand(g1, g3)
    n11 = nand(g3, g6)
    n16 = nand(g2, n11)
    n19 = nand(n11, g7)
    return (nand(n10, n16), nand(n16, n19))


def assert_matches(net, names, golden):
    for bits in product((0, 1), repeat=len(names)):
        assign = dict(zip(names, bits))
        assert outputs_of(net, assign) == golden(bits), f"mismatch at {bits}"


# -- AIG --------------------------------------------------------------------

def test_aig_constant_folding():
    aig = Aig()
    a = aig.add_pi("a")
    assert aig.add_and(a, Aig.CONST0) == Aig.CONST0
    assert aig.add_and(a, Aig.CONST1) == a
    assert aig.add_and(a, a) == a
    assert aig.add_and(a, a ^ 1) == Aig.CONST0


def test_aig_structural_hashing():
    aig = Aig()
    a, b = aig.add_pi("a"), aig.add_pi("b")
    before = aig.num_ands
    x = aig.add_and(a, b)
    y = aig.add_and(b, a)
    assert x == y
    assert aig.num_ands == before + 1


def test_aig_or_and_xor():
    aig = Aig()
    a, b = aig.add_pi("a"), aig.add_pi("b")
    for lits, fn in (
        (aig.add_or(a, b), lambda p, q: p | q),
        (aig.add_and(a, b), lambda p, q: p & q),
        (aig.add_xor(a, b), lambda p, q: p ^ q),
    ):
        for pv, qv in product((0, 1), repeat=2):
            vals = {a >> 1: pv, b >> 1: qv}
            assert aig.simulate_lit(lits, vals) == fn(pv, qv)


def test_levels():
    aig = Aig()
    a, b, c = aig.add_pi("a"), aig.add_pi("b"), aig.add_pi("c")
    x = aig.add_and(a, b)
    y = aig.add_and(x, c)
    lev = aig.levels()
    assert lev[a >> 1] == 0
    assert lev[x >> 1] == 1
    assert lev[y >> 1] == 2


# -- bench reader -----------------------------------------------------------

def test_read_c17():
    aig = read_bench(C17, name="c17")
    assert aig.pi_names == ["1", "2", "3", "6", "7"]
    assert aig.po_names == ["22", "23"]
    assert aig.num_ands > 0


def test_bench_gate_semantics():
    for op, fn in (
        ("AND", lambda p, q: p & q),
        ("OR", lambda p, q: p | q),
        ("NAND", lambda p, q: 1 - (p & q)),
        ("NOR", lambda p, q: 1 - (p | q)),
        ("XOR", lambda p, q: p ^ q),
        ("XNOR", lambda p, q: 1 - (p ^ q)),
    ):
        src = f"INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = {op}(a, b)\n"
        net, _ = map_to_tlg(read_bench(src))
        for pv, qv in product((0, 1), repeat=2):
            assert outputs_of(net, {"a": pv, "b": qv}) == (fn(pv, qv),), op


def test_bench_not_and_buf():
    for op, fn in (("NOT", lambda p: 1 - p), ("BUF", lambda p: p)):
        src = f"INPUT(a)\nOUTPUT(z)\nz = {op}(a)\n"
        net, _ = map_to_tlg(read_bench(src))
        for v in (0, 1):
            assert outputs_of(net, {"a": v}) == (fn(v),), op


def test_bench_multi_input_gate():
    src = "INPUT(a)\nINPUT(b)\nINPUT(c)\nOUTPUT(z)\nz = AND(a, b, c)\n"
    net, _ = map_to_tlg(read_bench(src))
    for bits in product((0, 1), repeat=3):
        assert outputs_of(net, dict(zip("abc", bits))) == (int(all(bits)),)


def test_bench_out_of_order_definitions():
    src = "INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = AND(t, b)\nt = NOT(a)\n"
    net, _ = map_to_tlg(read_bench(src))
    for pv, qv in product((0, 1), repeat=2):
        assert outputs_of(net, {"a": pv, "b": qv}) == ((1 - pv) & qv,)


def test_bench_comments_ignored():
    aig = read_bench("# hello\nINPUT(a)  # inline\nOUTPUT(a)\n")
    assert aig.pi_names == ["a"]


def test_bench_rejects_sequential():
    with pytest.raises(SynthError, match="combinational"):
        read_bench("INPUT(a)\nOUTPUT(z)\nz = DFF(a)\n")


def test_bench_rejects_undriven_output():
    with pytest.raises(SynthError, match="never driven"):
        read_bench("INPUT(a)\nOUTPUT(q)\n")


def test_bench_rejects_cycle():
    with pytest.raises(SynthError, match="cyclic"):
        read_bench("INPUT(a)\nOUTPUT(z)\nz = AND(a, y)\ny = AND(z, a)\n")


def test_bench_rejects_unknown_gate():
    with pytest.raises(SynthError, match="unsupported"):
        read_bench("INPUT(a)\nOUTPUT(z)\nz = MAJORITY(a, a, a)\n")


def test_bench_rejects_wrong_arity():
    with pytest.raises(SynthError, match="needs 1"):
        read_bench("INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = NOT(a, b)\n")


# -- blif reader ------------------------------------------------------------

def test_read_blif_majority():
    net, _ = map_to_tlg(read_blif(MAJ_BLIF))
    for bits in product((0, 1), repeat=3):
        assert outputs_of(net, dict(zip("abc", bits))) == (int(sum(bits) >= 2),)


def test_blif_dont_care_cube():
    src = ".model m\n.inputs a b\n.outputs f\n.names a b f\n1- 1\n.end\n"
    net, _ = map_to_tlg(read_blif(src))
    for pv, qv in product((0, 1), repeat=2):
        assert outputs_of(net, {"a": pv, "b": qv}) == (pv,)


def test_blif_offset_cover():
    """A cover with '0' output values describes the off-set."""
    src = ".model m\n.inputs a b\n.outputs f\n.names a b f\n11 0\n.end\n"
    net, _ = map_to_tlg(read_blif(src))
    for pv, qv in product((0, 1), repeat=2):
        assert outputs_of(net, {"a": pv, "b": qv}) == (1 - (pv & qv),)


def test_blif_line_continuation():
    src = ".model m\n.inputs a \\\n b\n.outputs f\n.names a b f\n11 1\n.end\n"
    net, _ = map_to_tlg(read_blif(src))
    assert outputs_of(net, {"a": 1, "b": 1}) == (1,)


def test_blif_rejects_latch():
    with pytest.raises(SynthError, match="combinational"):
        read_blif(".model m\n.inputs a\n.outputs q\n.latch a q 0\n.end\n")


def test_blif_rejects_bad_cube_char():
    src = ".model m\n.inputs a\n.outputs f\n.names a f\nX 1\n.end\n"
    with pytest.raises(SynthError, match="bad cube"):
        read_blif(src)


# -- cuts -------------------------------------------------------------------

def test_cuts_are_k_feasible():
    aig = read_bench(C17)
    for k in (3, 4, 6):
        for node, cuts in enumerate_cuts(aig, k=k).items():
            for cut in cuts:
                assert len(cut) <= k


def test_trivial_cut_always_present():
    aig = read_bench(C17)
    cuts = enumerate_cuts(aig, k=6)
    for node, cs in cuts.items():
        if node == 0:
            continue
        assert frozenset({node}) in cs


def test_cut_limit_respected():
    aig = read_bench(C17)
    for node, cs in enumerate_cuts(aig, k=6, limit=4).items():
        assert len(cs) <= 4


def test_larger_k_gives_at_least_as_many_cuts():
    aig = read_bench(C17)
    small = enumerate_cuts(aig, k=3, limit=99)
    large = enumerate_cuts(aig, k=6, limit=99)
    assert sum(len(v) for v in large.values()) >= sum(len(v) for v in small.values())


def test_cut_function_matches_simulation():
    """A cut's tabulated function must match direct AIG evaluation."""
    aig = read_bench(C17)
    cuts = enumerate_cuts(aig, k=4)
    for node in aig.topo_ids():
        for cut in cuts[node]:
            if cut == frozenset({node}):
                continue
            support = sorted(cut)
            table = cut_function(aig, node, cut)
            for idx, bits in enumerate(product((0, 1), repeat=len(support))):
                vals = dict(zip(support, bits))
                assert aig.simulate_lit(node << 1, dict(vals)) == table[idx]


def test_bad_k_rejected():
    with pytest.raises(ValueError):
        enumerate_cuts(read_bench(C17), k=0)


# -- mapping ----------------------------------------------------------------

def test_c17_maps_correctly():
    net, stats = map_to_tlg(read_bench(C17, name="c17"))
    net.validate()
    assert_matches(net, ["1", "2", "3", "6", "7"], c17_golden)
    assert stats.gates > 0 and stats.depth > 0


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_c17_correct_at_every_cut_size(k):
    net, _ = map_to_tlg(read_bench(C17), k=k)
    assert_matches(net, ["1", "2", "3", "6", "7"], c17_golden)


def test_larger_cuts_do_not_increase_gate_count():
    small, _ = map_to_tlg(read_bench(C17), k=2)
    large, _ = map_to_tlg(read_bench(C17), k=6)
    assert len(large.gates) <= len(small.gates)


def test_mapped_network_roundtrips():
    from tlglock.thfile import parse_th
    net, _ = map_to_tlg(read_bench(C17))
    text = net.to_text()
    assert parse_th(text).to_text() == text


def test_xor_maps_despite_not_being_a_single_threshold_gate():
    """XOR is not linearly separable, so it must map to more than one TLG."""
    net, _ = map_to_tlg(read_bench(XOR_BENCH))
    for pv, qv in product((0, 1), repeat=2):
        assert outputs_of(net, {"a": pv, "b": qv}) == (pv ^ qv,)
    assert len([g for g in net.gates if g.fanin >= 2]) >= 2


def test_every_emitted_gate_is_a_real_threshold_function():
    """Sanity: the mapper must not emit a gate identify() would reject."""
    from tlglock.separable import gate_to_table
    net, _ = map_to_tlg(read_bench(C17))
    for g in net.gates:
        if g.fanin == 0 or g.fanin > 10:
            continue
        assert identify(gate_to_table(g), g.fanin) is not None


def test_inverted_output_handled():
    src = "INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = NAND(a, b)\n"
    net, _ = map_to_tlg(read_bench(src))
    for pv, qv in product((0, 1), repeat=2):
        assert outputs_of(net, {"a": pv, "b": qv}) == (1 - (pv & qv),)


def test_output_driven_by_primary_input():
    src = "INPUT(a)\nINPUT(b)\nOUTPUT(z)\nz = BUF(a)\n"
    net, _ = map_to_tlg(read_bench(src))
    assert outputs_of(net, {"a": 1, "b": 0}) == (1,)
    assert outputs_of(net, {"a": 0, "b": 1}) == (0,)


def test_stats_are_consistent():
    net, stats = map_to_tlg(read_bench(C17))
    assert stats.gates == len(net.gates)
    assert stats.cuts_separable <= stats.cuts_tried
    assert 0.0 <= stats.separable_fraction <= 1.0
    assert stats.max_fanin <= 6
