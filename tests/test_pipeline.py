"""
End-to-end pipeline tests.

Each stage is unit-tested elsewhere; what these check is that the stages
compose. The property carried all the way through is that the original
netlist's function survives synthesis, collapse and locking, and that the
attack against the result recovers a key that reproduces it.
"""

from itertools import product

import pytest

from tlglock import (
    Status,
    collapse,
    lock,
    map_to_tlg,
    oracle_from,
    outputs_of,
    read_bench,
    sat_attack,
    verify_recovered_key,
)
from tlglock.cli import main
from tlglock.metrics import corruption_rate, is_equivalent_under_correct_key

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

MAJ = """\
INPUT(a)
INPUT(b)
INPUT(c)
OUTPUT(m)
x = AND(a, b)
y = AND(b, c)
z = AND(a, c)
m = OR(x, y, z)
"""

ADDER = """\
INPUT(a)
INPUT(b)
INPUT(cin)
OUTPUT(s)
OUTPUT(cout)
t = XOR(a, b)
s = XOR(t, cin)
u = AND(a, b)
v = AND(t, cin)
cout = OR(u, v)
"""


def golden_c17(bits):
    nand = lambda p, q: 1 - (p & q)
    g1, g2, g3, g6, g7 = bits
    n10, n11 = nand(g1, g3), nand(g3, g6)
    n16, n19 = nand(g2, n11), nand(n11, g7)
    return (nand(n10, n16), nand(n16, n19))


def golden_maj(bits):
    return (int(sum(bits) >= 2),)


def golden_adder(bits):
    a, b, cin = bits
    return (a ^ b ^ cin, int(a + b + cin >= 2))


CIRCUITS = [
    ("c17", C17, ["1", "2", "3", "6", "7"], golden_c17),
    ("maj", MAJ, ["a", "b", "c"], golden_maj),
    ("adder", ADDER, ["a", "b", "cin"], golden_adder),
]


@pytest.mark.parametrize("name,src,inputs,golden", CIRCUITS)
def test_synthesis_preserves_function(name, src, inputs, golden):
    net, _ = map_to_tlg(read_bench(src, name=name))
    for bits in product((0, 1), repeat=len(inputs)):
        assert outputs_of(net, dict(zip(inputs, bits))) == golden(bits)


@pytest.mark.parametrize("name,src,inputs,golden", CIRCUITS)
def test_collapse_preserves_function(name, src, inputs, golden):
    net, _ = map_to_tlg(read_bench(src, name=name))
    net, _ = collapse(net)
    for bits in product((0, 1), repeat=len(inputs)):
        assert outputs_of(net, dict(zip(inputs, bits))) == golden(bits)


@pytest.mark.parametrize("name,src,inputs,golden", CIRCUITS)
def test_locking_preserves_function_under_correct_key(name, src, inputs, golden):
    net, _ = map_to_tlg(read_bench(src, name=name))
    net, _ = collapse(net)
    report = lock(net, percent=100, keys_per_gate=2, seed=0)
    for bits in product((0, 1), repeat=len(inputs)):
        assign = dict(zip(inputs, bits))
        got = outputs_of(report.locked_network, {**assign, **report.correct_key})
        assert got == golden(bits)


@pytest.mark.parametrize("name,src,inputs,golden", CIRCUITS)
def test_full_pipeline_then_attack(name, src, inputs, golden):
    net, _ = map_to_tlg(read_bench(src, name=name))
    net, _ = collapse(net)
    report = lock(net, percent=100, keys_per_gate=2, mode="balanced", seed=0)

    assert is_equivalent_under_correct_key(net, report)
    assert corruption_rate(net, report) > 0.0

    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=120
    )
    assert res.status is Status.UNSAT
    assert verify_recovered_key(net, report.locked_network, res.key)


def test_c17_is_breakable_matching_table_one():
    """
    Table I reports c17 as solved rather than timed out. The flow here
    reproduces that qualitative result: c17 is small enough that the attack
    terminates and recovers a working key in a handful of iterations.

    The absolute counts are not comparable -- the paper used MiniSAT+ and this
    uses the built-in DPLL -- and the paper's own Result column is ambiguous
    (see CLAUDE.md finding 4), so only the "breakable, quickly" claim is
    asserted.
    """
    net, _ = map_to_tlg(read_bench(C17, name="c17"))
    net, _ = collapse(net)
    report = lock(net, percent=50, keys_per_gate=2, seed=0)
    res = sat_attack(
        report.locked_network, report.key_names, oracle_from(net), timeout=120
    )
    assert res.status is Status.UNSAT
    assert res.key is not None
    assert res.iterations < 32


@pytest.mark.parametrize("percent", [25, 50, 75, 100])
def test_pipeline_at_each_locking_percentage(percent):
    net, _ = map_to_tlg(read_bench(C17, name="c17"))
    net, _ = collapse(net)
    report = lock(net, percent=percent, keys_per_gate=2, seed=0)
    assert is_equivalent_under_correct_key(net, report)


def test_pipeline_is_deterministic():
    def once():
        net, _ = map_to_tlg(read_bench(C17, name="c17"))
        net, _ = collapse(net)
        return lock(net, percent=100, keys_per_gate=2, seed=7)

    a, b = once(), once()
    assert a.locked_network.to_text() == b.locked_network.to_text()
    assert a.key_string == b.key_string


# -- CLI --------------------------------------------------------------------


@pytest.fixture
def bench_file(tmp_path):
    p = tmp_path / "c17.bench"
    p.write_text(C17)
    return p


def test_cli_synth_writes_netlist(bench_file, tmp_path):
    out = tmp_path / "c17.th"
    assert main(["synth", str(bench_file), "-o", str(out)]) == 0
    assert out.exists()
    from tlglock import read_th

    read_th(str(out)).validate()


def test_cli_synth_to_stdout(bench_file, capsys):
    assert main(["synth", str(bench_file)]) == 0
    assert ".threshold" in capsys.readouterr().out


def test_cli_lock_roundtrip(bench_file, tmp_path):
    th = tmp_path / "c17.th"
    locked = tmp_path / "c17_locked.th"
    keyf = tmp_path / "key.txt"
    main(["synth", str(bench_file), "-o", str(th)])
    assert (
        main(
            [
                "lock", str(th), "-o", str(locked),
                "--percent", "100", "--keys", "2", "--key-file", str(keyf),
            ]
        )
        == 0
    )
    assert locked.exists() and keyf.exists()
    assert set(keyf.read_text().strip()) <= {"0", "1"}


def test_cli_attack_recovers_key(bench_file, tmp_path, capsys):
    th = tmp_path / "c17.th"
    locked = tmp_path / "c17_locked.th"
    main(["synth", str(bench_file), "-o", str(th)])
    main(["lock", str(th), "-o", str(locked), "--percent", "100", "--keys", "1"])
    capsys.readouterr()

    assert (
        main(["attack", str(locked), "--original", str(th), "--timeout", "60"]) == 0
    )
    out = capsys.readouterr().out
    assert "equivalent: True" in out


def test_cli_attack_rejects_unlocked_input(bench_file, tmp_path):
    th = tmp_path / "c17.th"
    main(["synth", str(bench_file), "-o", str(th)])
    assert main(["attack", str(th), "--original", str(th)]) == 2


def test_cli_run_emits_a_table_row(bench_file, capsys):
    assert (
        main(
            ["run", str(bench_file), "--percent", "100", "--keys", "2",
             "--timeout", "60"]
        )
        == 0
    )
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("circuit,n_keys,percent")
    fields = out[1].split(",")
    assert fields[0] == "c17"
    assert fields[6] in ("BROKEN", "Timeout", "NO_KEY")


def test_cli_run_reports_timeout_cleanly(bench_file, capsys):
    main(
        ["run", str(bench_file), "--percent", "100", "--keys", "4",
         "--timeout", "0.001"]
    )
    row = capsys.readouterr().out.strip().splitlines()[1]
    assert "Timeout" in row and "---" in row


def test_cli_rejects_missing_abc(bench_file):
    from tlglock import abc_available

    if abc_available():
        pytest.skip("abc is installed in this environment")
    assert main(["synth", str(bench_file), "--abc", "yes"]) == 2
