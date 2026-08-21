"""
Table I regression data and internal-consistency checks.

The flow cannot yet reproduce Table I end to end -- that needs the ABC driver
and MiniSAT+, both still to be written. What these tests *can* do now is check
the table against the numeric claims the paper's own prose makes about it, so
that any later regression run has a trustworthy target. Two of the abstract's
headline numbers are checked here.
"""

import csv
import pathlib

import pytest

GOLDEN = pathlib.Path(__file__).parent / "golden" / "table1.csv"


def load_table() -> list[dict]:
    with open(GOLDEN) as fh:
        rows = [r for r in fh if not r.startswith("#")]
    return list(csv.DictReader(rows))


@pytest.fixture(scope="module")
def table():
    return load_table()


def _f(row, key):
    return float(row[key])


def test_table_has_fifteen_benchmarks(table):
    assert len(table) == 15


def test_suites_are_the_ones_claimed(table):
    """The paper evaluates on ISCAS'85, ISCAS'89, ITC'99 and MCNC."""
    assert {r["suite"] for r in table} == {"ISCAS85", "ISCAS89", "ITC99", "MCNC"}


def test_results_are_valid_solver_outcomes(table):
    assert {r["result"] for r in table} <= {"SAT", "UNSAT", "TIMEOUT"}


def test_timeouts_have_no_solver_statistics(table):
    """A timeout means the solver never finished, so no conflict counts."""
    for r in table:
        if r["result"] == "TIMEOUT":
            assert r["conflicts"] == ""
            assert r["decisions"] == ""
            assert r["cpu_time_s"] == ""


def test_completed_runs_have_statistics(table):
    for r in table:
        if r["result"] in ("SAT", "UNSAT"):
            assert r["conflicts"] != "" and r["cpu_time_s"] != ""


def test_unsat_runs_have_zero_conflicts(table):
    """Both UNSAT cases were resolved by preprocessing alone."""
    for r in table:
        if r["result"] == "UNSAT":
            assert int(r["conflicts"]) == 0
            assert int(r["decisions"]) == 0


def test_large_designs_time_out(table):
    """Section IV-A calls out c1355, c7552, i10 and b17 as timeouts."""
    named = {"c1355", "c7552", "i10", "b17"}
    for r in table:
        if r["circuit"] in named:
            assert r["result"] == "TIMEOUT", f"{r['circuit']} should time out"


def test_c17_is_solved_quickly(table):
    """The smallest benchmark is expected to fall in well under a second."""
    row = next(r for r in table if r["circuit"] == "c17")
    assert row["result"] == "SAT"
    assert _f(row, "cpu_time_s") < 1.0


def test_crtl_beats_lctl_on_every_metric(table):
    """
    The paper's central PPA claim: CRTL is at least as good as LCTL on area,
    power and delay for every benchmark.
    """
    for r in table:
        for metric in ("area_um2", "power_uw", "delay_ns"):
            lctl, crtl = _f(r, f"lctl_{metric}"), _f(r, f"crtl_{metric}")
            assert crtl <= lctl, f"{r['circuit']}: CRTL {metric} worse than LCTL"


def _saving(row, metric):
    return 1 - _f(row, f"crtl_{metric}") / _f(row, f"lctl_{metric}")


def _best(table, metric, exclude=()):
    rows = [r for r in table if r["circuit"] not in exclude]
    top = max(rows, key=lambda r: _saving(r, metric))
    return top["circuit"], _saving(top, metric)


def test_best_savings_are_where_we_think_they_are(table):
    """
    Pin the actual maxima, so a later data edit that moves them gets caught.
    c17 is listed separately: at 5 um^2 and 2.5 uW its ratios are dominated by
    rounding, and the paper's own "up to" figures clearly exclude it.
    """
    assert _best(table, "area_um2") == ("c17", 0.40)
    assert _best(table, "power_uw") == ("c17", 0.52)

    area = _best(table, "area_um2", exclude={"c17"})
    power = _best(table, "power_uw", exclude={"c17"})
    delay = _best(table, "delay_ns", exclude={"c17"})

    assert area[0] == "s386" and abs(area[1] - 0.333) < 0.001
    assert power[0] == "s713" and abs(power[1] - 0.288) < 0.001
    assert delay[0] == "c1908" and abs(delay[1] - 0.636) < 0.001


def test_conclusion_power_claim_matches_s713(table):
    """Conclusion: 'up to 29% lower power' -- s713, 28.8%. Consistent."""
    _, best = _best(table, "power_uw", exclude={"c17"})
    assert round(best * 100) == 29


def test_abstract_savings_are_conservative_not_overstated(table):
    """
    The abstract claims up to 30% area, 50% delay, 20% power. Every one of
    those is at or below the table's actual maximum, so the abstract
    understates rather than overstates. That is the safe direction, but the
    journal version could legitimately claim more.
    """
    for metric, claimed in (
        ("area_um2", 0.30),
        ("delay_ns", 0.50),
        ("power_uw", 0.20),
    ):
        _, actual = _best(table, metric, exclude={"c17"})
        assert actual >= claimed, (
            f"{metric}: abstract claims {claimed:.0%} but table maxes at {actual:.1%}"
        )


def test_conclusion_b17_area_figure_does_not_match_table_one(table):
    """
    DISCREPANCY. The conclusion states "on b17 it reduces area by 26% while
    retaining security". Table I gives b17 as 7200 -> 6100 um^2, a 15.3%
    reduction, not 26%.

    26% is not a stray number: it is what c1355, c1908, c2670, s1494, s526 and
    s5378 all reduce by, so the sentence most likely picked up the wrong row.
    b17's own standout figure is delay, at 54.5%.

    This test asserts the table, and fails deliberately if someone later edits
    the CSV to match the prose instead of fixing the prose. Section 5 of
    ch5_tlglock.tex carries the same sentence into the dissertation.
    """
    row = next(r for r in table if r["circuit"] == "b17")
    assert abs(_saving(row, "area_um2") - 0.153) < 0.001
    assert abs(_saving(row, "delay_ns") - 0.545) < 0.001

    at_26 = {
        r["circuit"] for r in table if abs(_saving(r, "area_um2") - 0.26) < 0.01
    }
    assert "b17" not in at_26
    assert at_26 == {"c1355", "c1908", "c2670", "s1494", "s526", "s5378"}


def test_conclusion_s386_crtl_numbers(table):
    """Conclusion: 'on s386/s526 it reaches 25 uW and 8 ns'."""
    row = next(r for r in table if r["circuit"] == "s386")
    assert _f(row, "crtl_power_uw") == 25.0
    assert _f(row, "crtl_delay_ns") == 8.0


def test_locking_percentages_are_in_range(table):
    for r in table:
        assert 0 < _f(r, "percent") <= 100


def test_key_counts_are_positive(table):
    for r in table:
        assert int(r["n_keys"]) > 0


def test_modified_comparator_flags_match_the_paper(table):
    """The paper stars des, b15 and b17 as using the 10x-width comparator."""
    starred = {r["circuit"] for r in table if r["comparator_10x"] == "1"}
    assert starred == {"des", "b15", "b17"}


@pytest.mark.golden
def test_reproduce_table1():
    """End-to-end regression. Needs the ABC driver and MiniSAT+."""
    pytest.skip("blocked on abc.py and the SAT attack loop -- see CLAUDE.md")
