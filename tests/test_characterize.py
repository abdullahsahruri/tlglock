"""
Characterization tests.

ngspice is not installed in CI, so the runner is injectable and every test
here uses a fake one. That is not a limitation of the tests -- everything
except the subprocess call is deterministic text manipulation, and text is
exactly what can go wrong: a malformed .meas line, a missing node, a width
that does not scale with the weight.

The one thing these cannot check is whether the reconstructed LCTL and CRTL
topologies match Fig. 2 of the paper. They do not; see the note in spice.py.
"""

import re

import pytest

from tlglock.characterize import (
    CellResult,
    MeasurementError,
    SimulatorError,
    characterize,
    key_size_sweep,
    ngspice_available,
    parse_measurements,
    run_deck,
    table_i_rows,
    write_csv,
)
from tlglock.spice import (
    PTM45_HP,
    PTM130,
    TECHNOLOGIES,
    CellSpec,
    area_um2,
    build_deck,
    device_count,
    subckt,
    subckt_name,
    transistors,
    worst_case_stimulus,
)


def spec(cell="CRTL", weights=(3, 2, 1, 1), keys=(2, -2), threshold=3, **kw):
    return CellSpec(
        cell=cell,
        weights=list(weights),
        key_weights=list(keys),
        threshold=threshold,
        **kw,
    )


FAKE_MEAS = {"tpd": 1.5e-11, "trf": 8.0e-12, "iavg": 3.2e-05, "ipeak": 9.1e-04}


def fake_runner(deck: str) -> dict:
    return dict(FAKE_MEAS)


# -- spec validation --------------------------------------------------------

def test_rejects_unknown_cell():
    with pytest.raises(ValueError, match="unknown cell"):
        CellSpec(cell="MAGIC", weights=[1])


def test_rejects_empty_cell():
    with pytest.raises(ValueError, match="no inputs"):
        CellSpec(cell="LCTL", weights=[])


def test_rejects_bad_comparator_scale():
    with pytest.raises(ValueError, match="comparator_scale"):
        CellSpec(cell="LCTL", weights=[1], comparator_scale=0)


def test_input_names_and_fanin():
    s = spec(weights=(1, 1, 1), keys=(2, -2))
    assert s.input_names == ["X1", "X2", "X3", "K1", "K2"]
    assert s.fanin == 5
    assert s.total_weight == 3 + 4


# -- netlist structure ------------------------------------------------------

@pytest.mark.parametrize("cell", ["LCTL", "CRTL"])
def test_subckt_is_well_formed(cell):
    text = subckt(spec(cell=cell))
    assert text.startswith(f".subckt {subckt_name(spec(cell=cell))}")
    assert text.rstrip().endswith(".ends")
    for line in text.splitlines():
        if line.startswith("M"):
            parts = line.split()
            assert len(parts) == 8, line
            assert parts[6].startswith("W=") and parts[7].startswith("L=")


@pytest.mark.parametrize("cell", ["LCTL", "CRTL"])
def test_every_input_appears_in_the_netlist(cell):
    s = spec(cell=cell)
    text = subckt(s)
    for name in s.input_names:
        assert re.search(rf"\b{name}\b", text), f"{name} missing from netlist"


@pytest.mark.parametrize("cell", ["LCTL", "CRTL"])
def test_port_list_matches_instantiation(cell):
    s = spec(cell=cell)
    header = subckt(s).splitlines()[0].split()[2:]
    deck = build_deck(s, worst_case_stimulus(s))
    inst = next(l for l in deck.splitlines() if l.startswith("XDUT")).split()[1:-1]
    assert header == inst


def test_crtl_has_a_clock_port_and_lctl_does_not():
    assert " E " in subckt(spec(cell="CRTL")).splitlines()[0]
    assert " E " not in subckt(spec(cell="LCTL")).splitlines()[0]


def test_crtl_includes_the_charge_recycling_switch():
    """The mechanism the paper's energy claim rests on must be present."""
    text = subckt(spec(cell="CRTL"))
    assert "recycling" in text
    assert re.search(r"^M7 ", text, re.MULTILINE)


def test_weights_map_to_device_widths():
    """A weight of w must produce a device w times the unit width."""
    s = CellSpec(cell="CRTL", weights=[1, 4], key_weights=[], threshold=2)
    devices = {n: w for n, w, _ in transistors(s)}
    assert devices["Mx2"] == pytest.approx(4 * devices["Mx1"])


def test_negative_weights_go_to_the_reference_branch():
    """
    A negative weight subtracts from the sum, which is equivalent to adding
    to the threshold the sum must clear -- so it is realised on the reference
    side rather than as a device that cannot exist.
    """
    s = CellSpec(cell="CRTL", weights=[2], key_weights=[-2], threshold=1)
    lines = [l for l in subckt(s).splitlines() if l.startswith("Mx")]
    by_node = {l.split()[0]: l.split()[1] for l in lines}
    assert by_node["Mx1"] == "nsum"
    assert by_node["Mx2"] == "nref"


def test_comparator_scale_widens_devices():
    """The paper's starred rows use a 10x comparator."""
    base = area_um2(spec(comparator_scale=1.0))
    wide = area_um2(spec(comparator_scale=10.0))
    assert wide > base


@pytest.mark.parametrize("cell", ["LCTL", "CRTL"])
def test_device_count_grows_with_key_count(cell):
    counts = [
        device_count(
            CellSpec(
                cell=cell,
                weights=[3, 2, 1, 1],
                key_weights=[2] * k,
                threshold=3,
            )
        )
        for k in (0, 2, 4, 8)
    ]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


# -- area -------------------------------------------------------------------

def test_area_is_positive_and_scales():
    small = area_um2(CellSpec(cell="LCTL", weights=[1, 1], threshold=1))
    large = area_um2(CellSpec(cell="LCTL", weights=[8, 8], threshold=8))
    assert 0 < small < large


def test_area_overhead_factors_are_multiplicative():
    s = spec()
    base = area_um2(s, diffusion_overhead=1.0, routing_factor=1.0)
    scaled = area_um2(s, diffusion_overhead=2.0, routing_factor=3.0)
    assert scaled == pytest.approx(6 * base)


def test_area_scales_with_technology():
    s45 = CellSpec(cell="LCTL", weights=[2, 1], threshold=2, tech=PTM45_HP)
    s130 = CellSpec(cell="LCTL", weights=[2, 1], threshold=2, tech=PTM130)
    assert area_um2(s130) > area_um2(s45)


# -- stimulus ---------------------------------------------------------------

def test_worst_case_stimulus_sits_on_the_boundary():
    """
    The static inputs plus the toggled one must land exactly on T, so the
    measurement captures the slowest resolution rather than an easy one.
    """
    s = CellSpec(cell="LCTL", weights=[3, 2, 1, 1], threshold=4)
    stim = worst_case_stimulus(s, rise=True)
    weights = dict(zip(s.input_names, s.all_weights))
    total = weights[stim.toggled] + sum(
        weights[n] for n, v in stim.static.items() if v
    )
    assert total == s.threshold


def test_worst_case_stimulus_falls_below_the_boundary():
    s = CellSpec(cell="LCTL", weights=[3, 2, 1, 1], threshold=4)
    stim = worst_case_stimulus(s, rise=False)
    weights = dict(zip(s.input_names, s.all_weights))
    total = sum(weights[n] for n, v in stim.static.items() if v)
    assert total <= s.threshold - 1


def test_stimulus_covers_every_input():
    s = spec()
    stim = worst_case_stimulus(s)
    assert {stim.toggled} | set(stim.static) == set(s.input_names)


# -- deck -------------------------------------------------------------------

@pytest.mark.parametrize("cell", ["LCTL", "CRTL"])
def test_deck_has_required_sections(cell):
    s = spec(cell=cell)
    deck = build_deck(s, worst_case_stimulus(s))
    for marker in (".include", ".subckt", ".tran", ".meas", ".end"):
        assert marker in deck, marker
    assert deck.rstrip().endswith(".end")


def test_deck_measures_all_four_quantities():
    s = spec()
    deck = build_deck(s, worst_case_stimulus(s))
    for m in ("tpd", "trf", "iavg", "ipeak"):
        assert f" {m} " in deck


def test_deck_drives_every_input():
    s = spec()
    deck = build_deck(s, worst_case_stimulus(s))
    for name in s.input_names:
        assert re.search(rf"^V{name} ", deck, re.MULTILINE), name


def test_crtl_deck_has_a_clock_source():
    s = spec(cell="CRTL")
    assert "VE E 0 PULSE" in build_deck(s, worst_case_stimulus(s))


def test_lctl_deck_has_no_clock_source():
    s = spec(cell="LCTL")
    assert "VE E 0" not in build_deck(s, worst_case_stimulus(s))


def test_rise_and_fall_decks_differ():
    s = spec()
    up = build_deck(s, worst_case_stimulus(s, rise=True))
    down = build_deck(s, worst_case_stimulus(s, rise=False))
    assert up != down
    assert "RISE=1" in up and "FALL=1" in down


def test_model_include_path_is_honoured():
    from dataclasses import replace

    tech = replace(PTM45_HP, model_include="/opt/ptm/45nm_HP.pm")
    s = CellSpec(cell="LCTL", weights=[1, 1], threshold=1, tech=tech)
    assert ".include /opt/ptm/45nm_HP.pm" in build_deck(s, worst_case_stimulus(s))


def test_load_capacitance_appears():
    s = CellSpec(cell="LCTL", weights=[1, 1], threshold=1, load_ff=7.5)
    assert "CL OUT 0 7.5f" in build_deck(s, worst_case_stimulus(s))


# -- measurement parsing ----------------------------------------------------

def test_parse_typical_ngspice_output():
    out = """
    tpd                 =  1.234000e-11 targ=  2.0e-09 trig=  1.9e-09
    trf                 =  8.000000e-12
    iavg                = -3.210000e-05 from=  0.0 to=  8.0e-09
    ipeak               =  9.100000e-04
    """
    vals = parse_measurements(out)
    assert vals["tpd"] == pytest.approx(1.234e-11)
    assert vals["iavg"] == pytest.approx(-3.21e-05)


def test_failed_measurement_raises():
    """
    ngspice reports a failed measurement inline and still exits zero, so a
    silently dropped data point would look like a successful run.
    """
    with pytest.raises(MeasurementError, match="could not measure"):
        parse_measurements("tpd = failed\ntrf = 1.0e-12\n")


def test_empty_output_raises():
    with pytest.raises(MeasurementError, match="no .meas results"):
        parse_measurements("ngspice-42 done\n")


def test_parser_is_case_insensitive():
    assert "tpd" in parse_measurements("TPD = 1.0e-11\n")


def test_missing_binary_gives_actionable_error():
    with pytest.raises(SimulatorError, match="ptm.asu.edu"):
        run_deck("* empty\n.end\n", binary="definitely-not-ngspice")


def test_ngspice_available_reports_absence():
    assert ngspice_available("definitely-not-ngspice") is False


# -- characterize -----------------------------------------------------------

def test_characterize_with_injected_runner():
    r = characterize(spec(), runner=fake_runner)
    assert isinstance(r, CellResult)
    assert r.cell == "CRTL"
    assert r.delay_ns == pytest.approx(0.015)
    assert r.power_uw == pytest.approx(32.0)
    assert r.area_um2 > 0
    assert r.devices > 0


def test_characterize_runs_both_directions():
    calls = []

    def counting(deck):
        calls.append(deck)
        return dict(FAKE_MEAS)

    characterize(spec(), runner=counting)
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_characterize_reports_the_slower_direction():
    decks = []

    def asymmetric(deck):
        decks.append(deck)
        slow = "FALL=1" in deck
        return {**FAKE_MEAS, "tpd": 5.0e-11 if slow else 1.0e-11}

    r = characterize(spec(), runner=asymmetric)
    assert r.delay_ns == pytest.approx(0.05)


def test_characterize_records_key_count():
    r = characterize(spec(keys=(2, -2, 2, -2)), runner=fake_runner)
    assert r.n_keys == 4


# -- sweep ------------------------------------------------------------------

def test_sweep_shape():
    res = key_size_sweep(key_sizes=[2, 4], runner=fake_runner)
    assert len(res) == 4
    assert {r.cell for r in res} == {"LCTL", "CRTL"}
    assert {r.n_keys for r in res} == {2, 4}


def test_sweep_area_grows_with_key_count():
    res = key_size_sweep(cells=("CRTL",), key_sizes=[2, 4, 8], runner=fake_runner)
    areas = [r.area_um2 for r in sorted(res, key=lambda x: x.n_keys)]
    assert areas == sorted(areas)
    assert areas[-1] > areas[0]


def test_sweep_compensates_the_threshold():
    """
    The measured cell must be the cell locking would actually produce, so the
    threshold carries the same compensation embed_keys() applies.
    """
    res = key_size_sweep(cells=("LCTL",), key_sizes=[4], runner=fake_runner)
    # Four alternating +/-2 weights sum to zero, so no shift for even counts.
    assert res[0].threshold == 3


def test_sweep_honours_technology():
    res = key_size_sweep(key_sizes=[2], tech=PTM130, runner=fake_runner)
    assert all(r.tech == "PTM130" for r in res)


def test_sweep_honours_comparator_scale():
    plain = key_size_sweep(key_sizes=[2], comparator_scale=1.0, runner=fake_runner)
    wide = key_size_sweep(key_sizes=[2], comparator_scale=10.0, runner=fake_runner)
    assert wide[0].area_um2 > plain[0].area_um2


# -- Table I formatting -----------------------------------------------------

def test_table_i_pairs_cells():
    rows = table_i_rows(key_size_sweep(key_sizes=[2, 4], runner=fake_runner))
    assert len(rows) == 2
    for r in rows:
        assert {"lctl_area_um2", "crtl_area_um2", "area_saving"} <= set(r)


def test_table_i_drops_unpaired_rows():
    """A partial sweep must not look like a complete comparison."""
    res = key_size_sweep(cells=("LCTL",), key_sizes=[2, 4], runner=fake_runner)
    assert table_i_rows(res) == []


def test_table_i_savings_are_fractions():
    rows = table_i_rows(key_size_sweep(key_sizes=[2], runner=fake_runner))
    for key in ("area_saving", "power_saving", "delay_saving"):
        assert -1.0 <= rows[0][key] <= 1.0


def test_write_csv_roundtrip(tmp_path):
    import csv

    res = key_size_sweep(key_sizes=[2], runner=fake_runner)
    path = tmp_path / "chars.csv"
    write_csv(res, str(path))
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(res)
    assert rows[0]["cell"] in ("LCTL", "CRTL")


def test_write_csv_rejects_empty():
    with pytest.raises(ValueError, match="no results"):
        write_csv([], "/tmp/nope.csv")


# -- technology registry ----------------------------------------------------

def test_registry_contains_expected_nodes():
    assert {"PTM45", "PTM45LP", "PTM130"} <= set(TECHNOLOGIES)


def test_technology_width_helper():
    assert PTM45_HP.w(3) == pytest.approx(3 * PTM45_HP.w_unit)


# -- CLI --------------------------------------------------------------------

def test_cli_characterize_reports_missing_ngspice(capsys):
    from tlglock.cli import main

    rc = main(["characterize", "--binary", "definitely-not-ngspice"])
    assert rc == 2
    assert "ptm.asu.edu" in capsys.readouterr().err
