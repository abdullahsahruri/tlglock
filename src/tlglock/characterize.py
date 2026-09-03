"""
Run ngspice and turn its output into the columns of Table I.

Three pieces: a runner that shells out to ngspice in batch mode, a parser for
the `.meas` results it prints, and a sweep that walks key size the way Fig. 5
does. Everything except the runner works without ngspice installed, which is
what makes the whole path testable.

The measurement parser is deliberately strict. ngspice reports a failed
measurement as `failed` rather than by exiting non-zero, so a deck whose
output never crosses the threshold yields a run that looks successful and
silently drops a data point. `MeasurementError` is raised instead.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .spice import (
    CellSpec,
    CellType,
    Technology,
    PTM45_HP,
    area_um2,
    build_deck,
    device_count,
    worst_case_stimulus,
)

DEFAULT_BINARY = "ngspice"

# ngspice prints e.g. "tpd                 =  1.234000e-11 targ= ..."
_MEAS = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?|failed)",
    re.IGNORECASE | re.MULTILINE,
)


class MeasurementError(RuntimeError):
    """A .meas statement did not produce a usable value."""


class SimulatorError(RuntimeError):
    """ngspice failed to run."""


@dataclass
class CellResult:
    cell: str
    fanin: int
    n_keys: int
    threshold: int
    tech: str
    area_um2: float
    power_uw: float
    delay_ns: float
    transition_ns: float = 0.0
    peak_current_ua: float = 0.0
    devices: int = 0
    comparator_scale: float = 1.0

    def as_row(self) -> dict:
        return asdict(self)


def ngspice_available(binary: str = DEFAULT_BINARY) -> bool:
    return shutil.which(binary) is not None


def parse_measurements(output: str) -> dict[str, float]:
    """
    Extract .meas results from ngspice output.

    Raises MeasurementError on any measurement ngspice reported as failed,
    rather than dropping it -- a silently missing delay would show up much
    later as a suspiciously flat curve.
    """
    values: dict[str, float] = {}
    failed: list[str] = []
    for name, raw in _MEAS.findall(output):
        key = name.lower()
        if raw.lower() == "failed":
            failed.append(key)
            continue
        values[key] = float(raw)
    if failed:
        raise MeasurementError(
            f"ngspice could not measure {sorted(set(failed))} -- the output "
            "probably never crossed the trigger level; check the stimulus "
            "and that the gate actually switches for this input pattern"
        )
    if not values:
        raise MeasurementError("no .meas results found in ngspice output")
    return values


def run_deck(
    deck: str,
    binary: str = DEFAULT_BINARY,
    timeout: float = 300.0,
    workdir: str | None = None,
) -> dict[str, float]:
    """Write a deck to a temp file, run ngspice in batch mode, parse results."""
    if not ngspice_available(binary):
        raise SimulatorError(
            f"'{binary}' not found on PATH. Install ngspice (apt install "
            "ngspice / brew install ngspice) and fetch PTM model cards from "
            "https://ptm.asu.edu/"
        )

    tmp = tempfile.mkdtemp() if workdir is None else workdir
    path = Path(tmp) / "deck.sp"
    path.write_text(deck)

    try:
        proc = subprocess.run(
            [binary, "-b", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir or None,
        )
    except subprocess.TimeoutExpired:
        raise SimulatorError(f"ngspice timed out after {timeout}s") from None

    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0 and "error" in combined.lower():
        raise SimulatorError(
            f"ngspice exited {proc.returncode}:\n{combined[-2000:]}"
        )
    return parse_measurements(combined)


Runner = Callable[[str], dict[str, float]]


def characterize(
    spec: CellSpec,
    runner: Runner | None = None,
    binary: str = DEFAULT_BINARY,
    workdir: str | None = None,
) -> CellResult:
    """
    Measure one cell: area, power, delay.

    Both switching directions are simulated and the slower is reported, since
    a threshold gate is generally asymmetric -- the pulldown network is
    width-weighted while the pullup is not.

    `runner` is injectable so the sweep can be tested, and so a cluster or
    an alternative simulator can be dropped in without touching this function.
    """
    run = runner or (lambda deck: run_deck(deck, binary=binary, workdir=workdir))

    delay = 0.0
    transition = 0.0
    iavg = 0.0
    ipeak = 0.0

    for rise in (True, False):
        stim = worst_case_stimulus(spec, rise=rise)
        meas = run(build_deck(spec, stim))

        # A measurement ngspice never emits is not the same as one it reports
        # as failed, and only the latter reaches parse_measurements. Without
        # this check a deck whose output never switches returns tpd=0.0 and
        # the cell looks infinitely fast -- the exact "suspiciously flat
        # curve" this module was written to rule out.
        missing = [k for k in ("tpd", "trf") if k not in meas]
        if missing:
            raise MeasurementError(
                f"{spec.cell} fanin={spec.fanin} T={spec.threshold} "
                f"(rise={rise}): ngspice produced no value for {missing}. The "
                "output almost certainly never crossed the trigger level, so "
                "the gate did not switch for this pattern."
            )
        delay = max(delay, abs(meas.get("tpd", 0.0)))
        transition = max(transition, abs(meas.get("trf", 0.0)))
        iavg = max(iavg, abs(meas.get("iavg", 0.0)))
        # ngspice cannot take ABS() inside .meas MAX, so the deck reports the
        # supply current's two extremes separately and the peak magnitude is
        # recombined here.
        peak = max(
            abs(meas.get("imax", 0.0)),
            abs(meas.get("imin", 0.0)),
            abs(meas.get("ipeak", 0.0)),
        )
        ipeak = max(ipeak, peak)

    return CellResult(
        cell=spec.cell,
        fanin=spec.fanin,
        n_keys=len(spec.key_weights),
        threshold=spec.threshold,
        tech=spec.tech.name,
        area_um2=round(area_um2(spec), 4),
        power_uw=round(iavg * spec.tech.vdd * 1e6, 4),
        delay_ns=round(delay * 1e9, 6),
        transition_ns=round(transition * 1e9, 6),
        peak_current_ua=round(ipeak * 1e6, 4),
        devices=device_count(spec),
        comparator_scale=spec.comparator_scale,
    )


def key_size_sweep(
    cells: Sequence[CellType] = ("LCTL", "CRTL"),
    key_sizes: Iterable[int] = range(2, 17, 2),
    data_weights: Sequence[int] = (3, 2, 1, 1),
    key_weight: int = 2,
    tech: Technology = PTM45_HP,
    comparator_scale: float = 1.0,
    runner: Runner | None = None,
    binary: str = DEFAULT_BINARY,
) -> list[CellResult]:
    """
    Sweep key size for both cells -- the experiment behind Fig. 5.

    Key weights alternate sign, matching the `balanced` locking mode, and the
    threshold is compensated the same way `embed_keys()` compensates it, so
    the cell being measured is the cell the locking pass would actually
    produce rather than an idealised one.
    """
    results: list[CellResult] = []
    base_threshold = max(1, sum(w for w in data_weights if w > 0) // 2)

    for cell in cells:
        for k in key_sizes:
            key_weights = [
                key_weight if j % 2 == 0 else -key_weight for j in range(k)
            ]
            # Correct key taken as all-ones, so the shift is the weight sum.
            shift = sum(key_weights)
            spec = CellSpec(
                cell=cell,
                weights=list(data_weights),
                key_weights=key_weights,
                threshold=base_threshold + shift,
                tech=tech,
                comparator_scale=comparator_scale,
            )
            results.append(characterize(spec, runner=runner, binary=binary))
    return results


def write_csv(results: Sequence[CellResult], path: str) -> None:
    if not results:
        raise ValueError("no results to write")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].as_row()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())


def table_i_rows(results: Sequence[CellResult]) -> list[dict]:
    """
    Fold paired LCTL/CRTL results into Table I's side-by-side layout.

    Pairs are matched on fan-in and key count. Rows without a counterpart are
    dropped rather than half-filled, so a partial sweep cannot masquerade as a
    complete comparison.
    """
    by_key: dict[tuple[int, int], dict[str, CellResult]] = {}
    for r in results:
        by_key.setdefault((r.fanin, r.n_keys), {})[r.cell] = r

    rows = []
    for (fanin, n_keys), pair in sorted(by_key.items()):
        if "LCTL" not in pair or "CRTL" not in pair:
            continue
        l, c = pair["LCTL"], pair["CRTL"]
        rows.append(
            {
                "fanin": fanin,
                "n_keys": n_keys,
                "lctl_area_um2": l.area_um2,
                "lctl_power_uw": l.power_uw,
                "lctl_delay_ns": l.delay_ns,
                "crtl_area_um2": c.area_um2,
                "crtl_power_uw": c.power_uw,
                "crtl_delay_ns": c.delay_ns,
                "area_saving": round(1 - c.area_um2 / l.area_um2, 4) if l.area_um2 else 0.0,
                "power_saving": round(1 - c.power_uw / l.power_uw, 4) if l.power_uw else 0.0,
                "delay_saving": round(1 - c.delay_ns / l.delay_ns, 4) if l.delay_ns else 0.0,
            }
        )
    return rows
