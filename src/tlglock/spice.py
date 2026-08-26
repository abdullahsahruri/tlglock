"""
SPICE netlist generation for the LCTL and CRTL threshold gates.

Targets ngspice with ASU PTM BSIM4 model cards -- plain text files, no PDK
install, nothing tied to a commercial tool. PTM 45nm keeps the node the same
as the GPDK045 numbers in Table I, so results stay roughly comparable.

    https://ptm.asu.edu/          (45nm HP / LSTP model cards)

                    === TOPOLOGY IS A RECONSTRUCTION ===

The original schematics did not survive, so the device-level netlists below
are reconstructed from the device labels in Fig. 2 of the paper together with
the cited sources -- LCTL from reference [15] (Ozdemir et al., capacitive /
latch-type threshold gate) and CRTL from reference [16] (charge-recycling
threshold logic).

What Fig. 2 gives us:

  LCTL   differential current-mode. An input branch carrying Iin with the
         data inputs X1..Xm on devices sized W/2L, a reference branch
         carrying Iref with the threshold inputs Y1..Ym likewise, tail
         devices M5 and M10 at W/L, and a cross-coupled latch (M1, M2, M6,
         M7) with load resistors R producing Vout and Voutn.

  CRTL   dynamic, single clock E with reset and evaluate phases. Cross-
         coupled inverter pair (M1..M4) precharged through E, input network
         X1..Xn against threshold network T1..Tn, and a charge-recycling
         switch M7 that shares charge between the two output nodes during
         reset instead of dumping it to ground -- the mechanism the paper's
         energy claims rest on.

Weights are realised as device width ratios: a weight of w gets a device of
width w * unit_width. This is the standard conductance-ratio encoding and is
what makes "how many weights?" in Fig. 3 a physical design question.

**Verify these against Fig. 2 before publishing any number produced from
them.** The harness around them -- sweeping, measuring, parsing, reporting --
is independent of the topology and is what this module is really for; swapping
in a corrected netlist means editing two functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

CellType = Literal["LCTL", "CRTL"]


@dataclass(frozen=True)
class Technology:
    """
    Process parameters. Defaults describe PTM 45nm HP.

    `model_include` is the path to the PTM card on the machine that will run
    ngspice; it is written into the deck verbatim.
    """

    name: str = "PTM45"
    model_include: str = "models/45nm_HP.pm"
    nmos_model: str = "NMOS_VTG"
    pmos_model: str = "PMOS_VTG"
    vdd: float = 1.0
    l_min: float = 45e-9
    w_unit: float = 90e-9      # unit device width, 2 * L_min
    temp: float = 27.0

    def w(self, multiple: float) -> float:
        return multiple * self.w_unit


PTM45_HP = Technology()
PTM45_LP = Technology(
    name="PTM45LP", model_include="models/45nm_LP.pm", vdd=1.1
)
PTM130 = Technology(
    name="PTM130",
    model_include="models/130nm_bulk.pm",
    vdd=1.2,
    l_min=130e-9,
    w_unit=260e-9,
)

TECHNOLOGIES = {t.name: t for t in (PTM45_HP, PTM45_LP, PTM130)}


@dataclass
class CellSpec:
    """
    One threshold gate to characterise.

    `weights` are the data-input weights and `key_weights` the key-input
    weights, kept separate so the sweep can vary key count independently --
    which is exactly what Fig. 5 of the paper plots.
    """

    cell: CellType
    weights: Sequence[int]
    key_weights: Sequence[int] = field(default_factory=tuple)
    threshold: int = 1
    tech: Technology = PTM45_HP
    comparator_scale: float = 1.0   # the starred rows use 10x
    load_ff: float = 2.0            # output load in femtofarads

    @property
    def fanin(self) -> int:
        return len(self.weights) + len(self.key_weights)

    @property
    def all_weights(self) -> list[int]:
        return list(self.weights) + list(self.key_weights)

    @property
    def input_names(self) -> list[str]:
        return [f"X{i+1}" for i in range(len(self.weights))] + [
            f"K{i+1}" for i in range(len(self.key_weights))
        ]

    @property
    def total_weight(self) -> int:
        return sum(abs(w) for w in self.all_weights)

    def __post_init__(self) -> None:
        if self.cell not in ("LCTL", "CRTL"):
            raise ValueError(f"unknown cell type '{self.cell}'")
        if not self.all_weights:
            raise ValueError("cell has no inputs")
        if self.comparator_scale <= 0:
            raise ValueError("comparator_scale must be positive")


def _fmt(value: float) -> str:
    """SPICE-friendly number."""
    return f"{value:.6g}"


def _device(name, d, g, s, b, model, w, l) -> str:
    return f"M{name} {d} {g} {s} {b} {model} W={_fmt(w)} L={_fmt(l)}"


# -- LCTL -------------------------------------------------------------------


def lctl_subckt(spec: CellSpec) -> str:
    """
    Latch-type conductance threshold logic.

    Data inputs steer current into the `sum` node through width-weighted
    NMOS devices; the threshold is a matching reference branch whose total
    width encodes T. The cross-coupled latch resolves whichever branch draws
    more current, so the gate fires when the weighted input conductance
    exceeds the reference -- the analog form of `sum w_i x_i >= T`.

    Negative weights are realised on the reference side, since a negative
    contribution to the sum is equivalent to a positive contribution to the
    threshold it must overcome.
    """
    t = spec.tech
    lines = [f".subckt LCTL_{spec.fanin} {' '.join(spec.input_names)} OUT VDD VSS"]
    lines.append("* input branch: width-weighted pulldowns into the sum node")

    idx = 0
    for name, weight in zip(spec.input_names, spec.all_weights):
        idx += 1
        node = "sum" if weight > 0 else "ref"
        lines.append(
            _device(
                f"in{idx}", node, name, "tail", "VSS",
                t.nmos_model, t.w(abs(weight)) / 2, t.l_min,
            )
        )

    lines.append("* reference branch: total width encodes the threshold T")
    ref_w = max(abs(spec.threshold), 1)
    lines.append(
        _device("ref", "ref", "VDD", "tailr", "VSS", t.nmos_model, t.w(ref_w) / 2, t.l_min)
    )

    lines.append("* tail current sources M5 / M10")
    tail_w = t.w(max(1.0, spec.total_weight / 2)) * spec.comparator_scale
    lines.append(_device("5", "tail", "VDD", "VSS", "VSS", t.nmos_model, tail_w, t.l_min))
    lines.append(_device("10", "tailr", "VDD", "VSS", "VSS", t.nmos_model, tail_w, t.l_min))

    lines.append("* cross-coupled latch M1/M2/M6/M7 with load devices M3/M8")
    lat_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("1", "outn", "out_i", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("2", "out_i", "outn", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("6", "outn", "out_i", "sum", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("7", "out_i", "outn", "ref", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("3", "sum", "VDD", "VSS", "VSS", t.nmos_model, lat_w / 2, t.l_min))
    lines.append(_device("8", "ref", "VDD", "VSS", "VSS", t.nmos_model, lat_w / 2, t.l_min))

    lines.append("* output buffer")
    lines.append(_device("o1", "OUT", "out_i", "VDD", "VDD", t.pmos_model, t.w(4), t.l_min))
    lines.append(_device("o2", "OUT", "out_i", "VSS", "VSS", t.nmos_model, t.w(2), t.l_min))

    lines.append(".ends")
    return "\n".join(lines)


# -- CRTL -------------------------------------------------------------------


def crtl_subckt(spec: CellSpec) -> str:
    """
    Charge-recycling threshold logic.

    Dynamic, clocked by E. During reset (E low) the output nodes are
    precharged and M7 shorts them together, recycling the charge from the
    previous evaluation rather than discharging it -- the energy mechanism the
    paper's power numbers depend on. During evaluate (E high) the input
    network competes against the threshold network and the cross-coupled pair
    resolves.
    """
    t = spec.tech
    lines = [f".subckt CRTL_{spec.fanin} {' '.join(spec.input_names)} E OUT VDD VSS"]

    lines.append("* precharge devices M3 / M4, gated by E")
    pre_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("3", "out_i", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))
    lines.append(_device("4", "outn", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))

    lines.append("* charge recycling switch M7: shorts the outputs during reset")
    lines.append(
        _device("7", "out_i", "En", "outn", "VDD", t.pmos_model, t.w(3.0), t.l_min)
    )

    lines.append("* cross-coupled resolving pair M1 / M2")
    lat_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("1", "out_i", "outn", "nsum", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("2", "outn", "out_i", "nref", "VSS", t.nmos_model, lat_w, t.l_min))

    lines.append("* input network: width-weighted evaluation devices")
    idx = 0
    for name, weight in zip(spec.input_names, spec.all_weights):
        idx += 1
        node = "nsum" if weight > 0 else "nref"
        lines.append(
            _device(
                f"x{idx}", node, name, "foot", "VSS",
                t.nmos_model, t.w(abs(weight)), t.l_min,
            )
        )

    lines.append("* threshold network T1..Tn")
    lines.append(
        _device(
            "t", "nref", "VDD", "foot", "VSS",
            t.nmos_model, t.w(max(abs(spec.threshold), 1)), t.l_min,
        )
    )

    lines.append("* evaluation footer M5 / M6")
    foot_w = t.w(max(2.0, spec.total_weight / 2)) * spec.comparator_scale
    lines.append(_device("5", "foot", "E", "VSS", "VSS", t.nmos_model, foot_w, t.l_min))

    lines.append("* clock inverter for En")
    lines.append(_device("i1", "En", "E", "VDD", "VDD", t.pmos_model, t.w(2), t.l_min))
    lines.append(_device("i2", "En", "E", "VSS", "VSS", t.nmos_model, t.w(1), t.l_min))

    lines.append("* output buffer")
    lines.append(_device("o1", "OUT", "out_i", "VDD", "VDD", t.pmos_model, t.w(4), t.l_min))
    lines.append(_device("o2", "OUT", "out_i", "VSS", "VSS", t.nmos_model, t.w(2), t.l_min))

    lines.append(".ends")
    return "\n".join(lines)


def subckt(spec: CellSpec) -> str:
    return lctl_subckt(spec) if spec.cell == "LCTL" else crtl_subckt(spec)


def subckt_name(spec: CellSpec) -> str:
    return f"{spec.cell}_{spec.fanin}"


# -- transistor accounting and area ----------------------------------------


def transistors(spec: CellSpec) -> list[tuple[str, float, float]]:
    """
    (name, W, L) for every device, parsed back out of the netlist.

    Reading the widths back from the emitted text rather than tracking them
    separately means the area model and the simulated netlist can never
    disagree: there is one source of truth.
    """
    out = []
    for line in subckt(spec).splitlines():
        if not line.startswith("M"):
            continue
        # M<name> <d> <g> <s> <b> <model> W=<w> L=<l>
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"malformed device line: {line!r}")
        name = parts[0]
        w = float(parts[6].split("=", 1)[1])
        l = float(parts[7].split("=", 1)[1])
        out.append((name, w, l))
    return out


def area_um2(
    spec: CellSpec,
    diffusion_overhead: float = 1.8,
    routing_factor: float = 1.35,
) -> float:
    """
    Analytical area estimate, in square micrometres.

    Active area is the sum of W * L over all devices. `diffusion_overhead`
    accounts for source/drain diffusion and contacts, `routing_factor` for
    intra-cell wiring and well spacing. Both are explicit because they are
    assumptions, not measurements.

    This is the standard estimate in the TLG literature and is defensible if
    stated as such. If a reviewer wants real numbers, lay one cell out in
    Magic or KLayout, fit the two factors against it, and rerun -- the whole
    sweep then rescales without further layout work.
    """
    active = sum(w * l for _, w, l in transistors(spec))
    return active * diffusion_overhead * routing_factor * 1e12


def device_count(spec: CellSpec) -> int:
    return len(transistors(spec))


# -- stimulus and deck ------------------------------------------------------


@dataclass
class Stimulus:
    """One transient measurement: hold inputs, toggle one, watch the output."""

    toggled: str
    static: dict[str, int]
    rise: bool = True
    period_ns: float = 4.0
    edge_ps: float = 20.0


def worst_case_stimulus(spec: CellSpec, rise: bool = True) -> Stimulus:
    """
    Drive the gate across its decision boundary by the smallest margin.

    Propagation delay in a threshold gate depends on how far the weighted sum
    sits from T: a sum that clears the threshold by one unit resolves far more
    slowly than one that clears it by ten. Measuring at the boundary is the
    honest worst case, and it is what makes the delay column comparable
    between cells with different weight spreads.
    """
    weights = spec.all_weights
    names = spec.input_names
    order = sorted(range(len(weights)), key=lambda i: -abs(weights[i]))

    toggled_idx = order[0]
    target = spec.threshold if rise else spec.threshold - 1
    static: dict[str, int] = {}
    total = weights[toggled_idx] if rise else 0

    for i in order[1:]:
        if total + weights[i] <= target:
            static[names[i]] = 1
            total += weights[i]
        else:
            static[names[i]] = 0

    return Stimulus(toggled=names[toggled_idx], static=static, rise=rise)


def build_deck(
    spec: CellSpec,
    stim: Stimulus,
    title: str | None = None,
) -> str:
    """
    A complete ngspice deck: models, subcircuit, stimulus, .tran and .meas.

    Measurements emitted:
      tpd     input 50% to output 50% propagation delay
      trf     output transition time, 20% to 80%
      iavg    average supply current over the window
      ipeak   peak supply current
    """
    t = spec.tech
    vdd = t.vdd
    half = vdd / 2
    per = stim.period_ns
    edge = stim.edge_ps / 1000.0

    lines = [
        f"* {title or f'{spec.cell} fanin={spec.fanin} T={spec.threshold}'}",
        f".include {t.model_include}",
        "",
        subckt(spec),
        "",
        f".temp {_fmt(t.temp)}",
        f"VVDD VDD 0 DC {_fmt(vdd)}",
        "VVSS VSS 0 DC 0",
    ]

    for name, value in stim.static.items():
        lines.append(f"V{name} {name} 0 DC {_fmt(vdd if value else 0.0)}")

    t0, t1 = per / 2, per / 2 + edge
    if stim.rise:
        lines.append(
            f"V{stim.toggled} {stim.toggled} 0 PWL(0 0 "
            f"{_fmt(t0)}n 0 {_fmt(t1)}n {_fmt(vdd)} {_fmt(per * 2)}n {_fmt(vdd)})"
        )
    else:
        lines.append(
            f"V{stim.toggled} {stim.toggled} 0 PWL(0 {_fmt(vdd)} "
            f"{_fmt(t0)}n {_fmt(vdd)} {_fmt(t1)}n 0 {_fmt(per * 2)}n 0)"
        )

    if spec.cell == "CRTL":
        lines.append(
            f"VE E 0 PULSE(0 {_fmt(vdd)} {_fmt(per * 0.75)}n "
            f"{_fmt(edge)}n {_fmt(edge)}n {_fmt(per * 0.6)}n {_fmt(per * 1.5)}n)"
        )

    lines.append(f"CL OUT 0 {_fmt(spec.load_ff)}f")

    ports = " ".join(spec.input_names)
    clock = " E" if spec.cell == "CRTL" else ""
    lines.append(f"XDUT {ports}{clock} OUT VDD VSS {subckt_name(spec)}")

    lines.append("")
    lines.append(f".tran {_fmt(edge / 20)}n {_fmt(per * 2)}n")
    lines.append("")

    direction = "RISE" if stim.rise else "FALL"
    lines.append(
        f".meas tran tpd TRIG V({stim.toggled}) VAL={_fmt(half)} {direction}=1 "
        f"TARG V(OUT) VAL={_fmt(half)} CROSS=1"
    )
    lines.append(
        f".meas tran trf TRIG V(OUT) VAL={_fmt(0.2 * vdd)} CROSS=1 "
        f"TARG V(OUT) VAL={_fmt(0.8 * vdd)} CROSS=1"
    )
    lines.append(f".meas tran iavg AVG I(VVDD) FROM=0 TO={_fmt(per * 2)}n")
    lines.append(f".meas tran ipeak MAX ABS(I(VVDD)) FROM=0 TO={_fmt(per * 2)}n")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines) + "\n"
