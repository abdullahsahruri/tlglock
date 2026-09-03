"""
SPICE netlists for three threshold-logic cell families.

Targets ngspice with ASU PTM BSIM4 model cards -- plain text, no PDK install,
nothing tied to a commercial tool. Staying at 45nm keeps the numbers roughly
comparable to the GPDK045 figures in the literature.

Each cell is transcribed from a published schematic rather than reconstructed:

  LCTL   Latch-type low power threshold logic. Fig. 8 of Beiu, Quintana &
         Avedillo, "Review of Differential Threshold Gate Implementations".
         Current-controlled latch over two parallel nMOS banks; weights are
         conductance ratios.

  CRTL   Charge recycling threshold logic. Fig. 2 of Celinski, Lopez,
         Al-Sarawi & Abbott, Microelectronics Journal 33 (2002) 1071-1077,
         and Fig. 4 of the same review. Capacitively coupled floating gate
         into a sense-amplifier latch; weights are *capacitor* ratios.

  DCSTL  Differential current-switch threshold logic. Section 3 of the same
         review, which reports it as having a better power-delay product than
         both LCTL and CIALTL. Conductance weighting with cascoded internal
         nodes to limit swing.

                === TWO WEIGHTING MECHANISMS, NOT ONE ===

The families split on how a weight is realised, and it is not a detail:

  conductance (LCTL, DCSTL)  weight w is a device of width w * w_unit.
                             A *negative* weight is the same device on the
                             opposite bank -- signed weights are free.

  capacitive  (CRTL)         weight w is a capacitor w * c_unit onto the
                             floating gate. Capacitors cannot be negative,
                             so a negative weight must drive the complement
                             of its input, costing an inverter.

That asymmetry has a consequence for logic locking specifically. A locking
scheme whose selling point is that a key input looks exactly like a data input
cannot afford to emit an inverter for half its key bits, which is what
alternating-sign key weights do in a capacitive cell. The conductance families
host signed keys without leaving a trace; CRTL does not. The inverters are
emitted and counted here rather than hidden, so the cost lands in the area
numbers where it can be seen.

All three cells are clocked -- see CLOCK_PORT -- so propagation delay is
measured from the evaluate edge, not from a data input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

CellType = Literal["LCTL", "CRTL", "DCSTL"]

# Every cell here is clocked, but they do not agree on what the clock is
# called: LCTL sequences precharge/evaluate from a reset signal PHIR, while
# CRTL and DCSTL use an enable E. The deck builder needs the right port name
# or the subcircuit call silently binds the clock to a data input.
CLOCK_PORT = {"LCTL": "PHIR", "CRTL": "E", "DCSTL": "E"}


@dataclass(frozen=True)
class Technology:
    """
    Process parameters. Defaults describe PTM 45nm HP.

    `model_include` is the path to the PTM card on the machine that will run
    ngspice; it is written into the deck verbatim.
    """

    name: str = "PTM45"
    model_include: str = "models/45nm_HP.pm"
    # The PTM cards declare `.model nmos nmos` / `.model pmos pmos`. Older
    # ASU releases used NMOS_VTG/PMOS_VTG and some forks still do, so this is
    # configurable -- but it must match the card actually being included or
    # ngspice fails on an undefined model.
    nmos_model: str = "nmos"
    pmos_model: str = "pmos"
    vdd: float = 1.0
    l_min: float = 45e-9
    w_unit: float = 90e-9      # unit device width, 2 * L_min
    temp: float = 27.0
    c_unit: float = 1e-15      # unit input capacitor for capacitive cells

    def w(self, multiple: float) -> float:
        return multiple * self.w_unit

    def c(self, multiple: float) -> float:
        return multiple * self.c_unit


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
        if self.cell not in CLOCK_PORT:
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


def _signed_banks(spec: CellSpec) -> tuple[list[tuple[str, int]], int]:
    """
    Split weights into a data bank and a threshold bank.

    A conductance TLG compares two parallel nMOS banks. A negative weight is
    not a device that cannot exist -- it is a device on the *other* bank,
    since subtracting w from the sum is the same as adding w to the threshold
    the sum must clear. So signed weights cost nothing here, which is exactly
    what a capacitive cell cannot do (see crtl_subckt).

    Returns (data-bank entries, threshold multiple), where the threshold
    multiple already absorbs the negative weights.
    """
    data: list[tuple[str, int]] = []
    thresh = spec.threshold
    for name, weight in zip(spec.input_names, spec.all_weights):
        if weight >= 0:
            data.append((name, weight))
        else:
            # x contributes -|w|*x; equivalently the reference bank gains
            # |w| when x is high. Realised by placing the device on the
            # threshold side, gated by the same signal.
            data.append((name, weight))
    return data, thresh


def _unit_bank(prefix, node, gate, count, tech, scale=1.0):
    """
    Emit `count` unit-width devices in parallel from `node` to `foot`.

    Both banks have to be built from *identical unit* devices, which is how
    the published cells describe them ("parallel-connected sets of unit nMOS
    transistors"). Collapsing a weight into one device of width w*w_unit looks
    equivalent and is not: a single wide device and w narrow ones differ in
    source/drain resistance and short-channel behaviour, so a data bank made
    of several devices does not match a reference made of one. That mismatch
    is invisible at large margins and decides the outcome at sum == T, which
    is exactly where a threshold gate has to be right.
    """
    out = []
    whole = int(count)
    for k in range(whole):
        out.append(_device(f"{prefix}{k}", node, gate, "foot", "VSS",
                           tech.nmos_model, tech.w(1.0) * scale, tech.l_min))
    frac = count - whole
    if frac > 0:
        out.append(_device(f"{prefix}h", node, gate, "foot", "VSS",
                           tech.nmos_model, tech.w(frac) * scale, tech.l_min))
    return out


def _balance_banks(lines, da_units, db_units, tech):
    """
    Pad the lighter bank with switched-off dummy devices until both banks
    carry the same device count.

    The LCTL description specifies "two input arrays having an *equal number*
    of parallel transistors", and that is a matching requirement rather than a
    description. The data bank holds one device per input, most of them off on
    any given pattern; the threshold bank holds only the reference. Off
    devices still contribute junction capacitance, so without padding the data
    node is far more heavily loaded and discharges more slowly no matter which
    bank has more conductance. At wide margins the conductance still wins; at
    sum == T it does not, and the gate resolves on parasitics.

    Dummies are gated to VSS so they never conduct and never alter the
    threshold -- they only equalise the capacitance.
    """
    out = []
    deficit = int(round(da_units - db_units))
    node = "db" if deficit > 0 else "da"
    for k in range(abs(deficit)):
        out.append(_device(f"pad{node}{k}", node, "VSS", "foot", "VSS",
                           tech.nmos_model, tech.w(1.0), tech.l_min))
    if out:
        lines.append(f"* {len(out)} dummy device(s) balancing bank capacitance")
        lines.extend(out)


def lctl_subckt(spec: CellSpec) -> str:
    """
    Latch-type low power threshold logic (LCTL).

    Topology follows Fig. 8 of Beiu, Quintana & Avedillo, "Review of
    Differential Threshold Gate Implementations", which in turn reproduces the
    original LCTL of Avedillo et al.:

      * a CMOS current-controlled latch, M2/M5 and M7/M10, producing OUT and
        its complement OUTN;
      * two input arrays of parallel nMOS devices -- the data bank on one
        side, the threshold bank on the other -- whose gates are the gate's
        inputs;
      * M1/M3 and M6/M8 select precharge or evaluate from the reset signal
        PHIR. With PHIR low, M1 and M6 conduct and both outputs sit at VDD;
        with PHIR high, M3 and M8 conduct and the outputs discharge through
        whichever bank sinks more current;
      * M4x and M9x, the two extra W/2L devices, exist specifically to settle
        the case where the weighted sum *equals* the threshold.

    That last pair is not optional here. TLGLock compares with `>=`, and
    threshold compensation puts the correct key exactly on the boundary for
    some assignments, so a cell that resolves ties arbitrarily would break the
    lock rather than merely slow it down.

    Weights are conductance ratios: weight w becomes a device of width
    w * w_unit. Negative weights go to the threshold bank.
    """
    t = spec.tech
    lines = [
        f".subckt LCTL_{spec.fanin} {' '.join(spec.input_names)} PHIR OUT OUTN VDD VSS"
    ]

    lines.append("* current-controlled latch M2/M5 and M7/M10")
    lat_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("2", "OUT", "OUTN", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("7", "OUTN", "OUT", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    # OUT discharges through the *threshold* bank and OUTN through the data
    # bank, so that a data side stronger than the reference leaves OUT high.
    # Wiring these the other way round inverts the gate: it would read 0
    # exactly when the weighted sum clears the threshold.
    lines.append(_device("5", "OUT", "OUTN", "db", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("10", "OUTN", "OUT", "da", "VSS", t.nmos_model, lat_w, t.l_min))

    lines.append("* precharge devices M1 / M6, gated by PHIR")
    pre_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("1", "OUT", "PHIR", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))
    lines.append(_device("6", "OUTN", "PHIR", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))
    # M3 / M8 belong in series with the discharge path, not across the banks.
    # Wiring them from da/db to foot puts a fixed conductance in parallel with
    # the weighted devices on *both* sides, which does not change the sign of
    # the comparison but does dilute it: at the sum == T boundary the half-unit
    # tie-breaker was competing against two extra units per side and the latch
    # resolved on parasitics instead. The footer below already gates
    # evaluation, so the sequencing is unaffected by dropping them.

    lines.append("* data bank (positive weights) and threshold bank (negative)")
    idx = 0
    for name, weight in zip(spec.input_names, spec.all_weights):
        idx += 1
        bank = "da" if weight > 0 else "db"
        lines.extend(_unit_bank(f"4{idx}_", bank, name, abs(weight), t))

    lines.append("* threshold bank reference devices, total width encodes T")
    # The reference constant is T itself. The negative-weight devices are
    # already on this bank and contribute |w| when their input is high, which
    # is exactly the  sum_{w>0} w x >= T + sum_{w<0} |w| x  rearrangement.
    # Subtracting the negatives here as well would count them twice and bias
    # every gate with a negative weight toward 0.
    ref = max(1, spec.threshold)
    lines.extend(_unit_bank("9r_", "db", "VDD", ref, t))

    lines.append("* M4n+1 / M9n+1: resolve the sum == threshold tie")
    tie_w = t.w(1.0) / 2
    lines.append(_device("4t", "da", "VDD", "foot", "VSS", t.nmos_model, tie_w, t.l_min))
    lines.append(_device("9t", "db", "VSS", "foot", "VSS", t.nmos_model, tie_w, t.l_min))

    da_units = sum(abs(w) for w in spec.all_weights if w > 0) + 0.5
    db_units = sum(abs(w) for w in spec.all_weights if w < 0) + ref
    _balance_banks(lines, da_units, db_units, t)

    lines.append("* evaluation footer")
    foot_w = t.w(max(2.0, spec.total_weight / 2)) * spec.comparator_scale
    lines.append(
        _device("f", "foot", "PHIR", "VSS", "VSS", t.nmos_model, foot_w, t.l_min)
    )

    lines.append(".ends")
    return "\n".join(lines)


# -- CRTL -------------------------------------------------------------------


def crtl_threshold_volts(spec: CellSpec) -> tuple[float, int]:
    """
    The analog threshold voltage for a CRTL cell, and the capacitive units.

    The floating gate sits at phi = (sum_i C_i x_i) / C_tot * VDD, so with
    C_i = |w_i| * c_unit the node carries the weighted sum scaled into volts.
    Negative weights are complemented (see crtl_subckt), which turns
    w_i*x_i into w_i + |w_i|*xbar_i, so the constant part shifts the
    threshold: comparing against T becomes comparing against T - sum(negatives).

    The comparison level is placed at (T' - 0.5) units rather than T'. The
    weighted sum only takes integer values, so half a unit below T' is the
    widest possible margin either side, and it makes `>=` unambiguous instead
    of leaving the sum == T case on a knife edge.
    """
    units = sum(abs(w) for w in spec.all_weights) or 1
    negative = sum(w for w in spec.all_weights if w < 0)
    level = spec.threshold - negative          # T' after complementing

    # A bias capacitor to VDD lifts the floating gate's operating range. With
    # no bias, phi spans 0..VDD and the comparison sits near VDD/2 -- which at
    # a 1.0V supply leaves both comparator devices at Vgs ~ 0.5V, barely into
    # conduction, so the tail current is tiny and the latch never regenerates
    # inside the evaluate window. (The original CRTL was published at a 2V
    # supply, where this is not a problem.) Adding B units of bias shifts the
    # range to [B, units+B] / (units+B), at the cost of proportionally less
    # swing per unit weight.
    bias = crtl_bias_units(spec)
    total = units + bias
    volts = (level - 0.5 + bias) / total * spec.tech.vdd
    return max(0.0, min(volts, spec.tech.vdd)), units


def crtl_bias_units(spec: CellSpec) -> float:
    """Bias-capacitor size, in the same units as the input capacitors."""
    return sum(abs(w) for w in spec.all_weights) or 1


def crtl_subckt(spec: CellSpec) -> str:
    """
    Charge recycling threshold logic (CRTL).

    Topology follows Fig. 2 of Celinski, Lopez, Al-Sarawi & Abbott, "Low
    depth, low power carry lookahead adders using threshold logic",
    Microelectronics Journal 33 (2002), and Fig. 4 of the Beiu/Quintana/
    Avedillo review.

      * sense amplifier from cross-coupled M1-M4, producing Y and Yi;
      * M8 and M9 equalize the two output nodes during the equalize phase --
        this is the charge recycling: the charge drawn during evaluation is
        re-used rather than dumped;
      * inputs are *capacitively* coupled through C1..Cn onto the floating
        gate phi of M5, with phi = sum_i C_i x_i / C_tot;
      * M6's gate voltage T sets the threshold, so M5/M6 form the comparator;
      * M7 is the evaluation footer, gated by E.

                  === WEIGHTS ARE CAPACITORS, NOT WIDTHS ===

    That is the substantive difference from the conductance cells, and it has
    a consequence for locking that the PPA literature has no reason to
    mention: a capacitor cannot be negative. A negative weight must be applied
    to the complement of its input, which costs an inverter.

    For TLGLock this is not merely area. The scheme's claim is that a key
    input is indistinguishable from a data input; in `balanced` mode roughly
    half the key weights are negative, so each one leaves an inverter that a
    data input of the same weight would not have. The lock acquires exactly
    the structural footprint the scheme says it does not have. The inverters
    are emitted here and counted, rather than hidden, so the cost shows up in
    the area numbers.
    """
    t = spec.tech
    volts, units = crtl_threshold_volts(spec)
    lines = [
        f".subckt CRTL_{spec.fanin} {' '.join(spec.input_names)} E OUT OUTN VDD VSS"
    ]

    lines.append("* sense amplifier: cross-coupled M1-M4")
    lat_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("1", "Yi", "Y", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("2", "Y", "Yi", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("3", "Yi", "Y", "nA", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("4", "Y", "Yi", "nB", "VSS", t.nmos_model, lat_w, t.l_min))

    lines.append("* precharge to VDD during the equalize phase (E low)")
    pre_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("p1", "Y", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))
    lines.append(_device("p2", "Yi", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))

    lines.append("* M9 / M8 equalize the two sides -- the charge recycling path")
    # Both equalizers must conduct in the *same* phase. M8 is nMOS on En, so
    # it shorts while En is high, i.e. while E is low. A pMOS on En would
    # conduct on the opposite phase and short the outputs during evaluation
    # instead of during reset, which prevents the latch from ever resolving.
    # The pMOS therefore takes E.
    lines.append(_device("9", "Y", "E", "Yi", "VDD", t.pmos_model, t.w(3.0), t.l_min))
    lines.append(_device("8", "nA", "En", "nB", "VSS", t.nmos_model, t.w(3.0), t.l_min))
    # The comparator's drain nodes need a defined reset level too, and it has
    # to be VDD rather than VSS. Holding them at VSS while the outputs are
    # precharged high leaves M3/M4 with a full VDD across them, so current
    # flows VDD -> precharge -> Y -> M3 -> nA -> reset -> VSS for the whole
    # reset phase: the outputs settle partway instead of reaching VDD, and the
    # cell burns static power it should not have. Precharging both sides high
    # turns M3/M4 off during reset; they come on as the comparator pulls
    # nA/nB down at the evaluate edge.
    lines.append(_device("r1", "nA", "E", "VDD", "VDD", t.pmos_model, t.w(2.0), t.l_min))
    lines.append(_device("r2", "nB", "E", "VDD", "VDD", t.pmos_model, t.w(2.0), t.l_min))

    lines.append("* comparator pair: M5 on the floating gate, M6 on the threshold")
    cmp_w = t.w(4.0) * spec.comparator_scale
    lines.append(_device("5", "nA", "phi", "tail", "VSS", t.nmos_model, cmp_w, t.l_min))
    lines.append(_device("6", "nB", "vth", "tail", "VSS", t.nmos_model, cmp_w, t.l_min))

    lines.append("* M7 evaluation footer")
    foot_w = t.w(max(2.0, units / 2.0)) * spec.comparator_scale
    lines.append(_device("7", "tail", "E", "VSS", "VSS", t.nmos_model, foot_w, t.l_min))

    lines.append(f"* input capacitor array, C_i = |w_i| * {_fmt(t.c_unit)}F")
    idx = 0
    inverted: list[str] = []
    for name, weight in zip(spec.input_names, spec.all_weights):
        idx += 1
        if weight < 0:
            # No negative capacitors: drive the complement instead. This
            # inverter is the structural footprint discussed above.
            node = f"{name}_n"
            inverted.append(name)
            lines.append(
                _device(f"n{idx}p", node, name, "VDD", "VDD", t.pmos_model, t.w(2), t.l_min)
            )
            lines.append(
                _device(f"n{idx}n", node, name, "VSS", "VSS", t.nmos_model, t.w(1), t.l_min)
            )
        else:
            node = name
        lines.append(f"C{idx} {node} phi {_fmt(t.c(abs(weight)))}")

    if inverted:
        lines.append(
            f"* {len(inverted)} inverter(s) added for negative weights: "
            + ", ".join(inverted)
        )
    # The floating gate needs a DC path or the operating point is undefined,
    # but a single resistor to ground pins phi at 0V regardless of the inputs
    # and the capacitive divider is lost. Pairing each capacitor with a
    # resistor of R = k / C instead makes the *resistive* divider reproduce
    # the capacitive one exactly -- phi = sum(C_i V_i) / sum(C_i) at DC -- while
    # the values stay large enough (>= 1e12 ohm here) that the RC is many
    # orders of magnitude beyond the transient window, so the capacitors still
    # set the dynamics.
    bias = crtl_bias_units(spec)
    lines.append(f"* bias capacitor to VDD, {_fmt(bias)} units")
    lines.append(f"Cbias VDD phi {_fmt(t.c(bias))}")

    lines.append("* DC-matched bias network: R_i = k / C_i reproduces the divider")
    k = 1e-3
    lines.append(f"Rbias VDD phi {_fmt(k / t.c(bias))}")
    for i, (name, weight) in enumerate(
        zip(spec.input_names, spec.all_weights), start=1
    ):
        node = f"{name}_n" if weight < 0 else name
        lines.append(f"Rb{i} {node} phi {_fmt(k / t.c(abs(weight)))}")

    lines.append("* threshold reference voltage")
    lines.append(f"Vth vth VSS DC {_fmt(volts)}")

    lines.append("* clock inverter for En")
    lines.append(_device("i1", "En", "E", "VDD", "VDD", t.pmos_model, t.w(2), t.l_min))
    lines.append(_device("i2", "En", "E", "VSS", "VSS", t.nmos_model, t.w(1), t.l_min))

    # Non-inverting buffers, two stages each. A single inverter here silently
    # flips the cell: OUT becomes NOT(Y), so the gate reads 0 exactly when the
    # weighted sum clears the threshold, and both outputs sit low during
    # precharge instead of high -- the opposite convention from the
    # conductance cells, which would make the two families incomparable.
    lines.append("* output buffers (two stages, non-inverting)")
    for tag, src, dst in (("a", "Y", "OUT"), ("b", "Yi", "OUTN")):
        mid = f"{dst}_i"
        lines.append(_device(f"o{tag}1", mid, src, "VDD", "VDD", t.pmos_model, t.w(2), t.l_min))
        lines.append(_device(f"o{tag}2", mid, src, "VSS", "VSS", t.nmos_model, t.w(1), t.l_min))
        lines.append(_device(f"o{tag}3", dst, mid, "VDD", "VDD", t.pmos_model, t.w(4), t.l_min))
        lines.append(_device(f"o{tag}4", dst, mid, "VSS", "VSS", t.nmos_model, t.w(2), t.l_min))

    lines.append(".ends")
    return "\n".join(lines)


def dcstl_subckt(spec: CellSpec) -> str:
    """
    Differential current-switch threshold logic (DCSTL).

    Section 3 of the Beiu/Quintana/Avedillo review reports DCSTL as having a
    better power-delay product than both other latch-based families it
    compares against, LCTL and CIALTL, which is why it is here.

    Structure: a clocked latched comparator over two parallel-connected banks
    of unit nMOS devices -- the data mapping bank and the threshold mapping
    bank -- placed at the *bottom* of the discharge path, as in SCSDL. Keeping
    the banks below the latch shortens the feedback path relative to LCTL,
    which is where the speed comes from. Internal node swing is restricted by
    the cascode devices M11/M12, which is where the power saving comes from.

    Weighting is conductance-based, so signed weights are free -- a negative
    weight is a device in the threshold bank. For a locking host that matters
    more than the PPA edge: unlike CRTL, no inverter is needed and no
    structural footprint is introduced.
    """
    t = spec.tech
    lines = [
        f".subckt DCSTL_{spec.fanin} {' '.join(spec.input_names)} E OUT OUTN VDD VSS"
    ]

    lines.append("* latched comparator, cross-coupled M1-M4")
    lat_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("1", "OUT", "OUTN", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    lines.append(_device("2", "OUTN", "OUT", "VDD", "VDD", t.pmos_model, lat_w, t.l_min))
    # As in LCTL: OUT discharges through the threshold bank (cb), OUTN through
    # the data bank (ca), so a data side that beats the reference leaves OUT
    # high rather than pulling it down.
    lines.append(_device("3", "OUT", "OUTN", "cb", "VSS", t.nmos_model, lat_w, t.l_min))
    lines.append(_device("4", "OUTN", "OUT", "ca", "VSS", t.nmos_model, lat_w, t.l_min))

    lines.append("* precharge, gated by E")
    pre_w = t.w(2.0) * spec.comparator_scale
    lines.append(_device("5", "OUT", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))
    lines.append(_device("6", "OUTN", "E", "VDD", "VDD", t.pmos_model, pre_w, t.l_min))

    lines.append("* cascodes M11 / M12 restrict internal node swing")
    cas_w = t.w(3.0) * spec.comparator_scale
    lines.append(_device("11", "ca", "VDD", "da", "VSS", t.nmos_model, cas_w, t.l_min))
    lines.append(_device("12", "cb", "VDD", "db", "VSS", t.nmos_model, cas_w, t.l_min))

    lines.append("* data mapping bank / threshold mapping bank, at the foot")
    idx = 0
    for name, weight in zip(spec.input_names, spec.all_weights):
        idx += 1
        bank = "da" if weight > 0 else "db"
        lines.extend(_unit_bank(f"d{idx}_", bank, name, abs(weight), t))

    # As in LCTL: the constant is T, since the negative-weight devices are
    # already on this bank and supply their own |w| when driven high.
    ref = max(1, spec.threshold)
    lines.extend(_unit_bank("tr_", "db", "VDD", ref, t))
    lines.append("* tie-breaking half unit, for weighted sum == threshold")
    lines.append(
        _device("tb", "da", "VDD", "foot", "VSS", t.nmos_model, t.w(1.0) / 2, t.l_min)
    )

    da_units = sum(abs(w) for w in spec.all_weights if w > 0) + 0.5
    db_units = sum(abs(w) for w in spec.all_weights if w < 0) + ref
    _balance_banks(lines, da_units, db_units, t)

    lines.append("* evaluation footer")
    foot_w = t.w(max(2.0, spec.total_weight / 2)) * spec.comparator_scale
    lines.append(_device("f", "foot", "E", "VSS", "VSS", t.nmos_model, foot_w, t.l_min))

    lines.append(".ends")
    return "\n".join(lines)


def subckt(spec: CellSpec) -> str:
    if spec.cell == "LCTL":
        return lctl_subckt(spec)
    if spec.cell == "CRTL":
        return crtl_subckt(spec)
    return dcstl_subckt(spec)


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
    area = active * diffusion_overhead * routing_factor * 1e12
    return area + capacitor_area_um2(spec)


def capacitors(spec: CellSpec) -> list[tuple[str, float]]:
    """(name, farads) for every input capacitor. Empty for conductance cells."""
    out = []
    for line in subckt(spec).splitlines():
        if not line.startswith("C"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            out.append((parts[0], float(parts[3])))
    return out


def capacitor_area_um2(spec: CellSpec, density_ff_per_um2: float = 2.0) -> float:
    """
    Silicon area of the input capacitor array.

    A capacitive TLG puts its weights in capacitors, so counting only
    transistors would make those weights look free and hand the capacitive
    family an area win it has not earned. MIM and poly-poly capacitors run on
    the order of 1-2 fF/um^2 at this node; the density is exposed because it
    is a process assumption, not a measurement.
    """
    total_ff = sum(c for _, c in capacitors(spec)) * 1e15
    return total_ff / density_ff_per_um2 if density_ff_per_um2 else 0.0


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

    # The toggled input has to be able to move the sum across the boundary on
    # its own, so it must carry positive weight.
    positive = [i for i, w in enumerate(weights) if w > 0]
    toggled_idx = max(positive, key=lambda i: weights[i]) if positive else 0
    others = [i for i in range(len(weights)) if i != toggled_idx]

    target = spec.threshold if rise else spec.threshold - 1
    base = weights[toggled_idx] if rise else 0
    need = target - base

    # Exact subset-sum over the static inputs. A greedy pass does not work
    # once weights can be negative -- taking a negative weight early moves the
    # running total *away* from the target, and the greedy has no way back --
    # which silently produced patterns that never crossed the threshold at
    # all, so the gate never switched and every delay measurement failed.
    #
    # Reachable sums are tracked with one witness assignment each; the space
    # is bounded by the total weight, which is small for a real cell.
    reach: dict[int, dict[str, int]] = {0: {}}
    for i in others:
        nxt = dict(reach)
        for total, assign in reach.items():
            for bit in (0, 1):
                s = total + weights[i] * bit
                if s not in nxt:
                    nxt[s] = {**assign, names[i]: bit}
        reach = nxt

    if need in reach:
        chosen = reach[need]
    else:
        # No exact hit: land as close as possible without overshooting, so the
        # measurement is still on the correct side of the boundary.
        below = [s for s in reach if s <= need]
        chosen = reach[max(below)] if below else reach[min(reach)]

    static = {names[i]: chosen.get(names[i], 0) for i in others}
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

    clk = CLOCK_PORT[spec.cell]
    lines.append(
        f"V{clk} {clk} 0 PULSE(0 {_fmt(vdd)} {_fmt(per * 0.75)}n "
        f"{_fmt(edge)}n {_fmt(edge)}n {_fmt(per * 0.6)}n {_fmt(per * 1.5)}n)"
    )

    lines.append(f"CL OUT 0 {_fmt(spec.load_ff)}f")
    lines.append(f"CLN OUTN 0 {_fmt(spec.load_ff)}f")

    ports = " ".join(spec.input_names)
    lines.append(f"XDUT {ports} {clk} OUT OUTN VDD VSS {subckt_name(spec)}")

    lines.append("")
    lines.append(f".tran {_fmt(edge / 20)}n {_fmt(per * 2)}n")
    lines.append("")

    # These are precharge cells: both outputs are pulled high during the
    # reset phase and exactly one of them falls during evaluation, so delay
    # is clock-to-Q and the node that moves depends on the logic value.
    #
    #   rise=True   sum == T      -> gate outputs 1, so OUTN is the one that
    #                                falls
    #   rise=False  sum == T - 1  -> gate outputs 0, so OUT falls
    #
    # Measuring the node that should *not* move would simply fail, which is a
    # useful property: a cell that resolves the wrong way fails loudly here
    # instead of quietly reporting a plausible delay.
    sense = "OUTN" if stim.rise else "OUT"
    clk_edge = per * 0.75
    lines.append(
        f".meas tran tpd TRIG V({clk}) VAL={_fmt(half)} RISE=1 "
        f"TARG V({sense}) VAL={_fmt(half)} FALL=1"
    )
    lines.append(
        f".meas tran trf TRIG V({sense}) VAL={_fmt(0.8 * vdd)} FALL=1 "
        f"TARG V({sense}) VAL={_fmt(0.2 * vdd)} FALL=1"
    )
    lines.append(f".meas tran iavg AVG I(VVDD) FROM=0 TO={_fmt(per * 2)}n")
    # ngspice has no ABS() inside .meas MAX, so take both extremes and let the
    # parser combine them.
    lines.append(f".meas tran imax MAX I(VVDD) FROM=0 TO={_fmt(per * 2)}n")
    lines.append(f".meas tran imin MIN I(VVDD) FROM=0 TO={_fmt(per * 2)}n")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines) + "\n"
