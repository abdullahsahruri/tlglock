"""
.th netlist format parser and writer.

Format (as given in Fig. 3 of Sahruri & Margala, VLSI-SoC 2025):

    .model c3_after_clp_locked.th
    .input K1 X1 X2 X3 Y2 Y3 K2
    .output Z
    .threshold K1 X1 X2 X3 Y2 Y3 K2 Z 3 2 1 3 2 1 -2 7

A `.threshold` line carries, in order:
    n input names, 1 output name, n integer weights, 1 integer threshold
so a well-formed line has exactly 2n + 2 tokens after the directive.

Gate semantics (Eq. 2, generalised by Eq. 3):
    out = 1  iff  sum_i w_i * x_i  >=  T
Key inputs are not distinguished structurally -- they are ordinary inputs
whose weights happen to have been assigned by the locking pass. That is the
whole point of the scheme: the lock has no structural footprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


class ThParseError(ValueError):
    """Raised on a malformed .th file."""


@dataclass
class ThGate:
    """A single threshold gate: out = 1 iff sum(w_i * x_i) >= threshold."""

    inputs: list[str]
    output: str
    weights: list[int]
    threshold: int

    def __post_init__(self) -> None:
        if len(self.inputs) != len(self.weights):
            raise ThParseError(
                f"gate '{self.output}': {len(self.inputs)} inputs but "
                f"{len(self.weights)} weights"
            )
        if len(set(self.inputs)) != len(self.inputs):
            dupes = {n for n in self.inputs if self.inputs.count(n) > 1}
            raise ThParseError(
                f"gate '{self.output}': repeated input name(s) {sorted(dupes)}"
            )

    @property
    def fanin(self) -> int:
        return len(self.inputs)

    def weight_of(self, name: str) -> int:
        try:
            return self.weights[self.inputs.index(name)]
        except ValueError:
            raise KeyError(f"'{name}' is not an input of gate '{self.output}'")

    def eval(self, values: dict[str, int]) -> int:
        """Evaluate this gate. `values` must bind every input name."""
        total = 0
        for name, w in zip(self.inputs, self.weights):
            try:
                total += w * values[name]
            except KeyError:
                raise KeyError(
                    f"gate '{self.output}': input '{name}' unbound"
                ) from None
        return 1 if total >= self.threshold else 0

    def weighted_sum(self, values: dict[str, int]) -> int:
        """The pre-comparison sum. Useful for margin / corruption analysis."""
        return sum(w * values[n] for n, w in zip(self.inputs, self.weights))

    def to_line(self) -> str:
        toks = (
            [".threshold"]
            + self.inputs
            + [self.output]
            + [str(w) for w in self.weights]
            + [str(self.threshold)]
        )
        return " ".join(toks)

    def copy(self) -> "ThGate":
        return ThGate(
            inputs=list(self.inputs),
            output=self.output,
            weights=list(self.weights),
            threshold=self.threshold,
        )


@dataclass
class ThNetwork:
    """A combinational network of threshold gates."""

    model: str = "unnamed"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    gates: list[ThGate] = field(default_factory=list)

    # -- structural queries -------------------------------------------------

    @property
    def driver(self) -> dict[str, ThGate]:
        """Map from signal name to the gate that drives it."""
        return {g.output: g for g in self.gates}

    @property
    def signals(self) -> set[str]:
        s = set(self.inputs)
        for g in self.gates:
            s.add(g.output)
            s.update(g.inputs)
        return s

    def fanout_count(self) -> dict[str, int]:
        """How many gates consume each signal (Algorithm 1's f_out)."""
        counts = {s: 0 for s in self.signals}
        for g in self.gates:
            for name in g.inputs:
                counts[name] += 1
        return counts

    def validate(self) -> None:
        """Check the network is well-formed and combinational."""
        drivers = {}
        for g in self.gates:
            if g.output in self.inputs:
                raise ThParseError(f"gate drives primary input '{g.output}'")
            if g.output in drivers:
                raise ThParseError(f"signal '{g.output}' has multiple drivers")
            drivers[g.output] = g

        known = set(self.inputs)
        for g in self.gates:
            for name in g.inputs:
                if name not in known and name not in drivers:
                    raise ThParseError(
                        f"gate '{g.output}': input '{name}' is undriven "
                        "and not a primary input"
                    )

        for o in self.outputs:
            if o not in drivers and o not in self.inputs:
                raise ThParseError(f"primary output '{o}' is undriven")

        # Combinational: a topological order must exist.
        self.topological_order()

    def topological_order(self) -> list[ThGate]:
        """Gates in evaluation order. Raises if the network has a cycle."""
        drivers = self.driver
        ready = set(self.inputs)
        remaining = list(self.gates)
        order: list[ThGate] = []

        while remaining:
            progressed = False
            still: list[ThGate] = []
            for g in remaining:
                if all(n in ready for n in g.inputs):
                    order.append(g)
                    ready.add(g.output)
                    progressed = True
                else:
                    still.append(g)
            remaining = still
            if not progressed:
                stuck = sorted(g.output for g in remaining)
                raise ThParseError(
                    f"combinational loop or undriven input among gates {stuck}"
                )
        return order

    def copy(self) -> "ThNetwork":
        return ThNetwork(
            model=self.model,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            gates=[g.copy() for g in self.gates],
        )

    def to_text(self) -> str:
        lines = [f".model {self.model}"]
        if self.inputs:
            lines.append(".input " + " ".join(self.inputs))
        if self.outputs:
            lines.append(".output " + " ".join(self.outputs))
        lines.extend(g.to_line() for g in self.gates)
        lines.append(".end")
        return "\n".join(lines) + "\n"


# -- parsing ---------------------------------------------------------------


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _join_continuations(lines: Iterable[str]) -> list[str]:
    """Fold backslash-continued lines together."""
    out: list[str] = []
    buf = ""
    for raw in lines:
        text = _strip_comment(raw).rstrip()
        if text.endswith("\\"):
            buf += text[:-1] + " "
            continue
        buf += text
        if buf.strip():
            out.append(buf.strip())
        buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _parse_threshold(tokens: Sequence[str], lineno: int) -> ThGate:
    if len(tokens) < 4 or len(tokens) % 2 != 0:
        raise ThParseError(
            f"line {lineno}: .threshold needs 2n+2 tokens for n inputs, "
            f"got {len(tokens)}"
        )
    n = (len(tokens) - 2) // 2
    names = list(tokens[: n + 1])
    nums = tokens[n + 1 :]
    try:
        vals = [int(t) for t in nums]
    except ValueError:
        bad = [t for t in nums if not _is_int(t)]
        raise ThParseError(
            f"line {lineno}: non-integer weight/threshold {bad}"
        ) from None

    try:
        return ThGate(
            inputs=names[:-1],
            output=names[-1],
            weights=vals[:-1],
            threshold=vals[-1],
        )
    except ThParseError as e:
        raise ThParseError(f"line {lineno}: {e}") from None


def _is_int(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def parse_th(text: str) -> ThNetwork:
    """Parse .th source into a ThNetwork. Does not validate connectivity."""
    net = ThNetwork()
    seen_model = False

    for lineno, line in enumerate(_join_continuations(text.splitlines()), start=1):
        toks = line.split()
        if not toks:
            continue
        directive, rest = toks[0], toks[1:]

        if directive == ".model":
            if not rest:
                raise ThParseError(f"line {lineno}: .model needs a name")
            net.model = rest[0]
            seen_model = True
        elif directive in (".input", ".inputs"):
            net.inputs.extend(rest)
        elif directive in (".output", ".outputs"):
            net.outputs.extend(rest)
        elif directive == ".threshold":
            net.gates.append(_parse_threshold(rest, lineno))
        elif directive == ".end":
            break
        else:
            raise ThParseError(f"line {lineno}: unknown directive '{directive}'")

    if not seen_model and not net.gates:
        raise ThParseError("empty or non-.th input")

    dupes = [n for n in net.inputs if net.inputs.count(n) > 1]
    if dupes:
        raise ThParseError(f"duplicate primary input(s) {sorted(set(dupes))}")

    return net


def read_th(path: str) -> ThNetwork:
    with open(path, "r") as fh:
        return parse_th(fh.read())


def write_th(net: ThNetwork, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(net.to_text())
