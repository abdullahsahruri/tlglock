"""
Synthesis front end: BENCH/BLIF -> AIG -> TLG network.

Steps 1-3 of Fig. 3 in the TLGLock paper. Boolean structure generation, then
threshold cut computation and covering.

Two paths in:

  * If the `abc` binary is on PATH, run_abc() drives it for structural
    hashing and optional rewriting. That is the preferred route for real
    benchmarks, since ABC's AIG optimisation is far better than anything
    reimplemented here.
  * If not, the built-in BENCH and BLIF readers construct the AIG directly.
    Sufficient for testing, and for circuits already in a reasonable form.

The mapping itself is native, because ABC has no notion of a threshold cut:
its `if -K` mapper accepts a cut when the cut is small enough, whereas a TLG
mapper accepts a cut when the cut's local function is linearly separable. That
test is an LP (see separable.py), it is not monotone in cut size, and it is
not cheap. Cost-model design under an LP feasibility test is an open problem;
what is implemented here is deliberately the simple version -- enumerate
k-feasible cuts, keep the separable ones, cover greedily by depth then size.
Good enough to produce correct netlists and to measure against.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .separable import ThresholdRealisation, identify
from .thfile import ThGate, ThNetwork

DEFAULT_CUT_SIZE = 6          # LUT6-sized cuts, matching the FPGA work
DEFAULT_CUT_LIMIT = 12        # priority cuts retained per node


class SynthError(RuntimeError):
    pass


# -- AIG --------------------------------------------------------------------


@dataclass
class AigNode:
    """An AND node. Fanin literals are 2*node_id + inverted."""

    id: int
    fan0: int = 0
    fan1: int = 0
    is_pi: bool = False

    @property
    def lit(self) -> int:
        return self.id << 1


@dataclass
class Aig:
    """And-inverter graph. Node 0 is the constant zero."""

    nodes: list[AigNode] = field(default_factory=lambda: [AigNode(0)])
    pi_names: list[str] = field(default_factory=list)
    po_names: list[str] = field(default_factory=list)
    po_lits: list[int] = field(default_factory=list)
    _pi_of: dict[str, int] = field(default_factory=dict)
    _strash: dict[tuple[int, int], int] = field(default_factory=dict)

    CONST0 = 0
    CONST1 = 1

    def add_pi(self, name: str) -> int:
        if name in self._pi_of:
            return self._pi_of[name]
        node = AigNode(len(self.nodes), is_pi=True)
        self.nodes.append(node)
        self.pi_names.append(name)
        self._pi_of[name] = node.lit
        return node.lit

    def pi_lit(self, name: str) -> int:
        if name not in self._pi_of:
            raise SynthError(f"unknown primary input '{name}'")
        return self._pi_of[name]

    def add_and(self, a: int, b: int) -> int:
        """Structurally hashed AND with constant folding."""
        if a == self.CONST0 or b == self.CONST0:
            return self.CONST0
        if a == self.CONST1:
            return b
        if b == self.CONST1:
            return a
        if a == b:
            return a
        if a == (b ^ 1):
            return self.CONST0
        key = (min(a, b), max(a, b))
        if key in self._strash:
            return self._strash[key]
        node = AigNode(len(self.nodes), fan0=key[0], fan1=key[1])
        self.nodes.append(node)
        self._strash[key] = node.lit
        return node.lit

    def add_or(self, a: int, b: int) -> int:
        return self.add_and(a ^ 1, b ^ 1) ^ 1

    def add_xor(self, a: int, b: int) -> int:
        return self.add_or(self.add_and(a, b ^ 1), self.add_and(a ^ 1, b))

    def add_po(self, name: str, lit: int) -> None:
        self.po_names.append(name)
        self.po_lits.append(lit)

    # -- queries ------------------------------------------------------------

    def node(self, lit: int) -> AigNode:
        return self.nodes[lit >> 1]

    def is_pi(self, lit: int) -> bool:
        return self.nodes[lit >> 1].is_pi

    def is_const(self, lit: int) -> bool:
        return (lit >> 1) == 0

    @property
    def num_ands(self) -> int:
        return sum(1 for n in self.nodes if not n.is_pi and n.id != 0)

    def topo_ids(self) -> list[int]:
        """AND node ids in dependency order. Construction already ensures it."""
        return [n.id for n in self.nodes if not n.is_pi and n.id != 0]

    def levels(self) -> dict[int, int]:
        lev = {0: 0}
        for n in self.nodes:
            if n.id == 0:
                continue
            if n.is_pi:
                lev[n.id] = 0
            else:
                lev[n.id] = 1 + max(lev[n.fan0 >> 1], lev[n.fan1 >> 1])
        return lev

    def simulate_lit(self, lit: int, values: dict[int, int]) -> int:
        """Evaluate one literal given node-id -> value for the support."""
        nid = lit >> 1
        inv = lit & 1
        if nid == 0:
            return inv
        if nid in values:
            return values[nid] ^ inv
        node = self.nodes[nid]
        if node.is_pi:
            raise SynthError(f"primary input node {nid} has no value")
        v = self.simulate_lit(node.fan0, values) & self.simulate_lit(node.fan1, values)
        values[nid] = v
        return v ^ inv


# -- readers ----------------------------------------------------------------


def read_bench(text: str, name: str = "bench") -> Aig:
    """
    Parse ISCAS .bench format.

    Recognises INPUT/OUTPUT declarations and AND/OR/NAND/NOR/NOT/BUF/XOR/XNOR
    gates with arbitrary fan-in. Sequential elements (DFF) are rejected: the
    flow is combinational, and silently dropping state would produce a netlist
    that looks fine and means something else.
    """
    aig = Aig()
    lits: dict[str, int] = {}
    assignments: list[tuple[str, str, list[str]]] = []
    outputs: list[str] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("INPUT(") and line.endswith(")"):
            sig = line[line.index("(") + 1 : -1].strip()
            lits[sig] = aig.add_pi(sig)
        elif upper.startswith("OUTPUT(") and line.endswith(")"):
            outputs.append(line[line.index("(") + 1 : -1].strip())
        elif "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if "(" not in rhs:
                raise SynthError(f"malformed bench assignment: {raw!r}")
            op = rhs[: rhs.index("(")].strip().upper()
            args = [
                a.strip()
                for a in rhs[rhs.index("(") + 1 : rhs.rindex(")")].split(",")
                if a.strip()
            ]
            if op in ("DFF", "LATCH"):
                raise SynthError(
                    f"sequential element '{op}' at {lhs}: this flow is "
                    "combinational only"
                )
            assignments.append((lhs, op, args))
        else:
            raise SynthError(f"unparsable bench line: {raw!r}")

    # Assignments may be listed out of order; resolve iteratively.
    pending = list(assignments)
    while pending:
        progressed = False
        still = []
        for lhs, op, args in pending:
            if all(a in lits for a in args):
                lits[lhs] = _apply_gate(aig, op, [lits[a] for a in args], lhs)
                progressed = True
            else:
                still.append((lhs, op, args))
        if not progressed:
            missing = {a for _, _, args in still for a in args if a not in lits}
            raise SynthError(f"undefined or cyclic signals: {sorted(missing)}")
        pending = still

    for o in outputs:
        if o not in lits:
            raise SynthError(f"output '{o}' is never driven")
        aig.add_po(o, lits[o])

    aig_name = name
    aig.name = aig_name  # type: ignore[attr-defined]
    return aig


def _apply_gate(aig: Aig, op: str, args: list[int], lhs: str) -> int:
    if op in ("BUF", "BUFF"):
        _need(args, 1, op, lhs)
        return args[0]
    if op in ("NOT", "INV"):
        _need(args, 1, op, lhs)
        return args[0] ^ 1
    if not args:
        raise SynthError(f"gate {op} at '{lhs}' has no inputs")

    if op == "AND":
        return _fold(aig.add_and, args)
    if op == "NAND":
        return _fold(aig.add_and, args) ^ 1
    if op == "OR":
        return _fold(aig.add_or, args)
    if op == "NOR":
        return _fold(aig.add_or, args) ^ 1
    if op == "XOR":
        return _fold(aig.add_xor, args)
    if op in ("XNOR", "NXOR"):
        return _fold(aig.add_xor, args) ^ 1
    raise SynthError(f"unsupported bench gate '{op}' at '{lhs}'")


def _need(args: list[int], n: int, op: str, lhs: str) -> None:
    if len(args) != n:
        raise SynthError(f"{op} at '{lhs}' needs {n} input(s), got {len(args)}")


def _fold(fn, args: list[int]) -> int:
    acc = args[0]
    for a in args[1:]:
        acc = fn(acc, a)
    return acc


def read_blif(text: str, name: str = "blif") -> Aig:
    """
    Parse a flat combinational BLIF.

    Handles .model/.inputs/.outputs/.names with SOP covers, including
    don't-care input patterns. Latches and subcircuits are rejected.
    """
    aig = Aig()
    lits: dict[str, int] = {}
    outputs: list[str] = []
    model = name

    lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            lines.append(buf.strip())
        buf = ""
    if buf.strip():
        lines.append(buf.strip())

    blocks: list[tuple[list[str], list[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        toks = line.split()
        if toks[0] == ".model":
            model = toks[1] if len(toks) > 1 else name
            i += 1
        elif toks[0] in (".inputs", ".input"):
            for s in toks[1:]:
                lits[s] = aig.add_pi(s)
            i += 1
        elif toks[0] in (".outputs", ".output"):
            outputs.extend(toks[1:])
            i += 1
        elif toks[0] == ".names":
            signals = toks[1:]
            rows = []
            i += 1
            while i < len(lines) and not lines[i].startswith("."):
                rows.append(lines[i].split())
                i += 1
            blocks.append((signals, rows))
        elif toks[0] == ".end":
            break
        elif toks[0] in (".latch", ".subckt"):
            raise SynthError(
                f"'{toks[0]}' found: this flow is combinational and flat only"
            )
        else:
            i += 1

    pending = list(blocks)
    while pending:
        progressed = False
        still = []
        for signals, rows in pending:
            ins, out = signals[:-1], signals[-1]
            if all(s in lits for s in ins):
                lits[out] = _sop_to_aig(aig, [lits[s] for s in ins], rows, out)
                progressed = True
            else:
                still.append((signals, rows))
        if not progressed:
            missing = {
                s for sig, _ in still for s in sig[:-1] if s not in lits
            }
            raise SynthError(f"undefined or cyclic signals: {sorted(missing)}")
        pending = still

    for o in outputs:
        if o not in lits:
            raise SynthError(f"output '{o}' is never driven")
        aig.add_po(o, lits[o])

    aig.name = model  # type: ignore[attr-defined]
    return aig


def _sop_to_aig(aig: Aig, ins: list[int], rows: list[list[str]], out: str) -> int:
    if not rows:
        return Aig.CONST0
    onset_rows = []
    off_polarity = False
    for row in rows:
        if len(ins) == 0:
            # constant node: a single output value
            val = row[-1]
            return Aig.CONST1 if val == "1" else Aig.CONST0
        if len(row) != 2:
            raise SynthError(f"malformed .names row for '{out}': {row}")
        cube, val = row
        if val == "0":
            off_polarity = True
        onset_rows.append(cube)

    if off_polarity:
        # An off-set cover; build it and invert.
        acc = Aig.CONST0
        for cube in onset_rows:
            acc = aig.add_or(acc, _cube_to_aig(aig, ins, cube, out))
        return acc ^ 1

    acc = Aig.CONST0
    for cube in onset_rows:
        acc = aig.add_or(acc, _cube_to_aig(aig, ins, cube, out))
    return acc


def _cube_to_aig(aig: Aig, ins: list[int], cube: str, out: str) -> int:
    if len(cube) != len(ins):
        raise SynthError(
            f"cube '{cube}' has {len(cube)} literals but '{out}' has {len(ins)} inputs"
        )
    acc = Aig.CONST1
    for lit, ch in zip(ins, cube):
        if ch == "1":
            acc = aig.add_and(acc, lit)
        elif ch == "0":
            acc = aig.add_and(acc, lit ^ 1)
        elif ch != "-":
            raise SynthError(f"bad cube character '{ch}' for '{out}'")
    return acc


# -- ABC binary -------------------------------------------------------------


def abc_available(binary: str = "abc") -> bool:
    return shutil.which(binary) is not None


def run_abc(
    path: str,
    script: str = "strash; rewrite -l; refactor -l; strash",
    binary: str = "abc",
    timeout: int = 600,
) -> str:
    """
    Run ABC on a circuit file and return optimised BLIF.

    Raises SynthError if the binary is missing, so a caller can fall back to
    the built-in readers rather than failing outright.
    """
    if not abc_available(binary):
        raise SynthError(
            f"'{binary}' not found on PATH -- install ABC or use the "
            "built-in read_bench/read_blif readers"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.blif")
        cmd = f'read {path}; {script}; write_blif {out}'
        proc = subprocess.run(
            [binary, "-q", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise SynthError(
                f"abc failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        with open(out) as fh:
            return fh.read()


def read_circuit(path: str, use_abc: bool | None = None) -> Aig:
    """
    Read a .bench or .blif file, optionally preprocessing through ABC.

    use_abc=None means "use it if present".
    """
    if use_abc is None:
        use_abc = abc_available()

    if use_abc:
        try:
            return read_blif(run_abc(path), name=os.path.basename(path))
        except SynthError:
            if use_abc is True:
                raise
    with open(path) as fh:
        text = fh.read()
    if path.lower().endswith(".bench"):
        return read_bench(text, name=os.path.basename(path))
    return read_blif(text, name=os.path.basename(path))


# -- cut enumeration --------------------------------------------------------


Cut = frozenset[int]  # node ids forming the cut's support


def enumerate_cuts(
    aig: Aig, k: int = DEFAULT_CUT_SIZE, limit: int = DEFAULT_CUT_LIMIT
) -> dict[int, list[Cut]]:
    """
    k-feasible cuts per node, by the standard bottom-up merge.

    Each node keeps the trivial cut plus merges of its fanins' cuts, pruned to
    `limit` entries ranked by size then by support id for determinism.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cuts: dict[int, list[Cut]] = {}

    for node in aig.nodes:
        if node.id == 0:
            cuts[0] = [frozenset()]
            continue
        trivial: Cut = frozenset({node.id})
        if node.is_pi:
            cuts[node.id] = [trivial]
            continue

        a, b = node.fan0 >> 1, node.fan1 >> 1
        merged: set[Cut] = set()
        for ca in cuts.get(a, [frozenset({a})]):
            for cb in cuts.get(b, [frozenset({b})]):
                u = ca | cb
                if len(u) <= k:
                    merged.add(u)
        merged.add(trivial)
        ranked = sorted(merged, key=lambda c: (len(c), sorted(c)))
        cuts[node.id] = ranked[:limit]
        if trivial not in cuts[node.id]:
            cuts[node.id][-1] = trivial
    return cuts


def cut_function(aig: Aig, root: int, cut: Cut) -> list[int]:
    """
    Truth table of `root` as a function of `cut`, in truth_bits() order.

    Support variables are ordered by node id, which is the order the emitted
    ThGate's inputs will use.
    """
    support = sorted(cut)
    n = len(support)
    if n > 20:
        raise SynthError(f"cut of size {n} is too large to tabulate")

    table = []
    for idx in range(1 << n):
        values = {
            nid: (idx >> (n - 1 - i)) & 1 for i, nid in enumerate(support)
        }
        table.append(aig.simulate_lit(root << 1, dict(values)))
    return table


# -- mapping ----------------------------------------------------------------


@dataclass
class MapStats:
    gates: int = 0
    depth: int = 0
    cuts_tried: int = 0
    cuts_separable: int = 0
    max_weight: int = 0
    max_fanin: int = 0

    @property
    def separable_fraction(self) -> float:
        return self.cuts_separable / self.cuts_tried if self.cuts_tried else 0.0


def map_to_tlg(
    aig: Aig,
    k: int = DEFAULT_CUT_SIZE,
    limit: int = DEFAULT_CUT_LIMIT,
    model: str | None = None,
) -> tuple[ThNetwork, MapStats]:
    """
    Cover the AIG with threshold gates.

    For each node, take the best separable cut -- ranked by depth of the cut's
    support, then by size -- and emit one TLG. Nodes not reachable from a
    primary output through the chosen cuts are dropped.

    The trivial two-input cut of an AND node is always separable, so a cover
    always exists; the interesting question is how much larger than that the
    mapper can go, which is what MapStats records.
    """
    cuts = enumerate_cuts(aig, k=k, limit=limit)
    levels = aig.levels()
    stats = MapStats()

    chosen: dict[int, tuple[Cut, ThresholdRealisation]] = {}
    cut_depth: dict[int, int] = {n.id: 0 for n in aig.nodes if n.is_pi or n.id == 0}

    for nid in aig.topo_ids():
        best: tuple[tuple[int, int], Cut, ThresholdRealisation] | None = None
        for cut in cuts[nid]:
            if cut == frozenset({nid}) or not cut:
                continue
            stats.cuts_tried += 1
            table = cut_function(aig, nid, cut)
            real = identify(table, len(cut))
            if real is None:
                continue
            stats.cuts_separable += 1
            depth = 1 + max((cut_depth.get(s, 0) for s in cut), default=0)
            score = (depth, len(cut))
            if best is None or score < best[0]:
                best = (score, cut, real)

        if best is None:
            # Fall back to the structural AND, which is always separable.
            node = aig.nodes[nid]
            a, b = node.fan0, node.fan1
            cut = frozenset({a >> 1, b >> 1})
            table = cut_function(aig, nid, cut)
            real = identify(table, len(cut))
            if real is None:
                raise SynthError(f"node {nid}: even the structural cut is not separable")
            best = ((1 + max(cut_depth.get(s, 0) for s in cut), len(cut)), cut, real)

        score, cut, real = best
        chosen[nid] = (cut, real)
        cut_depth[nid] = score[0]

    # -- emit, keeping only what the outputs need ---------------------------
    net = ThNetwork(model=model or getattr(aig, "name", "mapped"))
    net.inputs = list(aig.pi_names)

    name_of: dict[int, str] = {0: "__const0"}
    for i, nm in enumerate(aig.pi_names, start=1):
        name_of[i] = nm
    for nid in aig.topo_ids():
        name_of[nid] = f"n{nid}"

    needed: set[int] = set()
    stack = [lit >> 1 for lit in aig.po_lits]
    while stack:
        nid = stack.pop()
        if nid in needed or nid == 0 or aig.nodes[nid].is_pi:
            continue
        needed.add(nid)
        stack.extend(chosen[nid][0])

    const_used = False
    for nid in aig.topo_ids():
        if nid not in needed:
            continue
        cut, real = chosen[nid]
        support = sorted(cut)
        if 0 in support:
            const_used = True
        net.gates.append(
            real.to_gate([name_of[s] for s in support], name_of[nid])
        )
        stats.max_weight = max(stats.max_weight, real.max_weight)
        stats.max_fanin = max(stats.max_fanin, len(support))

    if const_used:
        # A gate with no inputs and threshold 1 is constant zero.
        net.gates.insert(0, ThGate(inputs=[], output="__const0", weights=[], threshold=1))

    # Primary outputs. .th has no output-polarity field, so an inverted PO
    # literal needs a real inverting gate; name it for the output directly
    # rather than chaining a buffer behind it.
    driven = {g.output for g in net.gates}
    for name, lit in zip(aig.po_names, aig.po_lits):
        src = name_of[lit >> 1]
        if name in driven or name in net.inputs:
            # The output name is already taken by the node that drives it, or
            # by a primary input. Either way it needs its own gate under a
            # distinct name only if the polarity differs.
            if lit & 1:
                raise SynthError(
                    f"output '{name}' is inverted but its name is already driven"
                )
            if src != name:
                net.gates.append(
                    ThGate(inputs=[src], output=name, weights=[1], threshold=1)
                )
                driven.add(name)
        elif lit & 1:
            net.gates.append(
                ThGate(inputs=[src], output=name, weights=[-1], threshold=0)
            )
            driven.add(name)
        else:
            net.gates.append(
                ThGate(inputs=[src], output=name, weights=[1], threshold=1)
            )
            driven.add(name)
        net.outputs.append(name)

    net.validate()
    stats.gates = len(net.gates)
    stats.depth = _depth_of(net)
    return net, stats


def _depth_of(net: ThNetwork) -> int:
    level = {n: 0 for n in net.inputs}
    d = 0
    for g in net.topological_order():
        level[g.output] = 1 + max((level.get(i, 0) for i in g.inputs), default=0)
        d = max(d, level[g.output])
    return d


def synthesize(
    path: str,
    k: int = DEFAULT_CUT_SIZE,
    limit: int = DEFAULT_CUT_LIMIT,
    use_abc: bool | None = None,
) -> tuple[ThNetwork, MapStats]:
    """Read a circuit file and map it to a TLG network. Steps 1-3 of Fig. 3."""
    aig = read_circuit(path, use_abc=use_abc)
    return map_to_tlg(aig, k=k, limit=limit)
