"""
Oracle-guided SAT attack on a locked TLG network.

This is the attack the paper's Table I measures. The threat model is the
standard one from Subramanyan et al. [18]: the attacker holds the locked
netlist and an activated chip that can be queried as a black box, and wants
the key.

The loop:

    1. Find a distinguishing input -- a pattern x and two key vectors k_A,
       k_B that make the locked circuit disagree on some output. Any key
       consistent with everything learned so far is still a candidate, so a
       disagreement means at least one of the two is wrong.
    2. Query the oracle for the correct response y on x.
    3. Constrain both key copies to reproduce y on x. This eliminates every
       key that disagrees, typically a large fraction of the remaining space.
    4. Repeat. When step 1 is UNSAT, no two surviving keys differ anywhere,
       so any survivor is functionally equivalent to the real key.

Each iteration adds two circuit copies to the formula, so the instance grows
linearly in DIP count while the search space it prunes shrinks exponentially.
That is why the attack is fast on XOR locking -- and why TLGLock's resistance
has to come from making each individual solve hard, not from needing many
iterations.

Solvers are pluggable. `PbSolver` is a small complete DPLL over pseudo-Boolean
constraints, which needs no external binary and is enough for benchmarks up to
roughly c17/s27 scale -- it makes the loop testable and demonstrable here.
`ExternalSolver` shells out to MiniSAT+ or any PB solver following the
competition output convention, which is what Table I's numbers require.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

from .opb import Constraint, OpbEncoder
from .sim import outputs_of
from .thfile import ThNetwork

DEFAULT_TIMEOUT = 3600.0  # the paper's 1-hour limit


class Status(str, Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class SolveResult:
    status: Status
    model: dict[int, int] = field(default_factory=dict)
    conflicts: int = 0
    decisions: int = 0
    seconds: float = 0.0


class Solver(Protocol):
    def solve(self, enc: OpbEncoder, timeout: float) -> SolveResult: ...


# -- built-in solver --------------------------------------------------------


class PbSolver:
    """
    Complete DPLL over pseudo-Boolean constraints, with propagation.

    For a constraint sum(c_i x_i) >= rhs under a partial assignment, let
    `fixed` be the contribution of assigned variables and `slack` the largest
    the remainder can still reach. If fixed + slack < rhs the branch is dead.
    Otherwise any unassigned variable whose *other* value would push the best
    case below rhs is forced. That is the whole propagation rule; branching
    handles the rest.

    Not competitive with a real solver -- no learning, no restarts, no
    watched literals -- but complete, dependency-free, and honest about
    conflict counts.
    """

    def __init__(self, max_conflicts: int = 2_000_000):
        self.max_conflicts = max_conflicts

    def solve(self, enc: OpbEncoder, timeout: float = DEFAULT_TIMEOUT) -> SolveResult:
        start = time.monotonic()
        n = enc.num_vars
        cons = [(c.terms, c.rhs) for c in enc.constraints]

        # Index constraints by the variables they mention.
        occurs: list[list[int]] = [[] for _ in range(n + 1)]
        for ci, (terms, _) in enumerate(cons):
            for _, v in terms:
                occurs[v].append(ci)

        assign: list[int | None] = [None] * (n + 1)
        stats = {"conflicts": 0, "decisions": 0}

        def undo(trail: Sequence[int]) -> None:
            for v in trail:
                assign[v] = None

        def check_and_propagate(queue: list[int]) -> list[int] | None:
            """
            Propagate to fixpoint. Returns the variables it assigned, or None
            on conflict.

            On conflict the partial trail is unwound before returning. Leaving
            it in place would silently corrupt the assignment for every later
            branch -- the solver would then report models that violate the
            very constraints it was given.
            """
            trail: list[int] = []
            touched = set(range(len(cons))) if not queue else set()
            for v in queue:
                touched.update(occurs[v])

            while touched:
                ci = touched.pop()
                terms, rhs = cons[ci]
                fixed = 0
                slack = 0
                unassigned = []
                for c, v in terms:
                    a = assign[v]
                    if a is None:
                        unassigned.append((c, v))
                        if c > 0:
                            slack += c
                    else:
                        fixed += c * a
                if fixed + slack < rhs:
                    undo(trail)
                    return None
                for c, v in unassigned:
                    if assign[v] is not None:
                        continue
                    if c > 0 and fixed + slack - c < rhs:
                        assign[v] = 1
                        trail.append(v)
                        touched.update(occurs[v])
                    elif c < 0 and fixed + slack + c < rhs:
                        assign[v] = 0
                        trail.append(v)
                        touched.update(occurs[v])
            return trail

        def search(depth: int) -> bool:
            if time.monotonic() - start > timeout:
                raise TimeoutError
            nxt = next((v for v in range(1, n + 1) if assign[v] is None), None)
            if nxt is None:
                return True
            stats["decisions"] += 1
            for val in (1, 0):
                assign[nxt] = val
                trail = check_and_propagate([nxt])
                if trail is None:
                    stats["conflicts"] += 1
                    assign[nxt] = None
                    if stats["conflicts"] > self.max_conflicts:
                        raise TimeoutError
                    continue
                if search(depth + 1):
                    return True
                undo(trail)
                assign[nxt] = None
            return False

        import sys as _sys

        old_limit = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(old_limit, 10 * (n + 100)))
        try:
            root = check_and_propagate([])
            if root is None:
                return SolveResult(
                    Status.UNSAT, {}, stats["conflicts"], stats["decisions"],
                    time.monotonic() - start,
                )
            ok = search(0)
        except (TimeoutError, RecursionError):
            return SolveResult(
                Status.TIMEOUT, {}, stats["conflicts"], stats["decisions"],
                time.monotonic() - start,
            )
        finally:
            _sys.setrecursionlimit(old_limit)

        elapsed = time.monotonic() - start
        if not ok:
            return SolveResult(
                Status.UNSAT, {}, stats["conflicts"], stats["decisions"], elapsed
            )
        model = {v: (assign[v] or 0) for v in range(1, n + 1)}
        return SolveResult(
            Status.SAT, model, stats["conflicts"], stats["decisions"], elapsed
        )


# -- external solver --------------------------------------------------------


class ExternalSolver:
    """
    Drive an external PB solver -- MiniSAT+, clasp, RoundingSat, or similar.

    Expects the PB competition output convention: an `s` line carrying the
    status and `v` lines carrying the model as signed literals. Conflict and
    decision counts are scraped from comment lines when the solver reports
    them, since Table I has columns for both.
    """

    def __init__(
        self,
        binary: str = "minisat+",
        args: Sequence[str] = (),
        conflict_pattern: str = "conflicts",
        decision_pattern: str = "decisions",
    ):
        self.binary = binary
        self.args = list(args)
        self.conflict_pattern = conflict_pattern
        self.decision_pattern = decision_pattern

    def solve(self, enc: OpbEncoder, timeout: float = DEFAULT_TIMEOUT) -> SolveResult:
        import os
        import tempfile

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "instance.opb")
            enc.write(path)
            try:
                proc = subprocess.run(
                    [self.binary, *self.args, path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"solver '{self.binary}' not found on PATH"
                ) from None
            except subprocess.TimeoutExpired:
                return SolveResult(
                    Status.TIMEOUT, {}, 0, 0, time.monotonic() - start
                )

        return self._parse(proc.stdout, time.monotonic() - start)

    def _parse(self, out: str, elapsed: float) -> SolveResult:
        status = Status.UNKNOWN
        model: dict[int, int] = {}
        conflicts = decisions = 0

        for line in out.splitlines():
            line = line.strip()
            if line.startswith("s "):
                body = line[2:].strip().upper()
                if "UNSATISFIABLE" in body:
                    status = Status.UNSAT
                elif "SATISFIABLE" in body or "OPTIMUM" in body:
                    status = Status.SAT
                elif "UNKNOWN" in body:
                    status = Status.UNKNOWN
            elif line.startswith("v "):
                for tok in line[2:].split():
                    neg = tok.startswith("-")
                    name = tok.lstrip("-")
                    if name.startswith("x") and name[1:].isdigit():
                        model[int(name[1:])] = 0 if neg else 1
            elif line.startswith("c "):
                low = line.lower()
                for pat, key in (
                    (self.conflict_pattern, "conflicts"),
                    (self.decision_pattern, "decisions"),
                ):
                    if pat in low:
                        nums = [t for t in low.replace(":", " ").split() if t.isdigit()]
                        if nums:
                            if key == "conflicts":
                                conflicts = int(nums[-1])
                            else:
                                decisions = int(nums[-1])
        return SolveResult(status, model, conflicts, decisions, elapsed)


# -- the attack -------------------------------------------------------------


@dataclass
class AttackResult:
    status: Status
    key: dict[str, int] | None = None
    dips: list[dict[str, int]] = field(default_factory=list)
    iterations: int = 0
    conflicts: int = 0
    decisions: int = 0
    seconds: float = 0.0
    key_is_correct: bool | None = None

    @property
    def table_row(self) -> str:
        """Format matching Table I's solver columns."""
        if self.status is Status.TIMEOUT:
            return "---,---,---,Timeout"
        return f"{self.conflicts},{self.decisions},{self.seconds:.2f},{self.status.value}"


def _encode_oracle_copy(
    enc: OpbEncoder,
    locked: ThNetwork,
    key_names: Sequence[str],
    key_suffix: str,
    tag: str,
    pattern: dict[str, int],
    response: Sequence[int],
) -> None:
    """
    Add "this key set reproduces `response` on `pattern`" to the formula.

    Internal signals get a private name so copies do not collide, but the key
    variables map onto the miter's own key variables -- that shared mapping is
    what makes the constraint bind.
    """
    rename = {k: f"{k}{key_suffix}" for k in key_names}
    mapping = enc.encode_network(locked, suffix=tag, share=(), rename=rename)

    for name, val in pattern.items():
        enc.fix(mapping.get(name, name), val)
    for out, val in zip(locked.outputs, response):
        enc.fix(mapping.get(out, out), val)


def build_attack_formula(
    locked: ThNetwork,
    key_names: Sequence[str],
    history: Sequence[tuple[dict[str, int], tuple[int, ...]]],
) -> OpbEncoder:
    """
    The iteration-i miter: two key sets that agree with every recorded
    observation but disagree somewhere on a fresh input.
    """
    keys = set(key_names)
    data_inputs = [n for n in locked.inputs if n not in keys]

    enc = OpbEncoder()
    for n in data_inputs:
        enc.var(n)
    for k in key_names:
        enc.var(f"{k}_A")
        enc.var(f"{k}_B")

    enc.encode_network(
        locked, suffix="_A", share=data_inputs,
        rename={k: f"{k}_A" for k in key_names},
    )
    enc.encode_network(
        locked, suffix="_B", share=data_inputs,
        rename={k: f"{k}_B" for k in key_names},
    )

    diffs = []
    for o in locked.outputs:
        d = enc.fresh(f"diff_{o}_")
        enc.encode_xor(enc.var(f"{o}_A"), enc.var(f"{o}_B"), d)
        diffs.append(d)
    enc.add([(1, d) for d in diffs], 1, "at least one output differs")

    for i, (pattern, response) in enumerate(history):
        for suffix in ("_A", "_B"):
            _encode_oracle_copy(
                enc, locked, key_names, suffix, f"{suffix}_o{i}", pattern, response
            )
    return enc


def build_key_formula(
    locked: ThNetwork,
    key_names: Sequence[str],
    history: Sequence[tuple[dict[str, int], tuple[int, ...]]],
) -> OpbEncoder:
    """Constraints a surviving key must satisfy, with no disagreement clause."""
    enc = OpbEncoder()
    for k in key_names:
        enc.var(f"{k}_A")
    for i, (pattern, response) in enumerate(history):
        _encode_oracle_copy(
            enc, locked, key_names, "_A", f"_A_o{i}", pattern, response
        )
    return enc


def sat_attack(
    locked: ThNetwork,
    key_names: Sequence[str],
    oracle: Callable[[dict[str, int]], tuple[int, ...]],
    solver: Solver | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_iterations: int = 10_000,
    on_iteration: Callable[[int, dict[str, int]], None] | None = None,
) -> AttackResult:
    """
    Run the oracle-guided attack.

    `oracle` maps a data-input pattern to the activated circuit's response --
    in an experiment, the original unlocked network. The attack never sees the
    correct key, only the oracle's answers.

    The timeout is a wall-clock budget across the whole attack, matching the
    paper's 1-hour-per-benchmark protocol, and is passed down to each
    individual solve so a single hard instance cannot overrun it.
    """
    solver = solver or PbSolver()
    keys = set(key_names)
    data_inputs = [n for n in locked.inputs if n not in keys]

    history: list[tuple[dict[str, int], tuple[int, ...]]] = []
    result = AttackResult(status=Status.UNKNOWN)
    start = time.monotonic()

    for it in range(max_iterations):
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            result.status = Status.TIMEOUT
            result.iterations = it
            result.seconds = time.monotonic() - start
            return result

        enc = build_attack_formula(locked, key_names, history)
        res = solver.solve(enc, timeout=remaining)
        result.conflicts += res.conflicts
        result.decisions += res.decisions

        if res.status is Status.TIMEOUT:
            result.status = Status.TIMEOUT
            result.iterations = it
            result.seconds = time.monotonic() - start
            return result

        if res.status is Status.UNSAT:
            result.iterations = it
            break

        dip = {n: res.model.get(enc.var(n), 0) for n in data_inputs}
        response = oracle(dip)
        history.append((dip, response))
        result.dips.append(dip)
        if on_iteration:
            on_iteration(it, dip)
    else:
        result.status = Status.UNKNOWN
        result.iterations = max_iterations
        result.seconds = time.monotonic() - start
        return result

    # UNSAT: every surviving key is equivalent. Extract one.
    remaining = timeout - (time.monotonic() - start)
    key_enc = build_key_formula(locked, key_names, history)
    key_res = solver.solve(key_enc, timeout=max(remaining, 1.0))
    result.seconds = time.monotonic() - start

    if key_res.status is not Status.SAT:
        # No observations recorded means the key never mattered: any value
        # works, so report the all-zero key rather than failing.
        if not history:
            result.status = Status.UNSAT
            result.key = {k: 0 for k in key_names}
            return result
        result.status = key_res.status
        return result

    result.status = Status.UNSAT
    result.key = {
        k: key_res.model.get(key_enc.var(f"{k}_A"), 0) for k in key_names
    }
    return result


def oracle_from(original: ThNetwork) -> Callable[[dict[str, int]], tuple[int, ...]]:
    """Build an oracle callable from the unlocked reference network."""

    def query(pattern: dict[str, int]) -> tuple[int, ...]:
        return outputs_of(original, pattern)

    return query


def verify_recovered_key(
    original: ThNetwork,
    locked: ThNetwork,
    key: dict[str, int],
    limit: int = 1 << 16,
) -> bool:
    """
    Did the attack actually succeed?

    Checks functional equivalence, not equality with the planted key: the
    attack recovers a key equivalent to the real one, and threshold weight
    degeneracy means those are often different bit strings. Reporting failure
    because the bits differ would understate the attack.
    """
    from itertools import product

    data_inputs = [n for n in locked.inputs if n not in key]
    if (1 << len(data_inputs)) > limit:
        raise ValueError(f"{len(data_inputs)} data inputs is too many to verify")

    for bits in product((0, 1), repeat=len(data_inputs)):
        assign = dict(zip(data_inputs, bits))
        if outputs_of(original, assign) != outputs_of(locked, {**assign, **key}):
            return False
    return True
