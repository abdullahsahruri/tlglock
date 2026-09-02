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

import random
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


# -- AppSAT -----------------------------------------------------------------


@dataclass
class AppSatResult:
    """
    Outcome of an AppSAT run.

    `status` records how the loop ended, in the same vocabulary sat_attack()
    uses -- stated explicitly here because Table I's own Result column is
    ambiguous about exactly this (see CLAUDE.md finding 4):

        UNSAT    the miter went UNSAT, so the loop ran to exact completion
                 and `key` is functionally equivalent to the real key
        SAT      the loop stopped early on the approximate criterion and
                 `key` is an approximate key
        TIMEOUT  the wall-clock budget expired first

    Prefer the `exact` / `settled` flags over `status` when branching; they
    say the same thing without the solver-verdict overloading.

    Two error figures are reported, because "output error" has two natural
    definitions and they are not the same number:

        error_patterns  fraction of sampled input patterns on which `key`
                        gets *any* primary output wrong
        error_bits      fraction of primary-output *bits* wrong, averaged
                        over sampled patterns -- what corruption_rate()
                        measures

    error_bits <= error_patterns always, with equality only on a single-output
    circuit or when every wrong pattern corrupts every output. AppSAT's
    epsilon bounds error_patterns. Reporting one and comparing it against the
    other compares different quantities, so both are recorded and the caller
    can say which it means.
    """

    status: Status
    key: dict[str, int] | None = None
    exact: bool = False
    settled: bool = False
    error_patterns: float = 1.0
    error_bits: float = 1.0
    iterations: int = 0
    rounds: int = 0
    queries: int = 0
    dips: list[dict[str, int]] = field(default_factory=list)
    conflicts: int = 0
    decisions: int = 0
    seconds: float = 0.0


def estimate_key_error(
    locked: ThNetwork,
    key: dict[str, int],
    oracle: Callable[[dict[str, int]], tuple[int, ...]],
    data_inputs: Sequence[str],
    samples: int,
    rng: random.Random,
) -> tuple[float, float, list[tuple[dict[str, int], tuple[int, ...]]]]:
    """
    Estimate the output error of `key` from random oracle queries.

    Returns (error_patterns, error_bits, counterexamples). The
    counterexamples are the (pattern, oracle response) pairs that disagreed --
    exactly the observations that refute `key` when fed back into the
    constraint set.

    A sample of size n cannot resolve an error rate below 1/n, so an epsilon
    below that is not measurable: use samples >= 1/epsilon if the bound is
    meant to be meaningful rather than decorative.
    """
    n_out = len(locked.outputs)
    if samples <= 0 or n_out == 0:
        return 0.0, 0.0, []

    bad_patterns = 0
    bad_bits = 0
    counterexamples: list[tuple[dict[str, int], tuple[int, ...]]] = []

    for _ in range(samples):
        x = {n: rng.randint(0, 1) for n in data_inputs}
        ref = oracle(x)
        got = outputs_of(locked, {**x, **key})
        diff = sum(1 for a, b in zip(ref, got) if a != b)
        if diff:
            bad_patterns += 1
            bad_bits += diff
            counterexamples.append((x, ref))

    return bad_patterns / samples, bad_bits / (samples * n_out), counterexamples


def _solve_for_key(
    locked: ThNetwork,
    key_names: Sequence[str],
    history: Sequence[tuple[dict[str, int], tuple[int, ...]]],
    solver: "Solver",
    timeout: float,
) -> tuple[dict[str, int] | None, SolveResult]:
    """A key consistent with every recorded observation, or None."""
    enc = build_key_formula(locked, key_names, history)
    res = solver.solve(enc, timeout=timeout)
    if res.status is not Status.SAT:
        return None, res
    return {k: res.model.get(enc.var(f"{k}_A"), 0) for k in key_names}, res


def appsat_attack(
    locked: ThNetwork,
    key_names: Sequence[str],
    oracle: Callable[[dict[str, int]], tuple[int, ...]],
    solver: Solver | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    round_size: int = 12,
    samples: int = 64,
    epsilon: float = 0.01,
    settle_rounds: int = 3,
    learn_from_queries: bool = True,
    max_iterations: int = 10_000,
    seed: int = 0,
    on_round: Callable[[int, float, float], None] | None = None,
) -> AppSatResult:
    """
    AppSAT: the oracle-guided attack with an approximate stopping rule.

    Shamsi et al., "AppSAT: Approximately Deobfuscating Integrated Circuits,"
    IEEE HOST 2017.

    AppSAT is *not* an approximate solver. The inner solve is the same exact
    call sat_attack() makes, on the same miter; what AppSAT relaxes is
    termination. Every `round_size` DIP iterations it extracts a candidate key
    consistent with everything observed so far, estimates that key's output
    error from `samples` random oracle queries, and stops once the estimate
    stays at or below `epsilon` for `settle_rounds` consecutive rounds.

    That distinction is the whole point when arguing resistance. A lock whose
    security rests on the DIP loop being long does not survive AppSAT, because
    AppSAT never has to finish the loop -- only to reach a key that is right
    often enough. A lock survives only if no low-error key exists at all,
    which is a claim about the corruption landscape rather than about solver
    tractability, and is measurable independently of how hard the solve is.

    Parameters worth setting deliberately:

      epsilon        error bound for stopping, on error_patterns. A negative
                     value disables early stopping entirely, which makes this
                     exactly sat_attack() with error reporting attached.
      samples        random queries per round. Cannot resolve an error rate
                     below 1/samples -- see estimate_key_error().
      settle_rounds  consecutive rounds required below epsilon, so one lucky
                     sample does not end the attack.

    `learn_from_queries` feeds the *disagreeing* random queries back into the
    constraint set. AppSAT as published adds every random query; adding only
    the counterexamples keeps the formula smaller and guarantees progress --
    the candidate just refuted cannot come back -- which matters here because
    the built-in PbSolver has no clause learning. Set it False to measure
    error without letting the queries prune the key space.
    """
    solver = solver or PbSolver()
    rng = random.Random(seed)
    keys = set(key_names)
    data_inputs = [n for n in locked.inputs if n not in keys]

    history: list[tuple[dict[str, int], tuple[int, ...]]] = []
    result = AppSatResult(status=Status.UNKNOWN)
    start = time.monotonic()
    streak = 0

    def elapsed() -> float:
        return time.monotonic() - start

    def timed_out() -> AppSatResult:
        result.status = Status.TIMEOUT
        result.seconds = elapsed()
        return result

    for it in range(max_iterations):
        remaining = timeout - elapsed()
        if remaining <= 0:
            return timed_out()

        enc = build_attack_formula(locked, key_names, history)
        res = solver.solve(enc, timeout=remaining)
        result.conflicts += res.conflicts
        result.decisions += res.decisions

        if res.status is Status.TIMEOUT:
            return timed_out()
        if res.status is Status.UNSAT:
            break

        dip = {n: res.model.get(enc.var(n), 0) for n in data_inputs}
        history.append((dip, oracle(dip)))
        result.dips.append(dip)
        result.iterations = it + 1

        if (it + 1) % round_size:
            continue

        # -- round boundary: is the best key so far already good enough? ----
        remaining = timeout - elapsed()
        if remaining <= 0:
            return timed_out()

        cand, kres = _solve_for_key(
            locked, key_names, history, solver, max(remaining, 1.0)
        )
        result.conflicts += kres.conflicts
        result.decisions += kres.decisions
        if cand is None:
            continue

        result.rounds += 1
        ep, eb, counter = estimate_key_error(
            locked, cand, oracle, data_inputs, samples, rng
        )
        result.queries += samples
        if on_round:
            on_round(result.rounds, ep, eb)
        if learn_from_queries and counter:
            history.extend(counter)

        if ep <= epsilon:
            streak += 1
            if streak >= settle_rounds:
                result.status = Status.SAT
                result.key = cand
                result.settled = True
                result.error_patterns = ep
                result.error_bits = eb
                result.seconds = elapsed()
                return result
        else:
            streak = 0
    else:
        result.status = Status.UNKNOWN
        result.seconds = elapsed()
        return result

    # -- the miter went UNSAT, so the loop completed exactly ----------------
    remaining = timeout - elapsed()
    cand, kres = _solve_for_key(
        locked, key_names, history, solver, max(remaining, 1.0)
    )
    result.conflicts += kres.conflicts
    result.decisions += kres.decisions

    if cand is None:
        if history:
            result.status = kres.status
            result.seconds = elapsed()
            return result
        # Nothing ever constrained the key, so no value of it can matter.
        cand = {k: 0 for k in key_names}

    ep, eb, _ = estimate_key_error(
        locked, cand, oracle, data_inputs, samples, rng
    )
    result.queries += samples
    result.status = Status.UNSAT
    result.key = cand
    result.exact = True
    result.error_patterns = ep
    result.error_bits = eb
    result.seconds = elapsed()
    return result
