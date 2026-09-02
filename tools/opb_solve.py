#!/usr/bin/env python3
"""
A pseudo-Boolean solver for tlglock's .opb files, built on PySAT.

`attack.ExternalSolver` shells out to a PB solver and parses the PB
competition output convention. MiniSAT+ is the one the paper used, but it is
long unmaintained and not packaged anywhere current. This gives the same
interface without needing root:

    from tlglock import ExternalSolver, sat_attack
    sat_attack(locked, keys, oracle,
               solver=ExternalSolver(binary="tools/opb_solve.py"))

It reads the OPB dialect opb.py emits, encodes each linear constraint to CNF
with a sequential/totalizer encoding, and hands the result to a real CDCL
solver. That is deliberately the encoding an actual attacker would use --
polynomial in fan-in and threshold, not the exponential prime-implicant
expansion. encoding.py measures the difference; this is the practical
consequence of it, and it means a timeout here is a statement about the
instance rather than about a bad choice of representation.

Deliberately outside src/tlglock/: the library itself has no runtime
dependencies and should keep it that way. This script needs
`pip install --user python-sat pypblib`.

Usage:
    opb_solve.py instance.opb [--solver cadical153]
"""

from __future__ import annotations

import argparse
import re
import sys

TERM = re.compile(r"([+-]?\d+)\s+x(\d+)")


def parse_opb(text: str) -> tuple[int, list[tuple[list[tuple[int, int]], int]]]:
    """
    Parse the OPB subset opb.py writes.

    Returns (n_vars, constraints), each constraint being (terms, rhs) with the
    relation always >=, which is what OpbEncoder normalises to.
    """
    n_vars = 0
    constraints: list[tuple[list[tuple[int, int]], int]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("*"):
            m = re.search(r"#variable=\s*(\d+)", line)
            if m:
                n_vars = max(n_vars, int(m.group(1)))
            continue
        if line.startswith("min:") or line.startswith("max:"):
            continue  # decision problem only; objective ignored
        if ">=" not in line:
            raise ValueError(f"unsupported constraint (expected >=): {line!r}")

        body, rhs_text = line.split(">=", 1)
        rhs = int(rhs_text.strip().rstrip(";").strip())
        terms = [(int(c), int(v)) for c, v in TERM.findall(body)]
        if not terms:
            raise ValueError(f"no terms parsed from: {line!r}")
        n_vars = max(n_vars, max(v for _, v in terms))
        constraints.append((terms, rhs))

    return n_vars, constraints


def to_cnf(n_vars, constraints):
    """
    Encode every PB constraint into CNF.

    Negative coefficients are normalised by literal flip rather than dropped:
    for w < 0, w*x = w + |w|*(not x), so the term becomes |w| on the negated
    literal and the bound rises by |w|. Getting this wrong is silent -- the
    instance still solves, just not the one you meant -- so the round-trip is
    checked by tests/test_opb_solve.py against the built-in solver.
    """
    from pysat.formula import IDPool
    from pysat.pb import PBEnc

    pool = IDPool(start_from=n_vars + 1)
    clauses: list[list[int]] = []

    for terms, rhs in constraints:
        lits, weights = [], []
        bound = rhs
        for coeff, var in terms:
            if coeff == 0:
                continue
            if coeff > 0:
                lits.append(var)
                weights.append(coeff)
            else:
                lits.append(-var)
                weights.append(-coeff)
                bound += -coeff          # rhs - w, with w negative

        if not lits:
            if bound > 0:
                return [[]], pool        # 0 >= positive: unsatisfiable
            continue
        if bound <= 0:
            continue                     # trivially satisfied
        if bound > sum(weights):
            return [[]], pool            # unreachable: unsatisfiable

        cnf = PBEnc.geq(
            lits=lits, weights=weights, bound=bound, vpool=pool
        )
        clauses.extend(cnf.clauses)

    return clauses, pool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance")
    ap.add_argument("--solver", default="cadical153")
    args = ap.parse_args(argv)

    with open(args.instance) as fh:
        n_vars, constraints = parse_opb(fh.read())

    clauses, _pool = to_cnf(n_vars, constraints)

    from pysat.solvers import Solver

    with Solver(name=args.solver, bootstrap_with=clauses) as solver:
        sat = solver.solve()
        stats = {}
        try:
            stats = solver.accum_stats() or {}
        except Exception:
            pass
        model = solver.get_model() if sat else None

    # PB competition output convention, as ExternalSolver._parse expects.
    print(f"c conflicts: {stats.get('conflicts', 0)}")
    print(f"c decisions: {stats.get('decisions', 0)}")
    if not sat:
        print("s UNSATISFIABLE")
        return 0

    print("s SATISFIABLE")
    assign = {abs(l): (l > 0) for l in model}
    # Only the original x1..xN matter; auxiliary variables stay internal.
    values = [
        f"x{v}" if assign.get(v, False) else f"-x{v}"
        for v in range(1, n_vars + 1)
    ]
    for i in range(0, len(values), 20):
        print("v " + " ".join(values[i : i + 20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
