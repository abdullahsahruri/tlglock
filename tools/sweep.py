#!/usr/bin/env python3
"""
Table I sweep: synth -> collapse -> lock -> attack, on the real benchmarks.

Produces the numbers the paper's attack columns need, measured rather than
modelled. Each row is written and flushed as soon as it is finished, so a run
that is killed partway still leaves usable results.

Both attacks are run on every circuit, including ones where the exact attack
times out -- that is the interesting case, because AppSAT stopping early where
the exact loop cannot finish is precisely the threat the resistance argument
has to rule out.

    tools/sweep.py --timeout 120 -o results/sweep.csv

Locking percentages follow Table I per circuit. Key counts do not and cannot:
Table I's "#Keys" is a percentage of *its* gate count, and this flow's mapper
and collapse pass produce a different one, so keys_per_gate is held fixed and
the resulting n_keys is reported for comparison rather than matched.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tlglock.abc import map_to_tlg, read_circuit          # noqa: E402
from tlglock.attack import (                              # noqa: E402
    ExternalSolver, Status, appsat_attack, oracle_from, sat_attack,
    verify_recovered_key,
)
from tlglock.collapse import collapse                     # noqa: E402
from tlglock.encoding import network_encoding             # noqa: E402
from tlglock.locking import lock                          # noqa: E402
from tlglock.metrics import corruption_rate               # noqa: E402

# circuit -> (suite, Table I locking percentage)
TABLE_I = {
    "c17":   ("ISCAS85", 50),
    "c1355": ("ISCAS85", 50),
    "c1908": ("ISCAS85", 50),
    "c2670": ("ISCAS85", 50),
    "c7552": ("ISCAS85", 50),
    "s386":  ("ISCAS89", 80),
    "s526":  ("ISCAS89", 80),
    "s713":  ("ISCAS89", 70),
    "s1494": ("ISCAS89", 20),
    "s5378": ("ISCAS89", 20),
    "i8":    ("MCNC",    80),
    "i10":   ("MCNC",    50),
    "des":   ("MCNC",     5),
    "b15":   ("ITC99",   50),
    "b17":   ("ITC99",   10),
}

FIELDS = [
    "circuit", "suite", "percent", "pis", "pos", "ands",
    "tlg_mapped", "tlg_collapsed", "depth", "map_s",
    "n_keys", "pb_constraints", "miter_constraints", "per_dip",
    "sat_status", "sat_iters", "sat_conflicts", "sat_decisions",
    "sat_s", "sat_equivalent",
    "app_status", "app_exact", "app_settled", "app_err_patterns",
    "app_err_bits", "app_iters", "app_rounds", "app_s",
    "corruption", "note",
]

# Exhaustive key verification is 2^(data inputs); above this, report n/a
# rather than pretend, and rely on the attack's own UNSAT termination.
VERIFY_LIMIT = 18


def find_circuit(directory: str, name: str) -> str | None:
    for ext in (".bench", "_comb.blif", ".blif"):
        path = os.path.join(directory, name + ext)
        if os.path.exists(path):
            return path
    return None


def run_one(name: str, path: str, args) -> dict:
    suite, percent = TABLE_I[name]
    row = {f: "" for f in FIELDS}
    row.update(circuit=name, suite=suite, percent=percent, note="")

    t0 = time.monotonic()
    aig = read_circuit(path, use_abc=False)
    net, mstats = map_to_tlg(aig, k=args.cut_size)
    mapped = len(net.gates)
    net, _ = collapse(net, max_support=args.max_support)
    row.update(
        pis=len(aig.pi_names), pos=len(aig.po_names), ands=aig.num_ands,
        tlg_mapped=mapped, tlg_collapsed=len(net.gates), depth=mstats.depth,
        map_s=f"{time.monotonic() - t0:.1f}",
    )

    report = lock(
        net, percent=percent, keys_per_gate=args.keys,
        mode=args.mode, seed=args.seed,
    )
    locked, keys = report.locked_network, report.key_names
    enc = network_encoding(locked, key_names=keys, max_fanin=0)
    row.update(
        n_keys=report.num_keys,
        pb_constraints=enc.pb_constraints,
        miter_constraints=enc.miter_pb_constraints,
        per_dip=enc.pb_constraints_per_dip,
    )

    solver = ExternalSolver(binary=args.solver)
    oracle = oracle_from(net)
    data_inputs = [n for n in locked.inputs if n not in set(keys)]

    # -- exact attack -------------------------------------------------------
    res = sat_attack(locked, keys, oracle, solver=solver, timeout=args.timeout)
    equiv = ""
    if res.key is not None and len(data_inputs) <= VERIFY_LIMIT:
        equiv = verify_recovered_key(net, locked, res.key)
    elif res.key is not None:
        equiv = "n/a"
    row.update(
        sat_status=res.status.value, sat_iters=res.iterations,
        sat_conflicts=res.conflicts, sat_decisions=res.decisions,
        sat_s=f"{res.seconds:.2f}", sat_equivalent=equiv,
    )

    # -- approximate attack -------------------------------------------------
    ares = appsat_attack(
        locked, keys, oracle, solver=solver, timeout=args.timeout,
        round_size=args.round_size, samples=args.samples,
        epsilon=args.epsilon, settle_rounds=args.settle_rounds,
        seed=args.seed,
    )
    # A timed-out AppSAT run never found a key, so AppSatResult's error fields
    # are still at their 1.0 default -- that is "not measured", not "100%
    # wrong". Emitting the default would put an unmeasured number in a results
    # column where every other row is measured, which is precisely the kind of
    # thing that later gets quoted as data. Leave it blank instead.
    measured = ares.key is not None
    row.update(
        app_status=ares.status.value, app_exact=ares.exact,
        app_settled=ares.settled,
        app_err_patterns=f"{ares.error_patterns:.4f}" if measured else "",
        app_err_bits=f"{ares.error_bits:.4f}" if measured else "",
        app_iters=ares.iterations, app_rounds=ares.rounds,
        app_s=f"{ares.seconds:.2f}",
    )

    row["corruption"] = f"{corruption_rate(net, report, input_limit=args.corruption_inputs, key_limit=args.corruption_keys):.4f}"
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="bench/circuits/tablei")
    ap.add_argument("-o", "--output", default="results/sweep.csv")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--solver", default="tools/opb_solve.py")
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cut-size", type=int, default=6)
    ap.add_argument("--max-support", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--round-size", type=int, default=5)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--settle-rounds", type=int, default=2)
    ap.add_argument("--corruption-inputs", type=int, default=64)
    ap.add_argument("--corruption-keys", type=int, default=32)
    ap.add_argument("--only", nargs="+", help="restrict to these circuits")
    ap.add_argument("--skip", nargs="+", default=["b15", "b17"],
                    help="circuits to skip (default: the two largest)")
    args = ap.parse_args(argv)

    names = args.only or [n for n in TABLE_I if n not in set(args.skip or [])]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        fh.flush()

        for name in names:
            path = find_circuit(args.dir, name)
            if path is None:
                print(f"{name}: not found in {args.dir}", file=sys.stderr, flush=True)
                continue
            print(f"--- {name} ({os.path.basename(path)}) ...",
                  file=sys.stderr, flush=True)
            start = time.monotonic()
            try:
                row = run_one(name, path, args)
            except Exception as e:                      # keep the sweep going
                row = {f: "" for f in FIELDS}
                row.update(
                    circuit=name, suite=TABLE_I[name][0],
                    percent=TABLE_I[name][1],
                    note=f"{type(e).__name__}: {str(e)[:120]}",
                )
            writer.writerow(row)
            fh.flush()
            print(
                f"    {name}: sat={row['sat_status']} app={row['app_status']} "
                f"keys={row['n_keys']} [{time.monotonic() - start:.1f}s] "
                f"{row['note']}",
                file=sys.stderr, flush=True,
            )

    print(f"wrote {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
