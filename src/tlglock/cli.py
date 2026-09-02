"""
Command-line driver for the TLGLock flow.

    python -m tlglock synth   c17.bench -o c17.th
    python -m tlglock lock    c17.th --percent 50 --keys 2 -o c17_locked.th
    python -m tlglock attack  c17_locked.th --original c17.th --timeout 3600
    python -m tlglock run     c17.bench --percent 50 --keys 2

`run` is the whole pipeline in one shot -- synthesise, collapse, lock, attack --
and prints a row in the column order of Table I, so a sweep over benchmarks can
be diffed directly against the published table.
"""

from __future__ import annotations

import argparse
import sys
import time

from .abc import abc_available, read_circuit, map_to_tlg, synthesize
from .attack import (
    DEFAULT_TIMEOUT,
    ExternalSolver,
    PbSolver,
    Status,
    oracle_from,
    sat_attack,
    verify_recovered_key,
)
from .characterize import (
    key_size_sweep, ngspice_available, table_i_rows, write_csv,
)
from .collapse import collapse
from .encoding import format_report
from .locking import lock
from .spice import TECHNOLOGIES
from .metrics import corruption_rate
from .thfile import read_th, write_th


def _solver(name: str):
    if name == "builtin":
        return PbSolver()
    return ExternalSolver(binary=name)


def cmd_synth(args) -> int:
    net, stats = synthesize(
        args.input, k=args.cut_size, use_abc=None if args.abc == "auto" else args.abc == "yes"
    )
    if not args.no_collapse:
        net, cstats = collapse(net, max_support=args.max_support)
        print(
            f"collapse: {cstats.gates_before} -> {cstats.gates_after} gates "
            f"({cstats.merges} merges), depth {cstats.depth_before} -> {cstats.depth_after}",
            file=sys.stderr,
        )
    print(
        f"map: {stats.gates} gates, depth {stats.depth}, max fanin {stats.max_fanin}, "
        f"max |w| {stats.max_weight}, {stats.separable_fraction:.1%} of cuts separable",
        file=sys.stderr,
    )
    _emit(net, args.output)
    return 0


def cmd_lock(args) -> int:
    net = read_th(args.input)
    report = lock(
        net,
        percent=args.percent,
        keys_per_gate=args.keys,
        mode=args.mode,
        seed=args.seed,
        strategy=args.strategy,
    )
    print(
        f"locked {len(report.locked_gates)} gates with {report.num_keys} keys "
        f"(mode={args.mode})",
        file=sys.stderr,
    )
    print(f"key: {report.key_string}", file=sys.stderr)
    if args.corruption:
        print(
            f"corruption: {corruption_rate(net, report):.4f}", file=sys.stderr
        )
    _emit(report.locked_network, args.output)
    if args.key_file:
        with open(args.key_file, "w") as fh:
            fh.write(report.key_string + "\n")
    return 0


def cmd_attack(args) -> int:
    locked = read_th(args.input)
    original = read_th(args.original)
    key_names = [n for n in locked.inputs if n not in original.inputs]
    if not key_names:
        print("no key inputs found -- is this a locked netlist?", file=sys.stderr)
        return 2

    res = sat_attack(
        locked,
        key_names,
        oracle_from(original),
        solver=_solver(args.solver),
        timeout=args.timeout,
    )
    ok = (
        verify_recovered_key(original, locked, res.key)
        if res.key is not None
        else None
    )
    print(f"result:     {res.status.value}")
    print(f"iterations: {res.iterations}")
    print(f"conflicts:  {res.conflicts}")
    print(f"decisions:  {res.decisions}")
    print(f"cpu time:   {res.seconds:.2f}s")
    if res.key is not None:
        print(f"key:        {''.join(str(res.key[k]) for k in key_names)}")
        print(f"equivalent: {ok}")
    return 0


def cmd_run(args) -> int:
    t0 = time.monotonic()
    net, mstats = synthesize(
        args.input, k=args.cut_size, use_abc=None if args.abc == "auto" else args.abc == "yes"
    )
    if not args.no_collapse:
        net, _ = collapse(net, max_support=args.max_support)

    report = lock(
        net,
        percent=args.percent,
        keys_per_gate=args.keys,
        mode=args.mode,
        seed=args.seed,
    )
    res = sat_attack(
        report.locked_network,
        report.key_names,
        oracle_from(net),
        solver=_solver(args.solver),
        timeout=args.timeout,
    )

    name = args.input.rsplit("/", 1)[-1].split(".")[0]

    # Table I's "Result" column is SAT / UNSAT / Timeout. Its intended meaning
    # is ambiguous -- see CLAUDE.md finding 4 -- so this reports the attack
    # outcome in its own terms and leaves the mapping to the reader.
    #
    #   BROKEN    the loop terminated and recovered a working key
    #   TIMEOUT   the budget expired first
    #   NO_KEY    the loop terminated but no key satisfies the observations,
    #             which would indicate a bug rather than a security property
    if res.status is Status.TIMEOUT:
        outcome, conflicts, decisions, cpu = "Timeout", "---", "---", "---"
    elif res.key is not None:
        outcome = "BROKEN"
        conflicts, decisions, cpu = res.conflicts, res.decisions, f"{res.seconds:.2f}"
    else:
        outcome = "NO_KEY"
        conflicts, decisions, cpu = res.conflicts, res.decisions, f"{res.seconds:.2f}"

    print("circuit,n_keys,percent,conflicts,decisions,cpu_time_s,outcome,dips,gates,depth")
    print(
        f"{name},{report.num_keys},{args.percent:g},{conflicts},{decisions},"
        f"{cpu},{outcome},{res.iterations},{len(net.gates)},{mstats.depth}"
    )
    if res.key is not None and not verify_recovered_key(
        net, report.locked_network, res.key
    ):
        print(
            "# WARNING: recovered key does not reproduce the original function",
            file=sys.stderr,
        )
    print(f"# wall clock {time.monotonic() - t0:.2f}s", file=sys.stderr)
    return 0


def cmd_encoding(args) -> int:
    net = read_th(args.input)
    keys = list(args.keys or [])
    if args.original:
        original = read_th(args.original)
        keys = [n for n in net.inputs if n not in original.inputs]
    print(format_report(net, key_names=keys, max_fanin=args.max_fanin))
    return 0


def cmd_characterize(args) -> int:
    if not ngspice_available(args.binary):
        print(
            f"'{args.binary}' not found on PATH.\n"
            "  Install:  apt install ngspice   (or brew install ngspice)\n"
            "  Models:   https://ptm.asu.edu/  -> 45nm_HP.pm\n"
            "  Then:     --models path/to/45nm_HP.pm",
            file=sys.stderr,
        )
        return 2

    tech = TECHNOLOGIES[args.tech]
    if args.models:
        from dataclasses import replace

        tech = replace(tech, model_include=args.models)

    results = key_size_sweep(
        cells=tuple(args.cells),
        key_sizes=range(args.min_keys, args.max_keys + 1, args.step),
        tech=tech,
        comparator_scale=args.comparator_scale,
        binary=args.binary,
    )
    if args.output:
        write_csv(results, args.output)
        print(f"wrote {len(results)} rows to {args.output}", file=sys.stderr)

    rows = table_i_rows(results)
    if not rows:
        print("no LCTL/CRTL pairs to compare", file=sys.stderr)
        return 1
    print(",".join(rows[0]))
    for r in rows:
        print(",".join(str(v) for v in r.values()))
    return 0


def _emit(net, output: str | None) -> None:
    if output:
        write_th(net, output)
    else:
        sys.stdout.write(net.to_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tlglock", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_synth_opts(sp):
        sp.add_argument("--cut-size", "-k", type=int, default=6)
        sp.add_argument("--max-support", type=int, default=8)
        sp.add_argument("--no-collapse", action="store_true")
        sp.add_argument("--abc", choices=["auto", "yes", "no"], default="auto")

    def add_lock_opts(sp):
        sp.add_argument("--percent", "-p", type=float, default=50.0)
        sp.add_argument("--keys", type=int, default=2, help="key bits per gate")
        sp.add_argument(
            "--mode", choices=["equal", "balanced", "high", "random"], default="balanced"
        )
        sp.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("synth", help="netlist -> TLG network")
    s.add_argument("input")
    s.add_argument("--output", "-o")
    add_synth_opts(s)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("lock", help="embed keys into a TLG network")
    s.add_argument("input")
    s.add_argument("--output", "-o")
    s.add_argument("--key-file")
    s.add_argument("--corruption", action="store_true")
    s.add_argument(
        "--strategy", choices=["fanin", "fanout", "random", "first"], default="fanin"
    )
    add_lock_opts(s)
    s.set_defaults(func=cmd_lock)

    s = sub.add_parser("attack", help="oracle-guided SAT attack")
    s.add_argument("input")
    s.add_argument("--original", required=True, help="unlocked netlist, as oracle")
    s.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    s.add_argument("--solver", default="builtin")
    s.set_defaults(func=cmd_attack)

    s = sub.add_parser("run", help="full pipeline, Table I row on stdout")
    s.add_argument("input")
    s.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    s.add_argument("--solver", default="builtin")
    add_synth_opts(s)
    add_lock_opts(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "encoding", help="how many equations the locking creates, per encoding"
    )
    s.add_argument("input")
    s.add_argument("--original", help="unlocked netlist, to identify key inputs")
    s.add_argument(
        "--keys", nargs="+", help="key input names, if --original is not given"
    )
    s.add_argument("--max-fanin", type=int, default=16)
    s.set_defaults(func=cmd_encoding)

    s = sub.add_parser(
        "characterize", help="ngspice sweep for the area/power/delay columns"
    )
    s.add_argument("--tech", choices=sorted(TECHNOLOGIES), default="PTM45")
    s.add_argument("--models", help="path to the PTM model card")
    s.add_argument("--binary", default="ngspice")
    s.add_argument("--cells", nargs="+", default=["LCTL", "CRTL"])
    s.add_argument("--min-keys", type=int, default=2)
    s.add_argument("--max-keys", type=int, default=16)
    s.add_argument("--step", type=int, default=2)
    s.add_argument("--comparator-scale", type=float, default=1.0)
    s.add_argument("--output", "-o", help="CSV of the raw per-cell results")
    s.set_defaults(func=cmd_characterize)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "abc", "auto") == "yes" and not abc_available():
        print("abc requested but not found on PATH", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
