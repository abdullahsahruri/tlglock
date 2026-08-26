# tlglock

Key-driven logic locking in threshold logic gates -- a reimplementation of the
flow in Sahruri & Margala, *"TLGLock: A New Approach in Logic Locking Using
Key-Driven Charge Recycling in Threshold Logic Gates,"* IFIP/IEEE VLSI-SoC
2025.

A threshold gate computes `1[sum_i w_i x_i >= T]`. TLGLock embeds key bits as
additional weighted inputs to that sum, so the lock has no distinct structure
to find: a key input looks exactly like a data input.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

No runtime dependencies. Python 3.10+.

## The flow

```
  .bench / .blif
       |  read_circuit          step 1   (ABC if available, else built-in)
       v
      AIG
       |  map_to_tlg            steps 2-3
       v
  TLG network
       |  collapse              step 4
       v
  collapsed network
       |  lock                  steps 5-7
       v
  locked network
       |  sat_attack            evaluation
       v
  recovered key / timeout
```

## Command line

```bash
python -m tlglock synth  c17.bench -o c17.th
python -m tlglock lock   c17.th --percent 50 --keys 2 -o c17_locked.th
python -m tlglock attack c17_locked.th --original c17.th --timeout 3600
python -m tlglock run    c17.bench --percent 50 --keys 2
```

`run` does the whole pipeline and prints a CSV row for sweeping benchmarks.

## Library

```python
from tlglock import (
    read_bench, map_to_tlg, collapse, lock,
    sat_attack, oracle_from, verify_recovered_key,
    corruption_rate, outputs_of,
)

aig      = read_bench(open("c17.bench").read())
net, _   = map_to_tlg(aig, k=6)
net, _   = collapse(net)
report   = lock(net, percent=50, keys_per_gate=2, mode="balanced", seed=0)

print(report.key_string)
print(corruption_rate(net, report))

res = sat_attack(report.locked_network, report.key_names, oracle_from(net))
print(res.status, res.iterations)
print(verify_recovered_key(net, report.locked_network, res.key))
```

Weight modes are `equal`, `balanced`, `high` and `random`, matching the
Section IV-B sweep. `balanced` is the default: alternating signs let a wrong
key push the weighted sum either way, which is what drives corruption up.

## Threshold identification

The mapper accepts a cut when the cut's local function is linearly separable,
not when the cut is small enough. That test is an exact LP over rationals:

```python
from tlglock import identify, truth_bits

table = [int(sum(b) >= 2) for b in truth_bits(3)]
print(identify(table, 3))     # weights (1, 1, 1), threshold 2

table = [b[0] ^ b[1] for b in truth_bits(2)]
print(identify(table, 2))     # None -- XOR is not a threshold function
```

The classifier is validated by exhaustive enumeration against OEIS A000609:
2, 4, 14, 104, 1882 threshold functions of 0..4 variables.

## Threshold compensation

Adding key terms to the weighted sum changes the function unless the threshold
moves with them. Equivalence under the correct key `k*` holds iff

```
T' = T + sum_j v_j k*_j
```

`embed_keys()` applies this, and `lock()` verifies equivalence before
returning rather than assuming it. The paper's worked example (Eq. 6) does not
satisfy the identity -- see `CLAUDE.md` and `tests/test_eq6.py`.

## Solvers

`PbSolver` is a complete DPLL over pseudo-Boolean constraints with no external
dependency, enough for small benchmarks. For real circuits use an external PB
solver:

```python
from tlglock import ExternalSolver, sat_attack

sat_attack(locked, keys, oracle, solver=ExternalSolver(binary="minisat+"))
```

## Layout

```
src/tlglock/
  thfile.py     .th parse/write, validation, topological order
  sim.py        combinational simulation
  lp.py         exact rational simplex
  separable.py  threshold identification
  abc.py        BENCH/BLIF frontend, cut enumeration, TLG mapping
  collapse.py   TLG collapse/merge
  locking.py    Algorithms 1 and 2
  metrics.py    corruption rate, equivalence, key-space collapse
  opb.py        pseudo-Boolean encoding and miter construction
  attack.py     oracle-guided SAT attack
  cli.py        command-line driver
tests/
  golden/table1.csv    Table I regression data
```

Not implemented: Cadence cell characterisation for the area, power and delay
columns. See `CLAUDE.md`.
