# tlglock

Key-driven logic locking in threshold logic gates — a reimplementation of the
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

## Use

```python
from tlglock import parse_th, lock, corruption_rate, outputs_of

net = parse_th("""
.model eq5
.input X1 X2 X3
.output Z
.threshold X1 X2 X3 Z 1 1 1 3
.end
""")

report = lock(net, percent=100, keys_per_gate=2, mode="balanced", seed=0)

print(report.key_string)                    # the correct key
print(corruption_rate(net, report))         # mean output corruption

# The correct key reproduces the original function.
x = {"X1": 1, "X2": 1, "X3": 1}
assert outputs_of(net, x) == outputs_of(report.locked_network, {**x, **report.correct_key})

print(report.locked_network.to_text())
```

Weight modes are `equal`, `balanced`, `high` and `random`, matching the
Section IV-B sweep. `balanced` is the default: alternating signs let a wrong
key push the weighted sum either way, which is what drives corruption up.

Export a SAT-attack instance:

```python
from tlglock import build_distinguishing_miter

enc = build_distinguishing_miter(report.locked_network, report.key_names)
enc.write("miter.opb")     # MiniSAT+ / PB competition format
```

## Threshold compensation

Adding key terms to the weighted sum changes the function unless the
threshold moves with them. Equivalence under the correct key `k*` holds iff

```
T' = T + sum_j v_j k*_j
```

`embed_keys()` applies this, and `lock()` verifies equivalence before
returning rather than assuming it. The paper's worked example (Eq. 6) does
not satisfy the identity — see `CLAUDE.md` and `tests/test_eq6.py`.

## Layout

```
src/tlglock/
  thfile.py    .th parse/write, validation, topological order
  sim.py       combinational simulation
  locking.py   Algorithms 1 and 2
  metrics.py   corruption rate, equivalence, key-space collapse
  opb.py       pseudo-Boolean encoding and miter construction
tests/
  golden/table1.csv    Table I regression data
```

Not yet implemented: the ABC synthesis driver, TLG collapse/merge, the SAT
attack outer loop, and Cadence cell characterisation. See `CLAUDE.md`.
