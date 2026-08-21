# TLGLock — handoff

Rebuild of the flow in Sahruri & Margala, *"TLGLock: A New Approach in Logic
Locking Using Key-Driven Charge Recycling in Threshold Logic Gates,"*
IFIP/IEEE VLSI-SoC 2025. The paper is the specification; there is no surviving
original source.

**This is the second rebuild.** The first (August 2026) was lost with a WSL
distro deletion before it reached version control. Push to a remote before
doing anything else.

## State

Five modules under `src/tlglock/`, 260 tests passing, one skipped (the
end-to-end Table I regression, blocked on the items below).

| Module | Purpose | Status |
|---|---|---|
| `thfile.py` | `.th` parse/write, network validation, topological order | complete |
| `sim.py` | combinational simulation, truth tables | complete |
| `locking.py` | Algorithms 1 and 2, key embedding | complete |
| `metrics.py` | corruption rate, equivalence, key-space collapse | complete |
| `opb.py` | pseudo-Boolean encoding, distinguishing miter | complete |

```bash
pip install -e ".[dev]"
pytest -q                 # ~2 s
pytest -m golden          # the blocked regression
```

## Three findings that are not in the paper

Each has an executable test. Do not "fix" these by editing the data to match
the prose — the tests assert the analysis and will fail if you do.

### 1. Eq. (6) does not preserve functionality (`tests/test_eq6.py`)

Eq. (5) is `x1 + x2 + x3 >= 3`. Eq. (6) locks it as
`x1 + x2 + x3 - 2*k1 + 3*k2 >= 2` with correct key `K = [1,1]`, and the text
says the correct key "neutralizes the effect of the added inputs."

It does not. Under `K = [1,1]` the key terms contribute `-2 + 3 = +1`, giving
an effective threshold of `2 - 1 = 1`. The locked gate computes an OR where
the original computed an AND.

The governing identity is

```
T' = T + sum_j v_j k*_j
```

Equivalence holds **iff** `T' - sum_j v_j k*_j == T`. The printed numbers
violate it. Two minimal repairs exist: keep the weights and set `T = 4`, or
keep `T = 2` and make the weights sum to `-1` (e.g. `v = (-2, 1)`). The text
alone cannot say which was intended. `embed_keys()` implements the identity,
and `lock()` refuses to return a network that violates it.

**Action for the journal version:** state the compensation identity
explicitly. It is currently implicit and the worked example contradicts it.

### 2. Table I does not support the b17 area claim (`tests/test_golden.py`)

The conclusion says *"on b17 it reduces area by 26%."* Table I gives b17 as
7200 → 6100 µm², which is **15.3%**. 26% is what c1355, c1908, c2670, s1494,
s526 and s5378 reduce by — the sentence appears to have picked up the wrong
row. b17's actual standout is delay, at 54.5%.

The same sentence is in `ch5_tlglock.tex` in the dissertation draft.

### 3. Corruption saturates rather than peaking (`tests/test_metrics.py`)

Section IV-B says corruption "peaks at intermediate total key weight (≈ 2–3)"
and Fig. 4 is drawn with a peak. Sweeping key weight under threshold
compensation gives monotone-increasing-then-flat instead: on a 4-input gate
with T=2, corruption climbs to 0.6875 by |v|=3 and stays there. 0.6875 is
exactly the distance from that gate's function to a constant output, which is
the natural ceiling — once the key alone decides the gate, more weight
changes nothing.

Lower confidence than the other two, since it may depend on the measurement
setup. **Check against the raw Fig. 4 data before repeating the "peaks"
wording.** The actionable claim survives either way: |v| = 2–3 already
reaches ~95% of the plateau, so moderate weights are the right choice.

## Remaining work, in dependency order

1. **`abc.py` — ABC synthesis driver.** Wrap `abc` for BLIF/BENCH → AIG → TLG
   (flow steps 1–3). Needs `&get`, `&if -g`, threshold cut computation per
   Neutzling et al. [24]. Everything downstream is blocked on real netlists;
   right now the only inputs are hand-written and randomly generated.

2. **`collapse.py` — TLG collapse/merge.** Flow step 4, linear combination of
   adjacent TLGs per Lee et al. [23]. Affects gate count and therefore key
   count, so Table I's `#Keys` column cannot be reproduced without it.

3. **`attack.py` — SAT attack loop.** The oracle-guided outer loop.
   `opb.build_distinguishing_miter()` and `add_oracle_constraint()` already
   build one iteration; what is missing is the loop, MiniSAT+ invocation, OPB
   result parsing, and the 1-hour timeout. This is what actually produces the
   SAT/UNSAT/TIMEOUT column.

4. **Cadence cell characterisation.** LCTL/CRTL in GPDK045 for the area,
   power and delay columns. Not scriptable from here.

## Design notes

- **Keys are not marked.** A key input is an ordinary input whose weight came
  from the locking pass. That is the scheme's whole claim to having no
  structural footprint, so nothing in `ThGate` distinguishes them; the key
  names live in `LockReport`.
- **PB, not CNF.** A threshold gate is natively a linear constraint. CNF
  requires enumerating minimal true points, which is exponential — that is
  the "not expressible in polynomial-size CNF" claim in the intro. Encoding
  in OPB gives the attacker the *better* representation, so a timeout under
  PB is a stronger result than a timeout under blown-up CNF.
- **Reification bounds.** `encode_gate()` derives slack from the true bounds
  (`Smin = sum of negative weights`, `Smax = sum of positive`), not from
  `sum|w|`. With mixed-sign weights — which `balanced` mode produces on every
  locked gate — those differ, and the symmetric bound silently makes one
  implication vacuous. `tests/test_opb.py` brute-forces every model against
  the simulator specifically to catch that class of error. It caught a real
  sign bug in the first rebuild.
- **Everything is seeded.** `lock(seed=...)` and all metrics take a seed and
  are reproducible.
- **Metrics sample past `EXHAUSTIVE_LIMIT`.** Below it they enumerate. So a
  `False` from `is_equivalent_under_correct_key` is always conclusive; a
  `True` is conclusive only in the exhaustive regime.

## Conventions

Python ≥ 3.10, no runtime dependencies, `pytest` for dev. Tests carry the
reasoning in their docstrings — when a test encodes a claim from the paper,
the docstring says which claim and where it appears.
