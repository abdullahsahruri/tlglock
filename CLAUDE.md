# TLGLock — handoff

Rebuild of the flow in Sahruri & Margala, *"TLGLock: A New Approach in Logic
Locking Using Key-Driven Charge Recycling in Threshold Logic Gates,"*
IFIP/IEEE VLSI-SoC 2025. The paper is the specification; there is no surviving
original source.

**This is the second rebuild.** The first (August 2026) was lost with a WSL
distro deletion before it reached version control. Push to a remote before
doing anything else.

## State

Eleven modules under `src/tlglock/`, 471 tests passing, one skipped.

| Module | Purpose | Status |
|---|---|---|
| `thfile.py` | `.th` parse/write, validation, topological order | complete |
| `sim.py` | combinational simulation, truth tables | complete |
| `lp.py` | exact rational simplex (LP feasibility) | complete |
| `separable.py` | threshold identification | complete |
| `abc.py` | BENCH/BLIF -> AIG -> TLG mapping (steps 1-3) | complete |
| `collapse.py` | TLG collapse/merge (step 4) | complete |
| `locking.py` | Algorithms 1 and 2, key embedding (steps 5-7) | complete |
| `metrics.py` | corruption rate, equivalence, key-space collapse | complete |
| `opb.py` | pseudo-Boolean encoding, distinguishing miter | complete |
| `attack.py` | oracle-guided SAT attack loop | complete |
| `cli.py` | command-line driver | complete |
| Cadence characterisation | LCTL/CRTL in GPDK045 | **missing** |

```bash
pip install -e ".[dev]"
pytest -q                    # ~10 s
pytest -m slow               # exhaustive 4-variable enumeration
python -m tlglock run bench/c17.bench --percent 50 --keys 2
```

## Four findings that are not in the paper

Each has an executable test. Do not "fix" these by editing data to match the
prose -- the tests assert the analysis and will fail if you do.

### 1. Eq. (6) does not preserve functionality (`tests/test_eq6.py`)

Eq. (5) is `x1 + x2 + x3 >= 3`. Eq. (6) locks it as
`x1 + x2 + x3 - 2*k1 + 3*k2 >= 2` with correct key `K = [1,1]`, and the text
says the correct key "neutralizes the effect of the added inputs."

It does not. Under `K = [1,1]` the key terms contribute `-2 + 3 = +1`, giving
an effective threshold of `2 - 1 = 1`. The locked gate computes an OR where
the original computed an AND.

The governing identity is `T' = T + sum_j v_j k*_j`, and equivalence holds iff
it is satisfied. Two minimal repairs exist: keep the weights and set `T = 4`,
or keep `T = 2` and make the weights sum to `-1`. The text alone cannot say
which was intended. `embed_keys()` implements the identity and `lock()`
refuses to return a network that violates it.

**Action:** state the compensation identity explicitly in the journal version.
It is currently implicit and the worked example contradicts it.

### 2. Table I does not support the b17 area claim (`tests/test_golden.py`)

The conclusion says *"on b17 it reduces area by 26%."* Table I gives b17 as
7200 -> 6100 um^2, which is **15.3%**. 26% is what c1355, c1908, c2670, s1494,
s526 and s5378 reduce by -- the sentence appears to have picked up the wrong
row. b17's actual standout is delay, at 54.5%.

The same sentence is in `ch5_tlglock.tex` in the dissertation draft.

### 3. Corruption saturates rather than peaking (`tests/test_metrics.py`)

Section IV-B says corruption "peaks at intermediate total key weight (~2-3)"
and Fig. 4 is drawn with a peak. Sweeping key weight under threshold
compensation gives monotone-increasing-then-flat: on a 4-input gate with T=2,
corruption climbs to 0.6875 by |v|=3 and stays there. 0.6875 is exactly the
distance from that gate's function to a constant output, which is the natural
ceiling -- once the key alone decides the gate, more weight changes nothing.

Lower confidence than the others, since it may depend on the measurement
setup. **Check against the raw Fig. 4 data before repeating the "peaks"
wording.** The actionable claim survives either way: |v| = 2-3 already reaches
~95% of the plateau.

### 4. Table I's "Result" column is ambiguous

The column takes values SAT / UNSAT / Timeout, but its meaning is not stated.
Two readings are possible and they disagree:

- *Attack outcome.* SAT = the attack recovered the key. This fits c17 (tiny,
  broken instantly) and the timeouts on large designs.
- *A single solver invocation's verdict.* This fits c2670 and s5378 being
  UNSAT with **zero** conflicts and zero decisions -- that is preprocessing
  resolving the instance, not an attack loop terminating.

The second reading is hard to square with a completed attack: an attack loop
terminates *because* the miter goes UNSAT, and that final solve typically
follows many non-trivial ones. Reporting zero conflicts for it suggests the
table records one solve rather than a loop.

This matters for reproduction, so the CLI does not guess: `run` reports
`BROKEN` / `Timeout` / `NO_KEY` in its own vocabulary. Resolve the intended
meaning before comparing counts against Table I, and define the column
explicitly in the journal version.

## Remaining work

**Cadence characterisation.** LCTL/CRTL in GPDK045 for the area, power and
delay columns. Not scriptable from here.

**Real benchmarks.** `bench/fetch.sh` points at the sources; only a
hand-written c17 is checked in. Getting ISCAS/ITC/MCNC circuits in is what
turns the flow from "correct" into "measured".

**Attack scale.** `PbSolver` is a complete but plain DPLL -- no clause
learning, no restarts, no watched literals. Fine to roughly c17/s27 scale.
Table I's numbers need `ExternalSolver(binary="minisat+")` or another PB
solver; the plumbing and output parsing are done and tested.

**Cut ranking.** The mapper uses depth-then-size over separable cuts. This is
deliberately the simple version. Designing a cost model for the case where
feasibility is an LP rather than a size check is an open research problem, and
a plausible paper -- see "TLG-aware cut ranking in ABC".

## Design notes

- **Keys are not marked.** A key input is an ordinary input whose weight came
  from the locking pass. That is the scheme's whole claim to having no
  structural footprint, so nothing in `ThGate` distinguishes them; key names
  live in `LockReport`.
- **Exact arithmetic in the LP.** A float solver reporting "feasible" with a
  1e-12 margin hands back weights that do not implement the function, and the
  error surfaces much later as a mismatched netlist. `lp.py` uses `Fraction`
  throughout. Instances are tiny after the unateness reduction, so the cost is
  irrelevant.
- **Separability is validated against OEIS A000609.** The number of threshold
  functions of n variables is 2, 4, 14, 104, 1882 for n = 0..4. The test suite
  enumerates every truth table and matches those counts exactly, which
  validates the classifier in both directions -- a classifier that accepted
  everything would pass a positive-only test.
- **PB, not CNF.** A threshold gate is natively a linear constraint. CNF
  requires enumerating minimal true points, which is exponential -- the "not
  expressible in polynomial-size CNF" claim in the intro. Encoding in OPB
  gives the attacker the *better* representation, so a timeout under PB is a
  stronger result than a timeout under blown-up CNF.
- **Reification bounds.** `encode_gate()` derives slack from the true bounds
  (`Smin` = sum of negative weights, `Smax` = sum of positive), not from
  `sum|w|`. With mixed-sign weights -- which `balanced` mode produces on every
  locked gate -- those differ, and the symmetric bound silently makes one
  implication vacuous. `tests/test_opb.py` brute-forces every model against
  the simulator to catch that class of error. It caught a real sign bug in the
  first rebuild.
- **Attack success is functional, not bitwise.** Weight degeneracy means many
  key vectors realise the same function. `verify_recovered_key()` checks
  equivalence; comparing bit strings would understate the attack.
- **Collapse verifies before accepting.** `compose()` checks the merged gate
  against the composed truth table rather than trusting the identification, so
  a merge can never silently change the function.

## Conventions

Python >= 3.10, no runtime dependencies, `pytest` for dev. Tests carry the
reasoning in their docstrings -- when a test encodes a claim from the paper,
the docstring says which claim and where it appears.

Several tests exist because an earlier assumption of *mine* was wrong, not the
code's: the Eq. (6) uniqueness claim, the "equal mode compresses more"
ordering, and the XOR collapse stopping point. Each is now written to assert
what was actually measured. If a test here looks over-specified, that is
usually why.
