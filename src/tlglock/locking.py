"""
Algorithms 1 and 2 of Sahruri & Margala, VLSI-SoC 2025.

Algorithm 1 (synthesis + key preparation) selects which gates to lock and
generates the key vector. Algorithm 2 (key integration) folds key inputs into
each selected gate's weighted sum.

  Eq. 2   out = 1  iff  sum_i w_i x_i >= T
  Eq. 3   out = 1  iff  sum_i w_i x_i + sum_j v_j k_j >= T
  Eq. 4   delta   = sum_j v_j (k'_j - k_j)      deviation under a wrong key


                    === THRESHOLD COMPENSATION ===

Eq. 3 as printed keeps T unchanged while adding sum_j v_j k_j to the left
side. Functional equivalence under the correct key k* therefore requires

    sum_i w_i x_i + sum_j v_j k*_j  >=  T'    <=>    sum_i w_i x_i  >=  T

which holds only if the locked gate uses a compensated threshold

    T'  =  T + sum_j v_j k*_j                                    (*)

The paper's worked example (Eq. 6) does NOT satisfy (*). It locks the gate of
Eq. 5 -- x1 + x2 + x3 >= 3 -- as

    1*x1 + 1*x2 + 1*x3 - 2*k1 + 3*k2  >=  2      with k* = [1, 1]

Under k* the key terms contribute -2 + 3 = +1, giving an effective threshold
of 2 - 1 = 1, i.e. the locked gate computes x1 + x2 + x3 >= 1 (an OR) where
the original computed x1 + x2 + x3 >= 3 (an AND). The two are not equivalent,
so the correct key does not restore the original function.

Applying (*) with the same key weights gives T' = 3 + 1 = 4, and

    1*x1 + 1*x2 + 1*x3 - 2*k1 + 3*k2  >=  4      with k* = [1, 1]

does reduce to x1 + x2 + x3 >= 3 as intended. The printed threshold of 2
appears to be a typo for 4.

This module implements (*). `lock()` verifies equivalence under the correct
key before returning, so the invariant is enforced rather than assumed. See
tests/test_eq6.py for the executable form of this analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .thfile import ThGate, ThNetwork

WeightMode = Literal["equal", "balanced", "high", "random"]

VALID_MODES: tuple[str, ...] = ("equal", "balanced", "high", "random")


@dataclass
class LockReport:
    """What Algorithm 2 did, for downstream metrics and for the paper tables."""

    locked_network: ThNetwork
    key_names: list[str]
    correct_key: dict[str, int]
    locked_gates: list[str]
    key_weights: dict[str, dict[str, int]] = field(default_factory=dict)
    threshold_shift: dict[str, int] = field(default_factory=dict)
    mode: str = "balanced"

    @property
    def num_keys(self) -> int:
        return len(self.key_names)

    @property
    def key_string(self) -> str:
        """Correct key as a bit string, in key-name order."""
        return "".join(str(self.correct_key[k]) for k in self.key_names)


# -- Algorithm 1: gate selection and key generation -------------------------


def select_lock_gates(
    net: ThNetwork,
    percent: float,
    rng: random.Random | None = None,
    strategy: str = "fanin",
) -> list[ThGate]:
    """
    Choose |G_TLG| * P/100 gates to lock (Algorithm 1, step 3).

    strategy:
      "fanin"  -- prefer high fan-in gates. A wide weighted sum has more room
                  to absorb key weights without distorting the decision, and
                  high-fanin gates sit on more output cones, so corruption
                  propagates further.
      "fanout" -- prefer gates driving many consumers.
      "random" -- uniform sample.
      "first"  -- deterministic prefix, for reproducible tests.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    if not net.gates:
        return []

    rng = rng or random.Random(0)
    count = int(round(len(net.gates) * percent / 100.0))
    if percent > 0:
        count = max(1, count)
    count = min(count, len(net.gates))
    if count == 0:
        return []

    if strategy == "random":
        return rng.sample(net.gates, count)
    if strategy == "first":
        return net.gates[:count]
    if strategy == "fanin":
        ranked = sorted(net.gates, key=lambda g: (-g.fanin, g.output))
        return ranked[:count]
    if strategy == "fanout":
        fo = net.fanout_count()
        ranked = sorted(net.gates, key=lambda g: (-fo.get(g.output, 0), g.output))
        return ranked[:count]
    raise ValueError(f"unknown strategy '{strategy}'")


def generate_key(num_keys: int, rng: random.Random | None = None) -> list[int]:
    """Generate a random correct key vector (Algorithm 1, step 4)."""
    rng = rng or random.Random(0)
    return [rng.randint(0, 1) for _ in range(num_keys)]


def assign_key_weights(
    gate: ThGate,
    num_keys: int,
    mode: WeightMode = "balanced",
    rng: random.Random | None = None,
) -> list[int]:
    """
    Key-input weights for one gate (flow step 5: "weights set proportionally
    to the sum of the input weights").

    The scale anchor is the gate's total input weight magnitude, W. Fig. 4 of
    the paper reports corruption peaking at moderate total key weight (~2-3 on
    a 4-input gate) with power and delay growing roughly linearly, so the
    modes span that range rather than maximising magnitude.

      equal    -- all key weights +u. Simple, but every key bit pushes the
                  sum the same direction, so wrong keys are only detected in
                  aggregate. Highest equivalence-class compression.
      balanced -- alternating signs. Wrong keys can push the sum either way,
                  which is what makes corruption approach 0.5 per output.
      high     -- larger magnitudes; more corruption, more area/power.
      random   -- uniform in [-u, u] \\ {0}, for the randomised-key-weight
                  sweep of Section IV-B.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")
    rng = rng or random.Random(0)

    W = sum(abs(w) for w in gate.weights)
    # Proportional scale, floored at 1 so tiny gates still get a real lock.
    unit = max(1, round(W / max(1, gate.fanin)))

    if mode == "equal":
        return [unit] * num_keys
    if mode == "balanced":
        return [unit if j % 2 == 0 else -unit for j in range(num_keys)]
    if mode == "high":
        hi = max(2, round(1.5 * unit))
        return [hi if j % 2 == 0 else -hi for j in range(num_keys)]
    # random
    span = max(1, unit)
    out = []
    for _ in range(num_keys):
        v = 0
        while v == 0:
            v = rng.randint(-span, span)
        out.append(v)
    return out


# -- Algorithm 2: key integration -------------------------------------------


def embed_keys(
    gate: ThGate,
    key_names: Sequence[str],
    key_weights: Sequence[int],
    correct_bits: Sequence[int],
) -> tuple[ThGate, int]:
    """
    Fold key inputs into one gate's weighted sum (Algorithm 2, steps 2-4).

    Returns the locked gate and the threshold shift applied per (*).
    The shift is what makes the correct key a no-op; without it the locked
    gate computes a different function than the original even under k*.
    """
    if not (len(key_names) == len(key_weights) == len(correct_bits)):
        raise ValueError("key_names, key_weights, correct_bits must be same length")
    clash = set(key_names) & set(gate.inputs)
    if clash:
        raise ValueError(
            f"gate '{gate.output}': key name(s) {sorted(clash)} collide with inputs"
        )
    if any(b not in (0, 1) for b in correct_bits):
        raise ValueError("correct_bits must be binary")

    shift = sum(v * b for v, b in zip(key_weights, correct_bits))

    locked = ThGate(
        inputs=list(gate.inputs) + list(key_names),
        output=gate.output,
        weights=list(gate.weights) + list(key_weights),
        threshold=gate.threshold + shift,
    )
    return locked, shift


def lock(
    net: ThNetwork,
    percent: float = 50.0,
    keys_per_gate: int = 2,
    mode: WeightMode = "balanced",
    seed: int = 0,
    strategy: str = "fanin",
    key_prefix: str = "K",
    verify: bool = True,
) -> LockReport:
    """
    Run Algorithms 1 and 2 end to end.

    Each locked gate receives `keys_per_gate` fresh key inputs, so the total
    key size is keys_per_gate * ceil(|G| * percent/100) -- matching the
    "#Keys" column of Table I, where key counts exceed the gate count of the
    small benchmarks.

    With verify=True (default) the result is checked for functional
    equivalence with the original network under the correct key, by
    exhaustive comparison when the input space is small enough and by
    random sampling otherwise.
    """
    rng = random.Random(seed)
    targets = select_lock_gates(net, percent, rng=rng, strategy=strategy)

    locked_net = net.copy()
    by_output = {g.output: i for i, g in enumerate(locked_net.gates)}

    key_names: list[str] = []
    correct_key: dict[str, int] = {}
    key_weights: dict[str, dict[str, int]] = {}
    shifts: dict[str, int] = {}
    locked_names: list[str] = []

    counter = 0
    for gate in targets:
        idx = by_output[gate.output]
        names = [f"{key_prefix}{counter + j + 1}" for j in range(keys_per_gate)]
        counter += keys_per_gate

        weights = assign_key_weights(gate, keys_per_gate, mode=mode, rng=rng)
        bits = generate_key(keys_per_gate, rng=rng)

        new_gate, shift = embed_keys(locked_net.gates[idx], names, weights, bits)
        locked_net.gates[idx] = new_gate

        key_names.extend(names)
        correct_key.update(dict(zip(names, bits)))
        key_weights[gate.output] = dict(zip(names, weights))
        shifts[gate.output] = shift
        locked_names.append(gate.output)

    locked_net.inputs.extend(key_names)
    locked_net.model = f"{net.model}_locked"

    report = LockReport(
        locked_network=locked_net,
        key_names=key_names,
        correct_key=correct_key,
        locked_gates=locked_names,
        key_weights=key_weights,
        threshold_shift=shifts,
        mode=mode,
    )

    if verify:
        from .metrics import is_equivalent_under_correct_key

        if not is_equivalent_under_correct_key(net, report):
            raise AssertionError(
                "locked network is not equivalent to the original under the "
                "correct key -- threshold compensation is broken"
            )
    return report
