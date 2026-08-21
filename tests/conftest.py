import random

import pytest

from tlglock.thfile import ThGate, ThNetwork, parse_th

# The gate of Eq. 5: a 3-input AND written as a threshold function.
EQ5_TEXT = """\
.model eq5
.input X1 X2 X3
.output Z
.threshold X1 X2 X3 Z 1 1 1 3
.end
"""

# The 7-input gate printed in Fig. 3 of the paper.
FIG3_TEXT = """\
.model c3_after_clp_locked
.input K1 X1 X2 X3 Y2 Y3 K2
.output Z
.threshold K1 X1 X2 X3 Y2 Y3 K2 Z 3 2 1 3 2 1 -2 7
.end
"""

# A small multi-level network: two hidden gates feeding an output gate.
MULTI_TEXT = """\
.model multi
.input a b c d
.output F G
.threshold a b n1 1 1 2
.threshold c d n2 1 1 1
.threshold n1 n2 F 1 1 2
.threshold a n2 G 2 -1 1
.end
"""


@pytest.fixture
def eq5():
    return parse_th(EQ5_TEXT)


@pytest.fixture
def fig3():
    return parse_th(FIG3_TEXT)


@pytest.fixture
def multi():
    return parse_th(MULTI_TEXT)


@pytest.fixture
def rng():
    return random.Random(12345)


def random_network(seed: int, n_in: int = 5, n_gates: int = 6) -> ThNetwork:
    """Generate a random acyclic threshold network for property testing."""
    rnd = random.Random(seed)
    inputs = [f"i{k}" for k in range(n_in)]
    net = ThNetwork(model=f"rand{seed}", inputs=list(inputs))
    pool = list(inputs)

    for g in range(n_gates):
        fanin = rnd.randint(1, min(4, len(pool)))
        src = rnd.sample(pool, fanin)
        weights = [rnd.choice([-3, -2, -1, 1, 2, 3]) for _ in range(fanin)]
        lo = sum(w for w in weights if w < 0)
        hi = sum(w for w in weights if w > 0)
        thr = rnd.randint(lo, hi + 1)
        name = f"g{g}"
        net.gates.append(
            ThGate(inputs=src, output=name, weights=weights, threshold=thr)
        )
        pool.append(name)

    # Last two gates are the primary outputs.
    net.outputs = [g.output for g in net.gates[-2:]]
    net.validate()
    return net
