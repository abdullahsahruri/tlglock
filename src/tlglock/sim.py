"""
Combinational simulation of threshold networks.

Evaluation is levelised via ThNetwork.topological_order(), so a network with
a loop raises rather than silently returning a fixed point.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Mapping, Sequence

from .thfile import ThNetwork


def simulate(net: ThNetwork, assignment: Mapping[str, int]) -> dict[str, int]:
    """
    Evaluate every signal in `net` under the given primary-input assignment.

    Returns a dict binding all primary inputs and all gate outputs.
    Missing primary inputs default to 0; unknown names in `assignment`
    are rejected so typos in key names surface immediately.
    """
    unknown = set(assignment) - net.signals
    if unknown:
        raise KeyError(f"assignment names unknown signals: {sorted(unknown)}")

    values: dict[str, int] = {name: 0 for name in net.inputs}
    for name, val in assignment.items():
        if val not in (0, 1):
            raise ValueError(f"signal '{name}': value {val} is not binary")
        values[name] = val

    for gate in net.topological_order():
        values[gate.output] = gate.eval(values)
    return values


def outputs_of(net: ThNetwork, assignment: Mapping[str, int]) -> tuple[int, ...]:
    """Primary-output response, in declared order."""
    values = simulate(net, assignment)
    return tuple(values[o] for o in net.outputs)


def enumerate_assignments(names: Sequence[str]) -> Iterator[dict[str, int]]:
    """All 2^len(names) binary assignments, in counting order."""
    for bits in product((0, 1), repeat=len(names)):
        yield dict(zip(names, bits))


def truth_table(
    net: ThNetwork, over: Sequence[str] | None = None, fixed: Mapping[str, int] | None = None
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Exhaustive (input pattern, output response) pairs.

    `over` selects which inputs to sweep (default: all primary inputs);
    `fixed` pins the rest, e.g. a key vector.
    """
    sweep = list(over) if over is not None else list(net.inputs)
    pinned = dict(fixed or {})

    rows = []
    for assign in enumerate_assignments(sweep):
        assign.update(pinned)
        rows.append(
            (tuple(assign[n] for n in sweep), outputs_of(net, assign))
        )
    return rows
