"""In-place Bellman update illustrating update-order effects."""

from __future__ import annotations

import json


def inplace_update(values: dict[str, float], gamma: float = 0.9) -> dict[str, float]:
    """Update ``L1`` before ``L2`` to make update order visible."""

    first = 0.5 * (-1 + gamma * values["L1"]) + 0.5 * (1 + gamma * values["L2"])
    values["L1"] = first
    second = 0.5 * (gamma * values["L1"]) + 0.5 * (-1 + gamma * values["L2"])
    values["L2"] = second
    return values


def run(
    *, threshold: float = 0.0001, gamma: float = 0.9, max_iterations: int = 1000
) -> dict[str, object]:
    """Run in-place backups and report the iteration count."""

    if not 0 <= gamma < 1 or threshold <= 0 or max_iterations < 1:
        raise ValueError("invalid Bellman iteration parameters")
    values = {"L1": 0.0, "L2": 0.0}
    for iteration in range(1, max_iterations + 1):
        old = values.copy()
        inplace_update(values, gamma)
        if max(abs(values[key] - old[key]) for key in values) < threshold:
            return {"values": values, "iterations": iteration}
    raise RuntimeError("in-place iteration did not converge within max_iterations")


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
