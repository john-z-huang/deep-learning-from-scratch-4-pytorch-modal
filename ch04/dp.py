"""Two-state Bellman update from the dynamic-programming lesson."""

from __future__ import annotations

import argparse
import json


def bellman_update(values: dict[str, float], gamma: float = 0.9) -> dict[str, float]:
    """Apply one synchronous Bellman expectation backup."""

    new_values = dict(values)
    new_values["L1"] = 0.5 * (-1 + gamma * values["L1"]) + 0.5 * (
        1 + gamma * values["L2"]
    )
    new_values["L2"] = 0.5 * (gamma * values["L1"]) + 0.5 * (-1 + gamma * values["L2"])
    return new_values


def run(
    *, threshold: float = 0.0001, gamma: float = 0.9, max_iterations: int = 1000
) -> dict[str, object]:
    """Iterate the backup until the two-state values are numerically stable."""

    if not 0 <= gamma < 1 or threshold <= 0 or max_iterations < 1:
        raise ValueError("invalid Bellman iteration parameters")
    values = {"L1": 0.0, "L2": 0.0}
    for iteration in range(1, max_iterations + 1):
        updated = bellman_update(values, gamma)
        if max(abs(updated[key] - values[key]) for key in values) < threshold:
            return {"values": updated, "iterations": iteration}
        values = updated
    raise RuntimeError("Bellman iteration did not converge within max_iterations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.0001)
    args = parser.parse_args()
    print(json.dumps(run(gamma=args.gamma, threshold=args.threshold), indent=2))
