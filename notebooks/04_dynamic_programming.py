"""Notebook-friendly Bellman backups for a two-state MDP."""

from __future__ import annotations

import argparse
import json


def bellman_backup(values: dict[str, float], gamma: float = 0.9) -> dict[str, float]:
    """Perform one synchronous expected-value backup."""

    return {
        "L1": 0.5 * (-1 + gamma * values["L1"]) + 0.5 * (1 + gamma * values["L2"]),
        "L2": 0.5 * gamma * values["L1"] + 0.5 * (-1 + gamma * values["L2"]),
    }


def value_iteration(
    *, gamma: float = 0.9, threshold: float = 1e-4, max_iterations: int = 1000
) -> tuple[dict[str, float], int]:
    """Iterate Bellman backups and return the converged table and count."""

    if not 0 <= gamma < 1 or threshold <= 0 or max_iterations < 1:
        raise ValueError("invalid value-iteration parameters")
    values = {"L1": 0.0, "L2": 0.0}
    for iteration in range(1, max_iterations + 1):
        updated = bellman_backup(values, gamma)
        if max(abs(updated[key] - values[key]) for key in values) < threshold:
            return updated, iteration
        values = updated
    raise RuntimeError("value iteration did not converge")


def run(*, gamma: float = 0.9) -> dict[str, object]:
    values, iterations = value_iteration(gamma=gamma)
    return {"gamma": gamma, "values": values, "iterations": iterations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()
    print(json.dumps(run(gamma=args.gamma), indent=2, sort_keys=True))
