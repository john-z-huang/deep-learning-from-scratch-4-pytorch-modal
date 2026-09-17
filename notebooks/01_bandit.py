"""Notebook-friendly epsilon-greedy bandit implementation."""

from __future__ import annotations

import argparse
import json

import numpy as np


def incremental_average(values: list[float]) -> list[float]:
    """Show how a sample average can be updated without storing history."""

    estimate = 0.0
    estimates = []
    for count, value in enumerate(values, 1):
        estimate += (value - estimate) / count
        estimates.append(float(estimate))
    return estimates


def run(
    *, steps: int = 20, arms: int = 10, epsilon: float = 0.1, seed: int = 0
) -> dict[str, object]:
    """Run a bandit lesson with local action-value estimates."""

    if steps < 1 or arms < 1 or not 0 <= epsilon <= 1:
        raise ValueError("invalid bandit parameters")
    rng = np.random.default_rng(seed)
    rates = rng.random(arms)
    values = np.zeros(arms)
    counts = np.zeros(arms)
    rewards = []
    for _ in range(steps):
        action = (
            int(rng.integers(arms))
            if rng.random() < epsilon
            else int(np.argmax(values))
        )
        reward = float(rng.random() < rates[action])
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]
        rewards.append(reward)
    return {
        "steps": steps,
        "seed": seed,
        "rewards": rewards,
        "estimates": values.tolist(),
        "mean_reward": float(np.mean(rewards)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(steps=args.steps, seed=args.seed), indent=2, sort_keys=True))
