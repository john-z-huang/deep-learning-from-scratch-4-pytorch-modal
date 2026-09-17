"""Importance-sampling estimate for a small discrete expectation."""

from __future__ import annotations

import numpy as np


def estimate(*, trials=100, seed=0):
    """Estimate ``E_target[value]`` from samples drawn by ``behavior``."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    rng = np.random.default_rng(seed)
    values = np.array([1, 2, 3])
    target = np.array([0.1, 0.1, 0.8])
    behavior = np.array([0.2, 0.2, 0.6])
    samples = []
    for _ in range(trials):
        index = int(rng.choice(3, p=behavior))
        samples.append(target[index] / behavior[index] * values[index])
    return {
        "expectation": float(np.dot(values, target)),
        "estimate": float(np.mean(samples)),
        "seed": seed,
    }


run = estimate


if __name__ == "__main__":
    print(estimate())
