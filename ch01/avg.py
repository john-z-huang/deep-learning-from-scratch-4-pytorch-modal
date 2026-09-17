"""Compare naive and incremental sample-average updates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import numpy as np


def incremental_average(rewards: Iterable[float]) -> list[float]:
    """Return every prefix mean using the incremental update rule."""

    estimate = 0.0
    values: list[float] = []
    for count, reward in enumerate(rewards, 1):
        estimate += (reward - estimate) / count
        values.append(float(estimate))
    return values


def run(*, samples: int = 10, seed: int = 0) -> dict[str, list[float] | int]:
    """Generate a reproducible stream and expose its prefix estimates."""

    if samples < 1:
        raise ValueError("samples must be at least 1")
    rng = np.random.default_rng(seed)
    rewards = rng.random(samples).tolist()
    return {"rewards": rewards, "estimates": incremental_average(rewards)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(samples=args.samples, seed=args.seed), indent=2))
