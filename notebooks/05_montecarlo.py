"""Notebook-friendly every-visit Monte Carlo return updates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def episode_returns(rewards: list[float], gamma: float = 0.9) -> list[float]:
    """Compute the return from each time step of one episode."""

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return list(reversed(returns))


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Estimate two state values from deterministic sample episodes."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    values = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        returns = episode_returns([0.0, 1.0])
        for state, value in zip((0, 1), returns):
            counts[state] += 1
            values[state] += (value - values[state]) / counts[state]
    return {"episodes": episodes, "seed": seed, "values": dict(values)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(episodes=args.episodes, seed=args.seed), indent=2, sort_keys=True
        )
    )
