"""Notebook-friendly TD(0) and Q-learning update equations."""

from __future__ import annotations

import argparse
import json


def td_zero_update(
    value: float,
    reward: float,
    next_value: float,
    *,
    alpha: float = 0.1,
    gamma: float = 0.9,
    done: bool = False,
) -> float:
    """Apply one state-value TD(0) update."""

    target = reward if done else reward + gamma * next_value
    return value + alpha * (target - value)


def q_learning_update(
    value: float,
    reward: float,
    next_action_values: list[float],
    *,
    alpha: float = 0.1,
    gamma: float = 0.9,
    done: bool = False,
) -> float:
    """Apply one off-policy max-action Q-learning update."""

    target = reward if done else reward + gamma * max(next_action_values)
    return value + alpha * (target - value)


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    values = {"state0": 0.0, "state1": 0.0}
    for _ in range(episodes):
        values["state1"] = td_zero_update(values["state1"], 1.0, 0.0, done=True)
        values["state0"] = td_zero_update(values["state0"], 0.0, values["state1"])
    return {"episodes": episodes, "seed": seed, "values": values}


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
