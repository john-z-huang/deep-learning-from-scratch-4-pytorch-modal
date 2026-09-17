"""Repeated bandit experiment used to estimate average action rates."""

from __future__ import annotations

import json
import numpy as np

from ch01.bandit import Bandit, Agent


def run(
    *, runs: int = 1, steps: int = 100, epsilon: float = 0.1, seed: int = 0
) -> dict[str, object]:
    """Estimate the mean reward of an epsilon-greedy sample-average agent."""

    if runs < 1 or steps < 1:
        raise ValueError("runs and steps must be at least 1")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(runs):
        bandit, agent = Bandit(rng=rng), Agent(epsilon, rng=rng)
        rewards = []
        for _ in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            rewards.append(reward)
        rates.append(float(np.mean(rewards)))
    return {
        "runs": runs,
        "steps": steps,
        "mean_rate": float(np.mean(rates)),
        "rates": rates,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
