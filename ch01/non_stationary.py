"""Non-stationary bandit and constant-step-size agent."""

from __future__ import annotations

import json
import numpy as np


class NonStatBandit:
    """Bandit whose reward probabilities drift after every interaction."""

    def __init__(self, arms: int = 10, rng: np.random.Generator | None = None):
        if arms < 1:
            raise ValueError("arms must be at least 1")
        self.rng = rng or np.random.default_rng()
        self.rates = self.rng.random(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        self.rates += 0.1 * self.rng.standard_normal(len(self.rates))
        return int(rate > self.rng.random())


class AlphaAgent:
    """Epsilon-greedy agent using a constant step size ``alpha``."""

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.8,
        actions: int = 10,
        rng: np.random.Generator | None = None,
    ):
        if actions < 1 or not 0 <= epsilon <= 1 or not 0 < alpha <= 1:
            raise ValueError("invalid agent parameters")
        self.epsilon, self.alpha, self.rng = (
            epsilon,
            alpha,
            (rng or np.random.default_rng()),
        )
        self.Qs = np.zeros(actions)

    def update(self, action: int, reward: float) -> None:
        self.Qs[action] += self.alpha * (reward - self.Qs[action])

    def get_action(self) -> int:
        return (
            int(self.rng.integers(len(self.Qs)))
            if self.rng.random() < self.epsilon
            else int(np.argmax(self.Qs))
        )


def run(*, steps: int = 100, seed: int = 0) -> dict[str, object]:
    """Run a bounded drifting-bandit experiment and return its reward trace."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    rng = np.random.default_rng(seed)
    bandit = NonStatBandit(rng=rng)
    agent = AlphaAgent(rng=rng)
    rewards = []
    for _ in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        rewards.append(reward)
    return {"steps": steps, "rewards": rewards, "mean_reward": float(np.mean(rewards))}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
