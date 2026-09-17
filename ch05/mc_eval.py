"""Monte Carlo value evaluation with a readable random policy agent."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


class RandomAgent:
    """Every-visit Monte Carlo evaluator for a fixed random behavior policy."""

    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.V = defaultdict(float)
        self.counts = defaultdict(int)
        self.memory = []

    def get_action(self, state, rng=None):
        return int((rng or np.random.default_rng()).integers(4))

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def evaluate(self):
        total = 0.0
        for state, _, reward in reversed(self.memory):
            total = self.gamma * total + reward
            self.counts[state] += 1
            self.V[state] += (total - self.V[state]) / self.counts[state]


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Evaluate a tiny two-step random-policy episode repeatedly."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = np.random.default_rng(seed)
    agent = RandomAgent()
    for _ in range(episodes):
        agent.reset()
        state = 0
        for step in range(2):
            action = agent.get_action(state, rng)
            reward = float(step == 1 and action == 0)
            agent.add(state, action, reward)
            state += 1
        agent.evaluate()
    return {
        "episodes": episodes,
        "seed": seed,
        "values": {str(state): float(value) for state, value in agent.V.items()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
