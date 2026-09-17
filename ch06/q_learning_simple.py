"""Minimal epsilon-greedy Q-learning agent for the chapter introduction."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(self, action_size=4, alpha=0.8, gamma=0.9, epsilon=0.1):
        self.action_size, self.alpha, self.gamma, self.epsilon = (
            action_size,
            alpha,
            gamma,
            epsilon,
        )
        self.Q = defaultdict(float)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(np.argmax([self.Q[state, a] for a in range(self.action_size)]))

    def update(self, state, action, reward, next_state, done):
        next_q = (
            0.0 if done else max(self.Q[next_state, a] for a in range(self.action_size))
        )
        self.Q[state, action] += self.alpha * (
            reward + self.gamma * next_q - self.Q[state, action]
        )


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Learn the optimal action in a two-state deterministic toy MDP."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    np.random.seed(seed)
    agent = QLearningAgent(epsilon=0.1)
    for _ in range(episodes):
        state = 0
        for _ in range(4):
            action = agent.get_action(state)
            next_state = 1 if action == 0 else state
            done = next_state == 1
            agent.update(state, action, float(done), next_state, done)
            if done:
                break
            state = next_state
    return {
        "episodes": episodes,
        "seed": seed,
        "q_values": {f"{state}:{action}": float(value) for (state, action), value in agent.Q.items()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
