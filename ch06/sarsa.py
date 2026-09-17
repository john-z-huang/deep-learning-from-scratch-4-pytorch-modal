"""On-policy SARSA update with explicit next-action bootstrapping."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


class SarsaAgent:
    def __init__(self, action_size=4, alpha=0.8, gamma=0.9):
        self.action_size, self.alpha, self.gamma, self.epsilon = (
            action_size,
            alpha,
            gamma,
            0.1,
        )
        self.Q = defaultdict(float)

    def get_action(self, state):
        return (
            int(np.random.randint(self.action_size))
            if np.random.rand() < self.epsilon
            else int(np.argmax([self.Q[state, a] for a in range(self.action_size)]))
        )

    def update(self, state, action, reward, next_state, next_action, done):
        target = (
            reward if done else reward + self.gamma * self.Q[next_state, next_action]
        )
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Run SARSA on the same tiny MDP used by ``q_learning_simple``."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    np.random.seed(seed)
    agent = SarsaAgent()
    for _ in range(episodes):
        state = 0
        action = agent.get_action(state)
        for _ in range(4):
            next_state = 1 if action == 0 else state
            done = next_state == 1
            next_action = agent.get_action(next_state)
            agent.update(state, action, float(done), next_state, next_action, done)
            if done:
                break
            state, action = next_state, next_action
    return {
        "episodes": episodes,
        "seed": seed,
        "q_values": {f"{state}:{action}": float(value) for (state, action), value in agent.Q.items()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
