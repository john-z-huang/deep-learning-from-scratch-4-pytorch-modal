"""Off-policy Monte Carlo control agent with target/behavior policies."""

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from ch05.mc_control import greedy_probs


class McOffPolicyAgent:
    """Weighted-return Monte Carlo control with separate ``pi`` and ``b``."""

    def __init__(self, *, action_size: int = 4, epsilon: float = 0.1, rng=None):
        self.gamma, self.epsilon, self.alpha = 0.9, epsilon, 0.2
        self.action_size = action_size
        self.rng = rng or np.random.default_rng()
        self.pi = defaultdict(lambda: {a: 0.25 for a in range(4)})
        self.b = defaultdict(lambda: {a: 0.25 for a in range(4)})
        self.Q = defaultdict(float)
        self.memory = []

    def get_action(self, state):
        probs = self.b[state]
        return int(self.rng.choice(list(probs), p=list(probs.values())))

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self) -> None:
        G, rho = 0.0, 1.0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            key = (state, action)
            self.Q[key] += self.alpha * rho * (G - self.Q[key])
            rho *= self.pi[state][action] / self.b[state][action]
            self.pi[state] = greedy_probs(self.Q, state)
            self.b[state] = greedy_probs(
                self.Q, state, self.epsilon, self.action_size
            )


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Learn from a bounded two-step trajectory without a rendering side effect."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = np.random.default_rng(seed)
    agent = McOffPolicyAgent(rng=rng)
    for _ in range(episodes):
        agent.reset()
        state = 0
        for step in range(2):
            action = agent.get_action(state)
            reward = float(state == 1 and action == 0)
            agent.add(state, action, reward)
            state += 1
        agent.update()
    q_values = [
        {"state": state, "action": action, "value": float(value)}
        for (state, action), value in sorted(agent.Q.items())
    ]
    return {"episodes": episodes, "seed": seed, "q_values": q_values}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
