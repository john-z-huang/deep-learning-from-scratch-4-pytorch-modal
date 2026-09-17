"""Off-policy SARSA surface retaining target and behavior policies."""

from __future__ import annotations

import json

import numpy as np

from ch06.sarsa import SarsaAgent


class SarsaOffPolicyAgent(SarsaAgent):
    """SARSA agent whose behavior policy may differ from its target policy."""

    def __init__(self, *, behavior_epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = behavior_epsilon

    def target_action(self, state):
        """Return the greedy target-policy action without exploration."""

        return int(np.argmax([self.Q[state, a] for a in range(self.action_size)]))

    def update_off_policy(self, state, action, reward, next_state, done):
        """Bootstrap from the greedy target policy while behaving epsilon-greedily."""

        next_action = self.target_action(next_state)
        self.update(state, action, reward, next_state, next_action, done)


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Run a bounded off-policy SARSA example on a deterministic toy MDP."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    np.random.seed(seed)
    agent = SarsaOffPolicyAgent()
    for _ in range(episodes):
        state = 0
        for _ in range(4):
            action = agent.get_action(state)
            next_state = 1 if action == 0 else state
            done = next_state == 1
            agent.update_off_policy(state, action, float(done), next_state, done)
            if done:
                break
            state = next_state
    return {
        "episodes": episodes,
        "seed": seed,
        "q_values": {f"{state}:{action}": float(value) for (state, action), value in agent.Q.items()},
    }
    return {"seed": seed, "agent": "SarsaOffPolicyAgent"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
