"""TD(0) state-value update for a fixed policy."""

from __future__ import annotations

from collections import defaultdict


class TdAgent:
    def __init__(self, alpha=0.01, gamma=0.9):
        self.alpha, self.gamma, self.V = alpha, gamma, defaultdict(float)

    def evaluate(self, state, reward, next_state, done):
        next_value = 0.0 if done else self.V[next_state]
        self.V[state] += self.alpha * (reward + self.gamma * next_value - self.V[state])


def run(*, episodes: int = 1, seed: int = 0) -> dict[str, object]:
    """Evaluate a deterministic reward sequence using TD(0) bootstrapping."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    agent = TdAgent()
    for _ in range(episodes):
        agent.evaluate(0, 0.0, 1, False)
        agent.evaluate(1, 1.0, 2, True)
    return {
        "episodes": episodes,
        "seed": seed,
        "values": {str(state): float(value) for state, value in agent.V.items()},
    }


if __name__ == "__main__":
    print(run())
