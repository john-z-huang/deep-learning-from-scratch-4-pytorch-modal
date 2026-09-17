"""Multi-armed bandit: readable teaching implementation with a bounded runner."""

from __future__ import annotations
import argparse
import json
import numpy as np
from pytorch.common import save_training_artifacts


class Bandit:
    def __init__(self, arms=10, rng=None):
        self.rng = rng or np.random.default_rng()
        self.rates = self.rng.random(arms)

    def play(self, arm):
        return int(self.rates[arm] > self.rng.random())


class Agent:
    def __init__(self, epsilon, action_size=10, rng=None):
        self.epsilon, self.rng = epsilon, (rng or np.random.default_rng())
        self.Qs, self.ns = np.zeros(action_size), np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        return (
            int(self.rng.integers(len(self.Qs)))
            if self.rng.random() < self.epsilon
            else int(np.argmax(self.Qs))
        )


def run(
    *, episodes=1, seed=0, device="cpu", output_dir="bandit-runs", checkpoint_interval=1
):
    if episodes < 1 or checkpoint_interval < 1:
        raise ValueError("episodes and checkpoint_interval must be at least 1")
    rng = np.random.default_rng(seed)
    bandit, agent, rewards = Bandit(rng=rng), Agent(0.1, rng=rng), []
    for _ in range(episodes):
        total = 0.0
        for _ in range(20):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total += reward
        rewards.append(total)
    metadata = {
        "schema_version": 1,
        "experiment": "bandit",
        "implementation": "chapter-ch01",
        "seed": seed,
        "requested_device": device,
        "device": "cpu",
        "arms": 10,
        "horizon": 20,
    }
    paths = save_training_artifacts(
        rewards,
        output_dir,
        metadata=metadata,
        checkpoint={"experiment": "bandit", "seed": seed, "Qs": agent.Qs.tolist()},
    )
    return {
        **metadata,
        "episode_count": episodes,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "output_paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", default="bandit-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
