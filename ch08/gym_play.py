"""Gymnasium reset/step API demonstration for the chapter 08 environment."""

from __future__ import annotations

import json


def rollout(seed: int = 0, *, steps: int = 5) -> dict[str, object]:
    """Run a random CartPole rollout and return observations/rewards, headlessly."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    try:
        observation, info = env.reset(seed=seed)
        rewards = []
        for _ in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "observation_shape": list(observation.shape),
            "rewards": rewards,
            "info_keys": sorted(info),
        }
    finally:
        env.close()


run = rollout


if __name__ == "__main__":
    print(json.dumps(rollout(), indent=2, sort_keys=True))
