"""Device-aware REINFORCE CartPole experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from pytorch.common import device_metadata, make_cartpole, resolve_device, save_training_artifacts, seed_everything


class Policy(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(), nn.Linear(128, action_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.layers(states), dim=-1)


def run(
    episodes: int = 3000,
    *,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    output_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    max_steps: int = 500,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    env = make_cartpole(seed)
    observation_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    policy = Policy(observation_size, action_size).to(resolved_device)
    optimizer = optim.Adam(policy.parameters(), lr=0.0002)
    gamma = 0.98
    rewards: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            log_probs: list[torch.Tensor] = []
            episode_rewards: list[float] = []
            for _ in range(max_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=resolved_device).unsqueeze(0)
                distribution = Categorical(policy(state_tensor)[0])
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
                state, reward, terminated, truncated, _ = env.step(int(action.item()))
                episode_rewards.append(float(reward))
                if terminated or truncated:
                    break
            returns: list[float] = []
            running = 0.0
            for reward in reversed(episode_rewards):
                running = reward + gamma * running
                returns.append(running)
            returns.reverse()
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=resolved_device)
            if len(returns_tensor) > 1:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
            loss = -(torch.stack(log_probs) * returns_tensor).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rewards.append(float(sum(episode_rewards)))
            if episode % max(1, checkpoint_interval) == 0:
                print(f"episode={episode} reward={rewards[-1]:.1f}")
    finally:
        env.close()
    metadata = {"experiment": "reinforce", "seed": seed, **device_metadata(resolved_device)}
    checkpoint = {"experiment": "reinforce", "model_state_dict": policy.state_dict(), "seed": seed}
    paths = save_training_artifacts(rewards, output_dir, metadata=metadata, checkpoint=checkpoint) if output_dir else {}
    return {**metadata, "rewards": rewards, "artifacts": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="reinforce-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
