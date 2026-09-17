"""Device-aware one-step Actor-Critic CartPole experiment."""

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


class PolicyNet(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(), nn.Linear(128, action_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.layers(states), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, observation_size: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def run(
    episodes: int = 2000,
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
    policy = PolicyNet(observation_size, action_size).to(resolved_device)
    value = ValueNet(observation_size).to(resolved_device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=0.0002)
    value_optimizer = optim.Adam(value.parameters(), lr=0.0005)
    gamma = 0.98
    rewards: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            for _ in range(max_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=resolved_device).unsqueeze(0)
                distribution = Categorical(policy(state_tensor)[0])
                action = distribution.sample()
                next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                next_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=resolved_device).unsqueeze(0)
                reward_tensor = torch.tensor(float(reward), dtype=torch.float32, device=resolved_device)
                done_tensor = torch.tensor(float(done), dtype=torch.float32, device=resolved_device)
                target = reward_tensor + gamma * value(next_tensor).squeeze(1).detach() * (1.0 - done_tensor)
                current_value = value(state_tensor).squeeze(1)
                value_loss = F.mse_loss(current_value, target)
                advantage = (target - current_value).detach()
                policy_loss = -distribution.log_prob(action) * advantage
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                state = next_state
                total_reward += float(reward)
                if done:
                    break
            rewards.append(total_reward)
            if episode % max(1, checkpoint_interval) == 0:
                print(f"episode={episode} reward={total_reward:.1f}")
    finally:
        env.close()
    metadata = {"experiment": "actor_critic", "seed": seed, **device_metadata(resolved_device)}
    checkpoint = {
        "experiment": "actor_critic",
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value.state_dict(),
        "seed": seed,
    }
    paths = save_training_artifacts(rewards, output_dir, metadata=metadata, checkpoint=checkpoint) if output_dir else {}
    return {**metadata, "rewards": rewards, "artifacts": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="actor-critic-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
