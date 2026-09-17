"""Episodic policy-gradient loss and CartPole training lesson."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from pytorch.common import (
    device_metadata,
    make_cartpole,
    resolve_device,
    save_training_artifacts,
    seed_everything,
)


class Policy(nn.Module):
    """A softmax policy whose output rows sum to one."""

    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(states), dim=-1)


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute reward-to-go values for an episodic policy update."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def episode_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return the score-function loss for one sampled episode."""

    if log_probs.shape != returns.shape:
        raise ValueError("log_probs and returns must have the same shape")
    normalized = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return -(log_probs * normalized.detach()).sum()


def run(
    episodes: int = 1,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    max_steps: int = 500,
) -> dict[str, object]:
    """Train the policy for a bounded number of CartPole episodes."""

    if episodes < 1 or max_steps < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, max_steps, and checkpoint_interval must be positive")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    env = make_cartpole(seed)
    policy = Policy(int(env.observation_space.shape[0]), int(env.action_space.n)).to(
        resolved_device
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    rewards = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            log_probs = []
            episode_rewards = []
            for _ in range(max_steps):
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=resolved_device
                ).unsqueeze(0)
                distribution = Categorical(policy(state_tensor)[0])
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
                state, reward, terminated, truncated, _ = env.step(int(action.item()))
                episode_rewards.append(float(reward))
                if terminated or truncated:
                    break
            returns = discounted_returns(episode_rewards, 0.99).to(resolved_device)
            loss = episode_loss(torch.stack(log_probs), returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rewards.append(float(sum(episode_rewards)))
    finally:
        env.close()
    metadata = {
        "experiment": "chapter09_policy_gradient",
        "seed": seed,
        **device_metadata(resolved_device),
    }
    artifacts = (
        save_training_artifacts(
            rewards,
            output_dir,
            metadata=metadata,
            checkpoint={
                "experiment": "chapter09_policy_gradient",
                "model_state_dict": policy.state_dict(),
            },
        )
        if output_dir is not None
        else {}
    )
    return {**metadata, "rewards": rewards, "artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="chapter09-policy-gradient-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
