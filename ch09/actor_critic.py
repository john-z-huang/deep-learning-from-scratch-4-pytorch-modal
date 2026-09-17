"""Actor-critic networks and one-episode training lesson for chapter 09."""

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


class PolicyNet(nn.Module):
    """Actor network producing a categorical action distribution."""

    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(states), dim=-1)


class ValueNet(nn.Module):
    """Critic network estimating the scalar state value V(s)."""

    def __init__(self, observation_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states).squeeze(-1)


def actor_critic_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actor and critic losses from one bootstrapped trajectory."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    targets = rewards + gamma * (1 - dones) * next_value.detach()
    advantages = targets.detach() - values
    actor_loss = -(log_probs * advantages.detach()).sum()
    critic_loss = nn.functional.mse_loss(values, targets.detach())
    return actor_loss, critic_loss


def run(
    episodes: int = 1,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    max_steps: int = 500,
) -> dict[str, object]:
    """Train actor and critic on bounded CartPole episodes."""

    if episodes < 1 or max_steps < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, max_steps, and checkpoint_interval must be positive")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    env = make_cartpole(seed)
    actor = PolicyNet(int(env.observation_space.shape[0]), int(env.action_space.n)).to(
        resolved_device
    )
    critic = ValueNet(int(env.observation_space.shape[0])).to(resolved_device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=2e-4
    )
    rewards = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            log_probs = []
            values = []
            step_rewards = []
            dones = []
            for _ in range(max_steps):
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=resolved_device
                ).unsqueeze(0)
                probabilities = actor(state_tensor)[0]
                action_distribution = Categorical(probabilities)
                action = action_distribution.sample()
                log_probs.append(action_distribution.log_prob(action))
                values.append(critic(state_tensor)[0])
                state, reward, terminated, truncated, _ = env.step(int(action.item()))
                step_rewards.append(float(reward))
                done = terminated or truncated
                dones.append(float(done))
                if done:
                    break
            next_state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=resolved_device
            ).unsqueeze(0)
            next_value = critic(next_state_tensor)[0]
            rewards_tensor = torch.tensor(
                step_rewards, dtype=torch.float32, device=resolved_device
            )
            dones_tensor = torch.tensor(
                dones, dtype=torch.float32, device=resolved_device
            )
            actor_loss, critic_loss = actor_critic_loss(
                torch.stack(log_probs),
                torch.stack(values),
                rewards_tensor,
                next_value,
                dones_tensor,
            )
            optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            optimizer.step()
            rewards.append(float(sum(step_rewards)))
    finally:
        env.close()
    metadata = {
        "experiment": "chapter09_actor_critic",
        "seed": seed,
        **device_metadata(resolved_device),
    }
    artifacts = (
        save_training_artifacts(
            rewards,
            output_dir,
            metadata=metadata,
            checkpoint={
                "experiment": "chapter09_actor_critic",
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
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
    parser.add_argument("--output-dir", default="chapter09-actor-critic-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
