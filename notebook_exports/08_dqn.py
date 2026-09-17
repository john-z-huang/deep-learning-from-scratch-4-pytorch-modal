"""Notebook 08 exported teaching code: a local DQN target and update loop."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map observations to action values."""

    def __init__(self, observation_size: int = 4, action_size: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 16),
            nn.ReLU(),
            nn.Linear(16, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def q_learning_target(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Compute the terminal-aware max-next-action target."""

    return rewards + (1 - done) * gamma * next_q_values.max(dim=1).values


def run(
    episodes: int = 1,
    *,
    max_steps: int = 20,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | None = None,
) -> dict[str, object]:
    """Run a bounded batch-only DQN lesson without importing a shared runner."""

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    states = torch.arange(16, dtype=torch.float32, device=resolved_device).reshape(4, 4) / 16
    next_states = states + 0.1
    actions = torch.tensor([0, 1, 0, 1], device=resolved_device)
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0], device=resolved_device)
    done = torch.tensor([0.0, 1.0, 0.0, 1.0], device=resolved_device)
    network = QNetwork().to(resolved_device)
    target_network = QNetwork().to(resolved_device)
    target_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    losses = []
    for _ in range(episodes * max_steps):
        target = q_learning_target(rewards, target_network(next_states), done)
        predicted = network(states).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.mse_loss(predicted, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    rewards_out = [float(-loss) for loss in losses[-episodes:]]
    artifacts = {}
    if output_dir is not None:
        from pytorch.common import save_training_artifacts

        artifacts = save_training_artifacts(
            rewards_out,
            output_dir,
            metadata={"experiment": "notebook08_dqn", "seed": seed},
            checkpoint={"model_state_dict": network.state_dict(), "seed": seed},
        )
    return {
        "experiment": "notebook08_dqn",
        "episodes": episodes,
        "seed": seed,
        "device": str(resolved_device),
        "rewards": rewards_out,
        "losses": losses,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 08 PyTorch DQN smoke")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(
        episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
