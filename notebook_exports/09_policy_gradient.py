"""Notebook 09 exported teaching code: a local policy-gradient update."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn


class Policy(nn.Module):
    """Categorical policy for a compact, inspectable feature batch."""

    def __init__(self, input_size: int = 2, action_size: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(states), dim=-1)


def discounted_returns(rewards: list[float], gamma: float = 0.9) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def policy_gradient_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Compute the negative score-function objective."""

    return -(log_probs * returns.detach()).sum()


def run(
    episodes: int = 1,
    *,
    max_steps: int = 20,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | None = None,
) -> dict[str, object]:
    """Run bounded local policy updates and optionally save a checkpoint."""

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    states = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=resolved_device)
    actions = torch.tensor([0, 1], device=resolved_device)
    rewards = [0.0, 1.0]
    policy = Policy().to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    losses = []
    for _ in range(episodes * max_steps):
        distribution = torch.distributions.Categorical(policy(states))
        log_probs = distribution.log_prob(actions)
        loss = policy_gradient_loss(
            log_probs, discounted_returns(rewards, 0.9).to(resolved_device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    artifacts = {}
    if output_dir is not None:
        from pytorch.common import save_training_artifacts

        artifacts = save_training_artifacts(
            [-loss for loss in losses[-episodes:]],
            output_dir,
            metadata={"experiment": "notebook09_policy_gradient", "seed": seed},
            checkpoint={"model_state_dict": policy.state_dict(), "seed": seed},
        )
    return {
        "experiment": "notebook09_policy_gradient",
        "episodes": episodes,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
        "rewards": [float(-loss) for loss in losses[-episodes:]],
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 09 PyTorch policy-gradient smoke")
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
