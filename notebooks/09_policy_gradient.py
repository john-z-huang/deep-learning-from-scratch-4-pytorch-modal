"""Notebook-friendly score-function policy-gradient lesson."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn


class Policy(nn.Module):
    """Map a small feature vector to categorical action probabilities."""

    def __init__(self, input_size: int = 2, action_size: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 16), nn.Tanh(), nn.Linear(16, action_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(states), dim=-1)


def discounted_returns(rewards: list[float], gamma: float = 0.9) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.tensor(list(reversed(returns)))


def policy_gradient_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Return the negative score-function objective for one episode."""

    return -(log_probs * returns.detach()).sum()


def run(
    *,
    steps: int = 4,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | None = None,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    states = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], device=resolved_device
    )
    actions = torch.tensor([0, 1], device=resolved_device)
    rewards = [0.0, 1.0]
    policy = Policy().to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        distribution = torch.distributions.Categorical(policy(states))
        log_probs = distribution.log_prob(actions)
        loss = policy_gradient_loss(
            log_probs, discounted_returns(rewards).to(resolved_device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    artifacts = {}
    if output_dir is not None:
        from pytorch.common import save_training_artifacts

        artifacts = save_training_artifacts(
            [-loss for loss in losses],
            output_dir,
            metadata={"experiment": "notebook_09_policy_gradient", "seed": seed},
            checkpoint={"model_state_dict": policy.state_dict(), "seed": seed},
        )
    return {
        "steps": steps,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(steps=args.steps, seed=args.seed), indent=2, sort_keys=True))
