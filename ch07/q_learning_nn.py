"""Neural Q-learning target and loss demonstration for chapter 07."""

from __future__ import annotations

import json

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map a vector observation to one Q value per discrete action."""

    def __init__(self, observation_size: int = 4, action_size: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def q_learning_target(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Compute the discounted max-next-action target with terminal masking."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    return rewards + (1 - done) * gamma * next_q_values.max(dim=1).values


def q_learning_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Select chosen-action values and calculate a squared TD error."""

    chosen = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    return nn.functional.mse_loss(chosen, targets.detach())


def run(
    *, steps: int = 4, seed: int = 0, device: str | torch.device = "cpu"
) -> dict[str, object]:
    """Train the local network on a deterministic batch of TD targets."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    states = torch.arange(
        16, dtype=torch.float32, device=resolved_device
    ).reshape(4, 4) / 16
    next_states = states + 0.1
    actions = torch.tensor([0, 1, 0, 1], device=resolved_device)
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0], device=resolved_device)
    done = torch.tensor([0.0, 1.0, 0.0, 1.0], device=resolved_device)
    network = QNetwork().to(resolved_device)
    target_network = QNetwork().to(resolved_device)
    target_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        targets = q_learning_target(rewards, target_network(next_states), done)
        loss = q_learning_loss(network(states), actions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "steps": steps,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
