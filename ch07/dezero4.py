"""Two-layer PyTorch network and one-step training lesson for chapter 07."""

from __future__ import annotations

import json

import torch
from torch import nn


class TwoLayerNet(nn.Module):
    """A small fully connected network suitable for inspecting in a lesson."""

    def __init__(self, hidden_size: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Perform one explicit forward/backward/update step and return its loss."""

    predictions = model(inputs)
    loss = nn.functional.mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def run(
    *, steps: int = 20, seed: int = 0, device: str | torch.device = "cpu"
) -> dict[str, object]:
    """Fit a two-layer regressor on a deterministic synthetic data set."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device.type).manual_seed(seed)
    inputs = torch.rand((64, 1), generator=generator, device=resolved_device)
    targets = 1.5 * inputs + 0.25
    model = TwoLayerNet().to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    losses = [train_step(model, optimizer, inputs, targets) for _ in range(steps)]
    return {
        "steps": steps,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
        "final_loss": losses[-1],
    }


def main() -> None:
    """Run a short command-line example without import-time work."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
