"""Notebook-friendly PyTorch two-layer regression lesson."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn


class TwoLayerNet(nn.Module):
    """A sigmoid hidden layer makes the computation graph easy to inspect."""

    def __init__(self, hidden_size: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def fit(
    epochs: int = 20,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | None = None,
) -> dict[str, object]:
    """Fit the local model and return the loss curve."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    inputs = torch.linspace(0, 1, 64, device=resolved_device).reshape(-1, 1)
    targets = 1.5 * inputs + 0.25
    model = TwoLayerNet().to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    losses = []
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(inputs), targets)
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
            metadata={"experiment": "notebook_07_neural_networks", "seed": seed},
            checkpoint={"model_state_dict": model.state_dict(), "seed": seed},
        )
    return {
        "epochs": epochs,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
        "artifacts": artifacts,
    }


run = fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(fit(args.epochs, seed=args.seed), indent=2, sort_keys=True))
