"""Notebook 07 exported teaching code: a local two-layer PyTorch regressor."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn


class TwoLayerNet(nn.Module):
    """The network used by the exported notebook lesson."""

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
    loss = nn.functional.mse_loss(model(inputs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run(
    epochs: int = 20,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | None = None,
) -> dict[str, object]:
    """Train locally and optionally write the standard lesson artifacts."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    inputs = torch.linspace(0, 1, 64, device=resolved_device).reshape(-1, 1)
    targets = 1.5 * inputs + 0.25
    model = TwoLayerNet().to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    losses = [train_step(model, optimizer, inputs, targets) for _ in range(epochs)]
    artifacts = {}
    if output_dir is not None:
        from pytorch.common import save_training_artifacts

        artifacts = save_training_artifacts(
            [-loss for loss in losses],
            output_dir,
            metadata={"experiment": "notebook07_neural_networks", "seed": seed},
            checkpoint={"model_state_dict": model.state_dict(), "seed": seed},
        )
    return {
        "experiment": "notebook07_neural_networks",
        "epochs": epochs,
        "seed": seed,
        "device": str(resolved_device),
        "losses": losses,
        "final_loss": losses[-1],
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 07 PyTorch regression smoke")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(epochs=args.epochs, device=args.device, output_dir=args.output_dir)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
