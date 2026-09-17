"""PyTorch adaptation of Notebook 07's two-layer regression example."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pytorch.common import device_metadata, resolve_device, save_training_artifacts, seed_everything


class TwoLayerNet(nn.Module):
    """Small sigmoid hidden-layer network used by the Notebook 07 lesson."""

    def __init__(self, hidden_size: int = 10, out_size: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, hidden_size), nn.Sigmoid(), nn.Linear(hidden_size, out_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def run(
    epochs: int = 1000,
    *,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train the lesson network without import-time work or GUI output."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    generator = torch.Generator(device=resolved_device.type).manual_seed(seed)
    inputs = torch.rand((100, 1), generator=generator, device=resolved_device)
    targets = torch.sin(2 * torch.pi * inputs) + torch.rand((100, 1), generator=generator, device=resolved_device)
    model = TwoLayerNet().to(resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    for _ in range(epochs):
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    metadata = {"experiment": "neural_networks", "seed": seed, "epochs": epochs, **device_metadata(resolved_device)}
    checkpoint = {"experiment": "neural_networks", "model_state_dict": model.state_dict(), "seed": seed}
    artifacts = save_training_artifacts(
        [-loss for loss in losses], output_dir, metadata=metadata, checkpoint=checkpoint
    ) if output_dir is not None else {}
    return {**metadata, "final_loss": losses[-1], "losses": losses, "artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="neural-network-runs")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
