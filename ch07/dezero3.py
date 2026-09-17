"""Chapter 07 linear-regression lesson using a PyTorch Module."""

from __future__ import annotations

import json

import torch
from torch import nn


def run(
    *,
    steps: int = 20,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> float:
    """Fit a linear model on a reproducible synthetic data set."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device.type).manual_seed(seed)
    inputs = torch.rand((100, 1), generator=generator, device=resolved_device)
    targets = 5 + 2 * inputs + torch.rand(
        (100, 1), generator=generator, device=resolved_device
    )
    model = nn.Linear(1, 1, device=resolved_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    return float(nn.functional.mse_loss(model(inputs), targets).detach().cpu())


if __name__ == "__main__":
    print(json.dumps({"device": "cpu", "loss": run()}))
