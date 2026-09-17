"""Chapter 07 linear-regression lesson using a PyTorch Module."""

import torch
from torch import nn


def run(*, steps: int = 20, seed: int = 0) -> float:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand((100, 1), generator=generator)
    targets = 5 + 2 * inputs + torch.rand((100, 1), generator=generator)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    return float(nn.functional.mse_loss(model(inputs), targets).detach())


if __name__ == "__main__":
    print(run())
