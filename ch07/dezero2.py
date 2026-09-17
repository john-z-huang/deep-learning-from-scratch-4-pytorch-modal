"""Chapter 07 autograd lesson using a PyTorch optimizer."""

import torch


def rosenbrock(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2


def run(*, steps: int = 20, learning_rate: float = 0.001) -> float:
    x0 = torch.tensor(0.0, requires_grad=True)
    x1 = torch.tensor(2.0, requires_grad=True)
    optimizer = torch.optim.SGD((x0, x1), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(x0, x1)
        loss.backward()
        optimizer.step()
    return float(rosenbrock(x0, x1).detach())


if __name__ == "__main__":
    print(run())
