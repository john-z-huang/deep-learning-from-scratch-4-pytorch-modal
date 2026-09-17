"""Chapter 07 tensor-operation lesson using PyTorch tensors."""

import torch


def run() -> torch.Tensor:
    left = torch.tensor([1.0, 2.0, 3.0])
    right = torch.tensor([4.0, 5.0, 6.0])
    return torch.matmul(left, right)


if __name__ == "__main__":
    print(run())
