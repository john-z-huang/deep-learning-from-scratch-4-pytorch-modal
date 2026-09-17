"""Chapter 07 tensor-operation lesson using PyTorch tensors."""

from __future__ import annotations

import json

import torch


def dot_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a tensor dot product without converting to NumPy."""

    return torch.matmul(left, right)


def run(*, device: str | torch.device = "cpu") -> float:
    """Run the tensor operation on the explicitly selected device."""

    resolved_device = torch.device(device)
    left = torch.tensor([1.0, 2.0, 3.0], device=resolved_device)
    right = torch.tensor([4.0, 5.0, 6.0], device=resolved_device)
    return float(dot_product(left, right).detach().cpu())


if __name__ == "__main__":
    print(json.dumps({"device": "cpu", "value": run()}))
