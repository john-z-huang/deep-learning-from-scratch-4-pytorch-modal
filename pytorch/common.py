"""Shared utilities for the PyTorch experiments and Modal runners."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a requested device without silently hiding an unavailable CUDA device."""

    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device!r}")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cartpole(seed: int, *, render_mode: str | None = None):
    """Create a non-GUI Gymnasium environment with deterministic episode seeding."""

    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def save_training_artifacts(
    rewards: list[float],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any],
    checkpoint: dict[str, Any] | None = None,
    checkpoint_name: str = "checkpoint.pt",
) -> dict[str, str]:
    """Write JSON metrics, a headless PNG curve, and an optional torch checkpoint."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    plot_path = directory / "rewards.png"
    metrics = {
        **metadata,
        "episode_count": len(rewards),
        "rewards": [float(value) for value in rewards],
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(metrics["rewards"])
    axis.set_xlabel("Episode")
    axis.set_ylabel("Total reward")
    axis.set_title(str(metadata.get("experiment", "training")))
    figure.tight_layout()
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)

    paths = {"metrics": str(metrics_path), "plot": str(plot_path)}
    if checkpoint is not None:
        checkpoint_path = directory / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        paths["checkpoint"] = str(checkpoint_path)
    return paths


def device_metadata(device: torch.device) -> dict[str, Any]:
    """Return JSON-safe runtime information for smoke-test evidence."""

    metadata: dict[str, Any] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        metadata["cuda_device"] = torch.cuda.get_device_name(device)
    return metadata
