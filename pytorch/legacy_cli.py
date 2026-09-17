"""Compatibility CLI used by the chapter teaching modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pytorch.tabular import run as run_tabular


def run_chapter(
    experiment: str,
    *,
    episodes: int = 1,
    seed: int = 0,
    device: str = "cpu",
    output_dir: str | Path = "chapter-runs",
) -> dict[str, Any]:
    return run_tabular(
        experiment, episodes=episodes, seed=seed, device=device, output_dir=output_dir
    )


def cli(experiment: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", default=f"{experiment}-runs")
    args = parser.parse_args()
    print(json.dumps(run_chapter(experiment, **vars(args)), indent=2, sort_keys=True))
