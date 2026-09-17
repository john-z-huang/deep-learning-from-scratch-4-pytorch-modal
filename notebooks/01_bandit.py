"""Notebook export: the bandit lesson uses the chapter's Bandit and Agent."""

import argparse
import json
from ch01.bandit import Agent, Bandit, run  # noqa: F401


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="/tmp/notebook-01-runs")
    args = p.parse_args()
    print(
        json.dumps(
            run(episodes=args.episodes, seed=args.seed, output_dir=args.output_dir),
            indent=2,
        )
    )
