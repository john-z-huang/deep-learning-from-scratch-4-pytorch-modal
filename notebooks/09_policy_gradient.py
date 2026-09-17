"""Executable chapter 09 notebook wrapper for PyTorch policy-gradient runs."""

import argparse
import json

from pytorch.actor_critic import run as run_actor_critic
from pytorch.reinforce import run as run_reinforce
from pytorch.simple_pg import run as run_policy_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("reinforce", "policy_gradient", "actor_critic"), default="reinforce")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", default="notebook-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    args = parser.parse_args()
    runner = {"reinforce": run_reinforce, "policy_gradient": run_policy_gradient, "actor_critic": run_actor_critic}[args.experiment]
    device = "cuda" if args.device == "gpu" else "cpu"
    print(json.dumps(runner(episodes=args.episodes, seed=args.seed, device=device, output_dir=args.output_dir, checkpoint_interval=args.checkpoint_interval), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
