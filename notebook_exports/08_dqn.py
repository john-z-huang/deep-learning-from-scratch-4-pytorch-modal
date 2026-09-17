"""Notebook 08 teaching entrypoint backed by the reusable PyTorch DQN."""

import argparse
import json

from pytorch.dqn import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 08 PyTorch DQN smoke")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(
        episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
