"""Notebook 07 teaching entrypoint backed by the reusable PyTorch module."""

import argparse
import json

from pytorch.neural_networks import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Notebook 07 PyTorch regression smoke")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(epochs=args.epochs, device=args.device, output_dir=args.output_dir)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
