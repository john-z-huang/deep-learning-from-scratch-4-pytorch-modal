"""Chapter 09 policy-gradient entry backed by the PyTorch implementation."""

from pytorch.simple_pg import Policy, main, run

__all__ = ["Policy", "main", "run"]


if __name__ == "__main__":
    main()
