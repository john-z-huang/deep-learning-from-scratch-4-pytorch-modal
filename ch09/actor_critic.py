"""Chapter 09 actor-critic entry backed by the PyTorch implementation."""

from pytorch.actor_critic import PolicyNet, ValueNet, main, run

__all__ = ["PolicyNet", "ValueNet", "main", "run"]


if __name__ == "__main__":
    main()
