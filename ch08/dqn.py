"""Chapter 08 DQN entry backed by the shared PyTorch implementation."""

from pytorch.dqn import DQNAgent, QNet, ReplayBuffer, main, run

__all__ = ["DQNAgent", "QNet", "ReplayBuffer", "main", "run"]


if __name__ == "__main__":
    main()
