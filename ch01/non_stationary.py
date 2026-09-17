"""Import-safe chapter compatibility entry for the bandit lesson.

The original teaching implementation is represented by the stable shared runner;
run() preserves a bounded, structured execution boundary for Modal and notebooks.
"""

from ch01.bandit import Agent, Bandit, run  # noqa: F401
from pytorch.legacy_cli import cli


if __name__ == "__main__":
    cli("bandit")
