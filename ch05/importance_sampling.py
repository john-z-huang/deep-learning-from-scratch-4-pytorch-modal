"""Import-safe chapter compatibility entry for the monte carlo lesson.

The original teaching implementation is represented by the stable shared runner;
run() preserves a bounded, structured execution boundary for Modal and notebooks.
"""

from ch05.mc_control import McAgent, greedy_probs, run  # noqa: F401
from pytorch.legacy_cli import cli


if __name__ == "__main__":
    cli("monte_carlo")
