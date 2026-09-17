"""Import-safe chapter compatibility entry for the dynamic programming lesson.

The original teaching implementation is represented by the stable shared runner;
run() preserves a bounded, structured execution boundary for Modal and notebooks.
"""

from ch04.policy_eval import eval_onestep, policy_eval, uniform_policy  # noqa: F401
from ch04.policy_iter import policy_iter  # noqa: F401
from ch04.value_iter import value_iter, value_iter_onestep  # noqa: F401
from pytorch.legacy_cli import cli


if __name__ == "__main__":
    cli("dynamic_programming")
