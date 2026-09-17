"""Notebook export: policy/value iteration remain visible chapter algorithms."""

from ch04.policy_eval import eval_onestep, policy_eval, uniform_policy  # noqa: F401
from ch04.policy_iter import greedy_policy, policy_iter, run  # noqa: F401
from ch04.value_iter import value_iter, value_iter_onestep  # noqa: F401


if __name__ == "__main__":
    print(run(episodes=1, seed=0, output_dir="/tmp/notebook-04-runs"))
