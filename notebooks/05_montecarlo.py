"""Notebook export: Monte Carlo agent and return update remain readable."""

from ch05.mc_control import McAgent, greedy_probs, run  # noqa: F401


if __name__ == "__main__":
    print(run(episodes=1, seed=0, output_dir="/tmp/notebook-05-runs"))
