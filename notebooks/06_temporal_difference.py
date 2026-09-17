"""Notebook export: tabular TD/Q-learning update remains readable."""

from ch06.q_learning import QLearningAgent, greedy_probs, run  # noqa: F401


if __name__ == "__main__":
    print(run(episodes=1, seed=0, output_dir="/tmp/notebook-06-runs"))
