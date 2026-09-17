"""Q-learning: readable tabular update rule and bounded chapter runner hooks."""

from collections import defaultdict
import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    best = int(np.argmax([Q[(state, a)] for a in range(action_size)]))
    base = epsilon / action_size
    return {a: base + (1 - epsilon if a == best else 0) for a in range(action_size)}


class QLearningAgent:
    def __init__(self):
        self.gamma, self.alpha, self.epsilon, self.action_size = 0.9, 0.8, 0.1, 4
        self.Q = defaultdict(float)
        self.b = defaultdict(lambda: {a: 0.25 for a in range(4)})

    def get_action(self, state):
        probs = self.b[state]
        return int(np.random.choice(list(probs), p=list(probs.values())))

    def update(self, state, action, reward, next_state, done):
        next_q = (
            0 if done else max(self.Q[(next_state, a)] for a in range(self.action_size))
        )
        self.Q[(state, action)] += self.alpha * (
            reward + self.gamma * next_q - self.Q[(state, action)]
        )
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


def run(
    *,
    episodes=1,
    seed=0,
    device="cpu",
    output_dir="temporal-difference-runs",
    checkpoint_interval=1,
):
    from common.gridworld import GridWorld
    from pytorch.common import save_training_artifacts

    np.random.seed(seed)
    env = GridWorld()
    agent = QLearningAgent()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(100):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total += reward
            if done:
                break
            state = next_state
        rewards.append(total)
    metadata = {
        "schema_version": 1,
        "experiment": "temporal_difference",
        "implementation": "chapter-ch06",
        "seed": seed,
        "requested_device": device,
        "device": "cpu",
    }
    paths = save_training_artifacts(
        rewards,
        output_dir,
        metadata=metadata,
        checkpoint={
            "experiment": "temporal_difference",
            "seed": seed,
            "Q": dict(agent.Q),
        },
    )
    return {
        **metadata,
        "episode_count": episodes,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "output_paths": paths,
    }
