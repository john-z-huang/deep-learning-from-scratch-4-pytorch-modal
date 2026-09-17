"""Monte Carlo control: readable agent update logic and bounded execution."""

from collections import defaultdict
import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    best = int(np.argmax(qs))
    base = epsilon / action_size
    return {
        action: base + (1 - epsilon if action == best else 0)
        for action in range(action_size)
    }


class McAgent:
    def __init__(self):
        self.gamma, self.epsilon, self.alpha, self.action_size = 0.9, 0.1, 0.1, 4
        self.pi = defaultdict(lambda: {a: 0.25 for a in range(4)})
        self.Q, self.memory = defaultdict(float), []

    def get_action(self, state):
        probs = self.pi[state]
        return int(np.random.choice(list(probs), p=list(probs.values())))

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for state, action, reward in reversed(self.memory):
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += self.alpha * (G - self.Q[key])
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


def run(
    *,
    episodes=1,
    seed=0,
    device="cpu",
    output_dir="monte-carlo-runs",
    checkpoint_interval=1,
):
    from common.gridworld import GridWorld
    from pytorch.common import save_training_artifacts

    np.random.seed(seed)
    env = GridWorld()
    agent = McAgent()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        agent.reset()
        total = 0.0
        for _ in range(100):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            total += reward
            if done:
                break
            state = next_state
        agent.update()
        rewards.append(total)
    metadata = {
        "schema_version": 1,
        "experiment": "monte_carlo",
        "implementation": "chapter-ch05",
        "seed": seed,
        "requested_device": device,
        "device": "cpu",
    }
    paths = save_training_artifacts(
        rewards,
        output_dir,
        metadata=metadata,
        checkpoint={"experiment": "monte_carlo", "seed": seed, "Q": dict(agent.Q)},
    )
    return {
        **metadata,
        "episode_count": episodes,
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "output_paths": paths,
    }
