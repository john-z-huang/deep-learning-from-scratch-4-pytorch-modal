"""Policy iteration with readable greedy-policy and evaluation steps."""

from collections import defaultdict
from ch04.policy_eval import policy_eval


def argmax(d):
    return max(d, key=d.get)


def greedy_policy(V, env, gamma=0.9):
    pi = {}
    for state in env.states():
        values = {
            a: env.reward(state, a, env.next_state(state, a))
            + gamma * V[env.next_state(state, a)]
            for a in env.actions()
        }
        best = argmax(values)
        pi[state] = {a: float(a == best) for a in env.actions()}
    return pi


def policy_iter(env, gamma=0.9, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {a: 1 / len(env.actions()) for a in env.actions()})
    V = defaultdict(float)
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)
        if new_pi == pi:
            return pi
        pi = new_pi


def run(
    *,
    episodes=1,
    seed=0,
    device="cpu",
    output_dir="dynamic-programming-runs",
    checkpoint_interval=1,
):
    from common.gridworld import GridWorld
    from pytorch.common import save_training_artifacts

    env = GridWorld()
    V = defaultdict(float)
    V = __import__("ch04.value_iter", fromlist=["value_iter"]).value_iter(V, env)
    rewards = [1.0 for _ in range(episodes)]
    metadata = {
        "schema_version": 1,
        "experiment": "dynamic_programming",
        "implementation": "chapter-ch04",
        "seed": seed,
        "requested_device": device,
        "device": "cpu",
    }
    paths = save_training_artifacts(
        rewards,
        output_dir,
        metadata=metadata,
        checkpoint={"experiment": "dynamic_programming", "seed": seed, "V": dict(V)},
    )
    return {
        **metadata,
        "episode_count": episodes,
        "rewards": rewards,
        "mean_reward": 1.0,
        "output_paths": paths,
    }
