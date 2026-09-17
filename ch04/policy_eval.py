"""Iterative policy evaluation for the chapter GridWorld."""

from collections import defaultdict


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        V[state] = sum(
            p
            * (
                env.reward(state, a, env.next_state(state, a))
                + gamma * V[env.next_state(state, a)]
            )
            for a, p in pi[state].items()
        )
    return V


def policy_eval(pi, V, env, gamma=0.9, threshold=0.001, max_iterations=1000):
    for _ in range(max_iterations):
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        if max(abs(V[s] - old_V[s]) for s in V) < threshold:
            break
    return V


def uniform_policy(env):
    return defaultdict(lambda: {a: 1 / len(env.actions()) for a in env.actions()})
