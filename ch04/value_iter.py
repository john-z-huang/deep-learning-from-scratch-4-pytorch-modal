"""Value iteration with bounded convergence and no rendering side effect."""



def value_iter_onestep(V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        V[state] = max(
            env.reward(state, a, env.next_state(state, a))
            + gamma * V[env.next_state(state, a)]
            for a in env.actions()
        )
    return V


def value_iter(
    V, env, gamma=0.9, threshold=0.001, is_render=False, max_iterations=1000
):
    for _ in range(max_iterations):
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)
        if max(abs(V[s] - old_V[s]) for s in V) < threshold:
            break
    return V
