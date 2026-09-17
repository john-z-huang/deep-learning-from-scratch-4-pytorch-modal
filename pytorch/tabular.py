"""Short, reproducible NumPy tabular-RL runners for chapters 01 and 04-06."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from pytorch.common import save_training_artifacts


class GridWorld:
    """Small chapter-compatible GridWorld without rendering or global state."""

    action_moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    start = (2, 0)
    goal = (0, 3)
    wall = (1, 1)

    @staticmethod
    def next_state(state: tuple[int, int], action: int) -> tuple[int, int]:
        y, x = state
        dy, dx = GridWorld.action_moves[action]
        candidate = (y + dy, x + dx)
        if (
            not (0 <= candidate[0] < 3 and 0 <= candidate[1] < 4)
            or candidate == GridWorld.wall
        ):
            return state
        return candidate

    @staticmethod
    def step(
        state: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float, bool]:
        next_state = GridWorld.next_state(state, action)
        return (
            next_state,
            1.0 if next_state == GridWorld.goal else 0.0,
            next_state == GridWorld.goal,
        )


def _episode(
    policy: Callable[[tuple[int, int]], int],
    update: Callable[[list[tuple[tuple[int, int], int, float]], bool], None]
    | None = None,
    *,
    max_steps: int = 100,
) -> float:
    state = GridWorld.start
    trajectory: list[tuple[tuple[int, int], int, float]] = []
    total = 0.0
    for _ in range(max_steps):
        action = int(policy(state))
        next_state, reward, done = GridWorld.step(state, action)
        trajectory.append((state, action, reward))
        total += reward
        state = next_state
        if done:
            break
    if update is not None:
        update(trajectory, state == GridWorld.goal)
    return total


def run_bandit(
    *, episodes: int, seed: int, device: str, output_dir: str | Path
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rates = rng.random(10)
    values = np.zeros(10, dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)
    rewards: list[float] = []
    for _ in range(episodes):
        episode_reward = 0.0
        for _ in range(20):
            action = (
                int(rng.integers(10)) if rng.random() < 0.1 else int(np.argmax(values))
            )
            reward = float(rng.random() < rates[action])
            counts[action] += 1
            values[action] += (reward - values[action]) / counts[action]
            episode_reward += reward
        rewards.append(episode_reward)
    return _save(
        "bandit", rewards, seed, device, output_dir, {"arms": 10, "horizon": 20}
    )


def _greedy_policy(
    values: dict[tuple[tuple[int, int], int], float],
    rng: np.random.Generator,
    epsilon: float,
) -> Callable[[tuple[int, int]], int]:
    def policy(state: tuple[int, int]) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(4))
        qs = np.asarray([values[(state, action)] for action in range(4)])
        best = np.flatnonzero(qs == qs.max())
        return int(rng.choice(best))

    return policy


def run_dynamic_programming(
    *, episodes: int, seed: int, device: str, output_dir: str | Path
) -> dict[str, object]:
    values = {state: 0.0 for state in _states()}
    for _ in range(100):
        updated = dict(values)
        for state in _states():
            if state == GridWorld.goal:
                continue
            updated[state] = max(
                GridWorld.step(state, action)[1]
                + 0.9 * values[GridWorld.step(state, action)[0]]
                for action in range(4)
            )
        if max(abs(updated[s] - values[s]) for s in values) < 1e-5:
            values = updated
            break
        values = updated
    rewards = [_evaluate_greedy(values, max_steps=100) for _ in range(episodes)]
    return _save(
        "dynamic_programming",
        rewards,
        seed,
        device,
        output_dir,
        {"gamma": 0.9, "iterations": 100},
    )


def run_monte_carlo(
    *, episodes: int, seed: int, device: str, output_dir: str | Path
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    q: dict[tuple[tuple[int, int], int], float] = defaultdict(float)
    counts: dict[tuple[tuple[int, int], int], int] = defaultdict(int)

    def update(trajectory: list[tuple[tuple[int, int], int, float]], _: bool) -> None:
        returns = 0.0
        for state, action, reward in reversed(trajectory):
            returns = reward + 0.9 * returns
            key = (state, action)
            counts[key] += 1
            q[key] += (returns - q[key]) / counts[key]

    rewards = []
    for _ in range(episodes):
        rewards.append(_episode(_greedy_policy(q, rng, 0.1), update))
    return _save(
        "monte_carlo", rewards, seed, device, output_dir, {"gamma": 0.9, "epsilon": 0.1}
    )


def run_temporal_difference(
    *, episodes: int, seed: int, device: str, output_dir: str | Path
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    q: dict[tuple[tuple[int, int], int], float] = defaultdict(float)

    def update(trajectory: list[tuple[tuple[int, int], int, float]], _: bool) -> None:
        for index, (state, action, reward) in enumerate(trajectory):
            next_state = (
                trajectory[index + 1][0]
                if index + 1 < len(trajectory)
                else GridWorld.goal
            )
            next_q = max(q[(next_state, next_action)] for next_action in range(4))
            q[(state, action)] += 0.1 * (reward + 0.9 * next_q - q[(state, action)])

    rewards = []
    for _ in range(episodes):
        rewards.append(_episode(_greedy_policy(q, rng, 0.1), update))
    return _save(
        "temporal_difference",
        rewards,
        seed,
        device,
        output_dir,
        {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1},
    )


def _states() -> list[tuple[int, int]]:
    return [(y, x) for y in range(3) for x in range(4) if (y, x) != GridWorld.wall]


def _evaluate_greedy(values: dict[tuple[int, int], float], *, max_steps: int) -> float:
    state = GridWorld.start
    for _ in range(max_steps):
        action_values = [
            GridWorld.step(state, action)[1]
            + 0.9 * values[GridWorld.step(state, action)[0]]
            for action in range(4)
        ]
        state, reward, done = GridWorld.step(state, int(np.argmax(action_values)))
        if done:
            return reward
    return 0.0


def _save(
    experiment: str,
    rewards: list[float],
    seed: int,
    device: str,
    output_dir: str | Path,
    extra: dict[str, object],
) -> dict[str, object]:
    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "implementation": "numpy-tabular",
        "seed": seed,
        "requested_device": device,
        "device": "cpu",
        **extra,
    }
    paths = save_training_artifacts(
        rewards,
        output_dir,
        metadata=metadata,
        checkpoint={"experiment": experiment, "seed": seed},
    )
    return {
        **metadata,
        "episode_count": len(rewards),
        "rewards": rewards,
        "mean_reward": float(np.mean(rewards)),
        "output_paths": paths,
    }


RUNNERS = {
    "bandit": run_bandit,
    "dynamic_programming": run_dynamic_programming,
    "monte_carlo": run_monte_carlo,
    "temporal_difference": run_temporal_difference,
}


def run(
    experiment: str,
    *,
    episodes: int,
    seed: int,
    device: str = "cpu",
    output_dir: str | Path = "tabular-runs",
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    # Chapter files own the algorithm; this module remains the stable dispatcher.
    if experiment == "bandit":
        from ch01.bandit import run as chapter_run
    elif experiment == "dynamic_programming":
        from ch04.policy_iter import run as chapter_run
    elif experiment == "monte_carlo":
        from ch05.mc_control import run as chapter_run
    elif experiment == "temporal_difference":
        from ch06.q_learning import run as chapter_run
    else:
        raise ValueError(f"unknown tabular experiment {experiment!r}")
    return chapter_run(
        episodes=episodes,
        seed=seed,
        device=device,
        output_dir=output_dir,
        checkpoint_interval=1,
    )
    try:
        runner = RUNNERS[experiment]
    except KeyError as exc:
        raise ValueError(
            f"unknown tabular experiment {experiment!r}; choose from {', '.join(sorted(RUNNERS))}"
        ) from exc
    return runner(episodes=episodes, seed=seed, device=device, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(RUNNERS), default="bandit")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", default="tabular-runs")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
