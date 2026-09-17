"""Modal entry point for short, headless PyTorch reinforcement-learning runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import modal


APP_NAME = "deep-learning-from-scratch-4-pytorch"
VOLUME_NAME = "deep-learning-from-scratch-4-results"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gymnasium[classic-control]>=1.0,<2",
        "matplotlib>=3.9,<4",
        "numpy>=1.26,<3",
        "torch>=2.5,<3",
    )
    .env({"MPLBACKEND": "Agg"})
    .add_local_dir("common", remote_path="/root/common")
    .add_local_dir("ch01", remote_path="/root/ch01")
    .add_local_dir("ch04", remote_path="/root/ch04")
    .add_local_dir("ch05", remote_path="/root/ch05")
    .add_local_dir("ch06", remote_path="/root/ch06")
    .add_local_dir("pytorch", remote_path="/root/pytorch")
)


def _run_experiment(
    experiment: str,
    *,
    episodes: int,
    seed: int,
    device: str,
    output_dir: str,
    checkpoint_interval: int,
) -> dict[str, Any]:
    from pytorch.actor_critic import run as run_actor_critic
    from pytorch.dqn import run as run_dqn
    from pytorch.reinforce import run as run_reinforce
    from pytorch.simple_pg import run as run_policy_gradient
    from pytorch.tabular import run as run_tabular
    from pytorch.neural_networks import run as run_neural_networks

    runners: dict[str, Callable[..., dict[str, Any]]] = {
        "dqn": run_dqn,
        "reinforce": run_reinforce,
        "policy_gradient": run_policy_gradient,
        "actor_critic": run_actor_critic,
        "neural_networks": run_neural_networks,
        "bandit": run_tabular,
        "dynamic_programming": run_tabular,
        "monte_carlo": run_tabular,
        "temporal_difference": run_tabular,
    }
    try:
        runner = runners[experiment]
    except KeyError as exc:
        choices = ", ".join(sorted(runners))
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from {choices}"
        ) from exc
    if experiment in {
        "bandit",
        "dynamic_programming",
        "monte_carlo",
        "temporal_difference",
    }:
        return runner(
            experiment,
            episodes=episodes,
            seed=seed,
            device=device,
            output_dir=output_dir,
        )
    if experiment == "neural_networks":
        return runner(epochs=episodes, seed=seed, device=device, output_dir=output_dir)
    return runner(
        episodes=episodes,
        seed=seed,
        device=device,
        output_dir=output_dir,
        checkpoint_interval=checkpoint_interval,
    )


@app.function(
    image=base_image,
    cpu=1,
    timeout=60 * 30,
    volumes={"/outputs": results_volume},
)
def cpu_experiment(
    experiment: str = "dqn",
    episodes: int = 1,
    seed: int = 0,
    output_dir: str = "modal-runs",
    checkpoint_interval: int = 1,
) -> dict[str, Any]:
    result = _run_experiment(
        experiment,
        episodes=episodes,
        seed=seed,
        device="cpu",
        output_dir=str(Path("/outputs") / output_dir),
        checkpoint_interval=checkpoint_interval,
    )
    results_volume.commit()
    return result


@app.function(
    image=base_image,
    gpu="T4",
    timeout=60 * 30,
    volumes={"/outputs": results_volume},
)
def gpu_experiment(
    experiment: str = "dqn",
    episodes: int = 1,
    seed: int = 0,
    output_dir: str = "modal-runs",
    checkpoint_interval: int = 1,
) -> dict[str, Any]:
    result = _run_experiment(
        experiment,
        episodes=episodes,
        seed=seed,
        device="cuda",
        output_dir=str(Path("/outputs") / output_dir),
        checkpoint_interval=checkpoint_interval,
    )
    results_volume.commit()
    return result


@app.local_entrypoint()
def main(
    experiment: str = "dqn",
    device: str = "cpu",
    episodes: int = 1,
    seed: int = 0,
    output_dir: str = "modal-runs",
    checkpoint_interval: int = 1,
) -> None:
    """Run a short experiment with ``modal run modal_app.py``."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    function = gpu_experiment if device == "gpu" else cpu_experiment
    result = function.remote(
        experiment=experiment,
        episodes=episodes,
        seed=seed,
        output_dir=output_dir,
        checkpoint_interval=checkpoint_interval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
