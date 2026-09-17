"""Modal entry point for short, headless PyTorch reinforcement-learning runs."""

from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    .add_local_dir("ch07", remote_path="/root/ch07")
    .add_local_dir("ch08", remote_path="/root/ch08")
    .add_local_dir("ch09", remote_path="/root/ch09")
    .add_local_dir("notebooks", remote_path="/root/notebooks")
    .add_local_dir("notebook_exports", remote_path="/root/notebook_exports")
    .add_local_dir("pytorch", remote_path="/root/pytorch")
)


@dataclass(frozen=True)
class Route:
    """Explicit module/function route for one remote teaching experiment."""

    module: str
    function: str
    kind: str


ROUTES = {
    "ch01_avg": Route("ch01.avg", "run", "average"),
    "ch01_bandit_avg": Route("ch01.bandit_avg", "run", "bandit_avg"),
    "ch01_bandit": Route("ch01.bandit", "run", "artifact"),
    "ch01_non_stationary": Route("ch01.non_stationary", "run", "non_stationary"),
    "ch04_dp": Route("ch04.dp", "run", "dp_plain"),
    "ch04_dp_inplace": Route("ch04.dp_inplace", "run", "dp_plain"),
    "ch04_gridworld_play": Route("ch04.gridworld_play", "run", "gridworld"),
    "ch04_policy_eval": Route("ch04.policy_eval", "policy_eval", "policy_eval"),
    "ch04_policy_iter": Route("ch04.policy_iter", "run", "artifact"),
    "ch04_value_iter": Route("ch04.value_iter", "value_iter", "value_iter"),
    "ch05_dice": Route("ch05.dice", "run", "dice"),
    "ch05_importance_sampling": Route(
        "ch05.importance_sampling", "estimate", "importance_sampling"
    ),
    "ch05_mc_control": Route("ch05.mc_control", "run", "artifact"),
    "ch05_mc_control_offpolicy": Route(
        "ch05.mc_control_offpolicy", "run", "episodes"
    ),
    "ch05_mc_eval": Route("ch05.mc_eval", "run", "episodes"),
    "ch06_q_learning": Route("ch06.q_learning", "run", "artifact"),
    "ch06_q_learning_simple": Route("ch06.q_learning_simple", "run", "episodes"),
    "ch06_sarsa": Route("ch06.sarsa", "run", "episodes"),
    "ch06_sarsa_off_policy": Route("ch06.sarsa_off_policy", "run", "episodes"),
    "ch06_td_eval": Route("ch06.td_eval", "run", "episodes"),
    "ch07_dezero1": Route("ch07.dezero1", "run", "tensor"),
    "ch07_dezero2": Route("ch07.dezero2", "run", "steps"),
    "ch07_dezero3": Route("ch07.dezero3", "run", "steps_seed"),
    "ch07_dezero4": Route("ch07.dezero4", "run", "steps_seed"),
    "ch07_q_learning_nn": Route("ch07.q_learning_nn", "run", "steps_seed"),
    "ch08_dqn": Route("ch08.dqn", "run", "artifact"),
    "ch08_gym_play": Route("ch08.gym_play", "rollout", "rollout"),
    "ch08_replay_buffer": Route("ch08.replay_buffer", "run", "buffer"),
    "ch09_reinforce": Route("ch09.reinforce", "run", "artifact"),
    "ch09_simple_pg": Route("ch09.simple_pg", "run", "artifact"),
    "ch09_actor_critic": Route("ch09.actor_critic", "run", "artifact"),
    "notebook_export_07": Route(
        "notebook_exports.07_neural_networks", "run", "notebook_epochs"
    ),
    "notebook_export_08": Route(
        "notebook_exports.08_dqn", "run", "notebook_dqn"
    ),
    "notebook_export_09": Route(
        "notebook_exports.09_policy_gradient", "run", "notebook_pg"
    ),
    "notebook_01_bandit": Route("notebooks.01_bandit", "run", "nb_bandit"),
    "notebook_04_dynamic_programming": Route(
        "notebooks.04_dynamic_programming", "run", "nb_dp"
    ),
    "notebook_05_montecarlo": Route(
        "notebooks.05_montecarlo", "run", "nb_mc"
    ),
    "notebook_06_temporal_difference": Route(
        "notebooks.06_temporal_difference", "run", "nb_td"
    ),
    "notebook_07_neural_networks": Route(
        "notebooks.07_neural_networks", "fit", "nb_nn"
    ),
    "notebook_08_dqn": Route("notebooks.08_dqn", "run", "nb_dqn"),
    "notebook_09_policy_gradient": Route(
        "notebooks.09_policy_gradient", "run", "nb_pg"
    ),
    "neural_networks": Route(
        "notebook_exports.07_neural_networks", "run", "notebook_epochs"
    ),
    # Keep the original short names while routing them to chapter-owned code.
    "bandit": Route("ch01.bandit", "run", "artifact"),
    "dynamic_programming": Route("ch04.policy_iter", "run", "artifact"),
    "monte_carlo": Route("ch05.mc_control", "run", "artifact"),
    "temporal_difference": Route("ch06.q_learning", "run", "artifact"),
    "dqn": Route("ch08.dqn", "run", "artifact"),
    "reinforce": Route("ch09.reinforce", "run", "artifact"),
    "policy_gradient": Route("ch09.simple_pg", "run", "artifact"),
    "actor_critic": Route("ch09.actor_critic", "run", "artifact"),
}


def _json_safe(value: Any) -> Any:
    """Convert chapter results to values suitable for a JSON artifact."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_result(result: dict[str, Any], output_dir: str) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path("/outputs") / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "modal-result.json"
    result_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result_path


def _run_experiment(
    experiment: str,
    *,
    episodes: int,
    seed: int,
    device: str,
    output_dir: str,
    checkpoint_interval: int,
    max_steps: int,
) -> dict[str, Any]:
    try:
        route = ROUTES[experiment]
    except KeyError as exc:
        choices = ", ".join(sorted(ROUTES))
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from {choices}"
        ) from exc
    module = importlib.import_module(route.module)
    handler = getattr(module, route.function)
    deep_learning = route.kind in {
        "artifact",
        "tensor",
        "steps",
        "steps_seed",
        "notebook_epochs",
        "notebook_dqn",
        "notebook_pg",
        "nb_nn",
        "nb_dqn",
        "nb_pg",
    }
    algorithm_device = device if deep_learning else "cpu"
    print(
        f"remote_module={module.__name__} remote_file={module.__file__} "
        f"route={experiment} requested_device={device} "
        f"algorithm_device={algorithm_device}"
    )
    if route.kind == "average":
        result = handler(samples=max_steps, seed=seed)
    elif route.kind == "bandit_avg":
        result = handler(runs=episodes, steps=max_steps, seed=seed)
    elif route.kind == "non_stationary":
        result = handler(steps=episodes * max_steps, seed=seed)
    elif route.kind == "artifact":
        kwargs = dict(
            episodes=episodes,
            seed=seed,
            device=algorithm_device,
            output_dir=output_dir,
            checkpoint_interval=checkpoint_interval,
        )
        if route.module.startswith(("ch08.", "ch09.")):
            kwargs["max_steps"] = max_steps
        result = handler(**kwargs)
    elif route.kind == "dice":
        result = handler(trials=episodes * max_steps, seed=seed)
    elif route.kind == "importance_sampling":
        result = handler(trials=episodes * max_steps, seed=seed)
    elif route.kind == "episodes":
        result = handler(episodes=episodes, seed=seed)
    elif route.kind == "dp_plain":
        result = handler()
    elif route.kind == "gridworld":
        result = handler(seed=seed)
    elif route.kind == "tensor":
        result = {"value": handler(device=algorithm_device)}
    elif route.kind == "steps":
        result = {
            "loss": handler(
                steps=max_steps, device=algorithm_device
            )
        }
    elif route.kind == "steps_seed":
        result = {
            "loss": handler(
                steps=max_steps, seed=seed, device=algorithm_device
            )
        }
    elif route.kind == "rollout":
        result = handler(seed=seed, steps=max_steps)
    elif route.kind == "buffer":
        result = handler(capacity=max(1, min(max_steps, 10_000)))
    elif route.kind == "policy_eval":
        from common.gridworld import GridWorld

        env = GridWorld()
        policy = module.uniform_policy(env)
        values = module.policy_eval(
            policy, defaultdict(float), env, max_iterations=max_steps
        )
        result = {"values": values}
    elif route.kind == "value_iter":
        from common.gridworld import GridWorld

        result = {
            "values": handler(
                defaultdict(float), GridWorld(), max_iterations=max_steps
            )
        }
    elif route.kind == "notebook_epochs":
        result = handler(
            epochs=max_steps, seed=seed, device=algorithm_device, output_dir=output_dir
        )
    elif route.kind in {"notebook_dqn", "notebook_pg"}:
        result = handler(
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            device=algorithm_device,
            output_dir=output_dir,
        )
    elif route.kind == "nb_bandit":
        result = handler(steps=max_steps, seed=seed)
    elif route.kind == "nb_dp":
        result = handler()
    elif route.kind in {"nb_mc", "nb_td"}:
        result = handler(episodes=episodes, seed=seed)
    elif route.kind == "nb_nn":
        result = handler(
            epochs=max_steps,
            seed=seed,
            device=algorithm_device,
            output_dir=output_dir,
        )
    elif route.kind in {"nb_dqn", "nb_pg"}:
        result = handler(
            steps=max_steps,
            seed=seed,
            device=algorithm_device,
            output_dir=output_dir,
        )
    else:
        raise RuntimeError(f"unhandled route kind {route.kind!r}")
    result = dict(result)
    result.update(
        {
            "modal_route": experiment,
            "remote_module": module.__name__,
            "remote_file": str(module.__file__),
            "requested_device": device,
            "algorithm_device": algorithm_device,
        }
    )
    result_path = Path(output_dir)
    if not result_path.is_absolute():
        result_path = Path("/outputs") / result_path
    result["modal_result_path"] = str(result_path / "modal-result.json")
    _write_result(result, output_dir)
    print(f"modal_result_path={result['modal_result_path']}")
    print(f"module_file_exists={Path(module.__file__).is_file()}")
    return result


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
    max_steps: int = 500,
) -> dict[str, Any]:
    result = _run_experiment(
        experiment,
        episodes=episodes,
        seed=seed,
        device="cpu",
        output_dir=str(Path("/outputs") / output_dir),
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
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
    max_steps: int = 500,
) -> dict[str, Any]:
    result = _run_experiment(
        experiment,
        episodes=episodes,
        seed=seed,
        device="cuda",
        output_dir=str(Path("/outputs") / output_dir),
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
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
    max_steps: int = 500,
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
        max_steps=max_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
