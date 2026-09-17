"""Readable, device-aware DQN implementation for the chapter 08 lesson."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ch08.replay_buffer import ReplayBuffer
from pytorch.common import (
    device_metadata,
    make_cartpole,
    resolve_device,
    save_training_artifacts,
    seed_everything,
)


class QNet(nn.Module):
    """Small multilayer perceptron that predicts all discrete action values."""

    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class DQNAgent:
    """Epsilon-greedy DQN agent with a target network and replay memory."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        device: torch.device,
        *,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        self.action_size = action_size
        self.gamma = 0.98
        self.epsilon = 0.1
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.memory = ReplayBuffer(10_000)
        self.batch_size = batch_size
        self.qnet = QNet(observation_size, action_size).to(device)
        self.target = QNet(observation_size, action_size).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=5e-4)

    def get_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(state_tensor).argmax(dim=1).item())

    def update(
        self, state, action: int, reward: float, next_state, done: bool
    ) -> float | None:
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size, self.rng)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )
        current = self.qnet(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).max(dim=1).values
            target = rewards + (1 - dones) * self.gamma * future
        loss = nn.functional.mse_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def sync_target(self) -> None:
        """Copy online parameters to the target network."""

        self.target.load_state_dict(self.qnet.state_dict())


def run(
    episodes: int = 1,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    output_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    max_steps: int = 500,
) -> dict[str, object]:
    """Run a bounded CartPole DQN experiment and optionally save artifacts."""

    if episodes < 1 or max_steps < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, max_steps, and checkpoint_interval must be positive")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    env = make_cartpole(seed)
    agent = DQNAgent(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        resolved_device,
        seed=seed,
    )
    rewards = []
    losses = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total = 0.0
            for _ in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                loss = agent.update(state, action, reward, next_state, done)
                if loss is not None:
                    losses.append(loss)
                total += float(reward)
                state = next_state
                if done:
                    break
            if episode % checkpoint_interval == 0:
                agent.sync_target()
            rewards.append(total)
    finally:
        env.close()
    metadata = {
        "experiment": "chapter08_dqn",
        "seed": seed,
        **device_metadata(resolved_device),
        "mean_loss": float(np.mean(losses)) if losses else None,
    }
    artifacts = (
        save_training_artifacts(
            rewards,
            output_dir,
            metadata=metadata,
            checkpoint={
                "experiment": "chapter08_dqn",
                "model_state_dict": agent.qnet.state_dict(),
            },
        )
        if output_dir is not None
        else {}
    )
    return {**metadata, "rewards": rewards, "artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="chapter08-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
