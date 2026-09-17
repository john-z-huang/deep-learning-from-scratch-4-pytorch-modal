"""Device-aware DQN CartPole experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch.common import (
    device_metadata,
    make_cartpole,
    resolve_device,
    save_training_artifacts,
    seed_everything,
)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self, device: torch.device):
        data = random.sample(self.buffer, self.batch_size)
        states = torch.as_tensor(np.stack([item[0] for item in data]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([item[1] for item in data], dtype=torch.int64, device=device)
        rewards = torch.as_tensor([item[2] for item in data], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([item[3] for item in data]), dtype=torch.float32, device=device)
        dones = torch.as_tensor([item[4] for item in data], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


class QNet(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    def __init__(self, observation_size: int, action_size: int, device: torch.device):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.action_size = action_size
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_size=10_000, batch_size=32)
        self.qnet = QNet(observation_size, action_size).to(device)
        self.qnet_target = copy.deepcopy(self.qnet).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(state_tensor).argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done) -> float | None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(self.device)
        q_values = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.qnet_target(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def sync_qnet(self) -> None:
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def run(
    episodes: int = 300,
    *,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    output_dir: str | Path | None = None,
    checkpoint_interval: int = 20,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Train DQN without side effects at import time and return JSON-safe results."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    seed_everything(seed)
    resolved_device = resolve_device(device)
    env = make_cartpole(seed)
    observation_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = DQNAgent(observation_size, action_size, resolved_device)
    rewards: list[float] = []
    losses: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            for _ in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                loss = agent.update(state, action, reward, next_state, done)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_reward += float(reward)
                if done:
                    break
            if episode % max(1, checkpoint_interval) == 0:
                agent.sync_qnet()
            rewards.append(total_reward)
            print(f"episode={episode} reward={total_reward:.1f}")
    finally:
        env.close()
    metadata = {"experiment": "dqn", "seed": seed, **device_metadata(resolved_device)}
    metadata["mean_loss"] = float(np.mean(losses)) if losses else None
    checkpoint = {
        "experiment": "dqn",
        "model_state_dict": agent.qnet.state_dict(),
        "target_state_dict": agent.qnet_target.state_dict(),
        "seed": seed,
    }
    paths = (
        save_training_artifacts(rewards, output_dir, metadata=metadata, checkpoint=checkpoint)
        if output_dir is not None
        else {}
    )
    return {**metadata, "rewards": rewards, "artifacts": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="dqn-runs")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))
