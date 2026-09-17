"""Standalone replay-memory data structure used by the chapter 08 lesson."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One transition sampled by a value-based agent."""

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with reproducible NumPy sampling."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self._items.append(Transition(state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> list[Transition]:
        """Sample without replacement, preserving transition structure."""

        if not 1 <= batch_size <= len(self._items):
            raise ValueError("batch_size must be between 1 and the buffer length")
        generator = rng or np.random.default_rng()
        indices = generator.choice(len(self._items), size=batch_size, replace=False)
        items = list(self._items)
        return [items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)


def run(*, capacity: int = 3) -> dict[str, object]:
    """Demonstrate FIFO eviction and a structured sample."""

    buffer = ReplayBuffer(capacity)
    for state in range(capacity + 1):
        buffer.add(state, 0, float(state), state + 1, False)
    sample = buffer.sample(min(2, len(buffer)), np.random.default_rng(0))
    return {"size": len(buffer), "states": [item.state for item in sample]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
