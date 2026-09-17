"""Headless GridWorld value initialization used by the chapter demonstration."""

from __future__ import annotations

import json
import numpy as np
from common.gridworld import GridWorld


def random_values(seed: int = 0) -> dict[tuple[int, int], float]:
    """Create deterministic values for every GridWorld coordinate."""

    rng = np.random.default_rng(seed)
    return {state: float(rng.standard_normal()) for state in GridWorld().states()}


def run(*, seed: int = 0) -> dict[str, object]:
    """Return a wall-free value table without opening a renderer."""

    values = random_values(seed)
    values.pop(GridWorld().wall_state, None)
    return {"seed": seed, "state_count": len(values), "values": values}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
