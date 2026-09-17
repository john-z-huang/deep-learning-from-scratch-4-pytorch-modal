"""Monte Carlo estimation of the expected sum of dice."""

import numpy as np


def sample(dices=2, rng=None):
    rng = rng or np.random.default_rng()
    return int(rng.integers(1, 7, size=dices).sum())


def run(*, trials=100, dices=2, seed=0):
    rng = np.random.default_rng(seed)
    estimate = 0.0
    for count in range(1, trials + 1):
        estimate += (sample(dices, rng) - estimate) / count
    return {"trials": trials, "estimate": float(estimate), "seed": seed}


if __name__ == "__main__":
    print(run())
