"""Non-interactive chapter 08 environment probe."""

from pytorch.common import make_cartpole


def run(seed: int = 0) -> tuple[int, ...]:
    env = make_cartpole(seed)
    try:
        observation, _ = env.reset(seed=seed)
        return tuple(observation.shape)
    finally:
        env.close()


if __name__ == "__main__":
    print(run())
