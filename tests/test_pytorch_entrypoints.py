"""Short, non-interactive regression checks for the four PyTorch entry points."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pytorch import actor_critic, dqn, reinforce, simple_pg


class EntrypointTest(unittest.TestCase):
    def test_short_cpu_runs_write_all_artifacts(self) -> None:
        runners = [dqn.run, reinforce.run, simple_pg.run, actor_critic.run]
        with tempfile.TemporaryDirectory() as temporary:
            for runner in runners:
                output_dir = Path(temporary) / runner.__module__.split(".")[-1]
                result = runner(
                    episodes=1,
                    seed=7,
                    device="cpu",
                    output_dir=output_dir,
                    checkpoint_interval=1,
                    max_steps=5,
                )
                self.assertEqual(result["device"], "cpu")
                self.assertEqual(len(result["rewards"]), 1)
                self.assertTrue((output_dir / "metrics.json").is_file())
                self.assertTrue((output_dir / "rewards.png").is_file())
                self.assertTrue((output_dir / "checkpoint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
