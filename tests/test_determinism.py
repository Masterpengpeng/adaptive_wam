from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sim_state_manager import SimStateManager
from real_fastwam_adapter import FASTWAM_ROOT, LIBERO_ROOT, _add_project_import_paths


class LiberoDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _add_project_import_paths(
            {
                "fastwam_root": FASTWAM_ROOT,
                "libero_root": LIBERO_ROOT,
            }
        )
        import importlib

        cls.libero = importlib.import_module("libero.libero")
        cls.eval_helpers = importlib.import_module("eval_libero_single")
        suite_factory = cls.libero.benchmark.get_benchmark_dict().get("libero_spatial")
        if suite_factory is None:
            raise RuntimeError("LIBERO benchmark does not provide libero_spatial")
        task = suite_factory().get_task(0)
        cls.env, cls.task_description = cls.eval_helpers.get_libero_env(task, 224, 123)

    @classmethod
    def tearDownClass(cls) -> None:
        env = getattr(cls, "env", None)
        if env is not None:
            env.close()

    @staticmethod
    def _run_actions(env: Any, actions: np.ndarray) -> dict[str, Any]:
        observation = None
        for action in actions:
            observation, _, terminated, _ = env.step(action)
            if terminated:
                raise AssertionError("Determinism test terminated before completing actions")
        if observation is None:
            raise AssertionError("No observation returned for empty action sequence")
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }

    def test_libero_spatial_state_restore_replays_identically(self) -> None:
        random.seed(13579)
        np.random.seed(13579)
        observation = self.env.reset()
        self.assertIn("agentview_image", observation)
        self.assertIn("robot0_eye_in_hand_image", observation)

        action_rng = np.random.default_rng(20261014)
        actions = action_rng.uniform(-1.0, 1.0, size=(15, 7)).astype(np.float32)
        self._run_actions(self.env, actions[:10])
        snapshot = SimStateManager.save_state(self.env)
        qpos_s10 = np.asarray(self.env.sim.data.qpos).copy()
        self.assertIn("act", snapshot["simulation"]["data_fields"])
        self.assertIn("global_rng", snapshot)

        obs_a = self._run_actions(self.env, actions[10:])
        state_a = {
            "qpos": np.asarray(self.env.sim.data.qpos).copy(),
            "qvel": np.asarray(self.env.sim.data.qvel).copy(),
            "time": float(self.env.sim.data.time),
        }

        restored_observation = SimStateManager.restore_state(self.env, snapshot)
        np.testing.assert_allclose(self.env.sim.data.qpos, qpos_s10, atol=1e-10, rtol=0.0)
        self.assertIsNotNone(restored_observation)
        obs_b = self._run_actions(self.env, actions[10:])
        state_b = {
            "qpos": np.asarray(self.env.sim.data.qpos).copy(),
            "qvel": np.asarray(self.env.sim.data.qvel).copy(),
            "time": float(self.env.sim.data.time),
        }

        np.testing.assert_allclose(state_a["qpos"], state_b["qpos"], atol=1e-6)
        np.testing.assert_allclose(state_a["qvel"], state_b["qvel"], atol=1e-6)
        self.assertAlmostEqual(state_a["time"], state_b["time"], places=12)
        for camera_key in ("agentview_image", "robot0_eye_in_hand_image"):
            image_a = np.asarray(obs_a[camera_key])
            image_b = np.asarray(obs_b[camera_key])
            self.assertEqual(image_a.shape, (224, 224, 3))
            self.assertEqual(image_b.shape, image_a.shape)
            self.assertEqual(
                int(np.max(np.abs(image_a.astype(np.int16) - image_b.astype(np.int16)))),
                0,
                msg=f"Camera image differs after restore: {camera_key}",
            )


if __name__ == "__main__":
    unittest.main()
