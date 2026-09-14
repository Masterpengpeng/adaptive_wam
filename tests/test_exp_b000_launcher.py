import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.launch_exp_b000 import (
    DEFAULT_CONFIG,
    build_environment,
    build_manager_command,
    hash_file,
    iter_manifest_files,
    load_config,
    load_manifest,
    validate_demo_config,
    validate_gpu_groups,
)


class ExpB000LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)

    def test_fixed_config_and_base_only_manifest_are_valid(self):
        validate_demo_config(self.config)
        manifest = load_manifest(self.config)
        files = list(iter_manifest_files(manifest))
        self.assertEqual(len(manifest["asset_classes"]), 4)
        self.assertEqual(sum(entry["size"] for _, entry in files), 24834476744)
        joined = " ".join(entry["remote_path"].lower() for _, entry in files)
        self.assertNotIn("optional", joined)
        self.assertNotIn("demonstration", joined)

    def test_gpu_group_validation_rejects_malformed_or_reused_ids(self):
        self.assertEqual(validate_gpu_groups([[0, 1, 2, 3]], 4), [[0, 1, 2, 3]])
        for groups in ([], [[0, 1, 2]], [[0, 1, 2, 2]], [[0, 1, 2, 4]]):
            with self.subTest(groups=groups):
                with self.assertRaises((TypeError, ValueError)):
                    validate_gpu_groups(groups, 4)

    def test_hash_file_supports_sha256_and_git_blob_sha1(self):
        payload = b"fastwam\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(payload)
            self.assertEqual(hash_file(path, "sha256"), hashlib.sha256(payload).hexdigest())
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            self.assertEqual(hash_file(path, "git_blob_sha1"), blob)

    def test_command_and_environment_pin_safe_multigpu_path(self):
        command = build_manager_command(self.config)
        environment = build_environment(self.config)
        self.assertIn("MULTIRUN.gpu_groups=[[0,1,2,3]]", command)
        self.assertIn("EVALUATION.compile_action_infer=false", command)
        self.assertIn("EVALUATION.visualize_future_video=false", command)
        self.assertIn("EVALUATION.text_encoder_devices=[cuda:2,cuda:3]", command)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(environment["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
        self.assertTrue(environment["PYTHONNOUSERSITE"] == "1")
        self.assertIn("repos/LIBERO", environment["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
