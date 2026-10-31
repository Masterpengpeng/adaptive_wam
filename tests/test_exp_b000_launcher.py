import hashlib
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from scripts import download_exp_b000_assets as download_exp_b000
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
        self.assertEqual(sum(entry["size"] for _, entry in files), 26243989748)
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

    def test_asset_download_disk_gate_checks_destination_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            destination = project_root / "checkpoints"
            destination.mkdir()
            entry = {
                "local_path": "checkpoints/weights.bin",
                "remote_path": "weights.bin",
                "size": 5,
                "hash_type": "sha256",
                "hash": "unused",
            }
            asset_class = {
                "id": "test",
                "source": "huggingface",
                "repository": "test/model",
                "revision": "test-revision",
            }
            session = Mock()
            available_space = download_exp_b000.MIN_FREE_BYTES + 4

            with (
                patch.object(download_exp_b000, "PROJECT_ROOT", project_root),
                patch.object(
                    download_exp_b000, "LOG_PATH", project_root / "download.log"
                ),
                patch.object(
                    download_exp_b000.shutil,
                    "disk_usage",
                    return_value=Mock(free=available_space),
                ) as disk_usage,
            ):
                with self.assertRaisesRegex(RuntimeError, "disk gate failed"):
                    download_exp_b000.download_one(session, asset_class, entry)

            disk_usage.assert_called_once_with(destination)
            session.get.assert_not_called()

    def test_command_and_environment_pin_safe_multigpu_path(self):
        command = build_manager_command(self.config)
        environment = build_environment(self.config)
        self.assertIn("MULTIRUN.gpu_groups=[[0,1,2,3]]", command)
        self.assertIn("EVALUATION.compile_action_infer=false", command)
        self.assertIn("EVALUATION.visualize_future_video=false", command)
        self.assertIn("EVALUATION.text_encoder_devices=[cuda:2,cuda:3]", command)
        self.assertIn("model.redirect_common_files=false", command)
        self.assertIn(
            "EVALUATION.output_dir=/home/lab416/ZP_projects/adaptive_wam/outputs/baseline/EXP-B000",
            command,
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(environment["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
        self.assertTrue(environment["PYTHONNOUSERSITE"] == "1")
        self.assertIn("repos/LIBERO", environment["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
