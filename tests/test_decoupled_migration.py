from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.configure_libero_paths import write_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_decoupled_asset_manifest_is_complete() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "results/manifest/decoupled_assets.json").read_text(
            encoding="utf-8"
        )
    )
    entries = [
        entry
        for asset_class in manifest["asset_classes"]
        for entry in asset_class["files"]
    ]
    assert manifest["experiment_id"] == "RQ2-DECOUPLED"
    assert len(entries) == 8
    assert sum(int(entry["size"]) for entry in entries) == manifest["total_bytes"]
    assert any("optional_idm" in entry["remote_path"] for entry in entries)


def test_decoupled_snapshot_links_are_portable() -> None:
    directory = PROJECT_ROOT / "snapshots_decoupled_core_v1"
    links = sorted(directory.glob("*.pkl"))
    assert len(links) == 20
    for link in links:
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
        assert link.exists()


def test_fastwam_patch_matches_lock() -> None:
    lock_lines = (PROJECT_ROOT / "repos/FastWAM.lock").read_text(
        encoding="utf-8"
    ).splitlines()
    lock = dict(line.split("=", 1) for line in lock_lines if "=" in line)
    patch = PROJECT_ROOT / lock["local_patch"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert digest == lock["local_patch_sha256"]
    assert lock["commit"] == "7faa71108368fbb3b6885649f112af607427a2d4"


def test_libero_runtime_config_uses_checkout_paths(tmp_path: Path) -> None:
    project = tmp_path / "moved checkout"
    benchmark = project / "repos/LIBERO/libero/libero"
    for directory in (benchmark / "bddl_files", benchmark / "init_files", benchmark / "assets"):
        directory.mkdir(parents=True)

    output = tmp_path / "runtime/libero/config.yaml"
    write_config(project, output)
    content = output.read_text(encoding="utf-8")

    assert str(project.resolve()) in content
    assert "/home/lab416/" not in content
    assert (project / "outputs/baseline/EXP-B000/no_demonstrations").is_dir()


def test_libero_runtime_config_requires_a_complete_clone(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing LIBERO"):
        write_config(tmp_path, tmp_path / "config.yaml")
