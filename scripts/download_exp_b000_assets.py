#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

if __package__:
    from .launch_exp_b000 import PROJECT_ROOT, hash_file, iter_manifest_files
else:
    from launch_exp_b000 import PROJECT_ROOT, hash_file, iter_manifest_files


MANIFEST_PATH = PROJECT_ROOT / "results/manifest/EXP-B000_assets.json"
LOG_PATH = PROJECT_ROOT / "logs/EXP-B000/asset_download.log"
MIN_FREE_BYTES = 12 * 1024**3
CHUNK_BYTES = 8 * 1024**2
ORDER = {
    "matching_dataset_statistics": 0,
    "umt5_tokenizer": 1,
    "wan_runtime_weights:Wan2.2_VAE.pth": 2,
    "fastwam_base_checkpoint": 3,
    "wan_runtime_weights:models_t5_umt5-xxl-enc-bf16.pth": 4,
}


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{timestamp()} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def download_source(asset_class: dict) -> dict:
    return asset_class.get("fallback", asset_class)


def source_url(asset_class: dict, entry: dict) -> str:
    source = download_source(asset_class)
    repository = source["repository"]
    revision = source["revision"]
    remote_path = quote(entry["remote_path"], safe="/")
    if source["source"] == "huggingface":
        return f"https://huggingface.co/{repository}/resolve/{revision}/{remote_path}"
    if source["source"] == "modelscope":
        encoded_path = quote(entry["remote_path"], safe="")
        return (
            f"https://www.modelscope.cn/api/v1/models/{repository}/repo"
            f"?Revision={revision}&FilePath={encoded_path}"
        )
    raise ValueError(f"Unsupported source: {source['source']}")


def priority(item: tuple[dict, dict]) -> tuple[int, str]:
    asset_class, entry = item
    key = f"{asset_class['id']}:{entry['remote_path']}"
    return ORDER.get(key, ORDER.get(asset_class["id"], 99)), entry["remote_path"]


def validate_completed(path: Path, entry: dict) -> None:
    expected_size = int(entry["size"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual_hash = hash_file(path, entry["hash_type"])
    if actual_hash != entry["hash"]:
        raise RuntimeError(
            f"hash mismatch for {path}: {actual_hash} != {entry['hash']}"
        )


def download_one(session: requests.Session, asset_class: dict, entry: dict) -> None:
    final_path = PROJECT_ROOT / entry["local_path"]
    incomplete_path = final_path.with_name(final_path.name + ".incomplete")
    expected_size = int(entry["size"])
    url = source_url(asset_class, entry)
    source = download_source(asset_class)
    revision = source["revision"]

    if final_path.exists():
        validate_completed(final_path, entry)
        log(f"SKIP verified path={final_path} size={expected_size}")
        return

    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_size = incomplete_path.stat().st_size if incomplete_path.exists() else 0
    if partial_size > expected_size:
        raise RuntimeError(
            f"incomplete file is oversized: {incomplete_path} ({partial_size})"
        )
    remaining = expected_size - partial_size
    free = shutil.disk_usage(final_path.parent).free
    projected_free = free - remaining
    log(
        f"GATE path={final_path} storage_path={final_path.parent} free={free} "
        f"remaining_download={remaining} "
        f"projected_free={projected_free} minimum_free={MIN_FREE_BYTES}"
    )
    if projected_free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"disk gate failed for {final_path}: projected free {projected_free} "
            f"is below {MIN_FREE_BYTES}"
        )

    if remaining == 0:
        validate_completed(incomplete_path, entry)
        os.replace(incomplete_path, final_path)
        log(f"COMPLETE resumed-full path={final_path} size={expected_size}")
        return

    headers = {"Accept-Encoding": "identity"}
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
    log(
        f"START path={final_path} source={source['source']} "
        f"repository={source['repository']} revision={revision} url={url} "
        f"resume_offset={partial_size} expected_size={expected_size}"
    )
    started = time.monotonic()
    last_progress = started
    try:
        with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if partial_size and response.status_code != 206:
                raise RuntimeError(
                    f"server ignored Range for {url}; keeping existing incomplete file"
                )
            mode = "ab" if incomplete_path.exists() else "xb"
            with incomplete_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    now = time.monotonic()
                    if now - last_progress >= 60:
                        current = incomplete_path.stat().st_size
                        rate = (current - partial_size) / max(now - started, 1e-9)
                        log(
                            f"PROGRESS path={final_path} bytes={current}/{expected_size} "
                            f"rate_Bps={rate:.0f}"
                        )
                        last_progress = now
        downloaded_size = incomplete_path.stat().st_size
        if downloaded_size != expected_size:
            raise RuntimeError(
                f"downloaded size mismatch for {final_path}: "
                f"{downloaded_size} != {expected_size}"
            )
        log(f"VERIFY start path={final_path} hash_type={entry['hash_type']}")
        validate_completed(incomplete_path, entry)
        os.replace(incomplete_path, final_path)
        elapsed = time.monotonic() - started
        rate = remaining / max(elapsed, 1e-9)
        log(
            f"COMPLETE path={final_path} size={expected_size} hash={entry['hash']} "
            f"elapsed_seconds={elapsed:.3f} average_Bps={rate:.0f}"
        )
    except Exception as exc:
        current = incomplete_path.stat().st_size if incomplete_path.exists() else 0
        log(
            f"ERROR path={final_path} incomplete_bytes={current} "
            f"error={type(exc).__name__}:{exc}"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(
            {
                "matching_dataset_statistics",
                "umt5_tokenizer",
                "wan_runtime_weights",
                "fastwam_base_checkpoint",
            }
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = sorted(iter_manifest_files(manifest), key=priority)
    if len(items) != 8:
        raise RuntimeError(f"Expected exactly 8 manifest files, found {len(items)}")
    if args.only:
        selected = set(args.only)
        items = [item for item in items if item[0]["id"] in selected]
    for asset_class in {item[0]["id"]: item[0] for item in items}.values():
        if asset_class.get("source_change_reason"):
            log(
                f"SOURCE_POLICY class={asset_class['id']} source={asset_class['source']} "
                f"repository={asset_class['repository']} revision={asset_class['revision']} "
                f"reason={asset_class['source_change_reason']}"
            )
    log(
        f"RUN_START manifest={MANIFEST_PATH} file_count={len(items)} "
        f"manifest_total_bytes={manifest['total_bytes']}"
    )
    session = requests.Session()
    for asset_class, entry in items:
        download_one(session, asset_class, entry)
    free = shutil.disk_usage(PROJECT_ROOT).free
    log(f"RUN_COMPLETE free_bytes={free}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"RUN_FAILED error={type(error).__name__}:{error}")
        sys.stderr.write(f"asset download failed: {error}\n")
        raise SystemExit(1)
