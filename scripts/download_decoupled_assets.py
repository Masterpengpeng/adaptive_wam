#!/usr/bin/env python3
"""Download and verify the assets required by the RQ2 decoupled experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "manifest" / "decoupled_assets.json"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "decoupled_asset_download.log"
MIN_FREE_BYTES = 12 * 1024**3
CHUNK_BYTES = 8 * 1024**2


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _log(path: Path, message: str) -> None:
    line = f"{_timestamp()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _hash_file(path: Path, hash_type: str) -> str:
    if hash_type == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif hash_type == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode()
    else:
        raise ValueError(f"Unsupported hash type: {hash_type}")
    digest.update(prefix)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path, entry: dict) -> None:
    expected_size = int(entry["size"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual_hash = _hash_file(path, str(entry["hash_type"]))
    if actual_hash != entry["hash"]:
        raise RuntimeError(
            f"Hash mismatch for {path}: {actual_hash} != {entry['hash']}"
        )


def _url(asset_class: dict, entry: dict) -> str:
    if asset_class["source"] != "huggingface":
        raise ValueError(f"Unsupported source: {asset_class['source']}")
    repository = asset_class["repository"]
    revision = asset_class["revision"]
    remote_path = quote(str(entry["remote_path"]), safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{remote_path}"


def _download(
    session: requests.Session,
    asset_class: dict,
    entry: dict,
    log_path: Path,
) -> None:
    destination = PROJECT_ROOT / entry["local_path"]
    incomplete = destination.with_name(destination.name + ".incomplete")
    expected_size = int(entry["size"])
    if destination.is_file():
        _validate(destination, entry)
        _log(log_path, f"SKIP verified path={destination} size={expected_size}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_size = incomplete.stat().st_size if incomplete.exists() else 0
    if partial_size > expected_size:
        raise RuntimeError(f"Partial download is oversized: {incomplete}")
    remaining = expected_size - partial_size
    free = shutil.disk_usage(destination.parent).free
    if free - remaining < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Disk gate failed for {destination}: downloading {remaining} bytes "
            f"would leave less than {MIN_FREE_BYTES} bytes free"
        )

    headers = {"Accept-Encoding": "identity"}
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
    url = _url(asset_class, entry)
    _log(
        log_path,
        f"START path={destination} size={expected_size} resume={partial_size} url={url}",
    )
    started = time.monotonic()
    last_progress = started
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if partial_size and response.status_code != 206:
            raise RuntimeError(f"Server ignored resume range for {url}")
        mode = "ab" if partial_size else "wb"
        with incomplete.open(mode) as output:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                output.write(chunk)
                now = time.monotonic()
                if now - last_progress >= 60:
                    current = incomplete.stat().st_size
                    _log(log_path, f"PROGRESS path={destination} bytes={current}/{expected_size}")
                    last_progress = now

    _validate(incomplete, entry)
    os.replace(incomplete, destination)
    elapsed = time.monotonic() - started
    _log(log_path, f"COMPLETE path={destination} size={expected_size} seconds={elapsed:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--only", nargs="+", help="Download only named asset classes")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = list(manifest["asset_classes"])
    known = {str(item["id"]) for item in classes}
    if args.only:
        unknown = set(args.only) - known
        if unknown:
            raise ValueError(f"Unknown asset classes: {sorted(unknown)}")
        classes = [item for item in classes if item["id"] in set(args.only)]

    all_entries = [entry for item in manifest["asset_classes"] for entry in item["files"]]
    total = sum(int(entry["size"]) for entry in all_entries)
    if total != int(manifest["total_bytes"]):
        raise ValueError(f"Manifest total mismatch: {total} != {manifest['total_bytes']}")

    entries = [(item, entry) for item in classes for entry in item["files"]]
    _log(log_path, f"RUN_START manifest={manifest_path} files={len(entries)}")
    with requests.Session() as session:
        for asset_class, entry in entries:
            _download(session, asset_class, entry, log_path)
    _log(log_path, "RUN_COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        sys.stderr.write(f"decoupled asset download failed: {error}\n")
        raise
