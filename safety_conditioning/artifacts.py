"""Explicit project paths, asset provenance and protected output directories."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Choose a fresh output directory: {path}")
    path.mkdir(parents=True)
    return path


def lock_inputs(output, files, config):
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "sha256": {
            str(Path(p).resolve().relative_to(PROJECT)): sha256(p) for p in files
        },
    }
    write_json(Path(output) / "lock.json", manifest)
    return manifest


def verify_lock(path):
    manifest = read_json(path)
    for name, digest in manifest["sha256"].items():
        current = recorded_path(name)
        if current.is_file() and sha256(current) == digest:
            continue
        # Historical source remains in version control instead of a redundant
        # live implementation. New-run locks still require their exact hashes.
        saved = subprocess.run(
            ["git", "show", f"pre-consolidation:{name}"],
            cwd=PROJECT,
            capture_output=True,
        )
        if saved.returncode or hashlib.sha256(saved.stdout).hexdigest() != digest:
            raise ValueError(f"Locked input unavailable or changed: {name}")
    return len(manifest["sha256"])


def recorded_path(name):
    """Resolve historical artifact paths without modifying recorded results."""
    migration = PROJECT / "records/path_migration.json"
    paths = read_json(migration)["paths"] if migration.exists() else {}
    return PROJECT / paths.get(str(name), str(name))


def expert_path(task, level="expert"):
    return PROJECT / "assets" / f"{task}-v5-SAC-{level}.zip"
