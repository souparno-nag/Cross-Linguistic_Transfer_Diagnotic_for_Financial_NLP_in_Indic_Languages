"""T-112 — freeze a task's corpus as an immutable release.

Freezing is what makes every number in this project citable: after it, the
files behind a result cannot quietly change. §4 rule 5 says `data/v1.0/` is
never modified, so this module writes it once, records a SHA-256 per file, and
takes the write bit off. Re-freezing a task refuses unless explicitly forced.

The manifest records the fingerprints as well as the hashes. A hash says the
bytes did not change; the fingerprints say *what produced them* — which model,
which decoding config, which similarity model — and artefacts are only
comparable within one of those (§8).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from datetime import datetime, timezone

from .corpus_io import CORPUS_ROOT, FROZEN_ROOT, read_split
from .ids import BLOCK_NATIVE_LANG, targets_for_block

MANIFEST_NAME = "manifest.json"


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_dir(task: int):
    return FROZEN_ROOT / f"task_{task}"


def expected_splits(task: int) -> list[tuple[str, str]]:
    """The 12 splits of §2: one native and three MT per block."""
    out = []
    for block, native in BLOCK_NATIVE_LANG.items():
        out.append((block, native))
        out.extend((block, target) for target in targets_for_block(block))
    return out


def set_writable(path, writable: bool) -> None:
    """Take the write bit off every file and directory, or put it back."""
    for item in sorted(path.rglob("*"), reverse=True):
        mode = item.stat().st_mode
        item.chmod(mode | stat.S_IWUSR if writable else mode & ~0o222)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWUSR if writable else mode & ~0o222)


def freeze_task(task: int, fingerprints: dict, force: bool = False) -> dict:
    """Copy every present split into `data/v1.0/task_{n}/` and seal it."""
    destination = frozen_dir(task)
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} is already frozen. §4 rule 5 makes it immutable — a "
                "new version gets a new directory. Pass force=True only to redo a "
                "freeze that was never released."
            )
        set_writable(destination, True)
        shutil.rmtree(destination)

    files = {}
    missing = []
    for block, lang in expected_splits(task):
        source = CORPUS_ROOT / f"task_{task}" / block / f"{lang}.parquet"
        if not source.exists():
            missing.append(f"{block}/{lang}")
            continue
        frame = read_split(task, block, lang)  # validates before it is sealed
        target = destination / block / f"{lang}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[f"{block}/{lang}.parquet"] = {
            "sha256": sha256(target),
            "rows": len(frame),
            "origin": sorted(set(frame["origin"])),
            "flagged_rows": int(sum(1 for f in frame["flags"] if len(f))),
        }

    # Created here, not only per file: a task with nothing generated yet still
    # gets a manifest that names what is missing, rather than crashing on the
    # write and leaving no record at all.
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": task,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": len(files),
        "rows": sum(entry["rows"] for entry in files.values()),
        "missing": missing,
        "fingerprints": fingerprints,
        "files": files,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    set_writable(destination, False)
    return manifest


def verify_task(task: int) -> dict:
    """Re-hash every file and compare against the manifest.

    Returns the mismatches rather than raising, so a caller can report all of
    them at once — a count is not a diagnosis (§11).
    """
    destination = frozen_dir(task)
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}; task {task} is not frozen")
    manifest = json.loads(manifest_path.read_text())

    changed, missing = [], []
    for name, entry in manifest["files"].items():
        path = destination / name
        if not path.exists():
            missing.append(name)
        elif sha256(path) != entry["sha256"]:
            changed.append(name)
    extra = sorted(
        str(p.relative_to(destination))
        for p in destination.rglob("*.parquet")
        if str(p.relative_to(destination)) not in manifest["files"]
    )
    return {
        "task": task,
        "files": len(manifest["files"]),
        "changed": changed,
        "missing": missing,
        "unexpected": extra,
        "verified": not (changed or missing or extra),
    }
