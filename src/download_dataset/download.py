"""Fetch the IndicFinNLP dataset and materialise it under data/base_paper/.

kagglehub downloads into its own cache with a ``versions/<n>`` layout and
upstream's doubled ``IndicFinNLP_data/IndicFinNLP_data`` nesting. This script
keeps that cache out of the tree (in .cache/) and copies a flat, uniformly
named tree into data/base_paper/, which *is* committed.

    python -m src.download_dataset.download
    python -m src.download_dataset.download --force

Re-runs are no-ops when every file already matches manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import (
    BASE_PAPER_DIR,
    DATASET_SLUG,
    DATASET_URL,
    KAGGLEHUB_CACHE,
    LANGUAGES,
    MANIFEST_PATH,
    RAW_DIR,
    REPO_ROOT,
    SOURCE_PATH,
    raw_path,
)

# kagglehub reads its cache location from the environment when the download is
# resolved, so this has to be set before the module is imported.
KAGGLEHUB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(KAGGLEHUB_CACHE)

import kagglehub  # noqa: E402  (import must follow KAGGLEHUB_CACHE)

DESCRIPTION = "Fetch the IndicFinNLP dataset and materialise it under data/base_paper/."

# Upstream docs copied alongside the data, by filename.
DOC_FILES = ("README.md", "license.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_language(filename: str) -> str:
    """Pull the language out of an upstream filename."""
    matches = [lang for lang in LANGUAGES if lang in filename.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"cannot determine language for {filename!r}; matched {matches}. "
            f"If upstream added a language, extend LANGUAGES in paths.py."
        )
    return matches[0]


def parse_task(spreadsheet: Path) -> int:
    """Task number comes from the containing ``task_<n>`` directory."""
    name = spreadsheet.parent.name
    if not name.startswith("task_") or not name[5:].isdigit():
        raise ValueError(f"unexpected task directory {name!r} for {spreadsheet}")
    return int(name[5:])


def discover(version_dir: Path) -> list[dict]:
    """Map every upstream spreadsheet onto its materialised location."""
    entries = []
    for spreadsheet in sorted(version_dir.rglob("task_*/*.xlsx")):
        task = parse_task(spreadsheet)
        language = parse_language(spreadsheet.name)
        target = raw_path(task, language)
        entries.append(
            {
                "task": task,
                "language": language,
                "upstream": spreadsheet.relative_to(version_dir).as_posix(),
                "local": target.relative_to(BASE_PAPER_DIR).as_posix(),
                "bytes": spreadsheet.stat().st_size,
                "sha256": sha256(spreadsheet),
                "_source": spreadsheet,
            }
        )

    duplicates = {}
    for entry in entries:
        duplicates.setdefault(entry["local"], []).append(entry["upstream"])
    collisions = {k: v for k, v in duplicates.items() if len(v) > 1}
    if collisions:
        raise ValueError(f"multiple upstream files map to the same target: {collisions}")
    return entries


def is_current(entries: list[dict], version: int) -> bool:
    """True when the materialised copy already matches these files."""
    if not MANIFEST_PATH.exists():
        return False
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("version") != version:
        return False
    recorded = {(e["local"], e["sha256"]) for e in manifest.get("files", [])}
    if recorded != {(e["local"], e["sha256"]) for e in entries}:
        return False
    # The manifest can be stale relative to the tree; check the files too.
    return all((BASE_PAPER_DIR / local).exists() for local, _ in recorded)


def materialise(entries: list[dict], version_dir: Path, version: int) -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    for entry in entries:
        target = BASE_PAPER_DIR / entry["local"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["_source"], target)

    docs = []
    for name in DOC_FILES:
        source = version_dir / name
        if not source.exists():
            continue
        shutil.copy2(source, BASE_PAPER_DIR / name)
        docs.append({"local": name, "sha256": sha256(source)})

    manifest = {
        "dataset": DATASET_SLUG,
        "url": DATASET_URL,
        "version": version,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kagglehub_version": getattr(kagglehub, "__version__", "unknown"),
        "docs": docs,
        "files": [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    SOURCE_PATH.write_text(render_source(manifest))


def render_source(manifest: dict) -> str:
    rows = "\n".join(
        f"| {e['task']} | {e['language']} | `{e['local']}` | `{e['upstream']}` |"
        for e in manifest["files"]
    )
    return f"""# Base paper dataset

IndicFinNLP, the dataset accompanying *"IndicFinNLP: Financial Natural
Language Processing for Indian Languages"*.

| | |
|---|---|
| Source | <{manifest['url']}> |
| Kaggle slug | `{manifest['dataset']}` |
| Version | {manifest['version']} |
| Retrieved (UTC) | {manifest['retrieved_utc']} |

Licence terms are in `license.txt`; upstream's column-level metadata is in
`README.md`. Both are copied verbatim from the release.

## Layout

Files are renamed on materialisation so that every task and language shares
one shape, `raw/task_<n>/<language>.xlsx`. `manifest.json` records the
upstream name and a SHA-256 for each file.

| Task | Language | Local | Upstream |
|---|---|---|---|
{rows}

## Regenerating

Do not edit this directory by hand:

```
python -m src.download_dataset.download          # no-op if already current
python -m src.download_dataset.download --force  # re-materialise
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-materialise even when the local copy is already current",
    )
    args = parser.parse_args(argv)

    version_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    version = int(version_dir.name) if version_dir.name.isdigit() else 0

    entries = discover(version_dir)
    if not entries:
        print(f"No task spreadsheets found under {version_dir}", file=sys.stderr)
        return 1

    if is_current(entries, version) and not args.force:
        print(f"data/base_paper/ is already current (version {version}, {len(entries)} files)")
        return 0

    BASE_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    materialise(entries, version_dir, version)
    print(
        f"Materialised {len(entries)} files (version {version}) into "
        f"{BASE_PAPER_DIR.relative_to(REPO_ROOT)}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
