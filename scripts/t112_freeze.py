"""T-112 — freeze corpus v1.0.

    python -m scripts.t112_freeze --task 3            # freeze
    python -m scripts.t112_freeze --task 3 --verify   # re-hash and compare

Copies every split of a task into `data/v1.0/task_{n}/`, writes a manifest with
a SHA-256 per file, and removes the write bit. A task freezes independently
once its own checks pass; the manifest covers whatever is present and names
whatever is not.

Freezing refuses to overwrite an existing release (§4 rule 5). `--force` exists
only for a freeze that was never released.
"""

from __future__ import annotations

import argparse
import sys

from src.download_dataset.paths import REPO_ROOT
from src.freeze import freeze_task, frozen_dir, verify_task
from src.labse_gate import load_config as load_verification_config
from src.labse_gate import tau_for, verification_fingerprint
from src.translate import decoding_fingerprint
from src.translate import load_config as load_translation_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.verify:
        try:
            result = verify_task(args.task)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 2
        print(f"task {args.task}: {result['files']} files")
        for kind in ("changed", "missing", "unexpected"):
            if result[kind]:
                print(f"  {kind}: {result[kind]}", file=sys.stderr)
        if result["verified"]:
            print("manifest verifies.")
            return 0
        print("MANIFEST DOES NOT VERIFY", file=sys.stderr)
        return 1

    translation = load_translation_config()
    verification = load_verification_config()
    fingerprints = {
        "translation_model": translation["model"],
        "decoding": decoding_fingerprint(translation),
        "similarity_model": verification["model"],
        "verification": verification_fingerprint(verification),
        # Per task, not the global default: task 3's flags were derived at its
        # own calibrated τ, and a manifest naming the wrong threshold would
        # misdescribe the very column it seals.
        "tau": tau_for(verification, args.task),
    }
    try:
        manifest = freeze_task(args.task, fingerprints, force=args.force)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"froze task {args.task}: {manifest['splits']} splits, {manifest['rows']} rows")
    if manifest["missing"]:
        print(f"  missing (not frozen): {manifest['missing']}")
    print(f"  -> {frozen_dir(args.task).relative_to(REPO_ROOT)} (read-only)")
    result = verify_task(args.task)
    print("manifest verifies." if result["verified"] else f"VERIFY FAILED: {result}")
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
