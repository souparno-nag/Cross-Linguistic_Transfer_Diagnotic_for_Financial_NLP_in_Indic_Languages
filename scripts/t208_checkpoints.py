"""T-208 — inspect and prune training checkpoints.

    python -m scripts.t208_checkpoints list
    python -m scripts.t208_checkpoints show task2_hin_indicbert_seed1
    python -m scripts.t208_checkpoints path task2_hin_indicbert_seed1
    python -m scripts.t208_checkpoints prune            # dry run
    python -m scripts.t208_checkpoints prune --apply

`list` / `show` / `path` retrieve any run's checkpoint from its run_id (§8's
done criterion). `prune` drops orphan directories — a superseded config or an
aborted run — keeping only checkpoints that are reported in `experiments.csv`
or match a current shipped config. See `src/checkpoints.py` for the policy.
"""

from __future__ import annotations

import argparse
import sys

from src import checkpoints as C
from src.env_check import require_python

_GiB = 1024**3


def _fmt(meta: dict) -> str:
    size = meta["size_bytes"] / _GiB
    f1 = meta["best_dev_macro_f1"]
    f1s = "—" if f1 is None else f"{f1:.4f}"
    tag = meta["run_hash"] or f"{meta['config_hash']} (train)"
    return (
        f"{meta['run_id']:<32} {tag:<26}  "
        f"epoch {meta['last_epoch']:<3} best {f1s} @ {meta['best_epoch']}  "
        f"{size:.2f} GiB"
    )


def main() -> int:
    require_python()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_show = sub.add_parser("show")
    p_show.add_argument("run_id")
    p_path = sub.add_parser("path")
    p_path.add_argument("run_id")
    p_prune = sub.add_parser("prune")
    p_prune.add_argument("--apply", action="store_true", help="actually delete")
    args = parser.parse_args()

    if args.cmd == "list":
        rows = C.list_all()
        if not rows:
            print("no checkpoints")
            return 0
        for meta in rows:
            print(_fmt(meta))
        return 0

    if args.cmd == "path":
        path = C.checkpoint_path(args.run_id)
        if not path.exists():
            print(f"no checkpoint for {args.run_id!r}", file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.cmd == "show":
        meta = C.metadata(args.run_id)
        if meta is None:
            print(f"no checkpoint for {args.run_id!r}", file=sys.stderr)
            return 1
        for key, value in meta.items():
            print(f"  {key:<20} {value}")
        return 0

    if args.cmd == "prune":
        groups = C.classify()
        for meta in groups["keep"]:
            print(f"keep   {meta['run_id']:<32} {meta['reason']}")
        for meta in groups["orphan"]:
            print(f"ORPHAN {meta['run_id']:<32} {meta['reason']}")
        orphans = C.prune(dry_run=not args.apply)
        if not orphans:
            print("\nnothing to prune")
        elif args.apply:
            print(f"\nremoved {len(orphans)} orphan checkpoint dir(s)")
        else:
            print(f"\n{len(orphans)} orphan(s) — re-run with --apply to delete")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
