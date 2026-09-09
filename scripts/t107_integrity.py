"""T-107 — structural integrity check.

    python -m scripts.t107_integrity --task 3            # report only
    python -m scripts.t107_integrity --task 3 --write-flags
    python -m scripts.t107_integrity --task 1 --strict   # findings fail too

Writes `reports/task_{n}/integrity.json` with the summary and
`integrity_rows.parquet` with every flagged row — §8 asks for failures
enumerated by row id, and at these volumes (task 1 flags thousands of rows)
the ids belong in Parquet rather than in the JSON (§5, rule 2).

Exits non-zero on a **blocking** failure: a block that stops joining 4-way, a
label that drifted from its source, a replacement character in native text.
Findings — script leakage, the two §3.4 corruption modes, suspiciously short
output — are counted and enumerated but do not fail the run, because §4 rule 1
keeps those rows and they are results rather than defects. `--strict` fails on
them too.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.audit import config_hash
from src.download_dataset.paths import REPO_ROOT
from src.integrity import INTEGRITY_FLAGS, TRUNCATION_RATIO, apply_flags, check_task

REPORT_ROOT = REPO_ROOT / "reports"
EXAMPLES_PER_FLAG = 5


def examples(flagged: pd.DataFrame, task: int) -> dict:
    """A few rows per flag, with their text. A count is not a diagnosis (§11)."""
    from src.corpus_io import read_split  # noqa: PLC0415

    out = {}
    for flag in INTEGRITY_FLAGS:
        subset = flagged[flagged["flags"].apply(lambda f, flag=flag: flag in f)]
        rows = []
        for _, row in subset.head(EXAMPLES_PER_FLAG).iterrows():
            mt = read_split(task, row["block"], row["tgt_lang"])
            text = mt.loc[mt["item_id"] == row["item_id"], "text"]
            rows.append(
                {
                    "item_id": row["item_id"],
                    "direction": f"{row['src_lang']}->{row['tgt_lang']}",
                    "length_ratio": row["length_ratio"],
                    "text": (text.iloc[0][:160] if len(text) else ""),
                }
            )
        if rows:
            out[flag] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--write-flags", action="store_true", help="write flags into the corpus")
    parser.add_argument("--strict", action="store_true", help="findings fail the run too")
    args = parser.parse_args(argv)

    report, flagged = check_task(args.task)
    config = {"task": args.task, "truncation_ratio": TRUNCATION_RATIO}
    report["config_hash"] = config_hash(config)
    report["examples"] = examples(flagged, args.task)

    print(f"task {args.task}  config {report['config_hash']}\n")
    frame = pd.DataFrame(report["directions"])
    print(frame.to_string(index=False))

    print("\nstructure:")
    for block, info in report["structure"].items():
        state = "ok" if info["four_way"] else "FAILED"
        print(f"  {block}: {info['joined_rows']}/{info['expected_rows']} rows 4-way  {state}")

    print("\nnative splits:")
    for name, info in report["natives"].items():
        print(
            f"  {name}: {info['rows']} rows, script leakage "
            f"{len(info['script_leakage'])}, replacement chars "
            f"{len(info['encoding_corruption'])}, not NFC {info['not_nfc']}"
        )

    total = report["findings_total"]
    print("\nfindings (flagged, rows kept — §4 rule 1):")
    for flag, count in total.items():
        print(f"  {flag:22s} {count}")

    for flag, rows in report["examples"].items():
        print(f"\n  {flag}:")
        for row in rows:
            print(f"    [{row['direction']}] {row['item_id']} ratio {row['length_ratio']}")
            print(f"      {row['text'][:120]}")

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "integrity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    flagged.to_parquet(report_dir / "integrity_rows.parquet", index=False)
    print(
        f"\nWrote {(report_dir / 'integrity.json').relative_to(REPO_ROOT)} and "
        f"integrity_rows.parquet ({len(flagged)} flagged rows)"
    )

    if args.write_flags:
        written = apply_flags(args.task, flagged)
        print(f"Flags written into {len(written)} splits: {sum(written.values())} rows flagged")

    if report["blocking"]:
        print("\nBLOCKING:", file=sys.stderr)
        for failure in report["blocking"]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    if args.strict and any(total.values()):
        print("\n--strict: findings present", file=sys.stderr)
        return 1
    print("\nall green: no blocking failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
