"""T-104 — verify the translation pipeline across all 9 directions.

    python -m scripts.t104_smoke --task 2                # 1B model
    python -m scripts.t104_smoke --task 2 --fallback     # 320M distilled
    python -m scripts.t104_smoke --task 2 --benchmark    # time both, compare

Translates 20 real sentences per direction and writes them side by side for
hand-checking, which is §8's acceptance criterion for this task. Also reports
numeral preservation per direction — not a substitute for reading the output,
but it catches the failure this project most cares about without waiting for
T-105.

Downloads the weights on first run: 4.8 GB for the 1B, 1.3 GB for the distilled
fallback. The repos are gated; authenticate first (§3.1).
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import pandas as pd

from src.corpus_io import read_split
from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG, targets_for_block
from src.translate import Translator, decoding_fingerprint, load_config

REPORT_ROOT = REPO_ROOT / "reports"
DIGIT_ZEROS = (0x0966, 0x09E6, 0x0C66, 0x0D66)


def to_ascii_digits(text: str) -> str:
    """Fold every Indic digit onto ASCII so numbers compare across scripts."""
    table = {zero + i: str(i) for zero in DIGIT_ZEROS for i in range(10)}
    return str(text).translate(table)


def numerals(text: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", to_ascii_digits(text)))


def directions() -> list[tuple[str, str, str]]:
    """All 9 (block, source, target) directions from §2."""
    return [
        (block, native, target)
        for block, native in BLOCK_NATIVE_LANG.items()
        for target in targets_for_block(block)
    ]


def run(task: int, model_name: str, config: dict, sample: int, device: str | None):
    translator = Translator(model_name=model_name, config=config, device=device)
    print(f"loaded {model_name} on {translator.device}\n")

    rows = []
    for block, source, target in directions():
        frame = read_split(task, block, source).head(sample)
        started = time.perf_counter()
        outputs = translator.translate([str(t) for t in frame["text"]], source, target)
        elapsed = time.perf_counter() - started

        kept = sum(
            1
            for src, out in zip(frame["text"], outputs)
            if numerals(src) == numerals(out)
        )
        empty = sum(1 for out in outputs if not str(out).strip())
        print(
            f"{source}->{target} ({block}): {elapsed:5.1f}s  "
            f"numerals preserved {kept}/{len(frame)}  empty {empty}"
        )
        for src, out in zip(frame["text"], outputs):
            rows.append(
                {
                    "block": block,
                    "src_lang": source,
                    "tgt_lang": target,
                    "source": src,
                    "translation": out,
                    "src_numerals": " ".join(numerals(src)),
                    "tgt_numerals": " ".join(numerals(out)),
                    "numerals_preserved": numerals(src) == numerals(out),
                    "empty": not str(out).strip(),
                    "seconds_for_direction": round(elapsed, 2),
                }
            )
    translator.unload()
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fallback", action="store_true", help="use the 320M model")
    parser.add_argument(
        "--benchmark", action="store_true", help="run both and compare (§8)"
    )
    args = parser.parse_args(argv)

    config = load_config()
    print(f"decoding fingerprint: {decoding_fingerprint(config)}")
    print(f"config: {config['decoding']}\n")

    chosen = []
    if args.benchmark:
        chosen = [("1B", config["model"]), ("320M", config["fallback_model"])]
    elif args.fallback:
        chosen = [("320M", config["fallback_model"])]
    else:
        chosen = [("1B", config["model"])]

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for label, model_name in chosen:
        print(f"===== {label}: {model_name} =====")
        try:
            frame = run(args.task, model_name, config, args.sample, args.device)
        except Exception as error:  # noqa: BLE001
            import traceback

            print(f"\n{label} FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            traceback.print_exc()
            return 1
        frame["model"] = label
        path = report_dir / f"t104_smoke_{label}.parquet"
        frame.to_parquet(path, index=False)
        print(f"\nWrote {path.relative_to(REPO_ROOT)}")
        summaries.append(
            {
                "model": label,
                "name": model_name,
                "rows": len(frame),
                "numerals_preserved": round(frame["numerals_preserved"].mean(), 4),
                "empty": int(frame["empty"].sum()),
                "total_seconds": round(
                    frame.groupby(["src_lang", "tgt_lang"])["seconds_for_direction"]
                    .first()
                    .sum(),
                    1,
                ),
            }
        )

    summary = pd.DataFrame(summaries)
    print("\n===== summary =====")
    print(summary.to_string(index=False))

    lines = [
        "# T-104 — translation pipeline smoke test",
        "",
        f"Decoding fingerprint `{decoding_fingerprint(config)}`, identical across all "
        "9 directions so drift figures stay comparable between them.",
        "",
        "```json",
        __import__("json").dumps(config["decoding"], indent=2),
        "```",
        "",
        summary.to_markdown(index=False),
        "",
        f"{args.sample} sentences per direction. Per-sentence output is in "
        f"`t104_smoke_*.parquet` for hand-checking — numeral preservation is a "
        "cheap proxy, not a substitute for reading the translations.",
        "",
    ]
    (report_dir / "t104_smoke.md").write_text("\n".join(lines))
    print(f"Wrote {(report_dir / 't104_smoke.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
