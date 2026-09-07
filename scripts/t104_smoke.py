"""T-104 — verify the translation pipeline across all 9 directions.

    python -m scripts.t104_smoke --task 2                # 1B model
    python -m scripts.t104_smoke --task 2 --fallback     # 320M distilled
    python -m scripts.t104_smoke --task 2 --benchmark    # time both, compare

Translates 20 real sentences per direction and writes them side by side for
hand-checking, which is §8's acceptance criterion for this task.

It also reports how often the *digits* are identical between source and
translation. Read that as a rough signal, not a score: a correct translation can
legitimately change the digits, because Indic and Western numbering systems
differ. `১০০ মিলিয়ন` (100 million) rendering as `10 करोड़` (10 crore) is right,
and this check counts it as a mismatch. Handling that properly is T-105's job.

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
from src.entities import ESCAPE_LEAK, PLACEHOLDER_LEAK
from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG, targets_for_block
from src.translate import (
    Translator,
    decoding_fingerprint,
    load_config,
    translate_adaptive,
)

REPORT_ROOT = REPO_ROOT / "reports"
DIGIT_ZEROS = (0x0966, 0x09E6, 0x0C66, 0x0D66)


def to_ascii_digits(text: str) -> str:
    """Fold every Indic digit onto ASCII so numbers compare across scripts."""
    table = {zero + i: str(i) for zero in DIGIT_ZEROS for i in range(10)}
    return str(text).translate(table)


# A comma is a thousands separator only when exactly three digits follow it.
# Without this, "जुलाई 22,2019" parses as the single number 22,2019 rather than
# 22 and 2019, and a correct translation is scored as a numeral change.
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def numerals(text: str) -> list[str]:
    """Numbers in a string, comparable across scripts and separators."""
    folded = THOUSANDS_SEPARATOR.sub("", to_ascii_digits(text))
    return sorted(re.findall(r"\d+(?:\.\d+)?", folded))


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
    size = config["batch_size"]

    rows = []
    for block, source, target in directions():
        frame = read_split(task, block, source).head(sample)
        texts = [str(t) for t in frame["text"]]
        started = time.perf_counter()
        # Chunk by the configured batch size. Sending the whole sample in one
        # generate() call OOMs the 4 GB card on task 1, whose Telugu split has
        # 280 rows over 1000 characters (max 2510) — beam 5 over 20 long
        # sequences at once does not fit.
        outputs = translate_adaptive(translator, texts, source, target, size)
        elapsed = time.perf_counter() - started
        # Reclaim allocator blocks before the next direction; without this the
        # run dies partway through with ~1 GB held beyond the model itself.
        translator.free()

        kept = sum(
            1
            for src, out in zip(frame["text"], outputs)
            if numerals(src) == numerals(out)
        )
        empty = sum(1 for out in outputs if not str(out).strip())
        placeholders = sum(1 for out in outputs if PLACEHOLDER_LEAK.search(str(out)))
        escapes = sum(1 for out in outputs if ESCAPE_LEAK.search(str(out)))
        corrupt = ""
        if placeholders or escapes:
            corrupt = f"  CORRUPT placeholder {placeholders} escape {escapes}"
        print(
            f"{source}->{target} ({block}): {elapsed:5.1f}s  "
            f"numerals identical {kept}/{len(frame)}  empty {empty}{corrupt}"
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
                    "numerals_identical": numerals(src) == numerals(out),
                    "empty": not str(out).strip(),
                    "placeholder_leak": bool(PLACEHOLDER_LEAK.search(str(out))),
                    "escape_leak": bool(ESCAPE_LEAK.search(str(out))),
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
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        frame["model"] = label
        path = report_dir / f"t104_smoke_{label}.parquet"
        frame.to_parquet(path, index=False)
        print(f"\nWrote {path.relative_to(REPO_ROOT)}")
        summaries.append(
            {
                "model": label,
                "name": model_name,
                "rows": len(frame),
                "numerals_identical": round(frame["numerals_identical"].mean(), 4),
                "empty": int(frame["empty"].sum()),
                "placeholder_leak": int(frame["placeholder_leak"].sum()),
                "escape_leak": int(frame["escape_leak"].sum()),
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
        "`t104_smoke_*.parquet` for hand-checking.",
        "",
        "`numerals_identical` counts rows whose digit set is unchanged. It is a "
        "rough signal, **not** an accuracy score: a correct translation can "
        "legitimately change the digits, because the two numbering systems "
        "differ. `১০০ মিলিয়ন` (100 million) → `10 करोड़` (10 crore) is correct and "
        "counts here as a mismatch. Scale-aware comparison is T-105.",
        "",
    ]
    (report_dir / "t104_smoke.md").write_text("\n".join(lines))
    print(f"Wrote {(report_dir / 't104_smoke.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
