"""T-106 — generate all 9 translation directions for one task.

    python -m scripts.t106_generate --task 3            # cheapest task first
    python -m scripts.t106_generate --task 3 --dry-run  # what would be run
    python -m scripts.t106_generate --task 1 --block H --targets ben

Block H → {ben, mal, tel}, block B → {hin, mal, tel}, block T → {hin, ben, mal},
written to `data/raw/task_{n}/{block}/{lang}.parquet` beside the native split
each was translated from.

Run the tasks in ascending cost (3, then 2, then 1): task 3 proves the whole
path end to end in under two hours, before a day of GPU time is committed to
task 1. Every direction checkpoints after each batch (T-104), so an interrupted
run resumes rather than restarting — including across invocations of this
script.

The model is loaded once for the whole run. Nine loads of a 4.8 GB model on a
4 GB card OOMs on the second (§3.5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from src.audit import config_hash
from src.corpus_io import is_numeral_task, read_split, split_path, write_split
from src.download_dataset.paths import REPO_ROOT
from src.generate import broadcast, build_mt_frame, check_split, representatives
from src.ids import BLOCK_NATIVE_LANG, targets_for_block
from src.translate import (
    DEFAULT_BATCH_TIMEOUT,
    Translator,
    checkpoint_path,
    decoding_fingerprint,
    load_checkpoint,
    load_config,
    translate_rows,
)

REPORT_ROOT = REPO_ROOT / "reports"


def directions(blocks: list[str], targets: list[str] | None) -> list[tuple[str, str, str]]:
    """The (block, source, target) directions to run, in §2's order."""
    return [
        (block, BLOCK_NATIVE_LANG[block], target)
        for block in blocks
        for target in targets_for_block(block)
        if targets is None or target in targets
    ]


def seed_everything(seed: int) -> None:
    """§4 rule 7. Beam search is already deterministic; this covers the rest."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_direction(
    translator, task, block, source_lang, target_lang, batch_size, dry_run, rebuild,
    timeout=None,
):
    native = read_split(task, block, source_lang)
    unique = representatives(native)
    print(
        f"\n{source_lang}->{target_lang} ({block}): {len(native)} rows, "
        f"{len(unique)} distinct sentences to translate"
    )
    if dry_run:
        return native, None, None
    if rebuild:
        # Rebuilding reads the checkpoint and re-derives the split from it. No
        # model, no GPU: the point is to re-run the frame-building half after a
        # fix without paying for the translation again, or while another task's
        # run has the card.
        done = load_checkpoint(checkpoint_path(task, block, source_lang, target_lang))
        outstanding = len(set(unique["item_id"]) - set(done["item_id"]))
        if outstanding:
            raise ValueError(
                f"{outstanding} of {len(unique)} sentences are not in the "
                "checkpoint; --rebuild cannot invent them, run without it"
            )

    started = time.perf_counter()
    translated = translate_rows(
        translator,
        unique,
        source_lang,
        target_lang,
        task,
        block,
        batch_size=batch_size,
        timeout=timeout if timeout is not None else DEFAULT_BATCH_TIMEOUT,
    )
    elapsed = time.perf_counter() - started
    # Reclaim allocator blocks before the next direction; without this the run
    # dies partway through with ~1 GB held beyond the model itself (§3.5).
    if translator is not None:
        translator.free()

    mt = build_mt_frame(task, native, broadcast(native, translated), target_lang)
    return native, mt, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--block", action="append", choices=sorted(BLOCK_NATIVE_LANG))
    parser.add_argument("--targets", help="comma-separated subset, e.g. ben,mal")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=DEFAULT_BATCH_TIMEOUT,
        help="seconds before a batch is treated as stalled; 0 disables",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--fallback", action="store_true", help="use the 320M model")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, load nothing")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-derive the splits from existing checkpoints; loads no model",
    )
    args = parser.parse_args(argv)

    config = load_config()
    model_name = config["fallback_model"] if args.fallback else config["model"]
    fingerprint = decoding_fingerprint(config)
    run_config = {
        "task": args.task,
        "model": model_name,
        "decoding_fingerprint": fingerprint,
    }
    print(f"model: {model_name}")
    print(f"decoding fingerprint: {fingerprint}")
    print(f"run config hash: {config_hash(run_config)}")

    blocks = args.block or sorted(BLOCK_NATIVE_LANG)
    targets = args.targets.split(",") if args.targets else None
    plan = directions(blocks, targets)
    if not plan:
        print("no directions selected", file=sys.stderr)
        return 2

    seed_everything(config["seed"])
    translator = None
    if not (args.dry_run or args.rebuild):
        translator = Translator(model_name=model_name, config=config, device=args.device)
        print(f"loaded on {translator.device}")

    summaries = []
    failed = []
    for block, source_lang, target_lang in plan:
        try:
            native, mt, elapsed = run_direction(
                translator, args.task, block, source_lang, target_lang,
                args.batch_size, args.dry_run, args.rebuild, args.batch_timeout,
            )
        except Exception as error:  # noqa: BLE001
            # §11: report the direction and carry on; do not silently substitute
            # a different model or quietly drop it.
            import traceback

            traceback.print_exc()
            print(f"  FAILED {source_lang}->{target_lang}: {error}", file=sys.stderr)
            failed.append(f"{block}:{source_lang}->{target_lang}")
            continue
        if args.dry_run:
            continue

        summary = check_split(args.task, native, mt)
        path = write_split(mt, args.task, block, target_lang)
        summary.update(
            {
                "block": block,
                "src_lang": source_lang,
                "tgt_lang": target_lang,
                "seconds": round(elapsed, 1),
            }
        )
        summaries.append(summary)
        extra = ""
        if is_numeral_task(args.task):
            extra = f", spans recovered {summary['span_recovery_rate']:.1%}"
        print(
            f"  wrote {path.relative_to(REPO_ROOT)}: {summary['rows']} rows, "
            f"{summary['empty']} empty{extra} ({elapsed:.0f}s)"
        )

    if translator is not None:
        translator.unload()
    if args.dry_run:
        print(f"\n{len(plan)} directions planned; nothing written.")
        return 0

    if not summaries:
        print("no direction completed", file=sys.stderr)
        return 1

    frame = pd.DataFrame(summaries)
    ordered = ["block", "src_lang", "tgt_lang", "rows", "unique_source_texts", "empty"]
    ordered += [c for c in frame.columns if c not in ordered]
    frame = frame[ordered]
    print("\n===== summary =====")
    print(frame.to_string(index=False))

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "t106_generate.json").write_text(
        json.dumps(
            {
                "config": run_config,
                "config_hash": config_hash(run_config),
                "batch_size": args.batch_size or config["batch_size"],
                "batch_timeout": args.batch_timeout,
                "directions": summaries,
                "failed": failed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    lines = [
        f"# T-106 — generated splits, task {args.task}",
        "",
        f"Model `{model_name}`, decoding fingerprint `{fingerprint}`, identical "
        "across all directions so drift figures stay comparable (§8).",
        "",
        frame.to_markdown(index=False),
        "",
        "`unique_source_texts` is what actually went to the model: identical "
        "sentences are translated once and the result shared, so one sentence "
        "carrying three annotated numbers cannot come back as three different "
        "sentences.",
        "",
    ]
    if is_numeral_task(args.task):
        lines += [
            "`span_recovered` counts MT rows where the annotated number was found "
            "again in the translation. A failure is flagged and kept (§4 rule 1); "
            "the rate is a headline result, not an error. `span_scale_shift` "
            "separates numbers rewritten as an equivalent quantity "
            "(`১০০ মিলিয়ন` → `10 करोड़`) from numbers genuinely lost.",
            "",
        ]
    if failed:
        lines += ["## Failed directions", "", *(f"- `{d}`" for d in failed), ""]
    (report_dir / "t106_generate.md").write_text("\n".join(lines))
    print(f"\nWrote {(report_dir / 't106_generate.md').relative_to(REPO_ROOT)}")

    if failed:
        print(f"{len(failed)} direction(s) failed: {failed}", file=sys.stderr)
        return 1
    if len(summaries) == len(plan) and not args.block and not args.targets:
        print(f"\nAll {len(summaries)} directions complete for task {args.task}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
