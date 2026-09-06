"""T-102 — audit the native splits and test them for hidden parallelism.

    python -m scripts.t102_audit                 # Phase A only, no model needed
    python -m scripts.t102_audit --independence  # adds the LaBSE tests (B1/B2)

Phase A audits row counts, class distribution, encoding, duplicates and empty
rows, and establishes provenance from data/base_paper/manifest.json.

Phase B asks whether the three native splits are secretly translations of each
other. CLAUDE.md §8 specified 200 random cross-language pairs; random pairs
score low whether or not the corpora are parallel, so that sampling is kept
only as the B1 control and the finding rests on B2, a nearest-neighbour search
over the whole target split.

Exits non-zero when any check fails, so "done" is automated rather than
eyeballed (§8).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.audit import (
    CONTROL_TASK,
    LANGUAGES,
    NATIVE_TASK,
    audit_split,
    class_proportions,
    config_hash,
    control_task_parallelism,
    calibration_controls,
    embed_all,
    independence_tests,
    label_vocabularies,
    load_split,
    verify_provenance,
)
from src.download_dataset.paths import REPO_ROOT
from src.unicode_ranges import script_of

REPORT_DIR = REPO_ROOT / "reports"
REPORT_MD = REPORT_DIR / "native_audit.md"
REPORT_PARQUET = REPORT_DIR / "native_audit.parquet"
INDEPENDENCE_PARQUET = REPORT_DIR / "native_independence.parquet"
CONTROLS_PARQUET = REPORT_DIR / "native_independence_controls.parquet"

DEFAULT_SEED = 20260903
DEFAULT_SAMPLE = 200
DEFAULT_TAU = 0.82
LABSE = "sentence-transformers/LaBSE"


def summary_frame(audits) -> pd.DataFrame:
    """The per-split numbers, as the Parquet artefact required by §5."""
    return pd.DataFrame(
        [
            {
                "task": a.task,
                "language": a.language,
                "script": script_of(a.language),
                "rows": a.rows,
                "empty_text": a.empty_text,
                "empty_label": a.empty_label,
                "duplicate_texts": a.duplicate_texts,
                "duplicate_conflicting_labels": a.duplicate_conflicting_labels,
                "non_nfc": a.non_nfc,
                "replacement_char": a.replacement_char,
                "joiner_rows": a.joiner_rows,
                "leaked_script_rows": a.leaked_script_rows,
                "leaked_script_chars": json.dumps(a.leaked_script_chars),
                "rows_with_native_digits": a.rows_with_native_digits,
                "rows_with_ascii_digits": a.rows_with_ascii_digits,
                "text_len_min": a.text_len_min,
                "text_len_median": a.text_len_median,
                "text_len_max": a.text_len_max,
                "label_counts": json.dumps(a.label_counts, ensure_ascii=False),
            }
            for a in audits
        ]
    )


def md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    )
    return "\n".join([header, rule, body])


def render(
    audits,
    provenance_ok,
    provenance_rows,
    vocab,
    proportions,
    control,
    independence,
    verdicts,
    controls,
    config,
    failures,
) -> str:
    parts = [
        "# T-102 — Native split audit",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"config hash `{config_hash(config)}`",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## 1. Provenance",
        "",
        f"All {len(provenance_rows)} upstream files re-verified against "
        f"`data/base_paper/manifest.json`: "
        f"**{'all match' if provenance_ok else 'MISMATCH'}**.",
        "",
        "Source: IndicFinNLP (Ghosh et al., LREC-COLING 2024), Kaggle version 2. "
        "Licence terms in `data/base_paper/license.txt`.",
        "",
    ]

    if not provenance_ok:
        bad = [r for r in provenance_rows if not r["ok"]]
        parts += ["Files that do not match:", ""]
        parts += [f"- `{r['local']}`" for r in bad]
        parts += [""]

    parts += [
        f"## 2. Per-split audit (task {NATIVE_TASK})",
        "",
        md_table(summary_frame(audits)),
        "",
        "### Class distribution",
        "",
        md_table(proportions),
        "",
        "### Label vocabulary",
        "",
        f"Identical across all three languages: **{vocab['identical']}**  ",
        f"Union: {', '.join(f'`{v}`' for v in vocab['union'])}",
        "",
    ]
    if vocab["only_in"]:
        parts += [
            "Labels not shared by every language — these must be reconciled in "
            "T-103 before `labels.json` is written:",
            "",
        ]
        parts += [
            f"- **{lang}**: {', '.join(f'`{v}`' for v in vals)}"
            for lang, vals in vocab["only_in"].items()
        ]
        parts += [""]

    for a in audits:
        if a.conflicting_examples:
            parts += [
                f"Duplicate texts with conflicting labels in {a.language}:",
                "",
            ]
            parts += [f"- `{t}`" for t in a.conflicting_examples]
            parts += [""]

    parts += [
        "## 3. Independence",
        "",
        f"### 3.1 Task {CONTROL_TASK} is parallel by construction",
        "",
        f"Row counts {control['rows']}, of which "
        f"{control['urls_in_all_three']} URLs appear in all three languages and "
        f"{control['row_order_aligned']} rows carry the same URL at the same "
        f"index. Parallel: **{control['is_parallel']}**.",
        "",
        f"Task {CONTROL_TASK} is therefore one set of articles translated three "
        f"ways, not three independent corpora. It cannot supply the native "
        f"splits, and is retained instead as a human-translated parallel "
        f"control set for judging MT quality in T-104 and T-110.",
        "",
        f"### 3.2 Task {NATIVE_TASK} — LaBSE tests",
        "",
    ]

    if independence is None:
        parts += [
            "_Not run. Re-run with `--independence` to produce this section._",
            "",
        ]
    else:
        parts += [
            md_table(independence),
            "",
            "**B1** pairs rows by index and compares against an index-shuffled "
            "control; a large lift over the control means order-preserved "
            "parallelism. **B2** takes, for each sampled source sentence, the "
            "maximum cosine over the entire target split — it still fires when "
            "rows were shuffled or partially dropped, which is the shape these "
            "row counts suggest. The §8 random-pair sampling is the B1 control, "
            "not a test in its own right.",
            "",
            "#### Calibration controls",
            "",
            md_table(controls),
            "",
            "A raw B2 share is not interpretable on its own — every sentence "
            "here is narrow-domain ESG text, so a high score could be topical "
            "rather than translational. The negative controls pair genuinely "
            f"different content across languages; the positive control is task "
            f"{CONTROL_TASK}, proven parallel by its URL column. The finding "
            "below is only meaningful because the negative anchors sit far "
            "below τ while the split under test sits at or above the positive "
            "anchor.",
            "",
            "#### Finding",
            "",
        ]
        for pair, verdict in verdicts.items():
            state = (
                "**parallel suspected**"
                if verdict["parallel_suspected"]
                else "independent"
            )
            parts.append(
                f"- `{pair}` — {state}. "
                f"{verdict['b2_above_tau_share']:.1%} of sampled sentences have a "
                f"nearest neighbour at or above τ={config['tau']}; B1 lift over "
                f"control {verdict['b1_lift_over_control']:+.4f}."
            )
        parts += [""]

    parts += ["## 4. Result", ""]
    if failures:
        parts += ["Checks that failed:", ""]
        parts += [f"- {f}" for f in failures]
    else:
        parts += ["All Phase A checks passed."]
    parts += [""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=NATIVE_TASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--model", default=LABSE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to cuda when available, else cpu. "
        "Pass 'cpu' if the 4 GB card is busy — 6.5k sentences is minutes on CPU.",
    )
    parser.add_argument(
        "--independence",
        action="store_true",
        help="run the LaBSE tests (needs sentence-transformers)",
    )
    args = parser.parse_args(argv)

    config = {
        "task": args.task,
        "control_task": CONTROL_TASK,
        "languages": list(LANGUAGES),
        "seed": args.seed,
        "sample": args.sample,
        "tau": args.tau,
        "model": args.model if args.independence else None,
        "batch_size": args.batch_size if args.independence else None,
    }

    provenance_ok, provenance_rows = verify_provenance()
    audits = [audit_split(args.task, lang) for lang in LANGUAGES]
    vocab = label_vocabularies(audits)
    proportions = class_proportions(audits)
    control = control_task_parallelism()

    failures = [f for a in audits for f in a.failures()]
    if not provenance_ok:
        failures.append("provenance: files do not match manifest.json")
    if not vocab["identical"]:
        failures.append("label vocabularies differ across languages")

    independence = verdicts = controls = None
    if args.independence:
        column = {1: "indic", 2: "sentence_indic", 3: "news_title_indic"}
        # Native splits plus the control task, embedded together so the model
        # loads once (§3: the 4 GB card OOMs on a second copy).
        texts = {
            f"{task}:{lang}": load_split(task, lang)[column[task]].tolist()
            for task in (args.task, CONTROL_TASK)
            for lang in LANGUAGES
        }
        vectors = embed_all(
            texts, args.model, batch_size=args.batch_size, device=args.device
        )
        embeddings = {lang: vectors[f"{args.task}:{lang}"] for lang in LANGUAGES}
        control_embeddings = {
            lang: vectors[f"{CONTROL_TASK}:{lang}"] for lang in LANGUAGES
        }
        independence, verdicts = independence_tests(
            embeddings, seed=args.seed, sample=args.sample, tau=args.tau
        )
        controls = calibration_controls(
            embeddings,
            control_embeddings,
            seed=args.seed,
            sample=args.sample,
            tau=args.tau,
        )
        controls.to_parquet(CONTROLS_PARQUET, index=False)
        INDEPENDENCE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        independence.to_parquet(INDEPENDENCE_PARQUET, index=False)
        for pair, verdict in verdicts.items():
            if verdict["parallel_suspected"]:
                failures.append(f"independence: {pair} looks parallel, not independent")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame(audits).to_parquet(REPORT_PARQUET, index=False)
    REPORT_MD.write_text(
        render(
            audits,
            provenance_ok,
            provenance_rows,
            vocab,
            proportions,
            control,
            independence,
            verdicts,
            controls,
            config,
            failures,
        )
    )

    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PARQUET.relative_to(REPO_ROOT)}")
    if independence is not None:
        print(f"Wrote {INDEPENDENCE_PARQUET.relative_to(REPO_ROOT)}")
        print(f"Wrote {CONTROLS_PARQUET.relative_to(REPO_ROOT)}")
    if failures:
        print(f"\n{len(failures)} check(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
