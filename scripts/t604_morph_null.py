"""Does a masked lexicon term make a transfer failure more likely? The morphology
module's validity check — its counterpart to `t605_baseline`.

    python -m scripts.t604_morph_null [--task 2 --task 3] [--encoder ...] [--allow-fallback]

The morphological-masking module looks only at the target sentence, so it can be
run on every prediction, not just the failures. Over every (item, seed) whose
source prediction was right, each target sentence is put in one of three groups:

* `masked` — the module fires: a lexicon term appears only inflected;
* `intact` — a lexicon term appears as a whole word, none masked;
* `none`   — no lexicon term at all.

and the failure rate (`pred_tgt != gold`) is compared across them. If masking
explains failures, `masked` fails more often than `intact` — the closest control,
since both carry an ESG term. If the rates match, the label marks where inflection
happens, not why the model failed, and is reported as such.

Also compared within each condition (`masked` vs `intact` in the same transfer
cell, weighted by the masked count), so a difference in condition mix cannot
manufacture or hide an effect. The LaBSE gate is not applied: this asks about
the module's evidence, not about attribution.

CPU only; deterministic. Refuses the suffix-list fallback like `t604_remorph`.
Writes `reports/morphology_null.{md,parquet}`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import morphology
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER
from src.diagnostics import load_esg_terms, path_encoder
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix
from src.mismatch import join_source_target
from src.predictions import read_prediction_log

REPORT = REPO_ROOT / "reports" / "morphology_null.md"
GROUPS = ("masked", "intact", "none")


def term_group(analyzer, text: str, concepts: list[dict]) -> str:
    if morphology.check_instance(analyzer, text, concepts).status == "fired":
        return "masked"
    words = {morphology._clean_word(w) for w in text.split()}
    terms = (morphology._clean_word(c.get(analyzer.lang) or "") for c in concepts)
    return "intact" if any(t and t in words for t in terms) else "none"


def condition_rows(task: int, encoder: str, condition: dict, analyzers, concepts, groups_cache) -> pd.DataFrame:
    enc = path_encoder(encoder)
    block_src, lang_src = _split_lang(condition["train"])
    joined = join_source_target(
        read_prediction_log(task, f"{block_src}_{lang_src}_native_ceiling", encoder=enc),
        read_prediction_log(task, condition["name"], encoder=enc),
    )
    joined = joined[joined["pred_src"] == joined["gold"]]
    block, lang = str(joined["block_id"].iloc[0]), str(joined["lang"].iloc[0])
    key = (task, block, lang)
    if key not in groups_cache:
        frame = read_split(task, block, lang, frozen=True)
        groups_cache[key] = {i: term_group(analyzers[lang], str(t), concepts)
                             for i, t in zip(frame["item_id"], frame["text"])}
    return pd.DataFrame({
        "task": task, "encoder": encoder, "condition_id": condition["name"], "tgt_lang": lang,
        "item_id": joined["item_id"].values, "seed": joined["seed"].values,
        "group": joined["item_id"].map(groups_cache[key]).values,
        "failed": (joined["pred_tgt"] != joined["gold"]).values,
    })


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (task, encoder): failure rate and n per group, plus the
    condition-matched masked-vs-intact comparison."""
    out = []
    for (task, encoder), g in rows.groupby(["task", "encoder"]):
        rec = {"task": task, "encoder": encoder}
        for grp in GROUPS:
            sel = g.loc[g["group"] == grp, "failed"]
            rec[f"{grp}_n"] = len(sel)
            rec[f"{grp}_fail"] = sel.mean() if len(sel) else float("nan")
        num = den_m = den_i = 0.0
        worse = cells = 0
        for _, c in g.groupby("condition_id"):
            m, i = c.loc[c["group"] == "masked", "failed"], c.loc[c["group"] == "intact", "failed"]
            if len(m) and len(i):
                num += len(m)
                den_m += m.mean() * len(m)
                den_i += i.mean() * len(m)
                worse += m.mean() > i.mean()
                cells += 1
        rec["matched_masked_fail"] = den_m / num if num else float("nan")
        rec["matched_intact_fail"] = den_i / num if num else float("nan")
        rec["cells_masked_worse"] = f"{worse}/{cells}"
        out.append(rec)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]

    analyzers = {lang: morphology.MorphAnalyzer(lang) for lang in ("hin", "ben", "tel", "mal")}
    rungs = {lang: a.rung for lang, a in analyzers.items()}
    print("segmentation rung per language:", rungs)
    if not args.allow_fallback and any(r != "library" for r in rungs.values()):
        raise SystemExit("Indic NLP library rung did not load for every language (set INDIC_RESOURCES_PATH); "
                         "refusing to run on the suffix-list fallback without --allow-fallback")
    concepts = load_esg_terms()
    cache: dict = {}
    frames = []
    for task in tasks:
        for encoder in encoders:
            for condition in load_matrix(task)["conditions"]:
                if not condition["name"].startswith("transfer_"):
                    continue
                try:
                    frames.append(condition_rows(task, encoder, condition, analyzers, concepts, cache))
                except FileNotFoundError:
                    print(f"task_{task} {encoder} {condition['name']}: no prediction log, skipped")
    rows = pd.concat(frames, ignore_index=True)
    rows.to_parquet(REPORT.with_suffix(".parquet"), index=False)
    summary = summarise(rows)
    by_lang = (rows.groupby(["task", "tgt_lang", "group"])["failed"].agg(["mean", "size"])
               .rename(columns={"mean": "fail_rate", "size": "n"}).reset_index())
    lines = ["# Morphological masking — failure rate with and without a masked term", "",
             "Generated by `python -m scripts.t604_morph_null`. Unit: (item, seed) with a correct source "
             "prediction; `fail` = target prediction wrong. `masked`: the module fires on the target sentence; "
             "`intact`: a lexicon term is present unmasked; `none`: no lexicon term. `matched_*`: masked vs intact "
             "within the same condition, weighted by the masked count.", "",
             f"Segmentation rung: {rungs}.", "",
             summary.round(3).to_markdown(index=False), "",
             "## By target language", "",
             by_lang.round(3).to_markdown(index=False), "",
             "If `masked` fails no more often than `intact`, a masked term does not make failure more likely. "
             "`morphological_masking` is still assigned as a causal label (CLAUDE4.md Scope decisions, "
             "2026-10-12); this table is the limitation reported beside it.", ""]
    REPORT.write_text("\n".join(lines))
    print(summary.round(3).to_string(index=False))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
