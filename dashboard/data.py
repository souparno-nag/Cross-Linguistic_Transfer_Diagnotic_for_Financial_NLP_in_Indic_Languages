"""Everything the dashboard reads, as plain pandas.

Read-only: every function here loads an artefact that an earlier phase wrote
and reshapes it for display. Nothing is recomputed and nothing is written.
No Streamlit import, so the tests can call these directly; `dashboard/cached.py`
wraps them for the app.
"""

from __future__ import annotations

import pandas as pd

from scripts.t408_source_comparison import find_nonconverged
from src.corpus_io import load_labels, read_block
from src.diagnostics import audit_path, path_encoder, read_diagnostics
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix
from src.findings import ENCODERS, LABEL_CAVEATS
from src.predictions import log_path
from src.results_tables import _source_id_for

SOURCE_COMPARISON = REPO_ROOT / "reports" / "source_comparison.parquet"
LABEL_SHARES = REPO_ROOT / "reports" / "phase7" / "label_shares.parquet"

TASK_NAMES = {2: "Task 2: sustainability (2 classes)", 3: "Task 3: ESG theme (10 classes)"}
LANG_NAMES = {"hin": "Hindi", "ben": "Bengali", "tel": "Telugu", "mal": "Malayalam"}
LANG_ORDER = ["Hindi", "Bengali", "Telugu", "Malayalam"]
ENCODER_NAMES = {"indicbert-v2": "IndicBERT-v2", "mbert-base": "mBERT"}
KIND_NAMES = {"transfer": "Native test text", "transfer_mt": "Machine-translated test text"}

#: Display order: unattributed first, so the share nothing explains is never
#: the last thing on the chart (CLAUDE7.md, "Lead with unattributed").
GROUP_ORDER = ["unattributed", "linguistic", "data", "model"]
LABEL_ORDER = [
    "unattributed",
    "translation_drift", "tokenizer_fragmentation", "morphological_masking", "terminology_gap",
    "provenance_specific", "cross_lingual_hard",
    "class_collapse", "seed_unstable",
]

GROUP_TEXT = {
    "unattributed": "No check fired. The pipeline cannot say why this failure happened.",
    "linguistic": "Something about the test sentence itself: its translation, how it is split into tokens, "
                  "grammatical endings, or which words the model attends to.",
    "data": "Something about the item: it fails only on one version of the test text, or in every test language.",
    "model": "Something about the trained model: it predicts one class for almost everything, "
             "or another training run gets the item right.",
}

LABEL_TEXT = {
    "unattributed": "No check fired.",
    "translation_drift": "The test sentence scores below the similarity threshold against the training-language "
                         "sentence it was paired with, so the two may not mean the same thing. When this fires, "
                         "no other check runs.",
    "tokenizer_fragmentation": "The model's tokenizer splits the test sentence into at least twice as many pieces "
                               "as the training-language sentence.",
    "morphological_masking": "A key ESG term appears in the test sentence only with a grammatical ending attached. "
                             "Never fires on Hindi, which writes those endings as separate words.",
    "terminology_gap": "The model pays much less attention to a key ESG term in the test language than in the "
                       "training language.",
    "provenance_specific": "The same item is answered correctly on the other version of the test text "
                           "(native vs machine-translated).",
    "cross_lingual_hard": "Trained on the same language, the model gets this item wrong in all three test languages.",
    "class_collapse": "The model predicts one class for at least 75% of the test sentences, and this answer is "
                      "that class.",
    "seed_unstable": "Another training run of the same setup gets this item right.",
}

#: Audit columns shown per run in the explorer, in the order the checks run.
CHECKS = [
    ("gate_status", "Translation similarity"),
    ("frag_status", "Token splitting"),
    ("morph_status", "Grammatical endings"),
    ("saliency_status", "Attention to ESG terms"),
    ("provenance_status", "Other version of test text"),
    ("xling_status", "Fails in every test language"),
    ("collapse_status", "One-class predictions"),
    ("seed_status", "Other training runs"),
]


def label_caveat(label: str) -> str | None:
    """The stated limitation that must travel with this label (CLAUDE7.md hard rule 4)."""
    return LABEL_CAVEATS.get(label)


def class_names(task: int) -> list[str]:
    return load_labels(task)["labels"]


def untrained_runs() -> dict[str, set[int]]:
    """Run-id prefix (`task3_tel_indicbert`) -> seeds that never fit their training data."""
    return find_nonconverged()


def run_prefix(task: int, source: str, encoder: str) -> str:
    slug = {"indicbert-v2": "indicbert", "mbert-base": "mbert"}[encoder]
    return f"task{task}_{source}_{slug}"


# --------------------------------------------------------------------------
# Transfer page
# --------------------------------------------------------------------------


def transfer_gaps(task: int) -> pd.DataFrame:
    """One row per (encoder, source, target) on native test text, from T-408.

    T-408 leaves out runs that never trained; `excluded_seeds` says which.
    """
    frame = pd.read_parquet(SOURCE_COMPARISON)
    frame = frame[frame["task"] == task].copy()
    frame["model"] = frame["encoder"].map(ENCODER_NAMES)
    frame["source_name"] = frame["source"].map(LANG_NAMES)
    frame["target_name"] = frame["target"].map(LANG_NAMES)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Causes page
# --------------------------------------------------------------------------


def cause_shares(task: int) -> pd.DataFrame:
    """Phase 7 label shares for every condition of one task, all seeds."""
    frame = pd.read_parquet(LABEL_SHARES)
    frame = frame[(frame["task"] == task) & (frame["seed_set"] == "all")].copy()
    frame["model"] = frame["encoder"].map(ENCODER_NAMES)
    frame["pair"] = frame["source"].map(LANG_NAMES) + " → " + frame["target"].map(LANG_NAMES)
    order = GROUP_ORDER + LABEL_ORDER
    frame["order"] = frame["measure"].map({m: i for i, m in enumerate(order)})
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Explorer page
# --------------------------------------------------------------------------


def conditions(task: int) -> list[dict]:
    """The 18 transfer conditions of one task with their splits spelled out."""
    out = []
    for c in load_matrix(task)["conditions"]:
        if c["kind"] not in KIND_NAMES:
            continue
        train_block, source = _split_lang(c["train"])
        eval_block, target = _split_lang(c["eval"])
        native = c["kind"] == "transfer" and target != "mal"
        out.append({
            "name": c["name"], "kind": c["kind"], "source": source, "target": target,
            "train_block": train_block, "eval_block": eval_block,
            "ceiling": _source_id_for(c),
            "eval_provenance": c.get("eval_provenance") or source,
            "native_eval": native,
        })
    return out


def describe_test_text(condition: dict) -> str:
    target = LANG_NAMES[condition["target"]]
    if condition["native_eval"]:
        return f"{target}, written by people"
    return f"{target}, machine-translated from {LANG_NAMES[condition['eval_provenance']]}"


def corpus_versions(task: int) -> pd.DataFrame:
    """Every frozen row of the task: each item in every block and language."""
    frames = []
    for block in ("H", "B", "T"):
        for frame in read_block(task, block, frozen=True).values():
            frames.append(frame[["block_id", "item_id", "lang", "origin", "src_lang", "text", "flags"]])
    return pd.concat(frames, ignore_index=True)


def failures(task: int, condition: dict) -> pd.DataFrame:
    """Every failed (item, seed) of the condition on both encoders, with the
    Phase 6 label and every check's status."""
    frames = []
    for encoder in ENCODERS:
        enc = path_encoder(encoder)
        labels = read_diagnostics(task, condition["name"], encoder=enc)
        audit = pd.read_parquet(audit_path(task, condition["name"], encoder=enc))
        status_cols = [col for col, _ in CHECKS] + ["morph_evidence", "saliency_divergence"]
        merged = labels.merge(audit[["item_id", "seed", *status_cols]], on=["item_id", "seed"], how="left",
                              validate="one_to_one")
        frames.append(merged.assign(encoder=encoder))
    return pd.concat(frames, ignore_index=True)


def predictions(task: int, condition: dict) -> pd.DataFrame:
    """(encoder, item, seed) -> gold, prediction in the training language and
    in the test language. Only items the test split contains."""
    frames = []
    for encoder in ENCODERS:
        enc = path_encoder(encoder)
        target = pd.read_parquet(log_path(task, condition["name"], encoder=enc),
                                 columns=["item_id", "seed", "gold", "pred"])
        source = pd.read_parquet(log_path(task, condition["ceiling"], encoder=enc),
                                 columns=["item_id", "seed", "pred"])
        joined = target.rename(columns={"pred": "pred_tgt"}).merge(
            source.rename(columns={"pred": "pred_src"}), on=["item_id", "seed"], how="left", validate="one_to_one"
        )
        frames.append(joined.assign(encoder=encoder))
    return pd.concat(frames, ignore_index=True)


def failed_items(fails: pd.DataFrame, versions: pd.DataFrame, condition: dict) -> pd.DataFrame:
    """One row per failed item: how many runs failed on each encoder, the
    labels it got, and its test sentence. Most consistent failures first."""
    counts = fails.pivot_table(index="item_id", columns="encoder", values="seed", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=list(ENCODERS), fill_value=0)
    labels = fails.groupby("item_id")["assigned_label"].agg(lambda s: ", ".join(sorted(set(s))))
    test = versions[(versions["block_id"] == condition["eval_block"]) & (versions["lang"] == condition["target"])]
    out = pd.DataFrame({
        "item_id": counts.index,
        "IndicBERT-v2 runs failed": counts["indicbert-v2"].to_numpy(),
        "mBERT runs failed": counts["mbert-base"].to_numpy(),
        "causes": labels.reindex(counts.index).to_numpy(),
    })
    out = out.merge(test[["item_id", "text"]], on="item_id", how="left")
    out["total"] = out["IndicBERT-v2 runs failed"] + out["mBERT runs failed"]
    return out.sort_values(["total", "item_id"], ascending=[False, True]).drop(columns="total").reset_index(drop=True)
