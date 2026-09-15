"""Capacity-dilution analysis (CLAUDE5.md T-505).

The question Phase 5 exists to answer: does the Indic-specialised encoder beat
the massively-multilingual one, and where? Conneau et al. [13] describe the
trade-off — pretraining on more languages buys positive transfer between them
but spends a fixed model capacity across more of them, and past some point the
dilution wins.

This module assembles the evidence for that question and computes nothing that
another module already owns. Task numbers are **read back** from
`reports/encoder_comparison.parquet` and `reports/baselines.parquet` rather
than recomputed, for the reason T-114 gives: a figure typed into a second place
drifts from the first the moment anything is regenerated, and it drifts
silently.

Two measurements are new here, because nothing else in the project makes them:

* **Where each encoder's parameters actually sit** — split into the embedding
  table and the transformer body. The split matters more than the total,
  because IndicBERT-v2 is ALBERT-based and shares one layer's weights across
  all twelve, so its body is far smaller than its parameter count suggests.
* **How finely each tokenizer cuts Indic text** — tokens per sentence, on the
  *same* sentences in four languages. This is the concrete form of the
  specialisation advantage an Indic encoder is chosen for, and it is
  measurable without training anything.

Both are cheap, CPU-only, and need no weights: the parameter profile builds on
the `meta` device, and tokenisation needs only the tokenizer.

This module reports the trade-off; it does not adjudicate a mechanism it cannot
observe. Where a reading of the numbers is a hypothesis rather than a
measurement, the renderer says so.
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from .data import ENCODERS, get_tokenizer, load_split
from .download_dataset.paths import REPO_ROOT

COMPARISON_PARQUET = REPO_ROOT / "reports" / "encoder_comparison.parquet"
BASELINES_PARQUET = REPO_ROOT / "reports" / "baselines.parquet"

# Block H holds one set of items in four languages (§2), so tokens-per-sentence
# is comparable across languages: the same content, differently written.
FRAGMENTATION_TASK = 2
FRAGMENTATION_BLOCK = "H"
FRAGMENTATION_LANGS = ("hin", "ben", "tel", "mal")

# ALBERT shares one layer's parameters across all hidden layers, so the body
# count below is what exists, not what is executed. Recording it keeps the
# "IndicBERT is 5x smaller" reading from hiding an 11x difference.
SHARES_LAYERS = {"indicbert-v2"}


# --------------------------------------------------------------------------
# Where the parameters sit
# --------------------------------------------------------------------------


def parameter_profile(encoders=None) -> pd.DataFrame:
    """Per encoder: total, embedding and body parameters, plus vocabulary size.

    Built on the ``meta`` device, so it allocates nothing and downloads no
    weights.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    keys = tuple(encoders) if encoders else tuple(ENCODERS)
    rows = []
    for key in keys:
        config = AutoConfig.from_pretrained(ENCODERS[key].hf_id)
        with torch.device("meta"):
            model = AutoModel.from_config(config)
        total = sum(p.numel() for p in model.parameters())
        embedding = sum(
            p.numel() for name, p in model.named_parameters() if "embeddings" in name
        )
        rows.append(
            {
                "encoder": key,
                "hf_id": ENCODERS[key].hf_id,
                "total_params": total,
                "embedding_params": embedding,
                "body_params": total - embedding,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "layers": config.num_hidden_layers,
                "shares_layer_params": key in SHARES_LAYERS,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# How finely each tokenizer cuts the same sentences
# --------------------------------------------------------------------------


def fragmentation_profile(
    encoders: tuple[str, ...],
    *,
    task: int = FRAGMENTATION_TASK,
    block: str = FRAGMENTATION_BLOCK,
    langs: tuple[str, ...] = FRAGMENTATION_LANGS,
) -> pd.DataFrame:
    """Mean tokens per sentence, per language, per encoder.

    Every language here holds the *same items* (§2's 4-way parallel block), so
    a difference between languages is a difference in how the tokenizer handles
    the script, not in what the sentences say. Special tokens are excluded so
    the count reflects the text alone.
    """
    tokenizers = {key: get_tokenizer(key) for key in encoders}
    rows = []
    for lang in langs:
        origin = "native" if lang == _native_lang(block) else "mt"
        frame = load_split(task, block, lang, origin)
        text = [str(x) for x in frame["text"]]
        record = {"lang": lang, "origin": origin, "n_sentences": len(text)}
        for key, tokenizer in tokenizers.items():
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            lengths = np.array([len(ids) for ids in encoded])
            unk_id = tokenizer.unk_token_id
            n_unk = (
                sum(int((np.asarray(ids) == unk_id).sum()) for ids in encoded)
                if unk_id is not None
                else 0
            )
            record[f"{key}__tokens_per_sentence"] = float(lengths.mean())
            record[f"{key}__unk_per_1k_tokens"] = (
                1000.0 * n_unk / max(1, int(lengths.sum()))
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _native_lang(block: str) -> str:
    from .ids import BLOCK_NATIVE_LANG

    return BLOCK_NATIVE_LANG[block]


def fragmentation_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """How many more tokens ``numerator`` needs than ``denominator``, per language."""
    return (
        frame[f"{numerator}__tokens_per_sentence"]
        / frame[f"{denominator}__tokens_per_sentence"]
    )


# --------------------------------------------------------------------------
# Where the task deficit falls: in-language versus cross-lingual
# --------------------------------------------------------------------------


def in_language_deficit(baseline: str, other: str, *, path=None) -> pd.DataFrame:
    """Per task, the held-out in-language macro-F1 of both encoders.

    This is the fair in-language comparison — the held-out **test** fold, not
    the in-language ceiling, which is the model's own training text and says
    nothing about generalisation.
    """
    frame = pd.read_parquet(path or BASELINES_PARQUET)
    rows = []
    for task, group in frame.groupby(frame["config"].str[4].astype(int), sort=True):
        record = {"task": int(task)}
        for encoder in (baseline, other):
            sub = group[group["encoder"] == encoder]
            if sub.empty:
                record[f"{encoder}__mean"] = None
                record[f"{encoder}__std"] = None
                continue
            f1s = [float(x) for x in sub["macro_f1"]]
            record[f"{encoder}__mean"] = statistics.fmean(f1s)
            record[f"{encoder}__std"] = (
                statistics.stdev(f1s) if len(f1s) > 1 else 0.0
            )
        if record.get(f"{baseline}__mean") is not None and record.get(f"{other}__mean") is not None:
            record["delta"] = record[f"{other}__mean"] - record[f"{baseline}__mean"]
        else:
            record["delta"] = None
        rows.append(record)
    return pd.DataFrame(rows)


def cross_lingual_deficit(baseline: str, other: str, *, path=None) -> pd.DataFrame:
    """Per task, the mean target-language deficit over the computed cells.

    Read back from T-504's parquet — the same rows that report renders, so the
    two documents cannot disagree.
    """
    frame = pd.read_parquet(path or COMPARISON_PARQUET)
    ok = frame[frame["status"] == "ok"]
    rows = []
    for task, group in ok.groupby("task", sort=True):
        record = {"task": int(task), "n_cells": 0}
        by_encoder = {
            enc: sub.set_index("condition")
            for enc, sub in group.groupby("encoder", sort=False)
        }
        if baseline not in by_encoder or other not in by_encoder:
            rows.append({**record, "delta": None, "baseline_mean": None, "other_mean": None})
            continue
        shared = by_encoder[baseline].index.intersection(by_encoder[other].index)
        if shared.empty:
            rows.append({**record, "delta": None, "baseline_mean": None, "other_mean": None})
            continue
        base_targets = by_encoder[baseline].loc[shared, "target_mean"]
        other_targets = by_encoder[other].loc[shared, "target_mean"]
        rows.append(
            {
                "task": int(task),
                "n_cells": int(len(shared)),
                "baseline_mean": float(base_targets.mean()),
                "other_mean": float(other_targets.mean()),
                "delta": float((other_targets - base_targets).mean()),
                "baseline_gap_mean": float(
                    by_encoder[baseline].loc[shared, "gap_mean"].mean()
                ),
                "other_gap_mean": float(
                    by_encoder[other].loc[shared, "gap_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def deficit_summary(baseline: str, other: str) -> pd.DataFrame:
    """In-language and cross-lingual deficits side by side, per task.

    The comparison the analysis turns on: if the weaker encoder is weak
    *everywhere*, that is a capacity story; if it is competitive in-language
    and collapses across languages, capacity alone does not explain it.
    """
    in_lang = in_language_deficit(baseline, other).set_index("task")
    cross = cross_lingual_deficit(baseline, other).set_index("task")
    rows = []
    for task in sorted(set(in_lang.index) | set(cross.index)):
        rows.append(
            {
                "task": task,
                "in_language_delta": in_lang["delta"].get(task),
                "cross_lingual_delta": cross["delta"].get(task),
                "n_cells": cross["n_cells"].get(task),
            }
        )
    frame = pd.DataFrame(rows)
    frame["amplification"] = frame.apply(
        lambda r: (
            None
            if not r["in_language_delta"] or r["cross_lingual_delta"] is None
            else r["cross_lingual_delta"] / r["in_language_delta"]
        ),
        axis=1,
    )
    return frame


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _m(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{digits}f}"


def _signed(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:+.{digits}f}"


def render_parameters(frame: pd.DataFrame) -> list[str]:
    lines = [
        "## Where each encoder's capacity sits",
        "",
        "Totals hide the thing that matters. IndicBERT-v2 is ALBERT-based and "
        "shares one layer's weights across all twelve, so most of its "
        "parameter count is the embedding table and its transformer body is "
        "very small. Counted on the `meta` device, so no weights were loaded.",
        "",
        "| encoder | total | embeddings | transformer body | vocab | layers | shares layer params |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['encoder']} | {row['total_params'] / 1e6:.1f}M | "
            f"{row['embedding_params'] / 1e6:.1f}M | {row['body_params'] / 1e6:.1f}M | "
            f"{int(row['vocab_size']):,} | {int(row['layers'])} | "
            f"{'yes' if row['shares_layer_params'] else 'no'} |"
        )
    lines.append("")
    return lines


def render_fragmentation(
    frame: pd.DataFrame, encoders: tuple[str, ...], *, numerator: str, denominator: str
) -> list[str]:
    lines = [
        "## Where the Indic-specialised encoder does win: tokenisation",
        "",
        "Mean tokens per sentence on **the same items** in four languages "
        "(task 2, block H — every arm holds the same content, §2), special "
        "tokens excluded. Fewer tokens for the same sentence means the "
        "vocabulary fits the script better, which is the concrete advantage an "
        "Indic-specialised encoder is chosen for.",
        "",
        "| language | origin | sentences | "
        + " | ".join(f"{e} tokens" for e in encoders)
        + f" | {numerator} ÷ {denominator} | "
        + " | ".join(f"{e} UNK/1k" for e in encoders)
        + " |",
        "|" + "---|" * (4 + 2 * len(encoders)),
    ]
    ratios = fragmentation_ratio(frame, numerator, denominator)
    for i, (_, row) in enumerate(frame.iterrows()):
        cells = [str(row["lang"]), str(row["origin"]), str(int(row["n_sentences"]))]
        cells += [_m(row[f"{e}__tokens_per_sentence"], 1) for e in encoders]
        cells.append(f"{ratios.iloc[i]:.2f}×")
        cells += [_m(row[f"{e}__unk_per_1k_tokens"], 2) for e in encoders]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_deficit(frame: pd.DataFrame, *, baseline: str, other: str) -> list[str]:
    lines = [
        "## Where the deficit falls: in-language versus across languages",
        "",
        f"`delta` is {other} minus {baseline}; positive means {other} scored "
        "higher. In-language is the held-out **test** fold (T-206), not the "
        "in-language ceiling, which is the model's own training text. "
        "Cross-lingual is the mean over the computed transfer cells (T-504). "
        "`amplification` is the ratio of the two — how much bigger the "
        "deficit becomes once the model has to cross a language boundary.",
        "",
        "| task | in-language Δ | cross-lingual Δ | cells | amplification |",
        "|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        amp = row["amplification"]
        lines.append(
            f"| {int(row['task'])} | {_signed(row['in_language_delta'])} | "
            f"{_signed(row['cross_lingual_delta'])} | "
            f"{'—' if row['n_cells'] is None else int(row['n_cells'])} | "
            f"{'—' if amp is None else f'{amp:.1f}×'} |"
        )
    lines.append("")
    return lines
