"""Stage 3 of the diagnostic router — data and model-behaviour attribution
(CLAUDE4.md "Stage 3", added 2026-10-12).

Four modules that explain a failure by the item or by the model rather than by
a linguistic property of the target sentence. They run after the four Stage-1/2
modules and only on gate-passed rows (the gate stays terminal, hard rule 2), and
outrank nothing linguistic: precedence puts them last, so they only claim what
Stages 1-2 left unattributed.

| module       | label                 | group | fires when |
|--------------|-----------------------|-------|------------|
| `provenance` | `provenance_specific` | data  | the same (item, seed) is right in the condition's twin — native target vs MT target of the same language, same source |
| `xling`      | `cross_lingual_hard`  | data  | the same (item, seed), trained on the same source, fails in all 3 target languages |
| `collapse`   | `class_collapse`      | model | the run predicts one class on >= 75% of the condition's target rows, and this failure predicted that class |
| `seed`       | `seed_unstable`       | model | the same item is right on another seed of the same condition (with that seed's source also right) |

Every input is already on disk: the condition's prediction log, its twin's,
the source ceiling log, and the other conditions' failure sets. CPU only.

The cutoffs (75%, "all 3", "another seed") were fixed on 2026-10-12 before
these modules first ran; they are not to be moved to change the distribution
(hard rule 3). 75% is well above any task's largest gold class share (task 2:
55%), so a run that predicts one class that often is not tracking the labels.
"""

from __future__ import annotations

import pandas as pd

COLLAPSE_SHARE = 0.75
XLING_MIN_LANGS = 3

MODULES = ("provenance", "xling", "collapse", "seed")
STATUS_COLUMNS = [f"{m}_status" for m in MODULES]


def twin_condition(condition_id: str) -> str:
    """`transfer_hin_to_ben` <-> `transfer_hin_to_ben_mt`."""
    return condition_id[: -len("_mt")] if condition_id.endswith("_mt") else condition_id + "_mt"


def source_family(condition_id: str) -> str:
    """Conditions sharing a training source and a target provenance:
    `transfer_hin_to_ben_mt` -> `transfer_hin|mt`."""
    head = condition_id.split("_to_")[0]
    return f"{head}|{'mt' if condition_id.endswith('_mt') else 'native'}"


def _correct(log: pd.DataFrame) -> dict:
    return dict(zip(zip(log["item_id"], log["seed"]), log["pred"] == log["gold"]))


def provenance_status(keys, twin_log: pd.DataFrame | None) -> list[str]:
    if twin_log is None:
        return ["not_applicable"] * len(keys)
    right = _correct(twin_log)
    return ["not_applicable" if k not in right else ("fired" if right[k] else "not_fired") for k in keys]


def xling_status(keys, family: str, failed_langs: dict) -> list[str]:
    """`failed_langs[(family, item, seed)]` = target languages that failure set holds."""
    return ["fired" if len(failed_langs.get((family, *k), ())) >= XLING_MIN_LANGS else "not_fired" for k in keys]


def collapse_status(keys, preds, target_log: pd.DataFrame) -> list[str]:
    counts = target_log.groupby("seed")["pred"].value_counts(normalize=True)
    dominant = {seed: (cls, share) for (seed, cls), share in counts.groupby(level=0).head(1).items()}
    out = []
    for (_, seed), pred in zip(keys, preds):
        cls, share = dominant[seed]
        out.append("fired" if share >= COLLAPSE_SHARE and pred == cls else "not_fired")
    return out


def seed_status(keys, target_log: pd.DataFrame, source_log: pd.DataFrame) -> list[str]:
    src_right = _correct(source_log)
    tgt = target_log.assign(ok=target_log["pred"] == target_log["gold"])
    tgt = tgt[[src_right.get(k, False) for k in zip(tgt["item_id"], tgt["seed"])]]
    right_seeds = tgt[tgt["ok"]].groupby("item_id")["seed"].apply(set).to_dict()
    return ["fired" if right_seeds.get(item, set()) - {seed} else "not_fired" for item, seed in keys]


def failed_languages(audits: dict[str, pd.DataFrame]) -> dict:
    """From every condition's audit of one cell: (family, item, seed) -> set of
    target languages it failed in."""
    out: dict = {}
    for cond, audit in audits.items():
        fam = source_family(cond)
        for item, seed, lang in zip(audit["item_id"], audit["seed"], audit["tgt_lang"]):
            out.setdefault((fam, item, int(seed)), set()).add(lang)
    return out


def annotate(audit: pd.DataFrame, condition_id: str, *, target_log, source_log, twin_log, failed_langs) -> pd.DataFrame:
    """`audit` with the four `*_status` columns set. Gate-fired rows get
    `not_evaluated` (Stage 1 is terminal)."""
    audit = audit.copy()
    keys = list(zip(audit["item_id"], audit["seed"].astype(int)))
    tgt = target_log.set_index(["item_id", "seed"])
    preds = [tgt.at[k, "pred"] for k in keys]
    statuses = {
        "provenance": provenance_status(keys, twin_log),
        "xling": xling_status(keys, source_family(condition_id), failed_langs),
        "collapse": collapse_status(keys, preds, target_log),
        "seed": seed_status(keys, target_log, source_log),
    }
    gated = (audit["gate_status"] != "passed").tolist()
    for module, values in statuses.items():
        audit[f"{module}_status"] = ["not_evaluated" if g else v for g, v in zip(gated, values)]
    return audit
