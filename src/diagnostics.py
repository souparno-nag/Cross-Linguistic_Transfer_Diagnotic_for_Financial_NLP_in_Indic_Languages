"""Diagnostic pipeline scaffold (CLAUDE4.md T-601) — the two-stage router.

Stage 1 (semantic gate, `translation_drift`) is terminal: a failing instance
whose LaBSE similarity is below τ goes no further, by architecture, not by
convenience — CLAUDE4.md's hard rule 2 ("never reorder the stages") and its
own rationale (noise from the translation would otherwise be misattributed
to tokenization or morphology). Every gate-*passed* instance runs **all
three** remaining modules unconditionally — fragmentation, morphological
masking, terminology gap — so the audit trail records every module that
fired, not just the one precedence eventually picks (hard rule 1). The
orthographic/numeral module from CLAUDE4.md's original five-module design is
out of scope for this build (explicit instruction) and is simply absent from
`PRECEDENCE` below; `unattributed` covers everything nothing else caught.

This module owns orchestration and I/O; the actual per-module logic lives in
`src.fragmentation`, `src.morphology` and `src.saliency`, and the gate reuses
`src.labse_gate.score_items` rather than reimplementing it (CLAUDE4.md's
explicit instruction). `score_items`, not `score_pair`, because a
native-family condition's two blocks are only *partially* aligned by
construction (T-102b) — scoring a failure set's specific items, not an
entire split, is what keeps a legitimately-unaligned row elsewhere in the
corpus from raising here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import fragmentation, morphology, saliency
from . import labse_gate as labse
from .corpus_io import read_split
from .data import DEFAULT_ENCODER, get_tokenizer
from .download_dataset.paths import REPO_ROOT
from .evaluate import _split_lang, load_matrix
from .failures import failure_path, read_failures
from .inference import load_frozen_model

DIAGNOSTICS_ROOT = REPO_ROOT / "data" / "diagnostics"
AUDIT_ROOT = DIAGNOSTICS_ROOT / "audit"
ESG_TERMS_PATH = REPO_ROOT / "configs" / "esg_terms.json"

LABEL_COLUMNS = [
    "condition_id", "encoder_id", "item_id", "block_id", "src_lang", "tgt_lang",
    "assigned_label", "modules_fired", "r_frag", "labse_sim", "seed",
]

AUDIT_COLUMNS = [
    "condition_id", "encoder_id", "item_id", "block_id", "src_lang", "tgt_lang", "seed",
    "gate_status", "labse_sim",
    "frag_status", "r_frag",
    "morph_status", "morph_evidence",
    "saliency_status", "saliency_divergence", "saliency_convergence_error",
    "assigned_label", "modules_fired",
]

# CLAUDE4.md's resolution precedence, minus the excluded orthographic rung
# (dropped per this build's scope). `gate` is handled specially in
# `diagnose_row` (it is terminal and skips the other three modules entirely)
# but is listed here too so `precedence_pick` is the single source of truth
# for "which fired module wins."
PRECEDENCE = [
    ("gate", "translation_drift"),
    ("frag", "tokenizer_fragmentation"),
    ("morph", "morphological_masking"),
    ("saliency", "terminology_gap"),
]


def precedence_pick(modules_fired) -> str:
    """Exactly one label, by CLAUDE4.md's precedence order. `unattributed`
    when nothing fired — a legitimate, reportable outcome, not a gap to
    paper over (hard rule: "a pipeline that labels 100% of failures is
    suspicious")."""
    fired = set(modules_fired)
    for module, label in PRECEDENCE:
        if module in fired:
            return label
    return "unattributed"


def load_esg_terms(path: Path = ESG_TERMS_PATH) -> list[dict]:
    lexicon = json.loads(path.read_text())
    return lexicon["concepts"]


def path_encoder(encoder_id: str) -> str | None:
    """`None` for the default encoder, whose artefacts keep the original
    un-namespaced paths (the convention `evaluate.py` and `failures.py` already
    follow); any other encoder gets its own directory level. Without it a second
    encoder's labels would overwrite the first's, and the `encoder_id` column
    would be the only thing left saying whose they were."""
    return None if encoder_id == DEFAULT_ENCODER else encoder_id


def diagnostics_path(task: int, condition_id: str, *, root: Path | None = None, encoder: str | None = None) -> Path:
    root = root if root is not None else DIAGNOSTICS_ROOT
    base = root / f"task_{task}"
    return (base / encoder if encoder else base) / f"{condition_id}.parquet"


def audit_path(task: int, condition_id: str, *, root: Path | None = None, encoder: str | None = None) -> Path:
    root = root if root is not None else AUDIT_ROOT
    base = root / f"task_{task}"
    return (base / encoder if encoder else base) / f"{condition_id}.parquet"


def write_diagnostics(
    task: int, condition_id: str, labels: pd.DataFrame, audit: pd.DataFrame, *, encoder: str | None = None
) -> tuple[Path, Path]:
    label_out = diagnostics_path(task, condition_id, encoder=encoder)
    audit_out = audit_path(task, condition_id, encoder=encoder)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    labels[LABEL_COLUMNS].to_parquet(label_out, index=False)
    audit[AUDIT_COLUMNS].to_parquet(audit_out, index=False)
    return label_out, audit_out


def read_diagnostics(
    task: int, condition_id: str, *, root: Path | None = None, encoder: str | None = None
) -> pd.DataFrame:
    path = diagnostics_path(task, condition_id, root=root, encoder=encoder)
    if not path.exists():
        raise FileNotFoundError(f"no diagnostics for condition {condition_id!r} at {path}")
    return pd.read_parquet(path)


def discover_conditions(task: int, *, root: Path | None = None, encoder: str | None = None) -> list[dict]:
    """Every condition with a failures parquet on disk for `encoder` (`None` =
    the default encoder's un-namespaced directory), matched against
    `configs/eval_conditions.json` for its source `(block, lang)`. Generic
    over whatever exists, so new tasks, source blocks and encoders are picked
    up with no code change."""
    task_dir = failure_path(task, "x", root=root, encoder=encoder).parent
    matrix = load_matrix(task)
    by_name = {c["name"]: c for c in matrix["conditions"]}
    out = []
    for path in sorted(task_dir.glob("*.parquet")):
        condition_id = path.stem
        condition = by_name.get(condition_id)
        if condition is None:
            raise ValueError(f"{condition_id!r} has no entry in eval_conditions.json task_{task}")
        block_src, lang_src = _split_lang(condition["train"])
        out.append({"condition_id": condition_id, "block_src": block_src, "lang_src": lang_src})
    return out


# --------------------------------------------------------------------------
# Per-row diagnosis — pure, no I/O, fully unit-testable
# --------------------------------------------------------------------------


def diagnose_row(
    *,
    labse_sim: float,
    tau: float,
    source_text: str,
    target_text: str,
    gold_label: int,
    tokenizer,
    morph_analyzer,
    concepts: list[dict],
    lang_src: str,
    lang_tgt: str,
    loaded=None,
    max_len: int = 128,
    frag_threshold: float = fragmentation.DEFAULT_THRESHOLD,
    saliency_threshold: float = saliency.DEFAULT_DIVERGENCE_THRESHOLD,
    n_steps: int = saliency.DEFAULT_N_STEPS,
    device: str = "cpu",
) -> dict:
    """One instance's full audit row. Stage 1 is terminal: a fired gate skips
    every Stage-2 module rather than merely outranking them, per CLAUDE4.md's
    "never reorder the stages." A passed gate runs all three unconditionally.
    """
    if labse_sim < tau:
        return {
            "gate_status": "fired", "labse_sim": labse_sim,
            "frag_status": "not_evaluated", "r_frag": None,
            "morph_status": "not_evaluated", "morph_evidence": None,
            "saliency_status": "not_evaluated", "saliency_divergence": None,
            "saliency_convergence_error": None,
            "modules_fired": ["gate"], "assigned_label": precedence_pick(["gate"]),
        }

    ratio = fragmentation.r_frag(tokenizer, source_text, target_text)
    frag_fired = fragmentation.fires(ratio, frag_threshold)

    morph_result = morphology.check_instance(morph_analyzer, target_text, concepts)

    if loaded is not None:
        sal_result = saliency.diagnose_instance(
            loaded, tokenizer, source_text, target_text, gold_label, concepts,
            lang_src, lang_tgt, max_len=max_len, threshold=saliency_threshold,
            n_steps=n_steps, device=device,
        )
    else:
        sal_result = saliency.SaliencyResult(status="unavailable")

    modules_fired = []
    if frag_fired:
        modules_fired.append("frag")
    if morph_result.status == "fired":
        modules_fired.append("morph")
    if sal_result.status == "fired":
        modules_fired.append("saliency")

    return {
        "gate_status": "passed", "labse_sim": labse_sim,
        "frag_status": "fired" if frag_fired else "not_fired", "r_frag": ratio,
        "morph_status": morph_result.status, "morph_evidence": morph_result.evidence,
        "saliency_status": sal_result.status, "saliency_divergence": sal_result.divergence,
        "saliency_convergence_error": sal_result.convergence_delta,
        "modules_fired": modules_fired, "assigned_label": precedence_pick(modules_fired),
    }


# --------------------------------------------------------------------------
# Per-condition orchestration
# --------------------------------------------------------------------------


def diagnose_condition(
    task: int,
    condition_id: str,
    encoder_id: str,
    *,
    tau: float | None = None,
    device: str = "cpu",
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every failing instance of one condition, gated and (where the gate
    passes) run through all three Stage-2 modules. Returns `(labels, audit)`
    — both empty-but-correctly-shaped if the condition has zero failures,
    matching `export_failures`'s own "writing an empty file records that the
    check ran" precedent."""
    matrix = load_matrix(task)
    by_name = {c["name"]: c for c in matrix["conditions"]}
    condition = by_name.get(condition_id)
    if condition is None:
        raise ValueError(f"{condition_id!r} has no entry in eval_conditions.json task_{task}")
    block_src, lang_src = _split_lang(condition["train"])

    failures = read_failures(task, condition_id, root=root, encoder=path_encoder(encoder_id))
    if failures.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS), pd.DataFrame(columns=AUDIT_COLUMNS)

    for col in ("block_id", "lang"):
        if failures[col].nunique() != 1:
            raise ValueError(
                f"{condition_id!r}: failures file mixes multiple {col!r} values "
                f"({sorted(failures[col].unique())}) -- one condition file must "
                "describe one target arm"
            )
    block_tgt = str(failures["block_id"].iloc[0])
    lang_tgt = str(failures["lang"].iloc[0])

    config = labse.load_config()
    tau = tau if tau is not None else labse.tau_for(config, task)
    embedder = labse.Embedder(config, device=device)
    sims = labse.score_items(
        embedder, task, block_src, lang_src, block_tgt, lang_tgt, failures["item_id"].tolist()
    )
    sim_by_item = dict(zip(sims["item_id"], sims["labse_sim"]))
    missing = [item for item in failures["item_id"] if item not in sim_by_item]
    if missing:
        raise ValueError(
            f"{condition_id!r}: {len(missing)} failing instances have no LaBSE score, "
            f"e.g. {missing[:3]} -- rule 1 forbids silently dropping them"
        )

    src_frame = read_split(task, block_src, lang_src, frozen=True)
    tgt_frame = read_split(task, block_tgt, lang_tgt, frozen=True)
    src_text_by_item = dict(zip(src_frame["item_id"], src_frame["text"]))
    tgt_text_by_item = dict(zip(tgt_frame["item_id"], tgt_frame["text"]))

    tokenizer = get_tokenizer(encoder_id)
    morph_analyzer = morphology.MorphAnalyzer(lang_tgt)
    concepts = load_esg_terms()

    loaded_by_run: dict[str, object] = {}

    label_rows, audit_rows = [], []
    for _, failure_row in failures.iterrows():
        item_id = failure_row["item_id"]
        labse_sim = float(sim_by_item[item_id])
        run_id = failure_row["tgt_run_id"]
        if run_id not in loaded_by_run:
            loaded_by_run[run_id] = load_frozen_model(run_id, device=device)
        loaded = loaded_by_run[run_id]
        if loaded.run_config.encoder != encoder_id:
            raise ValueError(
                f"checkpoint {run_id!r} was trained with encoder "
                f"{loaded.run_config.encoder!r}, not the requested {encoder_id!r} -- "
                "the fragmentation ratio and saliency modules must use the same "
                "tokenizer the checkpoint was trained with"
            )

        row = diagnose_row(
            labse_sim=labse_sim,
            tau=tau,
            source_text=str(src_text_by_item[item_id]),
            target_text=str(tgt_text_by_item[item_id]),
            gold_label=int(failure_row["gold"]),
            tokenizer=tokenizer,
            morph_analyzer=morph_analyzer,
            concepts=concepts,
            lang_src=lang_src,
            lang_tgt=lang_tgt,
            loaded=loaded,
            max_len=loaded.run_config.max_len,
            device=device,
        )
        label_rows.append({
            "condition_id": condition_id, "encoder_id": encoder_id, "item_id": item_id,
            "block_id": block_tgt, "src_lang": lang_src, "tgt_lang": lang_tgt,
            "assigned_label": row["assigned_label"], "modules_fired": row["modules_fired"],
            "r_frag": row["r_frag"], "labse_sim": row["labse_sim"], "seed": int(failure_row["seed"]),
        })
        audit_rows.append({
            "condition_id": condition_id, "encoder_id": encoder_id, "item_id": item_id,
            "block_id": block_tgt, "src_lang": lang_src, "tgt_lang": lang_tgt,
            "seed": int(failure_row["seed"]), **row,
        })

    return pd.DataFrame(label_rows)[LABEL_COLUMNS], pd.DataFrame(audit_rows)[AUDIT_COLUMNS]
