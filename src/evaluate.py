"""T-305 — run every evaluation condition, 3 seeds each.

Reads `configs/eval_conditions.json` (T-113) rather than constructing
conditions ad hoc, per the working agreement. Phase 2 only trained Hindi
baselines (`H_nat`) — Bengali and Telugu ones are deferred, not cancelled — so
a condition whose checkpoint does not exist yet is *blocked*, not skipped
silently: `plan()` reports every condition either way, and running the plan
only executes the ones whose checkpoint is on disk.

One checkpoint, `task{N}_{lang}_{slug}_seed{S}` where `slug` is the encoder's
short name (`src.data`), serves every condition
whose model is "the one trained on `lang`" — the `transfer`/`transfer_mt`
family (train language named directly) and `translationese_{lang}` (the
model at home in that language, evaluated on its own three provenances). It
is loaded once per (task, lang, seed) and reused across every condition that
needs it, rather than once per condition (`prediction_log_from_loaded`).

Each condition contributes one row per **arm** to `experiments.csv` —
`transfer`/`transfer_mt` have one arm, `translationese` has three — and its
full per-instance predictions to `data/predictions/task_{n}/{condition}.parquet`
(T-302), so both the aggregate and the instance-level record land in the
outputs CLAUDE3.md's own table names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import checkpoints
from .audit import config_hash as _hash_dict
from .conditions import DEFAULT_SEED as CONDITIONS_SEED
from .conditions import partition_of
from .data import DEFAULT_ENCODER, num_labels, resolve_encoder
from .download_dataset.paths import REPO_ROOT
from .experiments import RESULTS_PATH, log_run
from .ids import BLOCK_NATIVE_LANG
from .inference import load_frozen_model
from .predictions import prediction_log_from_loaded, write_prediction_log
from .transfer import DEFAULT_N_BOOT, bootstrap_metric

CONDITIONS_PATH = REPO_ROOT / "configs" / "eval_conditions.json"
ENCODER_KEY = DEFAULT_ENCODER
SEEDS = (0, 1, 2)


def load_matrix(task: int) -> dict:
    """The committed T-113 matrix for one task -- never rebuilt ad hoc."""
    all_matrices = json.loads(CONDITIONS_PATH.read_text())
    key = f"task_{task}"
    if key not in all_matrices:
        raise KeyError(f"{CONDITIONS_PATH} has no {key!r}; run scripts.t113_conditions first")
    return all_matrices[key]


def run_id_for(task: int, lang: str, seed: int, encoder: str = ENCODER_KEY) -> str:
    """The checkpoint name T-206 writes for this (task, lang, encoder, seed).

    Built from the encoder's ``slug`` (``src.data``) rather than a literal, so
    a second encoder's checkpoints are addressable. This function is called for
    languages that have *no* checkpoint — Bengali and Telugu baselines are
    deferred (CLAUDE2.md) — and must still return the name that *would* be used
    so `plan` can report the condition as blocked instead of raising.
    """
    return f"task{task}_{lang}_{resolve_encoder(encoder).slug}_seed{seed}"


def _split_lang(split: str) -> tuple[str, str]:
    """`"task_2/H/hin"` -> `("H", "hin")`."""
    _, block, lang = split.split("/")
    return block, lang


def _origin_for(block: str, lang: str) -> str:
    return "native" if BLOCK_NATIVE_LANG[block] == lang else "mt"


def _item_filter_for(condition: dict, partition_seed: int):
    """A `str -> bool` predicate restricting a condition's arm to its own
    `eval_items` -- `None` (no filtering) unless the condition says
    `"partition:eval"` (`transfer`/`transfer_mt` on a partitioned task, T-113).

    Evaluating the *unfiltered* split would include items whose `item_id` was
    in the model's own training half in another language -- for tasks 2/3
    that item_id is shared across languages (T-102b), so that half is not
    zero-shot at all, just the same content relayed through a script change.
    `translationese` conditions set `eval_items: "all"` deliberately (their
    native arm *is* the training text, labelled explicitly per hard rule 4)
    and are correctly left unfiltered here.
    """
    if condition.get("eval_items") != "partition:eval":
        return None
    return lambda item_id: partition_of(item_id, seed=partition_seed) == "eval"


def _model_lang(condition: dict) -> str:
    """Which trained language's checkpoint evaluates this condition.

    `transfer`/`transfer_mt`: the language named in `train`. `translationese`:
    the language every arm is *in* -- its first arm is always that language's
    own native block (§8: "one written by people, two machine translated").
    """
    if condition["kind"] == "translationese":
        _, lang = _split_lang(condition["eval"][0])
        return lang
    _, lang = _split_lang(condition["train"])
    return lang


# --------------------------------------------------------------------------
# Planning: every (condition, seed), whether runnable or blocked
# --------------------------------------------------------------------------


def plan(
    task: int,
    matrix: dict | None = None,
    *,
    seeds: tuple[int, ...] = SEEDS,
    encoder: str = ENCODER_KEY,
) -> list[dict]:
    """Every (condition, seed) pair the matrix calls for.

    Each entry names the checkpoint it needs and whether that checkpoint
    currently exists -- callers report the blocked ones rather than silently
    dropping them (working agreement: "report row-count anomalies
    immediately, do not work around them" applies just as much to a missing
    checkpoint as a missing row).
    """
    matrix = matrix if matrix is not None else load_matrix(task)
    out = []
    for condition in matrix["conditions"]:
        lang = _model_lang(condition)
        for seed in seeds:
            run_id = run_id_for(task, lang, seed, encoder)
            out.append(
                {
                    "condition": condition["name"],
                    "kind": condition["kind"],
                    "encoder": encoder,
                    "model_lang": lang,
                    "seed": seed,
                    "run_id": run_id,
                    "runnable": checkpoints.exists(run_id),
                }
            )
    return out


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _EvalRunConfig:
    """Duck-typed for `experiments.log_run` (encoder / split_id / seed / hash)
    -- a zero-shot evaluation row has no `RunConfig` of its own to reuse."""

    encoder: str
    _split: str
    seed: int
    _hash: str

    def split_id(self) -> str:
        return self._split

    def hash(self) -> str:
        return self._hash


def run_condition(
    task: int,
    condition: dict,
    run_id: str,
    seed: int,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    n_boot: int = DEFAULT_N_BOOT,
    results_path=RESULTS_PATH,
    partition_seed: int = CONDITIONS_SEED,
    encoder: str = ENCODER_KEY,
) -> list[dict]:
    """Predict every arm of one condition from one checkpoint, write the
    prediction log, bootstrap each arm's macro-F1, and log a row per arm to
    `experiments.csv`. Returns the logged rows.

    `partition_seed` must match whatever seed built `eval_conditions.json`
    (T-113) -- it is what turns `condition["eval_items"] ==
    "partition:eval"` into an actual item filter (`_item_filter_for`).
    """
    loaded = load_frozen_model(run_id, device=device)
    if loaded.run_config.encoder != encoder:
        raise ValueError(
            f"checkpoint {run_id!r} was trained with encoder "
            f"{loaded.run_config.encoder!r}, not the requested {encoder!r} -- "
            "refusing to file its predictions under the wrong encoder"
        )
    # The default encoder keeps the un-namespaced path its committed artefacts
    # already occupy; every other encoder gets its own level (predictions.log_path).
    path_encoder = None if encoder == DEFAULT_ENCODER else encoder
    arms = condition["eval"] if isinstance(condition["eval"], list) else [condition["eval"]]
    labels = num_labels(task)
    item_filter = _item_filter_for(condition, partition_seed)
    rows = []
    for arm in arms:
        block, lang = _split_lang(arm)
        origin = _origin_for(block, lang)
        frame = prediction_log_from_loaded(
            loaded, condition["name"], task, block, lang, origin, seed,
            batch_size=batch_size, item_filter=item_filter,
        )
        write_prediction_log(frame, task, condition["name"], encoder=path_encoder)

        boot = bootstrap_metric(
            frame["gold"].tolist(), frame["pred"].tolist(), labels,
            n_boot=n_boot, seed=seed,
        )
        accuracy = float((frame["gold"] == frame["pred"]).mean())
        cfg = _EvalRunConfig(
            encoder=loaded.run_config.encoder,
            _split=f"{condition['name']}::{arm}",
            seed=seed,
            _hash=_hash_dict(
                {
                    "checkpoint_run_id": run_id,
                    "condition": condition["name"],
                    "arm": arm,
                    "n_boot": n_boot,
                    "bootstrap_seed": seed,
                }
            ),
        )
        row = log_run(
            cfg,
            f"{run_id}__{condition['name']}__{lang}",
            {"accuracy": accuracy, "macro_f1": boot.point},
            path=results_path,
        )
        rows.append({**row, "ci_low": boot.ci_low, "ci_high": boot.ci_high, "n": len(frame)})
    return rows


def run_all(
    task: int,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    n_boot: int = DEFAULT_N_BOOT,
    results_path=RESULTS_PATH,
    progress=None,
    encoder: str = ENCODER_KEY,
) -> dict:
    """Run every runnable (condition, seed) pair for one task.

    Returns `{"ran": [...], "blocked": [...]}` -- `blocked` entries name the
    checkpoint that would be needed, so the gap is visible rather than silent.
    """
    say = progress or (lambda _msg: None)
    matrix = load_matrix(task)
    conditions_by_name = {c["name"]: c for c in matrix["conditions"]}

    ran, blocked = [], []
    for entry in plan(task, matrix, encoder=encoder):
        if not entry["runnable"]:
            blocked.append(entry)
            say(f"BLOCKED {entry['condition']} seed{entry['seed']}: no checkpoint {entry['run_id']}")
            continue
        rows = run_condition(
            task, conditions_by_name[entry["condition"]], entry["run_id"], entry["seed"],
            device=device, batch_size=batch_size, n_boot=n_boot, results_path=results_path,
            partition_seed=matrix["seed"], encoder=encoder,
        )
        ran.extend(rows)
        say(f"ran {entry['condition']} seed{entry['seed']}: {len(rows)} arm(s)")
    return {"ran": ran, "blocked": blocked}
