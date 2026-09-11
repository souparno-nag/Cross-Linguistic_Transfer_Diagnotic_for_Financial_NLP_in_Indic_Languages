"""T-305 — run every evaluation condition, 3 seeds each.

Reads `configs/eval_conditions.json` (T-113) rather than constructing
conditions ad hoc, per the working agreement. Phase 2 only trained Hindi
baselines (`H_nat`) — Bengali and Telugu ones are deferred, not cancelled — so
a condition whose checkpoint does not exist yet is *blocked*, not skipped
silently: `plan()` reports every condition either way, and running the plan
only executes the ones whose checkpoint is on disk.

One checkpoint, `task{N}_{lang}_indicbert_seed{S}`, serves every condition
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
from .data import num_labels
from .download_dataset.paths import REPO_ROOT
from .experiments import RESULTS_PATH, log_run
from .ids import BLOCK_NATIVE_LANG
from .inference import load_frozen_model
from .predictions import prediction_log_from_loaded, write_prediction_log
from .transfer import DEFAULT_N_BOOT, bootstrap_metric

CONDITIONS_PATH = REPO_ROOT / "configs" / "eval_conditions.json"
ENCODER_KEY = "indicbert-v2"
SEEDS = (0, 1, 2)


def load_matrix(task: int) -> dict:
    """The committed T-113 matrix for one task -- never rebuilt ad hoc."""
    all_matrices = json.loads(CONDITIONS_PATH.read_text())
    key = f"task_{task}"
    if key not in all_matrices:
        raise KeyError(f"{CONDITIONS_PATH} has no {key!r}; run scripts.t113_conditions first")
    return all_matrices[key]


def run_id_for(task: int, lang: str, seed: int) -> str:
    return f"task{task}_{lang}_indicbert_seed{seed}"


def _split_lang(split: str) -> tuple[str, str]:
    """`"task_2/H/hin"` -> `("H", "hin")`."""
    _, block, lang = split.split("/")
    return block, lang


def _origin_for(block: str, lang: str) -> str:
    return "native" if BLOCK_NATIVE_LANG[block] == lang else "mt"


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


def plan(task: int, matrix: dict | None = None, *, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
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
            run_id = run_id_for(task, lang, seed)
            out.append(
                {
                    "condition": condition["name"],
                    "kind": condition["kind"],
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
) -> list[dict]:
    """Predict every arm of one condition from one checkpoint, write the
    prediction log, bootstrap each arm's macro-F1, and log a row per arm to
    `experiments.csv`. Returns the logged rows.
    """
    loaded = load_frozen_model(run_id, device=device)
    arms = condition["eval"] if isinstance(condition["eval"], list) else [condition["eval"]]
    labels = num_labels(task)
    rows = []
    for arm in arms:
        block, lang = _split_lang(arm)
        origin = _origin_for(block, lang)
        frame = prediction_log_from_loaded(
            loaded, condition["name"], task, block, lang, origin, seed, batch_size=batch_size
        )
        write_prediction_log(frame, task, condition["name"])

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
) -> dict:
    """Run every runnable (condition, seed) pair for one task.

    Returns `{"ran": [...], "blocked": [...]}` -- `blocked` entries name the
    checkpoint that would be needed, so the gap is visible rather than silent.
    """
    say = progress or (lambda _msg: None)
    matrix = load_matrix(task)
    conditions_by_name = {c["name"]: c for c in matrix["conditions"]}

    ran, blocked = [], []
    for entry in plan(task, matrix):
        if not entry["runnable"]:
            blocked.append(entry)
            say(f"BLOCKED {entry['condition']} seed{entry['seed']}: no checkpoint {entry['run_id']}")
            continue
        rows = run_condition(
            task, conditions_by_name[entry["condition"]], entry["run_id"], entry["seed"],
            device=device, batch_size=batch_size, n_boot=n_boot, results_path=results_path,
        )
        ran.extend(rows)
        say(f"ran {entry['condition']} seed{entry['seed']}: {len(rows)} arm(s)")
    return {"ran": ran, "blocked": blocked}
