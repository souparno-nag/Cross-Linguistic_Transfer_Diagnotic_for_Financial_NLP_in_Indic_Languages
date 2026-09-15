"""T-401's static audit, and the cross-block parity rule it enforces.

CLAUDE6.md hard rule 1 — "identical protocol to Phases 2/3 and 5 ... anything
that moves besides source block and encoder invalidates the comparison" — is
the rule this file turns into something CI checks rather than something a
reviewer has to remember. The failure it guards against is quiet: a Bengali
baseline carrying a learning rate Hindi never used still trains, still logs,
and still produces a number for T-408's table. The table would then report a
training-recipe difference as a source-language difference, and nothing in the
artefacts would say so.

The parity check is deliberately stricter than T-500's encoder one, which
permits a VRAM-forced re-shaping of `batch_size`/`grad_accum`. Changing the
source language changes neither the model nor the sequence length, so it
cannot force a different batch shape — there is no hardware reason for a block
to deviate, and a config that did would be tuning.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from scripts import t401_smoke as T
from src.config import TRAIN_CONFIG_DIR, load_run_config

PHASE4_LANGS = ("ben", "tel")


def _configs():
    return sorted(TRAIN_CONFIG_DIR.glob("*.yaml"))


def _phase4_configs():
    return [p for p in _configs() if load_run_config(p).lang in PHASE4_LANGS]


def _reference(task: int = 2, encoder: str = "indicbert-v2"):
    path = TRAIN_CONFIG_DIR / f"task{task}_hin_{'indicbert' if encoder == 'indicbert-v2' else 'mbert'}.yaml"
    return load_run_config(path), path


# --------------------------------------------------------------------------
# The shipped configs — hard rules 1 and 6
# --------------------------------------------------------------------------


def test_at_least_one_phase4_config_is_shipped():
    assert _phase4_configs(), (
        "no config names a Phase 4 source block — T-401 needs one added as a "
        "config entry"
    )


@pytest.mark.parametrize("path", _phase4_configs(), ids=lambda p: p.stem)
def test_phase4_config_matches_its_block_h_sibling(path):
    """Identical protocol across source blocks: only `lang` may differ."""
    run = load_run_config(path)
    check = T.check_block_parity(run, path)
    assert check["ok"], check["detail"]


@pytest.mark.parametrize("path", _phase4_configs(), ids=lambda p: p.stem)
def test_phase4_config_has_a_native_split_with_every_class(path):
    run = load_run_config(path)
    check = T.check_native_split_available(run)
    assert check["ok"], check["detail"]


@pytest.mark.parametrize("path", _phase4_configs(), ids=lambda p: p.stem)
def test_phase4_config_unblocks_real_conditions(path):
    """Training a cell has to turn blocked conditions into runnable ones.

    A checkpoint no condition names would be GPU time that bought nothing, and
    the naming convention is the only thing connecting the two.
    """
    run = load_run_config(path)
    check = T.check_conditions_unblocked(run)
    assert check["ok"], check["detail"]


def test_every_source_block_is_covered_for_each_encoder_and_task():
    """The grid is complete as configuration, whatever has actually been run.

    Phase 4 may stop part-way and report which cells ran (working agreement),
    but the configs should not be the reason a cell is missing.
    """
    shipped = {
        (r.task, r.lang, r.encoder)
        for r in (load_run_config(p) for p in _configs())
    }
    missing = [
        (task, lang, encoder)
        for task in (2, 3)
        for lang in ("hin", "ben", "tel")
        for encoder in ("indicbert-v2", "mbert-base")
        if (task, lang, encoder) not in shipped
    ]
    assert not missing, f"no config for {missing}"


# --------------------------------------------------------------------------
# The parity check itself — does it actually catch anything?
# --------------------------------------------------------------------------


def test_parity_catches_a_tuned_hyperparameter(tmp_path):
    """A learning rate tuned for one block is what hard rule 6 forbids."""
    reference, _ = _reference()
    tuned = dataclasses.replace(reference, lang="ben", lr=reference.lr * 2)
    check = T.check_block_parity(tuned, TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml")
    assert not check["ok"]
    assert "lr" in check["detail"]
    assert check["blocker"]


def test_parity_catches_a_changed_schedule(tmp_path):
    reference, _ = _reference()
    longer = dataclasses.replace(reference, lang="tel", epochs=reference.epochs + 10)
    check = T.check_block_parity(longer, TRAIN_CONFIG_DIR / "task2_tel_indicbert.yaml")
    assert not check["ok"]
    assert "epochs" in check["detail"]


def test_parity_catches_a_changed_holdout_fraction():
    """A different dev/test split changes what the metric even means, and it
    is not a hyperparameter so it is easy to overlook."""
    reference, _ = _reference()
    reshaped = dataclasses.replace(reference, lang="ben", test_fraction=0.2)
    check = T.check_block_parity(reshaped, TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml")
    assert not check["ok"]
    assert "test_fraction" in check["detail"]


def test_parity_rejects_a_vram_style_batch_reshape():
    """T-500 permits `batch 8 x accum 2` for `batch 16 x accum 1` because VRAM
    forces it for a bigger *model*. Nothing about a different *language* can
    force it, so the same re-shaping is tuning here and must fail — this is
    the one place the two parity rules deliberately disagree."""
    reference, _ = _reference()
    reshaped = dataclasses.replace(
        reference, lang="ben", batch_size=8, grad_accum=2
    )
    assert reshaped.batch_size * reshaped.grad_accum == (
        reference.batch_size * reference.grad_accum
    )
    check = T.check_block_parity(reshaped, TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml")
    assert not check["ok"], (
        "an effective-batch-preserving re-shape is permitted across encoders but "
        "not across blocks; nothing about the source language forces it"
    )


def test_parity_passes_when_only_the_language_differs():
    reference, _ = _reference()
    twin = dataclasses.replace(reference, lang="ben")
    check = T.check_block_parity(twin, TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml")
    assert check["ok"], check["detail"]


def test_block_moves_with_lang_and_is_not_a_difference():
    """`block` is derived from `lang`, so it always differs alongside it. If
    the check counted it, every Phase 4 config would fail for the one field
    that is *supposed* to move."""
    reference, _ = _reference()
    twin = dataclasses.replace(reference, lang="ben")
    assert twin.resolved_block() != reference.resolved_block()
    assert T.check_block_parity(twin, TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml")["ok"]


def test_reference_config_is_not_compared_against_itself():
    """A Hindi config has no block-H sibling; that is not a failure."""
    reference, path = _reference()
    check = T.check_block_parity(reference, path)
    assert check["ok"]
    assert not check["blocker"]
    assert "reference" in check["detail"]


def test_parity_fails_loudly_when_no_reference_exists(tmp_path):
    """Without a Hindi sibling the rule cannot be confirmed, and an
    unconfirmable rule must read as a blocker rather than a pass."""
    reference, _ = _reference()
    orphan = dataclasses.replace(reference, lang="ben", task=2, encoder="xlm-r-base")
    check = T.check_block_parity(orphan, tmp_path / "task2_ben_xlmr.yaml")
    assert not check["ok"]
    assert check["blocker"]
    assert "cannot confirm" in check["detail"]


# --------------------------------------------------------------------------
# The audit as a whole
# --------------------------------------------------------------------------


def test_audit_runs_offline_and_reports_every_check():
    audit = T.run_audit(
        TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml", device="cpu"
    )
    names = [c["name"] for c in audit]
    assert "protocol parity vs block H" in names
    assert "native split loads" in names
    assert "condition matrix has work for this block" in names
    assert all(isinstance(c["ok"], bool) for c in audit)


def test_audit_passes_for_every_shipped_phase4_config():
    """The shipped grid is clean as configuration — T-401's "completes with no
    code change", checked without a GPU."""
    for path in _phase4_configs():
        audit = T.run_audit(path, device="cpu")
        failed = [c for c in audit if not c["ok"] and c["blocker"]]
        assert not failed, f"{path.name}: {[(c['name'], c['detail']) for c in failed]}"


def test_smoke_leg_is_reused_from_t500_not_reimplemented():
    """The end-to-end leg must be the same code Phase 5 proved. A separate
    implementation would test a path the real runs never take."""
    from scripts import t500_smoke

    assert T.run_smoke is t500_smoke.run_smoke


def test_default_config_is_the_gate_cell():
    """CLAUDE6.md orders the grid `B_nat x IndicBERT-v2` first, so the bare
    command smoke-tests that cell rather than an arbitrary one."""
    run = load_run_config(T.DEFAULT_CONFIG)
    assert (run.lang, run.encoder) == ("ben", "indicbert-v2")
    assert Path(T.DEFAULT_CONFIG).exists()
