"""Tests for T-500's model-agnosticism smoke test (CLAUDE5.md).

The audit leg is pure and needs no model, so all of it is testable fast. The
most valuable test here is not of the script but of the *configs*: CLAUDE5.md
hard rule 1 says a cross-encoder comparison is only valid if nothing but the
encoder moved, and a YAML pair can drift apart silently months from now. The
parity test turns that rule into something CI enforces rather than something
a reader has to notice.
"""

from dataclasses import fields

import pytest

from scripts import t500_smoke as T
from src.config import TRAIN_CONFIG_DIR, RunConfig, load_run_config

PHASE5_ENCODERS = ("xlm-r-base", "mbert-base")


def _all_configs():
    """Every shipped config, deferred ones included.

    `configs/train/deferred/` holds configs T-206 must not pick up — XLM-R,
    excluded on VRAM grounds at T-500 — but they stay under test so that if the
    project ever moves to a larger card they can be run without first
    re-deriving whether they drifted from the protocol.
    """
    return sorted(TRAIN_CONFIG_DIR.glob("*.yaml")) + sorted(
        (TRAIN_CONFIG_DIR / "deferred").glob("*.yaml")
    )


def _phase5_configs():
    return [p for p in _all_configs() if load_run_config(p).encoder in PHASE5_ENCODERS]


# --------------------------------------------------------------------------
# The configs themselves — hard rules 1 and 2
# --------------------------------------------------------------------------


def test_at_least_one_phase5_config_is_shipped():
    assert _phase5_configs(), (
        "no config names a Phase 5 encoder — T-500 needs one added as a config entry"
    )


def test_deferred_configs_are_outside_the_t206_glob():
    """`t206_baseline` globs `configs/train/*.yaml` non-recursively, and that
    is the only thing keeping it from trying to train XLM-R and OOMing. If a
    deferred config is ever moved back up a level, this fails."""
    trainable = {p.stem for p in TRAIN_CONFIG_DIR.glob("*.yaml")}
    deferred = {p.stem for p in (TRAIN_CONFIG_DIR / "deferred").glob("*.yaml")}
    assert deferred, "expected XLM-R to be shipped as a deferred config"
    assert not (trainable & deferred)
    for path in (TRAIN_CONFIG_DIR / "deferred").glob("*.yaml"):
        assert load_run_config(path).encoder == "xlm-r-base"


@pytest.mark.parametrize("path", _phase5_configs(), ids=lambda p: p.stem)
def test_phase5_config_matches_its_indicbert_sibling(path):
    """Identical protocol across encoders: only `encoder` may differ."""
    run = load_run_config(path)
    check = T.check_protocol_parity(run, path)
    assert check["ok"], check["detail"]


@pytest.mark.parametrize("path", _phase5_configs(), ids=lambda p: p.stem)
def test_phase5_config_resolves_to_a_real_encoder(path):
    run = load_run_config(path)
    assert T.check_registry(run)["ok"]


def test_phase5_config_stems_are_unique_per_encoder():
    """T-208 maps a run_id back to `configs/train/<stem>.yaml`, and T-206
    builds run_id from the stem, so two encoders sharing a stem would share a
    checkpoint directory."""
    stems = [p.stem for p in _all_configs()]
    assert len(stems) == len(set(stems))


# --------------------------------------------------------------------------
# The audit's own logic
# --------------------------------------------------------------------------


def test_parity_check_catches_a_tuned_hyperparameter(tmp_path):
    """A per-encoder tuned field must fail the check, naming the field."""
    reference = load_run_config(TRAIN_CONFIG_DIR / "task2_hin_indicbert.yaml")
    tuned = RunConfig(
        **{
            **{f.name: getattr(reference, f.name) for f in fields(RunConfig)},
            "encoder": "xlm-r-base",
            "lr": reference.lr * 2,
        }
    )
    check = T.check_protocol_parity(tuned, TRAIN_CONFIG_DIR / "task2_hin_mbert.yaml")
    assert not check["ok"]
    assert "lr" in check["detail"]


def test_parity_check_passes_when_only_the_encoder_differs():
    reference = load_run_config(TRAIN_CONFIG_DIR / "task2_hin_indicbert.yaml")
    twin = RunConfig(
        **{
            **{f.name: getattr(reference, f.name) for f in fields(RunConfig)},
            "encoder": "mbert-base",
        }
    )
    check = T.check_protocol_parity(twin, TRAIN_CONFIG_DIR / "task2_hin_mbert.yaml")
    assert check["ok"], check["detail"]


def test_checkpoint_naming_check_passes_for_indicbert():
    """The reference encoder must pass — otherwise the check is just broken
    rather than detecting anything."""
    path = TRAIN_CONFIG_DIR / "task2_hin_indicbert.yaml"
    check = T.check_checkpoint_naming(load_run_config(path), path)
    assert check["ok"], check["detail"]


def test_audit_returns_one_entry_per_check_with_the_expected_shape():
    audit = T.run_audit(TRAIN_CONFIG_DIR / "task2_hin_mbert.yaml", device="cpu")
    # registry, parity, VRAM floor, checkpoint naming, three artefact checks
    assert len(audit) == 7
    for check in audit:
        assert set(check) == {"name", "ok", "blocker", "detail"}
        assert isinstance(check["ok"], bool)
        assert check["detail"].strip()


def test_audit_judges_the_shipped_yaml_not_the_smoke_overrides():
    """Leg B loads the same YAML with `epochs=1, patience=1`; auditing that
    object would report the smoke's own shortcut as a hard-rule-2 violation.
    Caught for real the first time this script ran."""
    path = TRAIN_CONFIG_DIR / "task2_hin_mbert.yaml"
    parity = next(
        c
        for c in T.run_audit(path, device="cpu")
        if c["name"] == "protocol parity vs IndicBERT"
    )
    assert parity["ok"], parity["detail"]


# --------------------------------------------------------------------------
# The VRAM floor — fixed parameter cost, which a batch-size sweep cannot see
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder,expected_millions",
    [("indicbert-v2", 34), ("xlm-r-base", 279), ("mbert-base", 179)],
)
def test_parameter_budget_matches_the_published_model_size(encoder, expected_millions):
    """Built on the `meta` device, so this needs no weights and no GPU. The
    counts are the published ones; a mismatch means the registry points at a
    different checkpoint than we think."""
    budget = T.parameter_budget(encoder)
    assert budget["n_params"] / 1e6 == pytest.approx(expected_millions, abs=1.5)
    # 16 bytes per parameter: fp32 weights + fp32 grads + two AdamW moments.
    assert budget["floor_gib"] == pytest.approx(budget["n_params"] * 16 / 1024**3)


def test_xlmr_full_finetune_does_not_fit_a_4gb_card():
    """The T-500 finding, pinned so a later change cannot quietly un-find it.

    XLM-R's fixed parameter cost alone exceeds this project's card, which is
    why the OOM is not a batch-size problem and T-503 exists.
    """
    assert T.parameter_budget("xlm-r-base")["floor_gib"] > 4.0
    assert T.parameter_budget("indicbert-v2")["floor_gib"] < 1.0


def test_vram_check_is_inconclusive_rather_than_falsely_green_without_cuda():
    run = load_run_config(TRAIN_CONFIG_DIR / "deferred" / "task2_hin_xlmr.yaml")
    check = T.check_optimizer_fits(run, "cpu")
    assert check["ok"] and not check["blocker"]
    assert "not compared" in check["detail"]


def test_subset_is_stratified_and_reproducible():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(100)],
            "label_id": [0] * 80 + [1] * 20,
            "text": ["x"] * 100,
        }
    )
    a = T._stratified_subset(frame, rows=20, seed=0)
    b = T._stratified_subset(frame, rows=20, seed=0)
    assert len(a) == 20
    assert list(a["item_id"]) == list(b["item_id"])
    # 80/20 in, 80/20 out — a subset that lost a class would make macro-F1
    # meaningless for a reason unrelated to the encoder.
    assert sorted(a["label_id"].value_counts().tolist(), reverse=True) == [16, 4]


def test_subset_returns_everything_when_rows_exceeds_the_split():
    import pandas as pd

    frame = pd.DataFrame(
        {"item_id": ["a", "b"], "label_id": [0, 1], "text": ["x", "y"]}
    )
    assert len(T._stratified_subset(frame, rows=200, seed=0)) == 2
