"""Tests for the T-102 audit.

These run against the real committed spreadsheets rather than fixtures: the
whole point of T-102 is what the actual upstream release contains, and the
files are pinned by SHA-256 in the manifest.
"""

import numpy as np
import pytest

from src.audit import (
    CONTROL_TASK,
    LANGUAGES,
    NATIVE_TASK,
    audit_split,
    class_proportions,
    config_hash,
    control_task_parallelism,
    independence_tests,
    label_vocabularies,
    verify_provenance,
)


@pytest.fixture(scope="module")
def audits():
    return [audit_split(NATIVE_TASK, lang) for lang in LANGUAGES]


def test_provenance_matches_manifest():
    ok, rows = verify_provenance()
    assert ok, [r["local"] for r in rows if not r["ok"]]
    assert len(rows) == 9


def test_config_hash_is_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_native_splits_are_clean(audits):
    for audit in audits:
        assert audit.failures() == []
        assert audit.rows > 2000
        assert audit.empty_text == 0
        assert audit.duplicate_conflicting_labels == 0
        assert audit.non_nfc == 0
        assert audit.replacement_char == 0


def test_no_script_leakage_in_native_splits(audits):
    """Regression guard for the danda false positive."""
    for audit in audits:
        assert audit.leaked_script_rows == 0, (
            f"{audit.language}: {audit.leaked_script_chars}"
        )


def test_label_vocabulary_is_shared(audits):
    vocab = label_vocabularies(audits)
    assert vocab["identical"]
    assert vocab["union"] == ["sustainable", "unsustainable"]
    assert vocab["only_in"] == {}


def test_class_proportions_cover_every_split(audits):
    frame = class_proportions(audits)
    assert set(frame["language"]) == set(LANGUAGES)
    for lang in LANGUAGES:
        assert frame[frame["language"] == lang]["share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_control_task_is_parallel():
    result = control_task_parallelism(CONTROL_TASK)
    assert result["is_parallel"]
    assert len(set(result["rows"].values())) == 1
    assert result["row_order_aligned"] == result["rows"]["hindi"]


def _unit(rows, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(rows, dim))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_independence_flags_identical_corpora():
    """A split compared against itself must come back as parallel."""
    embeddings = {"a": _unit(300), "b": None}
    embeddings["b"] = embeddings["a"].copy()
    _, verdicts = independence_tests(embeddings, seed=1, sample=100, tau=0.82)
    verdict = verdicts["a-b"]
    assert verdict["parallel_suspected"]
    assert verdict["b2_above_tau_share"] == 1.0
    assert verdict["b1_lift_over_control"] > 0.5


def test_independence_passes_unrelated_corpora():
    embeddings = {"a": _unit(300, seed=1), "b": _unit(300, seed=2)}
    frame, verdicts = independence_tests(embeddings, seed=1, sample=100, tau=0.82)
    assert not verdicts["a-b"]["parallel_suspected"]
    assert set(frame["test"]) == {
        "B1_row_aligned", "B1_shuffled_control", "B2_nearest_neighbour"
    }


def test_independence_is_deterministic():
    embeddings = {"a": _unit(300, seed=1), "b": _unit(300, seed=2)}
    first, _ = independence_tests(embeddings, seed=7, sample=100, tau=0.82)
    second, _ = independence_tests(embeddings, seed=7, sample=100, tau=0.82)
    assert first.equals(second)
