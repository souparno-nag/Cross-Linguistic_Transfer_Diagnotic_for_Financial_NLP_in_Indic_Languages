"""Tests for T-112, including §8's tamper test."""

import json

import pandas as pd
import pytest

from src import freeze as F


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A miniature frozen release, so the test never touches data/v1.0."""
    corpus = tmp_path / "raw"
    release = tmp_path / "v1.0"
    monkeypatch.setattr(F, "CORPUS_ROOT", corpus)
    monkeypatch.setattr(F, "FROZEN_ROOT", release)

    frame = pd.DataFrame(
        {"item_id": ["a"], "text": ["वाक्य"], "origin": ["native"], "flags": [[]]}
    )
    for block, lang in (("H", "hin"), ("H", "ben")):
        path = corpus / "task_9" / block / f"{lang}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(F, "read_split", lambda *a, **k: frame)
    monkeypatch.setattr(F, "expected_splits", lambda task: [("H", "hin"), ("H", "ben")])
    return F.freeze_task(9, {"decoding": "abc"})


def test_freeze_writes_a_manifest_that_verifies(frozen):
    assert frozen["splits"] == 2
    assert all(len(e["sha256"]) == 64 for e in frozen["files"].values())
    assert F.verify_task(9)["verified"]


def test_a_tampered_file_fails_verification(frozen):
    """§8's acceptance criterion: modify one file, re-verify, expect failure."""
    path = F.frozen_dir(9) / "H" / "hin.parquet"
    F.set_writable(F.frozen_dir(9), True)
    path.write_bytes(path.read_bytes() + b"\x00")

    result = F.verify_task(9)
    assert not result["verified"]
    assert result["changed"] == ["H/hin.parquet"]


def test_a_deleted_file_is_reported_not_ignored(frozen):
    F.set_writable(F.frozen_dir(9), True)
    (F.frozen_dir(9) / "H" / "ben.parquet").unlink()
    result = F.verify_task(9)
    assert result["missing"] == ["H/ben.parquet"] and not result["verified"]


def test_an_extra_file_is_reported(frozen):
    """A file nobody recorded is as much a problem as a changed one."""
    F.set_writable(F.frozen_dir(9), True)
    (F.frozen_dir(9) / "H" / "tel.parquet").write_bytes(b"x")
    assert F.verify_task(9)["unexpected"] == ["H/tel.parquet"]


def test_a_frozen_release_is_read_only(frozen):
    path = F.frozen_dir(9) / "H" / "hin.parquet"
    assert not path.stat().st_mode & 0o222, "rule 5: frozen files are immutable"


def test_refreezing_is_refused(frozen):
    """Rule 5: a new version gets a new directory, never an overwrite."""
    with pytest.raises(FileExistsError, match="already frozen"):
        F.freeze_task(9, {})


def test_missing_splits_are_named_in_the_manifest(tmp_path, monkeypatch):
    corpus = tmp_path / "raw"
    monkeypatch.setattr(F, "CORPUS_ROOT", corpus)
    monkeypatch.setattr(F, "FROZEN_ROOT", tmp_path / "v1.0")
    monkeypatch.setattr(F, "expected_splits", lambda task: [("H", "mal")])
    manifest = F.freeze_task(8, {})
    assert manifest["missing"] == ["H/mal"] and manifest["splits"] == 0
