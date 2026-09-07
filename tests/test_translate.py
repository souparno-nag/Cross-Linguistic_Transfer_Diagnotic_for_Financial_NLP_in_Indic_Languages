"""Tests for the T-104 pipeline.

The model itself is not loaded — these cover the parts that decide whether a
long run survives interruption, which is where the bugs that cost hours live.
"""

import pandas as pd
import pytest

from src import translate as T


class FakeTranslator:
    """Stands in for a loaded model, and counts what it was asked to do."""

    def __init__(self, config, fail_on=None):
        self.config = config
        self.calls = []
        self.fail_on = fail_on

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        if self.fail_on is not None and self.fail_on in texts:
            raise RuntimeError("simulated CUDA failure")
        return [f"[{tgt_lang}] {t}" for t in texts]


@pytest.fixture
def config():
    return T.load_config()


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"item_id": [f"i{i}" for i in range(10)], "text": [f"वाक्य {i}" for i in range(10)]}
    )


@pytest.fixture(autouse=True)
def isolated_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CHECKPOINT_ROOT", tmp_path / "translate")


def test_config_is_frozen_and_deterministic(config):
    assert config["decoding"]["do_sample"] is False, "sampling would break rule 7"
    assert config["decoding"]["num_beams"] >= 1
    assert T.decoding_fingerprint(config) == T.decoding_fingerprint(config)


def test_fingerprint_ignores_batch_size_but_tracks_decoding(config):
    changed = {**config, "batch_size": config["batch_size"] + 1}
    assert T.decoding_fingerprint(changed) == T.decoding_fingerprint(config)
    changed = {**config, "decoding": {**config["decoding"], "num_beams": 99}}
    assert T.decoding_fingerprint(changed) != T.decoding_fingerprint(config)


def test_flores_codes(config):
    assert T.flores_code("mal", config) == "mal_Mlym"
    with pytest.raises(ValueError, match="no FLORES code"):
        T.flores_code("eng", config)


def test_translates_everything_in_input_order(config, frame):
    fake = FakeTranslator(config)
    out = T.translate_rows(fake, frame, "hin", "ben", 2, "H", batch_size=4, progress=lambda *_: None)
    assert list(out["item_id"]) == list(frame["item_id"])
    assert out["translation"].iloc[0] == "[ben] वाक्य 0"
    assert len(out) == len(frame)


def test_resumes_without_redoing_work(config, frame):
    """An interrupted run must not re-translate what it already finished."""
    first = FakeTranslator(config)
    T.translate_rows(first, frame.head(4), "hin", "ben", 2, "H", batch_size=4, progress=lambda *_: None)
    assert sum(len(c) for c in first.calls) == 4

    second = FakeTranslator(config)
    out = T.translate_rows(second, frame, "hin", "ben", 2, "H", batch_size=4, progress=lambda *_: None)
    # Only the 6 unseen rows go to the model; the first 4 come from checkpoint.
    assert sum(len(c) for c in second.calls) == 6
    assert len(out) == 10
    assert out["translation"].notna().all()


def test_checkpoint_survives_a_failed_batch(config, frame):
    """A failure flags the rows and keeps going; nothing is dropped (rule 1)."""
    fake = FakeTranslator(config, fail_on="वाक्य 4")
    out = T.translate_rows(fake, frame, "hin", "ben", 2, "H", batch_size=4, progress=lambda *_: None)
    assert len(out) == 10
    failed = out[out["translation"] == ""]
    assert len(failed) == 4
    assert all("empty_output" in f for f in failed["flags"])


def test_empty_output_is_flagged_not_dropped(config, frame):
    class Blank(FakeTranslator):
        def translate(self, texts, src_lang, tgt_lang):
            return ["" for _ in texts]

    out = T.translate_rows(Blank(config), frame, "hin", "ben", 2, "H", batch_size=5, progress=lambda *_: None)
    assert len(out) == len(frame)
    assert all("empty_output" in f for f in out["flags"])


def test_checkpoint_path_separates_tasks_and_directions():
    a = T.checkpoint_path(2, "H", "hin", "ben")
    assert a != T.checkpoint_path(3, "H", "hin", "ben")
    assert a != T.checkpoint_path(2, "H", "hin", "tel")


class OOMOnce(FakeTranslator):
    """Fails on any batch larger than `limit`, like a GPU running out."""

    def __init__(self, config, limit):
        super().__init__(config)
        self.limit = limit
        self.freed = 0

    def free(self):
        self.freed += 1

    def translate(self, texts, src_lang, tgt_lang):
        import torch

        self.calls.append(list(texts))
        if len(texts) > self.limit:
            raise torch.OutOfMemoryError("simulated")
        return [f"[{tgt_lang}] {t}" for t in texts]


def test_adaptive_batching_backs_off_and_completes(config):
    """One oversized batch must not lose the whole direction."""
    texts = [f"वाक्य {i}" for i in range(10)]
    fake = OOMOnce(config, limit=2)
    out = T.translate_adaptive(fake, texts, "hin", "ben", batch_size=8)
    assert out == [f"[ben] {t}" for t in texts]
    assert fake.freed > 0, "should release cached blocks before retrying"
    assert max(len(c) for c in fake.calls) == 8, "should try the full batch first"
    assert all(len(c) <= 2 for c in fake.calls if len(c) <= 2)


def test_adaptive_batching_reraises_when_a_single_row_cannot_fit(config):
    """A row too big even alone is a real error, not something to swallow."""
    import torch

    fake = OOMOnce(config, limit=0)
    with pytest.raises(torch.OutOfMemoryError):
        T.translate_adaptive(fake, ["वाक्य"], "hin", "ben", batch_size=4)


def test_adaptive_batching_preserves_order(config):
    texts = [f"s{i}" for i in range(17)]
    out = T.translate_adaptive(OOMOnce(config, limit=3), texts, "hin", "ben", batch_size=16)
    assert out == [f"[ben] {t}" for t in texts]
