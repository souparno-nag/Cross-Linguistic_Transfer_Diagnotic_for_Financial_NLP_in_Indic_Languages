"""Tests for T-108.

The model is loaded once, for the §10 fixture that a known-similar pair and a
known-dissimilar pair land on opposite sides of τ. Everything else is
arithmetic and bookkeeping and runs without it.
"""

import numpy as np
import pandas as pd
import pytest

from src import labse_gate as L


@pytest.fixture(scope="module")
def config():
    return L.load_config()


def test_cosine_of_normalised_vectors_is_a_dot_product():
    a = np.array([[1.0, 0.0], [0.6, 0.8]])
    b = np.array([[1.0, 0.0], [0.8, 0.6]])
    assert L.cosine(a, b) == pytest.approx([1.0, 0.96])


def test_fingerprint_tracks_what_changes_a_score_and_ignores_batch_size(config):
    assert L.verification_fingerprint({**config, "batch_size": 999}) == L.verification_fingerprint(config)
    assert L.verification_fingerprint({**config, "max_seq_length": 64}) != L.verification_fingerprint(config)


def test_tau_is_the_documented_threshold(config):
    assert config["tau"] == 0.82, "τ is inherited from the brief; T-110 may revise it"
    assert config["normalize_embeddings"] is True


def test_embeddings_are_cached_per_block_and_language():
    assert L.embedding_path(1, "H", "ben") != L.embedding_path(1, "B", "ben")
    assert L.embedding_path(1, "H", "ben") != L.embedding_path(2, "H", "ben")


def test_a_stale_cache_is_not_reused(tmp_path, monkeypatch):
    """A regenerated split must not inherit vectors for a different row count.

    Reusing them would attach every score to the wrong sentence while looking
    perfectly healthy.
    """
    frame = pd.DataFrame({"item_id": ["a", "b", "c"], "text": ["x", "y", "z"]})
    monkeypatch.setattr(L, "read_split", lambda *a, **k: frame)
    path = tmp_path / "H_ben.npy"
    np.save(path, np.zeros((2, 4)))  # two vectors, three rows
    monkeypatch.setattr(L, "embedding_path", lambda *a, **k: path)

    with pytest.raises(RuntimeError, match="no embeddings cached"):
        L.embed_split(None, 1, "H", "ben", progress=lambda *_: None)


def test_cached_embeddings_are_reused_when_they_match(tmp_path, monkeypatch):
    frame = pd.DataFrame({"item_id": ["a", "b"], "text": ["x", "y"]})
    monkeypatch.setattr(L, "read_split", lambda *a, **k: frame)
    path = tmp_path / "H_ben.npy"
    np.save(path, np.ones((2, 4)))
    monkeypatch.setattr(L, "embedding_path", lambda *a, **k: path)
    assert L.embed_split(None, 1, "H", "ben", progress=lambda *_: None).shape == (2, 4)


@pytest.mark.slow
def test_known_similar_and_dissimilar_pairs_fall_the_right_side_of_tau(config):
    """§10's requirement, on the real model."""
    embedder = L.Embedder(config, device="cpu")
    pairs = [
        ("कंपनी ने 500 करोड़ रुपये जुटाए।", "সংস্থাটি 500 কোটি টাকা সংগ্রহ করেছে।", True),
        ("कंपनी ने 500 करोड़ रुपये जुटाए।", "আজ আবহাওয়া খুব ভালো।", False),
    ]
    vectors = embedder.encode([t for pair in pairs for t in pair[:2]])
    for i, (_, _, similar) in enumerate(pairs):
        score = float(L.cosine(vectors[2 * i : 2 * i + 1], vectors[2 * i + 1 : 2 * i + 2])[0])
        if similar:
            assert score >= config["tau"], f"translation pair scored {score:.3f}"
        else:
            assert score < config["tau"], f"unrelated pair scored {score:.3f}"
