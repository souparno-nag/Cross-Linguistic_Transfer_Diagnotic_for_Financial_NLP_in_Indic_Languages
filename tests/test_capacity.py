"""Tests for T-505's capacity-dilution analysis (CLAUDE5.md).

The task numbers are read back from T-504's and T-206's artefacts and tested
there. What is tested here is what this module adds: the parameter split that
separates an embedding table from a transformer body, the tokenisation
measurement, and the in-language-versus-cross-lingual comparison the written
argument turns on.
"""

import pandas as pd
import pytest

from src import capacity as C


# --------------------------------------------------------------------------
# Where the parameters sit
# --------------------------------------------------------------------------


def test_parameter_profile_splits_embeddings_from_the_body():
    """The totals hide the thing the analysis needs. Built on `meta`, so this
    loads no weights and needs no GPU."""
    frame = C.parameter_profile(("indicbert-v2", "mbert-base")).set_index("encoder")
    for encoder in ("indicbert-v2", "mbert-base"):
        row = frame.loc[encoder]
        assert row["total_params"] == row["embedding_params"] + row["body_params"]
        assert row["body_params"] > 0


def test_albert_body_is_far_smaller_than_its_total_suggests():
    """IndicBERT-v2 shares one layer across twelve, so reading its 5x smaller
    total as a 5x smaller model understates the confound by half."""
    frame = C.parameter_profile(("indicbert-v2", "mbert-base")).set_index("encoder")
    total_ratio = (
        frame.loc["mbert-base", "total_params"]
        / frame.loc["indicbert-v2", "total_params"]
    )
    body_ratio = (
        frame.loc["mbert-base", "body_params"]
        / frame.loc["indicbert-v2", "body_params"]
    )
    assert body_ratio > 2 * total_ratio
    assert frame.loc["indicbert-v2", "shares_layer_params"]
    assert not frame.loc["mbert-base", "shares_layer_params"]


def test_mbert_and_xlmr_bodies_are_identical():
    """The controlled comparison XLM-R would have provided: same capacity,
    different breadth. The analysis rests on this, so it is pinned."""
    frame = C.parameter_profile(("mbert-base", "xlm-r-base")).set_index("encoder")
    assert frame.loc["mbert-base", "body_params"] == frame.loc["xlm-r-base", "body_params"]
    assert frame.loc["mbert-base", "vocab_size"] != frame.loc["xlm-r-base", "vocab_size"]


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def test_fragmentation_ratio_divides_the_named_encoders():
    frame = pd.DataFrame(
        {
            "lang": ["hin", "mal"],
            "a__tokens_per_sentence": [50.0, 60.0],
            "b__tokens_per_sentence": [100.0, 90.0],
        }
    )
    ratio = C.fragmentation_ratio(frame, "b", "a")
    assert list(ratio) == pytest.approx([2.0, 1.5])


# --------------------------------------------------------------------------
# In-language versus cross-lingual
# --------------------------------------------------------------------------


def _baselines_frame(tmp_path):
    frame = pd.DataFrame(
        [
            dict(config="task2_hin_indicbert.yaml", encoder="indicbert-v2", seed=s,
                 macro_f1=0.80 + 0.01 * s, accuracy=0.8)
            for s in (0, 1, 2)
        ]
        + [
            dict(config="task2_hin_mbert.yaml", encoder="mbert-base", seed=s,
                 macro_f1=0.90 + 0.01 * s, accuracy=0.9)
            for s in (0, 1, 2)
        ]
    )
    path = tmp_path / "baselines.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_in_language_deficit_is_other_minus_baseline(tmp_path):
    frame = C.in_language_deficit(
        "indicbert-v2", "mbert-base", path=_baselines_frame(tmp_path)
    )
    row = frame[frame["task"] == 2].iloc[0]
    assert row["delta"] == pytest.approx(0.10)
    assert row["indicbert-v2__mean"] == pytest.approx(0.81)


def _comparison_frame(tmp_path, *, base_target, other_target):
    rows = []
    for encoder, target in (("indicbert-v2", base_target), ("mbert-base", other_target)):
        for i, condition in enumerate(["c1", "c2"]):
            rows.append(
                dict(task=2, condition=condition, kind="transfer", quadrant="q",
                     encoder=encoder, status="ok", n_seeds=3, target_mean=target,
                     gap_mean=0.4, gap_std=0.01)
            )
    path = tmp_path / "encoder_comparison.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_cross_lingual_deficit_averages_over_shared_cells(tmp_path):
    path = _comparison_frame(tmp_path, base_target=0.50, other_target=0.80)
    frame = C.cross_lingual_deficit("indicbert-v2", "mbert-base", path=path)
    row = frame.iloc[0]
    assert row["n_cells"] == 2
    assert row["delta"] == pytest.approx(0.30)


def test_cross_lingual_deficit_ignores_blocked_cells(tmp_path):
    rows = [
        dict(task=2, condition="c1", kind="transfer", quadrant="q", encoder=e,
             status="ok", n_seeds=3, target_mean=t, gap_mean=0.4, gap_std=0.01)
        for e, t in (("indicbert-v2", 0.5), ("mbert-base", 0.8))
    ] + [
        dict(task=2, condition="c2", kind="transfer", quadrant="q", encoder=e,
             status="blocked", n_seeds=0, target_mean=None, gap_mean=None, gap_std=None)
        for e in ("indicbert-v2", "mbert-base")
    ]
    path = tmp_path / "c.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    frame = C.cross_lingual_deficit("indicbert-v2", "mbert-base", path=path)
    assert frame.iloc[0]["n_cells"] == 1
    assert frame.iloc[0]["delta"] == pytest.approx(0.30)


def test_cross_lingual_deficit_reports_nothing_when_an_encoder_is_absent(tmp_path):
    rows = [
        dict(task=2, condition="c1", kind="transfer", quadrant="q",
             encoder="indicbert-v2", status="ok", n_seeds=3, target_mean=0.5,
             gap_mean=0.4, gap_std=0.01)
    ]
    path = tmp_path / "c.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    frame = C.cross_lingual_deficit("indicbert-v2", "mbert-base", path=path)
    assert frame.iloc[0]["delta"] is None


# --------------------------------------------------------------------------
# Rendering keeps the caveats attached
# --------------------------------------------------------------------------


def test_parameter_table_flags_shared_layers():
    md = "\n".join(C.render_parameters(C.parameter_profile(("indicbert-v2",))))
    assert "shares layer params" in md
    assert "| yes |" in md


def test_deficit_table_renders_the_amplification():
    frame = pd.DataFrame(
        [{"task": 2, "in_language_delta": 0.05, "cross_lingual_delta": 0.25,
          "n_cells": 6, "amplification": 5.0}]
    )
    md = "\n".join(C.render_deficit(frame, baseline="a", other="b"))
    assert "5.0×" in md
    assert "+0.0500" in md and "+0.2500" in md


def test_deficit_table_survives_a_missing_amplification():
    """A zero in-language delta would divide by zero; it must render as
    missing rather than crash or print a spurious number."""
    frame = pd.DataFrame(
        [{"task": 3, "in_language_delta": 0.0, "cross_lingual_delta": 0.25,
          "n_cells": 6, "amplification": None}]
    )
    md = "\n".join(C.render_deficit(frame, baseline="a", other="b"))
    assert "—" in md
