"""Tests for the results dashboard (`dashboard/`).

The dashboard only reads artefacts, so what can go wrong is a quiet misjoin:
a sentence shown beside the wrong prediction, a cause with no explanation, or
a page that breaks because a report it reads was renamed. These run on the
real artefacts and skip when they are not built.
"""

import pandas as pd
import pytest

from dashboard import data as D
from src import findings as F

needs_artefacts = pytest.mark.skipif(
    not (D.SOURCE_COMPARISON.exists() and D.LABEL_SHARES.exists()), reason="reports not built"
)


def test_every_label_and_group_has_an_explanation_and_a_place():
    assert set(D.LABEL_TEXT) == set(F.LABELS) == set(D.LABEL_ORDER)
    assert set(D.GROUP_TEXT) == set(F.GROUPS) == set(D.GROUP_ORDER)
    assert D.LABEL_ORDER[0] == D.GROUP_ORDER[0] == "unattributed"


def test_the_two_weak_labels_carry_their_limitation():
    assert D.label_caveat("morphological_masking") and D.label_caveat("terminology_gap")
    assert D.label_caveat("class_collapse") is None


@needs_artefacts
@pytest.mark.parametrize("task", F.TASKS)
def test_transfer_and_cause_tables_cover_every_setup(task):
    gaps = D.transfer_gaps(task)
    assert len(gaps) == 2 * 9 and set(gaps["model"]) == set(D.ENCODER_NAMES.values())
    shares = D.cause_shares(task)
    l1 = shares[shares["level"] == "L1"].groupby(["encoder", "condition_id"])["share"].sum()
    assert len(l1) == 2 * 18
    assert l1.round(9).eq(1.0).all()


@needs_artefacts
@pytest.mark.parametrize("task", F.TASKS)
def test_every_failure_lines_up_with_its_predictions_and_texts(task):
    """The explorer joins three files per setup. A failure must show a correct
    training-language prediction, a wrong test-language one, and both texts."""
    versions = D.corpus_versions(task)
    conditions = D.conditions(task)
    assert len(conditions) == 18
    for condition in conditions:
        fails = D.failures(task, condition)
        preds = D.predictions(task, condition)
        joined = fails.merge(preds, on=["encoder", "item_id", "seed"], how="left", validate="one_to_one")
        assert joined["gold"].notna().all(), condition["name"]
        assert (joined["pred_src"] == joined["gold"]).all(), condition["name"]
        assert (joined["pred_tgt"] != joined["gold"]).all(), condition["name"]
        assert joined["gate_status"].notna().all(), condition["name"]

        items = D.failed_items(fails, versions, condition)
        assert items["text"].notna().all(), condition["name"]
        train = versions[(versions["block_id"] == condition["train_block"]) & (versions["lang"] == condition["source"])]
        assert set(items["item_id"]) <= set(train["item_id"]), condition["name"]


@needs_artefacts
def test_malayalam_is_never_described_as_written_by_people():
    for condition in D.conditions(2):
        text = D.describe_test_text(condition)
        if condition["target"] == "mal":
            assert "machine-translated" in text
        elif condition["kind"] == "transfer":
            assert "written by people" in text


@needs_artefacts
@pytest.mark.parametrize("task", F.TASKS)
def test_every_page_renders(task):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../dashboard/app.py", default_timeout=120)
    at.session_state["task"] = task
    at.run()
    assert not at.exception
    for page in ("pages/causes.py", "pages/explorer.py"):
        at.switch_page(page).run()
        assert not at.exception, page
    at.session_state["items"] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    at.run()
    assert not at.exception
    assert any(isinstance(df.value, pd.DataFrame) and "Cause" in df.value.columns for df in at.dataframe)
