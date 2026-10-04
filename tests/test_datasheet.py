"""Tests for T-114.

The datasheet's failure mode is silent staleness: a figure typed in once,
correct that day, quietly wrong after the next regeneration. So these check
both that §8's required content is present and that the numbers actually come
from the artefacts rather than from the text.
"""

import pandas as pd
import pytest

from scripts.t114_datasheet import build
from src.datasheet import TASKS, direction_table, load_artefacts
from src.ids import BLOCK_NATIVE_LANG, targets_for_block


@pytest.fixture(scope="module")
def datasheet():
    return build()


@pytest.mark.parametrize(
    "required",
    [
        "Direction matrix",
        "ID scheme",
        "Machine translation",
        "Storage format",
        "Known limitations",
        "CC BY-NC-SA 4.0",
    ],
)
def test_every_required_section_is_present(datasheet, required):
    assert required in datasheet


def test_the_translation_model_and_decoding_are_named(datasheet):
    assert "indictrans2-indic-indic-1B" in datasheet
    assert "beam 5" in datasheet and "max length 256" in datasheet
    assert "Decoding fingerprint" in datasheet


def test_all_nine_directions_appear_for_every_task(datasheet):
    for task in TASKS:
        table = direction_table(load_artefacts(task), task)
        assert len(table) == 9
        for block, native in BLOCK_NATIVE_LANG.items():
            for target in targets_for_block(block):
                row = table[(table["source"] == native) & (table["target"] == target)]
                assert len(row) == 1, f"task {task}: {native}->{target} missing"


def test_the_rates_match_the_artefacts_they_came_from():
    """§8 requires a drift rate and an entity rate per direction — real ones."""
    for task in TASKS:
        artefacts = load_artefacts(task)
        table = direction_table(artefacts, task)
        for _, source in artefacts["ranking"].iterrows():
            row = table[
                (table["source"] == source["src_lang"])
                & (table["target"] == source["tgt_lang"])
            ].iloc[0]
            assert row["below τ"] == f"{round(source['drift_rate'] * 100, 1)}%"
            assert row["entities kept"] == f"{round(source['entity_preserved'] * 100, 1)}%"


def test_the_two_named_limitations_are_stated_plainly(datasheet):
    """§8 names both; neither may be softened into a footnote."""
    assert "No native Malayalam" in datasheet
    assert "Human verification covers a sample, not the corpus" in datasheet
    assert "cannot establish that a sentence means the right thing" in datasheet


def test_the_verification_sample_is_quantified_not_just_claimed():
    """"Verified" without coverage is the claim a reader would most easily misread.

    The hand-check is 180 rows per task against 68,358 / 19,614 / 4,788 MT rows,
    so the datasheet states the percentage and who signed it off rather than
    reporting the corpus as verified.
    """
    text = build()
    assert "0.3% of task 1, 0.9% of task 2 and 3.8% of task 3" in text
    assert "signed off by the project maintainer" in text
    assert "rather than an independent audit" in text


def test_share_alike_obligations_are_spelled_out_not_just_named(datasheet):
    assert "no commercial use" in datasheet
    assert "must be distributed under CC BY-NC-SA 4.0" in datasheet
    assert "Ghosh et al." in datasheet


def test_the_metric_caveat_survives_into_the_datasheet(datasheet):
    """A reader must not take labse_sim for a quality score.

    Machine translations outscore human ones on it, so a datasheet that
    reported the number without the caveat would mislead every downstream user.
    """
    assert "rewards literalness" in datasheet
    assert "score *higher* than human translations" in datasheet


def test_row_and_split_totals_agree_with_the_manifests(datasheet):
    total = sum(load_artefacts(task)["manifest"]["rows"] for task in TASKS)
    assert f"{total:,} rows" in datasheet


def test_tau_is_reported_per_task_with_its_basis(datasheet):
    assert "inherited, not calibrated" in datasheet, "task 1 has no human anchor"
    for task in TASKS:
        assert f"| `task_{task}` |" in datasheet


def test_regenerating_is_deterministic_apart_from_the_date():
    first, second = build(), build()
    assert first == second


def test_the_bengali_digit_system_difference_is_stated_not_hidden(datasheet):
    """Native Bengali uses Bengali digits, MT output is ASCII; the datasheet must say
    so, and must not turn it into a claimed cause of failure (reports/digit_audit.md)."""
    assert "Digit systems differ between native and machine-translated Bengali" in datasheet
    assert "not a measured cause of error" in datasheet
    assert "reports/digit_audit.md" in datasheet
