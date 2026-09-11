"""Tests for T-113.

The matrix exists to stop a model being evaluated on sentences it trained on,
in translation. These check that the validator actually notices when that
happens, rather than passing because the split names differ.
"""

import pytest

from src import conditions as C


def test_the_partition_is_a_property_of_the_item_not_its_position():
    """Every language must agree on which half an item is in, without talking."""
    assert C.partition_of("t2_abc123") == C.partition_of("t2_abc123")
    assert C.partition_of("t2_abc123", seed=1) in {"train", "eval"}


def test_the_partition_splits_roughly_in_half():
    ids = [f"t2_{i:06d}" for i in range(2000)]
    train = sum(1 for i in ids if C.partition_of(i) == "train")
    assert 0.45 < train / len(ids) < 0.55, f"{train}/2000 in the training half"


def test_a_different_seed_gives_a_different_partition():
    ids = [f"t2_{i:06d}" for i in range(500)]
    a = [C.partition_of(i, seed=1) for i in ids]
    b = [C.partition_of(i, seed=2) for i in ids]
    assert a != b


def test_all_nine_transfer_cells_and_four_quadrants_are_covered():
    matrix = C.build(2)
    assert C.validate(matrix) == []
    transfer = [c for c in matrix["conditions"] if c["kind"] == "transfer"]
    assert len(transfer) == 9
    assert len({c["quadrant"] for c in transfer}) == 4


def test_translationese_conditions_exist_for_all_three_native_languages():
    matrix = C.build(2)
    names = {c["name"] for c in matrix["conditions"] if c["kind"] == "translationese"}
    assert names == {"translationese_hin", "translationese_ben", "translationese_tel"}
    for condition in matrix["conditions"]:
        if condition["kind"] == "translationese":
            assert len(condition["eval"]) == 3, "native plus two translated versions"


def test_same_source_mt_conditions_cover_all_nine_cells_within_their_own_block():
    """The H/B/T-sourced MT arm: train and eval share a block, unlike `transfer`."""
    matrix = C.build(2)
    assert C.validate(matrix) == []
    transfer_mt = [c for c in matrix["conditions"] if c["kind"] == "transfer_mt"]
    assert len(transfer_mt) == 9
    for condition in transfer_mt:
        train_block = condition["train"].split("/")[1]
        eval_block = condition["eval"].split("/")[1]
        assert train_block == eval_block, condition["name"]
        assert condition["eval_provenance"] == condition["train"].split("/")[2]


def test_same_source_mt_conditions_pair_up_with_the_native_transfer_cells():
    """For every native transfer cell there is a matching same-source-MT one."""
    matrix = C.build(2)
    transfer = {c["name"]: c for c in matrix["conditions"] if c["kind"] == "transfer"}
    transfer_mt = {
        c["name"]: c for c in matrix["conditions"] if c["kind"] == "transfer_mt"
    }
    assert {f"{name}_mt" for name in transfer} == set(transfer_mt)
    for name, native_cell in transfer.items():
        mt_cell = transfer_mt[f"{name}_mt"]
        assert mt_cell["train"] == native_cell["train"]
        assert mt_cell["quadrant"] == native_cell["quadrant"]


def test_same_source_mt_still_keeps_train_and_eval_items_disjoint():
    """Same-block eval must not leak into the training half either."""
    matrix = C.build(2)
    condition = next(c for c in matrix["conditions"] if c["kind"] == "transfer_mt")
    ids = {f"t2_{i:06d}" for i in range(500)}
    failures = C.validate(
        {**matrix, "conditions": [condition]},
        {condition["train"]: ids, condition["eval"]: ids},
    )
    assert not [f for f in failures if "items" in f], failures


def test_a_split_used_for_both_training_and_evaluation_is_caught():
    matrix = C.build(2)
    matrix["conditions"][0]["eval"] = matrix["conditions"][0]["train"]
    failures = C.validate(matrix)
    assert any("both training source and evaluation target" in f for f in failures)


def test_item_overlap_is_caught_even_when_the_splits_differ():
    """The check that matters for tasks 2 and 3.

    Their splits hold the same items in every language, so a name-only check
    passes while the model is evaluated on its training sentences translated.
    """
    matrix = C.build(2)
    condition = matrix["conditions"][0]
    shared = {"t2_shared"}
    # Both halves claim the same item: the partition is being ignored.
    matrix["conditions"] = [dict(condition, train_items="all", eval_items="all")]
    failures = C.validate(matrix, {condition["train"]: shared, condition["eval"]: shared})
    assert any("shares 1 items" in f for f in failures)


def test_task_1_needs_no_partition_because_its_blocks_differ():
    matrix = C.build(1)
    assert matrix["partitioned"] is False
    assert all(
        c["train_items"] == "all"
        for c in matrix["conditions"]
        if c["kind"] == "transfer"
    )


def test_a_partitioned_condition_keeps_the_halves_apart():
    """The same item set in both splits is fine once the partition applies."""
    matrix = C.build(2)
    condition = next(c for c in matrix["conditions"] if c["kind"] == "transfer")
    ids = {f"t2_{i:06d}" for i in range(500)}
    failures = C.validate(
        {**matrix, "conditions": [condition]},
        {condition["train"]: ids, condition["eval"]: ids},
    )
    # The whole-matrix checks still fire on a one-condition matrix; what this
    # asserts is that no *item* leaked between the halves.
    assert not [f for f in failures if "items" in f], failures


def test_evaluation_prefers_native_text_over_machine_translation():
    """Evaluating on MT confounds transfer failure with translation failure."""
    matrix = C.build(2)
    for condition in matrix["conditions"]:
        if condition["kind"] != "transfer":
            continue
        target = condition["name"].split("_to_")[1]
        if target == "mal":
            assert condition["eval_provenance"] != "native", "no native Malayalam (§2)"
            assert not condition["eval"].startswith(condition["train"].rsplit("/", 2)[0] + "/" + condition["train"].split("/")[1] + "/"), \
                "Malayalam evaluation must not come from the block just trained on"
        else:
            assert condition["eval_provenance"] == "native", target


def test_task_1_evaluates_on_a_different_block_so_the_sentences_differ():
    """Task 1's block H Bengali is block H Hindi translated — the same items.

    Training on it and evaluating on it would be testing the model on its own
    training sentences in another script.
    """
    matrix = C.build(1)
    for condition in matrix["conditions"]:
        if condition["kind"] == "transfer":
            assert condition["train"] != condition["eval"]
            train_block = condition["train"].split("/")[1]
            eval_block = condition["eval"].split("/")[1]
            assert train_block != eval_block, condition["name"]
