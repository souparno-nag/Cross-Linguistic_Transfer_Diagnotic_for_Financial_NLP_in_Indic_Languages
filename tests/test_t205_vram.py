"""Tests for T-205's report assembly.

The probing itself is covered by `tests/test_vram.py`; what is tested here is
the report merge, which exists because a narrowed run (`--encoders`,
`--config`) used to overwrite the whole table with its own handful of rows.
That silently deleted measurements nobody had asked to re-measure — the same
failure mode CLAUDE.md records for T-106's per-direction report.
"""

from scripts import t205_vram as T


def test_new_rows_replace_same_key_rows():
    existing = [
        {"encoder": "mbert-base", "max_len": 192, "max_batch": 16},
        {"encoder": "indicbert-v2", "max_len": 192, "max_batch": 64},
    ]
    new = [{"encoder": "mbert-base", "max_len": 192, "max_batch": 32}]
    merged = T._merge_rows(new, existing, ("encoder", "max_len"))
    assert len(merged) == 2
    by_encoder = {r["encoder"]: r for r in merged}
    assert by_encoder["mbert-base"]["max_batch"] == 32  # replaced
    assert by_encoder["indicbert-v2"]["max_batch"] == 64  # untouched, kept


def test_untouched_rows_survive_a_narrowed_run():
    """The actual bug: measuring one encoder must not delete the other two."""
    existing = [
        {"encoder": e, "max_len": 128, "max_batch": 8}
        for e in ("indicbert-v2", "xlm-r-base", "mbert-base")
    ]
    merged = T._merge_rows(
        [{"encoder": "mbert-base", "max_len": 128, "max_batch": 48}],
        existing,
        ("encoder", "max_len"),
    )
    assert {r["encoder"] for r in merged} == {
        "indicbert-v2",
        "xlm-r-base",
        "mbert-base",
    }


def test_merge_on_a_single_key_works_for_epoch_rows():
    existing = [
        {"config": "task2_hin_mbert.yaml", "result": "OOM: ..."},
        {"config": "task3_hin_mbert.yaml", "result": "ok"},
    ]
    merged = T._merge_rows(
        [{"config": "task2_hin_mbert.yaml", "result": "ok"}], existing, ("config",)
    )
    assert len(merged) == 2
    assert {r["config"]: r["result"] for r in merged}["task2_hin_mbert.yaml"] == "ok"


def test_merge_is_order_stable():
    """Two merges of the same inputs produce the same row order, so the
    committed report does not churn between runs."""
    existing = [{"encoder": "b", "max_len": 64}, {"encoder": "a", "max_len": 128}]
    new = [{"encoder": "c", "max_len": 32}]
    first = T._merge_rows(new, existing, ("encoder", "max_len"))
    second = T._merge_rows(new, existing, ("encoder", "max_len"))
    assert first == second


def test_empty_new_rows_leave_the_report_intact():
    """`--skip-probe` contributes no probe rows and must not wipe the grid."""
    existing = [{"encoder": "mbert-base", "max_len": 192, "max_batch": 32}]
    assert T._merge_rows([], existing, ("encoder", "max_len")) == existing
