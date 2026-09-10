"""Tests for T-205's VRAM probing.

The search logic (largest fitting batch, grid assembly) is checked with a fake
probe so it runs on CPU. The real GPU probe is one ``slow`` test, skipped when
there is no CUDA device.
"""

import torch
import pytest

from src import vram as V


def fake_probe(fits_at_or_below: int):
    """A probe that 'fits' when batch_size <= threshold, peak scaled to batch."""

    def probe(encoder, batch_size, max_len, *, device, grad_accum, fp16):
        ok = batch_size <= fits_at_or_below
        return V.ProbeResult(
            encoder=encoder,
            max_len=max_len,
            batch_size=batch_size,
            grad_accum=grad_accum,
            fp16=fp16,
            fits=ok,
            peak_bytes=batch_size * (1024**3) // 8 if ok else 0,
            error=None if ok else "OOM: synthetic",
        )

    return probe


def test_largest_batch_returns_the_biggest_that_fits():
    best = V.largest_batch(
        "mbert-base", 128, device="cpu",
        batches=(64, 32, 16, 8, 4), probe=fake_probe(20),
    )
    assert best.fits
    assert best.batch_size == 16  # 32 fails, 16 is the next tried


def test_largest_batch_reports_failure_when_nothing_fits():
    best = V.largest_batch(
        "xlm-r-base", 256, device="cpu",
        batches=(8, 4, 2, 1), probe=fake_probe(0),
    )
    assert not best.fits
    assert best.batch_size == 1
    assert "OOM" in best.error


def test_budget_table_covers_the_grid_and_streams_rows():
    seen = []
    rows = V.budget_table(
        encoders=("mbert-base", "xlm-r-base"),
        max_lens=(64, 128),
        device="cpu",
        batches=(16, 8, 4),
        probe=fake_probe(10),
        on_row=seen.append,
    )
    assert len(rows) == 4
    assert seen == rows
    for r in rows:
        assert set(r) == {"encoder", "max_len", "fp16", "max_batch", "peak_gib", "note"}
        assert r["max_batch"] == 8  # 16 fails, 8 fits


def test_peak_gib_conversion():
    r = V.ProbeResult("e", 128, 16, 1, True, True, peak_bytes=3 * 1024**3)
    assert r.peak_gib == pytest.approx(3.0)


def test_synthetic_batch_shapes():
    ids, mask, labels = V._synthetic_batch(30000, 8, 64, "cpu")
    assert ids.shape == (8, 64)
    assert mask.shape == (8, 64)
    assert labels.shape == (8,)
    assert mask.min() == 1 and ids.min() >= 10


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_real_probe_step_reports_peak():
    r = V.probe_step("mbert-base", 8, 64, device="cuda", fp16=True)
    assert r.fits
    assert r.peak_bytes > 0
