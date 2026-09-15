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
        assert set(r) == {
            "encoder", "max_len", "fp16", "max_batch",
            "peak_gib", "floor_gib", "activation_gib", "note",
        }
        assert r["max_batch"] == 8  # 16 fails, 8 fits


def test_peak_gib_conversion():
    r = V.ProbeResult("e", 128, 16, 1, True, True, peak_bytes=3 * 1024**3)
    assert r.peak_gib == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The fixed floor — the cost a batch sweep cannot see (T-500, CLAUDE5.md)
# --------------------------------------------------------------------------


def test_fixed_floor_is_sixteen_bytes_per_parameter():
    """fp32 weights + fp32 gradients + AdamW's two fp32 moments."""
    assert V.fixed_floor_bytes(1_000_000) == 16_000_000


def test_activation_estimate_is_peak_minus_floor():
    r = V.ProbeResult(
        "e", 128, 16, 1, True, True, peak_bytes=3 * 1024**3, n_params=100_000_000
    )
    assert r.floor_gib == pytest.approx(100_000_000 * 16 / 1024**3)
    assert r.activation_gib == pytest.approx(r.peak_gib - r.floor_gib)


def test_activation_estimate_never_goes_negative():
    """A peak below the floor is the T-500 bug, not a negative activation
    cost; the property clamps rather than reporting nonsense."""
    r = V.ProbeResult(
        "e", 128, 16, 1, True, True, peak_bytes=1024**3, n_params=300_000_000
    )
    assert r.floor_gib > r.peak_gib
    assert r.activation_gib == 0.0


def test_largest_batch_short_circuits_when_the_floor_exceeds_the_device(monkeypatch):
    """An encoder that cannot fit at any batch size must be reported without
    probing every candidate — ten model loads to learn one arithmetic fact."""
    monkeypatch.setattr(V, "device_total_bytes", lambda device="cuda": 4 * 1024**3)
    monkeypatch.setattr(V, "encoder_param_count", lambda encoder: 279_000_000)
    calls = []

    def probe(*args, **kwargs):
        calls.append(args)
        raise AssertionError("should not probe an encoder that cannot fit")

    best = V.largest_batch(
        "xlm-r-base", 128, device="cuda", batches=(64, 32, 16, 8, 1), probe=probe
    )
    assert not best.fits
    assert calls == []
    assert "does not fit at any batch size" in best.error
    assert "4.16 GiB" in best.error  # 279M params x 16 bytes / 1024**3


def test_largest_batch_still_probes_when_the_floor_does_fit(monkeypatch):
    monkeypatch.setattr(V, "device_total_bytes", lambda device="cuda": 4 * 1024**3)
    monkeypatch.setattr(V, "encoder_param_count", lambda encoder: 34_000_000)
    best = V.largest_batch(
        "indicbert-v2", 128, device="cuda", batches=(64, 32, 16), probe=fake_probe(20)
    )
    assert best.fits and best.batch_size == 16


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


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_real_probe_step_includes_the_optimizer_state():
    """The T-500 regression test.

    GradScaler skips `optimizer.step()` when the first scaled gradients
    overflow, and AdamW allocates its moment buffers inside that call — so the
    probe used to report a peak *below* the encoder's own fixed floor, which is
    arithmetically impossible for a completed step. That is what made
    `xlm-r-base` look like it fitted in 2.76 GiB when it needs 4.14 GiB.
    """
    r = V.probe_step("indicbert-v2", 8, 64, device="cuda", fp16=True)
    assert r.fits
    assert r.n_params > 0
    assert r.peak_bytes >= V.fixed_floor_bytes(r.n_params), (
        f"peak {r.peak_gib:.2f} GiB is below the {r.floor_gib:.2f} GiB floor for "
        f"{r.n_params / 1e6:.0f}M parameters — the optimiser step did not allocate "
        "its state, so this measurement understates a real run"
    )
