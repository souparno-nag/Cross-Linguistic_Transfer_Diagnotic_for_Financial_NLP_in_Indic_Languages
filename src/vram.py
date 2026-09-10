"""VRAM probing for the training pipeline (CLAUDE2.md T-205).

The 3050 has 4 GB and the desktop session already holds ~1 GB (T-203), so the
usable budget is nearer 3 GB and a batch size that "should" fit often does not.
This module finds, per encoder and sequence length, the largest batch that
survives a real forward → backward → optimiser step, so the shipped configs can
be set from measurement rather than hope.

The probe mirrors ``train.py``'s step exactly — same AMP path, same optimiser,
``attention_mask`` all ones so every row is full length (the worst case) — and
runs the optimiser step because AdamW allocates its moment buffers lazily on the
first one, which is a real jump in peak memory.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch

from .train import Classifier

# Batch sizes tried, largest first. The probe returns the first that fits.
DEFAULT_BATCHES = (64, 48, 32, 24, 16, 12, 8, 4, 2, 1)
DEFAULT_MAX_LENS = (64, 128, 256)
DEFAULT_ENCODERS = ("indicbert-v2", "xlm-r-base", "mbert-base")

_GiB = 1024**3


@dataclass
class ProbeResult:
    encoder: str
    max_len: int
    batch_size: int
    grad_accum: int
    fp16: bool
    fits: bool
    peak_bytes: int = 0
    error: str | None = None

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / _GiB


def free_gpu() -> None:
    """Return activations *and* cached blocks so the next probe starts clean."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _synthetic_batch(vocab_size: int, batch_size: int, max_len: int, device: str):
    ids = torch.randint(10, max(11, vocab_size), (batch_size, max_len), device=device)
    mask = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    return ids, mask, labels


def probe_step(
    encoder: str,
    batch_size: int,
    max_len: int,
    *,
    device: str = "cuda",
    grad_accum: int = 1,
    fp16: bool = True,
    seed: int = 0,
) -> ProbeResult:
    """Run one training step at this shape; report whether it fits and its peak.

    Any other RuntimeError (a real bug, not exhaustion) propagates.
    """
    torch.manual_seed(seed)
    result = ProbeResult(
        encoder=encoder,
        max_len=max_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        fp16=fp16,
        fits=False,
    )
    model = optimizer = None
    try:
        free_gpu()
        model = Classifier(_resolve(encoder), num_labels=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        use_amp = fp16 and device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        vocab = model.encoder.config.vocab_size

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            ids, mask, labels = _synthetic_batch(vocab, batch_size, max_len, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ids, mask)
                loss = torch.nn.functional.cross_entropy(logits, labels) / grad_accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
            result.peak_bytes = int(torch.cuda.max_memory_allocated())
        result.fits = True
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        result.error = f"OOM: {exc}".split("\n")[0]
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.error = f"OOM: {exc}".split("\n")[0]
    finally:
        del model, optimizer
        free_gpu()
    return result


def _resolve(encoder: str) -> str:
    from .data import resolve_encoder

    return resolve_encoder(encoder).hf_id


def largest_batch(
    encoder: str,
    max_len: int,
    *,
    device: str = "cuda",
    batches=DEFAULT_BATCHES,
    grad_accum: int = 1,
    fp16: bool = True,
    probe=probe_step,
) -> ProbeResult:
    """Largest batch from ``batches`` that fits at this sequence length.

    Returns the failing 1-row probe if not even a single row fits.
    """
    last = None
    for batch_size in sorted(batches, reverse=True):
        last = probe(
            encoder,
            batch_size,
            max_len,
            device=device,
            grad_accum=grad_accum,
            fp16=fp16,
        )
        if last.fits:
            return last
    return last


def budget_table(
    *,
    encoders=DEFAULT_ENCODERS,
    max_lens=DEFAULT_MAX_LENS,
    device: str = "cuda",
    fp16: bool = True,
    batches=DEFAULT_BATCHES,
    probe=probe_step,
    on_row=None,
) -> list[dict]:
    """Grid over (encoder, max_len) → the largest fitting batch and its peak."""
    rows: list[dict] = []
    for encoder in encoders:
        for max_len in sorted(max_lens):
            best = largest_batch(
                encoder,
                max_len,
                device=device,
                batches=batches,
                fp16=fp16,
                probe=probe,
            )
            row = {
                "encoder": encoder,
                "max_len": max_len,
                "fp16": fp16,
                "max_batch": best.batch_size if best.fits else 0,
                "peak_gib": round(best.peak_gib, 3) if best.fits else None,
                "note": "" if best.fits else (best.error or "does not fit"),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows
