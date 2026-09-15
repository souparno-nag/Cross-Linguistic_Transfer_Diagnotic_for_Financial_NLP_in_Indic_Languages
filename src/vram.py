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

**That last part silently did not happen until T-500 (CLAUDE5.md).**
``GradScaler.step`` *skips* ``optimizer.step()`` when the first scaled gradients
overflow, which on a synthetic batch of random token ids with an untrained head
they reliably do — so AdamW's two moment buffers were never allocated and the
probe under-reported peak memory by 8 bytes per parameter. The symptom was a
table saying ``xlm-r-base`` fits in 2.76 GiB when its weights, gradients and
optimiser state alone need 4.14 GiB, which is more than the card has. Any
recorded peak below an encoder's fixed floor is that bug; see
``fixed_floor_bytes``. ``probe_step`` now forces the allocation when the scaler
skipped it, and every result carries the floor alongside the measured peak so
the two can be compared.
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
    n_params: int = 0
    forced_optimizer_step: bool = False
    error: str | None = None

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / _GiB

    @property
    def floor_gib(self) -> float:
        """Fixed cost of weights + gradients + AdamW state, in GiB."""
        return fixed_floor_bytes(self.n_params) / _GiB

    @property
    def activation_gib(self) -> float:
        """What the measured peak leaves over the fixed floor — the only part
        that batch size and sequence length actually move."""
        return max(0.0, self.peak_gib - self.floor_gib)


def fixed_floor_bytes(n_params: int) -> int:
    """Bytes a full fine-tune needs before a single activation.

    Under AMP the master weights stay fp32 and AdamW keeps two fp32 moments
    per parameter, so the cost is 4 (weights) + 4 (gradients) + 8 (optimiser
    state) = **16 bytes per parameter**, independent of batch size and
    sequence length. An encoder whose floor exceeds the card cannot be fully
    fine-tuned on it at any shape, which is not something a batch sweep can
    discover.
    """
    return n_params * 16


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

        result.n_params = sum(p.numel() for p in model.parameters())

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

        # `scaler.step` is a no-op when the scaled gradients overflowed, and on
        # a synthetic batch at GradScaler's initial scale they usually do. AdamW
        # allocates `exp_avg` and `exp_avg_sq` lazily *inside* `step()`, so a
        # skipped first step leaves 8 bytes per parameter unmeasured and the
        # probe reports a peak a real run sails straight past. Force it. The
        # resulting weights are meaningless, which does not matter: this
        # function measures allocation, not learning.
        if not optimizer.state:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()
            result.forced_optimizer_step = True

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


def encoder_param_count(encoder: str) -> int:
    """Parameter count without downloading weights or touching the GPU.

    Built on the ``meta`` device from the model's config, so it allocates
    nothing. This is what lets :func:`largest_batch` refuse to probe an
    encoder whose fixed floor already exceeds the card, instead of OOMing
    once per candidate batch size.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(_resolve(encoder))
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    return sum(p.numel() for p in model.parameters())


def device_total_bytes(device: str = "cuda") -> int | None:
    """Total VRAM of ``device``, or ``None`` when it is not a visible GPU."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None
    index = int(device.split(":")[1]) if ":" in device else 0
    return int(torch.cuda.get_device_properties(index).total_memory)


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

    Short-circuits when the encoder's fixed floor already exceeds the card:
    no batch size can help, so probing every candidate would just be ten
    identical OOMs and ten model loads.
    """
    total = device_total_bytes(device)
    if total is not None:
        try:
            n_params = encoder_param_count(encoder)
        except Exception:  # config unreachable — fall through to real probing
            n_params = 0
        if n_params and fixed_floor_bytes(n_params) > total:
            floor = fixed_floor_bytes(n_params)
            return ProbeResult(
                encoder=encoder,
                max_len=max_len,
                batch_size=min(batches),
                grad_accum=grad_accum,
                fp16=fp16,
                fits=False,
                n_params=n_params,
                error=(
                    f"does not fit at any batch size: weights + gradients + AdamW "
                    f"state need {floor / _GiB:.2f} GiB before any activation, and "
                    f"the device has {total / _GiB:.2f} GiB"
                ),
            )

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
            # The floor is reported next to the peak so the T-500 failure mode
            # is visible in the table itself: a peak *below* the floor means
            # the optimiser state was never allocated and the row is wrong.
            row = {
                "encoder": encoder,
                "max_len": max_len,
                "fp16": fp16,
                "max_batch": best.batch_size if best.fits else 0,
                "peak_gib": round(best.peak_gib, 3) if best.fits else None,
                "floor_gib": round(best.floor_gib, 3) if best.n_params else None,
                "activation_gib": (
                    round(best.activation_gib, 3) if best.fits and best.n_params else None
                ),
                "note": "" if best.fits else (best.error or "does not fit"),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows
