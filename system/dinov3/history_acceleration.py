"""Optional FP32 CUDA history adaptation with the NumPy protocol as fallback.

PyTorch is loaded only for sufficiently large jobs whose encoder uses CUDA.
Grouping, causal history selection, classification and calibrated gates remain
in their existing implementations.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .context import HistoryBatch, adapt_with_history

logger = logging.getLogger(__name__)

_CUDA_MIN_ROWS = 512
_CUDA_MAX_CHUNK = 512
_CUDA_MIN_CHUNK = 32


class _CudaHistoryUnavailable(RuntimeError):
    """A CUDA operation failed; the whole adaptation may be retried on CPU."""


def _cuda_chunk(torch, raw, normalized, history, begin, temperature, alpha):
    end = begin + len(history)
    valid = history >= 0
    rows = history.clamp(min=0)
    logits = torch.einsum("bd,bmd->bm", normalized[begin:end], normalized[rows]) / temperature
    logits = logits.masked_fill(~valid, -torch.inf)
    maximum = logits.max(dim=1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.exp(logits - maximum).masked_fill(~valid, 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    # Pool original vectors, not their normalized versions: this distinction
    # matters for non-unit cached features and radial rejection calibration.
    pooled = torch.einsum("bm,bmd->bd", weights, raw[rows])
    adapted = torch.nn.functional.normalize(
        (1.0 - alpha) * raw[begin:end] + alpha * pooled, dim=-1, eps=1e-12,
    )
    return torch.where(valid.any(dim=1, keepdim=True), adapted, raw[begin:end])


def _adapt_with_history_cuda(raw, history, *, device, temperature, alpha, progress_fn=None):
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        raise _CudaHistoryUnavailable(f"PyTorch cannot be loaded: {exc}") from exc

    from .encoder import _INFERENCE_LOCK

    # Share the encoder's lock because matmul precision is process-wide.
    # Explicitly disable ambient autocast and restore the caller's precision.
    with _INFERENCE_LOCK, torch.inference_mode():
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type="cuda", enabled=False):
                try:
                    if not torch.cuda.is_available():
                        raise _CudaHistoryUnavailable("CUDA is unavailable")
                    free, _ = torch.cuda.mem_get_info(device)
                    persistent = raw.nbytes * 2 + history.nbytes
                    bytes_per_query = max(1, history.shape[1] * raw.shape[1] * 4 * 3 + raw.shape[1] * 32)
                    budget = int(free * 0.8) - persistent
                    minimum = min(_CUDA_MIN_CHUNK, len(raw))
                    if budget < bytes_per_query * minimum:
                        raise _CudaHistoryUnavailable("insufficient free VRAM for history buffers")
                    chunk = min(_CUDA_MAX_CHUNK, len(raw), budget // bytes_per_query)
                    values = torch.from_numpy(raw).to(device=device)
                    normalized = torch.nn.functional.normalize(values, dim=-1, eps=1e-12)
                    indices = torch.from_numpy(history).to(device=device)
                except (RuntimeError, AssertionError) as exc:
                    raise _CudaHistoryUnavailable(str(exc)) from exc

                output = raw.copy()
                begin = 0
                while begin < len(raw):
                    end = min(begin + chunk, len(raw))
                    block = None
                    retry = False
                    try:
                        block = _cuda_chunk(torch, values, normalized, indices[begin:end], begin, temperature, alpha)
                        # Download completed chunks so cancellation/progress is
                        # checked after actual execution, with bounded VRAM.
                        output[begin:end] = block.cpu().numpy()
                    except torch.OutOfMemoryError as exc:
                        if chunk <= minimum:
                            raise _CudaHistoryUnavailable(f"CUDA OOM at chunk size {chunk}: {exc}") from exc
                        chunk = max(minimum, chunk // 2)
                        retry = True
                        logger.warning("CUDA history OOM; retrying with chunk size %d", chunk)
                    except RuntimeError as exc:
                        raise _CudaHistoryUnavailable(str(exc)) from exc
                    finally:
                        block = None
                    if retry:
                        # The exception/failed tensor frame has left scope.
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                        continue
                    begin = end
                    # Callback errors (including cancellation) must propagate;
                    # they are not CUDA failures and must never trigger a retry.
                    if progress_fn is not None:
                        progress_fn(begin, len(raw))
                return output
        finally:
            torch.set_float32_matmul_precision(precision)


def adapt_history_for_job(
    raw: np.ndarray,
    history: np.ndarray | HistoryBatch,
    *,
    device: str | None = None,
    temperature: float = 0.1,
    alpha: float = 0.1,
    progress_fn=None,
) -> np.ndarray:
    """Use the encoder's CUDA device for large jobs; otherwise use NumPy.

    Never changes precision in response to an OOM. A failed CUDA attempt is
    discarded and the entire result recomputed using the CPU reference.
    """
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    indices = np.ascontiguousarray(history.indices if isinstance(history, HistoryBatch) else history, dtype=np.int64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("raw features must be a finite [N, D] matrix")
    if indices.ndim != 2 or len(indices) != len(raw):
        raise ValueError("history must be a [N, H] matrix aligned with raw features")
    if indices.size and indices.max() >= len(raw):
        raise ValueError("history indices must refer to raw feature rows")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(alpha):
        raise ValueError("temperature must be positive and alpha must be finite")

    started = time.perf_counter()
    published = 0

    def report(done, total):
        nonlocal published
        published = max(published, done)
        if progress_fn is not None:
            progress_fn(published, total)

    report(0, len(raw))
    if not np.any(indices >= 0):
        report(len(raw), len(raw))
        logger.info("DINOv3 history backend=none rows=%d (no causal history)", len(raw))
        return raw.copy()

    if isinstance(device, str) and device.startswith("cuda") and len(raw) >= _CUDA_MIN_ROWS:
        try:
            output = _adapt_with_history_cuda(
                raw, indices, device=device, temperature=temperature, alpha=alpha, progress_fn=report,
            )
        except _CudaHistoryUnavailable as exc:
            logger.warning("DINOv3 CUDA history unavailable; using CPU: %s", exc)
        else:
            logger.info("DINOv3 history backend=cuda device=%s rows=%d elapsed=%.3fs",
                        device, len(raw), time.perf_counter() - started)
            return output

    output = adapt_with_history(raw, indices, temperature=temperature, alpha=alpha, progress_fn=report)
    logger.info("DINOv3 history backend=cpu rows=%d elapsed=%.3fs", len(raw), time.perf_counter() - started)
    return output
