"""DINOv2 encoder component with strict local asset validation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .checkpoint import DINO_FEATURE_DIM, DINO_MODEL_REVISION, DinoV2Checkpoint
from .preprocess import ImageInput, prepare_dinov2_batch


class EncoderAssetError(RuntimeError):
    """Raised when the offline DINOv2 encoder assets are missing or incompatible."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the DINOv2 encoder") from exc
    return torch


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _encoder_weight_name(checkpoint: DinoV2Checkpoint) -> str:
    name = Path(checkpoint.encoder_weights).name
    if not name or name in {".", ".."}:
        raise EncoderAssetError("Invalid DINOv2 encoder_weights metadata")
    return name


def resolve_encoder_dir(
    checkpoint: DinoV2Checkpoint,
    *,
    search_roots: Iterable[str | Path] = (),
) -> Path:
    weight_name = _encoder_weight_name(checkpoint)
    attempted: list[Path] = []
    raw = Path(checkpoint.encoder_weights).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw.parent)
    if checkpoint.path is not None:
        candidates.append(checkpoint.path.parent / "dinov2-base")
        candidates.append(checkpoint.path.parent / "encoder")
        candidates.append(checkpoint.path.parent)
    for root in search_roots:
        base = Path(root).expanduser()
        candidates.extend((base / "dinov2-base", base / "encoder", base))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        attempted.append(resolved)
        if resolved.is_dir() and (resolved / weight_name).is_file():
            return resolved
    raise EncoderAssetError(
        "DINOv2 encoder directory not found; attempted: "
        + ", ".join(map(str, attempted))
    )


def _default_model_factory(model_dir: str | Path):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise EncoderAssetError(
            "Transformers is required for the DINOv2 component"
        ) from exc

    # The packaged GUI backend can run without a console-backed stderr handle.
    # Transformers uses tqdm during weight loading, and tqdm flushes stderr;
    # disabling that redundant progress UI prevents OSError on Windows while
    # leaving model loading and inference unchanged. Keep the logging import
    # optional so lightweight test/dummy Transformers modules remain valid.
    try:
        from transformers.utils import logging as transformers_logging
    except (ImportError, ModuleNotFoundError):
        transformers_logging = None
    if transformers_logging is not None:
        transformers_logging.disable_progress_bar()

    resolved = Path(model_dir).expanduser().resolve()
    return AutoModel.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=False,
    )


class DinoV2Encoder:
    def __init__(
        self,
        checkpoint: DinoV2Checkpoint,
        *,
        encoder_dir: str | Path | None = None,
        search_roots: Iterable[str | Path] = (),
        model_factory: Callable[[str | Path], Any] | None = None,
        device: str | None = None,
        use_fp16: bool = False,
    ) -> None:
        torch = _require_torch()
        self.checkpoint = checkpoint
        self.encoder_dir = (
            Path(encoder_dir).expanduser().resolve()
            if encoder_dir is not None
            else resolve_encoder_dir(checkpoint, search_roots=search_roots)
        )
        weight_path = self.encoder_dir / _encoder_weight_name(checkpoint)
        if not weight_path.is_file():
            raise EncoderAssetError(f"DINOv2 encoder weights not found: {weight_path}")
        actual = _sha256(weight_path)
        if actual != checkpoint.encoder_sha256:
            raise EncoderAssetError(
                f"DINOv2 encoder hash mismatch: expected {checkpoint.encoder_sha256}, got {actual}"
            )
        config_path = self.encoder_dir / "config.json"
        if not config_path.is_file():
            raise EncoderAssetError(f"DINOv2 encoder config not found: {config_path}")
        self.device = self._resolve_device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        factory = model_factory or _default_model_factory
        self.model = factory(self.encoder_dir)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if self.use_fp16 and hasattr(self.model, "half"):
            self.model.half()
        self.revision = DINO_MODEL_REVISION

    @staticmethod
    def _resolve_device(requested: str | None):
        torch = _require_torch()
        value = (requested or "").strip().lower()
        if value:
            device = torch.device(value)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise EncoderAssetError("CUDA was requested but is not available")
            return device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _last_hidden_state(output: Any):
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, dict) and "last_hidden_state" in output:
            return output["last_hidden_state"]
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise EncoderAssetError("DINOv2 encoder output does not contain last_hidden_state")

    def encode(
        self,
        images: Sequence[ImageInput],
        *,
        array_color: str = "rgb",
        batch_size: int = 16,
    ) -> np.ndarray:
        if not images:
            raise ValueError("At least one crop is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        torch = _require_torch()
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = prepare_dinov2_batch(
                images[start : start + batch_size],
                array_color=array_color,
            )
            tensor = torch.from_numpy(batch).to(self.device)
            if self.use_fp16:
                tensor = tensor.half()
            with torch.inference_mode():
                encoded = self.model(pixel_values=tensor)
            hidden = self._last_hidden_state(encoded)
            if hidden.ndim != 3 or hidden.shape[-1] != DINO_FEATURE_DIM:
                raise EncoderAssetError(
                    f"Expected DINOv2 hidden state (*, *, {DINO_FEATURE_DIM}), got {tuple(hidden.shape)}"
                )
            cls = hidden[:, 0, :].detach().float().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
                raise EncoderAssetError("DINOv2 encoder produced invalid embeddings")
            outputs.append(cls / norms)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
