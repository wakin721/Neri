"""Offline Hugging Face DINOv2 ViT-B/14 encoder loader."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import DINO_FEATURE_DIM, DinoV2Checkpoint
from .preprocess import ImageInput, preprocess_image


class EncoderAssetError(RuntimeError):
    """Raised when local DINOv2 encoder assets cannot be verified or loaded."""


def resolve_encoder_dir(
    checkpoint: DinoV2Checkpoint,
    *,
    encoder_dir: str | Path | None = None,
    search_roots: Iterable[str | Path] = (),
) -> Path:
    declared = Path(checkpoint.encoder_weights)
    weight_name = declared.name
    candidates: list[Path] = []
    if encoder_dir is not None:
        candidates.append(Path(encoder_dir).expanduser())
    else:
        if checkpoint.path is not None:
            candidates.extend(
                [
                    checkpoint.path.parent / declared.parent,
                    checkpoint.path.parent / declared.parent.name,
                    checkpoint.path.parent / "model",
                ]
            )
        for root in search_roots:
            root_path = Path(root).expanduser()
            candidates.extend(
                [
                    root_path / declared.parent,
                    root_path / declared.parent.name,
                    root_path / "model",
                    root_path,
                ]
            )
        candidates.extend([Path.cwd() / declared.parent, Path.cwd() / "model"])

    attempted: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
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
        model_factory: Callable[[Path], object] | None = None,
        device: str | None = None,
        batch_size: int = 8,
        use_fp16: bool = False,
    ):
        self.checkpoint = checkpoint
        self.encoder_dir = resolve_encoder_dir(
            checkpoint,
            encoder_dir=encoder_dir,
            search_roots=search_roots,
        )
        weights = self.encoder_dir / Path(checkpoint.encoder_weights).name
        with weights.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual.lower() != checkpoint.encoder_sha256.lower():
            raise EncoderAssetError(
                f"DINOv2 encoder weight hash mismatch: expected {checkpoint.encoder_sha256}, got {actual}"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise EncoderAssetError("CUDA was requested but is not available")
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")

        factory = model_factory or _default_model_factory
        model = factory(self.encoder_dir)
        self.model = model.eval().requires_grad_(False).to(self.device)

    def encode(
        self,
        crops: Sequence[ImageInput],
        *,
        array_color: str = "rgb",
    ) -> np.ndarray:
        if not crops:
            raise ValueError("At least one crop is required")
        parts = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.stack(
                    [
                        preprocess_image(crop, array_color=array_color)
                        for crop in crops[start : start + self.batch_size]
                    ]
                ).to(self.device)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.use_fp16
                    else nullcontext()
                )
                with context:
                    outputs = self.model(pixel_values=batch)
                    hidden = getattr(outputs, "last_hidden_state", None)
                    if hidden is None or hidden.ndim != 3 or hidden.shape[0] != len(batch):
                        raise EncoderAssetError("DINOv2 did not return a valid last_hidden_state")
                    features = hidden[:, 0]
                if features.ndim != 2 or features.shape[1] != DINO_FEATURE_DIM:
                    raise EncoderAssetError(
                        f"DINOv2 returned shape {tuple(features.shape)}, expected (N, 768)"
                    )
                if not bool(torch.isfinite(features).all()):
                    raise EncoderAssetError("DINOv2 returned non-finite features")
                parts.append(F.normalize(features.float(), dim=1).cpu())
        result = torch.cat(parts, dim=0).numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise EncoderAssetError("DINOv2 returned non-finite normalized features")
        return result
