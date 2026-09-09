"""Lazy native DINOv3 ViT-B/16 encoder loader."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import DINO_FEATURE_DIM, DinoV3Checkpoint
from .component import DINO_BACKBONE_SHA256, dinov3_component_paths
from .preprocess import ImageInput, preprocess_image


class EncoderAssetError(RuntimeError):
    pass


def resolve_encoder_weights(
    checkpoint: DinoV3Checkpoint,
    *,
    weights_path=None,
    search_roots: Iterable[str | Path] = (),
) -> Path:
    raw = Path(weights_path or checkpoint.encoder_weights).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if checkpoint.path is not None:
            candidates.extend(
                (
                    checkpoint.path.parent / raw,
                    checkpoint.path.parent / raw.name,
                )
            )
        if weights_path is None and checkpoint.encoder_sha256.lower() == DINO_BACKBONE_SHA256:
            candidates.append(dinov3_component_paths().backbone)
        candidates.extend(Path(root).expanduser() / raw for root in search_roots)
        candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise EncoderAssetError(
        "DINOv3 encoder weights not found; attempted: "
        + ", ".join(map(str, candidates))
    )


def _default_model_factory(source_root=None):
    configured = source_root
    if configured is None:
        installed_root = dinov3_component_paths().source_root
        if (installed_root / "dinov3").is_dir():
            configured = installed_root
    if configured is None:
        configured = os.environ.get("NERI_DINOV3_SOURCE")
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.name == "dinov3":
            root = root.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("dinov3.hub.backbones")
    except ImportError as exc:
        raise EncoderAssetError(
            "Meta DINOv3 source is unavailable; install the DINOv3 component or set NERI_DINOV3_SOURCE"
        ) from exc
    return getattr(module, "dinov3_vitb16")


def _load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


class DinoV3Encoder:
    def __init__(
        self,
        checkpoint: DinoV3Checkpoint,
        *,
        weights_path=None,
        source_root=None,
        search_roots: Iterable[str | Path] = (),
        model_factory: Callable | None = None,
        device: str | None = None,
        batch_size: int = 8,
        use_fp16: bool = False,
    ):
        self.checkpoint = checkpoint
        self.weights_path = resolve_encoder_weights(
            checkpoint,
            weights_path=weights_path,
            search_roots=search_roots,
        )
        with self.weights_path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual.lower() != checkpoint.encoder_sha256.lower():
            raise EncoderAssetError(
                f"DINOv3 encoder weight hash mismatch: expected {checkpoint.encoder_sha256}, got {actual}"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise EncoderAssetError("CUDA was requested but is not available")
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")
        factory = model_factory or _default_model_factory(source_root)
        try:
            model = factory(pretrained=False)
        except TypeError:
            model = factory()
        model.load_state_dict(_load_state_dict(self.weights_path), strict=True)
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
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16
            else nullcontext()
        )
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.stack(
                    [
                        preprocess_image(crop, array_color=array_color)
                        for crop in crops[start : start + self.batch_size]
                    ]
                ).to(self.device)
                with autocast:
                    features = self.model.forward_features(batch)["x_norm_clstoken"]
                if features.ndim != 2 or features.shape[1] != DINO_FEATURE_DIM:
                    raise EncoderAssetError(
                        f"Native DINOv3 returned shape {tuple(features.shape)}, expected (N, 768)"
                    )
                parts.append(F.normalize(features.float(), dim=1).cpu())
        return torch.cat(parts, dim=0).numpy().astype(np.float32, copy=False)
