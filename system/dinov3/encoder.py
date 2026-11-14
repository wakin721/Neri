"""Offline, float32 DINOv3 ViT-B/16 extraction using the official local source."""
from __future__ import annotations

import hashlib
import importlib
from itertools import islice
from pathlib import Path
import sys
import threading

import numpy as np
import torch

from system.dino.preprocess import preprocess_image

FEATURE_DIM = 768
PREPROCESSING = "letterbox224_imagenet"
_INFERENCE_LOCK = threading.RLock()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _official_model(source):
    source = Path(source).resolve()
    if not (source / "dinov3/hub/backbones.py").is_file():
        raise ValueError("DINOv3 encoder_code must contain the official dinov3 package")
    existing = sys.modules.get("dinov3")
    if existing is not None:
        actual = Path(existing.__file__).resolve()
        # A running process must not silently mix two versions of encoder code.
        if actual != source / "dinov3/__init__.py":
            def inventory(package):
                return {p.relative_to(package).as_posix(): sha256(p)
                        for p in package.rglob("*.py")}
            if inventory(actual.parent) != inventory(source / "dinov3"):
                raise RuntimeError("Different DINOv3 source is already loaded; restart Neri")
    sys.path.insert(0, str(source))
    try:
        factory = importlib.import_module("dinov3.hub.backbones").dinov3_vitb16
        return factory(pretrained=False)
    finally:
        sys.path.remove(str(source))


class DinoV3Encoder:
    """Extract only x_norm_clstoken, with frozen weights and no autocast."""

    def __init__(self, weights, source, encoder_sha256, *, device=None,
                 batch_size=32, model_factory=None, **_kwargs):
        self.weights = Path(weights).resolve()
        if sha256(self.weights) != encoder_sha256:
            raise ValueError("DINOv3 encoder weight SHA-256 does not match the model")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("DINOv3 batch_size must be a positive integer")
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        with _INFERENCE_LOCK:
            self.model = (model_factory or _official_model)(source)
            self.model.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True), strict=True)
            self.model = self.model.float().eval().requires_grad_(False).to(self.device)

    @property
    def use_fp16(self):
        return False

    @use_fp16.setter
    def use_fp16(self, _value):
        # Detector preferences do not override the calibrated encoder protocol.
        pass

    def encode(self, crops, *, array_color="rgb"):
        if not len(crops):
            raise ValueError("At least one crop is required")
        tensors = (preprocess_image(crop, array_color=array_color) for crop in crops)
        return self._encode_tensors(tensors, len(crops))

    def encode_preprocessed(self, tensors):
        """Encode tensors produced by preprocess_image, e.g. in decode workers."""
        if not len(tensors):
            raise ValueError("At least one preprocessed crop is required")
        return self._encode_tensors(iter(tensors), len(tensors))

    def _encode_tensors(self, tensors, count):
        output = []
        # Restore process-wide precision flags after every call, including OOM.
        with _INFERENCE_LOCK, torch.inference_mode():
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            old_cudnn = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = True
                for begin in range(0, count, self.batch_size):
                    batch = torch.stack(list(islice(tensors, self.batch_size)))
                    if batch.shape[1:] != (3, 224, 224) or batch.dtype != torch.float32 or not bool(torch.isfinite(batch).all()):
                        raise ValueError("Expected finite float32 preprocessed crops [N,3,224,224]")
                    batch = batch.to(self.device)
                    with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
                        values = self.model.forward_features(batch)["x_norm_clstoken"]
                    if values.shape != (len(batch), FEATURE_DIM) or not bool(torch.isfinite(values).all()):
                        raise ValueError("DINOv3 returned invalid CLS features")
                    if bool((torch.linalg.vector_norm(values, dim=1) <= 1e-12).any()):
                        raise ValueError("DINOv3 returned zero CLS features")
                    output.append(torch.nn.functional.normalize(values.float(), dim=1).cpu().numpy())
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul
                torch.backends.cudnn.allow_tf32 = old_cudnn
        return np.concatenate(output).astype(np.float32, copy=False)
