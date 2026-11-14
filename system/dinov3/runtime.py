"""Versioned deployment manifests for the Ridge + Within DINOv3 model."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath

from .encoder import FEATURE_DIM, PREPROCESSING, DinoV3Encoder, sha256

ARCHITECTURE = "dinov3_vitb16"
HEAD_TYPE = "ridge_within"


def resolve_manifest_asset(manifest, raw):
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\\" in raw:
        raise ValueError("DINOv3 asset must be a relative POSIX path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ":" in raw or any(p in {"", ".", ".."} for p in raw.split("/")):
        raise ValueError("Unsafe DINOv3 asset path")
    root = Path(manifest).resolve().parent
    candidate = root.joinpath(*relative.parts).resolve()
    candidate.relative_to(root)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def read_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(schema_version=1, backend="dinov3", architecture=ARCHITECTURE,
                    head_type=HEAD_TYPE, feature_dim=FEATURE_DIM, preprocessing=PREPROCESSING,
                    event_aggregation="complete_sequence", requires_detector=True,
                    supports_video_fast=False, supports_video_all=False)
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported DINOv3 Ridge/Within manifest contract")
    for field in ("encoder_sha256", "checkpoint_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid DINOv3 {field}")
    for field in ("checkpoint", "encoder_weights", "encoder_code"):
        resolve_manifest_asset(path, payload.get(field))
    if "calibration" in payload or "calibration_sha256" in payload:
        digest = payload.get("calibration_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid DINOv3 calibration_sha256")
        raw = payload.get("calibration")
        if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw or any(
            part in {"", ".", ".."} for part in raw.split("/")
        ) or PurePosixPath(raw).is_absolute():
            raise ValueError("Unsafe DINOv3 calibration path")
    return payload


@dataclass
class DinoV3Runtime:
    manifest_path: Path
    model: object
    classifier: object
    registry: object = None
    feedback: object = None

    @property
    def checkpoint(self):
        return self.model

    def close(self):
        self.classifier.encoder = None


def load_dinov3_model(model_path, *, device=None, encoder_factory=None, **_kwargs):
    from .model import DinoV3RidgeWithinModel
    from .classifier import DinoV3Classifier

    path = Path(model_path).expanduser().resolve()
    payload = read_manifest(path)
    checkpoint = resolve_manifest_asset(path, payload["checkpoint"])
    if sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("DINOv3 classifier SHA-256 mismatch")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    if model.dimension != FEATURE_DIM:
        raise ValueError("DINOv3 model must use 768-dimensional CLS features")
    if model.provenance.get("encoder_sha256") != payload["encoder_sha256"]:
        raise ValueError("DINOv3 model and encoder provenance differ")
    if model.provenance.get("preprocessing") != PREPROCESSING:
        raise ValueError("DINOv3 preprocessing provenance differs")
    if model.calibration is None:
        raise ValueError("DINOv3 model requires calibration before deployment")
    weights = resolve_manifest_asset(path, payload["encoder_weights"])
    if sha256(weights) != payload["encoder_sha256"]:
        raise ValueError("DINOv3 encoder SHA-256 mismatch")
    encoder = (encoder_factory or DinoV3Encoder)(
        weights,
        resolve_manifest_asset(path, payload["encoder_code"]), payload["encoder_sha256"], device=device)
    return DinoV3Runtime(path, model, DinoV3Classifier(model, encoder=encoder))
