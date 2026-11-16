"""Versioned deployment manifests for the Ridge + Within DINOv3 model."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath

from .encoder import FEATURE_DIM, PREPROCESSING, sha256
from .experiment import PRODUCTION_RECIPE, head_type
from .display import model_display_name

ARCHITECTURE = "dinov3_vitb16"
HEAD_TYPE = "ridge_within"
ENCODER_BACKENDS = ("pytorch", "litert")


def _validate_sha256(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid DINOv3 {field}")
    return value


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
                    feature_dim=FEATURE_DIM, preprocessing=PREPROCESSING,
                    event_aggregation="complete_sequence", requires_detector=True,
                    supports_video_all=False)
    if (
        not isinstance(payload, dict)
        or any(payload.get(k) != v for k, v in expected.items())
        or not isinstance(payload.get("supports_video_fast"), bool)
        or not isinstance(payload.get("supports_video_all"), bool)
    ):
        raise ValueError("Unsupported DINOv3 Ridge/Within manifest contract")
    if payload.get("head_type") != head_type(payload.get("inference_recipe", PRODUCTION_RECIPE)):
        raise ValueError("DINOv3 head type and inference recipe differ")
    encoder_backend = payload.get("encoder_backend", "pytorch")
    if not isinstance(encoder_backend, str) or encoder_backend not in ENCODER_BACKENDS:
        raise ValueError(f"Unsupported DINOv3 encoder_backend: {encoder_backend!r}")
    for field in ("encoder_sha256", "checkpoint_sha256"):
        _validate_sha256(payload.get(field), field)
    checkpoint = resolve_manifest_asset(path, payload.get("checkpoint"))
    if not checkpoint.is_file():
        raise ValueError("DINOv3 checkpoint must be a file")
    weights = resolve_manifest_asset(path, payload.get("encoder_weights"))
    if encoder_backend == "litert":
        _validate_sha256(payload.get("encoder_litert_sha256"), "encoder_litert_sha256")
        if not weights.is_file() or weights.suffix.lower() not in {".tflite", ".litert"}:
            raise ValueError("DINOv3 LiteRT encoder_weights must be a .tflite or .litert file")
    else:
        if not weights.is_file() or weights.suffix.lower() not in {".pth", ".pt"}:
            raise ValueError("DINOv3 PyTorch encoder_weights must be a .pth or .pt file")
        code = resolve_manifest_asset(path, payload.get("encoder_code"))
        if not code.is_dir():
            raise ValueError("DINOv3 encoder_code must be a directory")
    if "calibration" in payload or "calibration_sha256" in payload:
        digest = payload.get("calibration_sha256")
        _validate_sha256(digest, "calibration_sha256")
        raw = payload.get("calibration")
        if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw or any(
            part in {"", ".", ".."} for part in raw.split("/")
        ) or PurePosixPath(raw).is_absolute():
            raise ValueError("Unsafe DINOv3 calibration path")
    return payload


def load_manifest_encoder(manifest_path, payload, device=None, *, encoder_factory=None, **kwargs):
    path = Path(manifest_path).expanduser().resolve()
    encoder_backend = payload.get("encoder_backend", "pytorch")
    if encoder_backend not in ENCODER_BACKENDS:
        raise ValueError(f"Unsupported DINOv3 encoder_backend: {encoder_backend!r}")
    weights = resolve_manifest_asset(path, payload.get("encoder_weights"))
    if encoder_backend == "litert":
        if not weights.is_file() or weights.suffix.lower() not in {".tflite", ".litert"}:
            raise ValueError("DINOv3 LiteRT encoder_weights must be a .tflite or .litert file")
        litert_sha256 = _validate_sha256(payload.get("encoder_litert_sha256"), "encoder_litert_sha256")
        if sha256(weights) != litert_sha256:
            raise ValueError("DINOv3 LiteRT encoder SHA-256 mismatch")
        if encoder_factory is None:
            from . import litert_encoder as _litert_encoder

            factory = _litert_encoder.LiteRtDinoV3Encoder
        else:
            factory = encoder_factory
        return factory(weights, litert_sha256, device="cpu", **kwargs)
    if not weights.is_file() or weights.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("DINOv3 PyTorch encoder_weights must be a .pth or .pt file")
    if sha256(weights) != payload["encoder_sha256"]:
        raise ValueError("DINOv3 encoder SHA-256 mismatch")
    code = resolve_manifest_asset(path, payload.get("encoder_code"))
    if not code.is_dir():
        raise ValueError("DINOv3 encoder_code must be a directory")
    if encoder_factory is None:
        from . import encoder as _encoder

        factory = _encoder.DinoV3Encoder
    else:
        factory = encoder_factory
    return factory(weights, code, payload["encoder_sha256"], device=device, **kwargs)


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


def load_dinov3_model(model_path, *, device=None, encoder_factory=None, use_shared_learning=True, **_kwargs):
    from .model import DinoV3RidgeWithinModel
    from .classifier import DinoV3Classifier

    path = Path(model_path).expanduser().resolve()
    payload = read_manifest(path)
    resource_path=path
    from .shared_learning import family_descriptor
    if use_shared_learning and family_descriptor(path,payload) is not None:
        from system.backend.dinov3_learning import resolve_learning_snapshot
        path,payload=resolve_learning_snapshot(path,payload)
    checkpoint = resolve_manifest_asset(path, payload["checkpoint"])
    if sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("DINOv3 classifier SHA-256 mismatch")
    model = DinoV3RidgeWithinModel.load(checkpoint)
    if model.inference_recipe != payload.get("inference_recipe", PRODUCTION_RECIPE):
        raise ValueError("DINOv3 checkpoint and manifest inference recipes differ")
    if model.dimension != FEATURE_DIM:
        raise ValueError("DINOv3 model must use 768-dimensional CLS features")
    if model.provenance.get("encoder_sha256") != payload["encoder_sha256"]:
        raise ValueError("DINOv3 model and encoder provenance differ")
    if model.provenance.get("preprocessing") != PREPROCESSING:
        raise ValueError("DINOv3 preprocessing provenance differs")
    if model.calibration is None:
        raise ValueError("DINOv3 model requires calibration before deployment")
    encoder = load_manifest_encoder(
        path,
        payload,
        device=device,
        encoder_factory=encoder_factory,
        **_kwargs,
    )
    classifier = DinoV3Classifier(model, encoder=encoder)
    classifier.display_name = model_display_name(payload)
    if payload.get("encoder_backend", "pytorch") == "litert":
        classifier.model_fingerprint = (
            f"litert:{payload['encoder_litert_sha256']}:"
            f"{payload['encoder_sha256']}:{payload['checkpoint_sha256']}"
        )
    else:
        classifier.model_fingerprint = f"{payload['encoder_sha256']}:{payload['checkpoint_sha256']}"
    return DinoV3Runtime(resource_path, model, classifier)
