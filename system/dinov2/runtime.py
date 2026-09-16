"""Manifest-driven DINOv2 runtime construction."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .checkpoint import (
    DINO_ARCHITECTURE_ID,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
    DinoV2Checkpoint,
    load_checkpoint,
)
from .state import default_dinov2_state_root


class DinoV2ManifestError(ValueError):
    """Raised when a model manifest is not a valid DINOv2 deployment manifest."""


@dataclass(frozen=True)
class DinoV2Runtime:
    manifest_path: Path
    checkpoint: DinoV2Checkpoint
    classifier: Any
    registry: Any
    feedback: Any
    owns_registry: bool
    owns_feedback: bool

    def close(self) -> None:
        if self.owns_feedback and self.feedback is not None:
            self.feedback.close()
        if self.owns_registry and self.registry is not None:
            self.registry.close()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV2ManifestError(f"Invalid DINOv2 manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise DinoV2ManifestError("DINOv2 manifest must be a JSON object")
    backend = payload.get("backend")
    if backend == "dinov3":
        raise DinoV2ManifestError("DINOv3 模型已不再支持；请安装或选择 DINOv2 模型。")
    if backend != "dinov2":
        raise DinoV2ManifestError("Selected classification model is not a DINOv2 manifest")
    if payload.get("schema_version") != 2:
        raise DinoV2ManifestError("Unsupported DINOv2 manifest schema_version")
    return payload


def resolve_dinov2_manifest(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name.endswith(".neri.json"):
        payload = _load_manifest(path)
        checkpoint_name = payload.get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise DinoV2ManifestError("DINOv2 manifest checkpoint is missing")
        checkpoint = (path.parent / checkpoint_name).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINOv2 classifier checkpoint not found: {checkpoint}")
        return path
    if path.is_file():
        legacy_seen = False
        for manifest in sorted(path.parent.glob("*.neri.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            checkpoint_name = payload.get("checkpoint")
            if not isinstance(checkpoint_name, str):
                continue
            if (manifest.parent / checkpoint_name).resolve() != path:
                continue
            if payload.get("backend") == "dinov3":
                legacy_seen = True
                continue
            return _load_manifest(manifest) and manifest.resolve()
        if legacy_seen:
            raise DinoV2ManifestError("DINOv3 模型已不再支持；请安装或选择 DINOv2 模型。")
    raise DinoV2ManifestError(f"No DINOv2 manifest found for classification model: {path}")


def _validate_manifest_contract(payload: dict[str, Any], checkpoint: DinoV2Checkpoint) -> None:
    if payload.get("architecture") != DINO_ARCHITECTURE_ID:
        raise DinoV2ManifestError(f"Unsupported DINOv2 architecture: {payload.get('architecture')}")
    if payload.get("feature_dim") != DINO_FEATURE_DIM:
        raise DinoV2ManifestError(f"Unsupported DINOv2 feature dimension: {payload.get('feature_dim')}")
    checks = (
        ("encoder_sha256", checkpoint.encoder_sha256),
        ("preprocessing", DINO_PREPROCESSING),
        ("event_aggregation", DINO_EVENT_AGGREGATION),
    )
    for field, expected in checks:
        if payload.get(field) != expected:
            raise DinoV2ManifestError(f"DINOv2 manifest {field} does not match checkpoint")


def _validate_store(store: Any, checkpoint: DinoV2Checkpoint, label: str) -> None:
    if store is None:
        return
    if getattr(store, "model_fingerprint", None) != checkpoint.fingerprint:
        raise ValueError(f"{label} model fingerprint does not match selected classifier")


def _create_registry(root: Path, checkpoint: DinoV2Checkpoint):
    from .registry import SpeciesRegistry, registry_path_for_fingerprint
    return SpeciesRegistry(
        registry_path_for_fingerprint(root, checkpoint.fingerprint),
        model_fingerprint=checkpoint.fingerprint,
    )


def _create_feedback(registry: Any, checkpoint: DinoV2Checkpoint):
    from .feedback import HumanFeedbackStore, feedback_path_for_registry
    return HumanFeedbackStore(
        feedback_path_for_registry(registry.path),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=checkpoint.rejection,
    )


def load_dinov2_model(
    model_path: str | Path,
    *,
    registry: Any = None,
    feedback: Any = None,
    state_root: str | Path | None = None,
    encoder_factory: Callable[..., Any] | None = None,
    device: str | None = None,
    use_fp16: bool = False,
) -> DinoV2Runtime:
    from .classifier import DinoV2Classifier
    from .encoder import DinoV2Encoder

    manifest_path = resolve_dinov2_manifest(model_path)
    payload = _load_manifest(manifest_path)
    checkpoint = load_checkpoint(manifest_path.parent / str(payload["checkpoint"]))
    _validate_manifest_contract(payload, checkpoint)

    owns_registry = registry is None
    owns_feedback = feedback is None
    created_registry = None
    created_feedback = None
    try:
        if registry is None:
            root = Path(state_root).expanduser().resolve() if state_root is not None else default_dinov2_state_root()
            registry = _create_registry(root, checkpoint)
            created_registry = registry
        else:
            _validate_store(registry, checkpoint, "Registry")

        if feedback is None:
            feedback = _create_feedback(registry, checkpoint)
            created_feedback = feedback
        else:
            _validate_store(feedback, checkpoint, "Feedback")

        factory = encoder_factory or DinoV2Encoder
        encoder = factory(
            checkpoint,
            search_roots=(manifest_path.parent,),
            device=device,
            use_fp16=use_fp16,
        )
        classifier = DinoV2Classifier(
            checkpoint,
            encoder=encoder,
            feedback=feedback,
            registry=registry,
        )
        return DinoV2Runtime(
            manifest_path=manifest_path,
            checkpoint=checkpoint,
            classifier=classifier,
            registry=registry,
            feedback=feedback,
            owns_registry=owns_registry,
            owns_feedback=owns_feedback,
        )
    except Exception:
        if created_feedback is not None:
            created_feedback.close()
        if created_registry is not None:
            created_registry.close()
        raise
