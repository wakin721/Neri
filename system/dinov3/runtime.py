"""Manifest-driven native DINOv3 runtime construction."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .checkpoint import (
    DINO_BACKBONE,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
    DinoV3Checkpoint,
    load_checkpoint,
)
from .feedback import HumanFeedbackStore, feedback_path_for_registry
from .registry import SpeciesRegistry, registry_path_for_fingerprint
from .rejection import MultiDualRejectionConfig, parse_multi_dual_rejection
from .state import default_dinov3_state_root

if TYPE_CHECKING:
    from .classifier import DinoV3Classifier


class DinoV3ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DinoV3Runtime:
    manifest_path: Path
    checkpoint: DinoV3Checkpoint
    classifier: "DinoV3Classifier"
    registry: SpeciesRegistry
    feedback: HumanFeedbackStore
    owns_registry: bool
    owns_feedback: bool

    def close(self) -> None:
        """Close stores owned by this runtime without touching injected stores."""
        if self.owns_feedback:
            self.feedback.close()
        if self.owns_registry:
            self.registry.close()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV3ManifestError(f"Invalid DINOv3 manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("backend") != "dinov3":
        raise DinoV3ManifestError("Selected classification model is not a DINOv3 manifest")
    return payload


def resolve_dinov3_manifest(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name.endswith(".neri.json"):
        payload = _load_manifest(path)
        checkpoint_name = payload.get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise DinoV3ManifestError("DINOv3 manifest checkpoint is missing")
        checkpoint = (path.parent / checkpoint_name).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINOv3 classifier checkpoint not found: {checkpoint}")
        return path
    if path.is_file():
        for manifest in sorted(path.parent.glob("*.neri.json")):
            try:
                payload = _load_manifest(manifest)
            except DinoV3ManifestError:
                continue
            checkpoint_name = payload.get("checkpoint")
            if isinstance(checkpoint_name, str) and (manifest.parent / checkpoint_name).resolve() == path:
                return manifest.resolve()
    raise DinoV3ManifestError(f"No DINOv3 manifest found for classification model: {path}")


def _validate_manifest_contract(payload: dict[str, Any], checkpoint: DinoV3Checkpoint) -> None:
    architecture = payload.get("architecture")
    feature_dim = payload.get("feature_dim")
    if architecture not in (None, "", DINO_BACKBONE):
        raise DinoV3ManifestError(f"Unsupported DINOv3 architecture: {architecture}")
    if feature_dim not in (None, DINO_FEATURE_DIM):
        raise DinoV3ManifestError(f"Unsupported DINOv3 feature dimension: {feature_dim}")
    checks = (
        ("encoder_sha256", checkpoint.encoder_sha256),
        ("preprocessing", DINO_PREPROCESSING),
        ("event_aggregation", DINO_EVENT_AGGREGATION),
    )
    for field, expected in checks:
        value = payload.get(field)
        if value not in (None, "", expected):
            raise DinoV3ManifestError(f"DINOv3 manifest {field} does not match checkpoint")


def _load_rejection_config(payload: dict[str, Any]) -> MultiDualRejectionConfig | None:
    try:
        return parse_multi_dual_rejection(payload.get("rejection"))
    except ValueError as exc:
        raise DinoV3ManifestError(f"Invalid DINOv3 rejection config: {exc}") from exc


def load_dinov3_model(
    model_path: str | Path,
    *,
    registry: SpeciesRegistry | None = None,
    feedback: HumanFeedbackStore | None = None,
    state_root: str | Path | None = None,
    encoder_factory: Callable[..., Any] | None = None,
    device: str | None = None,
    use_fp16: bool = False,
) -> DinoV3Runtime:
    from .classifier import DinoV3Classifier
    from .encoder import DinoV3Encoder

    manifest_path = resolve_dinov3_manifest(model_path)
    payload = _load_manifest(manifest_path)
    checkpoint_name = str(payload["checkpoint"])
    checkpoint = load_checkpoint(manifest_path.parent / checkpoint_name)
    _validate_manifest_contract(payload, checkpoint)
    rejection = _load_rejection_config(payload)

    owns_registry = registry is None
    owns_feedback = feedback is None
    created_registry: SpeciesRegistry | None = None
    created_feedback: HumanFeedbackStore | None = None

    try:
        if registry is None:
            root = (
                Path(state_root).expanduser().resolve()
                if state_root is not None
                else default_dinov3_state_root()
            )
            registry = SpeciesRegistry(
                registry_path_for_fingerprint(root, checkpoint.fingerprint),
                model_fingerprint=checkpoint.fingerprint,
            )
            created_registry = registry
        elif registry.model_fingerprint != checkpoint.fingerprint:
            raise ValueError("Registry model fingerprint does not match selected classifier")

        if feedback is None:
            feedback = HumanFeedbackStore(
                feedback_path_for_registry(registry.path),
                model_fingerprint=checkpoint.fingerprint,
                checkpoint_classes=checkpoint.classes,
                threshold=(
                    rejection.cosine_threshold
                    if rejection is not None
                    else checkpoint.threshold
                ),
            )
            created_feedback = feedback
        elif feedback.model_fingerprint != checkpoint.fingerprint:
            raise ValueError("Feedback model fingerprint does not match selected classifier")

        factory = encoder_factory or DinoV3Encoder
        encoder = factory(
            checkpoint,
            search_roots=(manifest_path.parent,),
            device=device,
            use_fp16=use_fp16,
        )
        if rejection is None:
            classifier = DinoV3Classifier(
                checkpoint,
                encoder=encoder,
                feedback=feedback,
                registry=registry,
            )
        else:
            from .multi_dual_classifier import MultiDualDinoV3Classifier

            classifier = MultiDualDinoV3Classifier(
                checkpoint,
                rejection=rejection,
                encoder=encoder,
                feedback=feedback,
                registry=registry,
            )
        return DinoV3Runtime(
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
