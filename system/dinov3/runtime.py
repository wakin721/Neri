"""Manifest-driven native DINOv3 runtime construction."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .checkpoint import DINO_BACKBONE, DINO_EVENT_AGGREGATION, DINO_FEATURE_DIM, DINO_PREPROCESSING, load_checkpoint
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
    checkpoint: Any
    classifier: "DinoV3Classifier"
    registry: SpeciesRegistry
    feedback: HumanFeedbackStore
    owns_registry: bool
    owns_feedback: bool

    def close(self):
        if self.owns_feedback:
            self.feedback.close()
        if self.owns_registry:
            self.registry.close()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("backend") != "dinov3":
        raise DinoV3ManifestError("Selected classification model is not a DINOv3 manifest")
    return payload


def resolve_dinov3_manifest(model_path):
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name.endswith(".neri.json"):
        return path
    raise DinoV3ManifestError(f"No DINOv3 manifest found for classification model: {path}")


def _validate_manifest_contract(payload, checkpoint):
    if payload.get("architecture") not in (None, "", DINO_BACKBONE):
        raise DinoV3ManifestError("Unsupported architecture")
    if payload.get("feature_dim") not in (None, DINO_FEATURE_DIM):
        raise DinoV3ManifestError("Unsupported feature dimension")


def load_dinov3_model(model_path, *, registry=None, feedback=None, state_root=None, encoder_factory=None, device=None, use_fp16=False):
    from .classifier import DinoV3Classifier
    from .encoder import DinoV3Encoder

    from .multi_dual_classifier import MultiDualDinoV3Classifier

    manifest_path = resolve_dinov3_manifest(model_path)
    payload = _load_manifest(manifest_path)
    checkpoint = load_checkpoint(manifest_path.parent / str(payload["checkpoint"]))
    _validate_manifest_contract(payload, checkpoint)
    rejection = parse_multi_dual_rejection(payload.get("rejection"))

    owns_registry = registry is None
    owns_feedback = feedback is None
    if registry is None:
        root = Path(state_root).expanduser().resolve() if state_root else default_dinov3_state_root()
        registry = SpeciesRegistry(registry_path_for_fingerprint(root, checkpoint.fingerprint), model_fingerprint=checkpoint.fingerprint)
    if feedback is None:
        feedback = HumanFeedbackStore(feedback_path_for_registry(registry.path), model_fingerprint=checkpoint.fingerprint, checkpoint_classes=checkpoint.classes, threshold=(rejection.cosine_threshold if rejection else checkpoint.threshold))

    encoder = (encoder_factory or DinoV3Encoder)(checkpoint, search_roots=(manifest_path.parent,), device=device, use_fp16=use_fp16)
    if rejection is None:
        classifier = DinoV3Classifier(checkpoint, encoder=encoder, feedback=feedback, registry=registry)
    else:
        classifier = MultiDualDinoV3Classifier(checkpoint, rejection=rejection, encoder=encoder, feedback=feedback, registry=registry)
    return DinoV3Runtime(manifest_path, checkpoint, classifier, registry, feedback, owns_registry, owns_feedback)
