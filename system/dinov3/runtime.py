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
from .registry import SpeciesRegistry, registry_path_for_fingerprint
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
    owns_registry: bool


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


def load_dinov3_model(
    model_path: str | Path,
    *,
    registry: SpeciesRegistry | None = None,
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

    owns_registry = registry is None
    if registry is None:
        root = Path(state_root).expanduser().resolve() if state_root is not None else default_dinov3_state_root()
        registry = SpeciesRegistry(
            registry_path_for_fingerprint(root, checkpoint.fingerprint),
            model_fingerprint=checkpoint.fingerprint,
        )
    elif registry.model_fingerprint != checkpoint.fingerprint:
        raise ValueError("Registry model fingerprint does not match selected classifier")

    factory = encoder_factory or DinoV3Encoder
    encoder = factory(
        checkpoint,
        search_roots=(manifest_path.parent,),
        device=device,
        use_fp16=use_fp16,
    )
    classifier = DinoV3Classifier(checkpoint, encoder=encoder, registry=registry)
    return DinoV3Runtime(
        manifest_path=manifest_path,
        checkpoint=checkpoint,
        classifier=classifier,
        registry=registry,
        owns_registry=owns_registry,
    )
