"""Backend helpers for DINOv2 checkpoint-scoped Registry state."""
from __future__ import annotations

import json

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.registry import SpeciesRegistry, registry_path_for_fingerprint
from system.dinov2.runtime import resolve_dinov2_manifest
from system.dinov2.state import default_dinov2_state_root


def load_checkpoint_for_model(classification_model_path: str):
    manifest = resolve_dinov2_manifest(classification_model_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint_name = payload.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise ValueError("DINOv2 manifest checkpoint is missing")
    return load_checkpoint(manifest.parent / checkpoint_name)


def open_registry_for_model(classification_model_path: str) -> SpeciesRegistry:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    return SpeciesRegistry(
        registry_path_for_fingerprint(
            default_dinov2_state_root(), checkpoint.fingerprint
        ),
        model_fingerprint=checkpoint.fingerprint,
        prototype_norm_power=checkpoint.prototype_norm_power,
    )
