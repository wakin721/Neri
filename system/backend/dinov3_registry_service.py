"""Backend helpers for opening the registry associated with a DINOv3 model."""
from __future__ import annotations

import json
from pathlib import Path

from system.dinov3.checkpoint import load_checkpoint
from system.dinov3.registry import SpeciesRegistry, registry_path_for_fingerprint
from system.dinov3.runtime import DinoV3ManifestError, resolve_dinov3_manifest
from system.dinov3.state import default_dinov3_state_root


def open_registry_for_model(classification_model_path: str) -> SpeciesRegistry:
    manifest_path = resolve_dinov3_manifest(classification_model_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV3ManifestError(f"Invalid DINOv3 manifest: {manifest_path}") from exc
    checkpoint_name = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise DinoV3ManifestError("DINOv3 manifest checkpoint is missing")
    checkpoint = load_checkpoint(manifest_path.parent / checkpoint_name)
    path = registry_path_for_fingerprint(default_dinov3_state_root(), checkpoint.fingerprint)
    return SpeciesRegistry(path, model_fingerprint=checkpoint.fingerprint)
