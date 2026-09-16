"""DINOv2 fingerprint-scoped Registry with CL2N overlay prototypes."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import registry_base as _base
from .prototype_bank import PrototypeBank, PrototypeRecord
from .simple_shot import deterministic_k_means

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def registry_path_for_fingerprint(root: str | Path, fingerprint: str) -> Path:
    safe = str(fingerprint).lower()
    if len(safe) != 64 or any(c not in "0123456789abcdef" for c in safe):
        raise ValueError("model fingerprint must be a SHA-256 hex digest")
    return (Path(root).expanduser().resolve() / "dinov2" / safe / "registry.sqlite3").resolve()


def _cl2n_rows(values: np.ndarray, feature_center: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(feature_center, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 768:
        raise ValueError("Expected embeddings with shape (N, 768)")
    if center.shape != (768,) or not np.isfinite(center).all():
        raise ValueError("feature_center must be a finite 768-dimensional vector")
    centered = array - center[None, :]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Registry evidence becomes zero after DINOv2 centering")
    return centered / norms


def _apply_prototype_norm_power(prototypes: np.ndarray, power: float) -> np.ndarray:
    value = float(power)
    if not np.isfinite(value) or value < 0:
        raise ValueError("prototype_norm_power must be finite and nonnegative")
    array = np.asarray(prototypes, dtype=np.float32)
    if value == 0.0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Learned prototype norm must be finite and non-zero")
    return (array / np.power(norms, value)).astype(np.float32, copy=False)


class SpeciesRegistry(_base.SpeciesRegistry):
    def __init__(self, *args, prototype_norm_power: float = 0.0, **kwargs):
        self.prototype_norm_power = float(prototype_norm_power)
        if not np.isfinite(self.prototype_norm_power) or self.prototype_norm_power < 0:
            raise ValueError("prototype_norm_power must be finite and nonnegative")
        super().__init__(*args, **kwargs)

    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:
        formal: list[PrototypeRecord] = []
        provisional: list[PrototypeRecord] = []
        for entry in self.list():
            if entry.status == "candidate":
                continue
            embeddings = self._embeddings(entry.id)
            if len(embeddings) < 4:
                continue
            transformed = _cl2n_rows(embeddings, feature_center)
            prototypes = deterministic_k_means(
                transformed,
                max_k=self._status_prototype_limit(entry.status),
            )
            prototypes = _apply_prototype_norm_power(
                prototypes,
                getattr(self, "prototype_norm_power", 0.0),
            )
            records = [
                PrototypeRecord(
                    species=entry.display_name,
                    embedding=prototype,
                    source="overlay",
                    registry_id=entry.id,
                    registration_status=entry.status,
                )
                for prototype in prototypes
            ]
            if entry.status == "provisional":
                provisional.extend(records)
            elif entry.status in {"confirmed", "mature"}:
                formal.extend(records)
        return PrototypeBank(tuple(formal), tuple(provisional))
