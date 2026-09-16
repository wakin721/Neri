"""DINOv2 human-feedback store with CL2N prototype generation."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import feedback_base as _base
from .checkpoint import DinoV2Rejection
from .simple_shot import deterministic_k_means

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def feedback_path_for_registry(registry_path: str | Path) -> Path:
    return Path(registry_path).expanduser().resolve().with_name("feedback.sqlite3")


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


def _cl2n_rows(values: np.ndarray, feature_center: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    center = np.asarray(feature_center, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 768:
        raise ValueError("Expected embeddings with shape (N, 768)")
    if center.shape != (768,) or not np.isfinite(center).all():
        raise ValueError("Expected finite feature_center with shape (768,)")
    centered = array - center[None, :]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Feedback evidence becomes zero after DINOv2 centering")
    return centered / norms


class HumanFeedbackStore(_base.HumanFeedbackStore):
    def __init__(
        self,
        path: str | Path,
        *,
        model_fingerprint: str,
        checkpoint_classes,
        rejection: DinoV2Rejection,
        prototype_norm_power: float = 0.0,
    ) -> None:
        if not isinstance(rejection, DinoV2Rejection):
            raise TypeError("rejection must be DinoV2Rejection")
        self.rejection = rejection
        self.prototype_norm_power = float(prototype_norm_power)
        if not np.isfinite(self.prototype_norm_power) or self.prototype_norm_power < 0:
            raise ValueError("prototype_norm_power must be finite and nonnegative")
        super().__init__(
            path,
            model_fingerprint=model_fingerprint,
            checkpoint_classes=checkpoint_classes,
            threshold=rejection.cosine_threshold,
        )

    def _build_candidate(
        self,
        positive_events,
        negative_events,
        feature_center: np.ndarray,
        *,
        status: str,
    ):
        positive_raw = np.stack([event.embedding for event in positive_events]).astype(
            np.float32, copy=False
        )
        positive_features = _cl2n_rows(positive_raw, feature_center)
        prototypes = deterministic_k_means(
            positive_features,
            max_k=self._prototype_limit(status),
        ).astype(np.float32)
        prototypes = _apply_prototype_norm_power(
            prototypes,
            getattr(self, "prototype_norm_power", 0.0),
        )
        positive_scores = self._scores(positive_features, prototypes)
        positive_coverage = float(np.mean(positive_scores >= self.threshold))
        if negative_events:
            negative_raw = np.stack([event.embedding for event in negative_events]).astype(
                np.float32, copy=False
            )
            negative_features = _cl2n_rows(negative_raw, feature_center)
            negative_scores = self._scores(negative_features, prototypes)
            false_accept_rate = float(np.mean(negative_scores >= self.threshold))
        else:
            false_accept_rate = 0.0
        quality_passed = (
            positive_coverage >= _base._POSITIVE_COVERAGE_MIN
            and (
                not negative_events
                or false_accept_rate <= _base._HARD_NEGATIVE_FALSE_ACCEPT_MAX
            )
        )
        return prototypes, positive_coverage, false_accept_rate, quality_passed

    def cluster_details(
        self,
        species: str,
        feature_center: np.ndarray,
    ) -> list[dict[str, object]]:
        positive_events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        if not positive_events:
            return []

        center = feature_center.numpy() if hasattr(feature_center, "numpy") else feature_center
        event_raw = np.stack([event.embedding for event in positive_events]).astype(
            np.float32, copy=False
        )
        event_vectors = _cl2n_rows(event_raw, np.asarray(center, dtype=np.float32))
        state = self.learning_state(species)
        active_generation = self._active_generation(species)
        active = active_generation is not None
        if active_generation is not None:
            prototype_values = self._generation_prototypes(int(active_generation["id"]))
            prototypes = np.stack(prototype_values).astype(np.float32, copy=False)
        else:
            prototypes = deterministic_k_means(event_vectors, max_k=1).astype(
                np.float32, copy=False
            )
            prototypes = _apply_prototype_norm_power(
                prototypes,
                getattr(self, "prototype_norm_power", 0.0),
            )

        deltas = event_vectors[:, None, :] - prototypes[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)
        labels = np.argmin(distances, axis=1)
        result: list[dict[str, object]] = []
        for prototype_index in range(len(prototypes)):
            member_indices = np.flatnonzero(labels == prototype_index).tolist()
            if not member_indices:
                continue
            ordered = sorted(
                member_indices,
                key=lambda index: (float(distances[index, prototype_index]), index),
            )
            refs: list[dict[str, object]] = []
            seen: set[str] = set()
            for event_index in ordered:
                for observation_id in positive_events[event_index].observation_ids:
                    if observation_id in seen:
                        continue
                    seen.add(observation_id)
                    refs.append(
                        {"kind": "observation", "observation_id": observation_id}
                    )
                    if len(refs) >= 3:
                        break
                if len(refs) >= 3:
                    break
            member_distances = distances[member_indices, prototype_index]
            result.append(
                {
                    "id": f"feedback:{species}:{prototype_index}",
                    "label": (
                        f"Feedback Cluster #{prototype_index + 1}"
                        if active
                        else "反馈证据（尚未形成 prototype）"
                    ),
                    "source": "feedback" if active else "feedback_evidence",
                    "prototype_index": prototype_index,
                    "event_count": len(member_indices),
                    "camera_count": len(
                        {positive_events[index].camera_id for index in member_indices}
                    ),
                    "sample_count": sum(
                        len(positive_events[index].observation_ids)
                        for index in member_indices
                    ),
                    "mean_squared_distance": float(np.mean(member_distances)),
                    "active": active,
                    "learning_status": state.status,
                    "example_refs": refs,
                }
            )
        return result
