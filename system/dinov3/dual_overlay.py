"""CL2N-compatible Registry and human-feedback overlays for Multi-dual."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .calibration import apply_multi_dual_gate, cl2n_transform, prototype_signals
from .feedback import (
    DINO_DIM,
    HumanFeedbackStore,
    SpeciesLearningState,
    _HARD_NEGATIVE_FALSE_ACCEPT_MAX,
    _POSITIVE_COVERAGE_MIN,
)
from .prototype_bank import PrototypeBank, PrototypeRecord
from .registry import SpeciesRegistry
from .rejection import MultiDualRejectionConfig
from .simple_shot import deterministic_k_means


def multi_dual_feedback_path_for_registry(
    registry_path: str | Path,
    rejection: MultiDualRejectionConfig | None = None,
) -> Path:
    """Keep Dual generations isolated from legacy and differently calibrated runs."""
    registry = Path(registry_path).expanduser().resolve()
    if rejection is None:
        return registry.with_name("feedback_multi_dual.sqlite3")
    material = (
        f"multi_dual|{rejection.cosine_threshold:.17g}|"
        f"{rejection.squared_distance_threshold:.17g}"
    ).encode("ascii")
    suffix = hashlib.sha256(material).hexdigest()[:16]
    return registry.with_name(f"feedback_multi_dual_{suffix}.sqlite3")


class MultiDualSpeciesRegistry(SpeciesRegistry):
    """Registry whose classifier overlay bank is rebuilt in full CL2N space.

    Registry matching/storage remains unchanged in raw normalized embedding space;
    only the dynamic prototypes exposed to the Multi-dual classifier are rebuilt
    after subtracting the checkpoint center and L2-normalizing each event vector.
    """

    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:
        center = np.asarray(feature_center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("feature_center must be a finite 768-dimensional vector")

        formal: list[PrototypeRecord] = []
        provisional: list[PrototypeRecord] = []
        for entry in self.list():
            if entry.status == "candidate":
                continue
            embeddings = self._embeddings(entry.id)
            if len(embeddings) < 4:
                continue
            transformed = cl2n_transform(embeddings, center)
            prototypes = deterministic_k_means(
                transformed,
                max_k=self._status_prototype_limit(entry.status),
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


class MultiDualHumanFeedbackStore(HumanFeedbackStore):
    """Human-feedback generations fitted and validated in Multi-dual CL2N space."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_fingerprint: str,
        checkpoint_classes,
        rejection: MultiDualRejectionConfig,
    ) -> None:
        if not isinstance(rejection, MultiDualRejectionConfig):
            raise TypeError("rejection must be MultiDualRejectionConfig")
        self.rejection = rejection
        super().__init__(
            path,
            model_fingerprint=model_fingerprint,
            checkpoint_classes=checkpoint_classes,
            threshold=rejection.cosine_threshold,
        )

    def _accepted_by_dual(self, features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.empty((0,), dtype=bool)
        signals = prototype_signals(features, prototypes)
        return apply_multi_dual_gate(
            signals["winner_cosine"],
            signals["winner_squared_distance"],
            cosine_threshold=self.rejection.cosine_threshold,
            squared_distance_threshold=self.rejection.squared_distance_threshold,
        )

    def _build_candidate(
        self,
        positive_events,
        negative_events,
        feature_center: np.ndarray,
        *,
        status: str,
    ) -> tuple[np.ndarray, float, float, bool]:
        center = np.asarray(feature_center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("Expected finite feature_center with shape (768,)")
        positive_raw = np.stack([event.embedding for event in positive_events]).astype(
            np.float32,
            copy=False,
        )
        positive_features = cl2n_transform(positive_raw, center)
        prototypes = deterministic_k_means(
            positive_features,
            max_k=self._prototype_limit(status),
        ).astype(np.float32)
        positive_accept = self._accepted_by_dual(positive_features, prototypes)
        positive_coverage = float(np.mean(positive_accept))

        if negative_events:
            negative_raw = np.stack(
                [event.embedding for event in negative_events]
            ).astype(np.float32, copy=False)
            negative_features = cl2n_transform(negative_raw, center)
            negative_accept = self._accepted_by_dual(negative_features, prototypes)
            false_accept_rate = float(np.mean(negative_accept))
        else:
            false_accept_rate = 0.0

        quality_passed = (
            positive_coverage >= _POSITIVE_COVERAGE_MIN
            and (
                not negative_events
                or false_accept_rate <= _HARD_NEGATIVE_FALSE_ACCEPT_MAX
            )
        )
        return prototypes, positive_coverage, false_accept_rate, quality_passed

    def cluster_details(
        self,
        species: str,
        feature_center: np.ndarray,
    ) -> list[dict[str, object]]:
        """Describe feedback clusters in the same CL2N geometry used by Dual."""
        positive_events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        if not positive_events:
            return []

        center = feature_center
        if hasattr(center, "numpy"):
            center = center.numpy()
        center = np.asarray(center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("Expected finite feature_center with shape (768,)")

        state: SpeciesLearningState = self.learning_state(species)
        active_generation = self._active_generation(species)
        active = active_generation is not None
        if active_generation is not None:
            prototype_values = self._generation_prototypes(int(active_generation["id"]))
            prototypes = np.stack(prototype_values).astype(np.float32, copy=False)
        else:
            event_raw = np.stack([event.embedding for event in positive_events]).astype(
                np.float32,
                copy=False,
            )
            event_vectors = cl2n_transform(event_raw, center)
            prototypes = deterministic_k_means(event_vectors, max_k=1).astype(
                np.float32,
                copy=False,
            )

        event_raw = np.stack([event.embedding for event in positive_events]).astype(
            np.float32,
            copy=False,
        )
        event_vectors = cl2n_transform(event_raw, center)
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
                        {
                            "kind": "observation",
                            "observation_id": observation_id,
                        }
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
