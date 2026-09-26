"""DINOv2 human-feedback store with CL2N prototype generation."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import feedback_base as _base
from .checkpoint import DinoV2Rejection
from .memory_bank import MemoryBank, MemoryExample
from .memory_checkpoint import MemoryCheckpoint
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
        memory_checkpoint: MemoryCheckpoint | None = None,
    ) -> None:
        if not isinstance(rejection, DinoV2Rejection):
            raise TypeError("rejection must be DinoV2Rejection")
        self.rejection = rejection
        self.memory_checkpoint = memory_checkpoint
        self.prototype_norm_power = float(prototype_norm_power)
        if not np.isfinite(self.prototype_norm_power) or self.prototype_norm_power < 0:
            raise ValueError("prototype_norm_power must be finite and nonnegative")
        super().__init__(
            path,
            model_fingerprint=model_fingerprint,
            checkpoint_classes=checkpoint_classes,
            threshold=rejection.cosine_threshold,
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback_generation_memory(
                generation_id INTEGER NOT NULL,
                event_index INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY(generation_id,event_index),
                FOREIGN KEY(generation_id) REFERENCES feedback_generations(id) ON DELETE CASCADE
            )"""
        )
        self._conn.commit()

    def _build_candidate(
        self,
        positive_events,
        negative_events,
        feature_center: np.ndarray,
        *,
        status: str,
        species: str | None = None,
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
        if getattr(self, "memory_checkpoint", None) is not None:
            if species is None:
                raise ValueError("Memory feedback candidate requires species")
            positive_coverage = self._memory_accept_rate(
                positive_events, species, positive_events
            )
        else:
            positive_scores = self._scores(positive_features, prototypes)
            positive_coverage = float(np.mean(positive_scores >= self.threshold))
        if negative_events:
            negative_raw = np.stack([event.embedding for event in negative_events]).astype(
                np.float32, copy=False
            )
            if getattr(self, "memory_checkpoint", None) is not None:
                false_accept_rate = self._memory_accept_rate(
                    negative_events, species, positive_events
                )
            else:
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

    def _memory_accept_rate(self, queries, species: str, positive_events) -> float:
        from .memory_classifier import MemoryDinoV2Classifier, _normalize_rows

        classifier = MemoryDinoV2Classifier(self.memory_checkpoint)
        bank = tuple(
            MemoryExample(species, event.embedding, event.camera_id, "feedback")
            for event in positive_events
        )
        raw = np.stack([event.embedding for event in queries]).astype(np.float32)
        centered = _normalize_rows(raw - classifier._center[None, :])
        scores, _, classes = classifier._scores(centered, bank)
        order = np.argsort(-scores, axis=1, kind="stable")
        winner = order[:, 0]
        runner = order[:, 1]
        knownness = scores[np.arange(len(scores)), winner] + (
            self.memory_checkpoint.margin_weight
            * (scores[np.arange(len(scores)), winner] - scores[np.arange(len(scores)), runner])
        )
        return float(np.mean((np.asarray(classes)[winner] == species) & (
            knownness >= self.memory_checkpoint.threshold
        )))

    def _activate_generation(self, species: str, **kwargs):
        generation = super()._activate_generation(species, **kwargs)
        events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        generation_id = int(generation["id"])
        self._conn.execute(
            "DELETE FROM feedback_generation_memory WHERE generation_id=?", (generation_id,)
        )
        self._conn.executemany(
            """INSERT INTO feedback_generation_memory(
                generation_id,event_index,camera_id,embedding
            ) VALUES(?,?,?,?)""",
            [
                (generation_id, index, event.camera_id,
                 np.asarray(event.embedding, dtype="<f4").tobytes())
                for index, event in enumerate(events)
            ],
        )
        return generation

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

    def memory_bank(self) -> MemoryBank:
        """Use only evidence captured when each generation was activated."""
        rows = self._conn.execute(
            """
            SELECT id,species,status,formal FROM feedback_generations
            WHERE active=1 ORDER BY species,id
            """
        ).fetchall()
        formal: list[MemoryExample] = []
        provisional: list[MemoryExample] = []
        for row in rows:
            species = str(row["species"])
            events = self._conn.execute(
                """SELECT camera_id,embedding FROM feedback_generation_memory
                   WHERE generation_id=? ORDER BY event_index""",
                (int(row["id"]),),
            ).fetchall()
            destination = formal if bool(row["formal"]) else provisional
            for event in events:
                destination.append(
                    MemoryExample(
                        species=species,
                        embedding=np.frombuffer(event["embedding"], dtype="<f4").copy(),
                        camera_id=str(event["camera_id"]),
                        source="feedback",
                        registration_status=str(row["status"]),
                    )
                )
        return MemoryBank(tuple(formal), tuple(provisional))
