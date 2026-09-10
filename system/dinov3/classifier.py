"""DINOv3 event classification with Multi-prototype open-set rejection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .checkpoint import DINO_FEATURE_DIM, DINO_MULTI_PROTOTYPE_HEAD, DinoV3Checkpoint
from .preprocess import ImageInput
from .prototype_bank import PrototypeBank, PrototypeRecord


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Embedding rows must be finite and non-zero")
    return array / norms


def _cosine_score(vector: np.ndarray, prototype: np.ndarray) -> float:
    vector_norm = float(np.linalg.norm(vector))
    prototype_norm = float(np.linalg.norm(prototype))
    if (
        vector_norm <= 0
        or prototype_norm <= 0
        or not np.isfinite(vector_norm)
        or not np.isfinite(prototype_norm)
    ):
        raise ValueError("Centered feature and prototype must be finite and non-zero")
    return float(np.dot(vector, prototype) / (vector_norm * prototype_norm))


def aggregate_event_embeddings(crop_embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(crop_embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DINO_FEATURE_DIM or len(array) == 0:
        raise ValueError("Expected crop embeddings with shape (N, 768)")
    return _normalize_rows(_normalize_rows(array).mean(axis=0, keepdims=True))[0]


@dataclass(frozen=True)
class DinoV3Observation:
    result_index: int
    box_index: int
    embedding: np.ndarray
    accepted: bool
    species: str
    source: str
    registry_id: int | None
    registration_status: str | None
    known_score: float
    threshold: float
    detection_confidence: float
    observation_id: str = ""
    best_known_species: str = ""
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,):
            raise ValueError("Expected observation embedding with shape (768,)")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)
        bbox = tuple(float(value) for value in self.bbox)
        if len(bbox) != 4 or not np.isfinite(np.asarray(bbox, dtype=np.float32)).all():
            raise ValueError("Expected observation bbox with four finite coordinates")
        object.__setattr__(self, "bbox", bbox)


@dataclass(frozen=True)
class DinoV3Prediction:
    species: str
    accepted: bool
    best_known_species: str
    head_species: str
    prototype_species: str
    head_prototype_consistent: bool
    known_score: float
    threshold: float
    candidates: tuple[dict[str, Any], ...]
    embedding: np.ndarray
    source: str = "checkpoint"
    registry_id: int | None = None
    registration_status: str | None = None
    assistive_match: bool = False
    nearest_prototype_index: int | None = None
    squared_distance: float | None = None

    def as_candidate(self, *, detection_confidence=None) -> dict[str, Any]:
        return {
            "name": self.species,
            "conf": self.known_score,
            "raw_cls_conf": self.known_score,
            "raw_det_conf": detection_confidence,
            "known_score": self.known_score,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "head_species": self.head_species,
            "prototype_species": self.prototype_species,
            "head_prototype_consistent": self.head_prototype_consistent,
            "source": self.source,
            "registry_id": self.registry_id,
            "registration_status": self.registration_status,
            "assistive_match": self.assistive_match,
            "nearest_prototype_index": self.nearest_prototype_index,
            "squared_distance": self.squared_distance,
        }


class DinoV3Classifier:
    def __init__(
        self,
        checkpoint: DinoV3Checkpoint,
        *,
        encoder=None,
        feedback=None,
        registry=None,
    ) -> None:
        if checkpoint.head_type != DINO_MULTI_PROTOTYPE_HEAD:
            raise ValueError(
                "DINOv3 classifier requires head_type=multi_prototype; Linear Head checkpoints are not supported"
            )
        if checkpoint.feature_center is None or checkpoint.prototype_class_indices is None:
            raise ValueError("Multi-prototype checkpoint is missing classifier metadata")

        self.checkpoint = checkpoint
        self.encoder = encoder
        self.feedback = feedback
        self.registry = registry
        self.names = {index: name for index, name in enumerate(checkpoint.classes)}
        self.backend = "dinov3"
        self._feature_center = checkpoint.feature_center.numpy().astype(
            np.float32,
            copy=False,
        )
        self._prototypes = checkpoint.prototypes.numpy().astype(np.float32, copy=False)
        self._prototype_class_indices = checkpoint.prototype_class_indices.numpy().astype(
            np.int64,
            copy=False,
        )
        self._base_records = tuple(
            PrototypeRecord(
                species=checkpoint.classes[int(class_index)],
                embedding=self._prototypes[prototype_index],
                source="base",
            )
            for prototype_index, class_index in enumerate(self._prototype_class_indices)
        )

    @property
    def classes(self) -> tuple[str, ...]:
        return self.checkpoint.classes

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != DINO_FEATURE_DIM:
            raise ValueError("Expected event features with shape (N, 768)")
        if not np.isfinite(array).all() or not np.allclose(
            np.linalg.norm(array, axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError("Expected finite L2-normalized event features")
        return array

    def _center_features(self, array: np.ndarray) -> np.ndarray:
        """Transform validated encoder features into prototype-comparison space.

        Legacy Multi-prototype checkpoints historically used the centered vector
        directly at runtime. Subclasses may override this hook when their
        calibrated rejection geometry requires an additional normalization step.
        """
        return array - self._feature_center[None, :]

    def _provider_bank(self, provider, *, name: str) -> PrototypeBank:
        if provider is None:
            return PrototypeBank(())
        bank_provider = getattr(provider, "prototype_bank", None)
        if not callable(bank_provider):
            return PrototypeBank(())
        bank = bank_provider(self._feature_center)
        if not isinstance(bank, PrototypeBank):
            raise TypeError(f"{name}.prototype_bank() must return PrototypeBank")
        return bank

    def _effective_bank(self) -> PrototypeBank:
        feedback = self._provider_bank(self.feedback, name="feedback")
        registry = self._provider_bank(self.registry, name="registry")
        return PrototypeBank(
            formal=(
                self._base_records
                + tuple(feedback.formal)
                + tuple(registry.formal)
            ),
            provisional=(
                tuple(feedback.provisional)
                + tuple(registry.provisional)
            ),
        )

    @staticmethod
    def _record_matrix(records: Sequence[PrototypeRecord]) -> np.ndarray:
        if not records:
            return np.empty((0, DINO_FEATURE_DIM), dtype=np.float32)
        return np.stack([record.embedding for record in records]).astype(np.float32, copy=False)

    @staticmethod
    def _distances(centered: np.ndarray, records: Sequence[PrototypeRecord]) -> np.ndarray:
        matrix = DinoV3Classifier._record_matrix(records)
        deltas = matrix - centered[None, :]
        return np.einsum("md,md->m", deltas, deltas, optimize=True)

    @staticmethod
    def _winning_record(
        centered: np.ndarray,
        records: Sequence[PrototypeRecord],
    ) -> tuple[int, PrototypeRecord, float, float]:
        if not records:
            raise ValueError("Prototype bank must contain at least one record")
        distances = DinoV3Classifier._distances(centered, records)
        winner_index = int(np.argmin(distances))
        winner = records[winner_index]
        return (
            winner_index,
            winner,
            float(distances[winner_index]),
            _cosine_score(centered, winner.embedding),
        )

    @staticmethod
    def _species_candidates(
        centered: np.ndarray,
        records: Sequence[PrototypeRecord],
    ) -> tuple[dict[str, Any], ...]:
        if not records:
            return ()
        distances = DinoV3Classifier._distances(centered, records)
        best_by_species: dict[str, dict[str, Any]] = {}
        for prototype_index, record in enumerate(records):
            distance = float(distances[prototype_index])
            existing = best_by_species.get(record.species)
            if existing is not None and float(existing["squared_distance"]) <= distance:
                continue
            best_by_species[record.species] = {
                "name": record.species,
                "nearest_prototype_index": prototype_index,
                "squared_distance": distance,
                "cosine_score": _cosine_score(centered, record.embedding),
                "source": record.source,
                "registry_id": record.registry_id,
                "registration_status": record.registration_status,
            }
        candidates = sorted(
            best_by_species.values(),
            key=lambda item: (float(item["squared_distance"]), str(item["name"])),
        )
        return tuple(candidates[:3])

    @staticmethod
    def _projection_y_axis(
        axis_x: np.ndarray,
        origin: np.ndarray,
        prototypes: Sequence[np.ndarray],
    ) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for prototype in prototypes:
            candidate = prototype - origin
            candidate = candidate - float(np.dot(candidate, axis_x)) * axis_x
            if float(np.linalg.norm(candidate)) > 1e-8:
                residuals.append(candidate.astype(np.float32, copy=False))

        if residuals:
            matrix = np.stack(residuals).astype(np.float32, copy=False)
            _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
            axis_y = vh[0].astype(np.float32, copy=False)
            axis_y = axis_y - float(np.dot(axis_y, axis_x)) * axis_x
            axis_norm = float(np.linalg.norm(axis_y))
            if axis_norm > 1e-8:
                axis_y = axis_y / axis_norm
                for residual in residuals:
                    alignment = float(np.dot(residual, axis_y))
                    if abs(alignment) <= 1e-8:
                        continue
                    if alignment < 0:
                        axis_y = -axis_y
                    return axis_y

        basis_index = int(np.argmin(np.abs(axis_x)))
        basis = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
        basis[basis_index] = 1.0
        basis = basis - float(np.dot(basis, axis_x)) * axis_x
        basis_norm = float(np.linalg.norm(basis))
        if basis_norm <= 1e-8:
            raise ValueError("Unable to construct local projection axis")
        return basis / basis_norm

    def explain_feature(self, feature: np.ndarray) -> dict[str, Any]:
        """Return a deterministic local 2-D explanation around the nearest two species.

        Classification still uses the full 768-dimensional centered feature space.
        The x-axis joins the prototype centroids of the two nearest species; the
        y-axis is derived only from nearby prototype residuals. The current sample
        is projected into this fixed local frame and never defines either axis.
        """
        value = np.asarray(feature, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("Expected one feature vector")
        array = self._validate_features(value[None, :])
        centered = self._center_features(array)[0]
        bank = self._effective_bank()
        records = tuple(bank.formal) + tuple(bank.provisional)
        if not records:
            raise ValueError("Prototype bank cannot be empty")

        nearest_species = list(self._species_candidates(centered, records)[:2])
        selected_species = [str(item["name"]) for item in nearest_species]
        selected_indices = [
            index
            for index, record in enumerate(records)
            if record.species in selected_species
        ]
        if not selected_indices:
            raise ValueError("No nearby prototypes are available")

        species_prototypes = {
            species: [
                records[index].embedding
                for index in selected_indices
                if records[index].species == species
            ]
            for species in selected_species
        }
        first = np.stack(species_prototypes[selected_species[0]]).mean(axis=0)
        if len(nearest_species) > 1:
            second = np.stack(species_prototypes[selected_species[1]]).mean(axis=0)
            origin = (first + second) * 0.5
            axis_x = second - first
            axis_norm = float(np.linalg.norm(axis_x))
            if axis_norm <= 1e-8:
                axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
                axis_x[0] = 1.0
            else:
                axis_x = axis_x / axis_norm
        else:
            origin = first.copy()
            axis_x = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
            axis_x[0] = 1.0

        selected_prototypes = [records[index].embedding for index in selected_indices]
        axis_y = self._projection_y_axis(axis_x, origin, selected_prototypes)

        def project(vector: np.ndarray) -> tuple[float, float]:
            delta = vector - origin
            return float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y))

        points: list[dict[str, Any]] = []
        for index in selected_indices:
            record = records[index]
            x, y = project(record.embedding)
            points.append(
                {
                    "kind": "prototype",
                    "species": record.species,
                    "source": record.source,
                    "registry_id": record.registry_id,
                    "registration_status": record.registration_status,
                    "prototype_index": index,
                    "x": x,
                    "y": y,
                }
            )

        prediction = self._classify_multi_prototype(array)[0]
        current_x, current_y = project(centered)
        points.append(
            {
                "kind": "current",
                "species": prediction.species,
                "source": prediction.source,
                "registry_id": prediction.registry_id,
                "registration_status": prediction.registration_status,
                "prototype_index": prediction.nearest_prototype_index,
                "x": current_x,
                "y": current_y,
            }
        )
        return {
            "species": prediction.species,
            "accepted": prediction.accepted,
            "best_known_species": prediction.best_known_species,
            "known_score": prediction.known_score,
            "threshold": prediction.threshold,
            "nearest_prototype_index": prediction.nearest_prototype_index,
            "squared_distance": prediction.squared_distance,
            "nearest_species": nearest_species,
            "projection": {
                "method": "nearest_two_species_axis",
                "species": selected_species,
                "points": points,
            },
        }

    def _classify_multi_prototype(self, array: np.ndarray) -> list[DinoV3Prediction]:
        bank = self._effective_bank()
        if not bank.formal:
            raise ValueError("Formal Multi-prototype bank cannot be empty")
        centered_rows = self._center_features(array)

        results: list[DinoV3Prediction] = []
        for row_index, centered in enumerate(centered_rows):
            formal_index, formal, formal_distance, formal_score = self._winning_record(
                centered,
                bank.formal,
            )
            candidates = self._species_candidates(centered, bank.formal)
            if formal_score >= self.checkpoint.threshold:
                results.append(
                    DinoV3Prediction(
                        species=formal.species,
                        accepted=True,
                        best_known_species=formal.species,
                        head_species=formal.species,
                        prototype_species=formal.species,
                        head_prototype_consistent=True,
                        known_score=formal_score,
                        threshold=self.checkpoint.threshold,
                        candidates=candidates,
                        embedding=array[row_index].copy(),
                        source=formal.source,
                        registry_id=formal.registry_id,
                        registration_status=formal.registration_status,
                        nearest_prototype_index=formal_index,
                        squared_distance=formal_distance,
                    )
                )
                continue

            if bank.provisional:
                provisional_index, provisional, provisional_distance, provisional_score = (
                    self._winning_record(centered, bank.provisional)
                )
                if provisional_score >= self.checkpoint.threshold:
                    results.append(
                        DinoV3Prediction(
                            species=provisional.species,
                            accepted=False,
                            best_known_species=formal.species,
                            head_species=formal.species,
                            prototype_species=provisional.species,
                            head_prototype_consistent=False,
                            known_score=provisional_score,
                            threshold=self.checkpoint.threshold,
                            candidates=candidates,
                            embedding=array[row_index].copy(),
                            source=provisional.source,
                            registry_id=provisional.registry_id,
                            registration_status=provisional.registration_status,
                            assistive_match=True,
                            nearest_prototype_index=len(bank.formal) + provisional_index,
                            squared_distance=provisional_distance,
                        )
                    )
                    continue

            results.append(
                DinoV3Prediction(
                    species="Unknown",
                    accepted=False,
                    best_known_species=formal.species,
                    head_species=formal.species,
                    prototype_species=formal.species,
                    head_prototype_consistent=True,
                    known_score=formal_score,
                    threshold=self.checkpoint.threshold,
                    candidates=candidates,
                    embedding=array[row_index].copy(),
                    source=formal.source,
                    registry_id=formal.registry_id,
                    registration_status=formal.registration_status,
                    nearest_prototype_index=formal_index,
                    squared_distance=formal_distance,
                )
            )
        return results

    def classify_features(self, features: np.ndarray) -> list[DinoV3Prediction]:
        array = self._validate_features(features)
        return self._classify_multi_prototype(array)

    def classify_crops(
        self,
        crops: Sequence[ImageInput],
        *,
        array_color="rgb",
    ) -> list[DinoV3Prediction]:
        if self.encoder is None:
            raise RuntimeError("DINOv3 encoder is not loaded")
        return self.classify_features(self.encoder.encode(crops, array_color=array_color))

    def classify_event(
        self,
        crops: Sequence[ImageInput],
        *,
        array_color="rgb",
    ) -> DinoV3Prediction:
        if self.encoder is None:
            raise RuntimeError("DINOv3 encoder is not loaded")
        event = aggregate_event_embeddings(
            self.encoder.encode(crops, array_color=array_color)
        )
        return self.classify_features(event[None, :])[0]
