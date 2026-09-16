"""DINOv2 Adaptive Multi-prototype classification with margin rejection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .checkpoint import DINO_FEATURE_DIM, DINO_MULTI_PROTOTYPE_HEAD, DinoV2Checkpoint
from .preprocess import ImageInput
from .prototype_bank import PrototypeBank, PrototypeRecord
from .rejection import accepts_margin_rejection, adjusted_distance_score, margin_route, rejection_metadata


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Embedding becomes zero after centering or contains non-finite values")
    return array / norms


def aggregate_event_embeddings(crop_embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(crop_embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DINO_FEATURE_DIM or len(array) == 0:
        raise ValueError("Expected crop embeddings with shape (N, 768)")
    if not np.isfinite(array).all():
        raise ValueError("Crop embeddings must be finite")
    return _normalize_rows(_normalize_rows(array).mean(axis=0, keepdims=True))[0]


@dataclass(frozen=True)
class DinoV2Observation:
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
    squared_distance: float | None = None
    class_margin: float | None = None
    adjusted_distance_score: float | None = None
    score_threshold: float | None = None
    registry_action: str = "candidate"

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,) or not np.isfinite(embedding).all():
            raise ValueError("Expected observation embedding with shape (768,)")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)
        bbox = tuple(float(value) for value in self.bbox)
        if len(bbox) != 4 or not np.isfinite(np.asarray(bbox, dtype=np.float32)).all():
            raise ValueError("Expected observation bbox with four finite coordinates")
        object.__setattr__(self, "bbox", bbox)


@dataclass(frozen=True)
class DinoV2Prediction:
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
    class_margin: float | None = None
    adjusted_distance_score: float | None = None
    score_threshold: float | None = None
    registry_action: str = "candidate"

    def as_candidate(self, *, detection_confidence=None) -> dict[str, Any]:
        return {
            "name": self.species,
            "conf": self.known_score,
            "raw_cls_conf": self.known_score,
            "raw_det_conf": detection_confidence,
            "known_score": self.known_score,
            "threshold": self.threshold,
            "score_threshold": self.score_threshold,
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
            "class_margin": self.class_margin,
            "adjusted_distance_score": self.adjusted_distance_score,
            "registry_action": self.registry_action,
        }


class DinoV2Classifier:
    def __init__(self, checkpoint: DinoV2Checkpoint, *, encoder=None, feedback=None, registry=None) -> None:
        if checkpoint.head_type != DINO_MULTI_PROTOTYPE_HEAD:
            raise ValueError("DINOv2 classifier requires head_type=multi_prototype")
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.feedback = feedback
        self.registry = registry
        self.names = {i: name for i, name in enumerate(checkpoint.classes)}
        self.backend = "dinov2"
        self._feature_center = checkpoint.feature_center.numpy().astype(np.float32, copy=False)
        self._prototypes = checkpoint.prototypes.numpy().astype(np.float32, copy=False)
        self._prototype_class_indices = checkpoint.prototype_class_indices.numpy().astype(np.int64, copy=False)
        self._base_records = tuple(
            PrototypeRecord(
                species=checkpoint.classes[int(class_index)],
                embedding=self._prototypes[index],
                source="base",
            )
            for index, class_index in enumerate(self._prototype_class_indices)
        )

    @property
    def classes(self) -> tuple[str, ...]:
        return self.checkpoint.classes

    @property
    def rejection_metadata(self) -> dict[str, object]:
        return rejection_metadata(self.checkpoint.rejection)

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != DINO_FEATURE_DIM:
            raise ValueError("Expected event features with shape (N, 768)")
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5):
            raise ValueError("Expected finite L2-normalized event features")
        return array

    def _center_features(self, array: np.ndarray) -> np.ndarray:
        return _normalize_rows(array - self._feature_center[None, :])

    def _provider_bank(self, provider, name: str) -> PrototypeBank:
        if provider is None or not callable(getattr(provider, "prototype_bank", None)):
            return PrototypeBank(())
        bank = provider.prototype_bank(self._feature_center)
        if not isinstance(bank, PrototypeBank):
            raise TypeError(f"{name}.prototype_bank() must return PrototypeBank")
        return bank

    def _effective_bank(self) -> PrototypeBank:
        feedback = self._provider_bank(self.feedback, "feedback")
        registry = self._provider_bank(self.registry, "registry")
        return PrototypeBank(
            formal=self._base_records + tuple(feedback.formal) + tuple(registry.formal),
            provisional=tuple(feedback.provisional) + tuple(registry.provisional),
        )

    @staticmethod
    def _matrix(records: Sequence[PrototypeRecord]) -> np.ndarray:
        if not records:
            return np.empty((0, DINO_FEATURE_DIM), dtype=np.float32)
        return np.stack([record.embedding for record in records]).astype(np.float32, copy=False)

    @classmethod
    def _geometry_rows(cls, vectors: np.ndarray, records: Sequence[PrototypeRecord]):
        matrix = cls._matrix(records)
        rows = np.asarray(vectors, dtype=np.float32)
        if not len(matrix):
            raise ValueError("Prototype bank must contain at least one record")
        if rows.ndim != 2 or rows.shape[1] != DINO_FEATURE_DIM:
            raise ValueError("Expected comparison features with shape (N, 768)")
        dot = rows @ matrix.T
        distances = np.maximum(
            np.square(rows).sum(axis=1, keepdims=True)
            + np.square(matrix).sum(axis=1)[None, :]
            - 2 * dot,
            0.0,
        ).astype(np.float64)
        row_norms = np.linalg.norm(rows, axis=1)
        prototype_norms = np.linalg.norm(matrix, axis=1)
        if np.any(row_norms <= 1e-12) or np.any(prototype_norms <= 1e-12):
            raise ValueError("Comparison vectors and prototypes must be non-zero")
        cosine = (dot / (row_norms[:, None] * prototype_norms[None, :])).astype(np.float64)
        return distances, cosine

    @classmethod
    def _geometry(cls, vector: np.ndarray, records: Sequence[PrototypeRecord]):
        distances, cosine = cls._geometry_rows(vector[None, :], records)
        return distances[0], cosine[0]

    def _class_scale(self, species: str) -> float:
        scales = self.checkpoint.rejection.class_distance_scales
        if scales is None or species not in self.checkpoint.classes:
            return 1.0
        return float(scales[self.checkpoint.classes.index(species)])

    def _signals(self, records, distances, cosine, selected_index: int) -> dict[str, Any]:
        selected = records[selected_index]
        selected_distance = float(distances[selected_index])
        best: dict[str, float] = {}
        for index, record in enumerate(records):
            distance = float(distances[index])
            if record.species not in best or distance < best[record.species]:
                best[record.species] = distance
        other = [distance for species, distance in best.items() if species != selected.species]
        if not other:
            raise ValueError("At least two prototype species are required for margin rejection")
        margin = float(min(other) - selected_distance)
        adjusted = adjusted_distance_score(
            winner_squared_distance=selected_distance,
            class_margin=margin,
            margin_weight=self.checkpoint.rejection.margin_weight,
            class_scale=self._class_scale(selected.species),
        )
        return {
            "record": selected,
            "index": selected_index,
            "squared_distance": selected_distance,
            "cosine": float(cosine[selected_index]),
            "class_margin": margin,
            "adjusted_distance_score": adjusted,
        }

    @classmethod
    def _candidates(cls, vector: np.ndarray, records: Sequence[PrototypeRecord]):
        if not records:
            return ()
        distances, cosine = cls._geometry(vector, records)
        best: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records):
            distance = float(distances[index])
            if record.species in best and float(best[record.species]["squared_distance"]) <= distance:
                continue
            best[record.species] = {
                "name": record.species,
                "nearest_prototype_index": index,
                "squared_distance": distance,
                "cosine_score": float(cosine[index]),
                "source": record.source,
                "registry_id": record.registry_id,
                "registration_status": record.registration_status,
            }
        return tuple(sorted(best.values(), key=lambda item: (float(item["squared_distance"]), str(item["name"])))[:3])

    def _prediction(self, raw, signals, candidates, *, accepted, species, best_known, assistive=False, nearest_index=None):
        record = signals["record"]
        cosine = signals["cosine"]
        adjusted = signals["adjusted_distance_score"]
        return DinoV2Prediction(
            species=species,
            accepted=accepted,
            best_known_species=best_known,
            head_species=best_known,
            prototype_species=record.species,
            head_prototype_consistent=record.species == best_known,
            known_score=cosine,
            threshold=self.checkpoint.rejection.cosine_threshold,
            score_threshold=self.checkpoint.rejection.adjusted_distance_score_threshold,
            candidates=candidates,
            embedding=raw.copy(),
            source=record.source,
            registry_id=record.registry_id,
            registration_status=record.registration_status,
            assistive_match=assistive,
            nearest_prototype_index=signals["index"] if nearest_index is None else nearest_index,
            squared_distance=signals["squared_distance"],
            class_margin=signals["class_margin"],
            adjusted_distance_score=adjusted,
            registry_action=margin_route(cosine, adjusted, self.checkpoint.rejection),
        )

    def _classify_multi_prototype(self, array: np.ndarray) -> list[DinoV2Prediction]:
        bank = self._effective_bank()
        if not bank.formal:
            raise ValueError("Formal Multi-prototype bank cannot be empty")
        centered_rows = self._center_features(array)
        distance_batch, cosine_batch = self._geometry_rows(centered_rows, bank.formal)
        results: list[DinoV2Prediction] = []
        for row_index, centered in enumerate(centered_rows):
            distances, cosine = distance_batch[row_index], cosine_batch[row_index]
            formal_index = int(np.argmin(distances))
            formal_signals = self._signals(bank.formal, distances, cosine, formal_index)
            formal = formal_signals["record"]
            candidates = self._candidates(centered, bank.formal)
            if accepts_margin_rejection(formal_signals["cosine"], formal_signals["adjusted_distance_score"], self.checkpoint.rejection):
                results.append(self._prediction(array[row_index], formal_signals, candidates, accepted=True, species=formal.species, best_known=formal.species))
                continue
            if bank.provisional:
                provisional_distances, _ = self._geometry(centered, bank.provisional)
                provisional_index = int(np.argmin(provisional_distances))
                combined = tuple(bank.formal) + tuple(bank.provisional)
                combined_distances, combined_cosine = self._geometry(centered, combined)
                combined_index = len(bank.formal) + provisional_index
                signals = self._signals(combined, combined_distances, combined_cosine, combined_index)
                provisional = signals["record"]
                if accepts_margin_rejection(signals["cosine"], signals["adjusted_distance_score"], self.checkpoint.rejection):
                    results.append(self._prediction(array[row_index], signals, candidates, accepted=False, species=provisional.species, best_known=formal.species, assistive=True, nearest_index=combined_index))
                    continue
            results.append(self._prediction(array[row_index], formal_signals, candidates, accepted=False, species="Unknown", best_known=formal.species))
        return results

    def classify_features(self, features: np.ndarray) -> list[DinoV2Prediction]:
        return self._classify_multi_prototype(self._validate_features(features))

    def classify_crops(self, crops: Sequence[ImageInput], *, array_color: str = "rgb") -> list[DinoV2Prediction]:
        if self.encoder is None:
            raise RuntimeError("DINOv2 encoder is not loaded")
        return self.classify_features(self.encoder.encode(crops, array_color=array_color))

    def classify_event(self, crops: Sequence[ImageInput], *, array_color: str = "rgb") -> DinoV2Prediction:
        if self.encoder is None:
            raise RuntimeError("DINOv2 encoder is not loaded")
        event = aggregate_event_embeddings(self.encoder.encode(crops, array_color=array_color))
        return self.classify_features(event[None, :])[0]

    def explain_feature(self, feature: np.ndarray) -> dict[str, Any]:
        array = self._validate_features(np.asarray(feature, dtype=np.float32)[None, :])
        centered = self._center_features(array)[0]
        bank = self._effective_bank()
        records = tuple(bank.formal) + tuple(bank.provisional)
        nearest = list(self._candidates(centered, records)[:2])
        names = [str(item["name"]) for item in nearest]
        indices = [i for i, record in enumerate(records) if record.species in names]
        if not indices:
            raise ValueError("No nearby prototypes are available")
        first = np.stack([records[i].embedding for i in indices if records[i].species == names[0]]).mean(axis=0)
        if len(names) > 1:
            second = np.stack([records[i].embedding for i in indices if records[i].species == names[1]]).mean(axis=0)
            origin = (first + second) * 0.5
            axis_x = second - first
            norm = float(np.linalg.norm(axis_x))
            axis_x = axis_x / norm if norm > 1e-8 else np.eye(1, DINO_FEATURE_DIM, 0, dtype=np.float32)[0]
        else:
            origin = first.copy()
            axis_x = np.eye(1, DINO_FEATURE_DIM, 0, dtype=np.float32)[0]
        basis = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
        basis[int(np.argmin(np.abs(axis_x)))] = 1.0
        axis_y = basis - float(np.dot(basis, axis_x)) * axis_x
        axis_y /= np.linalg.norm(axis_y)
        def project(vector):
            delta = vector - origin
            return float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y))
        points = []
        for index in indices:
            x, y = project(records[index].embedding)
            points.append({"kind": "prototype", "species": records[index].species, "prototype_index": index, "x": x, "y": y})
        prediction = self.classify_features(array)[0]
        x, y = project(centered)
        points.append({"kind": "current", "species": prediction.species, "prototype_index": prediction.nearest_prototype_index, "x": x, "y": y})
        return {
            "species": prediction.species,
            "accepted": prediction.accepted,
            "best_known_species": prediction.best_known_species,
            "known_score": prediction.known_score,
            "threshold": prediction.threshold,
            "score_threshold": prediction.score_threshold,
            "nearest_prototype_index": prediction.nearest_prototype_index,
            "squared_distance": prediction.squared_distance,
            "class_margin": prediction.class_margin,
            "adjusted_distance_score": prediction.adjusted_distance_score,
            "rejection": self.rejection_metadata,
            "nearest_species": nearest,
            "projection": {"method": "nearest_two_species_axis", "species": names, "points": points},
        }
