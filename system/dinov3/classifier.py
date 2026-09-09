"""DINOv3 event classification with Multi-prototype open-set rejection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .checkpoint import DINO_FEATURE_DIM, DINO_MULTI_PROTOTYPE_HEAD, DinoV3Checkpoint
from .preprocess import ImageInput


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
class PrototypeRecord:
    species: str
    embedding: np.ndarray
    source: str
    registry_id: int | None = None
    registration_status: str | None = None

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,) or not np.isfinite(embedding).all():
            raise ValueError("Expected a finite 768-dimensional prototype")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class PrototypeBank:
    formal: tuple[PrototypeRecord, ...]
    provisional: tuple[PrototypeRecord, ...] = ()


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

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,):
            raise ValueError("Expected observation embedding with shape (768,)")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


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
            # Kept as response compatibility metadata while the Linear Head
            # implementation is removed. They do not participate in decisions.
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
        registry=None,
    ) -> None:
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.registry = registry
        self.names = {index: name for index, name in enumerate(checkpoint.classes)}
        self.backend = "dinov3"

        if checkpoint.head_type == DINO_MULTI_PROTOTYPE_HEAD:
            if checkpoint.feature_center is None or checkpoint.prototype_class_indices is None:
                raise ValueError("Multi-prototype checkpoint is missing classifier metadata")
            self._feature_center = (
                checkpoint.feature_center.numpy().astype(np.float32, copy=False)
            )
            self._prototypes = checkpoint.prototypes.numpy().astype(
                np.float32,
                copy=False,
            )
            self._prototype_class_indices = (
                checkpoint.prototype_class_indices.numpy().astype(np.int64, copy=False)
            )
            self._weight = None
            self._bias = None
        else:
            # Temporary migration compatibility. Production DINOv3 components
            # switch to Multi-prototype; Task 7 removes the obsolete head path.
            if checkpoint.head_weight is None or checkpoint.head_bias is None:
                raise ValueError("Legacy DINOv3 checkpoint is missing Linear Head state")
            self._weight = checkpoint.head_weight.numpy().astype(
                np.float32,
                copy=False,
            )
            self._bias = checkpoint.head_bias.numpy().astype(np.float32, copy=False)
            self._prototypes = _normalize_rows(
                checkpoint.prototypes.numpy().astype(np.float32, copy=False)
            )
            self._feature_center = np.zeros(DINO_FEATURE_DIM, dtype=np.float32)
            self._prototype_class_indices = np.arange(
                len(checkpoint.classes),
                dtype=np.int64,
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

    def _species_candidates(
        self,
        centered: np.ndarray,
        distances: np.ndarray,
    ) -> tuple[dict[str, Any], ...]:
        candidates: list[dict[str, Any]] = []
        for class_index, species in enumerate(self.checkpoint.classes):
            prototype_indices = np.flatnonzero(
                self._prototype_class_indices == class_index
            )
            if len(prototype_indices) == 0:
                continue
            local = int(np.argmin(distances[prototype_indices]))
            prototype_index = int(prototype_indices[local])
            prototype = self._prototypes[prototype_index]
            candidates.append(
                {
                    "name": species,
                    "nearest_prototype_index": prototype_index,
                    "squared_distance": float(distances[prototype_index]),
                    "cosine_score": _cosine_score(centered, prototype),
                    "source": "base",
                    "registration_status": None,
                }
            )
        candidates.sort(key=lambda item: (item["squared_distance"], item["name"]))
        return tuple(candidates[:3])

    def _classify_multi_prototype(
        self,
        array: np.ndarray,
    ) -> list[DinoV3Prediction]:
        centered_rows = array - self._feature_center[None, :]
        deltas = centered_rows[:, None, :] - self._prototypes[None, :, :]
        distances = np.einsum("nmd,nmd->nm", deltas, deltas, optimize=True)
        winning_indices = np.argmin(distances, axis=1)

        results: list[DinoV3Prediction] = []
        for row_index, raw_winner in enumerate(winning_indices):
            winner = int(raw_winner)
            class_index = int(self._prototype_class_indices[winner])
            known_species = self.checkpoint.classes[class_index]
            centered = centered_rows[row_index]
            prototype = self._prototypes[winner]
            score = _cosine_score(centered, prototype)
            accepted = score >= self.checkpoint.threshold
            candidates = self._species_candidates(
                centered,
                distances[row_index],
            )
            results.append(
                DinoV3Prediction(
                    species=known_species if accepted else "Unknown",
                    accepted=accepted,
                    best_known_species=known_species,
                    head_species=known_species,
                    prototype_species=known_species,
                    head_prototype_consistent=True,
                    known_score=score,
                    threshold=self.checkpoint.threshold,
                    candidates=candidates,
                    embedding=array[row_index].copy(),
                    source="base",
                    nearest_prototype_index=winner,
                    squared_distance=float(distances[row_index, winner]),
                )
            )
        return results

    def _classify_legacy(self, array: np.ndarray) -> list[DinoV3Prediction]:
        assert self._weight is not None and self._bias is not None
        logits = array @ self._weight.T + self._bias
        sims = array @ self._prototypes.T
        heads = logits.argmax(axis=1)
        nearest = sims.argmax(axis=1)
        results: list[DinoV3Prediction] = []
        for row, (head_index, prototype_index) in enumerate(zip(heads, nearest)):
            hi = int(head_index)
            pi = int(prototype_index)
            score = float(sims[row, pi])
            consistent = hi == pi
            accepted = score >= self.checkpoint.threshold and consistent
            head = self.checkpoint.classes[hi]
            proto = self.checkpoint.classes[pi]
            order = np.argsort(logits[row])[::-1][:3]
            candidates = tuple(
                {
                    "name": self.checkpoint.classes[int(index)],
                    "logit": float(logits[row, int(index)]),
                    "prototype_score": float(sims[row, int(index)]),
                }
                for index in order
            )
            prediction = DinoV3Prediction(
                head if accepted else "Unknown",
                accepted,
                proto,
                head,
                proto,
                consistent,
                score,
                self.checkpoint.threshold,
                candidates,
                array[row].copy(),
            )
            if not accepted and self.registry is not None:
                matched = self.registry.match(array[row])
                if matched is not None:
                    prediction = DinoV3Prediction(
                        matched["display_name"],
                        bool(matched["accepted"]),
                        proto,
                        head,
                        proto,
                        consistent,
                        float(matched["score"]),
                        float(matched["threshold"]),
                        candidates,
                        array[row].copy(),
                        "registry",
                        int(matched["id"]),
                        str(matched["status"]),
                    )
            results.append(prediction)
        return results

    def classify_features(self, features: np.ndarray) -> list[DinoV3Prediction]:
        array = self._validate_features(features)
        if self.checkpoint.head_type == DINO_MULTI_PROTOTYPE_HEAD:
            return self._classify_multi_prototype(array)
        return self._classify_legacy(array)

    def classify_crops(
        self,
        crops: Sequence[ImageInput],
        *,
        array_color="rgb",
    ) -> list[DinoV3Prediction]:
        if self.encoder is None:
            raise RuntimeError("DINOv3 encoder is not loaded")
        return self.classify_features(
            self.encoder.encode(crops, array_color=array_color)
        )

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
