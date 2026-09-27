"""DINOv2 Memory classifier with optional class-centroid score mixing."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .classifier import (
    DinoV2Classifier,
    DinoV2Prediction,
    aggregate_event_embeddings,
)
from .memory_bank import MemoryBank, MemoryExample
from .memory_checkpoint import MemoryCheckpoint
from .preprocess import ImageInput


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Embedding becomes zero after Memory centering")
    return (values / norms).astype(np.float32, copy=False)


class MemoryDinoV2Classifier:
    def __init__(self, checkpoint: MemoryCheckpoint, *, encoder=None, feedback=None, registry=None):
        if checkpoint.head_type not in {"memory", "memory_no_centroid"}:
            raise ValueError("Expected Memory checkpoint")
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.feedback = feedback
        self.registry = registry
        self.names = {index: name for index, name in enumerate(checkpoint.classes)}
        self.backend = "dinov2"
        self._center = checkpoint.feature_center.numpy()

    @property
    def classes(self) -> tuple[str, ...]:
        return self.checkpoint.classes

    @property
    def rejection_metadata(self) -> dict[str, object]:
        return {
            "mode": self.checkpoint.head_type,
            "threshold": self.checkpoint.threshold,
            "neighbors": self.checkpoint.neighbors,
            "camera_pooling": True,
            "centroid_weight": self.checkpoint.centroid_weight,
            "margin_weight": self.checkpoint.margin_weight,
        }

    def _provider_bank(self, provider) -> MemoryBank:
        if provider is None or not callable(getattr(provider, "memory_bank", None)):
            return MemoryBank(())
        bank = provider.memory_bank()
        if not isinstance(bank, MemoryBank):
            raise TypeError("memory_bank() must return a MemoryBank")
        return bank

    def _arrays(self, examples: tuple[MemoryExample, ...]):
        if not examples:
            return self.checkpoint.features, self.checkpoint.labels, self.checkpoint.cameras
        raw = np.stack([example.embedding for example in examples])
        additional = _normalize_rows(raw - self._center[None, :])
        return (
            np.concatenate((self.checkpoint.features, additional)),
            np.concatenate((self.checkpoint.labels, [example.species for example in examples])),
            np.concatenate((self.checkpoint.cameras, [example.camera_id for example in examples])),
        )

    def _scores(self, centered: np.ndarray, bank: tuple[MemoryExample, ...] = ()):
        features, labels, cameras_all = self._arrays(bank)
        classes = tuple(dict.fromkeys((*self.classes, *(example.species for example in bank))))
        similarity = centered @ features.T
        scores = np.empty((len(centered), len(classes)), dtype=np.float32)
        nearest = np.empty((len(centered), len(classes)), dtype=np.int64)
        for class_index, species in enumerate(classes):
            mask = labels == species
            values = similarity[:, mask]
            cameras = cameras_all[mask]
            pooled = np.column_stack(
                [values[:, cameras == camera].max(axis=1) for camera in np.unique(cameras)]
            )
            k = min(self.checkpoint.neighbors, pooled.shape[1])
            memory_score = np.partition(pooled, pooled.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
            if self.checkpoint.centroid_weight:
                if species in self.checkpoint.classes and not any(
                    example.species == species for example in bank
                ):
                    centroid = self.checkpoint.centroids[self.checkpoint.classes.index(species)]
                else:
                    centroid = _normalize_rows(features[mask].mean(axis=0, keepdims=True))[0]
                scores[:, class_index] = (
                    (1.0 - self.checkpoint.centroid_weight) * memory_score
                    + self.checkpoint.centroid_weight * (centered @ centroid)
                )
            else:
                scores[:, class_index] = memory_score
            nearest[:, class_index] = np.flatnonzero(mask)[np.argmax(values, axis=1)]
        return scores, nearest, classes

    def classify_features(self, features: np.ndarray) -> list[DinoV2Prediction]:
        raw = DinoV2Classifier._validate_features(features)
        centered = _normalize_rows(raw - self._center[None, :])
        registry = self._provider_bank(self.registry)
        feedback = self._provider_bank(self.feedback)
        formal_examples = feedback.formal + registry.formal
        provisional_examples = feedback.provisional + registry.provisional
        scores, nearest, classes = self._scores(centered, formal_examples)
        results: list[DinoV2Prediction] = []
        for index, row in enumerate(scores):
            order = np.argsort(-row, kind="stable")
            winner, runner_up = int(order[0]), int(order[1])
            margin = float(row[winner] - row[runner_up])
            knownness = float(row[winner] + self.checkpoint.margin_weight * margin)
            accepted = knownness >= self.checkpoint.threshold
            species = classes[winner]
            nearest_index = int(nearest[index, winner])
            selected = (
                formal_examples[nearest_index - len(self.checkpoint.features)]
                if nearest_index >= len(self.checkpoint.features)
                else None
            )
            candidates = tuple(
                {"name": classes[int(j)], "known_score": float(row[j]), "source": "base"}
                for j in order[:3]
            )
            assistive = None
            if not accepted and provisional_examples:
                combined = formal_examples + provisional_examples
                combined_scores, combined_nearest, combined_classes = self._scores(centered[index:index + 1], combined)
                combined_row = combined_scores[0]
                combined_order = np.argsort(-combined_row, kind="stable")
                combined_winner = int(combined_order[0])
                combined_runner_up = int(combined_order[1])
                combined_index = int(combined_nearest[0, combined_winner])
                if combined_index >= len(self.checkpoint.features) + len(formal_examples):
                    combined_knownness = float(
                        combined_row[combined_winner]
                        + self.checkpoint.margin_weight
                        * (combined_row[combined_winner] - combined_row[combined_runner_up])
                    )
                    if combined_knownness >= self.checkpoint.threshold:
                        assistive = combined[combined_index - len(self.checkpoint.features)]
            results.append(
                DinoV2Prediction(
                    species=species if accepted else (assistive.species if assistive else "Unknown"),
                    accepted=accepted,
                    best_known_species=species,
                    head_species=species,
                    prototype_species=(assistive.species if assistive else species),
                    head_prototype_consistent=assistive is None,
                    known_score=knownness,
                    threshold=self.checkpoint.threshold,
                    candidates=candidates,
                    embedding=raw[index].copy(),
                    source=(assistive.source if assistive else (selected.source if selected else "base")),
                    registry_id=(assistive.registry_id if assistive else (selected.registry_id if selected else None)),
                    registration_status=(assistive.registration_status if assistive else (selected.registration_status if selected else None)),
                    assistive_match=assistive is not None,
                    class_margin=margin,
                    registry_action="match" if accepted else "candidate",
                )
            )
        return results

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
        raw = DinoV2Classifier._validate_features(np.asarray(feature, dtype=np.float32)[None, :])
        prediction = self.classify_features(raw)[0]
        centered = _normalize_rows(raw - self._center[None, :])
        formal = self._provider_bank(self.feedback).formal + self._provider_bank(self.registry).formal
        scores, nearest, classes = self._scores(centered, formal)
        order = np.argsort(-scores[0], kind="stable")[:2]
        features, _, _ = self._arrays(formal)
        representative = [features[int(nearest[0, int(index)])] for index in order]
        first, second = representative
        origin = (first + second) * 0.5
        axis_x = second - first
        axis_x_norm = float(np.linalg.norm(axis_x))
        if axis_x_norm <= 1e-12:
            axis_x = np.eye(1, len(first), 0, dtype=np.float32)[0]
        else:
            axis_x = axis_x / axis_x_norm
        basis = np.zeros_like(axis_x)
        basis[int(np.argmin(np.abs(axis_x)))] = 1.0
        axis_y = basis - float(basis @ axis_x) * axis_x
        axis_y /= np.linalg.norm(axis_y)

        def project(vector):
            delta = vector - origin
            return float(delta @ axis_x), float(delta @ axis_y)

        points = []
        nearest_species = []
        for rank, class_index in enumerate(order):
            exemplar_index = int(nearest[0, int(class_index)])
            example = (
                formal[exemplar_index - len(self.checkpoint.features)]
                if exemplar_index >= len(self.checkpoint.features)
                else None
            )
            species = classes[int(class_index)]
            x, y = project(representative[rank])
            points.append(
                {
                    "kind": "prototype", "species": species,
                    "source": example.source if example else "base",
                    "prototype_index": exemplar_index, "x": x, "y": y,
                }
            )
            cosine = float(centered[0] @ representative[rank])
            nearest_species.append(
                {
                    "name": species,
                    "nearest_prototype_index": exemplar_index,
                    "squared_distance": max(0.0, 2.0 - 2.0 * cosine),
                    "cosine_score": float(scores[0, int(class_index)]),
                    "source": example.source if example else "base",
                    "registry_id": example.registry_id if example else None,
                    "registration_status": example.registration_status if example else None,
                }
            )
        current_x, current_y = project(centered[0])
        points.append(
            {
                "kind": "current", "species": prediction.species,
                "source": prediction.source, "prototype_index": None,
                "x": current_x, "y": current_y,
            }
        )
        return {
            "species": prediction.species,
            "accepted": prediction.accepted,
            "best_known_species": prediction.best_known_species,
            "known_score": prediction.known_score,
            "threshold": prediction.threshold,
            "score_threshold": None,
            "nearest_prototype_index": None,
            "squared_distance": None,
            "class_margin": prediction.class_margin,
            "adjusted_distance_score": None,
            "rejection": self.rejection_metadata,
            "nearest_species": nearest_species,
            "projection": {
                "method": "nearest_two_species_axis",
                "species": [classes[int(index)] for index in order],
                "points": points,
            },
        }
