from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.feedback import FeedbackObservation, HumanFeedbackStore
from system.dinov3.registry import SpeciesRegistry
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def _unit(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _normalized(*values: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(768, dtype=np.float32)
    for index, weight in values:
        vector[index] = weight
    return vector / np.linalg.norm(vector)


def _feedback_observation(index: int, embedding: np.ndarray) -> FeedbackObservation:
    return FeedbackObservation(
        id=f"known-{index}",
        source_path=f"/tmp/known-{index}.jpg",
        media_kind="image",
        box_index=0,
        frame_index=None,
        timestamp_seconds=None,
        bbox=(1.0, 2.0, 30.0, 40.0),
        camera_id=f"camera-{index % 2}",
        captured_at=datetime(2026, 9, 10, 10, 0, 0) + timedelta(minutes=31 * index),
        predicted_species="盘羊",
        best_known_species="盘羊",
        accepted=True,
        prediction_source="base",
        registry_id=None,
        known_score=0.8,
        threshold=-1.0,
        embedding=embedding,
    )


def test_catalog_exposes_known_species_feedback_clusters(tmp_path):
    from system.dinov3.api import build_registry_catalog

    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(classes=("盘羊", "鹅喉羚"), threshold=-1.0)
    )
    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )
    try:
        for index in range(4):
            observation = _feedback_observation(index, _normalized((0, 1.0), (2, 0.05 * index)))
            feedback.persist_observation(observation)
            feedback.record_feedback(
                observation.id,
                operation_id=f"known-op-{index}",
                action="correct",
            )
        feedback.recompute_species("盘羊", np.zeros(768, dtype=np.float32))

        registry = SimpleNamespace(list=lambda status=None: [])
        catalog = build_registry_catalog(checkpoint, registry, feedback=feedback)
        sheep = next(item for item in catalog if item["display_name"] == "盘羊")

        assert sheep["feedback_event_count"] == 4
        assert sheep["feedback_prototype_count"] == 1
        assert sheep["learning_status"] == "provisional"
        sources = [cluster["source"] for cluster in sheep["clusters"]]
        assert sources.count("checkpoint") == checkpoint.prototypes_per_class[0]
        assert "feedback" in sources
        feedback_cluster = next(
            cluster for cluster in sheep["clusters"] if cluster["source"] == "feedback"
        )
        assert feedback_cluster["event_count"] == 4
        assert feedback_cluster["active"] is True
        assert feedback_cluster["example_refs"]
        assert feedback_cluster["example_refs"][0]["kind"] == "observation"
    finally:
        feedback.close()


def test_registry_cluster_details_group_events_under_current_prototypes(tmp_path):
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint="b" * 64,
        consistency_threshold=0.0,
    )
    try:
        first = registry.record_unknown(
            _unit(0),
            camera_id="cam-a",
            captured_at=None,
            source_path="/tmp/event-0.jpg",
            bbox=(0, 0, 20, 20),
        )
        for index in range(1, 4):
            registry.record_observation(
                first.id,
                _unit(index % 2),
                camera_id=f"cam-{index % 2}",
                captured_at=None,
                source_path=f"/tmp/event-{index}.jpg",
                bbox=(0, 0, 20, 20),
            )
        registry.set_identity(first.id, common_name="赤麂")
        registry.register(first.id)
        for index in range(4, 10):
            registry.record_observation(
                first.id,
                _unit(index % 2),
                camera_id=f"cam-{index % 2}",
                captured_at=None,
                source_path=f"/tmp/event-{index}.jpg",
                bbox=(0, 0, 20, 20),
            )

        detail = registry.get(first.id)
        assert detail.status == "confirmed"
        clusters = registry.cluster_details(first.id)

        assert len(clusters) == 2
        assert sum(cluster["event_count"] for cluster in clusters) == 10
        assert all(cluster["source"] == "registry" for cluster in clusters)
        assert all(cluster["example_refs"] for cluster in clusters)
        assert all(cluster["mean_squared_distance"] >= 0 for cluster in clusters)
        event_ids = {
            ref["event_id"]
            for cluster in clusters
            for ref in cluster["example_refs"]
        }
        assert event_ids
    finally:
        registry.close()


def test_local_projection_axes_depend_only_on_prototypes_not_current_sample():
    payload = make_multi_prototype_payload(
        classes=("A", "B"),
        threshold=-1.0,
        selection_k=2,
    )
    prototypes = torch.stack(
        [
            torch.tensor(_unit(0)),
            torch.tensor(_normalized((0, 1.0), (2, 0.35))),
            torch.tensor(_unit(1)),
            torch.tensor(_normalized((1, 1.0), (2, -0.25))),
        ]
    )
    payload["prototypes"] = prototypes
    payload["prototype_class_indices"] = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    payload["prototypes_per_class"] = [2, 2]
    checkpoint = validate_checkpoint(payload)
    classifier = DinoV3Classifier(checkpoint)

    above = classifier.explain_feature(_normalized((0, 1.0), (2, 0.12)))
    below = classifier.explain_feature(_normalized((0, 1.0), (2, -0.12)))

    def prototype_coordinates(result):
        return {
            (point["species"], point["prototype_index"]): (point["x"], point["y"])
            for point in result["projection"]["points"]
            if point["kind"] == "prototype"
        }

    assert prototype_coordinates(above) == pytest.approx(prototype_coordinates(below))
    above_current = next(
        point for point in above["projection"]["points"] if point["kind"] == "current"
    )
    below_current = next(
        point for point in below["projection"]["points"] if point["kind"] == "current"
    )
    assert above_current["y"] * below_current["y"] < 0
