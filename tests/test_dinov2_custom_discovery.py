from __future__ import annotations

import numpy as np
import pytest

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.custom_discovery import discover, _event_labels
from tests.test_dinov2_memory import _memory_model, _unit


def test_custom_gate_rejects_novel_batch_and_keeps_known_assignments(tmp_path):
    model, _ = _memory_model(tmp_path, weight=0.5)
    checkpoint = load_checkpoint(model)
    calibration = np.stack([_unit(0) for _ in range(20)])
    query = np.stack([_unit(4) for _ in range(8)] + [_unit(0)])
    result = discover(
        checkpoint, calibration, query,
        [f"cal-{i}" for i in range(20)], [f"query-{i}" for i in range(9)],
        calibration_camera_ids=[f"cal-camera-{i}" for i in range(20)],
        query_camera_ids=[f"query-camera-{i}" for i in range(9)],
    )
    assert result["target_known_calibration_frr"] == 0.05
    assert result["empirical_calibration_frr"] <= 0.05
    assert result["cross_camera_density"] is True
    assert result["rejected_count"] >= 8
    assert all(index >= 0 for index in result["novel_cluster"][:8])
    assert result["novel_cluster"][-1] == -1
    assert result["known_species"][-1] == "A"


def test_custom_density_rejects_batches_without_five_eligible_neighbors(tmp_path):
    model, _ = _memory_model(tmp_path)
    with pytest.raises(ValueError, match="five neighbors"):
        discover(
            load_checkpoint(model), np.stack([_unit(0)] * 20), np.stack([_unit(4)] * 5),
            [f"cal-{i}" for i in range(20)], ["same-sequence"] * 5,
        )


def test_hdbscan_automatically_separates_two_novel_groups(tmp_path):
    model, _ = _memory_model(tmp_path)
    calibration = np.stack([_unit(0) for _ in range(20)])
    query = np.stack([_unit(4) for _ in range(6)] + [_unit(5) for _ in range(6)])
    result = discover(
        load_checkpoint(model), calibration, query,
        [f"cal-{i}" for i in range(20)], [f"query-{i}" for i in range(12)],
    )
    labels = result["novel_cluster"]
    assert result["rejected_count"] == 12
    assert result["occupied_clusters"] == 2
    assert len(set(labels[:6])) == len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_burst_images_do_not_supply_multiple_density_events():
    features = np.stack([_unit(4)] * 9 + [_unit(5)])
    labels = _event_labels(features, np.array(["burst"] * 9 + ["single"]),
                           min_cluster_size=4, min_samples=None)
    assert labels.tolist() == [-1] * 10


def test_discovery_distinguishes_novel_noise_from_known_rows(tmp_path):
    model, _ = _memory_model(tmp_path)
    result = discover(
        load_checkpoint(model), np.stack([_unit(0)] * 20),
        np.stack([_unit(4)] * 6 + [_unit(0)]),
        [f"cal-{i}" for i in range(20)], [f"query-{i}" for i in range(7)],
        min_cluster_size=10,
    )
    assert result["algorithm"] == "hdbscan"
    assert result["novel_cluster"] == [-2] * 6 + [-1]
    assert result["rejected_mask"] == [True] * 6 + [False]
    assert result["noise_count"] == result["rejected_count"] == 6
    assert result["occupied_clusters"] == 0


def test_discovery_api_uses_hdbscan_settings(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from system.dinov2.api import dinov2_registry_router
    from system.backend import dinov2_registry_service
    model, _ = _memory_model(tmp_path)
    monkeypatch.setattr(dinov2_registry_service, "load_checkpoint_for_model",
                        lambda path: load_checkpoint(model))
    app = FastAPI()
    app.include_router(dinov2_registry_router())
    client = TestClient(app)
    payload = {
        "classification_model_path": str(model),
        "calibration_features": np.stack([_unit(0)] * 20).tolist(),
        "query_features": np.stack([_unit(4)] * 6 + [_unit(5)] * 6).tolist(),
        "calibration_sequence_ids": [f"cal-{i}" for i in range(20)],
        "query_sequence_ids": [f"query-{i}" for i in range(12)],
    }
    response = client.post("/api/dinov2/discovery/hdbscan", json=payload)
    assert response.status_code == 200
    assert response.json()["occupied_clusters"] == 2
    assert response.json()["algorithm"] == "hdbscan"
    for invalid in ({"min_cluster_size": 1}, {"min_samples": 0},
                    {"n_clusters": 2}, {"seed": 123}, {"min_cluster_size": True}):
        assert client.post("/api/dinov2/discovery/hdbscan",
                           json={**payload, **invalid}).status_code == 422
    assert client.post("/api/dinov2/discovery/custom-weighted-kmeans", json=payload).status_code == 404
