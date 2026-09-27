from __future__ import annotations

import numpy as np
import pytest

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.custom_discovery import discover, _sequence_weights
from tests.test_dinov2_memory import _memory_model, _unit


def test_custom_gate_rejects_novel_batch_and_keeps_known_assignments(tmp_path):
    model, _ = _memory_model(tmp_path, weight=0.5)
    checkpoint = load_checkpoint(model)
    calibration = np.stack([_unit(0) for _ in range(20)])
    query = np.stack([_unit(4) for _ in range(8)] + [_unit(0)])
    result = discover(
        checkpoint, calibration, query,
        [f"cal-{i}" for i in range(20)], [f"query-{i}" for i in range(9)],
        n_clusters=2,
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
            n_clusters=1,
        )


def test_weighted_kmeans_separates_two_novel_groups(tmp_path):
    model, _ = _memory_model(tmp_path)
    calibration = np.stack([_unit(0) for _ in range(20)])
    query = np.stack([_unit(4) for _ in range(6)] + [_unit(5) for _ in range(6)])
    result = discover(
        load_checkpoint(model), calibration, query,
        [f"cal-{i}" for i in range(20)], [f"query-{i}" for i in range(12)],
        n_clusters=2,
    )
    labels = result["novel_cluster"]
    assert result["rejected_count"] == 12
    assert result["occupied_clusters"] == 2
    assert len(set(labels[:6])) == len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_burst_weights_reduce_repeated_sequence_mass():
    weights = _sequence_weights(np.array(["burst"] * 9 + ["single"]))
    assert weights[0] < weights[-1]
    assert weights.mean() == pytest.approx(1.0)
