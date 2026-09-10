from __future__ import annotations

import numpy as np
import pytest

from system.dinov3.calibration import (
    apply_multi_dual_gate,
    calibrate_multi_dual,
    cl2n_transform,
    prototype_signals,
    species_macro_rejection,
)


def test_cl2n_subtracts_center_then_normalizes():
    raw = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    center = np.asarray([0.25, 0.0], dtype=np.float32)

    out = cl2n_transform(raw, center)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], [1.0, 0.0])


def test_prototype_signals_use_squared_euclidean_winner_and_winner_cosine():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prototypes = np.asarray([[0.5, 0.0], [0.0, 0.2]], dtype=np.float32)

    signals = prototype_signals(features, prototypes)

    assert signals["winner_index"].tolist() == [0, 1]
    assert np.allclose(signals["winner_cosine"], [1.0, 1.0])
    assert np.allclose(signals["winner_squared_distance"], [0.25, 0.64])


def test_dual_gate_requires_both_conditions():
    accepted = apply_multi_dual_gate(
        np.asarray([0.8, 0.4, 0.8]),
        np.asarray([0.2, 0.2, 0.8]),
        cosine_threshold=0.5,
        squared_distance_threshold=0.5,
    )

    assert accepted.tolist() == [True, False, False]


def test_species_macro_rejection_weights_species_equally():
    accepted = np.asarray([False, False, True, True, True, True])
    labels = np.asarray(["U1", "U1", "U2", "U2", "U2", "U2"])

    assert species_macro_rejection(accepted, labels) == pytest.approx(0.5)


def test_joint_calibration_obeys_known_frr_cap_and_uses_complementary_signals():
    known_cos = np.asarray([0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74])
    known_dist = np.asarray([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    proxy_cos = np.asarray([0.83, 0.81, 0.79, 0.40, 0.35, 0.30])
    proxy_dist = np.asarray([0.70, 0.65, 0.60, 0.09, 0.08, 0.07])
    proxy_labels = np.asarray(["U1", "U1", "U1", "U2", "U2", "U2"])

    calibration = calibrate_multi_dual(
        known_cos,
        known_dist,
        proxy_cos,
        proxy_dist,
        proxy_labels,
        max_known_frr=0.10,
        grid_size=64,
    )
    known_accept = apply_multi_dual_gate(
        known_cos,
        known_dist,
        cosine_threshold=calibration["cosine_threshold"],
        squared_distance_threshold=calibration["squared_distance_threshold"],
    )
    proxy_accept = apply_multi_dual_gate(
        proxy_cos,
        proxy_dist,
        cosine_threshold=calibration["cosine_threshold"],
        squared_distance_threshold=calibration["squared_distance_threshold"],
    )

    assert np.mean(~known_accept) <= 0.10 + 1e-12
    assert np.mean(~proxy_accept) >= 5 / 6
    assert calibration["mode"] == "multi_dual"
