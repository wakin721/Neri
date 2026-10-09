from __future__ import annotations

import numpy as np
import pytest

from system.dinov2.within_seq import (
    WithinSeqModel, calibration_threshold, cluster_rejected_sequences,
    discover_within_seq, hierarchical_weights, select_neighbors,
)


def unit(values):
    values = np.asarray(values, dtype=float)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def bank(dimension=4):
    rng = np.random.default_rng(23)
    x = rng.normal(scale=0.12, size=(24, dimension))
    x[:12, 0] += 1
    x[12:, 1] += 1
    return unit(x), ['A'] * 12 + ['B'] * 12, [f'c{i // 4}' for i in range(24)], [f's{i // 2}' for i in range(24)]


def test_hierarchy_equalizes_classes_cameras_sequences_and_burst_images():
    labels = np.array(['A'] * 5 + ['B'])
    cameras = np.array(['a', 'a', 'a', 'a', 'b', 'c'])
    seq = np.array(['s', 's', 's', 't', 'u', 'v'])
    weights = hierarchical_weights(labels, cameras, seq)
    np.testing.assert_allclose(weights, [1/24, 1/24, 1/24, 1/8, 1/4, 1/2])
    assert weights.sum() == pytest.approx(1)


def test_val90_includes_order_statistic_ties_and_uses_strict_rejection():
    values = np.arange(10, dtype=float)
    threshold = calibration_threshold(values)
    assert threshold == np.nextafter(8.0, np.inf)
    assert np.mean(values < threshold) == 0.9
    assert not threshold < threshold
    assert np.mean(np.ones(10) < calibration_threshold(np.ones(10))) == 1.0
    with pytest.raises(ValueError):
        calibration_threshold([np.nan])


def test_adaptive_k_uses_seventeenth_neighbor_and_smaller_tie():
    distance = np.r_[np.full(16, 0.1), 0.9]
    assert select_neighbors((1-distance)[None, :]).tolist() == [16]
    assert select_neighbors(np.full((2, 17), 0.9)).tolist() == [1, 1]
    assert select_neighbors(np.ones((1, 1))).tolist() == [1]


def test_whitening_matches_hierarchical_covariance_and_remains_fixed():
    x, labels, cameras, seq = bank()
    model = WithinSeqModel(x, labels, cameras, seq, dimension=4)
    labels, cameras, seq = map(np.asarray, (labels, cameras, seq))
    weights = hierarchical_weights(labels, cameras, seq)
    residual = x.copy()
    for species in np.unique(labels):
        rows = labels == species
        residual[rows] -= np.average(x[rows], axis=0, weights=weights[rows])
    covariance = (residual * weights[:, None]).T @ residual
    covariance = 0.9 * covariance + 0.1 * np.trace(covariance) / 4 * np.eye(4)
    np.testing.assert_allclose(model.whitening @ model.whitening.T, np.linalg.inv(covariance))
    before = model.whitening.copy()
    model.score_batch(x[::-1], cameras, seq)
    np.testing.assert_array_equal(model.whitening, before)


def test_sequence_mean_follows_image_scoring_and_is_camera_scoped():
    x, labels, cameras, seq = bank()
    model = WithinSeqModel(x, labels, cameras, seq, dimension=4)
    qcam = ['q1', 'q1', 'q2', 'q2']
    qseq = ['same'] * 4
    small = model.score_batch(x[[0, 12, 3, 4]], qcam, qseq, batch_size=1)
    large = model.score_batch(x[[0, 12, 3, 4]], qcam, qseq, batch_size=128)
    np.testing.assert_allclose(small['sequence_knownness'], large['sequence_knownness'])
    np.testing.assert_allclose(small['sequence_knownness'][:2], small['image_knownness'][:2].mean())
    assert small['sequence_knownness'][0] != small['sequence_knownness'][2]


def test_frozen_memory_predictions_match_independent_camera_top_three():
    x, labels, cameras, seq = bank()
    model = WithinSeqModel(x, labels, cameras, seq, dimension=4)
    result = model.score_batch(x, cameras, seq)
    query = unit(x - model.classification_center)
    expected = []
    for q in query:
        scores = []
        for species in model.classes:
            mask = model.labels == species
            similarities = q @ model.memory.T
            pooled = [max(similarities[mask & (model.cameras == c)]) for c in np.unique(model.cameras[mask])]
            local = np.mean(sorted(pooled, reverse=True)[:3])
            centroid = unit(model.memory[mask].mean(axis=0)[None, :])[0]
            scores.append(0.5 * local + 0.5 * float(q @ centroid))
        expected.append(model.classes[np.argmax(scores)])
    assert result['known_species'] == expected


def test_small_candidate_pool_retains_noise_and_distinguishes_accepted():
    x = unit([[1, .1], [1, .2], [.1, 1]])
    labels, count = cluster_rejected_sequences(x, ['a', 'a', 'b'], ['s', 's', 's'], [True, True, False])
    assert labels == [-1, -1, None]
    assert count == 1


def test_hdbscan_separates_raw_sequence_features_without_requested_k():
    rng = np.random.default_rng(9)
    x = unit(np.r_[rng.normal([1, 0, 0], .015, (12, 3)), rng.normal([0, 1, 0], .015, (12, 3))])
    labels, count = cluster_rejected_sequences(x, [f'c{i}' for i in range(24)], ['s']*24, [True]*24)
    assert count == 24
    assert len(set(labels[:12])) == len(set(labels[12:])) == 1
    assert labels[0] != labels[12]
    assert -1 not in labels


def test_batch_discovery_uses_aux_unknown_and_rejects_sequence_leakage():
    x, labels, cameras, seq = bank(768)
    auxiliary = unit(np.eye(768)[[10, 11, 12, 13]])
    args = [x, labels, cameras, seq, auxiliary, ['aux']*4, ['a','b','c','d'], auxiliary.copy(), ['query']*4, ['a','b','c','d']]
    result = discover_within_seq(*args)
    assert result['classification_neighbors'] == 3
    assert result['empirical_aux_unknown_recall'] >= .9
    assert result['novel_cluster'] == [-1]*4
    assert result['review_budget_images'] == 1
    args[-2] = ['aux']*4
    with pytest.raises(ValueError, match='disjoint'):
        discover_within_seq(*args)


@pytest.mark.parametrize('bad', [np.zeros((2,4)), np.full((2,4), np.nan), np.ones((2,3))])
def test_rejects_invalid_features(bad):
    with pytest.raises(ValueError):
        WithinSeqModel(bad, ['A','B'], ['a','b'], ['s','t'], dimension=4)


def test_api_validates_input_and_returns_json(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from system.dinov2.api import dinov2_registry_router
    import system.dinov2.within_seq as module
    app = FastAPI()
    app.include_router(dinov2_registry_router())
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {'classification_neighbors': 3, 'novel_cluster': [None, -1]}
    monkeypatch.setattr(module, 'discover_within_seq', fake)
    sample = {'features': [[1.,0.]], 'camera_ids': ['a'], 'sequence_ids': ['s']}
    with TestClient(app) as client:
        response = client.post('/api/dinov2/discovery/within-seq', json={
            'bank': {**sample, 'labels': ['A']}, 'auxiliary_unknown': sample, 'query': sample,
        })
        assert response.status_code == 200
        assert response.json()['novel_cluster'] == [None, -1]
        assert calls[0][1]['target'] == .9
        assert client.post('/api/dinov2/discovery/within-seq', json={}).status_code == 422
        monkeypatch.undo()
        response = client.post('/api/dinov2/discovery/within-seq', json={
            'bank': {**sample, 'labels': ['A']}, 'auxiliary_unknown': sample, 'query': sample,
        })
        assert response.status_code == 400
        assert '768' in response.json()['detail']
