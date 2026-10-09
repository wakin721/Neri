from __future__ import annotations

import numpy as np
import pytest

from system.dinov2.clustering import hdbscan_labels, hdbscan_prototypes


def _groups():
    values = np.zeros((13, 768), dtype=np.float32)
    values[:6, 0] = 1
    values[6:12, 0] = 1 / np.sqrt(1.25)
    values[6:12, 1] = 0.5 / np.sqrt(1.25)
    values[12, 2] = 1
    return values


def test_hdbscan_finds_groups_and_leaves_outlier_as_noise():
    labels = hdbscan_labels(_groups())
    assert len(set(labels[:6])) == len(set(labels[6:12])) == 1
    assert labels[0] >= 0 and labels[6] >= 0 and labels[0] != labels[6]
    assert labels[-1] == -1


def test_hdbscan_labels_are_repeatable_under_row_permutation():
    values = _groups()
    order = np.random.default_rng(7).permutation(len(values))
    np.testing.assert_array_equal(hdbscan_labels(values), hdbscan_labels(values))
    permuted = hdbscan_labels(values[order])
    np.testing.assert_array_equal(hdbscan_labels(values)[order], permuted)


def test_prototypes_exclude_noise_and_obey_learning_stage_budget():
    values = _groups()
    prototypes = hdbscan_prototypes(values, max_prototypes=3)
    assert prototypes.shape == (2, 768)
    assert np.all(prototypes[:, 2] == 0)
    assert hdbscan_prototypes(values, max_prototypes=1).shape == (1, 768)


def test_small_labelled_evidence_has_a_summary_but_no_discovery_cluster():
    values = _groups()[:3]
    assert hdbscan_labels(values).tolist() == [-1] * 3
    np.testing.assert_allclose(hdbscan_prototypes(values, max_prototypes=3), values.mean(axis=0, keepdims=True))
    assert hdbscan_labels(np.empty((0, 768))).shape == (0,)


@pytest.mark.parametrize("settings", [
    {"min_cluster_size": 1}, {"min_cluster_size": True},
    {"min_cluster_size": 2.5}, {"min_samples": 0},
    {"min_samples": False}, {"cluster_selection_epsilon": float("nan")},
])
def test_invalid_hdbscan_settings_fail_explicitly(settings):
    with pytest.raises(ValueError):
        hdbscan_labels(_groups(), **settings)


def test_nonfinite_features_and_invalid_prototype_budget_are_rejected():
    values = _groups()
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        hdbscan_labels(values)
    with pytest.raises(ValueError):
        hdbscan_prototypes(_groups(), max_prototypes=0)
