from __future__ import annotations

import numpy as np

from system.dinov3.simple_shot import deterministic_k_means


def _cluster(center_index: int, count: int, *, offset_index: int) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for sample in range(count):
        vector = np.zeros(768, dtype=np.float32)
        vector[center_index] = 1.0
        vector[offset_index] = 0.01 * (sample + 1)
        rows.append(vector)
    return rows


def test_three_supported_clusters_produce_three_prototypes_repeatably():
    embeddings = np.stack(
        _cluster(0, 4, offset_index=10)
        + _cluster(1, 4, offset_index=11)
        + _cluster(2, 4, offset_index=12)
    )

    first = deterministic_k_means(embeddings, max_k=3)
    second = deterministic_k_means(embeddings, max_k=3)

    assert first.shape == (3, 768)
    np.testing.assert_allclose(first, second, atol=1e-7)


def test_single_outlier_cannot_create_a_third_prototype():
    outlier = np.zeros(768, dtype=np.float32)
    outlier[50] = 1.0
    embeddings = np.stack(
        _cluster(0, 6, offset_index=10)
        + _cluster(1, 6, offset_index=11)
        + [outlier]
    )

    prototypes = deterministic_k_means(embeddings, max_k=3)

    assert prototypes.shape == (2, 768)
