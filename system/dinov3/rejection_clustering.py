"""Task-wide HDBSCAN grouping of rejected crops; never changes species or gates.

Representatives are target tracks, with byte-identical crop aliases collapsed
across input directories. Saved JSON metadata is sufficient to reopen a task:
there is no UI-triggered clustering and no partitioning at API page boundaries.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3
PCA_DIMENSIONS = 32
ALGORITHM_VERSION = 1


class ClusteringCancelled(RuntimeError):
    pass


def _check_cancelled(callback):
    if callback is not None and callback():
        raise ClusteringCancelled("rejection clustering cancelled")


def _unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cluster_rejections(features, observation_ids, cameras, tracks, contents, *,
                       model_fingerprint, scope_id="", is_cancelled=None):
    """Return JSON-ready metadata per rejected observation, across all folders.

    Inputs must contain rejected crops only. Track IDs must describe individual
    targets, including for the experiment recipe whose gate uses whole events.
    This function does not alter any of its inputs or perform model learning.
    """
    _check_cancelled(is_cancelled)
    values = np.asarray(features, dtype=np.float32)
    count = len(observation_ids)
    if not count:
        return {}
    if values.ndim != 2 or len(values) != count or not np.isfinite(values).all():
        raise ValueError("rejection features must be finite [N, D]")
    if np.any(np.linalg.norm(values, axis=1) <= 1e-12):
        raise ValueError("rejection features must be nonzero")
    if any(len(x) != count for x in (cameras, tracks, contents)):
        raise ValueError("rejection metadata must align with features")
    ids = list(map(str, observation_ids))
    if len(set(ids)) != count:
        raise ValueError("rejection observation IDs must be unique")
    # Stable order makes PCA, HDBSCAN tie resolution, run IDs and numbering
    # independent of folder selection / inference batch order.
    order = sorted(range(count), key=lambda i: ids[i])
    values = _unit(values[order])
    ids = [ids[i] for i in order]
    cameras, tracks, contents = ([str(x[i]) for i in order] for x in (cameras, tracks, contents))
    parent = list(range(count))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    by_track, by_content = {}, {}
    for i in range(count):
        if i % 1024 == 0:
            _check_cancelled(is_cancelled)
        if tracks[i]:
            key = (cameras[i], tracks[i])
            merge(i, by_track.setdefault(key, i))
        if contents[i]:
            merge(i, by_content.setdefault(contents[i], i))
    units = defaultdict(list)
    for i in range(count):
        units[root(i)].append(i)
    members = list(units.values())
    representatives = []
    for rows in members:
        unique = {}
        for i in rows:
            unique.setdefault(contents[i] or ids[i], values[i])
        representatives.append(_unit(np.mean(list(unique.values()), axis=0)))
    representatives = np.asarray(representatives, dtype=np.float32)
    digest = hashlib.sha256()
    from sklearn import __version__ as sklearn_version
    config = dict(version=ALGORITHM_VERSION, algorithm="sklearn.cluster.HDBSCAN",
                  implementation_version=sklearn_version,
                  min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                  pca_dimensions=PCA_DIMENSIONS, cluster_selection_method="eom")
    digest.update(json.dumps([scope_id, model_fingerprint, config],
                             ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for rows in members:
        digest.update(b"[")
        for i in rows:
            digest.update(json.dumps(ids[i], ensure_ascii=False).encode("utf-8"))
            digest.update(b",")
        digest.update(b"]")
    digest.update(memoryview(representatives))
    run_id = digest.hexdigest()
    labels = np.full(len(members), -1, dtype=int)
    strengths = np.zeros(len(members))
    _check_cancelled(is_cancelled)
    if len(members) > MIN_CLUSTER_SIZE:
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import PCA
        from threadpoolctl import threadpool_limits

        projected = representatives
        with threadpool_limits(limits=4):
            if min(projected.shape) > PCA_DIMENSIONS:
                projected = PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized",
                                random_state=0, whiten=False).fit_transform(projected)
            _check_cancelled(is_cancelled)
            fitted = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                             metric="euclidean", cluster_selection_method="eom",
                             n_jobs=4, copy=True).fit(projected)
        labels, strengths = fitted.labels_, fitted.probabilities_
    _check_cancelled(is_cancelled)
    # HDBSCAN numeric labels are local implementation details, not durable IDs.
    grouped = defaultdict(list)
    for j, label in enumerate(labels):
        if label >= 0:
            grouped[int(label)].append(j)
    clusters = sorted(grouped, key=lambda label: min(ids[i] for j in grouped[label] for i in members[j]))
    numbers = {label: index + 1 for index, label in enumerate(clusters)}
    sizes = {label: len(grouped[label]) for label in clusters}
    output = {}
    for j, rows in enumerate(members):
        label = int(labels[j])
        noise = label < 0
        number = numbers.get(label, 0)
        for i in rows:
            output[ids[i]] = dict(
                cluster_id=f"{run_id}:{number}" if not noise else f"{run_id}:noise",
                candidate_number=number, model_fingerprint=str(model_fingerprint),
                event_count=sizes.get(label, 1), min_cluster_events=MIN_CLUSTER_SIZE,
                density_verified=not noise, is_noise=noise,
                status="noise" if noise else "clustered", run_id=run_id,
                membership_strength=float(strengths[j]), independent_target_count=len(members),
                algorithm=config,
            )
    return output


def failed_rejections(observation_ids, *, model_fingerprint, scope_id=""):
    """Persist an explicit failure without discarding successful classification."""
    run_id = hashlib.sha256(json.dumps([scope_id, str(model_fingerprint), sorted(observation_ids)]).encode()).hexdigest()
    return {oid: dict(cluster_id=f"{run_id}:failed", candidate_number=0,
                      model_fingerprint=str(model_fingerprint), event_count=0,
                      min_cluster_events=MIN_CLUSTER_SIZE, density_verified=False,
                      is_noise=False, status="failed", run_id=run_id)
            for oid in observation_ids}
