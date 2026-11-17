"""Unit and integration tests for task-wide DINOv3 rejection clustering."""
from __future__ import annotations

from datetime import datetime, timedelta
from importlib.metadata import version as pkg_version
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.backend import services
from system.backend.models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ProcessingOptions,
)
from system.dinov3.observation_support import make_dinov3_observation_id
from system.dinov3.processor import (
    DinoV3CropCollector,
    DinoV3CropRecord,
    SEQUENCE_STATUS_COMPLETE,
    SEQUENCE_STATUS_FAILED,
    _annotate_sequence_state,
    finalize_photo_job,
    mark_dinov3_sequence_incomplete,
)
import system.dinov3.rejection_clustering as rc
from system.dinov3.rejection_clustering import (
    ALGORITHM_VERSION,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    PCA_DIMENSIONS,
    ClusteringCancelled,
    cluster_rejections,
    failed_rejections,
)
from tests.test_dinov3_processing import MockPrediction


def _make_population(rng: np.random.Generator, center_idx: int, count: int, dim: int, noise: float = 0.02) -> np.ndarray:
    center = np.zeros(dim, dtype=np.float32)
    center[center_idx] = 1.0
    vecs = center[None, :] + noise * rng.standard_normal((count, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-12)).astype(np.float32)


def test_cluster_rejections_separates_two_populations_across_directories():
    """Real HDBSCAN separates two synthetic populations across directory-like IDs without mocking sklearn."""
    rng = np.random.default_rng(20261114)
    dim = 64
    pop_a = _make_population(rng, 0, 7, dim)
    pop_b = _make_population(rng, 1, 7, dim)
    outlier_1 = _make_population(rng, 2, 1, dim, noise=0.0)
    outlier_2 = _make_population(rng, 3, 1, dim, noise=0.0)
    features = np.vstack([pop_a, pop_b, outlier_1, outlier_2])

    dirs = ("north_ridge/2026-11-01", "south_valley/2026-11-02", "west_creek/sub_a")
    obs_ids = [
        f"{dirs[i % len(dirs)]}/IMG_{i:04d}.JPG#box0"
        for i in range(len(features))
    ]
    cameras = [f"cam_{i % len(dirs)}" for i in range(len(features))]
    tracks = [f"track_{i}" for i in range(len(features))]
    contents = [f"digest_{i}" for i in range(len(features))]

    result = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="enc-abc:ckpt-123",
        scope_id="job-multi-dir",
    )

    assert set(result.keys()) == set(obs_ids)
    group_a = [result[oid] for oid in obs_ids[:7]]
    group_b = [result[oid] for oid in obs_ids[7:14]]
    outliers = [result[oid] for oid in obs_ids[14:]]

    cluster_ids_a = {item["cluster_id"] for item in group_a}
    cluster_ids_b = {item["cluster_id"] for item in group_b}
    assert len(cluster_ids_a) == 1
    assert len(cluster_ids_b) == 1
    assert cluster_ids_a != cluster_ids_b

    cand_a = {item["candidate_number"] for item in group_a}
    cand_b = {item["candidate_number"] for item in group_b}
    assert cand_a | cand_b == {1, 2}

    for item in group_a + group_b:
        assert item["density_verified"] is True
        assert item["is_noise"] is False
        assert item["status"] == "clustered"
        assert item["event_count"] == 7
        assert item["min_cluster_events"] == MIN_CLUSTER_SIZE
        assert item["independent_target_count"] == 16
        assert item["model_fingerprint"] == "enc-abc:ckpt-123"
        assert item["membership_strength"] > 0.0
        assert item["algorithm"]["version"] == ALGORITHM_VERSION
        assert item["algorithm"]["algorithm"] == "sklearn.cluster.HDBSCAN"
        assert item["algorithm"]["implementation_version"] == pkg_version("scikit-learn")
        assert item["algorithm"]["min_cluster_size"] == MIN_CLUSTER_SIZE
        assert item["algorithm"]["min_samples"] == MIN_SAMPLES
        assert item["algorithm"]["pca_dimensions"] == PCA_DIMENSIONS

    run_id = group_a[0]["run_id"]
    for item in outliers:
        assert item["run_id"] == run_id
        assert item["is_noise"] is True
        assert item["density_verified"] is False
        assert item["status"] == "noise"
        assert item["candidate_number"] == 0
        assert item["cluster_id"] == f"{run_id}:noise"


def test_cluster_rejections_runs_real_pca_when_above_pca_dimensions():
    """When independent units exceed PCA_DIMENSIONS=32, real PCA + HDBSCAN still separates 2 populations."""
    rng = np.random.default_rng(42)
    dim = 64
    pop_a = _make_population(rng, 0, 18, dim)
    pop_b = _make_population(rng, 1, 18, dim)
    features = np.vstack([pop_a, pop_b])
    obs_ids = [f"site_{i % 4}/dir_{i % 3}/img_{i:03d}.jpg#0" for i in range(36)]
    cameras = [f"cam_{i % 4}" for i in range(36)]
    tracks = [f"trk_{i}" for i in range(36)]
    contents = [f"sha_{i}" for i in range(36)]

    result = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="fp-pca",
        scope_id="scope-pca",
    )

    assert {result[oid]["candidate_number"] for oid in obs_ids[:18]} == {1}
    assert {result[oid]["candidate_number"] for oid in obs_ids[18:]} == {2}
    assert all(result[oid]["density_verified"] is True for oid in obs_ids)
    assert all(result[oid]["event_count"] == 18 for oid in obs_ids)
    assert all(result[oid]["independent_target_count"] == 36 for oid in obs_ids)


def test_cluster_rejections_pca_preserves_unequal_norms_for_imbalanced_populations(monkeypatch):
    """Centered PCA skips post-projection renormalization so imbalanced majority does not absorb outliers."""
    from sklearn.cluster import HDBSCAN

    rng = np.random.default_rng(2026111401)
    dim = 64
    majority = _make_population(rng, 0, 30, dim, noise=0.02)
    minority = _make_population(rng, 1, 6, dim, noise=0.02)
    outliers = np.vstack([
        _make_population(rng, 2, 1, dim, noise=0.0),
        _make_population(rng, 3, 1, dim, noise=0.0),
    ])
    features = np.vstack([majority, minority, outliers])
    obs_ids = [f"img_{i:03d}/site_{i % 4}/crop.jpg#0" for i in range(len(features))]
    cameras = [f"cam_{i % 4}" for i in range(len(features))]
    tracks = [f"trk_{i}" for i in range(len(features))]
    contents = [f"sha_{i}" for i in range(len(features))]

    seen_inputs = []
    orig_fit = HDBSCAN.fit

    def spy_fit(self, X, y=None):
        seen_inputs.append(np.array(X, copy=True))
        return orig_fit(self, X, y)

    monkeypatch.setattr(HDBSCAN, "fit", spy_fit)

    result = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="fp-imbalanced",
        scope_id="scope-imbalanced",
    )

    assert len(seen_inputs) == 1
    projected = seen_inputs[0]
    assert projected.shape == (38, PCA_DIMENSIONS)
    norms = np.linalg.norm(projected, axis=1)
    # obs_ids sort lexicographically ("img_000".."img_037"), matching rows 0..37.
    assert norms.max() / norms.min() > 2.0
    assert norms[:30].mean() < norms[30:36].mean()

    maj_clusters = {result[oid]["cluster_id"] for oid in obs_ids[:30]}
    min_clusters = {result[oid]["cluster_id"] for oid in obs_ids[30:36]}
    assert len(maj_clusters) == 1
    assert len(min_clusters) == 1
    assert maj_clusters != min_clusters
    assert all(result[oid]["density_verified"] is True and result[oid]["event_count"] == 30 for oid in obs_ids[:30])
    assert all(result[oid]["density_verified"] is True and result[oid]["event_count"] == 6 for oid in obs_ids[30:36])
    for oid in obs_ids[36:]:
        assert result[oid]["is_noise"] is True
        assert result[oid]["density_verified"] is False
        assert result[oid]["candidate_number"] == 0


def test_cluster_rejections_deterministic_reverse_order_and_scoped_ids():
    """Reverse input order yields identical output; scope_id and model_fingerprint scope run/cluster IDs."""
    rng = np.random.default_rng(7)
    dim = 48
    features = np.vstack([
        _make_population(rng, 0, 6, dim),
        _make_population(rng, 1, 6, dim),
    ])
    obs_ids = [f"folder_{i % 3}/photo_{i:02d}.jpg#0" for i in range(12)]
    cameras = [f"cam_{i % 3}" for i in range(12)]
    tracks = [f"trk_{i}" for i in range(12)]
    contents = [f"hash_{i}" for i in range(12)]

    forward = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="enc1:ckpt1",
        scope_id="job-1",
    )
    reversed_out = cluster_rejections(
        features[::-1],
        list(reversed(obs_ids)),
        list(reversed(cameras)),
        list(reversed(tracks)),
        list(reversed(contents)),
        model_fingerprint="enc1:ckpt1",
        scope_id="job-1",
    )
    assert forward == reversed_out

    diff_scope = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="enc1:ckpt1",
        scope_id="job-2",
    )
    diff_fp = cluster_rejections(
        features,
        obs_ids,
        cameras,
        tracks,
        contents,
        model_fingerprint="enc2:ckpt2",
        scope_id="job-1",
    )

    for oid in obs_ids:
        assert diff_scope[oid]["run_id"] != forward[oid]["run_id"]
        assert diff_scope[oid]["cluster_id"] != forward[oid]["cluster_id"]
        assert diff_scope[oid]["candidate_number"] == forward[oid]["candidate_number"]
        assert diff_scope[oid]["event_count"] == forward[oid]["event_count"]

        assert diff_fp[oid]["run_id"] != forward[oid]["run_id"]
        assert diff_fp[oid]["cluster_id"] != forward[oid]["cluster_id"]
        assert diff_fp[oid]["model_fingerprint"] == "enc2:ckpt2"
        assert diff_fp[oid]["candidate_number"] == forward[oid]["candidate_number"]


def test_cluster_rejections_collapses_tracks_and_duplicates_without_faking_density():
    """Repeated track frames and byte-identical cross-directory copies collapse and cannot fake min-5 density."""
    rng = np.random.default_rng(99)
    dim = 32

    # Part 1: Only 3 independent units behind 10 identical-center rows -> never passes MIN_CLUSTER_SIZE=5.
    small_feats = _make_population(rng, 0, 10, dim, noise=0.005)
    small_ids = [f"dir_{i % 4}/dup_{i}.jpg#0" for i in range(10)]
    small_cams = ["cam_a"] * 5 + [f"cam_{i}" for i in range(5)]
    small_tracks = ["track_1"] * 5 + [f"t_{i}" for i in range(5)]
    # Last 5 rows across different cameras share only 2 content hashes.
    small_contents = [f"c_{i}" for i in range(5)] + ["alias_x", "alias_x", "alias_x", "alias_y", "alias_y"]

    small_res = cluster_rejections(
        small_feats,
        small_ids,
        small_cams,
        small_tracks,
        small_contents,
        model_fingerprint="fp",
        scope_id="small",
    )
    for oid in small_ids:
        assert small_res[oid]["independent_target_count"] == 3
        assert small_res[oid]["density_verified"] is False
        assert small_res[oid]["is_noise"] is True
        assert small_res[oid]["status"] == "noise"
        assert small_res[oid]["candidate_number"] == 0

    # Part 2: Two valid 6-unit populations (A and C) so standard EOM HDBSCAN (allow_single_cluster=False)
    # verifies real clusters, alongside 10 rows in population B
    # that collapse via track + byte-identical content + transitive merge to only 3 independent units (<5).
    pop_a = _make_population(rng, 0, 6, dim, noise=0.01)
    pop_b = _make_population(rng, 1, 10, dim, noise=0.005)
    pop_c = _make_population(rng, 2, 6, dim, noise=0.01)
    features = np.vstack([pop_a, pop_b, pop_c])

    ids_a = [f"dir_a/real_{i}.jpg#0" for i in range(6)]
    cams_a = [f"cam_real_{i}" for i in range(6)]
    tracks_a = [f"trk_real_{i}" for i in range(6)]
    contents_a = [f"hash_real_{i}" for i in range(6)]

    ids_c = [f"dir_c/real_{i}.jpg#0" for i in range(6)]
    cams_c = [f"cam_c_{i}" for i in range(6)]
    tracks_c = [f"trk_c_{i}" for i in range(6)]
    contents_c = [f"hash_c_{i}" for i in range(6)]

    ids_b = [f"dir_b_{i % 3}/fake_{i}.jpg#0" for i in range(10)]
    # Unit 1: 4 frames in same (cam_b0, trk_b0) with distinct contents
    # Unit 2: 3 byte-identical copies across 3 cameras/directories sharing hash_alias
    # Unit 3: 3 rows linked transitively via track then content
    cams_b = ["cam_b0"] * 4 + ["cam_b1", "cam_b2", "cam_b3"] + ["cam_b4", "cam_b4", "cam_b5"]
    tracks_b = ["trk_b0"] * 4 + ["trk_b1", "trk_b2", "trk_b3"] + ["trk_b4", "trk_b4", "trk_b5"]
    contents_b = (
        [f"hash_b0_{i}" for i in range(4)]
        + ["hash_alias", "hash_alias", "hash_alias"]
        + ["hash_t0", "hash_t1", "hash_t1"]
    )

    mixed_res = cluster_rejections(
        features,
        ids_a + ids_b + ids_c,
        cams_a + cams_b + cams_c,
        tracks_a + tracks_b + tracks_c,
        contents_a + contents_b + contents_c,
        model_fingerprint="fp",
        scope_id="mixed",
    )

    assert {mixed_res[oid]["candidate_number"] for oid in ids_a} == {1}
    assert {mixed_res[oid]["candidate_number"] for oid in ids_c} == {2}
    for oid in ids_a + ids_c:
        assert mixed_res[oid]["independent_target_count"] == 15
        assert mixed_res[oid]["density_verified"] is True
        assert mixed_res[oid]["status"] == "clustered"
        assert mixed_res[oid]["event_count"] == 6

    for oid in ids_b:
        assert mixed_res[oid]["independent_target_count"] == 15
        assert mixed_res[oid]["density_verified"] is False
        assert mixed_res[oid]["is_noise"] is True
        assert mixed_res[oid]["status"] == "noise"
        assert mixed_res[oid]["candidate_number"] == 0


def test_cluster_rejections_empty_validation_cancellation_and_failed_helper():
    """Verify empty inputs, invalid input errors, ClusteringCancelled, and failed_rejections."""
    assert cluster_rejections(np.empty((0, 8), dtype=np.float32), [], [], [], [], model_fingerprint="fp") == {}

    with pytest.raises(ValueError, match="finite"):
        cluster_rejections(np.array([[np.nan, 1.0]], dtype=np.float32), ["a"], ["c"], ["t"], ["h"], model_fingerprint="fp")

    with pytest.raises(ValueError, match="nonzero"):
        cluster_rejections(np.zeros((1, 4), dtype=np.float32), ["a"], ["c"], ["t"], ["h"], model_fingerprint="fp")

    with pytest.raises(ValueError, match="unique"):
        cluster_rejections(np.ones((2, 4), dtype=np.float32), ["dup", "dup"], ["c1", "c2"], ["t1", "t2"], ["h1", "h2"], model_fingerprint="fp")

    with pytest.raises(ValueError, match="align"):
        cluster_rejections(np.ones((2, 4), dtype=np.float32), ["a", "b"], ["c1"], ["t1", "t2"], ["h1", "h2"], model_fingerprint="fp")

    with pytest.raises(ClusteringCancelled):
        cluster_rejections(
            np.ones((2, 4), dtype=np.float32),
            ["a", "b"],
            ["c1", "c2"],
            ["t1", "t2"],
            ["h1", "h2"],
            model_fingerprint="fp",
            is_cancelled=lambda: True,
        )

    failed_1 = failed_rejections(["obs-b", "obs-a"], model_fingerprint="fp-1", scope_id="job-1")
    failed_2 = failed_rejections(["obs-a", "obs-b"], model_fingerprint="fp-1", scope_id="job-1")
    failed_diff_scope = failed_rejections(["obs-a", "obs-b"], model_fingerprint="fp-1", scope_id="job-2")
    assert failed_1 == failed_2
    assert failed_1["obs-a"]["status"] == "failed"
    assert failed_1["obs-a"]["density_verified"] is False
    assert failed_1["obs-a"]["is_noise"] is False
    assert failed_1["obs-a"]["candidate_number"] == 0
    assert failed_1["obs-a"]["event_count"] == 0
    assert failed_1["obs-a"]["model_fingerprint"] == "fp-1"
    assert failed_1["obs-a"]["cluster_id"] == f"{failed_1['obs-a']['run_id']}:failed"
    assert failed_1["obs-a"]["run_id"] != failed_diff_scope["obs-a"]["run_id"]


def _build_finalizer_fixture(tmp_path: Path, *, include_accepted: bool = True):
    """Create multi-directory files, collector crops, DetectionItems, and MockPredictions."""
    rng = np.random.default_rng(20261114)
    dim = 64
    pop_a = _make_population(rng, 0, 6, dim)
    pop_b = _make_population(rng, 1, 6, dim)
    pop_known = _make_population(rng, 2, 3, dim)

    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector.dinov3_classifier.model_fingerprint = "enc-sha:ckpt-sha"
    detector._dinov3_collector = collector

    base_time = datetime(2026, 11, 1, 8, 0, 0)
    items: list[DetectionItem] = []
    predictions: list[MockPrediction] = []
    rejected_obs_ids: list[str] = []
    accepted_obs_ids: list[str] = []

    def add_photo(subdir: str, name: str, hour_offset: int, crops: list[tuple[np.ndarray, bool, str, tuple[float, float, float, float]]]):
        folder = tmp_path / subdir
        folder.mkdir(parents=True, exist_ok=True)
        img_path = folder / name
        img_path.write_bytes(f"{subdir}/{name}".encode("utf-8"))
        date_taken = (base_time + timedelta(hours=hour_offset)).strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            DetectionItem(
                filename=name,
                path=str(img_path),
                file_type="image",
                date_taken=date_taken,
                detection_boxes=[{} for _ in crops],
                detection_data={"sequence_pending": True},
            )
        )
        for box_idx, (emb, accepted, species, bbox) in enumerate(crops):
            collector.record_crop(DinoV3CropRecord(str(img_path), box_idx, bbox, 0.85, emb))
            obs_id = make_dinov3_observation_id(img_path, box_idx, bbox)
            if accepted:
                accepted_obs_ids.append(obs_id)
                predictions.append(
                    MockPrediction(
                        species=species,
                        accepted=True,
                        best_known_species=species,
                        known_score=2.4,
                        threshold=1.0,
                        embedding=emb,
                        confidence=0.88,
                    )
                )
            else:
                rejected_obs_ids.append(obs_id)
                predictions.append(
                    MockPrediction(
                        species=species,
                        accepted=False,
                        best_known_species=species,
                        known_score=-0.4,
                        threshold=1.0,
                        embedding=emb,
                        confidence=0.19,
                    )
                )

    for i in range(6):
        subdir = "CamNorth/sub1" if i % 2 == 0 else "CamSouth/sub2"
        if include_accepted and i == 0:
            # Multi-box photo: box 0 rejected (pop A), box 1 accepted (known species)
            add_photo(
                subdir,
                f"rej_a_{i}.jpg",
                i,
                [
                    (pop_a[i], False, "Red fox", (10, 10, 90, 90)),
                    (pop_known[0], True, "Roe deer", (120, 20, 220, 140)),
                ],
            )
        else:
            add_photo(subdir, f"rej_a_{i}.jpg", i, [(pop_a[i], False, "Red fox", (10, 10, 90, 90))])

    for i in range(6):
        subdir = "CamNorth/sub1" if i % 2 == 1 else "CamSouth/sub2"
        add_photo(subdir, f"rej_b_{i}.jpg", 20 + i, [(pop_b[i], False, "Badger", (20, 20, 100, 100))])

    if include_accepted:
        for j in range(1, 3):
            add_photo("CamEast/sub3", f"acc_{j}.jpg", 40 + j, [(pop_known[j], True, "Wild boar", (30, 30, 110, 110))])

    detector.dinov3_classifier.classify_features.return_value = predictions
    return detector, items, rejected_obs_ids, accepted_obs_ids


def test_finalize_photo_job_clusters_only_rejected_rows_normalizes_and_reloads_json(monkeypatch, tmp_path):
    """Finalizer passes only rejected rows to real HDBSCAN, copies rejection_cluster via services._apply_detection_data, and survives JSON reload."""
    detector, items, rejected_obs_ids, accepted_obs_ids = _build_finalizer_fixture(tmp_path)
    captured_calls = []
    real_cluster = rc.cluster_rejections

    def spy_cluster(features, observation_ids, cameras, tracks, contents, **kwargs):
        captured_calls.append(
            {
                "shape": features.shape,
                "observation_ids": list(observation_ids),
                "model_fingerprint": kwargs.get("model_fingerprint"),
                "scope_id": kwargs.get("scope_id"),
            }
        )
        return real_cluster(features, observation_ids, cameras, tracks, contents, **kwargs)

    monkeypatch.setattr(rc, "cluster_rejections", spy_cluster)

    saved_batches = []
    updated = finalize_photo_job(
        detector,
        items,
        tmp_path,
        save_batch_fn=lambda batch, root: saved_batches.append(batch),
        apply_data_fn=services._apply_detection_data,
        job_id="job-scope-1",
        complete_sequence=True,
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["shape"] == (12, 64)
    assert captured_calls[0]["observation_ids"] == rejected_obs_ids
    assert not set(captured_calls[0]["observation_ids"]) & set(accepted_obs_ids)
    assert captured_calls[0]["model_fingerprint"] == "enc-sha:ckpt-sha"
    assert captured_calls[0]["scope_id"] == "job-scope-1"
    assert len(saved_batches) == 1
    assert detector._dinov3_collector.get_all_crops() == []

    rejected_clusters_by_prefix = {"rej_a_": [], "rej_b_": []}
    for item in updated:
        raw_boxes = item.detection_data["检测框"]
        norm_boxes = item.detection_boxes
        assert len(raw_boxes) == len(norm_boxes)

        reloaded_item = DetectionItem.model_validate_json(item.model_dump_json())
        for raw_box, norm_box, rel_norm_box, rel_raw_box in zip(
            raw_boxes,
            norm_boxes,
            reloaded_item.detection_boxes,
            reloaded_item.detection_data["检测框"],
        ):
            if raw_box["accepted"]:
                assert "rejection_cluster" not in raw_box
                assert "rejection_cluster" not in norm_box
                assert "rejection_cluster" not in rel_norm_box
            else:
                assert "rejection_cluster" in raw_box
                assert "rejection_cluster" in norm_box
                assert norm_box["rejection_cluster"] == raw_box["rejection_cluster"]
                assert norm_box["rejection_cluster"] is not raw_box["rejection_cluster"]
                assert rel_norm_box["rejection_cluster"] == norm_box["rejection_cluster"]
                assert rel_raw_box["rejection_cluster"] == raw_box["rejection_cluster"]
                for prefix in rejected_clusters_by_prefix:
                    if item.filename.startswith(prefix):
                        rejected_clusters_by_prefix[prefix].append(rel_norm_box["rejection_cluster"])

    # Persisted JobSummary JSON reload also retains normalized rejection_cluster on DetectionItem.
    now = services.utc_now()
    job_summary = JobSummary(
        id="job-scope-1",
        state=JobState.COMPLETED,
        input_dir=str(tmp_path),
        total=len(updated),
        processed=len(updated),
        results=updated,
        created_at=now,
        updated_at=now,
    )
    reloaded_job = JobSummary.model_validate_json(job_summary.model_dump_json())
    assert len(reloaded_job.results) == len(updated)
    assert reloaded_job.results[0].detection_boxes[0]["rejection_cluster"] == updated[0].detection_boxes[0]["rejection_cluster"]
    assert "rejection_cluster" not in reloaded_job.results[0].detection_boxes[1]

    # The actual SQLite checkpoint, reopened through a new store instance,
    # must retain the same snapshot without consulting a live clusterer.
    from system.backend.job_state_store import JobStateStore
    checkpoint_path = tmp_path / "job_state.json"
    JobStateStore(checkpoint_path).save({job_summary.id: job_summary}, {})
    restored = JobSummary.model_validate(
        JobStateStore(checkpoint_path).load()["jobs"][job_summary.id])
    assert restored.results == reloaded_job.results

    # Real HDBSCAN separated both populations into distinct density-verified clusters of 6 events.
    clusters_a = rejected_clusters_by_prefix["rej_a_"]
    clusters_b = rejected_clusters_by_prefix["rej_b_"]
    assert len(clusters_a) == 6 and len(clusters_b) == 6
    assert len({c["cluster_id"] for c in clusters_a}) == 1
    assert len({c["cluster_id"] for c in clusters_b}) == 1
    assert clusters_a[0]["cluster_id"] != clusters_b[0]["cluster_id"]
    assert {clusters_a[0]["candidate_number"], clusters_b[0]["candidate_number"]} == {1, 2}
    assert all(c["density_verified"] is True and c["status"] == "clustered" and c["event_count"] == 6 for c in clusters_a + clusters_b)


def test_finalize_photo_job_handles_clustering_failure_gracefully(monkeypatch, tmp_path):
    """Unexpected clustering failure preserves sequence classification and attaches failed metadata."""
    detector, items, _, _ = _build_finalizer_fixture(tmp_path)

    def broken_cluster(*args, **kwargs):
        raise RuntimeError("synthetic HDBSCAN crash")

    monkeypatch.setattr(rc, "cluster_rejections", broken_cluster)

    saved_batches = []
    updated = finalize_photo_job(
        detector,
        items,
        tmp_path,
        save_batch_fn=lambda batch, root: saved_batches.append(batch),
        apply_data_fn=services._apply_detection_data,
        job_id="job-fail-1",
        complete_sequence=True,
    )

    assert len(saved_batches) == 1
    assert detector._dinov3_collector.get_all_crops() == []

    # Multi-box photo 0 keeps accepted box intact and gives rejected box failed cluster metadata.
    reloaded = DetectionItem.model_validate_json(updated[0].model_dump_json())
    rej_box, acc_box = reloaded.detection_boxes[0], reloaded.detection_boxes[1]
    assert acc_box["species"] == "Roe deer"
    assert "rejection_cluster" not in acc_box
    assert rej_box["species"] == "Unknown"
    failed_meta = rej_box["rejection_cluster"]
    assert failed_meta["status"] == "failed"
    assert failed_meta["density_verified"] is False
    assert failed_meta["is_noise"] is False
    assert failed_meta["candidate_number"] == 0
    assert failed_meta["event_count"] == 0
    assert failed_meta["model_fingerprint"] == "enc-sha:ckpt-sha"
    assert failed_meta["cluster_id"] == f"{failed_meta['run_id']}:failed"


def test_finalize_photo_job_cancellation_during_clustering_skips_save(monkeypatch, tmp_path):
    """Cancellation during or right after clustering does not save batches or mutate job state."""
    detector, items, _, _ = _build_finalizer_fixture(tmp_path, include_accepted=False)
    cancelled = {"flag": False}
    real_cluster = rc.cluster_rejections

    def cancel_inside_cluster(features, observation_ids, cameras, tracks, contents, **kwargs):
        cancelled["flag"] = True
        return real_cluster(features, observation_ids, cameras, tracks, contents, **kwargs)

    monkeypatch.setattr(rc, "cluster_rejections", cancel_inside_cluster)

    saved_batches = []
    mutated = []
    returned = finalize_photo_job(
        detector,
        items,
        tmp_path,
        is_cancelled=lambda: cancelled["flag"],
        mutate_job=lambda **kw: mutated.append(kw),
        save_batch_fn=lambda batch, root: saved_batches.append(batch),
        apply_data_fn=services._apply_detection_data,
        job_id="job-cancel-1",
        complete_sequence=True,
    )

    assert saved_batches == []
    assert mutated == []
    assert returned is items
    assert all("rejection_cluster" not in (box or {}) for item in returned for box in item.detection_boxes)
    assert detector._dinov3_collector.get_all_crops() == []


def test_finalize_photo_job_mixed_valid_and_error_cancellation_skips_failed_save(monkeypatch, tmp_path):
    """Cancellation during clustering in a mixed valid+error job must not save failed-file payloads afterward."""
    detector, valid_items, _, _ = _build_finalizer_fixture(tmp_path, include_accepted=False)
    failed_path = tmp_path / "broken.jpg"
    failed_path.write_bytes(b"corrupt")
    failed_item = DetectionItem(
        filename=failed_path.name,
        path=str(failed_path),
        file_type="image",
        error="decode failed",
    )

    cancelled = {"flag": False}
    real_cluster = rc.cluster_rejections

    def cancel_inside_cluster(features, observation_ids, cameras, tracks, contents, **kwargs):
        cancelled["flag"] = True
        return real_cluster(features, observation_ids, cameras, tracks, contents, **kwargs)

    monkeypatch.setattr(rc, "cluster_rejections", cancel_inside_cluster)

    saved_batches = []
    mutated = []
    returned = finalize_photo_job(
        detector,
        valid_items + [failed_item],
        tmp_path,
        is_cancelled=lambda: cancelled["flag"],
        mutate_job=lambda **kw: mutated.append(kw),
        save_batch_fn=lambda batch, root: saved_batches.append(batch),
        apply_data_fn=services._apply_detection_data,
        job_id="job-mixed-cancel",
        complete_sequence=True,
    )

    assert saved_batches == []
    assert mutated == []
    assert len(returned) == len(valid_items) + 1
    assert returned[-1] is failed_item
    assert returned[-1].detection_data == {}
    assert all("rejection_cluster" not in (box or {}) for item in returned for box in item.detection_boxes)
    assert detector._dinov3_collector.get_all_crops() == []


def test_finalize_photo_job_clustering_invariant_to_root_and_experiment_full_tracks(monkeypatch, tmp_path):
    """Clustering uses full source directories regardless of selected root, and keeps per-target tracks under experiment_full."""
    rng = np.random.default_rng(2026111402)
    dim = 64
    pop_a = _make_population(rng, 0, 5, dim)
    pop_b = _make_population(rng, 1, 6, dim)

    base_time = datetime(2026, 11, 2, 9, 0, 0)
    specs = [
        # Multi-box frame in CamNorth/siteA/cam1: same 30-min event, distinct per-target tracks.
        ("CamNorth/siteA/cam1", "a_0.jpg", 0, [
            ((10, 10, 80, 80), pop_a[0], "Red fox"),
            ((320, 220, 410, 310), pop_a[1], "Red fox"),
        ]),
        # Remaining 3 targets of pop_a across nested directories under CamNorth and CamSouth.
        ("CamNorth/siteB/cam2", "a_2.jpg", 7200, [((15, 15, 90, 90), pop_a[2], "Red fox")]),
        ("CamSouth/siteA/cam1", "a_3.jpg", 10800, [((20, 20, 95, 95), pop_a[3], "Red fox")]),
        ("CamSouth/siteB/cam2", "a_4.jpg", 14400, [((25, 25, 100, 100), pop_a[4], "Red fox")]),
    ]
    for i in range(6):
        subdir = "CamNorth/siteB/cam2" if i % 2 == 0 else "CamSouth/siteB/cam2"
        specs.append((subdir, f"b_{i}.jpg", 20000 + i * 3600, [((40, 40, 120, 120), pop_b[i], "Badger")]))

    def run_for_root(root_dir: Path, *, recipe: str):
        collector = DinoV3CropCollector()
        detector = MagicMock()
        detector.dinov3_classifier = MagicMock()
        detector.dinov3_classifier.backend = "dinov3"
        detector.dinov3_classifier.model_fingerprint = "enc-fp:ckpt-fp"
        detector.dinov3_classifier.model.inference_recipe = recipe
        detector._dinov3_collector = collector

        items = []
        preds = []
        for subdir, name, sec_offset, crops in specs:
            folder = tmp_path / subdir
            folder.mkdir(parents=True, exist_ok=True)
            img_path = folder / name
            img_path.write_bytes(f"{subdir}/{name}".encode("utf-8"))
            dt = (base_time + timedelta(seconds=sec_offset)).strftime("%Y-%m-%d %H:%M:%S")
            items.append(
                DetectionItem(
                    filename=name,
                    path=str(img_path),
                    file_type="image",
                    date_taken=dt,
                    detection_boxes=[{} for _ in crops],
                    detection_data={"sequence_pending": True},
                )
            )
            for box_idx, (bbox, emb, sp) in enumerate(crops):
                collector.record_crop(DinoV3CropRecord(str(img_path), box_idx, bbox, 0.85, emb))
                preds.append(MockPrediction(sp, False, sp, -0.4, 1.0, emb, confidence=0.2))

        detector.dinov3_classifier.classify_features.return_value = preds
        cluster_calls = []
        real_cluster = rc.cluster_rejections

        def spy_cluster(features, observation_ids, cameras, tracks, contents, **kwargs):
            cluster_calls.append({"cameras": list(cameras), "tracks": list(tracks)})
            return real_cluster(features, observation_ids, cameras, tracks, contents, **kwargs)

        monkeypatch.setattr(rc, "cluster_rejections", spy_cluster)
        updated = finalize_photo_job(
            detector,
            items,
            root_dir,
            apply_data_fn=services._apply_detection_data,
            job_id="job-root-invariance",
            complete_sequence=True,
        )
        cls_kwargs = detector.dinov3_classifier.classify_features.call_args.kwargs
        return updated, cls_kwargs, cluster_calls[0]

    north_items, north_cls, north_cluster_in = run_for_root(tmp_path / "CamNorth", recipe="production_dual")
    south_items, south_cls, south_cluster_in = run_for_root(tmp_path / "CamSouth", recipe="production_dual")

    # Original classifier camera_ids still depend on selected root, while clustering inputs and metadata do not.
    assert north_cls["camera_ids"] != south_cls["camera_ids"]
    assert north_cluster_in == south_cluster_in
    assert [it.detection_boxes[0]["rejection_cluster"] for it in north_items] == [
        it.detection_boxes[0]["rejection_cluster"] for it in south_items
    ]

    # Both inference recipes keep distinct targets in separate classification
    # sequences; clustering also keeps pop_a's 5 independent target units.
    exp_items, exp_cls, exp_cluster_in = run_for_root(tmp_path, recipe="experiment_full")
    assert exp_cls["sequence_ids"][0] != exp_cls["sequence_ids"][1]
    assert exp_cls["sequence_ids"][0].startswith("event:")
    assert exp_cluster_in["tracks"][0] != exp_cluster_in["tracks"][1]

    pop_a_clusters = [box["rejection_cluster"] for it in exp_items[:4] for box in it.detection_boxes]
    pop_b_clusters = [it.detection_boxes[0]["rejection_cluster"] for it in exp_items[4:]]
    assert all(c["independent_target_count"] == 11 for c in pop_a_clusters + pop_b_clusters)
    assert all(c["density_verified"] is True and c["event_count"] == 5 for c in pop_a_clusters)
    assert all(c["density_verified"] is True and c["event_count"] == 6 for c in pop_b_clusters)
    assert pop_a_clusters[0]["cluster_id"] != pop_b_clusters[0]["cluster_id"]


@pytest.mark.parametrize("status,error_text", [
    ("partial", None),
    ("pending", None),
    ("partial", "decode error"),
])
def test_incomplete_and_failed_sequence_state_clears_stale_rejection_cluster(tmp_path, status, error_text):
    """_annotate_sequence_state / mark_dinov3_sequence_incomplete clears stale rejection_cluster when complete=False."""
    img_path = tmp_path / "stale.jpg"
    img_path.write_bytes(b"stale")
    stale_cluster = {
        "cluster_id": "old-run:1",
        "candidate_number": 1,
        "status": "clustered",
        "density_verified": True,
        "is_noise": False,
    }
    raw_box = {
        "物种": "Unknown",
        "置信度": 0.2,
        "边界框": [10.0, 10.0, 50.0, 50.0],
        "候选项": [{"name": "Unknown", "conf": 0.2, "accepted": False}],
        "accepted": False,
        "rejection_cluster": dict(stale_cluster),
    }
    item = DetectionItem(
        filename=img_path.name,
        path=str(img_path),
        file_type="image",
        error=error_text,
        detection_boxes=[{"species": "Unknown", "bbox": [10.0, 10.0, 50.0, 50.0], "rejection_cluster": dict(stale_cluster)}],
        detection_data={"检测框": [raw_box], "sequence_completed": True},
    )

    # Complete annotation retains rejection_cluster when no error forces failed status.
    kept_item, kept_data = _annotate_sequence_state(
        item,
        status=SEQUENCE_STATUS_COMPLETE,
        apply_data_fn=services._apply_detection_data,
    )
    assert "rejection_cluster" in kept_data["检测框"][0]
    assert "rejection_cluster" in kept_item.detection_boxes[0]

    # Incomplete/partial/pending/failed annotation clears stale rejection_cluster.
    cleared_items = mark_dinov3_sequence_incomplete(
        [item],
        reason="cancelled" if status == "partial" else "pending_resume",
        status=status,
        apply_data_fn=services._apply_detection_data,
    )
    cleared = cleared_items[0]
    assert cleared.detection_data["sequence_completed"] is False
    if error_text:
        assert cleared.detection_data["sequence_status"] == SEQUENCE_STATUS_FAILED
    assert "rejection_cluster" not in cleared.detection_data["检测框"][0]
    assert "rejection_cluster" not in cleared.detection_boxes[0]

    # Partial finalize_photo_job also drops stale rejection_cluster and skips cluster_rejections.
    collector = DinoV3CropCollector()
    emb = _make_population(np.random.default_rng(1), 0, 1, 16)[0]
    collector.record_crop(DinoV3CropRecord(str(img_path), 0, (10.0, 10.0, 50.0, 50.0), 0.9, emb))
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector
    detector.dinov3_classifier.classify_features.return_value = [
        MockPrediction("Fox", False, "Fox", -0.3, 1.0, emb, confidence=0.2)
    ]
    partial_items = finalize_photo_job(
        detector,
        [item],
        tmp_path,
        apply_data_fn=services._apply_detection_data,
        complete_sequence=False,
        incomplete_reason="cancelled",
    )
    assert "rejection_cluster" not in partial_items[0].detection_data["检测框"][0]
    assert "rejection_cluster" not in partial_items[0].detection_boxes[0]


def test_services_legacy_full_completion_passes_job_id_to_finalize(monkeypatch, tmp_path):
    """services_legacy._run_job passes job_id and services._apply_detection_data on full DINOv3 completion."""
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    photos = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in photos:
        path.write_bytes(b"fixture")

    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, photos))
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", lambda p: Path(p))
    monkeypatch.setattr(legacy, "_dino_manifest_payload", lambda p: dict(backend="dinov3"))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda p: DetectionItem(filename=p.name, path=str(p), file_type="jpg"))
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda *a: {})
    monkeypatch.setattr(legacy, "_persist_dino_observations", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_selected_inference_class_ids", lambda *a: None)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: type("Detector", (), {})())
    monkeypatch.setattr(legacy, "_detect_image_batch", lambda detector, batch, batch_items, *a, **kw: batch_items)

    import system.dinov3.processor as processor
    captured_kwargs = []

    def fake_finalize(detector, results, root, **kwargs):
        captured_kwargs.append(kwargs)
        return [
            item.model_copy(
                update={
                    "detection_data": {
                        **(item.detection_data or {}),
                        "sequence_completed": True,
                        "sequence_pending": False,
                        "sequence_status": "complete",
                    }
                }
            )
            for item in results
        ]

    monkeypatch.setattr(processor, "finalize_photo_job", fake_finalize)
    request = CreateJobRequest(
        input_dir=str(tmp_path),
        options=ProcessingOptions(
            enable_detection=True,
            model_path="detector.pt",
            classification_model_path="joint.neri.json",
            batch_size=1,
            video_mode="skip",
        ),
    )
    manager = services.ProcessingJobManager()
    try:
        now = services.utc_now()
        manager._jobs["job-real-id"] = JobSummary(
            id="job-real-id",
            state=JobState.QUEUED,
            input_dir=str(tmp_path),
            created_at=now,
            updated_at=now,
        )
        manager._job_requests["job-real-id"] = request
        manager._run_job("job-real-id", request, [])
        finished = manager.get_job("job-real-id")
        assert finished.state == JobState.COMPLETED, finished.error
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["job_id"] == "job-real-id"
        assert captured_kwargs[0]["complete_sequence"] is True
        assert captured_kwargs[0]["apply_data_fn"] is legacy._apply_detection_data
    finally:
        manager._executor.shutdown()
