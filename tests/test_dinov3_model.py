"""Comprehensive unit and regression tests for DinoV3RidgeWithinModel.

Covers:
1. Update vs full-refit mathematical equivalence (partial_fit vs fit)
2. Reference numerical values and faithful algorithmic components
3. Save/load roundtripping with .neri-model and .zip archives (no pickle)
4. Subsequent partial_fit on loaded model and calibration invalidation
5. Rejection gate calibration and recalibrate API
6. Empty and invalid inputs validation
7. Transactional atomicity on partial_fit failures
8. Distinction between cameras (gate aggregation) and sites (classification fusion)
"""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

from system.dinov3 import (
    DinoV3RidgeWithinModel,
    WithinReferenceUpdater,
    adaptive_sequence_fuse,
    fit_unknown_threshold,
    resolve_target_tracks,
    score_cpu,
    sequence_mean,
    standardize,
    top75,
    unit,
)


def make_synthetic_dataset(seed=20261107, n_per_class=12, dim=16, classes=("sp_A", "sp_B", "sp_C")):
    rng = np.random.default_rng(seed)
    num_classes = len(classes)
    features = []
    labels = []
    cameras = []
    sequences = []
    sites = []
    contents = []

    for c_idx, cls_name in enumerate(classes):
        center = rng.standard_normal(dim)
        center /= np.linalg.norm(center)
        for i in range(n_per_class):
            noise = 0.2 * rng.standard_normal(dim)
            feat = center + noise
            features.append(feat)
            labels.append(cls_name)
            cam_id = f"cam_{i % 3}"
            seq_id = f"seq_{cls_name}_{i // 4}"
            site_id = f"site_{i % 2}"
            content_id = f"content_{cls_name}_{i // 2}"
            cameras.append(cam_id)
            sequences.append(seq_id)
            sites.append(site_id)
            contents.append(content_id)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=str),
        np.asarray(cameras, dtype=str),
        np.asarray(sequences, dtype=str),
        np.asarray(sites, dtype=str),
        np.asarray(contents, dtype=str),
    )


class TestEquivalenceAndUpdates:
    """Test mathematical equivalence of incremental updates vs full re-fitting."""

    def test_partial_fit_equivalence_old_and_new_species(self):
        dim = 16
        raw_init, y_init, cam_init, seq_init, _, _ = make_synthetic_dataset(
            seed=101, n_per_class=15, dim=dim, classes=("sp_A", "sp_B", "sp_C")
        )

        # Fixed geometry so we compare Ridge moments and Within references cleanly
        rng = np.random.default_rng(42)
        origin = rng.standard_normal(dim)
        random_mat = rng.standard_normal((dim, dim))
        transform = np.linalg.cholesky(np.eye(dim) + 0.1 * (random_mat @ random_mat.T))

        model_incremental = DinoV3RidgeWithinModel(ridge=0.01, weighting="hierarchical")
        model_incremental.fit(
            raw_init, y_init, cam_init, seq_init, origin=origin, transform=transform
        )

        # Supplements: additional samples for old species A and B, plus brand-new species D
        raw_supp, y_supp, cam_supp, seq_supp, _, _ = make_synthetic_dataset(
            seed=102, n_per_class=10, dim=dim, classes=("sp_A", "sp_B", "sp_D")
        )

        model_incremental.partial_fit(raw_supp, y_supp, cam_supp, seq_supp)

        # Full refit from scratch with combined data and exact same geometry
        raw_full = np.concatenate([raw_init, raw_supp], axis=0)
        y_full = np.concatenate([y_init, y_supp], axis=0)
        cam_full = np.concatenate([cam_init, cam_supp], axis=0)
        seq_full = np.concatenate([seq_init, seq_supp], axis=0)

        model_full = DinoV3RidgeWithinModel(ridge=0.01, weighting="hierarchical")
        model_full.fit(
            raw_full, y_full, cam_full, seq_full, origin=origin, transform=transform
        )

        # Check classes alignment
        assert np.array_equal(model_incremental.classes_, model_full.classes_)

        # Check Ridge parameters
        np.testing.assert_allclose(
            model_incremental.ridge_head.coef_,
            model_full.ridge_head.coef_,
            atol=1e-10,
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            model_incremental.ridge_head.intercept_,
            model_full.ridge_head.intercept_,
            atol=1e-10,
            rtol=1e-9,
        )

        # Check Within bank
        b_inc, l_inc, c_inc = model_incremental.within_updater.bank()
        b_full, l_full, c_full = model_full.within_updater.bank()
        assert np.array_equal(l_inc, l_full)
        assert np.array_equal(c_inc, c_full)
        np.testing.assert_allclose(b_inc, b_full, atol=1e-10, rtol=1e-9)

        # Check scoring
        test_feats = rng.standard_normal((20, dim))
        np.testing.assert_allclose(
            model_incremental.scores(test_feats),
            model_full.scores(test_feats),
            atol=1e-10,
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            model_incremental._compute_within_scores(test_feats),
            model_full._compute_within_scores(test_feats),
            atol=1e-10,
            rtol=1e-9,
        )


class TestNumericalReferenceComponents:
    """Verify algorithmic subcomponents match reference specifications."""

    def test_unit_normalization(self):
        vec = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
        res = unit(vec)
        np.testing.assert_allclose(np.linalg.norm(res[0]), 1.0, atol=1e-9)
        assert np.isfinite(res).all()

    def test_top75_margin_computation(self):
        # 4 items in sequence: top 75% -> ceil(0.75 * 4) = 3 items
        values = np.array([1.0, 5.0, 3.0, 2.0, 10.0])
        cameras = np.array(["c1", "c1", "c1", "c1", "c2"])
        sequences = np.array(["s1", "s1", "s1", "s1", "s2"])
        out = top75(values, cameras, sequences)
        # For c1-s1: sorted is [1, 2, 3, 5], top 3 are [2, 3, 5], mean is 10/3
        np.testing.assert_allclose(out[:4], 10.0 / 3.0, atol=1e-9)
        # For c2-s2: single item -> 10.0
        np.testing.assert_allclose(out[4], 10.0, atol=1e-9)

    def test_sequence_mean_computation(self):
        values = np.array([2.0, 4.0, 9.0])
        cameras = np.array(["c1", "c1", "c2"])
        sequences = np.array(["s1", "s1", "s2"])
        out = sequence_mean(values, cameras, sequences)
        np.testing.assert_allclose(out, [3.0, 3.0, 9.0], atol=1e-9)

    def test_standardize_known_mad(self):
        values = np.array([10.0, 12.0, 14.0, 100.0])
        known = np.array([True, True, True, False])
        z, meta = standardize(values, known)
        assert meta["median"] == 12.0
        # deviations: |10-12|=2, |12-12|=0, |14-12|=2 -> median dev is 2.0 -> scale is 1.4826 * 2.0
        expected_scale = 1.4826 * 2.0
        np.testing.assert_allclose(meta["scale"], expected_scale, atol=1e-6)
        np.testing.assert_allclose(z[1], 0.0, atol=1e-9)

    def test_score_cpu_reference_formula(self):
        dim = 4
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        # Event bank: 2 events for sp_A (across 2 cameras), 1 event for sp_B
        bank = np.array([
            [1.0, 0.0, 0.0, 0.0],  # sp_A, cam1 -> sim=1.0
            [0.0, 1.0, 0.0, 0.0],  # sp_A, cam2 -> sim=0.0
            [0.0, 0.0, 1.0, 0.0],  # sp_B, cam1 -> sim=0.0
        ], dtype=np.float64)
        labels = np.array(["sp_A", "sp_A", "sp_B"])
        cameras = np.array(["cam1", "cam2", "cam1"])
        classes = ["sp_A", "sp_B"]

        scores = score_cpu(query, bank, labels, cameras, classes)
        # For sp_A:
        # cam1 max = 1.0, cam2 max = 0.0 -> pooled = [1.0, 0.0]
        # k=min(3, 2)=2 -> local = (1.0 + 0.0)/2 = 0.5
        # center = unit(mean([1,0,0,0], [0,1,0,0])) = [1/sqrt(2), 1/sqrt(2), 0, 0]
        # global = query @ center = 1/sqrt(2)
        # expected sp_A = 0.5 * 0.5 + 0.5 * (1/sqrt(2)) = 0.25 + 0.5 / sqrt(2)
        expected_sp_a = 0.25 + 0.5 / np.sqrt(2)
        np.testing.assert_allclose(scores[0, 0], expected_sp_a, atol=1e-6)

    def test_adaptive_sequence_fuse(self):
        scores = np.array([
            [0.8, 0.2],
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        sites = np.array(["site1", "site1", "site2"])
        sequences = np.array(["seq1", "seq1", "seq2"])
        contents = np.array(["item1", "item2", "item3"])

        fused = adaptive_sequence_fuse(scores, sites, sequences, contents, tau=1.0, consensus_boost=True)
        # seq1 has 2 contents, votes are [0, 0], consensus >= 2/3 -> w_seq boosted to 0.80
        # group_score for seq1 is mean([0.8, 0.2], [0.9, 0.1]) = [0.85, 0.15]
        # fused[0] = 0.2 * [0.8, 0.2] + 0.8 * [0.85, 0.15] = [0.16 + 0.68, 0.04 + 0.12] = [0.84, 0.16]
        np.testing.assert_allclose(fused[0], [0.84, 0.16], atol=1e-6)


class TestCalibrationAndPrediction:
    """Test calibration workflow, threshold setting, and prediction gating."""

    def test_calibration_and_predict_flow(self):
        dim = 16
        raw, y, cam, seq, sites, contents = make_synthetic_dataset(
            seed=200, n_per_class=20, dim=dim, classes=("sp_A", "sp_B", "sp_C")
        )

        model = DinoV3RidgeWithinModel(ridge=0.01)
        model.fit(raw, y, cam, seq, metadata={"encoder_sha256": "abc12345"})

        assert model.dimension == dim
        assert model.provenance["encoder_sha256"] == "abc12345"
        assert model.classes is not None
        assert not model.is_calibrated
        assert model.threshold is None
        assert model.calibration is None

        # Prediction before calibration raises RuntimeError
        with pytest.raises(RuntimeError, match="not calibrated"):
            model.predict(raw[:5], cam[:5], seq[:5])

        # Prepare calibration set containing known and unknown samples
        cal_raw_known, cal_y_known, cal_cam_k, cal_seq_k, _, _ = make_synthetic_dataset(
            seed=201, n_per_class=10, dim=dim, classes=("sp_A", "sp_B")
        )
        cal_raw_unk, cal_y_unk, cal_cam_u, cal_seq_u, _, _ = make_synthetic_dataset(
            seed=202, n_per_class=10, dim=dim, classes=("unk_X", "unk_Y")
        )

        cal_raw = np.concatenate([cal_raw_known, cal_raw_unk], axis=0)
        cal_y = np.concatenate([cal_y_known, cal_y_unk], axis=0)
        cal_cam = np.concatenate([cal_cam_k, cal_cam_u], axis=0)
        cal_seq = np.concatenate([cal_seq_k, cal_seq_u], axis=0)

        cal_info = model.calibrate(cal_raw, cal_y, cal_cam, cal_seq, target_recall=0.90)
        assert model.is_calibrated
        assert model.threshold is not None
        assert model.calibration is not None
        assert "margin" in cal_info and "radial" in cal_info

        # Now prediction succeeds
        pred_out = model.predict(
            cal_raw,
            cal_cam,
            cal_seq,
            sites=cal_cam,
            contents=np.asarray([f"c_{i}" for i in range(len(cal_raw))]),
        )
        assert "classes" in pred_out
        assert "predictions" in pred_out
        assert "accepted" in pred_out
        assert "known_score" in pred_out
        assert "threshold" in pred_out
        assert "ridge_scores" in pred_out
        assert "within_scores" in pred_out
        assert "fused_scores" in pred_out

        assert len(pred_out["predictions"]) == len(cal_raw)
        assert len(pred_out["accepted"]) == len(cal_raw)
        assert pred_out["accepted"].dtype == bool
        assert pred_out["threshold"] == model.threshold

        # Unknown rejection rate on the calibration set unknown subset meets macro target
        unk_mask = np.isin(cal_y, ("unk_X", "unk_Y"))
        rejected_unknowns = ~pred_out["accepted"][unk_mask]
        assert rejected_unknowns.mean() >= 0.85

    def test_calibration_invalidation_after_partial_fit(self):
        dim = 16
        raw, y, cam, seq, _, _ = make_synthetic_dataset(seed=300, n_per_class=15, dim=dim)
        model = DinoV3RidgeWithinModel(ridge=0.01)
        model.fit(raw, y, cam, seq)

        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=301, n_per_class=10, dim=dim, classes=("sp_A", "unk_Z")
        )
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq)
        assert model.is_calibrated

        # Partial fit with new data
        supp_raw, supp_y, supp_cam, supp_seq, _, _ = make_synthetic_dataset(
            seed=302, n_per_class=5, dim=dim, classes=("sp_A", "sp_B")
        )
        model.partial_fit(supp_raw, supp_y, supp_cam, supp_seq)

        # Must be invalidated
        assert not model.is_calibrated
        assert model.threshold is None
        assert model.calibration is None

        with pytest.raises(RuntimeError, match="not calibrated"):
            model.predict(raw[:5], cam[:5], seq[:5])

        # Recalibrate restores capability
        model.recalibrate(cal_raw, cal_y, cal_cam, cal_seq)
        assert model.is_calibrated
        res = model.predict(raw[:5], cam[:5], seq[:5])
        assert len(res["accepted"]) == 5


class TestSerialization:
    """Test save and load with .neri-model and .zip archives without pickle."""

    def test_atomic_save_load_roundtrip_both_extensions(self, tmp_path):
        dim = 16
        raw, y, cam, seq, sites, contents = make_synthetic_dataset(seed=400, n_per_class=15, dim=dim)
        model = DinoV3RidgeWithinModel(
            ridge=0.01,
            tau=1.5,
            consensus_boost=True,
            target_recall=0.90,
            metadata={"encoder_sha256": "deadbeef123", "pipeline": "test"},
        )
        model.fit(raw, y, cam, seq)

        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=401, n_per_class=10, dim=dim, classes=("sp_A", "unk_W")
        )
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq)

        pred_before = model.predict(raw[:10], cam[:10], seq[:10], sites=sites[:10], contents=contents[:10])

        for ext in (".neri-model", ".zip"):
            save_path = tmp_path / f"model_test{ext}"
            model.save(save_path)
            assert save_path.exists()

            # Verify zip archive structure and absence of pickle
            with zipfile.ZipFile(save_path, "r") as archive:
                names = archive.namelist()
                assert "manifest.json" in names
                assert "geometry.npz" in names
                assert "ridge_head.npz" in names
                assert "within_updater.npz" in names
                assert "calibration.json" in names
                for name in names:
                    assert not name.endswith(".pkl")
                    assert not name.endswith(".pickle")

                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                assert manifest["head_type"] == "ridge_within"
                assert manifest["schema_version"] == 1
                assert manifest["backend"] == "dinov3"
                assert manifest["metadata"]["encoder_sha256"] == "deadbeef123"
                assert manifest["is_calibrated"] is True

            loaded_model = DinoV3RidgeWithinModel.load(save_path)
            assert loaded_model.dimension == model.dimension
            assert loaded_model.provenance == model.provenance
            assert np.array_equal(loaded_model.classes_, model.classes_)
            assert loaded_model.is_calibrated is True
            assert loaded_model.threshold == model.threshold

            pred_after = loaded_model.predict(
                raw[:10], cam[:10], seq[:10], sites=sites[:10], contents=contents[:10]
            )
            assert np.array_equal(pred_after["predictions"], pred_before["predictions"])
            assert np.array_equal(pred_after["accepted"], pred_before["accepted"])
            np.testing.assert_allclose(pred_after["known_score"], pred_before["known_score"], atol=1e-10)
            np.testing.assert_allclose(pred_after["ridge_scores"], pred_before["ridge_scores"], atol=1e-10)
            np.testing.assert_allclose(pred_after["within_scores"], pred_before["within_scores"], atol=1e-10)
            np.testing.assert_allclose(pred_after["fused_scores"], pred_before["fused_scores"], atol=1e-10)

            # Subsequent partial_fit on loaded model works and invalidates calibration
            supp_raw, supp_y, supp_cam, supp_seq, _, _ = make_synthetic_dataset(
                seed=402, n_per_class=5, dim=dim, classes=("sp_A", "sp_C")
            )
            loaded_model.partial_fit(supp_raw, supp_y, supp_cam, supp_seq)
            assert not loaded_model.is_calibrated
            with pytest.raises(RuntimeError, match="not calibrated"):
                loaded_model.predict(raw[:5], cam[:5], seq[:5])


class TestAtomicityAndValidation:
    """Test input validation and atomicity under exceptions."""

    def test_partial_fit_failure_rollback_atomicity(self):
        dim = 16
        raw, y, cam, seq, _, _ = make_synthetic_dataset(seed=500, n_per_class=10, dim=dim)
        model = DinoV3RidgeWithinModel()
        model.fit(raw, y, cam, seq)

        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=501, n_per_class=5, dim=dim, classes=("sp_A", "unk_Z")
        )
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq)

        orig_coef = model.ridge_head.coef_.copy()
        orig_groups = deepcopy(model.within_updater.groups)
        orig_threshold = model.threshold
        orig_calibrated = model.is_calibrated

        # Introduce invalid update (NaN features)
        bad_raw = np.zeros((4, dim), dtype=np.float32)
        bad_raw[1, 2] = np.nan
        bad_y = ["sp_A", "sp_B", "sp_C", "sp_D"]
        bad_cam = ["c1", "c1", "c2", "c2"]
        bad_seq = ["s1", "s1", "s2", "s2"]

        with pytest.raises(ValueError):
            model.partial_fit(bad_raw, bad_y, bad_cam, bad_seq)

        # State must be completely restored, including calibration
        np.testing.assert_array_equal(model.ridge_head.coef_, orig_coef)
        assert model.within_updater.groups.keys() == orig_groups.keys()
        for k in orig_groups:
            assert model.within_updater.groups[k][0] == orig_groups[k][0]
            np.testing.assert_array_equal(model.within_updater.groups[k][1], orig_groups[k][1])
        assert model.threshold == orig_threshold
        assert model.is_calibrated == orig_calibrated

    def test_empty_and_illegal_inputs(self):
        model = DinoV3RidgeWithinModel()

        # Call before fit
        with pytest.raises(RuntimeError):
            model.scores(np.zeros((2, 16)))
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((2, 16)), ["c1", "c2"], ["s1", "s2"])
        with pytest.raises(RuntimeError):
            model.partial_fit(np.zeros((2, 16)), ["c1", "c2"], ["c1", "c2"], ["s1", "s2"])

        # Empty features in fit
        with pytest.raises(ValueError, match="nonempty"):
            model.fit(np.empty((0, 16)), [], [], [])

        # 1D features in fit
        with pytest.raises(ValueError, match="matrix"):
            model.fit(np.zeros(16), ["a"], ["c"], ["s"])

        # Single class in fit
        with pytest.raises(ValueError, match="distinct classes"):
            model.fit(np.zeros((4, 16)), ["single", "single", "single", "single"], ["c"]*4, ["s"]*4)

        # Valid fit
        raw, y, cam, seq, _, _ = make_synthetic_dataset(seed=600, n_per_class=10, dim=16)
        model.fit(raw, y, cam, seq)

        # Dimension mismatch in scores/predict
        with pytest.raises(ValueError, match="dimension"):
            model.scores(np.zeros((2, 32)))

        # Calibration with no unknown samples
        with pytest.raises(ValueError, match="unknown sample"):
            model.calibrate(raw, y, cam, seq, unknown_mask=np.zeros(len(raw), dtype=bool))

        # Calibration with no known samples
        with pytest.raises(ValueError, match="known sample"):
            model.calibrate(raw, y, cam, seq, unknown_mask=np.ones(len(raw), dtype=bool))

        # Calibration target out of bounds
        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=601, n_per_class=5, dim=16, classes=("sp_A", "unk_X")
        )
        with pytest.raises(ValueError, match="target_recall"):
            model.calibrate(cal_raw, cal_y, cal_cam, cal_seq, target_recall=1.5)

        # Empty prediction query when calibrated returns empty dict arrays
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq, target_recall=0.9)
        empty_res = model.predict(np.empty((0, 16)), [], [])
        assert len(empty_res["predictions"]) == 0
        assert len(empty_res["accepted"]) == 0
        assert empty_res["threshold"] == model.threshold

    def test_malformed_archive_missing_required_files(self, tmp_path):
        dim = 16
        raw, y, cam, seq, _, _ = make_synthetic_dataset(seed=550, n_per_class=10, dim=dim)
        model = DinoV3RidgeWithinModel()
        model.fit(raw, y, cam, seq)

        valid_path = tmp_path / "valid_model.zip"
        model.save(valid_path)

        # 1. Missing manifest.json
        missing_manifest_path = tmp_path / "missing_manifest.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(missing_manifest_path, "w") as dst:
            for item in src.namelist():
                if item != "manifest.json":
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="Archive is missing required model files"):
            DinoV3RidgeWithinModel.load(missing_manifest_path)

        # 2. Missing geometry.npz
        missing_geo_path = tmp_path / "missing_geo.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(missing_geo_path, "w") as dst:
            for item in src.namelist():
                if item != "geometry.npz":
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="Archive is missing required model files"):
            DinoV3RidgeWithinModel.load(missing_geo_path)

        # 3. Missing ridge_head.npz
        missing_ridge_path = tmp_path / "missing_ridge.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(missing_ridge_path, "w") as dst:
            for item in src.namelist():
                if item != "ridge_head.npz":
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="Archive is missing|Invalid archive fields"):
            DinoV3RidgeWithinModel.load(missing_ridge_path)

        # 4. Missing within_updater.npz
        missing_within_path = tmp_path / "missing_within.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(missing_within_path, "w") as dst:
            for item in src.namelist():
                if item != "within_updater.npz":
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="Archive is missing|Invalid archive fields"):
            DinoV3RidgeWithinModel.load(missing_within_path)

    def test_malformed_archive_contract_violations(self, tmp_path):
        dim = 16
        raw, y, cam, seq, _, _ = make_synthetic_dataset(seed=560, n_per_class=10, dim=dim)
        model = DinoV3RidgeWithinModel()
        model.fit(raw, y, cam, seq)
        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=561, n_per_class=5, dim=dim, classes=("sp_A", "unk_Z")
        )
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq)

        valid_path = tmp_path / "calibrated_valid.zip"
        model.save(valid_path)

        # 1. Manifest specifies is_calibrated=True but calibration.json is missing
        corrupt_cal_missing = tmp_path / "corrupt_cal_missing.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_cal_missing, "w") as dst:
            for item in src.namelist():
                if item != "calibration.json":
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="calibrated state|calibration.json"):
            DinoV3RidgeWithinModel.load(corrupt_cal_missing)

        # 2. Calibration threshold is NaN
        corrupt_thresh_nan = tmp_path / "corrupt_thresh_nan.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_thresh_nan, "w") as dst:
            for item in src.namelist():
                if item == "calibration.json":
                    cal_obj = json.loads(src.read(item).decode("utf-8"))
                    cal_obj["threshold"] = "NaN"
                    dst.writestr(item, json.dumps(cal_obj))
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="threshold"):
            DinoV3RidgeWithinModel.load(corrupt_thresh_nan)

        # 3. Calibration scale is zero or negative
        corrupt_scale_zero = tmp_path / "corrupt_scale_zero.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_scale_zero, "w") as dst:
            for item in src.namelist():
                if item == "calibration.json":
                    cal_obj = json.loads(src.read(item).decode("utf-8"))
                    cal_obj["margin"]["scale"] = 0.0
                    dst.writestr(item, json.dumps(cal_obj))
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="scale"):
            DinoV3RidgeWithinModel.load(corrupt_scale_zero)

        # 4. Manifest dimension != ridge_head feature_dim
        corrupt_dim = tmp_path / "corrupt_dim.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_dim, "w") as dst:
            for item in src.namelist():
                if item == "manifest.json":
                    man_obj = json.loads(src.read(item).decode("utf-8"))
                    man_obj["dimension"] = 32
                    dst.writestr(item, json.dumps(man_obj))
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="geometry shape|dimension|Ridge state"):
            DinoV3RidgeWithinModel.load(corrupt_dim)

        # 5. Non-finite geometry in geometry.npz
        corrupt_geo_nan = tmp_path / "corrupt_geo_nan.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_geo_nan, "w") as dst:
            for item in src.namelist():
                if item == "geometry.npz":
                    with np.load(io.BytesIO(src.read(item))) as geo:
                        bad_origin = geo["origin"].copy()
                        bad_origin[0] = np.nan
                        bio = io.BytesIO()
                        np.savez_compressed(bio, origin=bad_origin, transform=geo["transform"])
                        dst.writestr(item, bio.getvalue())
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="geometry|origin"):
            DinoV3RidgeWithinModel.load(corrupt_geo_nan)

        # 6. Within counts containing zero or negative
        corrupt_counts = tmp_path / "corrupt_counts.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_counts, "w") as dst:
            for item in src.namelist():
                if item == "within_updater.npz":
                    with np.load(io.BytesIO(src.read(item))) as u_data:
                        bad_counts = u_data["counts"].copy()
                        bad_counts[0] = 0
                        bio = io.BytesIO()
                        np.savez_compressed(bio, labels=u_data["labels"], cameras=u_data["cameras"],
                                            sequences=u_data["sequences"], counts=bad_counts, totals=u_data["totals"])
                        dst.writestr(item, bio.getvalue())
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="counts"):
            DinoV3RidgeWithinModel.load(corrupt_counts)

        # 7. Within updater classes not matching ridge head classes
        corrupt_within_cls = tmp_path / "corrupt_within_cls.zip"
        with zipfile.ZipFile(valid_path, "r") as src, zipfile.ZipFile(corrupt_within_cls, "w") as dst:
            for item in src.namelist():
                if item == "within_updater.npz":
                    with np.load(io.BytesIO(src.read(item))) as u_data:
                        bad_labels = np.array(["unmatched_class"] * len(u_data["labels"]))
                        bio = io.BytesIO()
                        np.savez_compressed(bio, labels=bad_labels, cameras=u_data["cameras"],
                                            sequences=u_data["sequences"], counts=u_data["counts"], totals=u_data["totals"])
                        dst.writestr(item, bio.getvalue())
                else:
                    dst.writestr(item, src.read(item))
        with pytest.raises(ValueError, match="classes"):
            DinoV3RidgeWithinModel.load(corrupt_within_cls)


class TestCamerasVsSitesRouting:
    """Test that cameras (gate location) and sites (classification fusion) are routed properly."""

    def test_distinct_cameras_and_sites(self):
        dim = 16
        raw, y, cam, seq, sites, contents = make_synthetic_dataset(seed=700, n_per_class=12, dim=dim)
        model = DinoV3RidgeWithinModel()
        model.fit(raw, y, cam, seq)

        cal_raw, cal_y, cal_cam, cal_seq, _, _ = make_synthetic_dataset(
            seed=701, n_per_class=6, dim=dim, classes=("sp_A", "unk_V")
        )
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq)

        # Query where sites differs from cameras
        q_raw = raw[:6]
        q_cam = cam[:6]  # Used for gate margin top75 and radial
        q_sites = np.array(["distinct_site_1"] * 6)  # Grouping for fusion
        q_seq = seq[:6]
        q_contents = np.array(["item_0", "item_1", "item_2", "item_3", "item_4", "item_5"])

        pred_distinct = model.predict(q_raw, q_cam, q_seq, sites=q_sites, contents=q_contents)
        pred_default = model.predict(q_raw, q_cam, q_seq, sites=None, contents=q_contents)

        # Ridge scores and within scores are identical
        np.testing.assert_allclose(pred_distinct["ridge_scores"], pred_default["ridge_scores"])
        np.testing.assert_allclose(pred_distinct["within_scores"], pred_default["within_scores"])
        # Gate known_score depends strictly on q_cam and q_seq, so identical
        np.testing.assert_allclose(pred_distinct["known_score"], pred_default["known_score"])
        # Fused scores may reflect the site grouping differences
        assert pred_distinct["fused_scores"].shape == (6, len(model.classes_))


class TestMultiTargetTrajectoryAndTwoStageGate:
    """Test same-photo multi-target separation, trajectory linking, and two-stage gating."""

    def test_same_photo_animal_and_background_box_do_not_share_score_or_acceptance(self):
        dim = 16
        raw, y, cam, seq, _, _ = make_synthetic_dataset(
            seed=801, n_per_class=30, dim=dim, classes=("rock_squirrel", "badger", "boar")
        )
        model = DinoV3RidgeWithinModel(ridge=0.01)
        bank_mask = np.zeros(len(raw), dtype=bool)
        for cls_name in ("rock_squirrel", "badger", "boar"):
            cls_idx = np.flatnonzero(y == cls_name)
            bank_mask[cls_idx[:20]] = True
        cal_k_mask = ~bank_mask
        model.fit(raw[bank_mask], y[bank_mask], cam[bank_mask], seq[bank_mask])

        cal_k_raw, cal_k_y, cal_k_cam, cal_k_seq = (
            raw[cal_k_mask],
            y[cal_k_mask],
            cam[cal_k_mask],
            seq[cal_k_mask],
        )
        cal_u_raw, cal_u_y, cal_u_cam, cal_u_seq, _, _ = make_synthetic_dataset(
            seed=803, n_per_class=15, dim=dim, classes=("fallen_log", "rock_bg")
        )
        cal_raw = np.concatenate([cal_k_raw, cal_u_raw], axis=0)
        cal_y = np.concatenate([cal_k_y, cal_u_y], axis=0)
        cal_cam = np.concatenate([cal_k_cam, cal_u_cam], axis=0)
        cal_seq = np.concatenate([cal_k_seq, cal_u_seq], axis=0)
        model.calibrate(cal_raw, cal_y, cal_cam, cal_seq, target_recall=0.90)

        assert model.box_threshold is not None
        assert model.threshold is not None

        # Frame IM_00052 has box 0 (rock_squirrel) and box 1 (fallen_log) in the same camera event,
        # followed by IM_00053 with box 0 (rock_squirrel) and box 1 (fallen_log).
        squirrel_1 = raw[y == "rock_squirrel"][0]
        squirrel_2 = raw[y == "rock_squirrel"][1]
        log_1 = cal_u_raw[0]
        log_2 = cal_u_raw[1]

        q_raw = np.stack([squirrel_1, log_1, squirrel_2, log_2], axis=0)
        q_cam = np.array(["cam_burst"] * 4)
        q_seq = np.array(["event:cam_burst:2026-11-01T10:00:00"] * 4)
        q_contents = np.array(["sha_00052:0", "sha_00052:1", "sha_00053:0", "sha_00053:1"])

        pred = model.predict(q_raw, q_cam, q_seq, contents=q_contents)

        # Same-photo boxes must have distinct per-box and final gate scores
        assert pred["box_known_score"][0] != pred["box_known_score"][1]
        assert pred["known_score"][0] != pred["known_score"][1]
        assert pred["sequence_known_score"][0] != pred["sequence_known_score"][1]

        # Rock squirrel boxes are accepted; fallen log boxes in the same photos are rejected
        assert bool(pred["box_accepted"][0]) is True
        assert bool(pred["accepted"][0]) is True
        assert bool(pred["box_accepted"][1]) is False
        assert bool(pred["accepted"][1]) is False
        assert bool(pred["box_accepted"][2]) is True
        assert bool(pred["accepted"][2]) is True
        assert bool(pred["box_accepted"][3]) is False
        assert bool(pred["accepted"][3]) is False

        # Verify animal trajectory score is not dragged down by the rejected fallen log boxes
        pred_animal_only = model.predict(
            np.stack([squirrel_1, squirrel_2], axis=0),
            np.array(["cam_burst", "cam_burst"]),
            np.array(["event:cam_burst:2026-11-01T10:00:00"] * 2),
            contents=np.array(["sha_00052:0", "sha_00053:0"]),
        )
        np.testing.assert_allclose(
            pred["sequence_known_score"][[0, 2]],
            pred_animal_only["sequence_known_score"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            pred["fused_scores"][[0, 2]],
            pred_animal_only["fused_scores"],
            atol=1e-10,
        )

    def test_resolve_target_tracks_links_two_animals_across_frames_without_merging_same_photo(self):
        v_a1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_a2 = np.array([0.98, 0.20, 0.0, 0.0], dtype=np.float32)
        v_b1 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        v_b2 = np.array([0.0, 0.15, 0.99, 0.0], dtype=np.float32)

        features = np.stack([v_a1, v_b1, v_a2, v_b2], axis=0)
        cameras = ["cam1"] * 4
        sequences = ["ev1"] * 4
        contents = ["frame1:0", "frame1:1", "frame2:0", "frame2:1"]
        tracks = resolve_target_tracks(features, cameras, sequences, contents)

        # Same-frame boxes never share a track
        assert tracks[0] != tracks[1]
        assert tracks[2] != tracks[3]
        # Target A links across frame1:0 and frame2:0; Target B links across frame1:1 and frame2:1
        assert tracks[0] == tracks[2]
        assert tracks[1] == tracks[3]
