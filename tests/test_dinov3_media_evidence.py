"""Tests for weighted mixed photo/video evidence aggregation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.experiment import (
    EXPERIMENT_RECIPE,
    event_fuse,
    pooled_evidence,
    top_fraction,
)
from system.dinov3.model import (
    PRODUCTION_RECIPE,
    DinoV3RidgeWithinModel,
    adaptive_sequence_fuse,
    resolve_target_tracks,
    sequence_mean,
    top75,
)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


@lru_cache(maxsize=8)
def _make_fitted_calibrated_model(
    *,
    recipe: str,
    dim: int = 16,
    seed: int = 20261116,
) -> tuple[DinoV3RidgeWithinModel, dict[str, np.ndarray]]:
    with threadpool_limits(limits=1):
        return _build_fitted_calibrated_model(recipe=recipe, dim=dim, seed=seed)


def _build_fitted_calibrated_model(
    *,
    recipe: str,
    dim: int = 16,
    seed: int = 20261116,
) -> tuple[DinoV3RidgeWithinModel, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = ("sp_A", "sp_B", "sp_C")
    centers = {}
    for idx, cls_name in enumerate(classes):
        vec = np.zeros(dim, dtype=np.float32)
        vec[idx * 2] = 1.0
        vec[idx * 2 + 1] = 0.25
        centers[cls_name] = _unit_rows(vec[None, :])[0]

    train_x, train_y, train_cam, train_seq = [], [], [], []
    for cls_name in classes:
        for i in range(12):
            noise = 0.04 * rng.standard_normal(dim).astype(np.float32)
            train_x.append(_unit_rows((centers[cls_name] + noise)[None, :])[0])
            train_y.append(cls_name)
            train_cam.append(f"cam_{i % 3}")
            train_seq.append(f"seq_{cls_name}_{i // 3}")

    cal_x, cal_y, cal_cam, cal_seq, cal_contents = [], [], [], [], []
    for cls_name in classes:
        for i in range(8):
            noise = 0.04 * rng.standard_normal(dim).astype(np.float32)
            cal_x.append(_unit_rows((centers[cls_name] + noise)[None, :])[0])
            cal_y.append(cls_name)
            cal_cam.append(f"cal_cam_{i % 2}")
            cal_seq.append(f"cal_seq_{cls_name}_{i // 2}")
            cal_contents.append(f"cal_{cls_name}_{i}:0")

    unk_center = np.zeros(dim, dtype=np.float32)
    unk_center[-1] = 1.0
    for i in range(10):
        noise = 0.05 * rng.standard_normal(dim).astype(np.float32)
        cal_x.append(_unit_rows((unk_center + noise)[None, :])[0])
        cal_y.append(f"unk_{i % 2}")
        cal_cam.append(f"cal_cam_{i % 2}")
        cal_seq.append(f"cal_seq_unk_{i // 2}")
        cal_contents.append(f"cal_unk_{i}:0")

    model = DinoV3RidgeWithinModel(ridge=0.01, inference_recipe=recipe)
    model.fit(
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y),
        np.asarray(train_cam),
        np.asarray(train_seq),
    )
    model.calibrate(
        np.asarray(cal_x, dtype=np.float32),
        np.asarray(cal_y),
        np.asarray(cal_cam),
        np.asarray(cal_seq),
        contents=np.asarray(cal_contents),
        target_recall=0.90,
    )
    return model, centers


class TestFocusedWeightedArithmetic:
    def test_weighted_top75_evidence_mass_and_fractional_boundary(self):
        values = np.array([10.0, 2.0, 1.0, 100.0], dtype=np.float64)
        cameras = np.array(["c1", "c1", "c1", "c2"])
        sequences = np.array(["s1", "s1", "s1", "s2"])
        # Group (c1, s1) total mass = 2.0, top 75% mass = 1.5 -> 1.0*10.0 + 0.5*2.0
        weights = np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64)

        actual = top75(values, cameras, sequences, weights=weights)
        expected_s1 = (1.0 * 10.0 + 0.5 * 2.0) / 1.5
        np.testing.assert_allclose(actual[:3], expected_s1, atol=1e-12)
        np.testing.assert_allclose(actual[3], 100.0, atol=1e-12)

    def test_weighted_top_fraction_experiment_matches_mass_definition(self):
        values = np.array([8.0, 4.0, 0.0], dtype=np.float64)
        cameras = np.array(["c1", "c1", "c1"])
        sequences = np.array(["s1", "s1", "s1"])
        # Total mass = 2.0, top 75% mass = 1.5 -> 0.5*8.0 + 1.0*4.0
        weights = np.array([0.5, 1.0, 0.5], dtype=np.float64)

        actual = top_fraction(values, cameras, sequences, weights=weights)
        expected = (0.5 * 8.0 + 1.0 * 4.0) / 1.5
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_weighted_sequence_mean_arithmetic(self):
        values = np.array([1.0, 5.0, 9.0], dtype=np.float64)
        cameras = np.array(["c1", "c1", "c2"])
        sequences = np.array(["s1", "s1", "s2"])
        weights = np.array([1.0, 3.0, 0.2], dtype=np.float64)

        actual = sequence_mean(values, cameras, sequences, weights=weights)
        np.testing.assert_allclose(actual[:2], (1.0 * 1.0 + 3.0 * 5.0) / 4.0, atol=1e-12)
        np.testing.assert_allclose(actual[2], 9.0, atol=1e-12)

    def test_weighted_repeat_content_collapsing_and_effective_count(self):
        # Content c1 has two rows with unequal weights 0.75 and 0.25 -> collapsed [7.5, 2.5], weight 1.0
        # Content c2 has one row with weight 1.0 -> [0.0, 10.0], weight 1.0
        scores = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
        sites = np.array(["site1", "site1", "site1"])
        sequences = np.array(["seq1", "seq1", "seq1"])
        contents = np.array(["c1", "c1", "c2"])
        weights = np.array([0.75, 0.25, 1.0], dtype=np.float64)

        fused = adaptive_sequence_fuse(
            scores, sites, sequences, contents, tau=1.0, consensus_boost=True, weights=weights
        )
        # c1 votes class 0 (weight 1.0), c2 votes class 1 (weight 1.0) -> no consensus boost
        # n_eff = 2.0, w_seq = 2 / (2 + 1) = 2/3
        group_score = 0.5 * np.array([7.5, 2.5]) + 0.5 * np.array([0.0, 10.0])
        w_seq = 2.0 / 3.0
        expected = (1.0 - w_seq) * scores + w_seq * group_score
        np.testing.assert_allclose(fused, expected, atol=1e-12)

    def test_weighted_consensus_boost_and_prevent_repeat_count_inflation(self):
        # 1 photo (class 0, w=1.0) + 10 video frames (class 1, w=0.1 each):
        # weighted consensus is 1.0 vs 1.0 (50/50), so no consensus boost and equal group mean.
        scores = np.vstack([np.array([[10.0, 0.0]]), np.tile([0.0, 10.0], (10, 1))])
        sites = np.array(["s1"] * 11)
        sequences = np.array(["seq1"] * 11)
        contents = np.array(["photo:0"] + [f"vid#{i}:0" for i in range(10)])
        weights = np.array([1.0] + [0.1] * 10, dtype=np.float64)

        fused_prod = adaptive_sequence_fuse(
            scores, sites, sequences, contents, tau=1.0, consensus_boost=True, weights=weights
        )
        fused_exp = event_fuse(scores, sites, sequences, contents, weights=weights)

        group_mean = np.array([5.0, 5.0])
        w_prod = 2.0 / (2.0 + 1.0)
        w_exp = 2.0 / (2.0 + 0.5)
        np.testing.assert_allclose(fused_prod, (1.0 - w_prod) * scores + w_prod * group_mean, atol=1e-12)
        np.testing.assert_allclose(fused_exp, (1.0 - w_exp) * scores + w_exp * group_mean, atol=1e-12)

        # Single video of 10 frames (all class 0, w=0.1 each, total=1.0) must NOT inflate n_eff > 1
        # or trigger consensus_boost on itself; fused score for row 0 vs row 1 stays weighted properly.
        vid_scores = np.vstack([np.tile([10.0, 0.0], (8, 1)), np.tile([0.0, 10.0], (2, 1))])
        vid_sites = np.array(["s1"] * 10)
        vid_seqs = np.array(["seq1"] * 10)
        vid_contents = np.array([f"vid#{i}:0" for i in range(10)])
        vid_weights = np.array([0.1] * 10, dtype=np.float64)

        vid_fused = adaptive_sequence_fuse(
            vid_scores, vid_sites, vid_seqs, vid_contents, tau=1.0, consensus_boost=True, weights=vid_weights
        )
        # n_eff = 1.0 < 2.0 -> w_seq = 1 / (1 + 1) = 0.5 (not boosted to 0.80)
        vid_group_mean = np.array([8.0, 2.0])
        np.testing.assert_allclose(vid_fused, 0.5 * vid_scores + 0.5 * vid_group_mean, atol=1e-12)

        # Repeated photo content with weight 1.0 each must not inflate n_eff above len(rows)
        rep_scores = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        rep_fused = adaptive_sequence_fuse(
            rep_scores,
            np.array(["s1", "s1"]),
            np.array(["seq1", "seq1"]),
            np.array(["same_photo:0", "same_photo:0"]),
            tau=1.0,
            consensus_boost=True,
            weights=np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(rep_fused, rep_scores, atol=1e-12)
        # Also verify effective count <= unique content count when distinct scores share repeated content + one other
        rep_mixed_scores = np.array([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        rep_no_boost = adaptive_sequence_fuse(
            rep_mixed_scores,
            np.array(["s1"] * 3),
            np.array(["seq1"] * 3),
            np.array(["same_photo:0", "same_photo:0", "other_photo:0"]),
            tau=1.0,
            consensus_boost=False,
            weights=np.array([1.0, 1.0, 1.0]),
        )
        # n_eff capped at len(rows) = 2 -> w_seq = 2 / 3 (not 3 / 4)
        expected_w = 2.0 / 3.0
        rep_group_mean = (2.0 * np.array([10.0, 0.0]) + 1.0 * np.array([0.0, 10.0])) / 3.0
        np.testing.assert_allclose(
            rep_no_boost, (1.0 - expected_w) * rep_mixed_scores + expected_w * rep_group_mean, atol=1e-12
        )

    def test_weighted_resolve_target_tracks_honors_weights_for_multibox(self):
        a0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        a1 = np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32)
        b1 = np.array([0.14, 0.99, 0.0, 0.0], dtype=np.float32)
        feats = np.stack([a0, b0, a1, b1])
        cams = ["c"] * 4
        seqs = ["e"] * 4
        contents = ["f0:0", "f0:1", "f1:0", "f1:1"]
        tracks = resolve_target_tracks(
            feats, cams, seqs, contents, weights=np.array([1.0, 1.0, 0.5, 0.5])
        )
        assert tracks[0] == tracks[2]
        assert tracks[1] == tracks[3]
        assert tracks[0] != tracks[1]

    @pytest.mark.parametrize("n_rep", [3, 5, 7, 11])
    def test_weighted_consensus_boost_and_row_order_invariance_for_odd_n_splits(self, n_rep):
        photo_score = np.array([10.0, 0.0], dtype=np.float64)
        video_score = np.array([6.0, 4.0], dtype=np.float64)
        group_mean = 0.5 * photo_score + 0.5 * video_score
        expected_photo = 0.2 * photo_score + 0.8 * group_mean
        expected_video = 0.2 * video_score + 0.8 * group_mean

        scores_fwd = np.vstack([photo_score[None, :], np.tile(video_score, (n_rep, 1))])
        weights_fwd = np.array([1.0] + [1.0 / n_rep] * n_rep, dtype=np.float64)
        contents_fwd = np.array(["photo0:0"] + [f"vid#{i}:0" for i in range(n_rep)])
        sites = np.array(["s1"] * (1 + n_rep))
        seqs = np.array(["seq1"] * (1 + n_rep))

        # Reversed and interleaved permutations of the exact same rows
        perm_rev = np.arange(1 + n_rep)[::-1]
        mid = n_rep // 2
        perm_mid = np.array(
            list(range(1, 1 + mid)) + [0] + list(range(1 + mid, 1 + n_rep)),
            dtype=int,
        )

        for fuse_fn in (adaptive_sequence_fuse, event_fuse):
            out_fwd = fuse_fn(scores_fwd, sites, seqs, contents_fwd, weights=weights_fwd)
            np.testing.assert_allclose(out_fwd[0], expected_photo, atol=1e-12)
            np.testing.assert_allclose(out_fwd[1:], np.tile(expected_video, (n_rep, 1)), atol=1e-12)

            for perm in (perm_rev, perm_mid):
                out_perm = fuse_fn(
                    scores_fwd[perm], sites, seqs, contents_fwd[perm], weights=weights_fwd[perm]
                )
                np.testing.assert_allclose(out_perm, out_fwd[perm], atol=1e-12)


class TestNoWeightsParity:
    @pytest.mark.parametrize("recipe", [PRODUCTION_RECIPE, EXPERIMENT_RECIPE])
    def test_none_weights_bitwise_parity_model_and_classifier(self, recipe):
        model, centers = _make_fitted_calibrated_model(recipe=recipe, dim=768, seed=101 + (recipe == EXPERIMENT_RECIPE))
        rng = np.random.default_rng(7)
        features = _unit_rows(
            np.stack([
                centers["sp_A"] + 0.03 * rng.standard_normal(768),
                centers["sp_A"] + 0.03 * rng.standard_normal(768),
                centers["sp_B"] + 0.03 * rng.standard_normal(768),
            ])
        )
        cameras = ["cam_0"] * 3
        sequences = ["seq_0"] * 3
        sites = ["site_0"] * 3
        contents = ["img0:0", "img1:0", "img2:0"]

        baseline = model.predict(features, cameras, sequences, sites=sites, contents=contents)
        explicit_none = model.predict(
            features, cameras, sequences, sites=sites, contents=contents, evidence_weights=None
        )
        for key in baseline:
            np.testing.assert_array_equal(explicit_none[key], baseline[key])

        clf = DinoV3Classifier(model)
        clf_base = clf.classify_features(
            features, camera_ids=cameras, sequence_ids=sequences, content_ids=contents, site_ids=sites
        )
        clf_none = clf.classify_features(
            features,
            camera_ids=cameras,
            sequence_ids=sequences,
            content_ids=contents,
            site_ids=sites,
            evidence_weights=None,
        )
        for p1, p2 in zip(clf_base, clf_none):
            assert p1.species == p2.species
            assert p1.accepted == p2.accepted
            assert p1.known_score == p2.known_score
            assert p1.ridge_score == p2.ridge_score
            assert p1.within_score == p2.within_score
            assert p1.sequence_known_score == p2.sequence_known_score
            assert p1.box_known_score == p2.box_known_score

    def test_top75_and_top_fraction_preserve_float32_legacy_when_none(self):
        values = np.asarray([0.75104207, 0.751042, 0.7510421, -1.0], dtype=np.float32)
        cameras = ["c"] * 4
        sequences = ["e"] * 4
        expected_exp = np.full(4, np.sort(values)[::-1][:3].mean())
        np.testing.assert_array_equal(top_fraction(values, cameras, sequences, weights=None), expected_exp)
        expected_prod = np.full(4, np.sort(values.astype(np.float64))[-3:].mean())
        np.testing.assert_array_equal(top75(values, cameras, sequences, weights=None), expected_prod)

    def test_pooled_evidence_none_parity(self):
        within = np.array([[1.0, 0.2, 0.1], [0.9, 0.3, 0.0], [0.8, 0.1, 0.2]], dtype=np.float64)
        norm = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        cams = np.array(["c"] * 3)
        seqs = np.array(["e"] * 3)
        m0, p0, r0 = pooled_evidence(within, norm, cams, seqs)
        m1, p1, r1 = pooled_evidence(within, norm, cams, seqs, weights=None)
        np.testing.assert_array_equal(m1, m0)
        np.testing.assert_array_equal(p1, p0)
        np.testing.assert_array_equal(r1, r0)


class TestCalibratedRecipesAndDuplicateInvariance:
    @pytest.mark.parametrize("recipe", [PRODUCTION_RECIPE, EXPERIMENT_RECIPE])
    @pytest.mark.parametrize("duplicate_mode", ["distinct_frames", "identical_content"])
    def test_mixed_photo_video_duplicate_invariance_both_recipes(self, recipe, duplicate_mode):
        model, centers = _make_fitted_calibrated_model(recipe=recipe, dim=768, seed=404 + (recipe == EXPERIMENT_RECIPE))
        thresh_before = model.threshold
        box_thresh_before = model.box_threshold

        rng = np.random.default_rng(99)
        photo_feat = _unit_rows((centers["sp_A"] + 0.03 * rng.standard_normal(768))[None, :])[0]
        # Ambiguous / slightly noisy sp_B video feature
        video_feat = _unit_rows((centers["sp_B"] + 0.06 * rng.standard_normal(768))[None, :])[0]

        # Base mixed sequence: 1 photo (weight=1.0) + 1 video frame (weight=1.0)
        base_feats = np.stack([photo_feat, video_feat], axis=0)
        base_cams = ["cam1", "cam1"]
        base_seqs = ["ev1", "ev1"]
        base_sites = ["site1", "site1"]
        base_contents = ["photo0:0", "vid#0:0"]
        base_weights = np.array([1.0, 1.0], dtype=np.float64)

        # Replicated mixed sequence: 1 photo (weight=1.0) + 12 identical video frames (weights=1/12 each)
        n_rep = 12
        rep_feats = np.vstack([photo_feat[None, :], np.tile(video_feat, (n_rep, 1))])
        rep_cams = ["cam1"] * (1 + n_rep)
        rep_seqs = ["ev1"] * (1 + n_rep)
        rep_sites = ["site1"] * (1 + n_rep)
        if duplicate_mode == "distinct_frames":
            rep_contents = ["photo0:0"] + [f"vid#{i}:0" for i in range(n_rep)]
        else:
            rep_contents = ["photo0:0"] + ["vid#0:0"] * n_rep
        rep_weights = np.array([1.0] + [1.0 / n_rep] * n_rep, dtype=np.float64)

        base_out = model.predict(
            base_feats, base_cams, base_seqs, sites=base_sites, contents=base_contents, evidence_weights=base_weights
        )
        rep_out = model.predict(
            rep_feats, rep_cams, rep_seqs, sites=rep_sites, contents=rep_contents, evidence_weights=rep_weights
        )

        # Thresholds remain unchanged
        assert model.threshold == thresh_before
        assert model.box_threshold == box_thresh_before

        # Photo row and video row decisions/scores remain invariant under video frame replication
        assert rep_out["predictions"][0] == base_out["predictions"][0]
        assert np.all(rep_out["predictions"][1:] == base_out["predictions"][1])
        assert bool(rep_out["accepted"][0]) == bool(base_out["accepted"][0])
        assert np.all(rep_out["accepted"][1:] == base_out["accepted"][1])
        np.testing.assert_allclose(rep_out["fused_scores"][0], base_out["fused_scores"][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            rep_out["fused_scores"][1:], np.tile(base_out["fused_scores"][1], (n_rep, 1)), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(rep_out["known_score"][0], base_out["known_score"][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(rep_out["known_score"][1:], base_out["known_score"][1], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            rep_out["sequence_known_score"][0], base_out["sequence_known_score"][0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rep_out["sequence_known_score"][1:], base_out["sequence_known_score"][1], rtol=1e-5, atol=1e-5
        )

        # Classifier end-to-end check on replicated mixed sequence
        clf = DinoV3Classifier(model)
        base_preds = clf.classify_features(
            base_feats,
            camera_ids=base_cams,
            sequence_ids=base_seqs,
            content_ids=base_contents,
            site_ids=base_sites,
            evidence_weights=base_weights,
        )
        rep_preds = clf.classify_features(
            rep_feats,
            camera_ids=rep_cams,
            sequence_ids=rep_seqs,
            content_ids=rep_contents,
            site_ids=rep_sites,
            evidence_weights=rep_weights,
        )
        assert rep_preds[0].species == base_preds[0].species
        assert rep_preds[0].accepted == base_preds[0].accepted
        assert np.isclose(rep_preds[0].known_score, base_preds[0].known_score, rtol=1e-5, atol=1e-5)
        assert all(p.species == base_preds[1].species for p in rep_preds[1:])
        assert all(p.accepted == base_preds[1].accepted for p in rep_preds[1:])

    @pytest.mark.parametrize("recipe", [PRODUCTION_RECIPE, EXPERIMENT_RECIPE])
    def test_dense_video_frames_do_not_overwhelm_photo_when_weighted(self, recipe):
        model, centers = _make_fitted_calibrated_model(recipe=recipe, dim=768, seed=707 + (recipe == EXPERIMENT_RECIPE))
        # Strong sp_A photo + 20 weak/uncertain sp_B video frames
        photo_feat = centers["sp_A"]
        video_feat = _unit_rows((0.65 * centers["sp_B"] + 0.35 * centers["sp_A"])[None, :])[0]

        n_frames = 20
        feats = np.vstack([photo_feat[None, :], np.tile(video_feat, (n_frames, 1))])
        cams = ["cam1"] * (1 + n_frames)
        seqs = ["ev1"] * (1 + n_frames)
        sites = ["site1"] * (1 + n_frames)
        contents = ["photo0:0"] + [f"vid#{i}:0" for i in range(n_frames)]

        unweighted = model.predict(feats, cams, seqs, sites=sites, contents=contents)
        weighted = model.predict(
            feats,
            cams,
            seqs,
            sites=sites,
            contents=contents,
            evidence_weights=np.array([1.0] + [1.0 / n_frames] * n_frames),
        )
        # Unweighted lets 20 video frames drag the photo's fused sp_A advantage down much more than weighted 1:1
        sp_a_idx = int(np.flatnonzero(model.classes_ == "sp_A")[0])
        sp_b_idx = int(np.flatnonzero(model.classes_ == "sp_B")[0])
        unweighted_gap = unweighted["fused_scores"][0, sp_a_idx] - unweighted["fused_scores"][0, sp_b_idx]
        weighted_gap = weighted["fused_scores"][0, sp_a_idx] - weighted["fused_scores"][0, sp_b_idx]
        assert weighted_gap > unweighted_gap

        # Furthermore, weighted matches 1 photo + 1 video frame with [1.0, 1.0]
        ref = model.predict(
            np.stack([photo_feat, video_feat]),
            ["cam1", "cam1"],
            ["ev1", "ev1"],
            sites=["site1", "site1"],
            contents=["photo0:0", "vid#0:0"],
            evidence_weights=[1.0, 1.0],
        )
        np.testing.assert_allclose(weighted["fused_scores"][0], ref["fused_scores"][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(weighted["known_score"][0], ref["known_score"][0], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("recipe", [PRODUCTION_RECIPE, EXPERIMENT_RECIPE])
    @pytest.mark.parametrize("n_rep", [3, 5, 7, 11])
    def test_consensus_replication_and_row_order_invariance_both_recipes(self, recipe, n_rep):
        model, centers = _make_fitted_calibrated_model(
            recipe=recipe, dim=768, seed=404 + (recipe == EXPERIMENT_RECIPE)
        )
        photo_feat = centers["sp_A"]
        # Same-class video feature with distinct margin/score so consensus_boost changes fused_scores
        video_feat = _unit_rows((0.75 * centers["sp_A"] + 0.25 * centers["sp_B"])[None, :])[0]

        base_feats = np.stack([photo_feat, video_feat], axis=0)
        base_cams = ["cam1", "cam1"]
        base_seqs = ["ev1", "ev1"]
        base_sites = ["site1", "site1"]
        base_contents = ["photo0:0", "vid#0:0"]
        base_weights = np.array([1.0, 1.0], dtype=np.float64)
        base_out = model.predict(
            base_feats, base_cams, base_seqs, sites=base_sites, contents=base_contents, evidence_weights=base_weights
        )

        feats_fwd = np.vstack([photo_feat[None, :], np.tile(video_feat, (n_rep, 1))])
        cams = np.array(["cam1"] * (1 + n_rep))
        seqs = np.array(["ev1"] * (1 + n_rep))
        sites = np.array(["site1"] * (1 + n_rep))
        contents_fwd = np.array(["photo0:0"] + [f"vid#{i}:0" for i in range(n_rep)])
        weights_fwd = np.array([1.0] + [1.0 / n_rep] * n_rep, dtype=np.float64)

        perm_rev = np.arange(1 + n_rep)[::-1]
        mid = n_rep // 2
        perm_mid = np.array(
            list(range(1, 1 + mid)) + [0] + list(range(1 + mid, 1 + n_rep)),
            dtype=int,
        )

        out_fwd = model.predict(
            feats_fwd, cams, seqs, sites=sites, contents=contents_fwd, evidence_weights=weights_fwd
        )
        np.testing.assert_allclose(out_fwd["fused_scores"][0], base_out["fused_scores"][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            out_fwd["fused_scores"][1:], np.tile(base_out["fused_scores"][1], (n_rep, 1)), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(out_fwd["known_score"][0], base_out["known_score"][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_fwd["known_score"][1:], base_out["known_score"][1], rtol=1e-5, atol=1e-5)

        for perm in (perm_rev, perm_mid):
            out_perm = model.predict(
                feats_fwd[perm],
                cams[perm],
                seqs[perm],
                sites=sites[perm],
                contents=contents_fwd[perm],
                evidence_weights=weights_fwd[perm],
            )
            assert np.array_equal(out_perm["predictions"], out_fwd["predictions"][perm])
            assert np.array_equal(out_perm["accepted"], out_fwd["accepted"][perm])
            np.testing.assert_allclose(out_perm["fused_scores"], out_fwd["fused_scores"][perm], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(out_perm["known_score"], out_fwd["known_score"][perm], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                out_perm["sequence_known_score"], out_fwd["sequence_known_score"][perm], rtol=1e-5, atol=1e-5
            )


class TestInvalidWeightsAndClassifierForwarding:
    @pytest.mark.parametrize(
        "bad_weights",
        [
            [1.0],
            [1.0, 1.0, 1.0],
            [[1.0, 1.0]],
            1.0,
            [1.0, 0.0],
            [1.0, -0.5],
            [1.0, np.nan],
            [1.0, np.inf],
            [1.0, -np.inf],
            [True, True],
            ["1.0", "bad"],
        ],
    )
    def test_invalid_weights_rejected_before_predict(self, bad_weights):
        model, centers = _make_fitted_calibrated_model(recipe=PRODUCTION_RECIPE, dim=768, seed=808)
        features = np.stack([centers["sp_A"], centers["sp_B"]], axis=0)

        with pytest.raises(ValueError):
            model.predict(features, ["c", "c"], ["s", "s"], evidence_weights=bad_weights)

        called = False

        class _GuardedModel:
            classes = np.array(["sp_A", "sp_B"])
            calibration = {}
            inference_recipe = PRODUCTION_RECIPE

            def predict(self, *args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("predict must not be called with invalid weights")

        clf = DinoV3Classifier(_GuardedModel())
        with pytest.raises(ValueError):
            clf.classify_features(features, evidence_weights=bad_weights)
        assert called is False

    def test_empty_input_with_mismatched_weights_raises_value_error(self):
        model, _ = _make_fitted_calibrated_model(recipe=PRODUCTION_RECIPE, dim=16, seed=909)
        empty = np.empty((0, 16), dtype=np.float32)
        ok = model.predict(empty, [], [], evidence_weights=[])
        assert len(ok["predictions"]) == 0
        with pytest.raises(ValueError):
            model.predict(empty, [], [], evidence_weights=[1.0])

    def test_classifier_forwards_evidence_weights_only_when_given(self):
        features = _unit_rows(np.ones((2, 768), dtype=np.float32))

        class _LegacyModelNoExtraKwargs:
            classes = np.array(["sp_A", "sp_B"])
            calibration = {}
            inference_recipe = PRODUCTION_RECIPE

            def predict(self, raw, cameras, sequences, contents=None, sites=None, adapted_features=None):
                return {
                    "classes": self.classes,
                    "predictions": np.array(["sp_A", "sp_B"]),
                    "accepted": np.array([True, True]),
                    "known_score": np.array([1.0, 1.0]),
                    "threshold": 0.0,
                    "ridge_scores": np.array([[2.0, 0.5], [0.5, 2.0]]),
                    "within_scores": np.array([[0.9, 0.1], [0.1, 0.9]]),
                    "fused_scores": np.array([[2.0, 0.5], [0.5, 2.0]]),
                }

        legacy_clf = DinoV3Classifier(_LegacyModelNoExtraKwargs())
        preds = legacy_clf.classify_features(features)
        assert [p.species for p in preds] == ["sp_A", "sp_B"]

        recorded_kwargs: list[dict] = []

        class _SpyModel(_LegacyModelNoExtraKwargs):
            def predict(self, raw, cameras, sequences, **kwargs):
                recorded_kwargs.append(dict(kwargs))
                return super().predict(raw, cameras, sequences, **{
                    k: v for k, v in kwargs.items() if k != "evidence_weights"
                })

        spy_clf = DinoV3Classifier(_SpyModel())
        spy_clf.classify_features(features, evidence_weights=None)
        assert "evidence_weights" not in recorded_kwargs[-1]

        spy_clf.classify_features(features, evidence_weights=[1.0, 0.5])
        assert "evidence_weights" in recorded_kwargs[-1]
        np.testing.assert_allclose(recorded_kwargs[-1]["evidence_weights"], [1.0, 0.5])
