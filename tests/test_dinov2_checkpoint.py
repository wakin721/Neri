from __future__ import annotations

import math

import pytest
import torch

from system.dinov2.checkpoint import (
    DINO_BACKBONE,
    DINO_FEATURE_DIM,
    CheckpointValidationError,
    validate_checkpoint,
)
from tests.dinov2_fixtures import clone_payload, make_dinov2_payload


def test_accepts_frozen_adaptive_multi_dual_margin_contract():
    checkpoint = validate_checkpoint(make_dinov2_payload())

    assert DINO_BACKBONE == "facebook/dinov2-base"
    assert DINO_FEATURE_DIM == 768
    assert checkpoint.backbone == "facebook/dinov2-base"
    assert checkpoint.classes == ("A", "B")
    assert tuple(checkpoint.feature_center.shape) == (768,)
    assert tuple(checkpoint.prototypes.shape) == (4, 768)
    assert checkpoint.prototypes_per_class == (1, 3)
    assert checkpoint.adaptive_k is True
    assert checkpoint.selection_k == 0
    assert checkpoint.k_by_class == {"A": 1, "B": 3}
    assert checkpoint.rejection.mode == "multi_dual_margin"
    assert checkpoint.rejection.cosine_threshold == pytest.approx(0.31)
    assert checkpoint.rejection.adjusted_distance_score_threshold == pytest.approx(-0.80)
    assert checkpoint.rejection.margin_weight == pytest.approx(2.0)
    assert checkpoint.rejection.class_distance_scales == pytest.approx((1.0, 1.25))
    assert len(checkpoint.fingerprint) == 64


def test_class_distance_scales_are_optional():
    checkpoint = validate_checkpoint(make_dinov2_payload(with_scales=False))
    assert checkpoint.rejection.class_distance_scales is None


def test_rejects_dinov3_feature_space_even_at_768_dimensions():
    payload = make_dinov2_payload()
    payload["backbone"] = "dinov3_vitb16"
    with pytest.raises(CheckpointValidationError, match="facebook/dinov2-base"):
        validate_checkpoint(payload)


def test_rejects_wrong_feature_dimension():
    payload = make_dinov2_payload()
    payload["feature_dim"] = 1024
    with pytest.raises(CheckpointValidationError, match="768"):
        validate_checkpoint(payload)


@pytest.mark.parametrize("field", ["feature_center", "prototypes"])
def test_rejects_non_finite_feature_tensors(field):
    payload = make_dinov2_payload()
    payload[field] = payload[field].clone()
    payload[field].view(-1)[0] = math.nan
    with pytest.raises(CheckpointValidationError, match=field):
        validate_checkpoint(payload)


def test_rejects_missing_feature_center():
    payload = make_dinov2_payload()
    payload.pop("feature_center")
    with pytest.raises(CheckpointValidationError, match="feature_center"):
        validate_checkpoint(payload)


def test_rejects_invalid_prototype_class_index():
    payload = make_dinov2_payload()
    payload["prototype_class_indices"] = torch.tensor([0, 1, 1, 9])
    with pytest.raises(CheckpointValidationError, match="prototype_class_indices"):
        validate_checkpoint(payload)


def test_rejects_prototype_counts_that_disagree_with_mapping():
    payload = make_dinov2_payload()
    payload["prototypes_per_class"] = [2, 2]
    with pytest.raises(CheckpointValidationError, match="prototypes_per_class"):
        validate_checkpoint(payload)


def test_rejects_adaptive_k_mapping_that_disagrees_with_counts():
    payload = make_dinov2_payload()
    payload["k_by_class"] = {"A": 2, "B": 2}
    with pytest.raises(CheckpointValidationError, match="k_by_class"):
        validate_checkpoint(payload)


def test_rejects_invalid_encoder_sha256():
    payload = make_dinov2_payload()
    payload["encoder_sha256"] = "abc"
    with pytest.raises(CheckpointValidationError, match="encoder_sha256"):
        validate_checkpoint(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cosine_threshold", math.nan),
        ("adjusted_distance_score_threshold", math.inf),
        ("margin_weight", -0.01),
    ],
)
def test_rejects_invalid_margin_rejection_scalars(key, value):
    payload = make_dinov2_payload()
    payload["rejection"][key] = value
    with pytest.raises(CheckpointValidationError, match=key):
        validate_checkpoint(payload)


def test_rejects_wrong_rejection_mode():
    payload = make_dinov2_payload()
    payload["rejection"]["mode"] = "multi_dual"
    with pytest.raises(CheckpointValidationError, match="multi_dual_margin"):
        validate_checkpoint(payload)


@pytest.mark.parametrize(
    "scales",
    [[1.0], [1.0, 0.0], [1.0, math.nan], "1,1"],
)
def test_rejects_invalid_class_distance_scales(scales):
    payload = make_dinov2_payload()
    payload["rejection"]["class_distance_scales"] = scales
    with pytest.raises(CheckpointValidationError, match="class_distance_scales"):
        validate_checkpoint(payload)


def test_threshold_must_match_cosine_gate():
    payload = make_dinov2_payload()
    payload["threshold"] = 0.99
    with pytest.raises(CheckpointValidationError, match="threshold"):
        validate_checkpoint(payload)


def test_fingerprint_changes_when_margin_rejection_changes():
    first = validate_checkpoint(make_dinov2_payload())
    payload = make_dinov2_payload()
    payload["rejection"]["margin_weight"] = 1.0
    second = validate_checkpoint(payload)
    assert first.fingerprint != second.fingerprint


def test_validation_does_not_mutate_input_tensors_or_metadata():
    payload = make_dinov2_payload()
    original = clone_payload(payload)
    validate_checkpoint(payload)
    assert torch.equal(payload["feature_center"], original["feature_center"])
    assert torch.equal(payload["prototypes"], original["prototypes"])
    assert payload["rejection"] == original["rejection"]


def test_load_checkpoint_uses_file_sha256_as_fingerprint(tmp_path):
    import hashlib
    from system.dinov2.checkpoint import load_checkpoint

    path = tmp_path / "candidate.pt"
    torch.save(make_dinov2_payload(), path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint = load_checkpoint(path)

    assert checkpoint.path == path.resolve()
    assert checkpoint.fingerprint == expected
