from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from system.dinov3.checkpoint import (
    DINO_FEATURE_DIM,
    CheckpointValidationError,
    validate_checkpoint,
)
from system.dinov3.classifier import DinoV3Classifier
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def test_multi_prototype_checkpoint_does_not_require_linear_head():
    checkpoint = validate_checkpoint(make_multi_prototype_payload())

    assert checkpoint.head_type == "multi_prototype"
    assert tuple(checkpoint.prototypes.shape) == (6, 768)
    assert tuple(checkpoint.feature_center.shape) == (768,)
    assert checkpoint.selection_k == 3


def test_multi_prototype_checkpoint_accepts_selection_k_from_checkpoint():
    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(selection_k=5)
    )

    assert checkpoint.selection_k == 5
    assert tuple(checkpoint.prototypes.shape) == (10, 768)
    assert checkpoint.prototypes_per_class == (5, 5)


def test_multi_prototype_checkpoint_rejects_invalid_class_indices():
    payload = make_multi_prototype_payload()
    payload["prototype_class_indices"][0] = 99

    with pytest.raises(CheckpointValidationError):
        validate_checkpoint(payload)


@pytest.mark.parametrize("selection_k", [0, -1, True, 3.0, "3", None])
def test_multi_prototype_checkpoint_rejects_invalid_selection_k(selection_k):
    payload = make_multi_prototype_payload()
    payload["selection_k"] = selection_k

    with pytest.raises(CheckpointValidationError, match="selection_k"):
        validate_checkpoint(payload)


def test_checkpoint_validator_rejects_legacy_linear_head():
    payload = make_multi_prototype_payload()
    classes = payload["classes"]
    assert isinstance(classes, list)
    class_count = len(classes)
    payload["head_type"] = "linear"
    payload["head_state"] = {
        "weight": torch.zeros((class_count, DINO_FEATURE_DIM)),
        "bias": torch.zeros(class_count),
    }
    payload["prototypes"] = torch.zeros((class_count, DINO_FEATURE_DIM))
    for index in range(class_count):
        payload["prototypes"][index, index] = 1.0

    with pytest.raises(CheckpointValidationError, match="multi_prototype"):
        validate_checkpoint(payload)


def test_classifier_rejects_direct_legacy_linear_checkpoint():
    checkpoint = validate_checkpoint(make_multi_prototype_payload())
    legacy = replace(
        checkpoint,
        head_type="linear",
        head_weight=torch.zeros((len(checkpoint.classes), DINO_FEATURE_DIM)),
        head_bias=torch.zeros(len(checkpoint.classes)),
    )

    with pytest.raises(ValueError, match="multi_prototype"):
        DinoV3Classifier(legacy)
