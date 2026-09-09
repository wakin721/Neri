from __future__ import annotations

import pytest

from system.dinov3.checkpoint import CheckpointValidationError, validate_checkpoint
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def test_multi_prototype_checkpoint_does_not_require_linear_head():
    checkpoint = validate_checkpoint(make_multi_prototype_payload())

    assert checkpoint.head_type == "multi_prototype"
    assert tuple(checkpoint.prototypes.shape) == (6, 768)
    assert tuple(checkpoint.feature_center.shape) == (768,)
    assert checkpoint.selection_k == 3


def test_multi_prototype_checkpoint_rejects_invalid_class_indices():
    payload = make_multi_prototype_payload()
    payload["prototype_class_indices"][0] = 99

    with pytest.raises(CheckpointValidationError):
        validate_checkpoint(payload)


def test_multi_prototype_checkpoint_rejects_non_k3_contract():
    payload = make_multi_prototype_payload()
    payload["selection_k"] = 5

    with pytest.raises(CheckpointValidationError):
        validate_checkpoint(payload)
