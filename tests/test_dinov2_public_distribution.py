import pytest


def test_distribution_path_is_dinov2_scoped():
    from system.dinov2.public_distribution import _relative_component_path

    assert (
        _relative_component_path("DINOv2/model/model.safetensors")
        == "model/model.safetensors"
    )
    with pytest.raises(Exception):
        _relative_component_path("DINOv3/model/model.safetensors")
