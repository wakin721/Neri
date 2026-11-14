import numpy as np
import pytest

from scripts.train_reviewed_dinov3 import make_partition


def row(identity, species, camera, sequence=None, digest=None, image=None):
    return dict(id=identity, species=species, camera=camera, sequence=sequence or identity,
                sha256=digest or identity, image_id=image or identity, source_sha256="")


def test_partition_keeps_requested_classes_and_never_splits_events_or_duplicate_aliases():
    rows = [row(f"{label}-{cam}-{i}", label, f"cam{cam}")
            for label in ("A", "B", "C") for cam in range(4) for i in range(4)]
    rows += [row("u0", "U", "protected", "shared"), row("a0", "A", "protected", "shared")]
    # A renamed identical crop links its second original image to the first
    # original image, including another species' crop from that original.
    rows += [row("duplicate", "A", "alias-camera", digest=rows[0]["sha256"], image="alias-image"),
             row("other-object", "B", "alias-camera", image="alias-image")]
    pools, info = make_partition(rows, ["A", "B", "C"])
    owner = {int(i): pool for pool, indices in pools.items() for i in indices}
    assert {rows[int(i)]["species"] for i in pools["bank"]} == {"A", "B", "C"}
    assert owner[len(rows)-4] == owner[len(rows)-3] == "calibration"
    assert len(rows)-2 not in owner
    assert owner[0] == owner[len(rows)-1]
    assert info["duplicate_crops_excluded"] == 1
    assert info["strict_acquisition_source_image_content_isolation"]
    assert len(pools["evaluation"])
    again, _ = make_partition(rows, ["A", "B", "C"])
    for pool in pools:
        np.testing.assert_array_equal(pools[pool], again[pool])


def test_conflicting_labels_are_quarantined_and_cannot_create_a_training_class():
    rows = [row("a", "A", "c1", digest="conflict"), row("b", "B", "c2", digest="conflict")]
    with pytest.raises(ValueError, match="training event|class coverage"):
        make_partition(rows, ["A", "B"])
