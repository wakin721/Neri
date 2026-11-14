"""Expansion preserves held-out cameras/species and truthful sample roles."""
import pytest

from scripts.expand_dinov3_calibration import audit_split, expansion_plan


def row(i, species, camera, image=None, source=""):
    return dict(id=str(i), species=species, camera=camera, sequence=f"event{i}",
                image_id=image or f"image{i}", sha256=f"hash{i}", source_sha256=source)


@pytest.fixture
def data():
    rows=[row(0,"A","bank1"), row(1,"B","bank1"),
          row(2,"A","bank2"), row(3,"B","bank2"),
          row(4,"A","cal"), row(5,"U","cal"),
          row(6,"A","test"), row(7,"V","test"),
          row(8,"U","bank1"), row(9,"U","test"),
          row(10,"V","bank2"), row(11,"U","bank2",image="image2")]
    uses={0:"bank",1:"bank",2:"bank",3:"bank",4:"calibration",5:"calibration_unknown",
          6:"evaluation",7:"evaluation_unknown",8:"future_support_reserve",9:"future_support_reserve",
          10:"future_support_reserve",11:"future_support_reserve"}
    return rows,uses


def test_fixed_model_calibration_excludes_test_species_cameras_and_original_aliases(data):
    rows,uses=data
    plan=expansion_plan(rows,uses,{"A","B"},{"U":"calibration","V":"evaluation"})
    assert plan["added_unknown_indices"]==[8]
    reasons={r["index"]:r["reason"] for r in plan["reserve_exclusions"]}
    assert reasons[9]=="evaluation_camera"
    assert reasons[10]=="heldout_unknown_species_or_non_unknown"
    assert reasons[11]=="B_or_E_event_original_or_content"
    assert plan["audits"]["expanded_same_model"]["B_C_shared_cameras"]==["bank1"]
    for arm in plan["arms"].values():
        assert arm["6"]=="evaluation" and arm["7"]=="evaluation_unknown"


def test_strict_expansion_moves_whole_training_camera_and_preserves_classes(data):
    rows,uses=data
    plan=expansion_plan(rows,uses,{"A","B"},{"U":"calibration","V":"evaluation"})
    assert plan["strict_added_unknown_indices"]==[8]
    assert plan["strict_moved_B_indices"]==[0,1]
    strict=plan["arms"]["expanded_camera_isolated"]
    assert strict["0"]==strict["1"]=="calibration"
    assert strict["2"]==strict["3"]=="bank"
    assert plan["audits"]["expanded_camera_isolated"]["full_camera_isolation"]


def test_last_known_support_blocks_strict_transfer_but_not_independent_reserve_calibration(data):
    rows,uses=data
    rows.append(row(12,"rare","bank1"))
    uses[12]="bank"
    plan=expansion_plan(rows,uses,{"A","B","rare"},{"U":"calibration","V":"evaluation"})
    assert plan["added_unknown_indices"]==[8]
    assert plan["strict_added_unknown_indices"]==[]
    assert plan["strict_blocked_camera_groups"][0]["would_remove_known_classes"]==["rare"]
    assert plan["arms"]["expanded_camera_isolated"]["12"]=="bank"


def test_full_camera_claim_is_rejected_when_only_events_are_independent(data):
    rows,uses=data
    uses[8]="calibration_unknown"
    assert audit_split(rows,uses,{"A","B"},False)["passed"]
    with pytest.raises(ValueError,match="B/C camera"):
        audit_split(rows,uses,{"A","B"},True)


def test_unknown_test_species_cannot_be_reused_in_calibration(data):
    rows,uses=data
    rows.append(row(12,"V","another-cal-camera"))
    uses[12]="calibration_unknown"
    with pytest.raises(ValueError,match="Unknown calibration/test species"):
        audit_split(rows,uses,{"A","B"},False)
