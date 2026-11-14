"""Protocol-level tests: camera aliases, held-out unknowns and class fusion."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.compare_local_dinov3_protocols import (
    camera_components, make_camera_partition, metrics, reference_recipe,
    validate_partition,
    freeze,
)
from system.dinov3.model import adaptive_sequence_fuse


def test_frozen_json_roundtrip_and_changed_protocol(tmp_path):
    path=tmp_path/"protocol.json"
    freeze(path,{"schemes":("one","two")})
    freeze(path,{"schemes":("one","two")})
    with pytest.raises(ValueError,match="Frozen protocol changed"):
        freeze(path,{"schemes":("different",)})


def row(i, species, camera, digest=None, image=None, source=""):
    return dict(id=str(i),species=species,camera=camera,sequence=f"event{i}",
                sha256=digest or f"hash{i}",image_id=image or f"image{i}",source_sha256=source)


def test_camera_aliases_and_duplicate_labels_cannot_cross_pools():
    rows=[row(0,"A","one"),row(1,"A","two",digest="hash0"),
          row(2,"B","two"),row(3,"B","three",source="same-original"),
          row(4,"A","four",source="same-original"),row(5,"A","five",image="image2")]
    groups,excluded=camera_components(rows)
    assert excluded=={1:"duplicate_crop"}
    assert any(set(g)=={0,2,5} for g in groups)
    assert any(set(g)=={3,4} for g in groups)
    rows.append(row(6,"B","six",digest="hash0"))
    _,excluded=camera_components(rows)
    assert all(excluded[i]=="label_conflict" for i in (0,1,6))


def test_camera_split_covers_known_without_putting_single_camera_species_in_test():
    rows=[]
    for camera in range(6):
        for species in ("A","B"):
            rows.append(row(len(rows),species,f"cam{camera}"))
    rows.append(row(len(rows),"single","only"))
    rows.append(row(len(rows),"two","cam0"))
    rows.append(row(len(rows),"two","cam1"))
    for label in ("U1","U2","U3","U4"):
        for camera in range(6):
            rows.append(row(len(rows),label,f"unknown-{label}-{camera}"))
    uses,info=make_camera_partition(rows,["A","B","single","two"])
    assert validate_partition(rows,uses,["A","B","single","two"])["passed"]
    assert uses[12]=="bank"
    for species in ("A","B"):
        assert {uses[i] for i,r in enumerate(rows) if r["species"]==species}=={"bank","calibration","evaluation"}
    assert {uses[i] for i,r in enumerate(rows) if r["species"]=="two"}=={"bank","evaluation"}
    assert sum(info["purpose_counts"].values())==len(rows)


def test_validation_rejects_shared_cameras_and_shared_unknown_species():
    rows=[row(0,"A","shared"),row(1,"A","shared"),
          row(2,"U","c-u"),row(3,"V","e-u")]
    with pytest.raises(ValueError,match="Cross-pool"):
        validate_partition(rows,{0:"bank",1:"evaluation",2:"calibration_unknown",3:"evaluation_unknown"},["A"])
    rows[1]["camera"]="test"
    rows[3]["species"]="U"
    with pytest.raises(ValueError,match="Unknown species"):
        validate_partition(rows,{0:"bank",1:"evaluation",2:"calibration_unknown",3:"evaluation_unknown"},["A"])


def test_macro_reports_do_not_hide_rare_species_behind_herd_counts():
    truth=np.asarray(["A"]*99+["B","U","V"])
    prediction=np.asarray(["A"]*len(truth))
    accepted=np.ones(len(truth),dtype=bool)
    accepted[99:]=False
    result=metrics(truth,prediction,accepted,np.asarray(["A","B"]))
    assert result["known_false_rejection_macro"]==.5
    assert result["known_false_rejection_micro"]==.01
    assert result["unknown_rejection_macro"]==1


def test_experiment_blend_uses_uncorrected_within_gate_and_c_only_threshold():
    def row_zscore(v):
        return (v-v.mean(1,keepdims=True))/np.maximum(v.std(1,keepdims=True),1e-6)
    def cardinality(cameras,labels,classes,alpha):
        return np.asarray([.1,-.1]),np.asarray([3,3])
    def pool(values,inverse,fraction):
        return np.asarray([np.sort(values[inverse==i])[-max(1,int(np.ceil(fraction*(inverse==i).sum()))):].mean() for i in inverse])
    reference=SimpleNamespace(row_zscore=row_zscore,cardinality_bias=cardinality,
                              adaptive_sequence_fuse=adaptive_sequence_fuse,compute_top_quantile_pooling=pool)
    model=SimpleNamespace(classes=np.asarray(["A","B"]))
    cq=dict(labels=np.asarray(["A","B","U","U"]),cameras=np.asarray(["c"]*4),
            sequences=np.asarray(["c0","c1","c2","c3"]),contents=np.asarray(["x0","x1","x2","x3"]),
            crop_hashes=np.asarray(["x0","x1","x2","x3"]))
    eq=dict(cq,labels=np.asarray(["A","B","V","V"]))
    scores=dict(within=np.asarray([[.9,.1],[.2,.8],[.4,.3],[.3,.2]]),
                ridge=np.asarray([[.1,.9],[.8,.2],[.6,.4],[.4,.6]]),
                margin=np.asarray([1.7,1.4,.5,.4]),norm=np.asarray([1.,2.,3.,4.]))
    result,cal=reference_recipe(model,cq,eq,scores,scores,np.asarray(["c"]),np.asarray(["A"]),reference,blend=True)
    expected=model.classes[(.55*row_zscore(scores["within"]-np.asarray([.1,-.1]))+.45*row_zscore(scores["ridge"])).argmax(1)]
    np.testing.assert_array_equal(result[1]["predictions"],expected)
    _,changed=reference_recipe(model,cq,dict(eq,labels=np.asarray(["new"]*4)),scores,
                               dict(scores,margin=scores["margin"]*100),np.asarray(["c"]),np.asarray(["A"]),reference,blend=True)
    assert cal==changed
