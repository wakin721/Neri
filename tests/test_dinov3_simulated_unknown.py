import numpy as np
from threadpoolctl import threadpool_limits

from scripts.simulate_dinov3_unknown import acquisition_units, proxy_order, species_folds, fit_shadow, aggregate_populations, evidence


def row(i, species, camera="camera", sequence=None, source=None):
    return dict(id=str(i),species=species,camera=camera,sequence=sequence or f"event{i}",
                image_id=f"image{i}",sha256=f"crop{i}",source_sha256=source or f"source{i}")


def test_original_aliases_and_repeated_frames_count_as_one_acquisition():
    rows=[row(0,"A",sequence="burst"),row(1,"A",sequence="burst"),
          row(2,"A",camera="alias-camera",source="source0"),row(3,"B")]
    assert acquisition_units(rows,np.arange(4))==[[0,1,2],[3]]


def test_nested_proxy_budgets_keep_species_and_unique_units():
    rows=[row(i,sp,camera=f"cam{i%3}") for i,sp in enumerate(["A"]+["B"]*8+["C"]*8)]
    units=acquisition_units(rows,np.arange(len(rows)))
    order=proxy_order(rows,units,20261114)
    previous=set()
    for budget in (3,8,15):
        sample=order[:budget]
        assert {r["species"] for r in sample}=={"A","B","C"}
        assert len({r["unit"] for r in sample})==len(sample)
        assert previous<=set(r["index"] for r in sample)
        previous=set(r["index"] for r in sample)


def test_rare_species_matching_does_not_lose_cooccurring_class():
    rows=[row(0,"A",sequence="shared"),row(1,"B",sequence="shared"),row(2,"B"),row(3,"C")]
    units=acquisition_units(rows,np.arange(4))
    order=proxy_order(rows,units,20261114)
    assert len(order)==3
    assert {r["species"] for r in order}=={"A","B","C"}


def test_every_species_is_held_once_and_retained_in_other_folds():
    rows=[row(i,sp) for i,sp in enumerate(["A"]*5+["B"]*4+["C"]*3+["D"]*2+["E"])]
    folds=species_folds(rows,acquisition_units(rows,np.arange(len(rows))),count=3)
    assert sorted(c for held in folds for c in held)==["A","B","C","D","E"]
    assert all(folds)


def test_shadow_geometry_and_classifiers_ignore_all_held_species_features():
    rows=[row(i,sp) for i,sp in enumerate(["A"]*3+["B"]*3+["C"]*3)]
    raw=np.random.default_rng(7).normal(size=(9,4)).astype(np.float32)
    changed=raw.copy()
    changed[6:]=100000
    with threadpool_limits(limits=1):
        first,train=fit_shadow(raw,rows,np.arange(9),{"C"},{})
        second,_=fit_shadow(changed,rows,np.arange(9),{"C"},{})
    np.testing.assert_array_equal(train,np.arange(6))
    assert set(first.classes)=={"A","B"}
    assert {k[0] for k in first.within_updater.groups}=={"A","B"}
    np.testing.assert_array_equal(first.origin_,second.origin_)
    np.testing.assert_array_equal(first.transform_,second.transform_)
    np.testing.assert_array_equal(first.ridge_head.coef_,second.ridge_head.coef_)


def test_shadow_aggregation_weights_species_not_model_photo_counts():
    records=[dict(role="known",species="A",photos=100,rejection=0.,correct_acceptance=1.,closed_accuracy=1.),
             dict(role="known",species="A",photos=100,rejection=1.,correct_acceptance=0.,closed_accuracy=1.),
             dict(role="known",species="B",photos=1,rejection=1.,correct_acceptance=0.,closed_accuracy=1.)]
    result=aggregate_populations(records)
    assert result["known_rejection_macro"]==.75
    assert result["known_species"]==2


def test_shadow_evidence_matches_native_experiment_inference():
    rows=[row(i,sp) for i,sp in enumerate(["A"]*3+["B"]*3+["C"]*3)]
    rng=np.random.default_rng(19)
    raw=rng.normal(size=(9,4)).astype(np.float32)
    x=rng.normal(size=(12,4)).astype(np.float32)
    q=dict(raw=x,adapted=x,labels=np.asarray(["A"]*4+["B"]*4+["C"]*4),
           cameras=np.asarray(["cal"]*12),sequences=np.asarray([f"seq{i//2}" for i in range(12)]),
           crop_hashes=np.asarray([f"hash{i}" for i in range(12)]))
    with threadpool_limits(limits=1):
        model,_=fit_shadow(raw,rows,np.arange(9),{"C"},{})
        actual,stats=evidence(model,q,known=np.isin(q["labels"],model.classes))
        model._calibration=dict(stats,threshold=0.,target=.9,fit={})
        model._threshold=0.
        model.is_calibrated=True
        native=model.predict(x,q["cameras"],q["sequences"],contents=q["crop_hashes"],adapted_features=x)
    np.testing.assert_array_equal(actual["predictions"],native["predictions"])
    np.testing.assert_array_equal(actual["known_score"],native["known_score"])
