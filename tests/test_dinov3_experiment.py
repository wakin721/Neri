"""Experimental recipe survives deployment, feedback updates and old models."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.deployment import publish_model
from system.dinov3.experiment import EXPERIMENT_RECIPE, event_fuse, top_fraction
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import load_dinov3_model, read_manifest, resolve_manifest_asset
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_learning import ready


@pytest.fixture
def experiment(deployed):
    manifest, base, raw = deployed
    labels = np.asarray(["A"]*8+["B"]*8)
    cameras = np.asarray([f"c{i//4}" for i in range(16)])
    sequences = np.asarray([f"s{i//2}" for i in range(16)])
    rng = np.random.default_rng(90)
    unknown = rng.normal(size=(8,768)).astype(np.float32)
    unknown /= np.linalg.norm(unknown,axis=1,keepdims=True)
    x = np.r_[raw,unknown]
    cal = dict(features=x,labels=np.r_[labels,["U"]*8],cameras=np.asarray(["cal"]*24),
               sequences=np.asarray([f"q{i//2}" for i in range(24)]),
               contents=np.asarray([f"frame{i//2}:{i%2}" for i in range(24)]))
    with threadpool_limits(limits=1):
        model = DinoV3RidgeWithinModel(inference_recipe=EXPERIMENT_RECIPE).fit(
            raw,labels,cameras,sequences,origin=base.origin_,transform=base.transform_,metadata=base.metadata)
        model.calibrate(x,cal["labels"],cal["cameras"],cal["sequences"],contents=cal["contents"])
    return manifest,model,raw,cal


def test_whole_event_fusion_does_not_silently_use_production_box_tracks():
    scores=np.asarray([[10.,0.],[0.,5.]])
    actual=event_fuse(scores,["cam"]*2,["event"]*2,["frame:0","frame:1"])
    expected=.2*scores+.8*scores.mean(0)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert not np.array_equal(actual,scores)


def test_top75_preserves_historical_float32_mean():
    values=np.asarray([.75104207,.751042,.7510421,-1],np.float32)
    actual=top_fraction(values,["c"]*4,["e"]*4)
    np.testing.assert_array_equal(actual,np.full(4,np.sort(values)[::-1][:3].mean()))


def test_experiment_prediction_save_load_and_incremental_update(experiment,tmp_path):
    _,model,raw,cal=experiment
    args=(cal["features"],cal["cameras"],cal["sequences"])
    with threadpool_limits(limits=1):
        before=model.predict(*args,contents=cal["contents"])
        path=tmp_path/"experiment.neri-model"
        model.save(path)
        loaded=DinoV3RidgeWithinModel.load(path)
        actual=loaded.predict(*args,contents=cal["contents"])
        assert loaded.inference_recipe==EXPERIMENT_RECIPE and loaded.tau==.5
        assert loaded.box_threshold is None
        assert "box_known_score" not in actual
        np.testing.assert_array_equal(actual["predictions"],before["predictions"])
        np.testing.assert_array_equal(actual["accepted"],before["accepted"])
        np.testing.assert_array_equal(actual["known_score"],before["known_score"])
        loaded.partial_fit(raw[:1],["C"],["new"],["new-event"])
        with pytest.raises(RuntimeError,match="not calibrated"):
            loaded.predict(*args)
        loaded.calibrate(cal["features"],cal["labels"],cal["cameras"],cal["sequences"],contents=cal["contents"])
        assert loaded.inference_recipe==EXPERIMENT_RECIPE
        assert set(loaded.classes)=={"A","B","C"}
        assert "box_threshold" not in loaded.calibration


def test_experiment_classification_uses_fused_within_and_ridge(experiment):
    _,model,_,cal=experiment
    with threadpool_limits(limits=1):
        result=model.predict(cal["features"],cal["cameras"],cal["sequences"],contents=cal["contents"])
        within=result["within_scores"]
        ridge=result["ridge_scores"]
        _,labels,cameras=model.within_updater.bank()
        count=np.asarray([len(set(cameras[labels==c])) for c in model.classes])
        def zscore(x):
            return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)
        expected=.55*zscore(within-.02*np.log(count/3)[None,:])+.45*zscore(ridge)
        np.testing.assert_allclose(result["classification_scores"],expected,atol=1e-12,rtol=0)
        predictions=DinoV3Classifier(model).classify_features(cal["features"],camera_ids=cal["cameras"],
            sequence_ids=cal["sequences"],content_ids=cal["contents"])
    candidate=predictions[0].as_candidate()
    assert candidate["gate_mode"]=="sequence"
    assert candidate["box_known_score"] is None
    assert candidate["box_threshold"] is None and candidate["box_accepted"] is None
    assert candidate["inference_recipe"]==EXPERIMENT_RECIPE


def test_manifest_runtime_and_feedback_loading_preserve_recipe(experiment):
    manifest,model,_,cal=experiment
    old=read_manifest(manifest)
    with threadpool_limits(limits=1):
        publish_model(model,manifest,existing_manifest=old,display_name="Neri 实验模型 B/16",calibration_data=cal)
        payload=read_manifest(manifest)
        assert payload["head_type"]=="ridge_within_experiment"
        runtime=load_dinov3_model(manifest,encoder_factory=lambda *a,**kw:SimpleNamespace())
        assert runtime.model.inference_recipe==EXPERIMENT_RECIPE
        from system.backend.dinov3_learning import _load_model
        assert _load_model(manifest)[2].inference_recipe==EXPERIMENT_RECIPE
        from system.backend.validation_fast import _is_joint_model
        assert _is_joint_model(manifest)
    payload.update(head_type="ridge_within",inference_recipe="production_dual")
    manifest.write_text(json.dumps(payload),"utf8")
    with pytest.raises(ValueError,match="inference recipes differ"):
        load_dinov3_model(manifest,encoder_factory=lambda *a,**kw:pytest.fail("encoder must not load"))


def test_invalid_recipe_is_not_silently_treated_as_dual():
    with pytest.raises(ValueError,match="inference recipe"):
        DinoV3RidgeWithinModel(inference_recipe="unknown")
    with pytest.raises(ValueError,match="tau"):
        DinoV3RidgeWithinModel(inference_recipe=EXPERIMENT_RECIPE,tau=1.)


def test_automatic_feedback_recalibrates_and_republishes_experiment(ready,tmp_path):
    from system.backend import dinov3_learning as learning
    from tests.test_dinov3_learning import stage,loaded,_write_image
    manifest,model,cal,_=ready
    model.inference_recipe=EXPERIMENT_RECIPE
    model.tau=.5
    with threadpool_limits(limits=1):
        model.calibrate(cal["features"],cal["labels"],cal["cameras"],cal["sequences"],contents=cal["contents"])
        publish_model(model,manifest,existing_manifest=read_manifest(manifest),calibration_data=cal)
        photo=tmp_path/"new-animal.jpg"
        _write_image(photo,130)
        stage(manifest,photo,"C")
        assert learning.update_model(manifest)["updated_count"]==1
        updated=loaded(manifest)
    assert updated.inference_recipe==EXPERIMENT_RECIPE
    assert set(updated.classes)=={"A","B","C"}
    assert read_manifest(manifest)["head_type"]=="ridge_within_experiment"
    rejection=learning.learning_status(manifest)["rejection"]
    assert rejection["gate_mode"]=="sequence" and rejection["box_threshold"] is None


def test_photo_finalization_passes_events_and_saves_only_experimental_gate(experiment,tmp_path):
    from unittest.mock import MagicMock
    from system.backend.models_legacy import DetectionItem
    from system.dinov3.processor import DinoV3CropCollector,DinoV3CropRecord,finalize_photo_job
    _,model,raw,_=experiment
    photo=tmp_path/"same-photo.jpg"
    photo.write_bytes(b"finalizer uses provided crop embeddings")
    collector=DinoV3CropCollector()
    collector.record_crop(DinoV3CropRecord(str(photo),0,(0,0,20,20),.9,raw[0]))
    collector.record_crop(DinoV3CropRecord(str(photo),1,(30,30,50,50),.8,raw[8]))
    detector=MagicMock()
    detector.dinov3_classifier=DinoV3Classifier(model)
    detector._dinov3_collector=collector
    item=DetectionItem(filename=photo.name,path=str(photo),file_type="image",date_taken="2026-11-14 10:00:00")
    with threadpool_limits(limits=1):
        result=finalize_photo_job(detector,[item],tmp_path)
    boxes=result[0].detection_boxes
    assert len(boxes)==2
    assert boxes[0]["track_id"]==boxes[1]["track_id"]==boxes[0]["event_id"]
    assert boxes[0]["sequence_known_score"]==boxes[1]["sequence_known_score"]
    for box in boxes:
        assert box["inference_recipe"]==EXPERIMENT_RECIPE and box["gate_mode"]=="sequence"
        assert box["box_known_score"] is None and box["box_accepted"] is None


def test_crop_hash_calibration_still_excludes_original_photo_from_feedback(ready,tmp_path):
    from system.backend import dinov3_learning as learning
    from system.dinov3.encoder import sha256
    from tests.test_dinov3_learning import stage,_write_image
    manifest,model,cal,_=ready
    photo=tmp_path/"calibration-source.jpg"
    _write_image(photo,130)
    cal=dict(cal,source_contents=np.asarray([sha256(photo)+":0"]*len(cal["labels"])))
    with threadpool_limits(limits=1):
        publish_model(model,manifest,existing_manifest=read_manifest(manifest),calibration_data=cal)
        stage(manifest,photo,"C")
        with pytest.raises(ValueError,match="校准集内容重叠"):
            learning.update_model(manifest)
