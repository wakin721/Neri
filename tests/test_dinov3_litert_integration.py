"""Bundled species classification remains usable in a torch-free installation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from system.model_sync.layout import get_model_layout


def litert_manifest(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'head.neri-model').write_bytes(b'head')
    (directory/'encoder.tflite').write_bytes(b'encoder')
    digest=lambda value:hashlib.sha256(value).hexdigest()
    payload=dict(schema_version=1,backend='dinov3',architecture='dinov3_vitb16',
        encoder_backend='litert',encoder_weights='encoder.tflite',
        encoder_litert_sha256=digest(b'encoder'),encoder_sha256='a'*64,
        checkpoint='head.neri-model',checkpoint_sha256=digest(b'head'),
        head_type='ridge_within_experiment',inference_recipe='experiment_full',
        feature_dim=768,preprocessing='letterbox224_imagenet',
        event_aggregation='complete_sequence',requires_detector=True,
        supports_video_fast=False,supports_video_all=False,
        display_name='Neri1.2preview（LiteRT FP32 CPU）')
    path=directory/'Neri1.2preview-litert.neri.json'
    path.write_text(json.dumps(payload),'utf8')
    return path


def test_bundled_classifier_discovery_and_api_metadata(tmp_path):
    from system.model_sync.catalog import discover_models
    from system.backend.model_services import _to_model_info
    layout=get_model_layout(tmp_path)
    path=litert_manifest(layout.cls_bundled)
    discovered=discover_models(layout,'cls')
    assert len(discovered)==1
    model=discovered[0]
    assert model.path==str(path.resolve()) and model.source=='bundled'
    assert model.backend=='dinov3' and model.encoder_backend=='litert'
    info=_to_model_info(model)
    assert info.encoder_backend=='litert' and info.requires_detector
    assert info.supports_video_fast and not info.supports_video_all


def test_litert_detector_and_classifier_collect_real_crop_contract_without_torch(tmp_path,monkeypatch):
    import cv2
    from system.backend import services
    from system.dinov3.classifier import DinoV3Prediction
    manifest=litert_manifest(tmp_path/'classifier')
    detector_path=tmp_path/'detector.tflite'
    detector_path.write_bytes(b'detector')
    photo=tmp_path/'animal.jpg'
    cv2.imwrite(str(photo),np.full((100,120,3),128,np.uint8))
    class FakeDetector:
        def __init__(self,*args,**kwargs): self.names={0:'animal'}
        def predict(self,image,**kwargs):
            assert image.shape==(100,120,3)
            return [SimpleNamespace(x1=10,y1=20,x2=70,y2=80,confidence=.95,class_id=0)]
    raw=np.zeros(768,np.float32)
    raw[0]=1
    def classify(crops,**kwargs):
        assert crops[0].shape==(72,72,3) and kwargs['array_color']=='rgb'
        return [DinoV3Prediction(species='岩羊',best_known_species='岩羊',accepted=True,
            known_score=.5,threshold=0.,embedding=raw,candidates=(),ridge_score=.8,within_score=.8)]
    classifier=SimpleNamespace(backend='dinov3',encoder=SimpleNamespace(device='cpu',use_fp16=False),classify_crops=classify)
    monkeypatch.setattr('system.dino.litert_image_processor.LiteRtDetector',FakeDetector)
    monkeypatch.setattr('system.dinov3.runtime.load_dinov3_model',lambda *args,**kwargs:SimpleNamespace(classifier=classifier))
    original=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None if name=='torch' else original(name))
    detector=services._load_detector(str(detector_path),str(manifest))
    output=detector.detect_batch_species([str(photo)],use_fp16=True)
    assert output[0]['物种名称']=='岩羊'
    records=detector._dinov3_collector.get_all_crops()
    assert len(records)==1 and records[0].image_path==str(photo)
    np.testing.assert_array_equal(records[0].raw_embedding,raw)
    serialized=services._legacy._serialize_detector_output(detector,output[0])
    assert serialized['detection_completed']
    assert serialized['检测框'][0]['sequence_pending']


@pytest.mark.parametrize('saved', [{},{'selected_classification_model':'builtin'},
    {'selected_classification_model':''},{'selected_classification_model':None}])
def test_torch_free_settings_default_preserve_selection_and_explicit_disable(monkeypatch,saved):
    from system.backend import main_core_legacy as core
    from system.backend.models import ModelInfo
    detect=ModelInfo(name='detector',path='detector',source='bundled',backend='litert')
    classifier=ModelInfo(name='builtin',path='builtin',kind='cls',source='bundled',backend='dinov3',encoder_backend='litert')
    monkeypatch.setattr(core,'list_available_models',lambda:[detect])
    monkeypatch.setattr(core,'list_available_classification_models',lambda:[classifier])
    monkeypatch.setattr(core,'missing_yolo_dependencies',lambda:['torch','ultralytics'])
    monkeypatch.setattr(core,'detect_gpu_available',lambda:False)
    monkeypatch.setattr(core.settings_manager,'load_settings',lambda:dict(saved))
    original=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None if name=='torch' else object() if name in {'ai_edge_litert','cv2','numpy'} else original(name))
    response=core.settings()
    expected='builtin' if not saved or saved.get('selected_classification_model') else None
    assert response.selected_classification_model==expected
    assert response.selected_model=='detector' and response.detection_ready
    assert response.model_dump()['available_classification_models'][0]['encoder_backend']=='litert'


def test_gpu_classifier_selection_survives_missing_ultralytics_with_litert_detector(monkeypatch):
    from system.backend import main_core_legacy as core
    from system.backend.models import ModelInfo
    detect=ModelInfo(name='detector',path='detector',source='bundled',backend='litert')
    cpu=ModelInfo(name='cpu',path='cpu',kind='cls',backend='dinov3',encoder_backend='litert')
    gpu=ModelInfo(name='gpu',path='gpu',kind='cls',backend='dinov3',encoder_backend='pytorch')
    monkeypatch.setattr(core,'list_available_models',lambda:[detect])
    monkeypatch.setattr(core,'list_available_classification_models',lambda:[cpu,gpu])
    monkeypatch.setattr(core,'missing_yolo_dependencies',lambda:['ultralytics'])
    monkeypatch.setattr(core,'detect_gpu_available',lambda:True)
    monkeypatch.setattr(core.settings_manager,'load_settings',lambda:{'selected_classification_model':'gpu'})
    original=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:object() if name in {'torch','ai_edge_litert','cv2','numpy'} else original(name))
    response=core.settings()
    assert response.selected_classification_model=='gpu'
    assert response.selected_model=='detector' and response.detection_ready


def test_settings_resolve_old_builtin_names_after_display_rename(monkeypatch):
    from system.backend import main_core_legacy as core
    from system.backend.models import ModelInfo
    detector=ModelInfo(name='MDV6-yolov10-c·LiteRT',path='/models/detect/bundled/MDV6-yolov10-c.tflite',
        source='bundled',backend='litert')
    classifier=ModelInfo(name='Neri1.2preview·LiteRT',path='/models/cls/bundled/Neri1.2preview-litert.neri.json',
        source='bundled',kind='cls',backend='dinov3',encoder_backend='litert')
    monkeypatch.setattr(core,'list_available_models',lambda:[detector])
    monkeypatch.setattr(core,'list_available_classification_models',lambda:[classifier])
    monkeypatch.setattr(core,'missing_yolo_dependencies',lambda:[])
    monkeypatch.setattr(core,'detect_gpu_available',lambda:False)
    monkeypatch.setattr(core.settings_manager,'load_settings',lambda:{
        'selected_model':'MDV6-yolov10-c.tflite','selected_classification_model':'Neri1.2preview（LiteRT FP32 CPU）'})
    response=core.settings()
    assert response.selected_model==detector.path
    assert response.selected_classification_model==classifier.path
