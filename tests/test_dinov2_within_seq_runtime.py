from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.memory_bank import MemoryBank, MemoryExample
from system.dinov2.memory_classifier import MemoryDinoV2Classifier
from system.dinov2.runtime import load_dinov2_model
from system.dinov2.within_seq import WithinSeqModel, calibration_threshold


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    rng=np.random.default_rng(7)
    raw=rng.normal(0,.015,(24,768)); raw[:12,0]+=1; raw[12:,1]+=1
    raw/=np.linalg.norm(raw,axis=1,keepdims=True)
    labels=np.array(['A']*12+['B']*12); cameras=np.array([f'cam:{i//4}' for i in range(24)]); sequences=np.array([f'seq:{i//2}' for i in range(24)])
    with threadpool_limits(limits=1):
        model=WithinSeqModel(raw,labels,cameras,sequences)
    path=tmp_path_factory.mktemp('within-runtime')/'memory_head.npz'
    metadata=dict(version=1,rejection_mode='within_seq',threshold=.5,
        config=dict(neighbors=3,centroid_weight=.5,camera_pooling=True,margin_weight=1.),
        calibration=dict(images=10,sequences=10,target_unknown_recall=.9),
        provenance=dict(encoder_sha256='a'*64,preprocessing='letterbox224_imagenet',feature_dim=768,gradient_updates=0))
    np.savez_compressed(path,metadata=json.dumps(metadata),center=model.classification_center,features=model.memory,
        centroids=model.memory_centroids,labels=labels,cameras=cameras,classes=model.classes,raw_features=raw,sequences=sequences)
    with threadpool_limits(limits=1): checkpoint=load_checkpoint(path)
    return path,checkpoint,raw


def test_runtime_uses_within_scores_and_camera_scoped_sequence_mean(saved):
    _,cp,raw=saved; classifier=MemoryDinoV2Classifier(cp)
    cameras=['q1','q1','q2','q2'];seq=['same']*4;query=raw[[0,12,3,4]]
    expected=cp.within_seq.score_batch(query,cameras,seq)
    predictions=classifier.classify_features(query,camera_ids=cameras,sequence_ids=seq)
    np.testing.assert_allclose([p.known_score for p in predictions],expected['sequence_knownness'],atol=2e-6)
    assert [p.best_known_species for p in predictions]==expected['known_species']
    assert predictions[0].known_score==predictions[1].known_score
    assert predictions[0].known_score!=predictions[2].known_score
    assert classifier.rejection_metadata['mode']=='memory_within_seq'
    np.testing.assert_allclose(expected['sequence_knownness'],cp.within_seq.score_batch(query,cameras,seq,batch_size=1)['sequence_knownness'])
    with pytest.raises(ValueError):classifier.classify_features(query,camera_ids=cameras)


def test_event_averages_image_scores_and_keeps_raw_event_embedding(saved):
    _,cp,raw=saved; features=raw[[0,12]]
    classifier=MemoryDinoV2Classifier(cp,encoder=SimpleNamespace(encode=lambda *a,**k:features))
    prediction=classifier.classify_event(['first','second'])
    expected=cp.within_seq.score_batch(features,['event']*2,['event']*2)['image_knownness'].mean()
    assert prediction.known_score==pytest.approx(expected)
    assert np.linalg.norm(prediction.embedding)==pytest.approx(1)
    assert prediction.accepted==(prediction.known_score>=cp.threshold)


def test_provisional_evidence_veto_survives_within_and_frozen_transforms(saved):
    _,cp,raw=saved
    query=np.eye(768,dtype=np.float32)[100]
    provisional=MemoryExample('new',query,'new-camera','registry',42,'provisional')
    classifier=MemoryDinoV2Classifier(replace(cp,threshold=-100),registry=SimpleNamespace(memory_bank=lambda:MemoryBank((),(provisional,))))
    prediction=classifier.classify_features(query[None])[0]
    assert not prediction.accepted
    assert prediction.species=='new' and prediction.assistive_match
    extended=cp.within_seq.with_examples((provisional,))
    np.testing.assert_array_equal(extended.whitening,cp.within_seq.whitening)
    np.testing.assert_array_equal(extended.within_center,cp.within_seq.within_center)
    assert 'new' in extended.classes and 'new' not in cp.within_seq.classes
    assert extended.score_batch(query[None],['query'],['query'])['known_species']==['new']
    np.testing.assert_array_equal(extended.sequence_memory[:len(cp.within_seq.sequence_memory)],cp.within_seq.sequence_memory)


def test_manifest_selects_within_classifier_and_load_rejects_missing_raw(saved,tmp_path):
    path,cp,raw=saved
    manifest=path.with_suffix('.neri.json')
    manifest.write_text(json.dumps(dict(schema_version=2,backend='dinov2',checkpoint=path.name,
        architecture='dinov2_vitb14',feature_dim=768,encoder_sha256='a'*64,
        preprocessing='letterbox224_imagenet',event_aggregation='mean_l2_normalized_crop_embeddings')))
    store=SimpleNamespace(model_fingerprint=cp.fingerprint)
    with threadpool_limits(limits=1):
        runtime=load_dinov2_model(manifest,registry=store,feedback=store,encoder_factory=lambda *a,**k:None)
    assert runtime.checkpoint.head_type=='memory_within_seq'
    assert runtime.classifier.classify_features(raw[:1])[0].known_score==pytest.approx(cp.within_seq.score_batch(raw[:1],['q'],['s'])['image_knownness'][0])
    with np.load(path) as z:bad={k:z[k] for k in z.files if k not in {'raw_features','sequences'}}
    bad_path=tmp_path/'bad.npz';np.savez_compressed(bad_path,**bad)
    with pytest.raises(ValueError,match='raw memory is missing'):load_checkpoint(bad_path)


def test_bundled_default_is_validated_within_head():
    path=Path(__file__).resolve().parents[1]/'res/dinov2/memory_head.npz'
    with threadpool_limits(limits=1):cp=load_checkpoint(path)
    assert cp.head_type=='memory_within_seq' and len(cp.classes)==42
    assert cp.calibration['target_unknown_recall']==.9
    assert cp.calibration['empirical_aux_unknown_recall']>=.9
    assert cp.threshold==pytest.approx(.592520534992218)
    assert cp.calibration['point']=='Val90'
    assert cp.calibration['images']==123
    assert cp.calibration['query_mode']=='independent_single_frame'
    manifest=json.loads(path.with_name('memory_head_manifest.json').read_text('utf8'))
    policy=manifest['class_role_policy']
    counts=policy['reviewed_class_counts']
    assert policy['min_known_reviewed_images']==10
    assert set(cp.classes)=={species for species,n in counts.items() if n>=10}
    assert set(policy['unknown_classes'])=={species for species,n in counts.items() if n<10}
    assert not set(cp.labels) & set(policy['unknown_classes'])
    assert len(cp.within_seq.raw)==3311
