"""Learning survives backend switches, legacy migration, undo and restarts."""
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.backend import dinov3_learning as learning
from system.dinov3 import shared_learning as shared
from system.dinov3.deployment import publish_model
from system.dinov3.encoder import sha256
from system.dinov3.runtime import read_manifest,resolve_manifest_asset,load_dinov3_model
from tests.test_dinov3_experiment import experiment,deployed
from tests.test_dinov3_box_explanation_and_feedback import _write_image
from tests.test_dinov3_learning import stage


@pytest.fixture
def pair(experiment,tmp_path,monkeypatch):
    original,model,raw,cal=experiment
    source=read_manifest(original)
    monkeypatch.setenv('NERI_DINO_STATE_DIR',str(tmp_path/'state'))
    gpu=tmp_path/'res/model/cls/user/preview-gpu.neri.json'
    with threadpool_limits(limits=1):
        publish_model(model,gpu,encoder_weights=resolve_manifest_asset(original,source['encoder_weights']),
            encoder_code=resolve_manifest_asset(original,source['encoder_code']),calibration_data=cal)
    p=read_manifest(gpu)
    p['shared_learning']=True
    gpu.write_text(json.dumps(p),'utf8')
    cpu=tmp_path/'res/model/cls/bundled/preview-cpu.neri.json'
    cpu.parent.mkdir(parents=True)
    assets=cpu.parent/'assets'
    assets.mkdir()
    for field,name in [('checkpoint','model.neri-model'),('calibration','calibration.npz')]:
        shutil.copy2(resolve_manifest_asset(gpu,p[field]),assets/name)
    (assets/'encoder.tflite').write_bytes(b'fake-compatible-fp32')
    q=dict(p,encoder_backend='litert',encoder_weights='assets/encoder.tflite',
        encoder_litert_sha256=sha256(assets/'encoder.tflite'),checkpoint='assets/model.neri-model',
        calibration='assets/calibration.npz')
    q.pop('encoder_code')
    cpu.write_text(json.dumps(q),'utf8')
    calls=[]
    def encoder_factory(manifest,payload,**kwargs):
        backend=payload.get('encoder_backend','pytorch')
        calls.append(('load',backend))
        def encode(crops,**_kwargs):
            calls.append(('encode',backend,len(crops)))
            result=np.zeros((len(crops),768),np.float32)
            for index,crop in enumerate(crops):
                result[index,2 if crop.mean()<180 else 3]=1
            return result
        return SimpleNamespace(encode=encode,device='cpu' if backend=='litert' else 'cuda',use_fp16=False)
    monkeypatch.setattr('system.dinov3.runtime.load_manifest_encoder',encoder_factory)
    return cpu,gpu,calls


def active_model(path):
    from system.dinov3.model import DinoV3RidgeWithinModel
    manifest,payload=learning.prepare_shared_learning(path,read_manifest(path))
    return DinoV3RidgeWithinModel.load(resolve_manifest_asset(manifest,payload['checkpoint']))


def test_cpu_gpu_share_new_species_feedback_and_undo_without_source_photos(pair,tmp_path):
    cpu,gpu,calls=pair
    original_bytes={p:p.read_bytes() for p in (cpu,gpu)}
    photo=tmp_path/'cpu.jpg'
    other=tmp_path/'gpu.jpg'
    _write_image(photo,120)
    _write_image(other,220)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','cpu-confirm')
        learning.update_model(cpu)
        assert set(active_model(gpu).classes)=={'A','B','C'}
        assert learning._key(cpu)==learning._key(gpu)
        photo.unlink()
        stage(gpu,other,'D','gpu-confirm')
        learning.update_model(gpu)
        assert set(active_model(cpu).classes)=={'A','B','C','D'}
        assert learning.learning_status(cpu)['learned_count']==2
        assert learning.learning_status(gpu)['learned_count']==2
        # Cached compatible features survive CPU->GPU and photo removal.
        learning.revert_staged_feedback(gpu,'cpu-confirm')
        learning.update_model(cpu)
        assert set(active_model(gpu).classes)=={'A','B','D'}
        runtime=load_dinov3_model(cpu)
        assert runtime.classifier.encoder.device=='cpu'
        runtime=load_dinov3_model(gpu)
        assert runtime.classifier.encoder.device=='cuda'
    assert all(p.read_bytes()==data for p,data in original_bytes.items())
    assert [row for row in calls if row[0]=='encode']==[('encode','litert',1),('encode','pytorch',1)]


def test_switch_then_relabel_and_undo_restores_shared_species(pair,tmp_path):
    cpu,gpu,_=pair
    photo=tmp_path/'photo.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','first')
        learning.update_model(cpu)
        stage(gpu,photo,'E','correct')
        learning.update_model(gpu)
        assert 'E' in active_model(cpu).classes and 'C' not in active_model(cpu).classes
        learning.revert_staged_feedback(cpu,'correct')
        learning.update_model(gpu)
        assert 'C' in active_model(cpu).classes and 'E' not in active_model(cpu).classes


def legacy_learning(monkeypatch,manifest,photo,species,operation):
    with monkeypatch.context() as old:
        old.setattr(shared,'family_descriptor',lambda *a,**k:None)
        stage(manifest,photo,species,operation)
        learning.update_model(manifest)


def test_legacy_cpu_and_gpu_merge_once_and_retain_undo_backups(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    one,two=tmp_path/'one.jpg',tmp_path/'two.jpg'
    _write_image(one,120)
    _write_image(two,220)
    with threadpool_limits(limits=1):
        legacy_learning(monkeypatch,cpu,one,'C','legacy-cpu')
        legacy_learning(monkeypatch,gpu,two,'D','legacy-gpu')
        before={p:p.read_bytes() for p in (cpu,gpu)}
        runtime=load_dinov3_model(gpu)
        assert set(runtime.classifier.classes)=={'A','B','C','D'}
        assert learning.learning_status(cpu)['learned_count']==2
        with closing(learning._connect()) as conn:
            assert conn.execute('SELECT count(*) FROM shared_learning_aliases').fetchone()[0]==2
            assert conn.execute('SELECT count(*) FROM samples WHERE model=?',(learning._path_key(cpu),)).fetchone()[0]==1
            assert conn.execute('SELECT count(*) FROM samples WHERE model=?',(learning._key(cpu),)).fetchone()[0]==2
        learning.revert_staged_feedback(cpu,'legacy-gpu')
        learning.update_model(cpu)
        assert set(active_model(gpu).classes)=={'A','B','C'}
        # Repeated registration cannot resurrect an undone legacy supplement.
        for _ in range(2):
            load_dinov3_model(gpu)
        assert set(active_model(cpu).classes)=={'A','B','C'}
        assert all(p.read_bytes()==data for p,data in before.items())


def test_legacy_conflicts_never_overwrite_annotations_or_source_models(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    photo=tmp_path/'same.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        legacy_learning(monkeypatch,cpu,photo,'C','cpu')
        legacy_learning(monkeypatch,gpu,photo,'D','gpu')
        before={p:p.read_bytes() for p in (cpu,gpu)}
        with pytest.raises(ValueError,match='不同校验'):
            load_dinov3_model(gpu)
    assert all(p.read_bytes()==data for p,data in before.items())
    with closing(learning._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM samples WHERE model LIKE ?',('family:%',)).fetchone()[0]==0
        assert conn.execute('SELECT count(*) FROM samples').fetchone()[0]==2


def test_resume_uses_family_manifest_and_skips_legacy_backup_rows(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    photo=tmp_path/'pending.jpg'
    _write_image(photo,120)
    stage(cpu,photo,'C','pending')
    scheduled=[]
    monkeypatch.setattr(learning,'schedule_learning',lambda path:scheduled.append(Path(path)))
    learning.resume_learning()
    target=shared.family_manifest(learning._state_root(),shared.family_id(cpu))
    assert scheduled==[target]
    with threadpool_limits(limits=1):
        learning.update_model(target)
    assert 'C' in active_model(gpu).classes


def test_same_names_with_different_base_checkpoint_stay_isolated(pair,tmp_path):
    from system.dinov3.model import DinoV3RidgeWithinModel
    cpu,gpu,_=pair
    other=tmp_path/'unrelated.neri.json'
    payload=read_manifest(gpu)
    model=DinoV3RidgeWithinModel.load(resolve_manifest_asset(gpu,payload['checkpoint']))
    model.metadata['different_training_version']=True
    model.save(tmp_path/'other.neri-model')
    # Reuse the fixture's encoder files through a complete separate deployment.
    with threadpool_limits(limits=1):
        publish_model(model,other,encoder_weights=resolve_manifest_asset(gpu,payload['encoder_weights']),
            encoder_code=resolve_manifest_asset(gpu,payload['encoder_code']))
    q=read_manifest(other)
    q['shared_learning']=True
    other.write_text(json.dumps(q),'utf8')
    assert shared.family_id(other)!=shared.family_id(cpu)
    assert learning._key(other)!=learning._key(cpu)


def test_backend_selection_during_fit_does_not_supersede_or_drop_new_peer(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    photo=tmp_path/'new.jpg'
    _write_image(photo,120)
    stage(gpu,photo,'C','concurrent')
    third_dir=tmp_path/'res/model/cls/sync'
    shutil.copytree(cpu.parent,third_dir)
    third=third_dir/cpu.name
    payload=read_manifest(third)
    weights=resolve_manifest_asset(third,payload['encoder_weights'])
    weights.write_bytes(b'another-explicitly-opted-in-compatible-conversion')
    payload['encoder_litert_sha256']=sha256(weights)
    third.write_text(json.dumps(payload),'utf8')
    original_factory=__import__('system.dinov3.runtime',fromlist=['load_manifest_encoder']).load_manifest_encoder
    def factory(*args,**kwargs):
        encoder=original_factory(*args,**kwargs)
        original_encode=encoder.encode
        def encode(crops,**kw):
            learning.learning_status(third)
            return original_encode(crops,**kw)
        encoder.encode=encode
        return encoder
    monkeypatch.setattr('system.dinov3.runtime.load_manifest_encoder',factory)
    with threadpool_limits(limits=1):
        result=learning.update_model(gpu)
    assert not result.get('superseded')
    target=shared.family_manifest(learning._state_root(),shared.family_id(cpu))
    current=read_manifest(target)
    assert len(current['learning_encoder_variants'])==3
    assert payload['encoder_litert_sha256'] in current['learning_encoder_hashes']
    assert 'C' in active_model(cpu).classes


def test_torch_free_restart_finishes_gpu_pending_work_with_registered_cpu(pair,tmp_path,monkeypatch):
    import importlib.util
    cpu,gpu,calls=pair
    photo=tmp_path/'resume.jpg'
    _write_image(photo,120)
    stage(gpu,photo,'C','restart')
    target=shared.family_manifest(learning._state_root(),shared.family_id(cpu))
    assert read_manifest(target)['encoder_backend']=='pytorch'
    scheduled=[]
    monkeypatch.setattr(learning,'schedule_learning',lambda path:scheduled.append(Path(path)))
    original=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None if name=='torch' else original(name))
    learning.resume_learning()
    assert scheduled==[target]
    assert read_manifest(target)['encoder_backend']=='litert'
    with threadpool_limits(limits=1):
        learning.update_model(target)
    assert ('encode','litert',1) in calls
    assert 'C' in active_model(cpu).classes


def test_equal_legacy_confirmations_deduplicate_and_keep_both_undo_histories(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    photo=tmp_path/'same-confirmation.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        legacy_learning(monkeypatch,cpu,photo,'C','old-cpu')
        legacy_learning(monkeypatch,gpu,photo,'C','old-gpu')
        load_dinov3_model(gpu)
        assert learning.learning_status(cpu)['learned_count']==1
        learning.revert_staged_feedback(gpu,'old-gpu')
        learning.update_model(gpu)
        assert 'C' in active_model(cpu).classes
        learning.revert_staged_feedback(cpu,'old-cpu')
        learning.update_model(cpu)
        assert 'C' not in active_model(gpu).classes


def test_interrupted_legacy_import_retries_without_losing_peer_learning(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    photo=tmp_path/'legacy.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        legacy_learning(monkeypatch,cpu,photo,'C','legacy')
        original=learning._import_alias
        def interrupted(*args,**kwargs):
            original(*args,**kwargs)
            raise RuntimeError('interrupted before SQLite commit')
        with monkeypatch.context() as failure:
            failure.setattr(learning,'_import_alias',interrupted)
            with pytest.raises(RuntimeError,match='interrupted'):
                load_dinov3_model(gpu)
        assert 'C' in load_dinov3_model(gpu).classifier.classes
        assert learning.learning_status(cpu)['learned_count']==1


def test_legacy_learning_in_another_installation_is_found_from_database(pair,tmp_path,monkeypatch):
    cpu,gpu,_=pair
    old_root=tmp_path/'old-install/res'
    shutil.copytree(tmp_path/'res',old_root)
    old_cpu=old_root/cpu.relative_to(tmp_path/'res')
    photo=tmp_path/'old-install-confirmed.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        legacy_learning(monkeypatch,old_cpu,photo,'C','old-install')
        assert 'C' in load_dinov3_model(gpu).classifier.classes
        assert learning.learning_status(cpu)['learned_count']==1


def test_new_installation_keeps_shared_learning_after_old_resources_removed(pair,tmp_path):
    cpu,gpu,_=pair
    photo=tmp_path/'confirmed.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','confirmed')
        learning.update_model(cpu)
        new_root=tmp_path/'new-install/res'
        shutil.copytree(tmp_path/'res',new_root)
        new_cpu=new_root/cpu.relative_to(tmp_path/'res')
        new_gpu=new_root/gpu.relative_to(tmp_path/'res')
        removed=(tmp_path/'res').resolve()
        removed.relative_to(tmp_path.resolve())
        shutil.rmtree(removed)
        assert 'C' in load_dinov3_model(new_gpu).classifier.classes
        assert learning.learning_status(new_cpu)['learned_count']==1
        learning.revert_staged_feedback(new_gpu,'confirmed')
        learning.update_model(new_cpu)
        assert 'C' not in active_model(new_gpu).classes


def test_shared_calibration_import_has_no_nested_write_lock(pair,monkeypatch):
    cpu,gpu,_=pair
    original=read_manifest(cpu)
    calibration=resolve_manifest_asset(cpu,original['calibration'])
    learning.prepare_shared_learning(cpu)
    monkeypatch.setattr(learning,'schedule_learning',lambda *_:None)
    with threadpool_limits(limits=1):
        learning.import_calibration(gpu,calibration)
    assert learning.learning_status(cpu)['calibration_available']


def test_api_classes_use_learned_head_and_distribution_check_can_load_base(pair,tmp_path):
    from system.backend.model_services import list_model_classes
    cpu,gpu,_=pair
    photo=tmp_path/'class-list.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','classes')
        learning.update_model(cpu)
        assert 'C' in {row.name for row in list_model_classes(str(gpu))}
        assert 'C' in load_dinov3_model(cpu).classifier.classes
        assert 'C' not in load_dinov3_model(cpu,use_shared_learning=False).classifier.classes


def test_reconfirmed_changed_photo_and_undo_use_committed_feature_backup(pair,tmp_path):
    cpu,gpu,_=pair
    photo=tmp_path/'changed.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','first')
        learning.update_model(cpu)
        _write_image(photo,220)
        stage(gpu,photo,'D','changed')
        learning.update_model(gpu)
        assert 'D' in active_model(cpu).classes and 'C' not in active_model(cpu).classes
        photo.unlink()
        learning.revert_staged_feedback(cpu,'changed')
        learning.update_model(cpu)
        assert 'C' in active_model(gpu).classes and 'D' not in active_model(gpu).classes


@pytest.mark.parametrize('pending_action',['correction','undo'])
def test_discard_pending_keeps_published_learning_and_original_undo(pair,tmp_path,pending_action):
    cpu,gpu,_=pair
    photo=tmp_path/'discard.jpg'
    _write_image(photo,120)
    with threadpool_limits(limits=1):
        stage(cpu,photo,'C','original')
        learning.update_model(cpu)
        if pending_action=='correction':
            stage(gpu,photo,'D','pending')
        else:
            learning.revert_staged_feedback(gpu,'original')
        assert learning.discard_pending(cpu)['pending_count']==0
        assert 'C' in active_model(gpu).classes
        photo.unlink()
        learning.revert_staged_feedback(gpu,'original')
        learning.update_model(cpu)
        assert 'C' not in active_model(gpu).classes


def test_missing_current_row_recovers_from_committed_learning(pair,tmp_path):
    cpu,gpu,_=pair
    one,two=tmp_path/'one.jpg',tmp_path/'two.jpg'
    _write_image(one,120)
    _write_image(two,220)
    with threadpool_limits(limits=1):
        stage(cpu,one,'C','one')
        learning.update_model(cpu)
        with learning._write_lock() as conn:
            conn.execute('DELETE FROM samples WHERE model=?',(learning._key(cpu),))
        one.unlink()
        stage(gpu,two,'D','two')
        learning.update_model(gpu)
        assert {'C','D'}<=set(active_model(cpu).classes)
