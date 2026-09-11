from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import pytest
from system.dinov3.registry import RegistrationConditionError,SpeciesRegistry,registry_path_for_fingerprint
FP='a'*64
BASE=datetime(2026,9,9,0,0,0)
def vector(index=0):
    v=np.zeros(768,dtype=np.float32);v[index]=1;return v
def test_registry_is_scoped_by_model_fingerprint(tmp_path):
    path=registry_path_for_fingerprint(tmp_path,FP);reg=SpeciesRegistry(path,model_fingerprint=FP);reg.close()
    with pytest.raises(ValueError,match='fingerprint'):SpeciesRegistry(path,model_fingerprint='b'*64)
def test_four_events_gate_explicit_provisional_registration(tmp_path):
    reg=SpeciesRegistry(tmp_path/'r.db',model_fingerprint=FP,consistency_threshold=.5)
    entry=None
    for i in range(4):entry=reg.record_unknown(vector(),camera_id='cam-0',captured_at=BASE+timedelta(hours=i),source_path=f'{i}.jpg')
    assert entry.status=='candidate';assert entry.event_count==4;assert entry.prototype_count==1;assert not entry.can_register
    entry=reg.set_identity(entry.id,common_name='豹猫',scientific_name='Prionailurus bengalensis')
    assert entry.event_count==4;assert entry.status=='candidate';assert entry.can_register
    registered=reg.register(entry.id);assert registered.status=='provisional';assert registered.display_name.startswith('豹猫')
    reg.close()
def test_same_identity_within_30_minutes_counts_once(tmp_path):
    reg=SpeciesRegistry(tmp_path/'r.db',model_fingerprint=FP)
    entry=reg.record_unknown(vector(),camera_id='cam',captured_at=BASE,source_path='a.jpg')
    entry=reg.record_observation(entry.id,vector(),camera_id='cam',captured_at=BASE+timedelta(minutes=29),source_path='b.jpg')
    assert entry.event_count==1
    entry=reg.record_observation(entry.id,vector(),camera_id='cam',captured_at=BASE+timedelta(minutes=59),source_path='c.jpg')
    assert entry.event_count==2;reg.close()
def test_register_refuses_failed_conditions(tmp_path):
    reg=SpeciesRegistry(tmp_path/'r.db',model_fingerprint=FP);entry=reg.record_unknown(vector(),camera_id='cam',captured_at=BASE,source_path='a.jpg')
    with pytest.raises(RegistrationConditionError):reg.register(entry.id)
    reg.close()

def test_unnamed_candidates_are_listed_by_event_count_descending(tmp_path):
    reg=SpeciesRegistry(tmp_path/'r.db',model_fingerprint=FP)
    first=reg.record_unknown(vector(100),camera_id='cam-a',captured_at=BASE,source_path='a-0.jpg')
    second=reg.record_unknown(vector(101),camera_id='cam-b',captured_at=BASE,source_path='b-0.jpg')
    third=reg.record_unknown(vector(102),camera_id='cam-c',captured_at=BASE,source_path='c-0.jpg')
    reg.record_observation(second.id,vector(101),camera_id='cam-b',captured_at=BASE+timedelta(hours=1),source_path='b-1.jpg')
    reg.record_observation(second.id,vector(101),camera_id='cam-b',captured_at=BASE+timedelta(hours=2),source_path='b-2.jpg')
    reg.record_observation(third.id,vector(102),camera_id='cam-c',captured_at=BASE+timedelta(hours=1),source_path='c-1.jpg')
    unnamed=[entry for entry in reg.list() if entry.status=='candidate' and not entry.common_name]
    assert [entry.id for entry in unnamed]==[second.id,third.id,first.id]
    assert [entry.event_count for entry in unnamed]==[3,2,1]
    reg.close()

def test_set_identity_immediately_merges_same_named_candidates(tmp_path):
    reg=SpeciesRegistry(tmp_path/'r.db',model_fingerprint=FP)
    first=reg.record_unknown(vector(200),camera_id='cam-a',captured_at=BASE,source_path='camel-a.jpg')
    second=reg.record_unknown(vector(201),camera_id='cam-b',captured_at=BASE+timedelta(hours=1),source_path='camel-b.jpg')
    first=reg.set_identity(first.id,common_name='骆驼')
    merged=reg.set_identity(second.id,common_name='骆驼')
    named=[entry for entry in reg.list() if entry.common_name=='骆驼']
    assert merged.id==first.id
    assert [entry.id for entry in named]==[first.id]
    assert named[0].event_count==2
    reg.close()
