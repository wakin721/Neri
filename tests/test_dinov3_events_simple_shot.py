from datetime import datetime,timedelta
import hashlib
import os
from pathlib import Path
import numpy as np
from system.dinov3.events import EventObservation,camera_id_for_path,group_independent_events
from system.dinov3.simple_shot import build_prototype,deterministic_event_sample
BASE=datetime(2026,9,9,0,0,0)
def obs(minutes,name):return EventObservation(name,'cam-1','species-a',BASE+timedelta(minutes=minutes))
def test_29_minutes_59_seconds_stays_in_one_event():
    items=[EventObservation('a','cam','x',BASE),EventObservation('b','cam','x',BASE+timedelta(minutes=29,seconds=59))]
    assert len(group_independent_events(items))==1
def test_30_minutes_starts_a_new_event():
    assert len(group_independent_events([obs(0,'a'),obs(30,'b')]))==2
def test_different_identities_never_mix():
    items=[EventObservation('a','cam','cat',BASE),EventObservation('b','cam','boar',BASE+timedelta(minutes=1))]
    assert len(group_independent_events(items))==2
def test_camera_id_is_sha256_of_normalized_absolute_camera_directory(tmp_path):
    camera_root=tmp_path/'card-a'/'DCIM'
    camera_root.mkdir(parents=True)
    source=camera_root/'001.jpg'
    normalized=os.path.normcase(str(camera_root.resolve())).replace('\\','/')
    expected=hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    assert camera_id_for_path(source,camera_root)==expected

def test_same_named_camera_directories_at_different_paths_get_different_ids(tmp_path):
    left=tmp_path/'card-a'/'DCIM'
    right=tmp_path/'card-b'/'DCIM'
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    assert camera_id_for_path(left/'001.jpg',left)!=camera_id_for_path(right/'001.jpg',right)

def test_common_root_hashes_first_level_camera_directory(tmp_path):
    root=tmp_path/'survey'
    camera=root/'CAM01'
    nested=camera/'session-a'
    nested.mkdir(parents=True)
    first=camera/'001.jpg'
    second=nested/'002.jpg'
    assert camera_id_for_path(first,root)==camera_id_for_path(second,root)

def test_simple_shot_prototype_and_sampling_are_reproducible():
    values=np.eye(768,dtype=np.float32)[:4];p=build_prototype(values)
    assert np.isclose(np.linalg.norm(p),1.0)
    first=deterministic_event_sample(list(range(20)),count=5,seed_material='model|species|1')
    assert first==deterministic_event_sample(list(range(20)),count=5,seed_material='model|species|1')
