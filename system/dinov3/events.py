"""Thirty-minute independent-event grouping for camera-trap observations."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
from typing import Iterable
EVENT_GAP_SECONDS=30*60
@dataclass(frozen=True)
class EventObservation:
    source_path:str; camera_id:str; identity_key:str; captured_at:datetime|None
@dataclass(frozen=True)
class IndependentEvent:
    key:str; camera_id:str; identity_key:str; started_at:datetime|None; ended_at:datetime|None
    observations:tuple[EventObservation,...]; timestamp_missing:bool=False
def camera_id_for_path(path:str|Path,root:str|Path|None=None)->str:
    source=Path(path).expanduser()
    camera_directory=source.parent.resolve()
    if root is not None:
        try:
            resolved_root=Path(root).expanduser().resolve()
            relative=source.resolve().relative_to(resolved_root)
            camera_directory=(resolved_root/relative.parts[0]).resolve() if len(relative.parts)>1 else resolved_root
        except (OSError,ValueError): pass
    normalized=os.path.normcase(str(camera_directory)).replace('\\','/')
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
def make_event_key(camera_id,identity_key,started_at,source_path):
    marker=started_at.isoformat() if started_at is not None else f"missing|{Path(source_path).resolve()}"
    return hashlib.sha256(f"{camera_id}|{identity_key}|{marker}".encode()).hexdigest()
def group_independent_events(observations:Iterable[EventObservation],*,gap_seconds:int=EVENT_GAP_SECONDS)->list[IndependentEvent]:
    grouped={}
    for obs in observations: grouped.setdefault((obs.camera_id,obs.identity_key),[]).append(obs)
    events=[]
    for (camera,identity),items in sorted(grouped.items()):
        missing=sorted((x for x in items if x.captured_at is None),key=lambda x:x.source_path)
        timed=sorted((x for x in items if x.captured_at is not None),key=lambda x:(x.captured_at,x.source_path))
        for item in missing:
            events.append(IndependentEvent(make_event_key(camera,identity,None,item.source_path),camera,identity,None,None,(item,),True))
        cluster=[]
        for item in timed:
            if cluster and (item.captured_at-cluster[-1].captured_at).total_seconds()>=gap_seconds:
                events.append(IndependentEvent(make_event_key(camera,identity,cluster[0].captured_at,cluster[0].source_path),camera,identity,cluster[0].captured_at,cluster[-1].captured_at,tuple(cluster)))
                cluster=[]
            cluster.append(item)
        if cluster: events.append(IndependentEvent(make_event_key(camera,identity,cluster[0].captured_at,cluster[0].source_path),camera,identity,cluster[0].captured_at,cluster[-1].captured_at,tuple(cluster)))
    return sorted(events,key=lambda e:(e.started_at is None,e.started_at or datetime.max,e.camera_id,e.identity_key,e.key))
