"""Discovery and compatibility resolution for user and synchronized models."""
from __future__ import annotations
from dataclasses import dataclass
import json, os
from pathlib import Path
from typing import Literal, Sequence
from .layout import ModelLayout
ModelKind=Literal['detect','cls']; ModelSource=Literal['user','sync']; ModelBackend=Literal['yolo','dinov3']
_DETECT_EXTENSIONS=frozenset({'.pt'}); _CLS_EXTENSIONS=frozenset({'.pt','.onnx','.engine'}); _MANIFEST_SUFFIX='.neri.json'
@dataclass(frozen=True)
class DiscoveredModel:
    name:str; path:str; size_bytes:int|None; source:ModelSource; kind:ModelKind
    backend:ModelBackend='yolo'; architecture:str|None=None; feature_dim:int|None=None
    requires_detector:bool=False; supports_video_fast:bool=True; supports_video_all:bool=True
    checkpoint_path:str|None=None
def _size(path):
    try:return path.stat().st_size
    except OSError:return None
def _manifest_model(path:Path,source:ModelSource):
    try:payload=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,UnicodeError,json.JSONDecodeError):return None
    if not isinstance(payload,dict) or payload.get('backend')!='dinov3':return None
    name=payload.get('checkpoint')
    if not isinstance(name,str) or not name.strip():return None
    checkpoint=(path.parent/name).resolve()
    if not checkpoint.is_file():return None
    dim=payload.get('feature_dim');dim=dim if isinstance(dim,int) else None
    return DiscoveredModel(str(payload.get('display_name') or path.name),str(path.resolve()),_size(checkpoint),source,'cls','dinov3',str(payload.get('architecture') or '') or None,dim,bool(payload.get('requires_detector',True)),bool(payload.get('supports_video_fast',True)),bool(payload.get('supports_video_all',False)),str(checkpoint)),checkpoint
def _scan(directory:Path,source:ModelSource,kind:ModelKind):
    extensions=_DETECT_EXTENSIONS if kind=='detect' else _CLS_EXTENSIONS
    if not directory.is_dir():return []
    items=[];referenced=set()
    if kind=='cls':
        for manifest in sorted(directory.glob(f'*{_MANIFEST_SUFFIX}'),key=lambda p:p.name.casefold()):
            resolved=_manifest_model(manifest,source)
            if resolved:items.append(resolved[0]);referenced.add(resolved[1])
    for path in sorted(directory.iterdir(),key=lambda p:p.name.casefold()):
        if path.is_file() and path.suffix.lower() in extensions and path.resolve() not in referenced:
            items.append(DiscoveredModel(path.name,str(path.resolve()),_size(path),source,kind))
    return sorted(items,key=lambda x:x.name.casefold())
def discover_models(layout:ModelLayout,kind:ModelKind):
    if kind=='detect':dirs=((layout.detect_user,'user'),(layout.detect_sync,'sync'))
    elif kind=='cls':dirs=((layout.cls_user,'user'),(layout.cls_sync,'sync'))
    else:raise ValueError(f'Unsupported model kind: {kind}')
    models=[]
    for directory,source in dirs:models.extend(_scan(directory,source,kind))
    return models
def _saved_path_matches(saved,current):
    if saved==current:return True
    return os.name=='nt' and os.path.normcase(os.path.normpath(saved))==os.path.normcase(os.path.normpath(current))
def resolve_saved_model_path(saved:object,models:Sequence[DiscoveredModel])->str|None:
    if not isinstance(saved,str) or not saved.strip():return None
    value=saved.strip()
    for model in models:
        if _saved_path_matches(value,model.path) or (model.checkpoint_path and _saved_path_matches(value,model.checkpoint_path)):return model.path
    named=[m for m in models if m.name==value or Path(m.path).name==value]
    if not named:return None
    return next((m.path for m in named if m.source=='user'),named[0].path)
