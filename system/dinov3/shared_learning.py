"""Stable learning-family identity and relocatable CPU/GPU learning assets.

Only the verified preview family (or an explicitly opted-in experiment) shares
state. Source deployment manifests and legacy feedback rows remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import zipfile

import numpy as np

from .encoder import sha256

PREVIEW_ENCODER='73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'
PREVIEW_BASE='52f5b703b2b552fcdac6c8289bfac927ebdd548c95f8ca13bd777177665219ed'
STATE_NAME='model.neri.json'


def checkpoint_info(path, payload):
    from .runtime import resolve_manifest_asset
    with zipfile.ZipFile(resolve_manifest_asset(path,payload['checkpoint'])) as archive:
        return json.loads(archive.read('manifest.json'))


def family_descriptor(path, payload=None):
    """Read identity only; catalog discovery must never migrate or write state."""
    from .runtime import resolve_manifest_asset
    path=Path(path).expanduser().resolve()
    try:
        payload=payload if payload is not None else json.loads(path.read_text('utf8'))
        if (payload.get('backend')!='dinov3' or payload.get('inference_recipe')!='experiment_full'
                or payload.get('architecture')!='dinov3_vitb16'
                or payload.get('preprocessing')!='letterbox224_imagenet'
                or payload.get('feature_dim')!=768):
            return None
        opted=payload.get('shared_learning') is True
        if not opted and payload.get('encoder_sha256')!=PREVIEW_ENCODER:
            return None
        if (not opted and payload.get('encoder_backend')=='litert'
                and payload.get('encoder_litert_sha256')!='dd2e4f5eb531512747e121a2e4149e0208ce9f37ce730305e76270267cd4971d'):
            return None
        if payload.get('learning_base') and payload.get('learning_base_sha256'):
            base=payload['learning_base']
            digest=payload['learning_base_sha256']
        elif payload.get('checkpoint_sha256')==PREVIEW_BASE:
            base,digest=payload['checkpoint'],payload['checkpoint_sha256']
        else:
            info=checkpoint_info(path,payload)
            original=info.get('metadata',{}).get('human_learning_base')
            base=(original or {}).get('checkpoint',payload['checkpoint'])
            digest=(original or {}).get('sha256',payload['checkpoint_sha256'])
        if not opted and digest!=PREVIEW_BASE:
            return None
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            return None
        base_path=resolve_manifest_asset(path,base)
        contract={key:payload[key] for key in ('encoder_sha256','architecture','preprocessing','feature_dim','inference_recipe')}
        contract['base_checkpoint_sha256']=digest
        identifier=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        claimed=payload.get('learning_family_id')
        if claimed is not None and claimed!=identifier:
            raise ValueError('模型共享学习身份与基础模型不一致。')
        return dict(id=identifier,key='family:'+identifier,base_path=base_path,base_sha256=digest,contract=contract)
    except (OSError,KeyError,TypeError,json.JSONDecodeError,zipfile.BadZipFile):
        return None


def family_id(path,payload=None):
    descriptor=family_descriptor(path,payload)
    return descriptor['id'] if descriptor else None


def family_manifest(state_root,identifier):
    return Path(state_root)/'learning-families'/identifier/STATE_NAME


def atomic_manifest(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_asset(source,target,digest):
    source,target=Path(source),Path(target)
    if sha256(source)!=digest:
        raise ValueError('共享学习源资产校验失败：'+str(source))
    if target.is_file():
        if sha256(target)!=digest:
            raise ValueError('共享学习资产校验失败：'+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    try:
        # These are immutable weight/base assets. A same-volume hard link also
        # survives removal/replacement of the original installation directory.
        try:
            os.link(source,temporary)
        except OSError:
            shutil.copy2(source,temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def calibration_identity(path):
    digest=hashlib.sha256()
    with np.load(path,allow_pickle=False) as arrays:
        for key in sorted(arrays.files):
            array=np.asarray(arrays[key])
            if array.dtype.hasobject:
                raise ValueError('校准包不能包含对象数组。')
            digest.update(json.dumps([key,array.dtype.str,array.shape],ensure_ascii=False).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def register_encoder(source,payload,target,current):
    from .runtime import resolve_manifest_asset
    target=Path(target)
    root=target.parent
    backend=payload.get('encoder_backend','pytorch')
    actual=payload['encoder_litert_sha256'] if backend=='litert' else payload['encoder_sha256']
    registered=current.get('learning_encoder_variants',{}).get(actual)
    if registered and (root/registered['encoder_weights']).is_file():
        updated=dict(current,**registered,display_name=payload.get('display_name',current.get('display_name')))
        if backend=='litert':
            updated.pop('encoder_code',None)
        else:
            updated.pop('encoder_litert_sha256',None)
        return updated
    weights=resolve_manifest_asset(source,payload['encoder_weights'])
    relative=f'encoders/{actual}/encoder'+('.tflite' if backend=='litert' else '.pth')
    _copy_asset(weights,root/relative,actual)
    updated=dict(current,encoder_backend=backend,encoder_weights=relative,
                 display_name=payload.get('display_name',current.get('display_name')))
    hashes=set(updated.get('learning_encoder_hashes',[]))
    hashes.add(actual)
    updated['learning_encoder_hashes']=sorted(hashes)
    if backend=='litert':
        updated['encoder_litert_sha256']=actual
        updated.pop('encoder_code',None)
    else:
        code=resolve_manifest_asset(source,payload['encoder_code'])
        code_relative=f'encoders/{actual}/encoder-code'
        if not (root/code_relative).is_dir():
            temporary=root/(code_relative+'.tmp-'+uuid.uuid4().hex)
            try:
                shutil.copytree(code,temporary,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                temporary.rename(root/code_relative)
            finally:
                if temporary.exists():
                    temporary.resolve().relative_to(root.resolve())
                    shutil.rmtree(temporary)
        updated['encoder_code']=code_relative
        updated.pop('encoder_litert_sha256',None)
    variants=dict(updated.get('learning_encoder_variants',{}))
    variants[actual]={key:updated[key] for key in ('encoder_backend','encoder_weights','encoder_code','encoder_litert_sha256') if key in updated}
    updated['learning_encoder_variants']=variants
    return updated


def initialize(source,payload,descriptor,state_root):
    """Prepare one family; the caller holds the learning SQLite write lock."""
    from .runtime import resolve_manifest_asset
    target=family_manifest(state_root,descriptor['id'])
    if target.is_file():
        existing=json.loads(target.read_text('utf8'))
        actual=family_descriptor(target,existing)
        if actual is None or actual['id']!=descriptor['id']:
            raise ValueError('共享学习目录中的模型身份不匹配。')
        return target,existing,False
    _copy_asset(descriptor['base_path'],target.parent/'base/model.neri-model',descriptor['base_sha256'])
    calibration=resolve_manifest_asset(source,payload['calibration'])
    _copy_asset(calibration,target.parent/'calibration/base.npz',payload['calibration_sha256'])
    current=dict(payload,shared_learning=True,learning_family_id=descriptor['id'],
        learning_base='base/model.neri-model',learning_base_sha256=descriptor['base_sha256'],
        learning_base_calibration_id=calibration_identity(calibration),
        checkpoint='base/model.neri-model',checkpoint_sha256=descriptor['base_sha256'],
        calibration='calibration/base.npz',calibration_sha256=payload['calibration_sha256'])
    if payload['checkpoint_sha256']!=descriptor['base_sha256']:
        from .model import DinoV3RidgeWithinModel
        checkpoint=resolve_manifest_asset(source,payload['checkpoint'])
        if sha256(checkpoint)!=payload['checkpoint_sha256']:
            raise ValueError('已学习模型校验失败，未迁移。')
        model=DinoV3RidgeWithinModel.load(checkpoint)
        model.metadata['human_learning_base']={'checkpoint':current['learning_base'],'sha256':descriptor['base_sha256']}
        generation=target.parent/'migration'/uuid.uuid4().hex
        generation.mkdir(parents=True)
        model.save(generation/'model.neri-model')
        current.update(checkpoint=(generation/'model.neri-model').relative_to(target.parent).as_posix(),
                       checkpoint_sha256=sha256(generation/'model.neri-model'))
    current=register_encoder(source,payload,target,current)
    atomic_manifest(target,current)
    return target,current,True


def peers(source,identifier):
    """Find already-installed compatible deployments, without installing anything."""
    source=Path(source).resolve()
    resource=next((p for p in source.parents if p.name.casefold()=='res'),None)
    if resource is None:
        return []
    candidates=[]
    for relative in ('model/cls/bundled','model/cls/user','model/cls/sync'):
        candidates.extend((resource/relative).glob('*.neri.json'))
    candidates.extend((resource/'model/DINOv3').rglob('*.neri.json'))
    return [p.resolve() for p in sorted(set(candidates)) if p.resolve()!=source and family_id(p)==identifier]


def encoder_compatible(data,payload):
    if data.get('encoder_sha256')!=payload['encoder_sha256']:
        return False
    if payload.get('learning_family_id'):
        actual=data.get('encoder_litert_sha256') or data['encoder_sha256']
        return actual in payload.get('learning_encoder_hashes',[])
    return data.get('encoder_litert_sha256')==payload.get('encoder_litert_sha256')
