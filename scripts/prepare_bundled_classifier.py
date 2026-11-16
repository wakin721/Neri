"""Explicit developer tool to reassemble local Neri1.2preview FP32 classifier shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid

ROOT=Path(__file__).resolve().parents[1]
NAME='Neri1.2preview-litert'
MANIFEST=NAME+'.neri.json'
ASSET_DIR=NAME+'-assets'
SOURCE_DIR=ROOT/'res/bundled-models/Neri1.2preview-litert-fp32'
DESTINATION=ROOT/'res/model/cls/bundled'
CHUNK_BYTES=80*1024**2
FILES={'encoder':ASSET_DIR+'/encoder.tflite','checkpoint':ASSET_DIR+'/model.neri-model',
       'calibration':ASSET_DIR+'/calibration.npz'}
SIDECARS=(NAME+'.LICENSE.md',NAME+'.SOURCE.md')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_path(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid bundled asset path')
    parts=relative.split('/')
    if PurePosixPath(relative).is_absolute() or any(p in {'','.','..'} for p in parts):
        raise ValueError('Unsafe bundled asset path')
    result=Path(root).joinpath(*parts)
    result.resolve().relative_to(Path(root).resolve())
    if result.is_symlink():
        raise ValueError('Bundled assets cannot be symlinks')
    return result


def load_index(source=SOURCE_DIR):
    source=Path(source)
    index=json.loads((source/'assets.json').read_text('utf8'))
    if index.get('schema_version')!=1 or index.get('name')!=NAME:
        raise ValueError('Unsupported builtin classifier source index')
    assets=index.get('assets',[])
    if len(assets)!=3 or {item.get('path') for item in assets}!=set(FILES.values()):
        raise ValueError('Unexpected builtin classifier asset allowlist')
    manifest=index.get('manifest',{})
    if (manifest.get('backend')!='dinov3' or manifest.get('encoder_backend')!='litert'
            or manifest.get('encoder_weights')!=FILES['encoder']
            or manifest.get('checkpoint')!=FILES['checkpoint']
            or manifest.get('calibration')!=FILES['calibration']):
        raise ValueError('Builtin classifier manifest contract differs')
    hashes={item['path']:item['sha256'] for item in assets}
    for key,field in [('encoder','encoder_litert_sha256'),('checkpoint','checkpoint_sha256'),('calibration','calibration_sha256')]:
        if manifest.get(field)!=hashes[FILES[key]]:
            raise ValueError('Builtin manifest and source asset checksums differ')
    seen=set()
    for item in assets:
        safe_path(source,item['path'])
        if not isinstance(item['size'],int) or item['size']<1 or not item.get('parts'):
            raise ValueError('Invalid builtin source asset')
        if sum(part['size'] for part in item['parts'])!=item['size']:
            raise ValueError('Builtin chunk sizes differ from asset size')
        for part in item['parts']:
            safe_path(source,part['path'])
            if part['path'] in seen or not 0<part['size']<=CHUNK_BYTES:
                raise ValueError('Duplicate or oversized builtin source chunk')
            seen.add(part['path'])
    return index


def prepare(source=SOURCE_DIR,destination=DESTINATION):
    source,destination=Path(source),Path(destination)
    index=load_index(source)
    destination.mkdir(parents=True,exist_ok=True)
    for item in index['assets']:
        target=safe_path(destination,item['path'])
        # Validate the checked-in source even when an assembled output exists.
        for part in item['parts']:
            path=safe_path(source,part['path'])
            if not path.is_file() or path.stat().st_size!=part['size'] or sha(path)!=part['sha256']:
                raise ValueError('Missing or corrupt builtin classifier chunk: '+part['path'])
        if target.is_file() and target.stat().st_size==item['size'] and sha(target)==item['sha256']:
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as output:
                for part in item['parts']:
                    with safe_path(source,part['path']).open('rb') as stream:
                        shutil.copyfileobj(stream,output,1024**2)
            if temporary.stat().st_size!=item['size'] or sha(temporary)!=item['sha256']:
                raise ValueError('Reassembled builtin classifier checksum mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    for name in SIDECARS:
        shutil.copy2(safe_path(source,name),safe_path(destination,name))
    manifest_path=destination/MANIFEST
    temporary=manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(index['manifest'],ensure_ascii=False,indent=2)+'\n','utf8')
    temporary.replace(manifest_path)
    return dict(passed=True,manifest=str(manifest_path.resolve()),assets=len(index['assets']),
                bytes=sum(item['size'] for item in index['assets']),offline=True)


def import_source(experiment,original_manifest,source=SOURCE_DIR):
    """Explicit maintainer operation to split a local classifier export into chunks."""
    experiment,original_manifest,source=Path(experiment),Path(original_manifest),Path(source)
    if source.exists() and any(source.iterdir()):
        raise FileExistsError('Import into an empty source directory')
    original=json.loads(original_manifest.read_text('utf8'))
    exported=json.loads((experiment/'Neri1.2preview-litert.json').read_text('utf8'))
    inputs={'encoder':experiment/'dinov3_vitb16_fp32.tflite',
            'checkpoint':experiment/'model.neri-model','calibration':experiment/'calibration.npz'}
    expected={'encoder':exported['encoder_fp32_sha256'],
              'checkpoint':original['checkpoint_sha256'],'calibration':original['calibration_sha256']}
    for key,path in inputs.items():
        if sha(path)!=expected[key]:
            raise ValueError('Experiment input checksum mismatch: '+key)
    payload=dict(original,display_name='Neri1.2preview·LiteRT',encoder_backend='litert',
        encoder_weights=FILES['encoder'],encoder_litert_sha256=expected['encoder'],
        checkpoint=FILES['checkpoint'],calibration=FILES['calibration'])
    payload.pop('encoder_code',None)
    source.mkdir(parents=True,exist_ok=True)
    assets=[]
    for key,path in inputs.items():
        item=dict(path=FILES[key],size=path.stat().st_size,sha256=expected[key],parts=[])
        with path.open('rb') as stream:
            number=0
            while data:=stream.read(CHUNK_BYTES):
                name=key+f'.part{number:03}'
                (source/name).write_bytes(data)
                item['parts'].append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
                number+=1
        assets.append(item)
    code=safe_path(original_manifest.parent,original['encoder_code'])
    shutil.copy2(code/'LICENSE.md',source/SIDECARS[0])
    (source/SIDECARS[1]).write_text(
        '# Neri1.2preview LiteRT FP32 CPU\n\n'
        'DINOv3 ViT-B/16 encoder converted from the official frozen pretrained weights.\n'
        'The classifier, full calibration set and experiment_full event recipe are unchanged.\n'
        'Local verification: 14,477 evaluation crops / 1,605 events; FP32 classification and acceptance agree with PyTorch.\n'
        f'Original encoder SHA-256: {original["encoder_sha256"]}\n'
        f'Converted FP32 encoder SHA-256: {expected["encoder"]}\n'
        'DINOv3 remains subject to the accompanying official license.\n'
        'Source chunks are kept below 80 MiB for optional local developer reassembly; release builds download via NeriCloud.\n','utf8')
    (source/'assets.json').write_text(json.dumps(dict(schema_version=1,name=NAME,manifest=payload,
                                                   assets=assets),ensure_ascii=False,indent=2)+'\n','utf8')
    load_index(source)
    return dict(passed=True,source=str(source.resolve()),chunks=sum(len(item['parts']) for item in assets))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE_DIR)
    parser.add_argument('--destination',type=Path,default=DESTINATION)
    parser.add_argument('--import-from',type=Path)
    parser.add_argument('--original-manifest',type=Path,default=ROOT/'res/model/cls/user/dinov3-experiment-local.neri.json')
    args=parser.parse_args()
    result=(import_source(args.import_from,args.original_manifest,args.source) if args.import_from
            else prepare(args.source,args.destination))
    print(json.dumps(result,ensure_ascii=False,indent=2))
