"""Stage verified CPU assets and activate a reversible NeriCloud CPU release.

Preparation runs as the SSH account. Activation requires an administrator and
preserves the existing GPU release, service credentials, and model database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

try:
    from . import publish_dinov3_cloud as legacy
except ImportError:
    import publish_dinov3_cloud as legacy


MANIFEST = 'Neri1.2preview.neri.json'
LICENSE = 'Neri1.2preview-litert.LICENSE.md'


def _file(root, relative):
    relative = legacy.safe_name(relative)
    target = Path(root).joinpath(*relative.split('/'))
    target.resolve().relative_to(Path(root).resolve())
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != Path(root) and Path(root) in p.parents):
        raise ValueError('Release symlink')
    return target


def _identity(path):
    return dict(size=path.stat().st_size, sha256=legacy.digest(path))


def verify_release(root):
    root = Path(root)
    receipt = json.loads((root / 'package_inventory.json').read_text('utf8'))
    expected = receipt['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(expected) | {'package_inventory.json'}:
        raise ValueError('Release inventory differs')
    for name, identity in expected.items():
        if _identity(_file(root, name)) != identity:
            raise ValueError('Release checksum mismatch: ' + name)
    payload = json.loads((root / MANIFEST).read_text('utf8'))
    if payload.get('encoder_backend') != 'litert':
        raise ValueError('CPU release must use LiteRT')
    return receipt


def prepare(index, assets, destination):
    index, assets, destination = Path(index), Path(assets), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source = json.loads(index.read_text('utf8'))
    payload = source['manifest']
    if payload.get('encoder_backend') != 'litert':
        raise ValueError('CPU release must use LiteRT')
    rows = source['assets']
    if len(rows) != 3 or {Path(row['path']).name for row in rows} != {
            'encoder.tflite', 'model.neri-model', 'calibration.npz'}:
        raise ValueError('Unexpected CPU asset inventory')
    for row in rows:
        legacy.safe_name(row['path'])
        path = _file(assets, Path(row['path']).name)
        if _identity(path) != {k: row[k] for k in ('size', 'sha256')}:
            raise ValueError('Source checksum mismatch: ' + path.name)
    license_path = _file(index.parent, LICENSE)
    if not license_path.is_file():
        raise ValueError('Missing DINOv3 license')
    destination.mkdir(parents=True)
    for row in rows:
        target = _file(destination, row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets / Path(row['path']).name, target)
    shutil.copy2(license_path, destination / LICENSE)
    (destination / MANIFEST).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', 'utf8')
    files = {p.relative_to(destination).as_posix(): _identity(p)
             for p in destination.rglob('*') if p.is_file()}
    known_species = None
    checkpoint = _file(destination, payload['checkpoint'])
    if zipfile.is_zipfile(checkpoint):
        with zipfile.ZipFile(checkpoint) as archive:
            known_species = len(json.loads(archive.read('manifest.json'))['classes'])
    receipt = dict(name=payload.get('display_name'), runtime='cpu', files=files,
                   inference_recipe=payload.get('inference_recipe'), known_species=known_species)
    (destination / 'package_inventory.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', 'utf8')
    return verify_release(destination)


def _json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def activate(source, bundle):
    if not hasattr(os, 'geteuid') or os.geteuid() != 0:
        raise PermissionError('Administrator required for service activation')
    import grp

    source, bundle = Path(source), Path(bundle)
    receipt = verify_release(source)
    with zipfile.ZipFile(bundle) as archive:
        raw = archive.read('service_inventory.json')
        inventory = dict(json.loads(raw)['files'])
        inventory['service_inventory.json'] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        legacy.verify_members(archive, inventory)
    origin = 'http://127.0.0.1:8002'
    before_gpu = _json(origin + '/v1/dinov3/manifest')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    backup = Path('/var/backups') / ('neri-litert-' + stamp)
    backup.mkdir(mode=0o700, exist_ok=False)
    for path in (legacy.DROPIN, legacy.ENV):
        legacy.snapshot(backup, path)
    code = Path('/opt/neri-models/releases') / ('litert-' + legacy.digest(bundle)[:12] + '-' + stamp)
    release = Path('/var/lib/neri-models/releases') / ('litert-cpu-' + stamp)
    legacy.unpack(bundle, code)
    shutil.copytree(source, release, symlinks=False)
    verify_release(release)
    gid = grp.getgrnam('neri-models').gr_gid
    for root in (code, release):
        for path in (root, *root.rglob('*')):
            os.chown(path, 0, gid)
            path.chmod(0o750 if path.is_dir() else 0o640)
    try:
        previous = legacy.ENV.read_text('utf8')
        lines = [line for line in previous.splitlines() if not line.startswith('NERI_DINOV3_CPU_ROOT=')]
        legacy.ENV.write_text('\n'.join(lines) + '\nNERI_DINOV3_CPU_ROOT=' + str(release) + '\n', 'utf8')
        legacy.ENV.chmod(0o600)
        legacy.DROPIN.write_text(f'[Service]\nWorkingDirectory={code}\nEnvironmentFile={legacy.ENV}\n', 'utf8')
        legacy.run('systemctl', 'daemon-reload')
        legacy.run('systemctl', 'restart', 'neri-models')
        for attempt in range(20):
            try:
                cpu = _json(origin + '/v1/dinov3/manifest?runtime=cpu')
                break
            except OSError:
                if attempt == 19:
                    raise
                time.sleep(1)
        expected = dict(receipt['files'])
        expected['package_inventory.json'] = _identity(release / 'package_inventory.json')
        actual = {row['path'].removeprefix('DINOv3/cpu/'): {k: row[k] for k in ('size', 'sha256')}
                  for row in cpu['files']}
        if actual != expected:
            raise ValueError('Published CPU inventory differs')
        if _json(origin + '/v1/dinov3/manifest') != before_gpu:
            raise ValueError('GPU manifest changed during CPU activation')
        legacy.run('systemctl', 'is-active', '--quiet', 'neri-models')
    except BaseException:
        legacy.restore(backup)
        raise
    result = dict(passed=True, backup=str(backup), code=str(code), release=str(release),
                  cpu_manifest_id=cpu['manifest_id'], gpu_manifest_unchanged=True,
                  cpu_bytes=sum(row['size'] for row in actual.values()))
    (backup / 'deployment.json').write_text(json.dumps(result, indent=2) + '\n', 'utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', type=Path)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.index, args.assets, args.root)
        result = dict(staged=True, files=len(result['files']), root=str(args.root))
    elif args.check:
        result = dict(verified=True, files=len(verify_release(args.root)['files']))
    elif args.rollback:
        legacy.restore(args.rollback)
        result = dict(restored=True, backup=str(args.rollback))
    else:
        result = activate(args.root, args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
