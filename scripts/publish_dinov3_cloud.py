"""Verify and publish the preview service/model with a reversible systemd drop-in.

Run --check as the SSH account first. --apply requires an administrator; it never
reads credentials or changes the old service code, OpenList, nginx, or model DB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile

MODEL_SHA256 = 'f7068b3a84ea32a42d3d89322eed79e750696086b8158fb028c46878bd95dbe1'
NAME = 'Neri1.2preview'
DROPIN = Path('/etc/systemd/system/neri-models.service.d/50-preview.conf')
ENV = Path('/etc/neri-models-preview.env')
PYTHON = '/opt/neri-models/venv/bin/python'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or '\x00' in name or name.startswith('/')
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Unsafe archive member')
    return PurePosixPath(name).as_posix()


def verify_members(archive, inventory, prefix=''):
    files = [info for info in archive.infolist() if not info.is_dir()]
    actual = [safe_name(info.filename) for info in files]
    if len(set(name.casefold() for name in actual)) != len(actual):
        raise ValueError('Duplicate archive member')
    if set(actual) != {prefix + name for name in inventory}:
        raise ValueError('Archive inventory differs')
    for name, expected in inventory.items():
        safe_name(name)
        info = archive.getinfo(prefix + name)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Archive symlink')
        with archive.open(info) as stream:
            actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if info.file_size != expected['size'] or actual_sha != expected['sha256']:
            raise ValueError('Archive checksum differs: ' + name)


def verify(model, bundle):
    if digest(model) != MODEL_SHA256:
        raise ValueError('Unexpected model archive SHA-256')
    with zipfile.ZipFile(model) as archive:
        inventory_name = NAME + '/package_inventory.json'
        receipt = json.loads(archive.read(inventory_name))
        if receipt['name'] != NAME or receipt['inference_recipe'] != 'experiment_full':
            raise ValueError('Unexpected model recipe')
        inventory = dict(receipt['files'])
        raw = archive.read(inventory_name)
        inventory['package_inventory.json'] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        verify_members(archive, inventory, NAME + '/')
    with zipfile.ZipFile(bundle) as archive:
        raw = archive.read('service_inventory.json')
        source = json.loads(raw)['files']
        inventory = dict(source)
        inventory['service_inventory.json'] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        verify_members(archive, inventory)
    return receipt, source


def unpack(archive_path, destination, prefix=''):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = safe_name(info.filename)
            if not name.startswith(prefix):
                raise ValueError('Unexpected prefix')
            target = destination / safe_name(name[len(prefix):])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)


def run(*args, **kwargs):
    return subprocess.run(args, check=True, timeout=120, **kwargs)


def snapshot(backup, path):
    name = backup/path.name
    if path.exists():
        shutil.copy2(path, name)
    else:
        (backup/(path.name+'.absent')).touch()


def restore(backup):
    for path in (DROPIN, ENV):
        saved = backup/path.name
        if saved.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, path)
        elif (backup/(path.name+'.absent')).is_file():
            path.unlink(missing_ok=True)
        else:
            raise ValueError('Incomplete rollback backup')
    run('systemctl', 'daemon-reload')
    run('systemctl', 'restart', 'neri-models')


def manifest(origin):
    with urllib.request.urlopen(origin + '/v1/dinov3/manifest', timeout=60) as response:
        return json.load(response)


def smoke(origin, receipt):
    value = manifest(origin)
    expected = dict(receipt['files'])
    # The publication also contains its self-describing inventory.
    actual = {row['path'].removeprefix('DINOv3/'): row for row in value['files']}
    if set(actual) != set(expected) | {'package_inventory.json'}:
        raise ValueError('Published file list differs')
    for name, entry in expected.items():
        if any(actual[name][key] != entry[key] for key in ('size', 'sha256')):
            raise ValueError('Published checksum differs: ' + name)
    entry = actual[NAME + '.neri.json']
    request = urllib.request.Request(origin + '/v1/dinov3/direct', data=json.dumps({
        'manifest_id': value['manifest_id'], 'path': entry['path'], 'sha256': entry['sha256'],
    }).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        capability = json.load(response)
    with urllib.request.urlopen(origin + '/v1/proxy/' + capability['proxy_token'], timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Published download failed')
    if json.loads(data)['display_name'] != NAME:
        raise ValueError('Published model name differs')
    return {'manifest_id': value['manifest_id'], 'files': len(actual),
            'total_bytes': sum(row['size'] for row in actual.values())}


def apply(model, bundle, receipt):
    import grp
    backup = Path('/var/backups') / ('neri-preview-' + time.strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(mode=0o700, exist_ok=False)
    snapshot(backup, DROPIN)
    snapshot(backup, ENV)
    code = Path('/opt/neri-models/releases') / ('preview-' + digest(bundle)[:16])
    release = Path('/var/lib/neri-models/releases') / (NAME + '-' + MODEL_SHA256[:16])
    # A new path for each attempt also tolerates an interrupted pre-activation copy.
    code = code.with_name(code.name + '-' + backup.name)
    release = release.with_name(release.name + '-' + backup.name)
    unpack(bundle, code)
    unpack(model, release, NAME + '/')
    gid = grp.getgrnam('neri-models').gr_gid
    for root in (code, release):
        # Root owns published files; the service can only read them.
        for path in (root, *root.rglob('*')):
            os.chown(path, 0, gid)
            path.chmod(0o750 if path.is_dir() else 0o640)
    for path in (code.parent, release.parent):
        os.chown(path, 0, gid)
        path.chmod(0o750)
    # Import only, with an isolated state directory and a placeholder secret.
    # The real EnvironmentFile is left private and is loaded only by systemd.
    with tempfile.TemporaryDirectory() as state:
        subprocess.run([PYTHON, '-c',
            'from pathlib import Path; from server.model_distribution.app import create_app; '
            'from server.model_distribution.config import DistributionConfig; '
            f'a=create_app(DistributionConfig(state_dir=Path({state!r}), '
            f'openlist_token="preflight", dinov3_root=Path({str(release)!r}))); '
            'assert "/v1/dinov3/manifest" in a.openapi()["paths"]'],
            cwd=code, check=True, timeout=120,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    try:
        ENV.write_text(f'NERI_DINOV3_ROOT={release}\nNERI_MODEL_REQUESTS_PER_MINUTE=1200\n', 'utf-8')
        ENV.chmod(0o600)
        DROPIN.parent.mkdir(parents=True, exist_ok=True)
        DROPIN.write_text(f'[Service]\nWorkingDirectory={code}\nEnvironmentFile={ENV}\n', 'utf-8')
        DROPIN.chmod(0o644)
        run('systemctl', 'daemon-reload')
        run('systemctl', 'restart', 'neri-models')
        for attempt in range(20):
            try:
                result = smoke('http://127.0.0.1:8002', receipt)
                break
            except (OSError, ValueError):
                if attempt == 19:
                    raise
                time.sleep(1)
        run('systemctl', 'is-active', '--quiet', 'neri-models')
    except BaseException:
        restore(backup)
        raise
    report = dict(passed=True, backup=str(backup), code=str(code), release=str(release), **result)
    (backup/'deployment.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--bundle', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    if (args.apply or args.rollback) and os.geteuid() != 0:
        parser.error('Administrator required; no password is read by this script.')
    if args.rollback:
        backup = args.rollback.resolve(strict=True)
        if backup.parent != Path('/var/backups') or not backup.name.startswith('neri-preview-'):
            parser.error('Unexpected rollback path')
        restore(backup)
        print(json.dumps({'restored': str(backup)}))
        return
    if args.model is None or args.bundle is None:
        parser.error('--model and --bundle are required')
    receipt, sources = verify(args.model, args.bundle)
    if args.check:
        print(json.dumps(dict(passed=True, model=NAME, known_species=receipt['known_species'],
                              model_files=len(receipt['files']), service_files=len(sources),
                              bundle_sha256=digest(args.bundle)), indent=2))
    else:
        print(json.dumps(apply(args.model, args.bundle, receipt), indent=2))


if __name__ == '__main__':
    main()
