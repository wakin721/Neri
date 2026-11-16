"""Publish both verified preview variants to OneDrive and enable direct links.

Run --check unprivileged; --start requires administrator access, runs in the
background, and keeps all OpenList credentials on the server. Existing local
release trees remain the manifest source and proxy fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

try:
    from . import publish_dinov3_cloud as legacy
except ImportError:
    import publish_dinov3_cloud as legacy

SERVICE = 'http://127.0.0.1:8002'
MIRROR_BASE = '/Neri_Data/Model/DINOv3-mirrors'
ENV_KEYS = {'gpu': 'NERI_DINOV3_GPU_DRIVE_ROOT', 'cpu': 'NERI_DINOV3_CPU_DRIVE_ROOT'}


def read_env(path):
    result = {}
    for line in Path(path).read_text('utf8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if sep:
            values = shlex.split(value, comments=True)
            if len(values) > 1:
                raise ValueError('Unsupported environment file syntax')
            result[key.strip()] = values[0] if values else ''
    return result


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVICE + path, data=data,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def onedrive_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or '').lower()
    return (parsed.scheme == 'https' and not parsed.username and not parsed.password
            and parsed.port in (None, 443)
            and host.endswith(('.1drv.com', '.sharepoint.com', '.microsoftpersonalcontent.com')))


def release_plan(snapshot, source, runtime):
    prefix = 'DINOv3/cpu/' if runtime == 'cpu' else 'DINOv3/'
    source = Path(source).resolve(strict=True)
    rows = []
    for row in snapshot['files']:
        if not row['path'].startswith(prefix):
            raise ValueError('Wrong runtime manifest')
        relative = legacy.safe_name(row['path'][len(prefix):])
        path = source / relative
        path.resolve(strict=True).relative_to(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Invalid model source: ' + relative)
        if path.stat().st_size != row['size'] or legacy.digest(path) != row['sha256']:
            raise ValueError('Model source checksum mismatch: ' + relative)
        rows.append(dict(row, relative=relative, source=str(path)))
    if not rows or len({r['relative'].casefold() for r in rows}) != len(rows):
        raise ValueError('Invalid release inventory')
    return rows


class Drive:
    def __init__(self):
        config = read_env('/etc/neri-models.env')
        self.origin = config.get('NERI_OPENLIST_URL', 'http://127.0.0.1:5244').rstrip('/')
        address = urllib.parse.urlsplit(self.origin)
        if (address.scheme != 'http' or address.hostname not in {'localhost', '127.0.0.1'}
                or address.username or address.password or address.path):
            raise ValueError('Expected loopback OpenList')
        self.address = address
        self.token = config['NERI_OPENLIST_TOKEN']
        self.cache = {}

    def call(self, endpoint, payload):
        for attempt in range(5):
            req = urllib.request.Request(self.origin+'/api/fs/'+endpoint,
                data=json.dumps(payload).encode(),
                headers={'Authorization': self.token, 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    value = json.load(response)
                if value.get('code') != 200:
                    raise RuntimeError('OpenList '+endpoint+' failed')
                return value.get('data')
            except OSError:
                if attempt == 4:
                    raise RuntimeError('OpenList '+endpoint+' unavailable') from None
                time.sleep(min(8, 2**attempt))

    def listing(self, path, *, refresh=False):
        if path in self.cache and not refresh:
            return self.cache[path]
        value = self.call('list', dict(path=path, password='', page=1, per_page=0, refresh=True))
        rows = value.get('content') or []
        if len(rows) != value.get('total'):
            raise ValueError('Incomplete drive listing')
        self.cache[path] = {r['name']: r for r in rows}
        return self.cache[path]

    def mkdir(self, path):
        parts = PurePosixPath(path).parts
        for index in range(4, len(parts)+1):
            current = '/' + '/'.join(parts[1:index])
            parent, name = current.rsplit('/', 1)
            if name not in self.listing(parent):
                self.call('mkdir', {'path': current})
                self.cache.pop(parent, None)

    def link(self, path):
        value = self.call('link', dict(path=path, password='')) or {}
        url = value.get('url') or value.get('link') or value.get('raw_url') or ''
        if not onedrive_url(url):
            raise ValueError('Drive did not return an allowed OneDrive link')
        return url

    def matches(self, remote, row):
        parent, name = remote.rsplit('/', 1)
        entry = self.listing(parent).get(name)
        if entry is None or entry.get('is_dir') or entry.get('size') != row['size']:
            return False
        with urllib.request.urlopen(self.link(remote), timeout=180) as response:
            actual = hashlib.file_digest(response, 'sha256').hexdigest()
        return actual == row['sha256']

    def upload(self, source, remote):
        connection = http.client.HTTPConnection(self.address.hostname,
            self.address.port or 80, timeout=3600)
        try:
            with Path(source).open('rb') as stream:
                connection.request('PUT', '/api/fs/put', stream, {
                    'Authorization': self.token, 'Content-Type': 'application/octet-stream',
                    'Content-Length': str(Path(source).stat().st_size),
                    'File-Path': urllib.parse.quote(remote, safe=''), 'As-Task': 'false'})
                response = connection.getresponse()
                value = json.loads(response.read(1024**2))
            if response.status != 200 or value.get('code') != 200:
                raise RuntimeError('OpenList upload failed')
        finally:
            connection.close()

    def copy(self, source, remote):
        src, name = source.rsplit('/', 1)
        dst, destination_name = remote.rsplit('/', 1)
        if name != destination_name:
            raise ValueError('Copy filenames differ')
        self.call('copy', dict(src_dir=src, dst_dir=dst, names=[name], overwrite=False))


def publish(plans, snapshots, status, *, reuse_cache=None):
    drive = Drive()
    verified = {}
    previous = {}
    if reuse_cache is not None and Path(reuse_cache).is_file():
        for sha, size, name, remote in json.loads(Path(reuse_cache).read_text('utf8')):
            if (remote.startswith(MIRROR_BASE+'/Neri1.2preview-')
                    and legacy.safe_name(remote.lstrip('/'))):
                previous[(sha,size,name)] = remote
    roots = {}
    done_bytes = 0
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for runtime in ('gpu', 'cpu'):
        root = (MIRROR_BASE + '/Neri1.2preview-' + runtime + '-'
                + snapshots[runtime]['manifest_id'][:12] + '-' + stamp)
        roots[runtime] = root
        drive.mkdir(root)
        for row in sorted(plans[runtime], key=lambda r: r['size'], reverse=True):
            remote = root + '/' + row['relative']
            parent, name = remote.rsplit('/', 1)
            drive.mkdir(parent)
            existing = drive.listing(parent).get(name)
            status(phase='checking', runtime=runtime, file=row['relative'], verified_bytes=done_bytes)
            if existing is not None:
                if not drive.matches(remote, row):
                    raise ValueError('Refusing to overwrite mismatched mirror file: '+row['relative'])
            else:
                key = (row['sha256'], row['size'], name)
                reusable = verified.get(key)
                if reusable is None and key in previous:
                    try:
                        if drive.matches(previous[key], row):
                            reusable = previous[key]
                    except (OSError, RuntimeError, ValueError):
                        pass
                # The existing OneDrive GPU component often already has the same encoder.
                if reusable is None and runtime == 'gpu':
                    candidate = '/Neri_Data/Model/' + row['path']
                    try:
                        if drive.matches(candidate, row):
                            reusable = candidate
                    except (OSError, RuntimeError, ValueError):
                        pass
                if reusable:
                    status(phase='copying', runtime=runtime, file=row['relative'], verified_bytes=done_bytes)
                    drive.copy(reusable, remote)
                else:
                    status(phase='uploading', runtime=runtime, file=row['relative'],
                           file_bytes=row['size'], verified_bytes=done_bytes)
                    drive.upload(row['source'], remote)
                deadline = time.monotonic()+(300 if reusable else 3600)
                while True:
                    ready = drive.listing(parent, refresh=True).get(name)
                    if ready and not ready.get('is_dir') and ready.get('size') == row['size']:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('OneDrive file publication timed out')
                    time.sleep(2)
                if not drive.matches(remote, row):
                    raise ValueError('Published OneDrive checksum mismatch: '+row['relative'])
            verified[(row['sha256'], row['size'], name)] = remote
            if reuse_cache is not None:
                previous.update(verified)
                cache_path = Path(reuse_cache)
                temporary = cache_path.with_suffix('.tmp')
                temporary.write_text(json.dumps([list(key)+[path]
                    for key,path in previous.items()]), 'utf8')
                temporary.chmod(0o644)
                temporary.replace(cache_path)
            done_bytes += row['size']
            status(phase='verified', runtime=runtime, file=row['relative'], verified_bytes=done_bytes)
    return roots


def wait_snapshot(suffix=''):
    for attempt in range(20):
        try:
            return request('/v1/dinov3/manifest'+suffix)
        except OSError:
            if attempt == 19:
                raise
            time.sleep(1)


def activate(bundle, roots, before, status):
    import grp
    with zipfile.ZipFile(bundle) as archive:
        raw = archive.read('service_inventory.json')
        inventory = dict(json.loads(raw)['files'])
        inventory['service_inventory.json'] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        legacy.verify_members(archive, inventory)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    backup = Path('/var/backups') / ('neri-onedrive-'+stamp)
    backup.mkdir(mode=0o700)
    for path in (legacy.DROPIN, legacy.ENV):
        legacy.snapshot(backup, path)
    code = Path('/opt/neri-models/releases') / ('onedrive-'+legacy.digest(bundle)[:12]+'-'+stamp)
    legacy.unpack(bundle, code)
    gid = grp.getgrnam('neri-models').gr_gid
    for path in (code, *code.rglob('*')):
        os.chown(path, 0, gid)
        path.chmod(0o750 if path.is_dir() else 0o640)
    try:
        lines = [line for line in legacy.ENV.read_text('utf8').splitlines()
                 if line.partition('=')[0] not in set(ENV_KEYS.values())]
        lines.extend(ENV_KEYS[r]+'='+roots[r] for r in ('gpu', 'cpu'))
        legacy.ENV.write_text('\n'.join(lines)+'\n', 'utf8')
        legacy.ENV.chmod(0o600)
        legacy.DROPIN.write_text(f'[Service]\nWorkingDirectory={code}\nEnvironmentFile={legacy.ENV}\n', 'utf8')
        legacy.run('systemctl', 'daemon-reload')
        legacy.run('systemctl', 'restart', 'neri-models')
        for runtime, snapshot in before.items():
            suffix = '?runtime=cpu' if runtime == 'cpu' else ''
            current = wait_snapshot(suffix)
            if current != snapshot:
                raise ValueError('Manifest changed during mirror activation')
            entry = next(e for e in current['files'] if e['path'].endswith('encoder.tflite' if runtime=='cpu' else 'encoder.pth'))
            capability = request('/v1/dinov3/direct', dict(manifest_id=current['manifest_id'],
                path=entry['path'], sha256=entry['sha256']))
            if not onedrive_url(capability.get('direct_url') or ''):
                raise ValueError('Active service did not offer OneDrive '+runtime+' link')
            headers = dict(capability.get('direct_headers') or {})
            headers['Range'] = 'bytes=0-1048575'
            req = urllib.request.Request(capability['direct_url'], headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 206 or len(response.read()) != 1048576:
                    raise ValueError('Active OneDrive range download failed')
        legacy.run('systemctl', 'is-active', '--quiet', 'neri-models')
    except BaseException:
        legacy.restore(backup)
        raise
    result = dict(passed=True, roots=roots, backup=str(backup), code=str(code),
                  gpu_direct=True, cpu_direct=True, manifests_unchanged=True)
    (backup/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
    status(state='completed', phase='completed', **result)
    return result


def run(args):
    import fcntl
    job = Path(__file__).resolve().parent
    lock = (job/'publish.lock').open('a')
    # A second invocation must not overwrite the running job's status.
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = {'state':'running'}
    def status(**fields):
        state.update(fields, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        temporary = job/'onedrive-status.tmp'
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n', 'utf8')
        temporary.chmod(0o644)
        temporary.replace(job/'onedrive-status.json')
        print(json.dumps(fields,ensure_ascii=False),flush=True)
    try:
        config = read_env(legacy.ENV)
        snapshots = {r:request('/v1/dinov3/manifest'+('?runtime=cpu' if r=='cpu' else ''))
                     for r in ('gpu', 'cpu')}
        plans = {r:release_plan(snapshots[r], config['NERI_DINOV3_CPU_ROOT' if r=='cpu' else 'NERI_DINOV3_ROOT'], r)
                 for r in snapshots}
        status(total_bytes=sum(row['size'] for rows in plans.values() for row in rows), phase='validated')
        roots = publish(plans, snapshots, status, reuse_cache=job/'onedrive-verified.json')
        status(phase='activating')
        return activate(args.bundle, roots, snapshots, status)
    except Exception as exc:
        # Neither OpenList credentials nor signed OneDrive URLs enter logs.
        reason = re.sub(r'https?://[^\s]+', '[URL hidden]', str(exc))[:500]
        status(state='failed', phase='failed', error_type=type(exc).__name__, reason=reason)
        raise RuntimeError('OneDrive publication failed; inspect safe status and retry') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', type=Path)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    if args.rollback:
        if not hasattr(os, 'geteuid') or os.geteuid()!=0:
            parser.error('Administrator required')
        backup = args.rollback.resolve(strict=True)
        if backup.parent != Path('/var/backups') or not backup.name.startswith('neri-onedrive-'):
            parser.error('Unexpected rollback backup')
        legacy.restore(backup)
        print(json.dumps(dict(restored=True,backup=str(backup))))
        return
    if args.bundle is None:
        parser.error('--bundle is required')
    if args.check:
        with zipfile.ZipFile(args.bundle) as archive:
            raw = archive.read('service_inventory.json')
            inventory = dict(json.loads(raw)['files'])
            inventory['service_inventory.json'] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            legacy.verify_members(archive, inventory)
        print(json.dumps(dict(verified=True,bundle_sha256=legacy.digest(args.bundle))))
        return
    if not hasattr(os,'geteuid') or os.geteuid()!=0:
        parser.error('Administrator required; credentials stay on server')
    if args.start:
        job = Path(__file__).resolve().parent
        log = (job/'onedrive-publish.log').open('ab')
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--apply',
            '--bundle',str(args.bundle.resolve())],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=job)
        print(json.dumps(dict(started=True,pid=process.pid,status=str(job/'onedrive-status.json'))))
        return
    run(args)


if __name__ == '__main__':
    main()
