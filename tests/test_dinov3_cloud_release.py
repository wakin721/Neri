"""Exercise cloud publication through the actual desktop download/installer."""
import hashlib
import io
import json
import ssl
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from server.model_distribution.app import create_app
from server.model_distribution.config import DistributionConfig
from server.model_distribution.local_component import LocalComponentOverlay, LOCAL_SCHEME
from server.model_distribution.service import DistributionService
from server.model_distribution.storage import StorageError, UpstreamLink
from system.dinov3.component import dinov3_component_status, install_dinov3_component
from system.dinov3.distribution import DinoV3DistributionClient, DinoV3DistributionError
from system.dinov3.runtime import read_manifest, resolve_manifest_asset
from system.model_sync.catalog import discover_models
from system.model_sync.layout import get_model_layout
from tests.test_dinov3_experiment import experiment
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_preview_package import prepare
from scripts.package_dinov3_preview import package
from tests.test_model_distribution import FakeStore


class LoopbackOpener:
    def __init__(self, client):
        self.client = client
        self.corrupt = False

    def __call__(self, request, timeout):
        response = self.client.request(
            request.method, request.full_url, content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code,
                                         response.text, response.headers, None)
        body = response.content
        if self.corrupt and '/proxy/' in request.full_url:
            body = b'x' * len(body)
        stream = io.BytesIO(body)
        stream.status = response.status_code
        stream.headers = response.headers
        return stream


@pytest.fixture
def cloud(experiment, tmp_path):
    manifest = prepare(experiment)
    output = tmp_path/'preview.zip'
    package(manifest, output)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(tmp_path/'published')
    manifest = tmp_path/'published'/'Neri1.2preview'/'Neri1.2preview.neri.json'
    old = FakeStore({
        '/Neri_Data/Model/tracker.yaml': {'data': b'tracker', 'modified': '1'},
        '/Neri_Data/Model/DINOv2/legacy.pt': {'data': b'old', 'modified': '2'},
    })
    config = DistributionConfig(state_dir=tmp_path/'state', openlist_token='test',
                                requests_per_minute=1200, dinov3_root=manifest.parent)
    with patch('server.model_distribution.app.OpenListModelStore', return_value=old):
        app = create_app(config=config)
    with TestClient(app) as http:
        opener = LoopbackOpener(http)
        client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener)
        yield manifest, http, client, opener


def test_cloud_download_installs_full_experiment_and_keeps_legacy_routes(cloud, tmp_path):
    source, http, client, _ = cloud
    manifest = client.fetch_manifest()
    assert any(row.relative_path.endswith('calibration.npz') for row in manifest.files)
    first = manifest.files[0]
    cap = http.post('/v1/dinov3/direct', json={
        'manifest_id': manifest.manifest_id, 'path': first.path, 'sha256': first.sha256,
    }).json()
    assert cap['direct_url'] is None and cap['direct_headers'] == {}
    assert LOCAL_SCHEME not in json.dumps(cap)
    updates = []
    layout = get_model_layout(tmp_path/'installed'/'res')
    root = layout.dinov3_root
    status = install_dinov3_component(root=root, distribution_client=client,
                                      on_progress=lambda *args: updates.append(args))
    assert status['healthy'] and status['display_name'] == 'Neri1.2preview'
    assert status['class_count'] == 2
    assert updates[-1][0] == 100
    for entry in manifest.files:
        data = (root/entry.relative_path).read_bytes()
        assert len(data) == entry.size and hashlib.sha256(data).hexdigest() == entry.sha256
    installed = read_manifest(root/status['manifest'])
    assert installed['inference_recipe'] == 'experiment_full'
    assert installed['checkpoint_sha256'] == read_manifest(source)['checkpoint_sha256']
    discovered = discover_models(layout, 'cls')
    assert any(row.name == 'Neri1.2preview' and row.source == 'sync' for row in discovered)
    legacy = http.get('/v1/dinov2/manifest').json()
    assert legacy['files'][0]['path'] == 'DINOv2/legacy.pt'
    direct = http.post('/v1/dinov2/direct', json={
        'manifest_id': legacy['manifest_id'], **{k: legacy['files'][0][k] for k in ('path', 'sha256')},
    })
    assert direct.status_code == 200
    ordinary = http.get('/v1/manifest').json()
    assert [row['path'] for row in ordinary['files']] == ['tracker.yaml']


def test_corrupt_cloud_download_leaves_installed_component_untouched(cloud, tmp_path):
    _, _, client, opener = cloud
    root = tmp_path/'installed'
    install_dinov3_component(root=root, distribution_client=client)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    opener.corrupt = True
    with pytest.raises(DinoV3DistributionError, match='sha256'):
        install_dinov3_component(root=root, distribution_client=client)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not list(root.parent.glob('.dinov3-stage-*'))


def test_calibration_corruption_marks_component_unhealthy(cloud, tmp_path):
    _, _, client, _ = cloud
    root = tmp_path/'installed'
    status = install_dinov3_component(root=root, distribution_client=client)
    manifest = root/status['manifest']
    resolve_manifest_asset(manifest, read_manifest(manifest)['calibration']).write_bytes(b'bad')
    status = dinov3_component_status(root=root)
    assert not status['healthy'] and '校准集 SHA-256' in status['message']


def test_failed_activation_restores_old_component(cloud, tmp_path):
    _, _, client, _ = cloud
    root = tmp_path/'installed'
    install_dinov3_component(root=root, distribution_client=client)
    before = (root/dinov3_component_status(root=root)['manifest']).read_bytes()
    original = Path.replace

    def fail_staging(path, target):
        if path.name.startswith('.dinov3-stage-'):
            raise OSError('activation failed')
        return original(path, target)

    with patch.object(Path, 'replace', fail_staging), pytest.raises(OSError, match='activation failed'):
        install_dinov3_component(root=root, distribution_client=client)
    assert (root/dinov3_component_status(root=root)['manifest']).read_bytes() == before


def test_local_files_support_ranges_and_reject_escape(tmp_path):
    root = tmp_path/'release'
    root.mkdir()
    (root/'encoder.pth').write_bytes(b'0123456789')
    store = LocalComponentOverlay(FakeStore({}), root)
    link = store.resolve_link('/Neri_Data/Model/DINOv3/encoder.pth')
    assert b''.join(store.iter_bytes(link, 'bytes=2-5')) == b'2345'
    assert b''.join(store.iter_bytes(link, 'bytes=7-')) == b'789'
    for path in ('../private', 'assets/../encoder.pth', 'a\\b', 'C:/private', 'a//b'):
        with pytest.raises(StorageError):
            store.resolve_link('/Neri_Data/Model/DINOv3/'+path)
    for value in ('bytes=9-1', 'bytes=10-', 'bytes=-3', 'bytes=1-2,4-5'):
        with pytest.raises(StorageError):
            b''.join(store.iter_bytes(link, value))


def test_proxy_range_uses_real_local_bytes(cloud):
    _, http, client, _ = cloud
    manifest = client.fetch_manifest()
    entry = next(row for row in manifest.files if row.size > 20)
    cap = http.post('/v1/dinov3/direct', json={
        'manifest_id': manifest.manifest_id, 'path': entry.path, 'sha256': entry.sha256,
    }).json()
    url = '/v1/proxy/'+cap['proxy_token']
    response = http.get(url, headers={'Range': 'bytes=10-19'})
    assert response.status_code == 206 and len(response.content) == 10
    assert response.headers['content-range'] == f'bytes 10-19/{entry.size}'
    whole = http.get(url).content
    assert response.content == whole[10:20]


@pytest.mark.parametrize('failure', ['tls', 'short'])
def test_segmented_download_retries_only_failed_segment_and_verifies_file(cloud, tmp_path, monkeypatch, failure):
    import system.dinov3.distribution as distribution
    _, _, client, _ = cloud
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 131072)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 65536)
    original = client.opener
    failed = False
    ranges = []

    def interrupted(request, timeout):
        nonlocal failed
        selected = request.get_header('Range')
        if selected:
            ranges.append(selected)
        response = original(request, timeout)
        if not failed and selected == 'bytes=65536-131071':
            failed = True
            if failure == 'tls':
                response.close()
                raise ssl.SSLError('record layer failure')
            truncated = io.BytesIO(response.read(100))
            truncated.status, truncated.headers = response.status, response.headers
            response.close()
            return truncated
        return response

    client.opener = interrupted
    root = tmp_path/'segmented'
    status = install_dinov3_component(root=root, distribution_client=client)
    assert failed and status['healthy']
    assert ranges.count('bytes=65536-131071') > 1
    for entry in client.fetch_manifest().files:
        data = (root/entry.relative_path).read_bytes()
        assert len(data) == entry.size and hashlib.sha256(data).hexdigest() == entry.sha256
    assert not list(root.rglob('*.download'))


def test_segmented_download_rejects_ignored_range_without_replacing_installed_model(cloud, tmp_path, monkeypatch):
    import system.dinov3.distribution as distribution
    _, _, client, _ = cloud
    root = tmp_path/'installed'
    install_dinov3_component(root=root, distribution_client=client)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 131072)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 65536)
    original = client.opener

    def ignore_range(request, timeout):
        response = original(request, timeout)
        if request.get_header('Range'):
            response.status = 200
        return response

    client.opener = ignore_range
    with pytest.raises(DinoV3DistributionError, match='range_mismatch'):
        install_dinov3_component(root=root, distribution_client=client)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_python_tls_failure_switches_to_native_transport_for_remaining_segments(cloud, tmp_path, monkeypatch):
    import system.dinov3.distribution as distribution
    _, _, client, _ = cloud
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 131072)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 65536)
    monkeypatch.setattr(distribution.shutil, 'which', lambda _: 'curl.exe')
    original = client.opener
    client._native_fallback = True
    native_ranges = []
    python_ranges = []

    def broken_tls(request, timeout):
        if request.get_header('Range'):
            python_ranges.append(request.get_header('Range'))
            raise ssl.SSLError('record layer failure')
        return original(request, timeout)

    def native(url, start, end, size, directory):
        import urllib.request
        native_ranges.append((start, end))
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'}, method='GET')
        with original(request, 90) as response:
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
            return response.read()

    client.opener = broken_tls
    monkeypatch.setattr(client, '_native_proxy_segment', native)
    status = install_dinov3_component(root=tmp_path/'native-installed', distribution_client=client)
    assert status['healthy'] and native_ranges
    # One Python range per large file; later ranges use the selected transport.
    assert len(python_ranges) == sum(start == 0 for start, _ in native_ranges)


@pytest.mark.parametrize('status,content_range', [('206', 'bytes 10-19/100'),
                                                ('200', 'bytes 10-19/100'),
                                                ('206', 'bytes 0-9/100')])
def test_native_transport_keeps_certificate_checks_and_validates_response(tmp_path, monkeypatch, status, content_range):
    import system.dinov3.distribution as distribution
    from types import SimpleNamespace
    client = DinoV3DistributionClient()
    monkeypatch.setattr(distribution.shutil, 'which', lambda _: 'curl.exe')

    def run(command, **options):
        assert command[1] == '--disable'
        assert '--insecure' not in command and '-k' not in command
        assert options['stdin'] == distribution.subprocess.DEVNULL
        assert not options.get('shell', False)
        Path(command[command.index('--output')+1]).write_bytes(b'0123456789')
        Path(command[command.index('--dump-header')+1]).write_text(
            f'HTTP/1.1 {status}\r\nContent-Range: {content_range}\r\n\r\n')
        return SimpleNamespace(returncode=0, stdout=status.encode(), stderr=b'')

    monkeypatch.setattr(distribution.subprocess, 'run', run)
    if status == '206' and content_range == 'bytes 10-19/100':
        assert client._native_proxy_segment('https://example.invalid/file', 10, 19, 100, tmp_path) == b'0123456789'
    else:
        with pytest.raises(DinoV3DistributionError, match='range_mismatch'):
            client._native_proxy_segment('https://example.invalid/file', 10, 19, 100, tmp_path)


def test_backup_cleanup_permission_error_keeps_healthy_install_and_records_deferred_cleanup(cloud, tmp_path, monkeypatch):
    import system.dinov3.component as component

    _, _, client, _ = cloud
    root = tmp_path / 'DINOv3'
    install_dinov3_component(root=root, distribution_client=client)
    backup = root.with_name('DINOv3.old')
    real_rmtree = component.shutil.rmtree

    def locked_rmtree(path, *args, **kwargs):
        if Path(path) == backup:
            raise PermissionError(13, 'Access is denied', str(Path(path) / 'dinov3-vitb16'))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(component.shutil, 'rmtree', locked_rmtree)
    events = []
    status = install_dinov3_component(root=root, distribution_client=client, on_event=events.append)
    assert status['healthy'] and dinov3_component_status(root=root)['healthy']
    assert backup.is_dir() and (backup / status['manifest']).is_file()
    assert any(
        event.get('phase') == 'cleanup'
        and event.get('state') == 'deferred'
        and 'deferred' in str(event.get('message', '')).lower()
        and str(backup) in str(event.get('path', ''))
        for event in events
    )


def test_reinstall_uses_unique_backup_suffix_when_existing_old_backup_is_locked(cloud, tmp_path, monkeypatch):
    import system.dinov3.component as component

    source, _, _, _ = cloud
    root = tmp_path / 'DINOv3'
    install_dinov3_component(root=root, seed_dir=source.parent)
    old_backup = root.with_name('DINOv3.old')
    (old_backup / 'dinov3-vitb16').mkdir(parents=True)
    (old_backup / 'dinov3-vitb16' / 'user-backup.txt').write_bytes(b'keep-old')
    second_backup = root.with_name('DINOv3.old.1')
    second_backup.mkdir()
    (second_backup / 'marker.txt').write_bytes(b'keep-1')
    locked = {old_backup, second_backup}
    real_rmtree = component.shutil.rmtree
    replaced_targets = []
    original_replace = Path.replace

    def locked_rmtree(path, *args, **kwargs):
        if Path(path) in locked:
            raise PermissionError(13, 'Access is denied', str(path))
        return real_rmtree(path, *args, **kwargs)

    def record_replace(path, target):
        if path == root:
            replaced_targets.append(Path(target))
        return original_replace(path, target)

    monkeypatch.setattr(component.shutil, 'rmtree', locked_rmtree)
    monkeypatch.setattr(Path, 'replace', record_replace)
    events = []
    status = install_dinov3_component(root=root, seed_dir=source.parent, on_event=events.append)
    assert status['healthy'] and dinov3_component_status(root=root)['healthy']
    assert replaced_targets == [root.with_name('DINOv3.old.2')]
    assert (old_backup / 'dinov3-vitb16' / 'user-backup.txt').read_bytes() == b'keep-old'
    assert (second_backup / 'marker.txt').read_bytes() == b'keep-1'
    assert not root.with_name('DINOv3.old.2').exists()
    assert any(event.get('phase') == 'cleanup' and event.get('state') == 'deferred' for event in events)


def test_failed_activation_restores_component_and_preserves_locked_old_backup(cloud, tmp_path, monkeypatch):
    import system.dinov3.component as component

    source, _, _, _ = cloud
    root = tmp_path / 'DINOv3'
    install_dinov3_component(root=root, seed_dir=source.parent)
    (root / 'learned-marker.txt').write_bytes(b'learned')
    old_backup = root.with_name('DINOv3.old')
    (old_backup / 'dinov3-vitb16').mkdir(parents=True)
    (old_backup / 'dinov3-vitb16' / 'user-backup.txt').write_bytes(b'keep-old')
    real_rmtree = component.shutil.rmtree
    original_replace = Path.replace

    def locked_rmtree(path, *args, **kwargs):
        if Path(path) == old_backup:
            raise PermissionError(13, 'Access is denied', str(path))
        return real_rmtree(path, *args, **kwargs)

    def fail_activation(path, target):
        if path.name.startswith('.dinov3-stage-'):
            raise OSError('activation failed')
        return original_replace(path, target)

    monkeypatch.setattr(component.shutil, 'rmtree', locked_rmtree)
    monkeypatch.setattr(Path, 'replace', fail_activation)
    events = []
    with pytest.raises(OSError, match='activation failed'):
        install_dinov3_component(root=root, seed_dir=source.parent, on_event=events.append)
    assert (root / 'learned-marker.txt').read_bytes() == b'learned'
    assert dinov3_component_status(root=root)['healthy']
    assert (old_backup / 'dinov3-vitb16' / 'user-backup.txt').read_bytes() == b'keep-old'
    assert not root.with_name('DINOv3.old.1').exists()
    assert not list(root.parent.glob('.dinov3-stage-*'))
