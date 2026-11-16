import asyncio
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.requests import Request

from server.model_distribution.app import create_app, DinoV2DirectRequest, DinoV3DirectRequest
from server.model_distribution.config import DistributionConfig
from server.model_distribution.local_component import LocalComponentOverlay, PREFIX
from server.model_distribution.service import DistributionService, DistributionError
from server.model_distribution.storage import StorageError, UpstreamLink
from system.dinov3.distribution import DinoV3DistributionClient
from tests.test_model_distribution import FakeStore, RecordingBudget


def service_for(tmp_path):
    root = tmp_path / 'release'
    (root / 'assets').mkdir(parents=True)
    (root / 'Neri1.2preview.neri.json').write_text('{"display_name":"Neri1.2preview"}')
    (root / 'assets/model.bin').write_bytes(b'0123456789abcdef')
    upstream = FakeStore({'/Neri_Data/Model/detect/lite.tflite': {'data': b'lite', 'modified': '1'}})
    overlay = LocalComponentOverlay(upstream, root)
    return DistributionService(tmp_path / 'state', overlay), root


def test_local_release_client_download_hashes_progress_and_range(tmp_path):
    service, root = service_for(tmp_path)
    budget = RecordingBudget()
    app = create_app(DistributionConfig(tmp_path / 'state'), service=service, budget=budget)
    routes = {getattr(route, 'path', ''): route for route in app.routes}
    request = Request({'type': 'http', 'headers': [], 'client': ('203.0.113.10', 123)})
    issued = []

    # Exercise the real desktop downloader against the real API handlers,
    # capability store, and byte stream (no optional HTTP test dependency).
    def opener(req, timeout):
        path = req.full_url.removeprefix('http://127.0.0.1:8002')
        if path == '/v1/dinov3/manifest':
            result = routes[path].endpoint(request)
        elif path == '/v1/dinov3/direct':
            payload = DinoV3DirectRequest(**json.loads(req.data))
            result = routes[path].endpoint(payload, request)
            issued.append(result)
        else:
            response = routes['/v1/proxy/{token}'].endpoint(path.rsplit('/', 1)[-1], request)
            async def read():
                return b''.join([chunk async for chunk in response.body_iterator])
            return io.BytesIO(asyncio.run(read()))
        return io.BytesIO(json.dumps(result).encode())

    progress = []
    target = tmp_path / 'installed'
    client = DinoV3DistributionClient('http://127.0.0.1:8002/v1', opener=opener)
    client.download_tree('', target, on_progress=lambda p, m: progress.append(p))
    for path in root.rglob('*'):
        if path.is_file():
            assert (target / path.relative_to(root)).read_bytes() == path.read_bytes()
    assert progress[-1] == 100 and progress == sorted(progress)
    assert all(item['direct_url'] is None and item['direct_headers'] == {} for item in issued)
    assert 'neri-local' not in json.dumps(issued)
    assert str(root) not in json.dumps(issued)
    snapshot = service.dinov3_manifest()
    entry = next(e for e in snapshot.files if e.path.endswith('model.bin'))
    cap = service.dinov3_direct(snapshot.manifest_id, entry.path, entry.sha256)
    range_request = Request({'type': 'http', 'headers': [(b'range', b'bytes=2-6')], 'client': ('203.0.113.10', 123)})
    response = routes['/v1/proxy/{token}'].endpoint(cap['proxy_token'], range_request)
    async def read_range():
        return b''.join([chunk async for chunk in response.body_iterator])
    assert asyncio.run(read_range()) == b'23456'
    assert response.status_code == 206
    assert response.headers['content-range'] == 'bytes 2-6/16'
    assert budget.proxy_reservations[-1] == ('203.0.113.10', 5)
    assert service.manifest().files[0].path == 'detect/lite.tflite'
    with pytest.raises(DistributionError, match='model_not_in_manifest'):
        service.dinov3_direct(snapshot.manifest_id, 'DINOv3/../secret', 'a' * 64)


@pytest.mark.parametrize('relative', ['../secret', 'assets/../../secret', '/assets/model.bin',
                                      'assets//model.bin', 'assets/./model.bin', 'assets\\model.bin', 'C:/secret'])
def test_local_release_rejects_unsafe_paths(tmp_path, relative):
    service, _ = service_for(tmp_path)
    with pytest.raises(StorageError):
        service.store.resolve_link(PREFIX + '/' + relative)


def test_local_release_rejects_symlink_and_ranges(tmp_path):
    service, root = service_for(tmp_path)
    link = service.store.resolve_link(PREFIX + '/assets/model.bin')
    assert b''.join(service.store.iter_bytes(link, 'bytes=5-')) == b'56789abcdef'
    assert b''.join(service.store.iter_bytes(link, 'bytes=10-999')) == b'abcdef'
    for value in ('bytes=-1', 'bytes=1-2,4-5', 'bytes=16-', 'bytes=5-4'):
        with pytest.raises(StorageError):
            list(service.store.iter_bytes(link, value))
    # Use a mock for the OS-specific symlink flag; containment is tested above.
    original = Path.is_symlink
    with patch.object(Path, 'is_symlink', lambda p: p == root / 'assets' or original(p)):
        with pytest.raises(StorageError, match='symlink'):
            service.store.resolve_link(PREFIX + '/assets/model.bin')


def test_local_release_overlay_is_selected_by_server_config(tmp_path):
    _, root = service_for(tmp_path)
    config = DistributionConfig(tmp_path / 'config-state', dinov3_root=root)
    app = create_app(config)
    routes = {getattr(route, 'path', ''): route for route in app.routes}
    request = Request({'type': 'http', 'headers': [], 'client': ('127.0.0.1', 123)})
    payload = routes['/v1/dinov3/manifest'].endpoint(request)
    assert len(payload['files']) == 2
    assert all(item['path'].startswith('DINOv3/') for item in payload['files'])


def test_legacy_dinov2_still_has_its_own_manifest_and_capabilities(tmp_path):
    service, _ = service_for(tmp_path)
    service.store.upstream.files['/Neri_Data/Model/DINOv2/legacy.pt'] = {'data': b'old', 'modified': '1'}
    app = create_app(DistributionConfig(tmp_path / 'state'), service=service, budget=RecordingBudget())
    routes = {getattr(route, 'path', ''): route for route in app.routes}
    request = Request({'type': 'http', 'headers': [], 'client': ('127.0.0.1', 123)})
    payload = routes['/v1/dinov2/manifest'].endpoint(request)
    assert [e['path'] for e in payload['files']] == ['DINOv2/legacy.pt']
    entry = payload['files'][0]
    result = routes['/v1/dinov2/direct'].endpoint(DinoV2DirectRequest(
        manifest_id=payload['manifest_id'], path=entry['path'], sha256=entry['sha256']), request)
    assert service.consume_proxy(result['proxy_token']).path == entry['path']
    assert result['sha256'] == hashlib.sha256(b'old').hexdigest()
    with pytest.raises(DistributionError):
        service.dinov3_direct(payload['manifest_id'], entry['path'], entry['sha256'])


def test_cpu_and_gpu_releases_have_independent_manifests_and_capabilities(tmp_path):
    from fastapi.testclient import TestClient

    gpu = tmp_path / 'gpu'
    cpu = tmp_path / 'cpu'
    gpu.mkdir()
    cpu.mkdir()
    (gpu / 'encoder.pth').write_bytes(b'gpu-weights')
    (cpu / 'encoder.tflite').write_bytes(b'cpu-weights')
    config = DistributionConfig(tmp_path / 'state', openlist_token='test',
                                dinov3_root=gpu, dinov3_cpu_root=cpu)
    with TestClient(create_app(config)) as client:
        gpu_manifest = client.get('/v1/dinov3/manifest').json()
        cpu_manifest = client.get('/v1/dinov3/manifest?runtime=cpu').json()
        assert [e['path'] for e in gpu_manifest['files']] == ['DINOv3/encoder.pth']
        assert [e['path'] for e in cpu_manifest['files']] == ['DINOv3/cpu/encoder.tflite']
        assert client.get('/v1/dinov3/manifest?runtime=unknown').status_code == 422
        # Fetching the other variant must not invalidate outstanding capabilities.
        for snapshot, expected in ((gpu_manifest, b'gpu-weights'), (cpu_manifest, b'cpu-weights')):
            entry = snapshot['files'][0]
            response = client.post('/v1/dinov3/direct', json={
                'manifest_id': snapshot['manifest_id'], 'path': entry['path'],
                'sha256': entry['sha256']})
            assert response.status_code == 200
            capability = response.json()
            assert capability['direct_url'] is None
            assert client.get('/v1/proxy/' + capability['proxy_token']).content == expected
        entry = cpu_manifest['files'][0]
        assert client.post('/v1/dinov3/direct', json={
            'manifest_id': gpu_manifest['manifest_id'], 'path': entry['path'],
            'sha256': entry['sha256']}).status_code == 409


def test_cpu_cloud_staging_checks_all_assets_before_publishing(tmp_path):
    from scripts.publish_dinov3_litert_cloud import prepare, verify_release

    assets = tmp_path / 'assets'
    metadata = tmp_path / 'metadata'
    assets.mkdir()
    metadata.mkdir()
    inventory = []
    for name in ('encoder.tflite', 'model.neri-model', 'calibration.npz'):
        data = ('verified-' + name).encode()
        (assets / name).write_bytes(data)
        inventory.append(dict(path='assets/' + name, size=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
    payload = dict(encoder_backend='litert', display_name='Neri1.2preview·LiteRT',
                   checkpoint='assets/model.neri-model', encoder_weights='assets/encoder.tflite',
                   calibration='assets/calibration.npz')
    index = metadata / 'assets.json'
    index.write_text(json.dumps(dict(manifest=payload, assets=inventory)), encoding='utf8')
    (metadata / 'Neri1.2preview-litert.LICENSE.md').write_text('license')
    release = tmp_path / 'published'
    prepare(index, assets, release)
    receipt = verify_release(release)
    assert receipt['name'] == 'Neri1.2preview·LiteRT'
    assert json.loads((release / 'Neri1.2preview.neri.json').read_text('utf8')) == payload
    (assets / 'calibration.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        prepare(index, assets, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
    (release / 'unexpected.bin').write_bytes(b'not in receipt')
    with pytest.raises(ValueError, match='inventory'):
        verify_release(release)
