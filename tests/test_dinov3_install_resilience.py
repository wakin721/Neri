import hashlib
import io
import json
import ssl
import urllib.error
from pathlib import Path

import pytest

from tests.test_dinov3_cloud_release import cloud
from tests.test_dinov3_experiment import experiment
from tests.test_dinov3_deployment import deployed

import system.dinov3.distribution as distribution
from system.dinov3.distribution import (DinoV3DistributionClient, DinoV3DistributionEntry,
    DinoV3DistributionManifest, DinoV3DistributionError)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(distribution.time, 'sleep', waits.append)
    return waits


def test_metadata_retries_transient_failure_and_records_sanitized_reason(no_wait):
    calls, events = [], []
    def opener(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.URLError(TimeoutError('temporary network failure'))
        return io.BytesIO(b'{"ok": true}')
    client = DinoV3DistributionClient(opener=opener, on_event=events.append)
    assert client._json_request('POST', '/dinov3/direct', {'path':'DINOv3/assets/encoder.pth'}) == {'ok':True}
    assert len(calls) == 3 and no_wait == [1, 2]
    failures = [event for event in events if event['state'] == 'retrying']
    assert all(row['file'] == 'DINOv3/assets/encoder.pth' for row in failures)
    assert failures[0]['causes'][0]['type'] == 'URLError'
    assert failures[0]['causes'][1]['type'] == 'TimeoutError'


def test_http_429_honors_retry_after_without_logging_tokens(no_wait):
    events, calls = [], []
    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, 'Too Many Requests',
                {'Retry-After':'3'}, io.BytesIO(json.dumps({
                    'detail':'request_rate_limited', 'proxy_token':'SECRET_CAPABILITY',
                    'direct_url':'https://example.test/x?sig=SECRET_SIGNATURE'}).encode()))
        return io.BytesIO(b'{}')
    client = DinoV3DistributionClient(opener=opener, on_event=events.append)
    client._json_request('POST', '/dinov3/direct')
    assert no_wait == [3]
    serialized = json.dumps(events)
    assert 'SECRET_CAPABILITY' not in serialized and 'SECRET_SIGNATURE' not in serialized
    assert events[1]['http_status'] == 429
    assert events[1]['server_detail'] == 'request_rate_limited'


@pytest.mark.parametrize('status', [401, 403, 404, 409, 422])
def test_permanent_http_errors_keep_status_without_blind_retry(status, no_wait):
    calls = []
    def opener(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, status, 'failure', {},
                                    io.BytesIO(b'{"detail":"stale_manifest"}'))
    client = DinoV3DistributionClient(opener=opener)
    with pytest.raises(DinoV3DistributionError) as caught:
        client._json_request('POST', '/dinov3/direct')
    assert len(calls) == 1 and no_wait == []
    assert caught.value.details['http_status'] == status


def test_metadata_tls_failure_can_use_windows_transport(monkeypatch):
    def opener(request, timeout):
        raise ssl.SSLError('TLS record failure')
    client = DinoV3DistributionClient(opener=opener)
    client._native_fallback = True
    monkeypatch.setattr(distribution.shutil, 'which', lambda _: 'curl.exe')
    native = []
    def native_response(request):
        native.append(request.get_method())
        return b'{"ok":true}'
    monkeypatch.setattr(client, '_native_json_response', native_response)
    assert client._json_request('POST', '/dinov3/direct') == {'ok':True}
    assert native == ['POST']


def test_transient_refresh_failure_retries_without_downloading_previous_segments_again(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    network = Network()
    network.fail_refresh = False
    failed = False
    def opener(request, timeout):
        nonlocal failed
        if request.get_method() == 'POST' and len(network.ranges) == 3 and not failed:
            failed = True
            raise urllib.error.URLError(TimeoutError('refresh failed once'))
        return network(request, timeout)
    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener)
    client.fetch_manifest = network.manifest
    target = tmp_path/'installed'
    client.download_tree('', target)
    assert failed and (target/'b.bin').read_bytes() == b'abcd'*4
    assert network.ranges == ['bytes=0-3','bytes=4-7','bytes=8-11','bytes=12-15']


class Network:
    def __init__(self):
        self.files = {'DINOv3/a.json':b'{}', 'DINOv3/b.bin':b'abcd'*4}
        self.tokens = {}
        self.ranges = []
        self.capabilities = []
        self.fail_refresh = True

    def manifest(self):
        return DinoV3DistributionManifest('a'*64, tuple(
            DinoV3DistributionEntry(name, name[7:], len(data), hashlib.sha256(data).hexdigest())
            for name, data in self.files.items()))

    def __call__(self, request, timeout):
        if request.get_method() == 'POST':
            body = json.loads(request.data)
            path = body['path']
            self.capabilities.append(path)
            if path.endswith('b.bin') and len(self.ranges) >= 3 and self.fail_refresh:
                raise urllib.error.URLError(TimeoutError('credential refresh interrupted'))
            token = str(len(self.capabilities)).zfill(24)
            self.tokens[token] = path
            data = self.files[path]
            return io.BytesIO(json.dumps(dict(path=path, size=len(data),
                sha256=hashlib.sha256(data).hexdigest(), proxy_token=token,
                direct_url=None, direct_headers={})).encode())
        path = self.tokens[request.full_url.rsplit('/', 1)[-1]]
        data = self.files[path]
        selected = request.get_header('Range')
        if selected:
            self.ranges.append(selected)
            start, end = map(int, selected[6:].split('-'))
            stream = io.BytesIO(data[start:end+1])
            stream.status, stream.headers = 206, {'Content-Range':f'bytes {start}-{end}/{len(data)}'}
            return stream
        return io.BytesIO(data)


@pytest.mark.parametrize('corrupt', [False, True])
def test_install_retry_reuses_verified_files_and_validated_partial_prefix(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    network = Network()
    cache = tmp_path/'cache'
    def client():
        result = DinoV3DistributionClient('http://127.0.0.1/v1', opener=network, cache_dir=cache)
        result.fetch_manifest = network.manifest
        return result
    with pytest.raises(DinoV3DistributionError) as caught:
        client().download_tree('', tmp_path/'first-staging')
    assert caught.value.details['phase'] == 'capability'
    # One initial capability + five refresh attempts, not 5 x 5 nested attempts.
    assert network.capabilities.count('DINOv3/b.bin') == 6
    partial = cache/('a'*64)/'b.bin.download'
    assert partial.read_bytes() == b'abcd'*3
    if corrupt:
        partial.write_bytes(b'X'+partial.read_bytes()[1:])
    network.fail_refresh = False
    network.capabilities.clear()
    network.ranges.clear()
    target = tmp_path/'second-staging'
    client().download_tree('', target)
    assert network.capabilities.count('DINOv3/a.json') == 0
    assert network.ranges[0] == ('bytes=0-3' if corrupt else 'bytes=12-15')
    assert (target/'b.bin').read_bytes() == b'abcd'*4
    assert not partial.exists()


def test_maintenance_failure_keeps_progress_and_writes_redacted_traceback(tmp_path, monkeypatch):
    from system.backend import dinov3_maintenance as worker
    from system.backend import maintenance
    monkeypatch.setattr(maintenance, 'maintenance_status_path', lambda: tmp_path/'status.json')
    monkeypatch.setattr(maintenance, 'maintenance_log_path', lambda: tmp_path/'maintenance.log')
    monkeypatch.setattr(maintenance, '_wait_for_process_exit', lambda _: None)
    monkeypatch.setattr(maintenance, '_wait_for_port_release', lambda: None)
    monkeypatch.setattr(maintenance, '_start_backend', lambda: None)
    monkeypatch.setattr(worker.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(worker, '_install_dino_python_dependencies', lambda *a, **k: None)
    def install(**kwargs):
        kwargs['on_event']({'phase':'capability', 'file':'assets/encoder.pth', 'attempt':5})
        kwargs['on_progress'](30, 'download in progress')
        try:
            raise urllib.error.URLError('https://example.test/proxy/SECRET_TOKEN?sig=SECRET_SIGN')
        except Exception as exc:
            raise DinoV3DistributionError('credential failed', details={
                'phase':'capability', 'file':'assets/encoder.pth', 'http_status':503}) from exc
    monkeypatch.setattr(worker, 'install_dinov3_component', install)
    assert worker._run(['--operation','install_dinov3','--backend-pid','0']) == 1
    status = json.loads((tmp_path/'status.json').read_text('utf-8'))
    assert status['state'] == 'failed' and status['progress'] == 66
    assert status['error_details']['http_status'] == 503
    assert status['error_details']['file'] == 'assets/encoder.pth'
    combined = (tmp_path/'status.json').read_text('utf-8') + (tmp_path/'maintenance.log').read_text('utf-8')
    assert 'SECRET_TOKEN' not in combined and 'SECRET_SIGN' not in combined
    assert 'URLError' in combined and 'traceback' in combined


def test_native_metadata_preserves_http_status_and_retry_headers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    client = DinoV3DistributionClient()
    monkeypatch.setattr(distribution.shutil, 'which', lambda _: 'curl.exe')
    def run(command, **options):
        assert command[1] == '--disable' and '--insecure' not in command
        Path(command[command.index('--output')+1]).write_bytes(b'{"detail":"request_rate_limited"}')
        Path(command[command.index('--dump-header')+1]).write_text(
            'HTTP/1.1 200 Connection established\n\nHTTP/1.1 429 Too Many Requests\nRetry-After: 7\n\n')
        return SimpleNamespace(returncode=0, stdout=b'429', stderr=b'')
    monkeypatch.setattr(distribution.subprocess, 'run', run)
    request = distribution.urllib.request.Request('https://example.test/dinov3/direct',
        data=b'{}', method='POST')
    with pytest.raises(urllib.error.HTTPError) as caught:
        client._native_json_response(request)
    assert caught.value.code == 429 and caught.value.headers['Retry-After'] == '7'


def test_diagnostics_remove_bare_proxy_tokens_and_nested_secret_fields():
    from system.dinov3.download_diagnostics import exception_details, event_line
    token = 'A' * 43
    detail = exception_details(OSError('Authorization: Bearer PRIVATE_BEARER; connection failed at /proxy/'+token))
    line = event_line(dict(details=detail, proxy_token='SECRET',
                           nested={'direct_url':'https://host.test/x?sig=SIGNED'}))
    assert token not in line and 'SECRET' not in line and 'SIGNED' not in line and 'PRIVATE_BEARER' not in line
    assert json.loads(line)['proxy_token'] == '[hidden]'




def test_failed_component_install_retains_verified_cache_and_old_component(cloud, tmp_path):
    from system.dinov3.component import install_dinov3_component
    source, _, client, _ = cloud
    root = tmp_path/'component'
    install_dinov3_component(root=root, seed_dir=source.parent)
    before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    client.cache_dir = tmp_path/'cache'
    original = client.opener
    def failing(request, timeout):
        if request.get_method() == 'POST' and json.loads(request.data)['path'].endswith('LICENSE.md'):
            raise urllib.error.URLError(TimeoutError('license credential temporarily unavailable'))
        return original(request, timeout)
    client.opener = failing
    with pytest.raises(DinoV3DistributionError):
        install_dinov3_component(root=root, distribution_client=client)
    assert before == {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert list(client.cache_dir.rglob('calibration.npz'))
    assert not list(root.parent.glob('.dinov3-stage-*'))
    credentials = []
    def restored(request, timeout):
        if request.get_method() == 'POST':
            credentials.append(json.loads(request.data)['path'])
        return original(request, timeout)
    client.opener = restored
    result = install_dinov3_component(root=root, distribution_client=client)
    assert result['healthy'] and result['class_count'] == 2
    assert not any(path.endswith('calibration.npz') for path in credentials)