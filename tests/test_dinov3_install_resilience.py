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
    # This tests download diagnostics, independently of host GPU hardware or pip.
    monkeypatch.setattr(worker, 'resolve_runtime_choice', lambda _: 'cpu')
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


def _seed_partial(target: Path, data: bytes, offset: int, segment_size: int, *, sha256: str | None = None):
    from system.dinov3.download_cache import checkpoint_partial, partial_paths

    target.parent.mkdir(parents=True, exist_ok=True)
    part, metadata = partial_paths(target)
    part.write_bytes(data[:offset])
    prefix_digest = hashlib.sha256(data[:offset])
    checkpoint_partial(
        metadata,
        size=len(data),
        sha256=sha256 or hashlib.sha256(data).hexdigest(),
        segment_size=segment_size,
        offset=offset,
        digest=prefix_digest,
    )
    return part, metadata


@pytest.mark.parametrize('runtime', ['gpu', 'cpu'])
def test_existing_valid_partial_resumes_via_onedrive_direct_for_cpu_and_gpu(tmp_path, monkeypatch, runtime):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'0123456789abcdef'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = 'b' * 64
    entry_path = 'DINOv3/cpu/model.bin' if runtime == 'cpu' else 'DINOv3/model.bin'
    cache = tmp_path / 'cache'
    cached = cache / manifest_id / 'model.bin'
    part, metadata = _seed_partial(cached, payload, offset=8, segment_size=4, sha256=expected_sha)
    part.write_bytes(payload[:8] + b'UNCOMMITTED')

    direct_url = 'https://tenant.sharepoint.com/files/model.bin?sig=SECRET_DIRECT_SIG'
    proxy_token = 'P' * 43
    direct_requests = []
    proxy_requests = []
    events = []

    def opener(request, timeout):
        if request.get_method() == 'POST':
            return io.BytesIO(json.dumps({
                'path': entry_path,
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': proxy_token,
                'direct_url': direct_url,
                'direct_headers': {'Referer': 'https://onedrive.live.com/', 'Authorization': 'DROP_ME'},
            }).encode())
        if '/proxy/' in request.full_url:
            proxy_requests.append(request.full_url)
            raise AssertionError('proxy should not be called when direct resume succeeds')
        assert request.full_url == direct_url
        assert request.get_header('Referer') == 'https://onedrive.live.com/'
        assert request.get_header('Authorization') is None
        rng = request.get_header('Range')
        direct_requests.append(rng)
        assert rng == 'bytes=8-15'
        stream = io.BytesIO(payload[8:])
        stream.status = 206
        stream.headers = {'Content-Range': 'bytes 8-15/16', 'Content-Length': '8'}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache, on_event=events.append, runtime=runtime)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry(entry_path, 'model.bin', len(payload), expected_sha),)
    )
    target = tmp_path / f'out-{runtime}'
    client.download_tree('', target)
    assert (target / 'model.bin').read_bytes() == payload
    assert direct_requests == ['bytes=8-15']
    assert proxy_requests == []
    assert not part.exists() and not metadata.exists()
    serialized = json.dumps(events)
    assert 'SECRET_DIRECT_SIG' not in serialized and proxy_token not in serialized


def test_corrupt_partial_is_discarded_and_restarts_via_onedrive_direct(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'abcdefghijklmno!'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = 'c' * 64
    cache = tmp_path / 'cache'
    cached = cache / manifest_id / 'weights.bin'
    part, _ = _seed_partial(cached, payload, offset=8, segment_size=4, sha256=expected_sha)
    part.write_bytes(b'CORRUPTED')

    direct_url = 'https://contoso.1drv.com/weights.bin?sig=SECRET_SIG'
    seen_ranges = []

    def opener(request, timeout):
        if request.get_method() == 'POST':
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/weights.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'Q' * 43,
                'direct_url': direct_url,
                'direct_headers': {},
            }).encode())
        assert request.full_url == direct_url
        seen_ranges.append(request.get_header('Range'))
        stream = io.BytesIO(payload)
        stream.status = 200
        stream.headers = {'Content-Length': str(len(payload))}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/weights.bin', 'weights.bin', len(payload), expected_sha),)
    )
    target = tmp_path / 'out'
    client.download_tree('', target)
    assert seen_ranges == [None]
    assert (target / 'weights.bin').read_bytes() == payload


def test_fresh_large_file_direct_checkpoints_1mib_and_resumes_after_drop_or_proxy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'0123456789abcdef'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = 'd' * 64
    cache = tmp_path / 'cache'

    class _DropStream:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.status = 200
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n=-1):
            if self._chunks:
                item = self._chunks.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
            return b''

    direct_calls = []
    proxy_ranges = []
    events = []

    def opener(request, timeout):
        if request.get_method() == 'POST':
            idx = len(direct_calls) + 1
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/large.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'R' * 43,
                'direct_url': f'https://cdn.microsoftpersonalcontent.com/large.bin?v={idx}&sig=SECRET_{idx}',
                'direct_headers': {},
            }).encode())
        if '/proxy/' in request.full_url:
            rng = request.get_header('Range')
            proxy_ranges.append(rng)
            start, end = map(int, rng[6:].split('-'))
            stream = io.BytesIO(payload[start:end + 1])
            stream.status = 206
            stream.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            return stream
        rng = request.get_header('Range')
        direct_calls.append(rng)
        if len(direct_calls) == 1:
            # Send 9 bytes (checkpoints 8 bytes = 2 x 4-byte segments) then drop connection.
            return _DropStream([payload[0:4], payload[4:8], payload[8:9], OSError('connection reset')])
        # On direct retry, server ignores Range and returns 200 -> reject without losing 8-byte prefix!
        stream = io.BytesIO(payload)
        stream.status = 200
        stream.headers = {'Content-Length': str(len(payload))}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache, on_event=events.append)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/large.bin', 'large.bin', len(payload), expected_sha),)
    )
    target = tmp_path / 'out'
    client.download_tree('', target)
    assert (target / 'large.bin').read_bytes() == payload
    assert direct_calls == [None, 'bytes=8-15']
    assert proxy_ranges == ['bytes=8-11', 'bytes=12-15']
    serialized = json.dumps(events)
    assert 'SECRET_1' not in serialized and 'SECRET_2' not in serialized and ('R' * 43) not in serialized


def test_direct_url_403_renews_bounded_and_falls_back_preserving_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'AAAABBBBCCCCDDDD'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = 'e' * 64
    cache = tmp_path / 'cache'
    cached = cache / manifest_id / 'model.bin'
    _seed_partial(cached, payload, offset=8, segment_size=4, sha256=expected_sha)

    # Case 1: 403 on first direct URL, renewed direct URL succeeds with 206.
    cap_calls = []
    direct_urls = []

    def opener_renew_ok(request, timeout):
        if request.get_method() == 'POST':
            cap_calls.append(len(cap_calls) + 1)
            idx = len(cap_calls)
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'S' * 43,
                'direct_url': f'https://tenant.sharepoint.com/model.bin?tok={idx}',
                'direct_headers': {},
            }).encode())
        direct_urls.append(request.full_url)
        if 'tok=1' in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, io.BytesIO(b'expired'))
        assert request.get_header('Range') == 'bytes=8-15'
        stream = io.BytesIO(payload[8:])
        stream.status = 206
        stream.headers = {'Content-Range': 'bytes 8-15/16'}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener_renew_ok, cache_dir=cache)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
    )
    out1 = tmp_path / 'out1'
    client.download_tree('', out1)
    assert (out1 / 'model.bin').read_bytes() == payload
    assert len(cap_calls) == 2
    assert len(direct_urls) == 2

    # Case 2: Persistent 503 on direct retries stays bounded and falls back to proxy from offset 8.
    cache2 = tmp_path / 'cache2'
    _seed_partial(cache2 / manifest_id / 'model.bin', payload, offset=8, segment_size=4, sha256=expected_sha)
    cap_calls2 = []
    direct_attempts = []
    proxy_ranges = []

    def opener_always_503(request, timeout):
        if request.get_method() == 'POST':
            cap_calls2.append(1)
            idx = len(cap_calls2)
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'T' * 43,
                'direct_url': f'https://tenant.sharepoint.com/model.bin?tok={idx}',
                'direct_headers': {},
            }).encode())
        if '/proxy/' in request.full_url:
            rng = request.get_header('Range')
            proxy_ranges.append(rng)
            start, end = map(int, rng[6:].split('-'))
            stream = io.BytesIO(payload[start:end + 1])
            stream.status = 206
            stream.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            return stream
        direct_attempts.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, 'Unavailable', {}, io.BytesIO(b'busy'))

    client2 = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener_always_503, cache_dir=cache2)
    client2.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
    )
    out2 = tmp_path / 'out2'
    client2.download_tree('', out2)
    assert (out2 / 'model.bin').read_bytes() == payload
    assert 1 <= len(direct_attempts) <= distribution.DIRECT_DOWNLOAD_ATTEMPTS
    assert len(cap_calls2) <= distribution.DIRECT_DOWNLOAD_ATTEMPTS
    assert proxy_ranges == ['bytes=8-11', 'bytes=12-15']


def test_local_http_direct_resume_with_native_curl_fallback_and_redirect_whitelist(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4096)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8192)
    payload = (b'NERI-ONEDRIVE-DIRECT-' * 600)[:12288]
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = 'f' * 64
    direct_ranges = []
    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or '0')
            self.rfile.read(length)
            host, port = self.server.server_address
            body = json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'U' * 43,
                'direct_url': f'http://{host}:{port}/redirect-ok?sig=SECRET_LOCAL_SIG',
                'direct_headers': {'Referer': 'https://onedrive.live.com/'},
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            host, port = self.server.server_address
            if self.path.startswith('/redirect-ok'):
                self.send_response(302)
                self.send_header('Location', f'http://{host}:{port}/direct-blob?sig=SECRET_HOP_SIG')
                self.end_headers()
                return
            if self.path.startswith('/redirect-evil'):
                self.send_response(302)
                self.send_header('Location', 'https://evil.example/payload.bin')
                self.end_headers()
                return
            if self.path.startswith('/direct-blob'):
                rng = self.headers.get('Range')
                direct_ranges.append(rng)
                assert rng == 'bytes=4096-12287'
                chunk = payload[4096:]
                self.send_response(206)
                self.send_header('Content-Range', f'bytes 4096-12287/{len(payload)}')
                self.send_header('Content-Length', str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        allowed_prefix = f'http://{host}:{port}/'
        monkeypatch.setattr(distribution, 'valid_direct_url', lambda u: u.startswith(allowed_prefix))
        cache = tmp_path / 'cache'
        _seed_partial(cache / manifest_id / 'model.bin', payload, offset=4096, segment_size=4096, sha256=expected_sha)

        client = DinoV3DistributionClient(f'http://{host}:{port}', cache_dir=cache, on_event=events.append)
        client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
            manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
        )
        # Force Python urllib direct open to fail TLS once so native curl.exe handles the 302 + 206 resume.
        def broken_direct(request, timeout):
            raise ssl.SSLError('simulated windows tls failure')
        monkeypatch.setattr(client, '_open_direct_url', broken_direct)
        client._native_fallback = True
        out = tmp_path / 'installed'
        client.download_tree('', out)
        assert (out / 'model.bin').read_bytes() == payload
        assert direct_ranges == ['bytes=4096-12287']
        serialized = json.dumps(events)
        assert 'SECRET_LOCAL_SIG' not in serialized and 'SECRET_HOP_SIG' not in serialized

        # Also verify native curl redirect whitelist rejects non-allowed Location targets.
        with pytest.raises(DinoV3DistributionError, match='unsafe_direct_redirect'):
            with client._open_native_stream(
                f'http://{host}:{port}/redirect-evil', headers={}, allow_direct_redirects=True
            ):
                pass
    finally:
        server.shutdown()
        server.server_close()


def test_local_http_urllib_direct_resume_and_safe_redirect_handler(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4096)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8192)
    payload = (b'URLLIB-DIRECT-RESUME-' * 500)[:8192]
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = '1' * 64
    seen_ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or '0')
            self.rfile.read(length)
            host, port = self.server.server_address
            body = json.dumps({
                'path': 'DINOv3/cpu/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'V' * 43,
                'direct_url': f'http://{host}:{port}/blob',
                'direct_headers': {},
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/blob':
                rng = self.headers.get('Range')
                seen_ranges.append(rng)
                chunk = payload[4096:]
                self.send_response(206)
                self.send_header('Content-Range', f'bytes 4096-8191/{len(payload)}')
                self.send_header('Content-Length', str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            if self.path == '/evil-hop':
                self.send_response(302)
                self.send_header('Location', 'https://evil.example/model.bin')
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        allowed_prefix = f'http://{host}:{port}/'
        monkeypatch.setattr(distribution, 'valid_direct_url', lambda u: u.startswith(allowed_prefix))
        cache = tmp_path / 'cache'
        _seed_partial(cache / manifest_id / 'model.bin', payload, offset=4096, segment_size=4096, sha256=expected_sha)
        client = DinoV3DistributionClient(f'http://{host}:{port}', cache_dir=cache, runtime='cpu')
        client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
            manifest_id, (DinoV3DistributionEntry('DINOv3/cpu/model.bin', 'model.bin', len(payload), expected_sha),)
        )
        out = tmp_path / 'out'
        client.download_tree('', out)
        assert (out / 'model.bin').read_bytes() == payload
        assert seen_ranges == ['bytes=4096-8191']

        req = distribution.urllib.request.Request(f'http://{host}:{port}/evil-hop', method='GET')
        with pytest.raises(DinoV3DistributionError, match='unsafe_direct_redirect'):
            client._open_direct_url(req, timeout=10)
    finally:
        server.shutdown()
        server.server_close()


def test_non_onedrive_direct_url_is_ignored_and_falls_back_preserving_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'0123456789abcdef'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = '2' * 64
    cache = tmp_path / 'cache'
    _seed_partial(cache / manifest_id / 'model.bin', payload, offset=8, segment_size=4, sha256=expected_sha)
    requested_urls = []

    def opener(request, timeout):
        requested_urls.append(request.full_url)
        if request.get_method() == 'POST':
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'W' * 43,
                'direct_url': 'https://untrusted.example.com/model.bin?sig=SECRET',
                'direct_headers': {},
            }).encode())
        rng = request.get_header('Range')
        start, end = map(int, rng[6:].split('-'))
        stream = io.BytesIO(payload[start:end + 1])
        stream.status = 206
        stream.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
    )
    out = tmp_path / 'out'
    client.download_tree('', out)
    assert (out / 'model.bin').read_bytes() == payload
    assert not any('untrusted.example.com' in url for url in requested_urls)
    assert len(requested_urls) == 3  # 1 capability + 2 proxy segments (8-11, 12-15)


def test_direct_stream_exit_failure_does_not_promote_target_or_redownload_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'0123456789abcdef'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = '3' * 64
    cache = tmp_path / 'cache'
    cached = cache / manifest_id / 'model.bin'
    direct_requests = []
    proxy_requests = []
    target_existed_during_exit = []

    class _FailingExitStream:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)
            self.status = 200
            self.headers = {'Content-Length': str(len(data))}

        def __enter__(self):
            return self

        def read(self, n=-1):
            return self._buf.read(n)

        def __exit__(self, exc_type, exc, tb):
            target_existed_during_exit.append(cached.exists())
            raise OSError('simulated curl exit code 18 after body')

    def opener(request, timeout):
        if request.get_method() == 'POST':
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'X' * 43,
                'direct_url': 'https://tenant.sharepoint.com/model.bin?sig=SECRET',
                'direct_headers': {},
            }).encode())
        if '/proxy/' in request.full_url:
            proxy_requests.append(request.get_header('Range'))
            raise AssertionError('proxy should not be called when verified partial already completed')
        direct_requests.append(request.get_header('Range'))
        return _FailingExitStream(payload)

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
    )
    out = tmp_path / 'out'
    client.download_tree('', out)
    assert target_existed_during_exit == [False]
    assert direct_requests == [None]
    assert proxy_requests == []
    assert (out / 'model.bin').read_bytes() == payload


def test_corrupt_completed_direct_partial_falls_back_to_proxy_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_BYTES', 4)
    monkeypatch.setattr(distribution, 'PROXY_SEGMENT_THRESHOLD', 8)
    payload = b'0123456789abcdef'
    bad_payload = b'XXXXXXXXXXXXXXXX'
    expected_sha = hashlib.sha256(payload).hexdigest()
    manifest_id = '4' * 64
    cache = tmp_path / 'cache'
    proxy_ranges = []

    def opener(request, timeout):
        if request.get_method() == 'POST':
            return io.BytesIO(json.dumps({
                'path': 'DINOv3/model.bin',
                'size': len(payload),
                'sha256': expected_sha,
                'proxy_token': 'Y' * 43,
                'direct_url': 'https://tenant.sharepoint.com/model.bin?sig=SECRET',
                'direct_headers': {},
            }).encode())
        if '/proxy/' in request.full_url:
            rng = request.get_header('Range')
            proxy_ranges.append(rng)
            start, end = map(int, rng[6:].split('-'))
            stream = io.BytesIO(payload[start:end + 1])
            stream.status = 206
            stream.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}
            return stream
        # Write full-size corrupt bytes (checkpoints offset==16), then extra byte triggers size_mismatch fallback.
        stream = io.BytesIO(bad_payload + b'EXTRA')
        stream.status = 200
        stream.headers = {}
        return stream

    client = DinoV3DistributionClient('http://127.0.0.1/v1', opener=opener, cache_dir=cache)
    client.fetch_manifest = lambda runtime=None: DinoV3DistributionManifest(
        manifest_id, (DinoV3DistributionEntry('DINOv3/model.bin', 'model.bin', len(payload), expected_sha),)
    )
    out = tmp_path / 'out'
    client.download_tree('', out)
    assert (out / 'model.bin').read_bytes() == payload
    assert proxy_ranges == ['bytes=0-3', 'bytes=4-7', 'bytes=8-11', 'bytes=12-15']


def test_open_native_stream_default_user_agent_and_wait_timeout_error(tmp_path, monkeypatch):
    import subprocess

    commands = []

    class _FakeStdout:
        def readline(self, _n=-1):
            return b''

        def close(self):
            pass

    class _FakeProc:
        def __init__(self):
            self.stdout = _FakeStdout()
            self._terminated = False

        def poll(self):
            return None if not self._terminated else 1

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(['curl.exe'], timeout or 10)

        def terminate(self):
            self._terminated = True

        def kill(self):
            self._terminated = True

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return _FakeProc()

    monkeypatch.setattr(distribution.shutil, 'which', lambda name: 'C:/Windows/System32/curl.exe')
    monkeypatch.setattr(distribution.subprocess, 'Popen', fake_popen)
    client = DinoV3DistributionClient('http://127.0.0.1/v1')
    with pytest.raises(OSError, match='timed out'):
        with client._open_native_stream('https://tenant.sharepoint.com/model.bin', scratch_dir=tmp_path):
            pass
    assert 'User-Agent: Neri DINOv3 installer' in commands[0]
    with pytest.raises(OSError, match='timed out'):
        with client._open_native_stream(
            'https://tenant.sharepoint.com/model.bin',
            headers={'user-agent': 'CustomUA/1.0'},
            scratch_dir=tmp_path,
        ):
            pass
    assert 'User-Agent: Neri DINOv3 installer' not in commands[1]
    assert 'user-agent: CustomUA/1.0' in commands[1]
