import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from server.model_distribution.manifest import ManifestBuilder, ManifestError
from server.model_distribution.storage import RemoteEntry, UpstreamLink


class FakeStore:
    def __init__(self, files):
        self.files = dict(files)
        self.read_count = 0

    def list_dir(self, path):
        prefix = path.rstrip('/') + '/'
        rows = []
        for full, meta in self.files.items():
            if not full.startswith(prefix):
                continue
            rest = full[len(prefix):]
            if '/' in rest:
                continue
            rows.append(RemoteEntry(rest, len(meta['data']), meta.get('modified'), False))
        return rows

    def stat(self, path):
        meta = self.files.get(path)
        if meta is None:
            return None
        return RemoteEntry(Path(path).name, len(meta['data']), meta.get('modified'), False)

    def resolve_link(self, path):
        return UpstreamLink('https://example.invalid/' + Path(path).name, {})

    def iter_bytes(self, link, range_header=None):
        self.read_count += 1
        name = Path(link.url).name
        for full, meta in self.files.items():
            if Path(full).name == name:
                data = meta['data']
                for index in range(0, len(data), 3):
                    yield data[index:index + 3]
                return
        raise KeyError(name)


class ManifestBuilderTests(unittest.TestCase):
    def test_short_or_oversized_stream_is_not_cached(self):
        from unittest.mock import patch
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'abc', 'modified': '1'}})
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            for body in [b'a', b'abcde']:
                with self.subTest(body=body), patch.object(store, 'iter_bytes', return_value=iter([body])):
                    with self.assertRaisesRegex(ManifestError, 'drive_stream_size_mismatch'):
                        builder.build()
            snapshot = builder.build()
            self.assertEqual(snapshot.files[0].sha256, hashlib.sha256(b'abc').hexdigest())

    def test_builds_allowlisted_sorted_manifest(self):
        files = {
            '/Neri_Data/Model/detect/z.pt': {'data': b'z', 'modified': '1'},
            '/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': '1'},
            '/Neri_Data/Model/cls/b.onnx': {'data': b'bb', 'modified': '2'},
            '/Neri_Data/Model/tracker.yaml': {'data': b'track', 'modified': '3'},
            '/Neri_Data/Model/detect/no.onnx': {'data': b'bad', 'modified': '4'},
            '/Neri_Data/Model/cls/readme.txt': {'data': b'bad', 'modified': '5'},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = ManifestBuilder(Path(temp_dir), FakeStore(files)).build()
        self.assertEqual([item.path for item in snapshot.files], ['cls/b.onnx', 'detect/a.pt', 'detect/z.pt', 'tracker.yaml'])
        self.assertEqual(snapshot.files[1].sha256, hashlib.sha256(b'aaa').hexdigest())
        self.assertEqual(len(snapshot.manifest_id), 64)

    def test_reuses_hash_only_when_size_and_modified_match(self):
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': 'same'}})
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            builder.build(); first = store.read_count
            builder.build(); second = store.read_count
            self.assertEqual(first, second)
            store.files['/Neri_Data/Model/detect/a.pt'] = {'data': b'bbb', 'modified': 'changed'}
            builder.build()
            self.assertGreater(store.read_count, second)

    def test_missing_modified_never_trusts_cache(self):
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': None}})
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            builder.build(); first = store.read_count
            builder.build()
            self.assertGreater(store.read_count, first)

    def test_rejects_casefold_duplicate_logical_paths(self):
        files = {
            '/Neri_Data/Model/detect/A.pt': {'data': b'a', 'modified': '1'},
            '/Neri_Data/Model/detect/a.pt': {'data': b'b', 'modified': '2'},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ManifestError):
                ManifestBuilder(Path(temp_dir), FakeStore(files)).build()

    def test_manifest_id_changes_when_content_changes(self):
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'a', 'modified': '1'}})
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            first = builder.build().manifest_id
            store.files['/Neri_Data/Model/detect/a.pt'] = {'data': b'b', 'modified': '2'}
            second = builder.build().manifest_id
        self.assertNotEqual(first, second)


class OpenListPathTests(unittest.TestCase):
    def test_list_rejects_malformed_or_incomplete_listing(self):
        from unittest.mock import patch
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.storage import OpenListModelStore, StorageError
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OpenListModelStore(DistributionConfig(Path(temp_dir), openlist_token='secret'))
            for data in [None, {}, {'content': None}, {'content': [], 'total': 2},
                         {'content': [{'name': 'a.pt'}], 'total': 1}]:
                with self.subTest(data=data), patch.object(store, '_call', return_value=data):
                    with self.assertRaises(StorageError):
                        store.list_dir('/Neri_Data/Model/detect')
            for content in [None, []]:
                with patch.object(store, '_call', return_value={'content': content, 'total': 0}):
                    self.assertEqual(store.list_dir('/Neri_Data/Model/cls'), [])

    def test_range_proxy_rejects_upstream_ignoring_range(self):
        from unittest.mock import patch
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.storage import OpenListModelStore, StorageError
        from tests.test_model_sync_client import _FakeResponse

        response = _FakeResponse(b'0123456789')
        response.status = 200
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OpenListModelStore(DistributionConfig(Path(temp_dir), openlist_token='secret'))
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(StorageError, 'drive_range_not_honored'):
                    list(store.iter_bytes(UpstreamLink('https://foo.1drv.com/model', {}), 'bytes=2-3'))

    def test_range_proxy_checks_content_range_and_limits_stream(self):
        from unittest.mock import patch
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.storage import OpenListModelStore, StorageError
        from tests.test_model_sync_client import _FakeResponse

        with tempfile.TemporaryDirectory() as temp_dir:
            store = OpenListModelStore(DistributionConfig(Path(temp_dir), openlist_token='secret'))
            for content_range, body, valid in [('bytes 2-3/10', b'23', True),
                                             ('bytes 0-1/10', b'01', False),
                                             ('bytes 2-3/10', b'234', False),
                                             ('bytes 2-3/10', b'2', False)]:
                with self.subTest(content_range=content_range, body=body):
                    response = _FakeResponse(body)
                    response.status = 206
                    response.headers['Content-Range'] = content_range
                    with patch('urllib.request.urlopen', return_value=response):
                        chunks = store.iter_bytes(UpstreamLink('https://foo.1drv.com/model', {}), 'bytes=2-3')
                        if valid:
                            self.assertEqual(b''.join(chunks), b'23')
                        else:
                            with self.assertRaises(StorageError):
                                list(chunks)

    def test_store_rejects_paths_outside_fixed_root(self):
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.storage import OpenListModelStore, StorageError
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OpenListModelStore(DistributionConfig(Path(temp_dir), openlist_token='x'))
            with self.assertRaises(StorageError):
                store._safe_path('/Neri_Data/Training/a.pt')
            self.assertEqual(store._safe_path('/Neri_Data/Model/detect/a.pt'), '/Neri_Data/Model/detect/a.pt')


class CapabilityTests(unittest.TestCase):
    def test_validates_microsoft_https_download_hosts(self):
        from server.model_distribution.capabilities import valid_direct_download_url
        self.assertTrue(valid_direct_download_url('https://foo.1drv.com/a'))
        self.assertTrue(valid_direct_download_url('https://tenant.sharepoint.com/a'))
        self.assertTrue(valid_direct_download_url('https://x.microsoftpersonalcontent.com/a'))
        self.assertFalse(valid_direct_download_url('http://foo.1drv.com/a'))
        self.assertFalse(valid_direct_download_url('https://evil.example/a'))
        self.assertFalse(valid_direct_download_url('https://user:pass@foo.1drv.com/a'))
        self.assertFalse(valid_direct_download_url('https://foo.1drv.com:444/a'))

    def test_filters_sensitive_direct_headers(self):
        from server.model_distribution.capabilities import safe_direct_headers
        self.assertEqual(
            safe_direct_headers({'Authorization': 'secret', 'Cookie': 'x', 'User-Agent': 'ua', 'Accept': '*/*'}),
            {'User-Agent': 'ua', 'Accept': '*/*'},
        )

    def test_proxy_capability_is_bound_expires_and_has_bounded_uses(self):
        from server.model_distribution.capabilities import CapabilityStore, CapabilityError
        from server.model_distribution.manifest import ManifestEntry
        with tempfile.TemporaryDirectory() as temp_dir:
            now = [1000.0]
            store = CapabilityStore(Path(temp_dir), ttl_seconds=10, clock=lambda: now[0])
            token, expires = store.issue(ManifestEntry('detect/a.pt', 3, 'a' * 64))
            self.assertEqual(expires, 1010.0)
            for _ in range(4):
                bound = store.consume(token)
                self.assertEqual(bound.path, 'detect/a.pt')
            with self.assertRaises(CapabilityError):
                store.consume(token)

            token, _ = store.issue(ManifestEntry('detect/b.pt', 3, 'b' * 64))
            now[0] = 1011.0
            with self.assertRaises(CapabilityError):
                store.consume(token)

    def test_request_budget_is_per_ip_and_resets_each_minute(self):
        from server.model_distribution.capabilities import BudgetError, BudgetStore
        with tempfile.TemporaryDirectory() as temp_dir:
            now = [120.0]
            budget = BudgetStore(
                Path(temp_dir),
                secret=b'test-secret',
                requests_per_minute=2,
                daily_ip_bytes=100,
                daily_total_bytes=1000,
                clock=lambda: now[0],
            )
            budget.check_request('203.0.113.10')
            budget.check_request('203.0.113.10')
            budget.check_request('203.0.113.11')
            with self.assertRaises(BudgetError):
                budget.check_request('203.0.113.10')
            now[0] = 180.0
            budget.check_request('203.0.113.10')

    def test_proxy_budget_enforces_per_ip_and_total_daily_bytes(self):
        from server.model_distribution.capabilities import BudgetError, BudgetStore
        with tempfile.TemporaryDirectory() as temp_dir:
            budget = BudgetStore(
                Path(temp_dir),
                secret=b'test-secret',
                requests_per_minute=100,
                daily_ip_bytes=10,
                daily_total_bytes=15,
                clock=lambda: 86400.0,
            )
            budget.reserve_proxy_bytes('203.0.113.10', 6)
            budget.reserve_proxy_bytes('203.0.113.11', 5)
            with self.assertRaises(BudgetError):
                budget.reserve_proxy_bytes('203.0.113.10', 5)
            with self.assertRaises(BudgetError):
                budget.reserve_proxy_bytes('203.0.113.11', 5)


class DistributionServiceTests(unittest.TestCase):
    def test_anonymous_client_uses_server_auth_and_receives_onedrive_link(self):
        import json
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from server.model_distribution.app import create_app
        from server.model_distribution.config import DistributionConfig
        from tests.test_model_sync_client import _FakeResponse

        api_calls = []
        def openlist(request, timeout):
            if request.full_url.startswith('https://foo.1drv.com/'):
                self.assertIsNone(request.get_header('Authorization'))
                return _FakeResponse(b'abc')
            api_calls.append(request)
            self.assertEqual(request.get_header('Authorization'), 'server-only-secret')
            body = json.loads(request.data)
            endpoint = request.full_url.rsplit('/', 1)[-1]
            if endpoint == 'list':
                content = ([{'name': 'a.pt', 'size': 3, 'modified': '1', 'is_dir': False}]
                           if body['path'].endswith('/detect') else [])
                data = {'content': content, 'total': len(content)}
            elif endpoint == 'get':
                return _FakeResponse(json.dumps({'code': 404, 'message': 'not found'}).encode())
            else:
                self.assertEqual(endpoint, 'link')
                data = {'url': 'https://foo.1drv.com/a', 'header': None}
            return _FakeResponse(json.dumps({'code': 200, 'data': data}).encode())

        with tempfile.TemporaryDirectory() as temp_dir, patch('urllib.request.urlopen', side_effect=openlist):
            app = create_app(DistributionConfig(Path(temp_dir), openlist_token='server-only-secret'))
            with TestClient(app) as client:
                response = client.get('/v1/manifest')
                self.assertEqual(response.status_code, 200)
                manifest = response.json()
                entry = manifest['files'][0]
                direct = client.post('/v1/direct', json={
                    'manifest_id': manifest['manifest_id'], 'path': entry['path'], 'sha256': entry['sha256'],
                })
                self.assertEqual(direct.status_code, 200)
                capability = direct.json()
                self.assertEqual(capability['direct_url'], 'https://foo.1drv.com/a')
                self.assertEqual(capability['direct_headers'], {})
                self.assertNotIn('server-only-secret', response.text + direct.text)
                proxy = client.get('/v1/proxy/' + capability['proxy_token'])
                self.assertEqual(proxy.content, b'abc')
                self.assertEqual(proxy.headers['content-length'], '3')
                self.assertEqual(hashlib.sha256(proxy.content).hexdigest(), entry['sha256'])
                self.assertTrue(api_calls)

    def test_direct_requires_current_manifest_identity(self):
        from server.model_distribution.service import DistributionService, DistributionError
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': '1'}})
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DistributionService(Path(temp_dir), store, ttl_seconds=60)
            manifest = service.manifest(); entry = manifest.files[0]
            with self.assertRaises(DistributionError):
                service.direct('0' * 64, entry.path, entry.sha256)
            with self.assertRaises(DistributionError):
                service.direct(manifest.manifest_id, entry.path, '0' * 64)

    def test_invalid_direct_link_still_returns_proxy_capability(self):
        from server.model_distribution.service import DistributionService
        store = FakeStore({'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': '1'}})
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DistributionService(Path(temp_dir), store, ttl_seconds=60)
            manifest = service.manifest(); entry = manifest.files[0]
            result = service.direct(manifest.manifest_id, entry.path, entry.sha256)
            self.assertIsNone(result['direct_url'])
            self.assertTrue(result['proxy_token'])
            self.assertEqual(service.consume_proxy(result['proxy_token']).path, entry.path)


class RecordingBudget:
    def __init__(self):
        self.requests = []
        self.proxy_reservations = []

    def check_request(self, client_ip):
        self.requests.append(client_ip)

    def reserve_proxy_bytes(self, client_ip, byte_count):
        self.proxy_reservations.append((client_ip, byte_count))


class DistributionRouteTests(unittest.TestCase):
    def _request(self, client='203.0.113.10', *, range_header=None, real_ip=None):
        from starlette.requests import Request
        headers = []
        if range_header is not None:
            headers.append((b'range', range_header.encode('ascii')))
        if real_ip is not None:
            headers.append((b'x-real-ip', real_ip.encode('ascii')))
        return Request({
            'type': 'http',
            'method': 'GET',
            'path': '/',
            'raw_path': b'/',
            'query_string': b'',
            'headers': headers,
            'client': (client, 12345),
            'server': ('testserver', 80),
            'scheme': 'http',
        })

    def test_proxy_reservation_matches_requested_range(self):
        from server.model_distribution.app import proxy_reserved_bytes
        self.assertEqual(proxy_reserved_bytes(100, None), 100)
        self.assertEqual(proxy_reserved_bytes(100, 'bytes=10-19'), 10)
        self.assertEqual(proxy_reserved_bytes(100, 'bytes=90-'), 10)
        with self.assertRaises(ValueError):
            proxy_reserved_bytes(100, 'bytes=100-')
        with self.assertRaises(ValueError):
            proxy_reserved_bytes(100, 'bytes=20-10')

    def test_public_routes_use_real_client_budget_and_proxy_byte_budget(self):
        from server.model_distribution.app import DirectRequest, create_app
        from server.model_distribution.capabilities import BoundCapability
        from server.model_distribution.config import DistributionConfig
        from server.model_distribution.manifest import ManifestEntry, ManifestSnapshot

        class RouteService:
            def __init__(self):
                self.store = FakeStore({
                    '/Neri_Data/Model/detect/a.pt': {'data': b'x' * 100, 'modified': '1'},
                })

            def manifest(self):
                entry = ManifestEntry('detect/a.pt', 100, 'a' * 64)
                return ManifestSnapshot('b' * 64, (entry,))

            def direct(self, manifest_id, path, sha256):
                return {'manifest_id': manifest_id, 'path': path, 'sha256': sha256}

            def consume_proxy(self, token):
                return BoundCapability('detect/a.pt', 'a' * 64, 100, 999999.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            budget = RecordingBudget()
            config = DistributionConfig(
                Path(temp_dir),
                openlist_token='secret',
                requests_per_minute=10,
                daily_proxy_ip_bytes=1000,
                daily_proxy_total_bytes=5000,
            )
            app = create_app(config=config, service=RouteService(), budget=budget)
            routes = {getattr(route, 'path', None): route for route in app.routes}
            request = self._request(client='127.0.0.1', real_ip='203.0.113.10')

            manifest_result = routes['/v1/manifest'].endpoint(request)
            self.assertEqual(manifest_result['manifest_id'], 'b' * 64)

            direct_result = routes['/v1/direct'].endpoint(
                DirectRequest(manifest_id='b' * 64, path='detect/a.pt', sha256='a' * 64),
                request,
            )
            self.assertEqual(direct_result['path'], 'detect/a.pt')

            proxy_request = self._request(
                client='127.0.0.1',
                real_ip='203.0.113.10',
                range_header='bytes=10-19',
            )
            response = routes['/v1/proxy/{token}'].endpoint('token-value-for-test-123456', proxy_request)
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.headers['content-range'], 'bytes 10-19/100')
            self.assertEqual(response.headers['content-length'], '10')
            self.assertEqual(budget.requests, ['203.0.113.10'] * 3)
            self.assertEqual(budget.proxy_reservations, [('203.0.113.10', 10)])

    def test_proxy_stream_rejects_bytes_beyond_reserved_budget(self):
        from server.model_distribution.app import bounded_proxy_stream
        from server.model_distribution.storage import StorageError
        self.assertEqual(list(bounded_proxy_stream(iter([b'12', b'34']), 4)), [b'12', b'34'])
        for chunks in [[b'12345'], [b'1']]:
            with self.subTest(chunks=chunks), self.assertRaises(StorageError):
                list(bounded_proxy_stream(iter(chunks), 4))


class ProxyRangeTests(unittest.TestCase):
    def test_app_helpers_import_without_openlist_token(self):
        env = dict(os.environ)
        env.pop('NERI_OPENLIST_TOKEN', None)
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'from server.model_distribution.app import valid_range_header; '
                'assert valid_range_header("bytes=0-1")',
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_only_single_byte_range(self):
        os.environ.setdefault('NERI_OPENLIST_TOKEN', 'test')
        from server.model_distribution.app import valid_range_header
        self.assertTrue(valid_range_header(None))
        self.assertTrue(valid_range_header('bytes=0-99'))
        self.assertTrue(valid_range_header('bytes=100-'))
        self.assertFalse(valid_range_header('bytes=0-1,3-4'))
        self.assertFalse(valid_range_header('items=0-1'))
        self.assertFalse(valid_range_header('bytes=-100'))


if __name__ == '__main__':
    unittest.main()
