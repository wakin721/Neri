import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_proxy_capability_is_bound_and_expires(self):
        from server.model_distribution.capabilities import CapabilityStore, CapabilityError
        from server.model_distribution.manifest import ManifestEntry
        with tempfile.TemporaryDirectory() as temp_dir:
            now = [1000.0]
            store = CapabilityStore(Path(temp_dir), ttl_seconds=10, clock=lambda: now[0])
            token, expires = store.issue(ManifestEntry('detect/a.pt', 3, 'a' * 64))
            self.assertEqual(expires, 1010.0)
            bound = store.consume(token)
            self.assertEqual(bound.path, 'detect/a.pt')
            now[0] = 1011.0
            with self.assertRaises(CapabilityError):
                store.consume(token)


class DistributionServiceTests(unittest.TestCase):
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
