import hashlib
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
        self.assertEqual(
            [item.path for item in snapshot.files],
            ['cls/b.onnx', 'detect/a.pt', 'detect/z.pt', 'tracker.yaml'],
        )
        self.assertEqual(snapshot.files[1].sha256, hashlib.sha256(b'aaa').hexdigest())
        self.assertEqual(len(snapshot.manifest_id), 64)

    def test_reuses_hash_only_when_size_and_modified_match(self):
        files = {'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': 'same'}}
        store = FakeStore(files)
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            builder.build()
            first = store.read_count
            builder.build()
            second = store.read_count
            self.assertEqual(first, second)
            store.files['/Neri_Data/Model/detect/a.pt'] = {'data': b'bbb', 'modified': 'changed'}
            builder.build()
            self.assertGreater(store.read_count, second)

    def test_missing_modified_never_trusts_cache(self):
        files = {'/Neri_Data/Model/detect/a.pt': {'data': b'aaa', 'modified': None}}
        store = FakeStore(files)
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(Path(temp_dir), store)
            builder.build()
            first = store.read_count
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
        files = {'/Neri_Data/Model/detect/a.pt': {'data': b'a', 'modified': '1'}}
        store = FakeStore(files)
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
            self.assertEqual(
                store._safe_path('/Neri_Data/Model/detect/a.pt'),
                '/Neri_Data/Model/detect/a.pt',
            )


if __name__ == '__main__':
    unittest.main()
