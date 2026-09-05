import unittest

from system.training.transport import HttpTransport, UploadError, UploadCancelled


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.transport = HttpTransport()
        self.job = {'sample_id': 'a' * 32, 'revision': 1, 'secret': 's' * 43, 'payload': {'species': ['赤狐']}}
        self.targets = [{'kind': kind, 'species': '赤狐', 'chunk_size': 5242880,
                         'upload_url': 'https://my.microsoftpersonalcontent.com/upload/' + token * 43,
                         'cancel_url': self.transport.base_url + '/v1/uploads/' + token * 43}
                        for kind, token in [('image', 'i'), ('annotation', 'a')]]
        self.response = {'sample_id': self.job['sample_id'], 'revision': 1, 'targets': self.targets}
        self.calls = []

        def request(url, method, body=None, headers=None):
            self.calls.append((url, method, body, headers))
            return self.response if url.endswith('/v1/submissions') else {'status': 'complete'}

        self.transport._request = request

    def test_committed_retry_receipt_finishes_without_resending(self):
        self.response.update(already_complete=True, targets=[])
        self.transport.upload(self.job, b'photo', b'annotation', lambda: True)
        self.assertEqual(len(self.calls), 1)

    def test_receipt_must_match_sample_and_revision(self):
        self.response.update(already_complete=True, targets=[], revision=2)
        with self.assertRaises(UploadError):
            self.transport.upload(self.job, b'photo', b'annotation', lambda: True)

    def test_only_microsoft_receives_photo_bytes_and_bounded_chunks(self):
        photo = b'x' * (5242880 + 7)
        self.transport.upload(self.job, photo, b'{}', lambda: True)
        puts = [call for call in self.calls if call[1] == 'PUT']
        self.assertEqual([len(call[2]) for call in puts], [5242880, 7, 2])
        self.assertEqual(puts[1][3]['Content-Range'], 'bytes 5242880-5242886/5242887')
        self.assertTrue(all('Authorization' not in call[3] for call in puts))
        self.assertTrue(all(call[0].startswith('https://my.microsoftpersonalcontent.com/') for call in puts))
        self.targets[0]['upload_url'] = 'https://myneri.top/api/training/upload-photos'
        with self.assertRaises(UploadError):
            self.transport.upload(self.job, photo, b'{}', lambda: True)

    def test_cancellation_revokes_relay_attempt(self):
        checks = iter([True, False])
        with self.assertRaises(UploadCancelled):
            self.transport.upload(self.job, b'photo', b'{}', lambda: next(checks))
        self.assertEqual(sum(call[1] == 'DELETE' for call in self.calls), 2)
        self.assertFalse(any(call[1] == 'PUT' for call in self.calls))


if __name__ == '__main__':
    unittest.main()
