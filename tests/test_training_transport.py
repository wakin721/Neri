import json
import unittest

from system.training.transport import HttpTransport, UploadError, UploadCancelled


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.transport = HttpTransport()
        self.job = {
            'sample_id': 'a' * 32,
            'revision': 1,
            'secret': 's' * 43,
            'payload': {'species': ['赤狐']},
        }
        self.targets = [
            {
                'kind': kind,
                'species': '赤狐',
                'chunk_size': 5242880,
                'upload_url': 'https://my.microsoftpersonalcontent.com/upload/' + token * 43,
                'cancel_url': self.transport.base_url + '/v1/uploads/' + token * 43,
            }
            for kind, token in [('image', 'i'), ('annotation', 'a'), ('classes', 'c')]
        ]
        self.response = {'sample_id': self.job['sample_id'], 'revision': 1, 'targets': self.targets}
        self.calls = []

        def request(url, method, body=None, headers=None):
            self.calls.append((url, method, body, headers))
            return self.response if url.endswith('/v1/submissions') else {'status': 'complete'}

        self.transport._request = request

    @staticmethod
    def package(label='0 0.5 0.5 1.0 1.0\n', class_name='Vulpes_vulpes\n'):
        return json.dumps({
            'yolo_labels': {'赤狐': label},
            'yolo_classes': {'赤狐': class_name},
        }, ensure_ascii=False).encode()

    def test_committed_retry_receipt_finishes_without_resending(self):
        self.response.update(already_complete=True, targets=[])
        self.transport.upload(self.job, b'photo', self.package(), lambda: True)
        self.assertEqual(len(self.calls), 1)

    def test_receipt_must_match_sample_and_revision(self):
        self.response.update(already_complete=True, targets=[], revision=2)
        with self.assertRaises(UploadError):
            self.transport.upload(self.job, b'photo', self.package(), lambda: True)

    def test_only_microsoft_receives_photo_and_yolo_bytes_in_bounded_chunks(self):
        photo = b'x' * (5242880 + 7)
        label = '0 0.5 0.5 1.0 1.0\n'
        classes = 'Vulpes_vulpes\n'
        self.transport.upload(self.job, photo, self.package(label, classes), lambda: True)
        puts = [call for call in self.calls if call[1] == 'PUT']
        self.assertEqual(
            [len(call[2]) for call in puts],
            [5242880, 7, len(label.encode()), len(classes.encode())],
        )
        self.assertEqual(puts[1][3]['Content-Range'], 'bytes 5242880-5242886/5242887')
        self.assertTrue(all('Authorization' not in call[3] for call in puts))
        self.assertTrue(all(call[0].startswith('https://my.microsoftpersonalcontent.com/') for call in puts))
        self.targets[0]['upload_url'] = 'https://myneri.top/api/training/upload-photos'
        with self.assertRaises(UploadError):
            self.transport.upload(self.job, photo, self.package(), lambda: True)

    def test_cancellation_revokes_all_three_upload_targets(self):
        checks = iter([True, False])
        with self.assertRaises(UploadCancelled):
            self.transport.upload(self.job, b'photo', self.package(), lambda: next(checks))
        self.assertEqual(sum(call[1] == 'DELETE' for call in self.calls), 3)
        self.assertFalse(any(call[1] == 'PUT' for call in self.calls))


if __name__ == '__main__':
    unittest.main()
