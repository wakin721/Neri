import json
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

from server.training_broker.app import Broker, BrokerConfig, BrokerError, Submission


class MemoryDrive:
    def __init__(self):
        self.files = {}
        self.sessions = {}
        self.cancelled = []
        self.folders = set()

    def mkdir(self, path):
        self.folders.add(path)

    def create_session(self, path, size):
        url = 'https://my.microsoftpersonalcontent.com/upload/' + str(len(self.sessions) + 1)
        self.sessions[url] = (path, size)
        return {'upload_url': url, 'chunk_size': 5242880}

    def stat(self, path):
        return {'size': self.files[path], 'is_dir': False} if path in self.files else None

    def delete(self, path):
        self.files.pop(path, None)

    def cancel(self, url):
        self.cancelled.append(url)

    def upload_all(self, response):
        self._upload_all(response)

    def upload(self, target, source, save_capability):
        save_capability('https://my.microsoftpersonalcontent.com/upload/' + target['token'])
        self.files[target['path']] = len(source.read_bytes())


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.drive = MemoryDrive()
        self.now = 1000.0
        self.config = BrokerConfig(state_dir=Path(self.temp.name), root_path='/Neri_Data/Training',
                                   requests_per_minute=100, daily_ip_bytes=1000000, daily_total_bytes=2000000)
        self.broker = Broker(self.config, self.drive, clock=lambda: self.now)
        self.secret = 's' * 43
        stream = io.BytesIO()
        Image.new('RGB', (12, 8), 'green').save(stream, format='JPEG')
        self.photo = stream.getvalue()
        self.annotation = self.annotation_for('a' * 32, 1, ['赤狐'])
        self.drive._upload_all = self.upload_all

    @staticmethod
    def annotation_for(sample_id, revision, species):
        return json.dumps({'schema_version': 1, 'sample_id': sample_id, 'revision': revision,
                           'species': species}, ensure_ascii=False).encode()

    def upload_all(self, response):
        for target in response['targets']:
            path, size = self.drive.sessions[target['upload_url']]
            self.drive.files[path] = size

    def tearDown(self):
        self.temp.cleanup()

    def submission(self, **kwargs):
        values = dict(sample_id='a' * 32, revision=1, secret=self.secret,
                      species=['赤狐'], image_bytes=len(self.photo))
        values.update(kwargs)
        values.setdefault('annotation_bytes', len(self.annotation_for(values['sample_id'], values['revision'], values['species'])))
        return Submission(**values)

    def test_pair_per_species_and_no_arbitrary_paths(self):
        response = self.broker.create(self.submission(species=['赤狐', '狼']), 'client1')
        self.assertEqual(len(response['targets']), 4)
        self.drive.upload_all(response)
        paths = list(self.drive.files)
        self.assertTrue(all(path.startswith('/Neri_Data/Training/') for path in paths))
        self.assertTrue(any('/赤狐/' in path for path in paths))
        self.assertTrue(any('/狼/' in path for path in paths))
        for bad in ['../private', '/root', '..', 'a\\b', '赤狐%2fsecret', 'name.', '空照片/../']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.submission(species=[bad])
        with self.assertRaises(ValueError):
            self.submission(species=['Fox', 'fox'])

    def test_cannot_complete_partial_or_wrong_sized_pair(self):
        response = self.broker.create(self.submission(), 'client1')
        first = response['targets'][0]
        path, size = self.drive.sessions[first['upload_url']]
        self.drive.files[path] = size
        with self.assertRaises(BrokerError) as error:
            self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.assertEqual(error.exception.status, 409)
        self.upload_all({**response, 'targets': response['targets'][1:]})
        self.assertEqual(self.broker.complete('a' * 32, 1, self.secret, 'client1')['status'], 'complete')

    def test_ownership_is_required_for_reuse_completion_and_delete(self):
        response = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(response)
        actions = [lambda: self.broker.create(self.submission(secret='x' * 43), 'client2'),
                   lambda: self.broker.complete('a' * 32, 1, 'x' * 43, 'client2'),
                   lambda: self.broker.delete('a' * 32, 'x' * 43, 'client2')]
        for action in actions:
            with self.assertRaises(BrokerError) as error:
                action()
            self.assertEqual(error.exception.status, 403)
        self.assertEqual(len(self.drive.files), 2)

    def test_new_revision_keeps_previous_until_complete_and_removes_old_category(self):
        response = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(response)
        self.broker.complete('a' * 32, 1, self.secret, 'client1')
        new = self.broker.create(self.submission(revision=2, species=['狼']), 'client1')
        self.assertTrue(any('/赤狐/' in p for p in self.drive.files))
        with self.assertRaises(BrokerError):
            self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.drive.upload_all(new)
        self.broker.complete('a' * 32, 2, self.secret, 'client1')
        self.assertEqual(len(self.drive.files), 2)
        self.assertTrue(all('/狼/' in p for p in self.drive.files))

    def test_retry_completed_revision_is_idempotent_and_survives_restart(self):
        response = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(response)
        self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.broker = Broker(self.config, self.drive, clock=lambda: self.now)
        again = self.broker.create(self.submission(), 'client1')
        self.assertTrue(again['already_complete'])
        self.assertFalse(again['targets'])
        self.assertEqual(len(self.drive.files), 2)

    def test_retries_cancel_old_capabilities_and_different_payload_same_revision_rejected(self):
        first = self.broker.create(self.submission(), 'client1')
        self.broker.create(self.submission(), 'client1')
        self.assertEqual(set(self.drive.cancelled), {t['upload_url'] for t in first['targets']})
        with self.assertRaises(BrokerError) as error:
            self.broker.create(self.submission(species=['狼']), 'client1')
        self.assertEqual(error.exception.status, 409)

    def test_delete_only_removes_owned_files_and_retains_other_submissions(self):
        own = self.broker.create(self.submission(), 'client1')
        other = self.broker.create(self.submission(sample_id='b' * 32, secret='x' * 43), 'client2')
        self.drive.upload_all(own)
        self.drive.upload_all(other)
        self.broker.delete('a' * 32, self.secret, 'client1')
        self.assertEqual(len(self.drive.files), 2)
        self.assertTrue(all('b' * 32 in p for p in self.drive.files))
        self.broker.delete('a' * 32, self.secret, 'client1')

    def test_daily_byte_budget_and_per_minute_limit_persist(self):
        self.config.daily_ip_bytes = len(self.photo) + len(self.annotation) + 10
        self.broker = Broker(self.config, self.drive, clock=lambda: self.now)
        self.broker.create(self.submission(), 'client1')
        with self.assertRaises(BrokerError) as error:
            self.broker.create(self.submission(sample_id='b' * 32), 'client1')
        self.assertEqual(error.exception.status, 429)
        self.config.requests_per_minute = 1
        self.broker = Broker(self.config, self.drive, clock=lambda: self.now)
        self.broker.create(self.submission(sample_id='c' * 32), 'client2')
        with self.assertRaises(BrokerError) as error:
            self.broker.create(self.submission(sample_id='d' * 32), 'client2')
        self.assertEqual(error.exception.status, 429)

    def test_expired_data_removed_without_touching_recent_submission(self):
        old = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(old)
        self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.now += 366 * 86400
        recent = self.broker.create(self.submission(sample_id='b' * 32), 'client2')
        self.drive.upload_all(recent)
        self.broker.complete('b' * 32, 1, self.secret, 'client2')
        self.broker.cleanup()
        self.assertEqual(len(self.drive.files), 2)
        self.assertTrue(all('b' * 32 in p for p in self.drive.files))

    def test_completion_rejects_and_removes_mismatched_actual_size(self):
        response = self.broker.create(self.submission(image_bytes=1), 'client1')
        self.drive.upload_all(response)
        path, _ = self.drive.sessions[response['targets'][0]['upload_url']]
        self.drive.files[path] = 500
        with self.assertRaises(BrokerError) as error:
            self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.assertEqual(error.exception.status, 413)
        self.assertFalse(self.drive.files)

    def test_cancel_removes_finished_partial_files_but_not_committed_sample(self):
        response = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(response)
        token = response['targets'][0]['cancel_url'].rsplit('/', 1)[1]
        self.broker.cancel_upload(token, 'client1')
        self.assertFalse(self.drive.files)
        second = self.broker.create(self.submission(), 'client1')
        self.drive.upload_all(second)
        self.broker.complete('a' * 32, 1, self.secret, 'client1')
        with self.assertRaises(BrokerError):
            self.broker.cancel_upload(second['targets'][0]['cancel_url'].rsplit('/', 1)[1], 'client1')
        self.assertEqual(len(self.drive.files), 2)

    def test_retry_partial_upload_removes_owned_files_and_uses_new_names(self):
        first = self.broker.create(self.submission(), 'client1')
        path, size = self.drive.sessions[first['targets'][0]['upload_url']]
        self.drive.files[path] = size
        old_paths = set(self.drive.files)
        second = self.broker.create(self.submission(), 'client1')
        self.assertFalse(self.drive.files)
        self.drive.upload_all(second)
        self.broker.complete('a' * 32, 1, self.secret, 'client1')
        self.assertFalse(old_paths & set(self.drive.files))
        self.broker.delete('a' * 32, self.secret, 'client1')
        self.assertFalse(self.drive.files)


if __name__ == '__main__':
    unittest.main()
