from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from system.training.events import EVENT_GAP_SECONDS, TriggerEventResolver
from system.training.media import prepare_sample
from system.training.queue import TrainingQueue
from system.training.species import SpeciesNameResolver
from system.training.transport import UploadCancelled


def save_camera_photo(path: Path, taken: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[36867] = taken.strftime('%Y:%m:%d %H:%M:%S')
    Image.new('RGB', (100, 50), 'green').save(path, exif=exif)


def create_species_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE species (中文名 TEXT PRIMARY KEY, 学名 TEXT, 物种类型 TEXT)')
        db.execute(
            'INSERT INTO species(中文名, 学名, 物种类型) VALUES (?,?,?)',
            ('棕脸鹟莺', 'Abroscopus albogularis', '鸟类'),
        )
        db.commit()


class MemoryCloud:
    def __init__(self) -> None:
        self.files: dict[str, dict] = {}

    def upload(self, job, photo, annotation, is_current):
        if not is_current():
            raise UploadCancelled()
        self.files[job['sample_id']] = json.loads(annotation)

    def delete(self, job, is_current):
        if not is_current():
            raise UploadCancelled()
        self.files.pop(job['sample_id'], None)


class SpeciesNameResolverTests(unittest.TestCase):
    def test_database_scientific_name_is_normalized_for_yolo_metadata(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'species.db'
            create_species_db(db_path)
            resolver = SpeciesNameResolver(db_path)
            self.assertEqual(
                resolver.resolve('棕脸鹟莺'),
                'Abroscopus_albogularis',
            )

    def test_missing_species_falls_back_to_underscore_pinyin(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'species.db'
            create_species_db(db_path)
            resolver = SpeciesNameResolver(db_path)
            self.assertEqual(resolver.resolve('测试鸟'), 'ce_shi_niao')


class TriggerEventResolverTests(unittest.TestCase):
    def test_adjacent_photos_within_gap_form_one_transitive_trigger_event(self):
        self.assertEqual(EVENT_GAP_SECONDS, 10)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'camera'
            start = datetime(2026, 9, 7, 12, 0, 0)
            first = folder / '001.jpg'
            second = folder / '002.jpg'
            third = folder / '003.jpg'
            separate = folder / '004.jpg'
            save_camera_photo(first, start)
            save_camera_photo(second, start + timedelta(seconds=8))
            save_camera_photo(third, start + timedelta(seconds=17))
            save_camera_photo(separate, start + timedelta(seconds=30))

            resolver = TriggerEventResolver()
            keys = [
                resolver.event_key(path, folder_key='physical-camera')
                for path in (first, second, third, separate)
            ]
            self.assertEqual(keys[0], keys[1])
            self.assertEqual(keys[1], keys[2])
            self.assertNotEqual(keys[2], keys[3])


class TrainingYoloMediaTests(unittest.TestCase):
    def test_yolo_label_uses_zero_class_and_scientific_name_sidecar(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'camera.jpg'
            save_camera_photo(path, datetime(2026, 9, 7, 12, 0, 0))
            photo, package = prepare_sample(
                path,
                {
                    'sample_id': 'a' * 32,
                    'revision': 1,
                    'payload': {
                        'species': ['棕脸鹟莺'],
                        'training_names': {'棕脸鹟莺': 'Abroscopus_albogularis'},
                        'model_boxes': [
                            {
                                'species': '棕脸鹟莺',
                                'bbox': [10, 5, 60, 30],
                                'confidence': 0.9,
                            }
                        ],
                    },
                },
            )
            self.assertTrue(photo.startswith(b'\xff\xd8'))
            data = json.loads(package)
            self.assertEqual(data['upload_species'], ['棕脸鹟莺'])
            self.assertEqual(
                data['yolo_labels']['棕脸鹟莺'],
                '0 0.350000 0.350000 0.500000 0.500000\n',
            )
            self.assertEqual(
                data['yolo_classes']['棕脸鹟莺'],
                'Abroscopus_albogularis\n',
            )

    def test_confirmed_empty_photo_has_empty_yolo_label(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.jpg'
            save_camera_photo(path, datetime(2026, 9, 7, 12, 0, 0))
            _, package = prepare_sample(
                path,
                {
                    'sample_id': 'b' * 32,
                    'revision': 1,
                    'payload': {
                        'species': ['空照片'],
                        'training_names': {'空照片': 'background'},
                        'empty': True,
                        'model_boxes': [],
                    },
                },
            )
            data = json.loads(package)
            self.assertEqual(data['upload_species'], ['空照片'])
            self.assertEqual(data['yolo_labels']['空照片'], '')
            self.assertEqual(data['yolo_classes']['空照片'], 'background\n')


class TrainingEventQueueTests(unittest.TestCase):
    def test_only_first_valid_photo_from_one_trigger_event_is_enqueued(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / 'species.db'
            create_species_db(db_path)
            cloud = MemoryCloud()
            queue = TrainingQueue(
                root / 'state',
                transport=cloud,
                clock=lambda: 1000.0,
                debounce_seconds=0,
                species_db_path=db_path,
            )
            try:
                queue.set_consent('2026-09-05', True)
                start = datetime(2026, 9, 7, 12, 0, 0)
                photos = [root / 'camera' / f'{i}.jpg' for i in range(3)]
                for path, seconds in zip(photos, (0, 4, 9)):
                    save_camera_photo(path, start + timedelta(seconds=seconds))
                mark = {
                    '物种名称': '棕脸鹟莺',
                    '检测框': [
                        {'物种': '棕脸鹟莺', '边界框': [10, 5, 60, 30], '置信度': 0.9}
                    ],
                }
                self.assertTrue(queue.enqueue(photos[0], mark))
                self.assertFalse(queue.enqueue(photos[1], mark))
                self.assertFalse(queue.enqueue(photos[2], mark))
                while queue.process_once():
                    pass
                self.assertEqual(len(cloud.files), 1)
                uploaded = next(iter(cloud.files.values()))
                self.assertEqual(
                    uploaded['yolo_classes']['棕脸鹟莺'],
                    'Abroscopus_albogularis\n',
                )
            finally:
                queue.stop()


if __name__ == '__main__':
    unittest.main()
