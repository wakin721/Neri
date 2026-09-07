from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from system.training.queue import TrainingQueue


class TrainingEventKeyMigrationTests(unittest.TestCase):
    def test_legacy_jobs_table_without_event_key_is_migrated_before_index_creation(self):
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            queue = TrainingQueue(state_dir, transport=object())
            queue.stop()

            with closing(sqlite3.connect(queue.db_path)) as db:
                db.execute("DROP INDEX IF EXISTS jobs_event")
                db.execute("ALTER TABLE jobs DROP COLUMN event_key")
                db.commit()

            migrated = TrainingQueue(state_dir, transport=object())
            try:
                with closing(sqlite3.connect(migrated.db_path)) as db:
                    columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
                    indexes = {row[1] for row in db.execute("PRAGMA index_list(jobs)")}
                self.assertIn("event_key", columns)
                self.assertIn("jobs_event", indexes)
            finally:
                migrated.stop()


if __name__ == "__main__":
    unittest.main()
