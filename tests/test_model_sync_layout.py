import tempfile
import unittest
from pathlib import Path

from system.model_sync.layout import (
    get_model_layout,
    migrate_legacy_layout,
    resolve_tracker_config,
)


class ModelLayoutMigrationTests(unittest.TestCase):
    def test_migrates_legacy_models_and_tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            (resource_root / "model").mkdir(parents=True)
            (resource_root / "model_cls").mkdir(parents=True)
            (resource_root / "model" / "detect.pt").write_bytes(b"det")
            (resource_root / "model_cls" / "classify.onnx").write_bytes(b"cls")
            (resource_root / "model_cls" / "tracker.yaml").write_text(
                "tracker_type: botsort\n", encoding="utf-8"
            )

            report = migrate_legacy_layout(resource_root)

            self.assertTrue((resource_root / "Model/detect/user/detect.pt").is_file())
            self.assertTrue((resource_root / "Model/cls/user/classify.onnx").is_file())
            self.assertTrue((resource_root / "Model/tracker.yaml").is_file())
            self.assertEqual(report.moved, 3)

    def test_migration_is_idempotent_and_preserves_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            legacy = resource_root / "model"
            target = resource_root / "Model/detect/user"
            legacy.mkdir(parents=True)
            target.mkdir(parents=True)
            (legacy / "same.pt").write_bytes(b"legacy")
            (target / "same.pt").write_bytes(b"user")

            first = migrate_legacy_layout(resource_root)
            second = migrate_legacy_layout(resource_root)

            self.assertEqual((target / "same.pt").read_bytes(), b"user")
            retained = sorted(target.glob("same.legacy-*.pt"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"legacy")
            self.assertEqual(first.moved, 1)
            self.assertEqual(second.moved, 0)

    def test_only_approved_extensions_migrate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            detect = resource_root / "model"
            cls = resource_root / "model_cls"
            detect.mkdir(parents=True)
            cls.mkdir(parents=True)
            (detect / "keep.pt").write_bytes(b"pt")
            (detect / "ignore.onnx").write_bytes(b"onnx")
            (detect / "readme.txt").write_text("x", encoding="utf-8")
            (cls / "keep.pt").write_bytes(b"pt")
            (cls / "keep.onnx").write_bytes(b"onnx")
            (cls / "keep.engine").write_bytes(b"engine")
            (cls / "ignore.txt").write_text("x", encoding="utf-8")

            migrate_legacy_layout(resource_root)

            layout = get_model_layout(resource_root)
            self.assertTrue((layout.detect_user / "keep.pt").exists())
            self.assertFalse((layout.detect_user / "ignore.onnx").exists())
            self.assertTrue((layout.cls_user / "keep.pt").exists())
            self.assertTrue((layout.cls_user / "keep.onnx").exists())
            self.assertTrue((layout.cls_user / "keep.engine").exists())
            self.assertTrue((detect / "ignore.onnx").exists())
            self.assertTrue((detect / "readme.txt").exists())
            self.assertTrue((cls / "ignore.txt").exists())

    def test_tracker_preference_is_canonical_then_cls_then_detect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            cls_tracker = resource_root / "model_cls/tracker.yaml"
            detect_tracker = resource_root / "model/tracker.yaml"
            cls_tracker.parent.mkdir(parents=True)
            detect_tracker.parent.mkdir(parents=True)
            cls_tracker.write_text("cls\n", encoding="utf-8")
            detect_tracker.write_text("detect\n", encoding="utf-8")

            migrate_legacy_layout(resource_root)

            canonical = resource_root / "Model/tracker.yaml"
            self.assertEqual(canonical.read_text(encoding="utf-8"), "cls\n")
            self.assertTrue(detect_tracker.exists())

            canonical.write_text("canonical\n", encoding="utf-8")
            cls_tracker.write_text("new-cls\n", encoding="utf-8")
            migrate_legacy_layout(resource_root)
            self.assertEqual(canonical.read_text(encoding="utf-8"), "canonical\n")
            self.assertTrue(cls_tracker.exists())

    def test_detect_tracker_is_fallback_when_cls_tracker_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            detect_tracker = resource_root / "model/tracker.yaml"
            detect_tracker.parent.mkdir(parents=True)
            detect_tracker.write_text("detect\n", encoding="utf-8")

            migrate_legacy_layout(resource_root)

            self.assertEqual(
                (resource_root / "Model/tracker.yaml").read_text(encoding="utf-8"),
                "detect\n",
            )

    def test_creates_all_canonical_directories_and_tracker_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_root = Path(temp_dir) / "res"
            layout = get_model_layout(resource_root)

            for path in (
                layout.detect_user,
                layout.detect_sync,
                layout.cls_user,
                layout.cls_sync,
            ):
                self.assertTrue(path.is_dir())
            self.assertEqual(resolve_tracker_config(resource_root), "botsort.yaml")
            layout.tracker.write_text("tracker_type: botsort\n", encoding="utf-8")
            self.assertEqual(resolve_tracker_config(resource_root), str(layout.tracker))


if __name__ == "__main__":
    unittest.main()
