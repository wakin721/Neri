import tempfile
import unittest
from pathlib import Path

from system.model_sync.layout import get_model_layout, migrate_legacy_layout


class ModelLayoutTests(unittest.TestCase):
    def test_canonical_model_root_is_lowercase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = get_model_layout(Path(tmp))
            self.assertEqual(layout.root, Path(tmp) / "model")
            self.assertEqual(layout.detect_user, Path(tmp) / "model" / "detect" / "user")
            self.assertEqual(layout.detect_sync, Path(tmp) / "model" / "detect" / "sync")
            self.assertEqual(layout.cls_user, Path(tmp) / "model" / "cls" / "user")
            self.assertEqual(layout.cls_sync, Path(tmp) / "model" / "cls" / "sync")
            self.assertEqual(layout.tracker, Path(tmp) / "model" / "tracker.yaml")

    def test_flat_lowercase_legacy_models_migrate_into_user_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_detect = root / "model"
            legacy_cls = root / "model_cls"
            legacy_detect.mkdir()
            legacy_cls.mkdir()
            (legacy_detect / "detect.pt").write_bytes(b"detect")
            (legacy_cls / "classify.pt").write_bytes(b"classify")
            (legacy_cls / "tracker.yaml").write_text("tracker", encoding="utf-8")

            report = migrate_legacy_layout(root)
            layout = get_model_layout(root)

            self.assertEqual(report.moved, 3)
            self.assertEqual((layout.detect_user / "detect.pt").read_bytes(), b"detect")
            self.assertEqual((layout.cls_user / "classify.pt").read_bytes(), b"classify")
            self.assertEqual(layout.tracker.read_text(encoding="utf-8"), "tracker")

    def test_existing_canonical_subdirectories_do_not_block_flat_file_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_detect = root / "model"
            (legacy_detect / "detect" / "user").mkdir(parents=True)
            (legacy_detect / "detect" / "sync").mkdir(parents=True)
            (legacy_detect / "cls" / "user").mkdir(parents=True)
            (legacy_detect / "cls" / "sync").mkdir(parents=True)
            (legacy_detect / "legacy.pt").write_bytes(b"legacy")

            report = migrate_legacy_layout(root)

            self.assertEqual(report.moved, 1)
            self.assertFalse((legacy_detect / "legacy.pt").exists())
            self.assertEqual(
                (legacy_detect / "detect" / "user" / "legacy.pt").read_bytes(),
                b"legacy",
            )

    def test_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model").mkdir()
            (root / "model" / "a.pt").write_bytes(b"a")
            first = migrate_legacy_layout(root)
            second = migrate_legacy_layout(root)
            self.assertEqual(first.moved, 1)
            self.assertEqual(second.moved, 0)

    def test_collision_uses_deterministic_legacy_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = get_model_layout(root)
            layout.detect_user.mkdir(parents=True)
            (layout.detect_user / "same.pt").write_bytes(b"new")
            (root / "model" / "same.pt").write_bytes(b"old")

            report = migrate_legacy_layout(root)

            self.assertEqual(report.moved, 1)
            self.assertEqual(report.collisions, ("same.legacy-1.pt",))
            self.assertEqual((layout.detect_user / "same.pt").read_bytes(), b"new")
            self.assertEqual((layout.detect_user / "same.legacy-1.pt").read_bytes(), b"old")

    def test_only_allowed_model_extensions_are_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model").mkdir()
            (root / "model_cls").mkdir()
            (root / "model" / "keep.txt").write_text("x", encoding="utf-8")
            (root / "model" / "ignore.onnx").write_bytes(b"x")
            (root / "model_cls" / "classify.onnx").write_bytes(b"onnx")
            (root / "model_cls" / "classify.engine").write_bytes(b"engine")

            migrate_legacy_layout(root)
            layout = get_model_layout(root)

            self.assertTrue((root / "model" / "keep.txt").exists())
            self.assertTrue((root / "model" / "ignore.onnx").exists())
            self.assertTrue((layout.cls_user / "classify.onnx").exists())
            self.assertTrue((layout.cls_user / "classify.engine").exists())

    def test_model_cls_tracker_is_preferred_over_flat_model_tracker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model").mkdir()
            (root / "model_cls").mkdir()
            (root / "model" / "tracker.yaml").write_text("detect", encoding="utf-8")
            (root / "model_cls" / "tracker.yaml").write_text("cls", encoding="utf-8")

            migrate_legacy_layout(root)
            layout = get_model_layout(root)

            self.assertEqual(layout.tracker.read_text(encoding="utf-8"), "cls")

    def test_flat_model_tracker_is_fallback_when_model_cls_tracker_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model").mkdir()
            (root / "model" / "tracker.yaml").write_text("detect", encoding="utf-8")

            migrate_legacy_layout(root)
            layout = get_model_layout(root)

            self.assertEqual(layout.tracker.read_text(encoding="utf-8"), "detect")

    def test_existing_canonical_tracker_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = get_model_layout(root)
            layout.root.mkdir(parents=True)
            layout.tracker.write_text("canonical", encoding="utf-8")
            (root / "model_cls").mkdir()
            (root / "model_cls" / "tracker.yaml").write_text("legacy", encoding="utf-8")

            migrate_legacy_layout(root)

            self.assertEqual(layout.tracker.read_text(encoding="utf-8"), "canonical")

    def test_migration_creates_all_canonical_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = get_model_layout(Path(tmp))
            migrate_legacy_layout(Path(tmp))
            for path in (
                layout.root,
                layout.detect_user,
                layout.detect_sync,
                layout.cls_user,
                layout.cls_sync,
            ):
                self.assertTrue(path.is_dir())


if __name__ == "__main__":
    unittest.main()
