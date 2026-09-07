import unittest
from pathlib import Path

from system.model_sync.layout import get_model_layout, migrate_legacy_layout
from system.version import BASE_VERSION, DEFAULT_BUILD_NUMBER


class Alpha3ReleaseContractTests(unittest.TestCase):
    def test_release_version_is_alpha3_build_460(self):
        self.assertEqual(BASE_VERSION, "3.0.6-alpha3")
        self.assertEqual(DEFAULT_BUILD_NUMBER, 460)

    def test_canonical_local_model_root_is_lowercase_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            layout = get_model_layout(Path(tmp))
            self.assertEqual(layout.root.name, "model")
            self.assertEqual(layout.detect_user, Path(tmp).resolve() / "model" / "detect" / "user")
            self.assertEqual(layout.cls_sync, Path(tmp).resolve() / "model" / "cls" / "sync")
            self.assertEqual(layout.tracker, Path(tmp).resolve() / "model" / "tracker.yaml")

    def test_flat_legacy_lowercase_model_is_migrated_in_place(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            resource_root = Path(tmp)
            legacy_root = resource_root / "model"
            legacy_root.mkdir(parents=True)
            (legacy_root / "legacy.pt").write_bytes(b"legacy")
            (legacy_root / "tracker.yaml").write_text("tracker_type: botsort\n", encoding="utf-8")

            report = migrate_legacy_layout(resource_root)
            layout = get_model_layout(resource_root)

            self.assertGreaterEqual(report.moved, 2)
            self.assertEqual((layout.detect_user / "legacy.pt").read_bytes(), b"legacy")
            self.assertTrue(layout.tracker.is_file())


if __name__ == "__main__":
    unittest.main()
