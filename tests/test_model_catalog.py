import os
import tempfile
import unittest
from pathlib import Path

from system.model_sync.catalog import discover_models, resolve_saved_model_path
from system.model_sync.layout import get_model_layout


class ModelCatalogTests(unittest.TestCase):
    def test_discovers_duplicate_names_from_user_and_sync(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_user / "bird.pt").write_bytes(b"user")
            (layout.detect_sync / "bird.pt").write_bytes(b"sync")

            models = discover_models(layout, "detect")

            self.assertEqual(
                [(item.name, item.source, item.kind) for item in models],
                [("bird.pt", "user", "detect"), ("bird.pt", "sync", "detect")],
            )
            self.assertNotEqual(models[0].path, models[1].path)

    def test_detect_and_cls_extensions_are_filtered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            for name in ("a.pt", "b.onnx"):
                (layout.detect_user / name).write_bytes(b"x")
            for name in ("a.pt", "b.onnx", "c.engine", "d.txt"):
                (layout.cls_user / name).write_bytes(b"x")

            self.assertEqual([m.name for m in discover_models(layout, "detect")], ["a.pt"])
            self.assertEqual(
                [m.name for m in discover_models(layout, "cls")],
                ["a.pt", "b.onnx", "c.engine"],
            )

    def test_orders_user_before_sync_and_names_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_user / "z.pt").write_bytes(b"z")
            (layout.detect_user / "A.pt").write_bytes(b"a")
            (layout.detect_sync / "b.pt").write_bytes(b"b")

            models = discover_models(layout, "detect")
            self.assertEqual(
                [(m.source, m.name) for m in models],
                [("user", "A.pt"), ("user", "z.pt"), ("sync", "b.pt")],
            )

    def test_saved_selection_prefers_exact_path_then_user_legacy_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_user / "bird.pt").write_bytes(b"user")
            (layout.detect_sync / "bird.pt").write_bytes(b"sync")
            models = discover_models(layout, "detect")
            user = next(m for m in models if m.source == "user")
            sync = next(m for m in models if m.source == "sync")

            self.assertEqual(resolve_saved_model_path(sync.path, models), sync.path)
            self.assertEqual(resolve_saved_model_path("bird.pt", models), user.path)
            self.assertIsNone(resolve_saved_model_path("missing.pt", models))

    @unittest.skipUnless(os.name == "nt", "case-only path compatibility is Windows-specific")
    def test_alpha2_uppercase_saved_path_resolves_after_lowercase_rename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.detect_sync / "bird.pt").write_bytes(b"sync")
            models = discover_models(layout, "detect")
            current = models[0].path
            old = current.replace("\\model\\", "\\Model\\")
            self.assertNotEqual(old, current)
            self.assertEqual(resolve_saved_model_path(old, models), current)

    def test_unique_legacy_filename_resolves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = get_model_layout(Path(temp_dir) / "res")
            (layout.cls_sync / "only.onnx").write_bytes(b"x")
            models = discover_models(layout, "cls")
            self.assertEqual(resolve_saved_model_path("only.onnx", models), models[0].path)


if __name__ == "__main__":
    unittest.main()
