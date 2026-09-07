import os
import tempfile
import unittest
from pathlib import Path

from system.model_sync.layout import get_model_layout
from system.utils import _canonical_resource_alias


class ModelResourcePathTests(unittest.TestCase):
    def test_legacy_model_resource_paths_redirect_to_lowercase_canonical_layout(self) -> None:
        self.assertEqual(
            os.path.normpath(_canonical_resource_alias("res/model/legacy.pt")),
            os.path.normpath("res/model/detect/user/legacy.pt"),
        )
        self.assertEqual(
            os.path.normpath(_canonical_resource_alias("res/model_cls/legacy.pt")),
            os.path.normpath("res/model/cls/user/legacy.pt"),
        )
        self.assertEqual(
            os.path.normpath(_canonical_resource_alias("res/model_cls/tracker.yaml")),
            os.path.normpath("res/model/tracker.yaml"),
        )

    def test_canonical_lowercase_paths_are_not_rewritten_as_legacy_files(self) -> None:
        for value in (
            "res/model/detect/user/bird.pt",
            "res/model/detect/sync/bird.pt",
            "res/model/cls/user/bird.pt",
            "res/model/cls/sync/bird.pt",
            "res/model/tracker.yaml",
            "res/model/.sync-state.json",
        ):
            self.assertEqual(
                os.path.normpath(_canonical_resource_alias(value)),
                os.path.normpath(value),
            )

    def test_model_layout_uses_lowercase_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resource_root = Path(tmp).resolve()
            layout = get_model_layout(resource_root)
            self.assertEqual(layout.root.name, "model")
            self.assertEqual(layout.root.parent, resource_root)


if __name__ == "__main__":
    unittest.main()
