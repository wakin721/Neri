import unittest
from pathlib import Path
from unittest.mock import patch

from system.model_sync.layout import get_model_layout
from system.utils import resource_path


class ModelResourcePathTests(unittest.TestCase):
    def test_legacy_model_resource_paths_redirect_to_lowercase_canonical_layout(self) -> None:
        with patch("system.utils._resource_root", return_value=Path(r"C:\Neri")):
            self.assertEqual(
                Path(resource_path("res/model/legacy.pt")),
                Path(r"C:\Neri\res\model\detect\user\legacy.pt"),
            )
            self.assertEqual(
                Path(resource_path("res/model_cls/legacy.pt")),
                Path(r"C:\Neri\res\model\cls\user\legacy.pt"),
            )
            self.assertEqual(
                Path(resource_path("res/model_cls/tracker.yaml")),
                Path(r"C:\Neri\res\model\tracker.yaml"),
            )

    def test_model_layout_uses_lowercase_model_directory(self) -> None:
        layout = get_model_layout(Path(r"C:\Neri\res"))
        self.assertEqual(layout.root, Path(r"C:\Neri\res\model"))


if __name__ == "__main__":
    unittest.main()
