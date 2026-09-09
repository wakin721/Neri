import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import system.dinov3 as dinov3
from system.model_sync.layout import get_model_layout


class DinoV3ComponentLayoutTests(unittest.TestCase):
    def test_model_layout_exposes_dinov3_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resource_root = Path(tmp).resolve()
            layout = get_model_layout(resource_root)
            self.assertEqual(layout.dinov3_root, resource_root / "model" / "DINOv3")

    def test_dinov3_package_exposes_component_status_api(self) -> None:
        self.assertTrue(hasattr(dinov3, "dinov3_component_status"))

    def test_existing_incomplete_component_is_installed_but_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            root.mkdir()
            status = dinov3.dinov3_component_status(root=root)
            self.assertTrue(status["installed"])
            self.assertFalse(status["healthy"])

    def test_component_paths_match_res_model_dinov3_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            paths = dinov3.dinov3_component_paths(root=root)
            self.assertEqual(paths.root, root.resolve())
            self.assertEqual(paths.source_package, root.resolve() / "source" / "dinov3")
            self.assertEqual(
                paths.backbone,
                root.resolve() / "dinov3-vitb16" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            )
            self.assertEqual(paths.license, root.resolve() / "LICENSE_DINOv3.md")
            self.assertEqual(paths.install_manifest, root.resolve() / "install.json")

    def test_complete_component_is_healthy_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            paths = dinov3.dinov3_component_paths(root=root)
            paths.source_package.mkdir(parents=True)
            paths.backbone.parent.mkdir(parents=True)
            (paths.source_package / "__init__.py").write_text("", encoding="utf-8")
            (paths.source_package / "hub").mkdir()
            (paths.source_package / "hub" / "backbones.py").write_text("", encoding="utf-8")
            (paths.source_package / "models").mkdir()
            (paths.source_package / "models" / "vision_transformer.py").write_text("", encoding="utf-8")
            paths.backbone.write_bytes(b"backbone")
            paths.classifier.write_bytes(b"classifier")
            paths.license.write_text("license", encoding="utf-8")
            paths.install_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "component": "dinov3",
                        "component_version": 1,
                        "architecture": "dinov3_vitb16",
                        "source": {"commit": dinov3.DINO_SOURCE_COMMIT},
                        "backbone": {"sha256": dinov3.DINO_BACKBONE_SHA256},
                    }
                ),
                encoding="utf-8",
            )
            paths.model_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "backend": "dinov3",
                        "checkpoint": paths.classifier.name,
                        "architecture": "dinov3_vitb16",
                        "feature_dim": 768,
                        "encoder_sha256": dinov3.DINO_BACKBONE_SHA256,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "system.dinov3.component._sha256_file",
                side_effect=lambda path: (
                    dinov3.DINO_BACKBONE_SHA256
                    if Path(path).name == paths.backbone.name
                    else dinov3.DINO_CLASSIFIER_SHA256
                ),
            ):
                status = dinov3.dinov3_component_status(root=root)
            self.assertTrue(status["installed"])
            self.assertTrue(status["healthy"])

    def test_remove_is_idempotent_and_keeps_shared_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "res" / "model" / "DINOv3"
            root.mkdir(parents=True)
            (root / "install.json").write_text("{}", encoding="utf-8")
            shared_torch = base / "toolkit" / "Lib" / "site-packages" / "torch" / "__init__.py"
            shared_torch.parent.mkdir(parents=True)
            shared_torch.write_text("shared", encoding="utf-8")

            dinov3.remove_dinov3_component(root=root)
            dinov3.remove_dinov3_component(root=root)

            self.assertFalse(root.exists())
            self.assertEqual(shared_torch.read_text(encoding="utf-8"), "shared")


if __name__ == "__main__":
    unittest.main()
