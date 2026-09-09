import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import system.dinov3 as dinov3
from system.model_sync.layout import get_model_layout


class _FakeDinoCloud:
    def __init__(self, *, fail_on_root: bool = False):
        self.fail_on_root = fail_on_root

    def download_tree(self, relative_path: str, target: Path, on_progress=None) -> None:
        if relative_path != "":
            raise AssertionError(f"strict mirror must download the root, got {relative_path!r}")
        if self.fail_on_root:
            raise RuntimeError("fake cloud failure")
        target.mkdir(parents=True, exist_ok=True)
        package = target / "source" / "dinov3"
        (package / "hub").mkdir(parents=True)
        (package / "models").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "hub" / "backbones.py").write_text("", encoding="utf-8")
        (package / "models" / "vision_transformer.py").write_text("", encoding="utf-8")
        backbone = target / "dinov3-vitb16" / dinov3.DINO_BACKBONE_FILENAME
        backbone.parent.mkdir(parents=True)
        backbone.write_bytes(b"backbone")
        (target / dinov3.DINO_CLASSIFIER_FILENAME).write_bytes(b"classifier")
        (target / dinov3.DINO_MODEL_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "dinov3",
                    "checkpoint": dinov3.DINO_CLASSIFIER_FILENAME,
                    "architecture": "dinov3_vitb16",
                    "feature_dim": 768,
                }
            ),
            encoding="utf-8",
        )
        (target / "LICENSE_DINOv3.md").write_text("license", encoding="utf-8")
        (target / "install.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "component": "dinov3",
                    "component_version": 2,
                    "architecture": "dinov3_vitb16",
                    "source": {"commit": dinov3.DINO_SOURCE_COMMIT},
                    "backbone": {
                        "filename": f"dinov3-vitb16/{dinov3.DINO_BACKBONE_FILENAME}",
                        "sha256": dinov3.DINO_BACKBONE_SHA256,
                    },
                    "classifier": {
                        "filename": dinov3.DINO_CLASSIFIER_FILENAME,
                        "sha256": dinov3.DINO_CLASSIFIER_SHA256,
                        "head_type": "multi_prototype",
                        "selection_k": 3,
                        "manifest": dinov3.DINO_MODEL_MANIFEST_FILENAME,
                    },
                    "files": [],
                }
            ),
            encoding="utf-8",
        )


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
                root.resolve()
                / "dinov3-vitb16"
                / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            )
            self.assertEqual(paths.classifier.name, "multi_prototype.pt")
            self.assertEqual(paths.model_manifest.name, "multi_prototype.neri.json")
            self.assertEqual(paths.license, root.resolve() / "LICENSE_DINOv3.md")
            self.assertEqual(paths.install_manifest, root.resolve() / "install.json")

    def test_complete_component_is_healthy_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            _FakeDinoCloud().download_tree("", root)
            paths = dinov3.dinov3_component_paths(root=root)
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
            shared_torch = (
                base / "toolkit" / "Lib" / "site-packages" / "torch" / "__init__.py"
            )
            shared_torch.parent.mkdir(parents=True)
            shared_torch.write_text("shared", encoding="utf-8")

            dinov3.remove_dinov3_component(root=root)
            dinov3.remove_dinov3_component(root=root)

            self.assertFalse(root.exists())
            self.assertEqual(shared_torch.read_text(encoding="utf-8"), "shared")

    def test_cloud_client_uses_canonical_nericloud_component_root(self) -> None:
        client = dinov3.DinoV3CloudClient(origin="https://cloud.example")
        self.assertEqual(
            client.mount_path("source/dinov3"),
            "/Neri_Data/Model/DINOv3/source/dinov3",
        )

    def test_install_mirrors_cloud_tree_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "res" / "model" / "DINOv3"
            with mock.patch(
                "system.dinov3.component._sha256_file",
                side_effect=lambda path: (
                    dinov3.DINO_BACKBONE_SHA256
                    if Path(path).name == dinov3.DINO_BACKBONE_FILENAME
                    else dinov3.DINO_CLASSIFIER_SHA256
                ),
            ):
                installed = dinov3.install_dinov3_component(
                    root=root,
                    cloud_client=_FakeDinoCloud(),
                )
                status = dinov3.dinov3_component_status(root=root)
            self.assertEqual(installed, root.resolve())
            self.assertTrue(status["healthy"])
            self.assertTrue((root / "source" / "dinov3" / "__init__.py").is_file())
            self.assertTrue((root / "LICENSE_DINOv3.md").is_file())
            self.assertTrue((root / "multi_prototype.pt").is_file())

    def test_failed_install_preserves_previous_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DINOv3"
            root.mkdir()
            marker = root / "existing.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                dinov3.install_dinov3_component(
                    root=root,
                    cloud_client=_FakeDinoCloud(fail_on_root=True),
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
