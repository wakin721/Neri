import os
import shutil
import subprocess
import sys
import textwrap
import tempfile
import unittest
from pathlib import Path

from scripts.package_litert_resources import copy_resources, verify_resources, CANONICAL_DIRECTORIES


class LitePackageContractTests(unittest.TestCase):
    def test_packaged_detector_survives_startup_sync_and_remains_discoverable(self):
        from system.model_sync.layout import get_model_layout
        from system.model_sync.catalog import discover_models
        from system.model_sync.manager import ModelSyncManager
        from system.model_sync.protocol import ModelManifest
        from tests.test_model_sync_manager import FakeClient
        from scripts.package_litert_resources import BUILTIN_DETECTOR
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'res'
            copy_resources(Path(__file__).resolve().parents[1]/'res', output)
            report = verify_resources(output, require_bundled_detector=True)
            self.assertTrue(any(row['path'] == BUILTIN_DETECTOR for row in report['models']))
            self.assertNotIn('classifier_assets', report)
            self.assertFalse((output / 'bundled-models').exists())
            self.assertTrue((output / 'model/cls/bundled').is_dir())
            self.assertEqual(list((output / 'model/cls/bundled').rglob('*')), [])
            layout = get_model_layout(output)
            status = ModelSyncManager(layout, FakeClient(ModelManifest('a'*64, ()))).sync_once_for_test()
            self.assertEqual(status.state, 'completed')
            model = next(row for row in discover_models(layout, 'detect')
                         if row.name == 'MDV6-yolov10-c·LiteRT')
            self.assertEqual(model.source, 'bundled')
            self.assertEqual(model.backend, 'litert')
            self.assertEqual(Path(model.path).name, 'MDV6-yolov10-c.tflite')
            self.assertTrue(Path(model.path).is_file())
            self.assertEqual(Path(model.path).read_bytes(),
                (Path(__file__).resolve().parents[1]/'res'/BUILTIN_DETECTOR).read_bytes())
            self.assertEqual([row for row in discover_models(layout, 'cls') if row.source == 'bundled'], [])

    def test_distribution_verifier_requires_the_builtin_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            other = source/'model/detect/user/another.tflite'
            other.parent.mkdir(parents=True)
            other.write_bytes(b'other model')
            copy_resources(source, output)
            with self.assertRaisesRegex(ValueError, 'Missing bundled LiteRT'):
                verify_resources(output, require_bundled_detector=True)

    def test_lite_model_catalog_and_component_api_import_without_torch(self):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
            import importlib.abc, sys, tempfile
            from pathlib import Path
            attempts=[]
            class MissingTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname=='torch' or fullname.startswith('torch.'):
                        attempts.append(fullname)
                        raise ModuleNotFoundError('PyTorch is optional', name='torch')
            sys.meta_path.insert(0,MissingTorch())
            from system.model_sync.catalog import discover_models
            from system.model_sync.layout import get_model_layout
            from system.dinov3.component import dinov3_component_status
            from system.backend.dinov3_environment import dinov3_environment_router
            with tempfile.TemporaryDirectory() as temp:
                layout=get_model_layout(Path(temp)/'res')
                (layout.detect_sync/'fixture.tflite').write_bytes(b'fixture')
                assert any(model.backend=='litert' for model in discover_models(layout,'detect'))
                assert not dinov3_component_status(root=layout.dinov3_root)['installed']
                assert dinov3_environment_router()
            assert attempts==[],attempts
        ''')], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_resources_include_litert_and_matching_sidecars_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'output'
            files = {
                'translate.json': '{}', 'demo/help.md': 'help',
                'model/tracker.yaml': 'tracker_type: botsort',
                'model/detect/sync/detector.tflite': 'LiteRT model',
                'model/detect/sync/detector.labels.txt': 'animal',
                'model/detect/sync/detector.LICENSE.md': 'license',
                'model/detect/sync/detector.SOURCE.md': 'source',
                'model/detect/user/another.litert': 'another LiteRT model',
                'model/detect/user/legacy.pt': 'legacy',
                'model/cls/user/classifier.onnx': 'classifier',
                'model/cls/user/experiment.neri.json': '{}',
                'model/cls/user/assets/encoder.pth': 'encoder',
                'model/cls/user/assets/model.neri-model': 'classifier head',
                'model/cls/user/assets/calibration.npz': 'calibration',
                'model/DINOv3/weights.safetensors': 'component',
                'model/detect/sync/legacy.labels.txt': 'stale labels',
                'Model/old.engine': 'uppercase legacy',
                'model_cls/old.pt': 'old directory',
                'demo/downloaded-model.zip': 'bundled archive',
            }
            for relative, contents in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, 'utf-8')
            result = copy_resources(source, output)
            self.assertEqual(result['policy'], 'litert-only')
            self.assertEqual(len(result['models']), 2)
            self.assertTrue((output/'translate.json').is_file())
            self.assertTrue((output/'demo/help.md').is_file())
            self.assertTrue((output/'model/tracker.yaml').is_file())
            self.assertTrue((output/'model/detect/sync/detector.LICENSE.md').is_file())
            self.assertFalse((output/'model/detect/sync/legacy.labels.txt').exists())
            self.assertFalse((output/'model/DINOv3').exists())
            self.assertNotIn('Model', [path.name for path in output.iterdir()])
            self.assertFalse((output/'model/old.engine').exists())
            self.assertFalse((output/'model_cls').exists())
            self.assertFalse((output/'demo/downloaded-model.zip').exists())
            for directory in CANONICAL_DIRECTORIES:
                self.assertTrue((output/'model'/directory).is_dir())
            for relative, contents in files.items():
                self.assertEqual((source/relative).read_text('utf-8'), contents)
            self.assertEqual(verify_resources(output), result)

    def test_model_verifier_rejects_unexpected_weights_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            model = source/'model/detect/sync/lite.tflite'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'LiteRT model')
            copy_resources(source, output)
            extra = output/'model/cls/user/unexpected.pt'
            extra.write_bytes(b'forbidden weights')
            with self.assertRaisesRegex(ValueError, 'Non-LiteRT'):
                verify_resources(output)
            extra.unlink()
            (output/'model/detect/sync/lite.tflite').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_resources(output)

    def test_resource_packager_does_not_overwrite_existing_user_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            source.mkdir()
            sentinel = output/'model/cls/user/user.pt'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'keep user model')
            with self.assertRaises(FileExistsError):
                copy_resources(source, output)
            self.assertEqual(sentinel.read_bytes(), b'keep user model')

    def test_pruner_removes_pytorch_but_preserves_unrelated_packages(self):
        repository_root = Path(__file__).resolve().parents[1]
        prune_script = repository_root / "scripts" / "prune_lite_python.ps1"
        self.assertTrue(
            prune_script.is_file(),
            "lite packaging must provide scripts/prune_lite_python.ps1",
        )
        if os.name != "nt":
            self.skipTest("PowerShell packaging regression runs on Windows")

        with tempfile.TemporaryDirectory() as tmp:
            toolkit = Path(tmp) / "toolkit"
            site_packages = toolkit / "Lib" / "site-packages"
            site_packages.mkdir(parents=True)

            pytorch_artifacts = (
                "torch",
                "torchvision",
                "torchvision.libs",
                "torchaudio",
                "torchaudio.lib",
                "torchaudio.libs",
                "functorch",
                "torchgen",
                "torch-2.13.0+cu130.dist-info",
                "torchvision-0.24.0+cu130.dist-info",
                "torchaudio-2.13.0+cu130.dist-info",
            )
            for name in pytorch_artifacts:
                artifact = site_packages / name
                artifact.mkdir(parents=True)
                (artifact / "sentinel.txt").write_text("remove", encoding="utf-8")

            preserved = site_packages / "fastapi"
            preserved.mkdir()
            (preserved / "sentinel.txt").write_text("keep", encoding="utf-8")

            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(prune_script),
                    "-ToolkitPath",
                    str(toolkit),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            for name in pytorch_artifacts:
                self.assertFalse((site_packages / name).exists(), name)
            self.assertEqual(
                (preserved / "sentinel.txt").read_text(encoding="utf-8"),
                "keep",
            )

    def test_package_windows_runs_pruner_after_copying_python(self):
        repository_root = Path(__file__).resolve().parents[1]
        package_script = (repository_root / "scripts" / "package_windows.ps1").read_text(
            encoding="utf-8"
        )
        copy_marker = 'Copy-Item "$pythonRoot\\*" $toolkit -Recurse -Force'
        prune_marker = 'prune_lite_python.ps1'
        self.assertIn(copy_marker, package_script)
        self.assertIn(prune_marker, package_script)
        self.assertGreater(
            package_script.index(prune_marker),
            package_script.index(copy_marker),
            "PyTorch pruning must happen after the build Python environment is copied",
        )
        self.assertIn('package_litert_resources.py', package_script)
        self.assertNotIn('Copy-Item "res" $package -Recurse -Force', package_script)

    def test_package_windows_and_ci_require_mdv6_only_without_bundled_classifier_or_shards(self):
        repository_root = Path(__file__).resolve().parents[1]
        package_script = (repository_root / "scripts" / "package_windows.ps1").read_text(encoding="utf-8")
        build_workflow = (repository_root / ".github" / "workflows" / "build-windows-x64.yml").read_text(encoding="utf-8")
        for content in (package_script, build_workflow):
            self.assertNotIn("prepare_bundled_classifier.py", content)
            self.assertNotIn("--require-bundled-classifier", content)
            self.assertIn("--require-bundled-detector", content)
            self.assertIn("verify_bundled_litert.py", content)
        for workflow_name in ("model-sync-pr.yml", "validation-performance.yml"):
            workflow = (repository_root / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertNotIn("prepare_bundled_classifier.py", workflow)
            self.assertNotIn("bundled-models", workflow)

        gitignore = (repository_root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/res/bundled-models/", gitignore)
        self.assertIn("/res/model/cls/bundled/Neri1.2preview-litert*", gitignore)

    def test_clean_checkout_packages_mdv6_without_repo_shards(self):
        repository_root = Path(__file__).resolve().parents[1]
        from scripts.package_litert_resources import BUILTIN_DETECTOR
        detector_source = repository_root / "res" / BUILTIN_DETECTOR
        with tempfile.TemporaryDirectory() as tmp:
            clean_res = Path(tmp) / "res"
            output = Path(tmp) / "packaged-res"
            (clean_res / "model" / "detect" / "bundled").mkdir(parents=True)
            shutil.copy2(detector_source, clean_res / BUILTIN_DETECTOR)
            for suffix in (".labels.txt", ".LICENSE.md", ".SOURCE.md"):
                sidecar = detector_source.with_name(detector_source.stem + suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, clean_res / "model" / "detect" / "bundled" / sidecar.name)
            report = copy_resources(clean_res, output)
            self.assertEqual([row["path"] for row in report["models"]], [BUILTIN_DETECTOR])
            self.assertEqual(verify_resources(output, require_bundled_detector=True), report)
            self.assertEqual(list((output / "model" / "cls" / "bundled").rglob("*")), [])

    def test_verify_bundled_litert_requires_mdv6_and_rejects_bundled_classifier(self):
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp) / "Neri"
            shutil.copytree(
                repository_root / "system",
                app_root / "system",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for relative in CANONICAL_DIRECTORIES:
                (app_root / "res" / "model" / relative).mkdir(parents=True, exist_ok=True)
            detector = app_root / "res" / "model" / "detect" / "bundled" / "MDV6-yolov10-c.tflite"
            detector.write_bytes(b"fake tflite detector")
            verify_script = repository_root / "scripts" / "verify_bundled_litert.py"
            runner = textwrap.dedent(f"""
                import json, sys, types
                import numpy as np
                fake_mod = types.ModuleType("ai_edge_litert")
                fake_interp_mod = types.ModuleType("ai_edge_litert.interpreter")
                class FakeInterpreter:
                    def __init__(self, model_path, num_threads=2):
                        self.model_path = model_path
                    def allocate_tensors(self):
                        pass
                    def get_input_details(self):
                        return [{{"index": 0, "shape": (1, 640, 640, 3), "dtype": np.float32}}]
                    def set_tensor(self, index, value):
                        pass
                    def invoke(self):
                        pass
                    def get_output_details(self):
                        return [{{"index": 0}}]
                    def get_tensor(self, index):
                        return np.zeros((1, 300, 6), dtype=np.float32)
                fake_interp_mod.Interpreter = FakeInterpreter
                fake_mod.interpreter = fake_interp_mod
                sys.modules["ai_edge_litert"] = fake_mod
                sys.modules["ai_edge_litert.interpreter"] = fake_interp_mod
                import importlib.util
                spec = importlib.util.spec_from_file_location("verify_bundled_litert", {str(verify_script)!r})
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                report = mod.verify({str(app_root)!r})
                assert report["passed"] is True, report
                assert report.get("bundled_classifier_present") is False, report
                assert "classifier_smoke" not in report, report
            """)
            ok = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True, timeout=30)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)

            cls_manifest = app_root / "res" / "model" / "cls" / "bundled" / "Neri1.2preview-litert.neri.json"
            cls_manifest.write_text("{}", encoding="utf-8")
            bad = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0, "verify_bundled_litert must reject bundled classifier assets")
            self.assertIn("ValueError", bad.stderr)


if __name__ == "__main__":
    unittest.main()
