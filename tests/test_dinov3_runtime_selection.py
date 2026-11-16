"""Tests for DINOv3 CPU/GPU runtime selection, download protocol, and maintenance."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from server.model_distribution.app import create_app
from server.model_distribution.config import DistributionConfig
from server.model_distribution.storage import UpstreamLink
from system.dinov3.deployment import publish_model
from system.dinov3.encoder import PREPROCESSING, sha256
from system.dinov3.runtime import read_manifest, resolve_manifest_asset
from tests.test_dinov3_box_explanation_and_feedback import _write_image
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_experiment import experiment
from tests.test_dinov3_learning import stage
from tests.test_dinov3_preview_package import prepare
from tests.test_model_distribution import FakeStore
from scripts.package_dinov3_preview import package


class _LoopbackOpener:
    def __init__(self, client):
        self.client = client

    def __call__(self, request, timeout):
        response = self.client.request(
            request.method,
            request.full_url,
            content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status_code,
                response.text,
                response.headers,
                None,
            )
        stream = io.BytesIO(response.content)
        stream.status = response.status_code
        stream.headers = response.headers
        return stream


def _make_litert_component_dir(experiment_fixture, target_dir: Path, *, shared_learning: bool = True) -> Path:
    original, _model, _raw, cal = experiment_fixture
    src = read_manifest(original)
    if src.get("inference_recipe") != "experiment_full":
        original = prepare(experiment_fixture)
        src = read_manifest(original)
    target_dir.mkdir(parents=True, exist_ok=True)
    assets = target_dir / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(resolve_manifest_asset(original, src["checkpoint"]), assets / "model.neri-model")
    if "calibration" in src:
        shutil.copy2(resolve_manifest_asset(original, src["calibration"]), assets / "calibration.npz")
    else:
        np.savez(assets / "calibration.npz", **cal)
    calibration_digest = sha256(assets / "calibration.npz")
    tflite_bytes = b"fake-litert-cpu-encoder-fp32"
    (assets / "encoder.tflite").write_bytes(tflite_bytes)
    payload = {
        "schema_version": 1,
        "backend": "dinov3",
        "architecture": "dinov3_vitb16",
        "encoder_backend": "litert",
        "encoder_weights": "assets/encoder.tflite",
        "encoder_litert_sha256": hashlib.sha256(tflite_bytes).hexdigest(),
        "encoder_sha256": src["encoder_sha256"],
        "checkpoint": "assets/model.neri-model",
        "checkpoint_sha256": src["checkpoint_sha256"],
        "calibration": "assets/calibration.npz",
        "calibration_sha256": calibration_digest,
        "head_type": src["head_type"],
        "inference_recipe": src["inference_recipe"],
        "feature_dim": 768,
        "preprocessing": PREPROCESSING,
        "event_aggregation": "complete_sequence",
        "requires_detector": True,
        "supports_video_fast": False,
        "supports_video_all": False,
        "display_name": "Neri1.2preview（LiteRT FP32 CPU）",
    }
    if shared_learning:
        payload["shared_learning"] = True
    manifest = target_dir / "Neri1.2preview.neri.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _make_gpu_component_dir(experiment_fixture, tmp_path: Path) -> Path:
    manifest = prepare(experiment_fixture)
    output = tmp_path / "preview.zip"
    package(manifest, output)
    gpu_dir = tmp_path / "gpu_published"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(gpu_dir)
    return gpu_dir / "Neri1.2preview" / "Neri1.2preview.neri.json"


def test_validate_runtime_choice_is_strict_three_values():
    from system.dinov3.runtime_selection import validate_runtime_choice

    assert validate_runtime_choice("auto") == "auto"
    assert validate_runtime_choice("cpu") == "cpu"
    assert validate_runtime_choice("gpu") == "gpu"
    for bad in ("Auto", "CPU", " cuda", "cpu ", "", None, 1, True):
        with pytest.raises(ValueError):
            validate_runtime_choice(bad)


def test_auto_runtime_probes_cuda_and_xpu_without_torch(monkeypatch):
    from system.backend import maintenance
    from system.dinov3 import runtime_selection

    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    monkeypatch.setattr(maintenance, "_detect_nvidia_cuda_pytorch_index", lambda: ("CUDA 12.8", "https://download.pytorch.org/whl/cu128"))
    monkeypatch.setattr(maintenance, "_detect_intel_xpu_pytorch_index", lambda: None)
    assert runtime_selection.resolve_runtime_choice("auto") == "gpu"

    monkeypatch.setattr(maintenance, "_detect_nvidia_cuda_pytorch_index", lambda: ("CPU Only (CUDA 10.2 过旧)", maintenance.PYTORCH_CPU_INDEX_URL))
    monkeypatch.setattr(maintenance, "XPU_ENABLED", False)
    assert runtime_selection.resolve_runtime_choice("auto") == "cpu"

    monkeypatch.setattr(maintenance, "_detect_nvidia_cuda_pytorch_index", lambda: None)
    monkeypatch.setattr(maintenance, "XPU_ENABLED", True)
    monkeypatch.setattr(maintenance, "_detect_intel_xpu_pytorch_index", lambda: ("Intel XPU", maintenance.PYTORCH_XPU_INDEX_URL))
    assert runtime_selection.resolve_runtime_choice("auto") == "gpu"

    monkeypatch.setattr(maintenance, "_detect_intel_xpu_pytorch_index", lambda: ("CPU Only", maintenance.PYTORCH_CPU_INDEX_URL))
    assert runtime_selection.resolve_runtime_choice("auto") == "cpu"


def test_explicit_cpu_and_gpu_skip_hardware_probe(monkeypatch):
    from system.dinov3 import runtime_selection

    monkeypatch.setattr(runtime_selection, "detect_supported_gpu", lambda: pytest.fail("should not probe"))
    assert runtime_selection.resolve_runtime_choice("cpu") == "cpu"
    assert runtime_selection.resolve_runtime_choice("gpu") == "gpu"


def test_distribution_client_cpu_and_gpu_manifest_protocol():
    from system.dinov3.distribution import DinoV3DistributionClient, DinoV3DistributionError

    seen_urls = []
    payloads = {
        "/v1/dinov3/manifest": {
            "schema_version": 1,
            "manifest_id": "a" * 64,
            "files": [{"path": "DINOv3/Neri1.2preview.neri.json", "size": 2, "sha256": "b" * 64}],
        },
        "/v1/dinov3/manifest?runtime=cpu": {
            "schema_version": 1,
            "manifest_id": "c" * 64,
            "files": [{"path": "DINOv3/cpu/Neri1.2preview-litert.neri.json", "size": 3, "sha256": "d" * 64}],
        },
    }

    def opener(request, timeout):
        seen_urls.append(request.full_url)
        for suffix, body in payloads.items():
            if request.full_url.endswith(suffix):
                return io.BytesIO(json.dumps(body).encode("utf-8"))
        raise AssertionError(f"unexpected url: {request.full_url}")

    client = DinoV3DistributionClient("http://127.0.0.1/v1", opener=opener)
    gpu_manifest = client.fetch_manifest()
    assert gpu_manifest.files[0].path == "DINOv3/Neri1.2preview.neri.json"
    assert gpu_manifest.files[0].relative_path == "Neri1.2preview.neri.json"

    gpu_explicit = client.fetch_manifest(runtime="gpu")
    assert gpu_explicit.files[0].relative_path == "Neri1.2preview.neri.json"

    cpu_manifest = client.fetch_manifest(runtime="cpu")
    assert cpu_manifest.files[0].path == "DINOv3/cpu/Neri1.2preview-litert.neri.json"
    assert cpu_manifest.files[0].relative_path == "Neri1.2preview-litert.neri.json"
    assert seen_urls == [
        "http://127.0.0.1/v1/dinov3/manifest",
        "http://127.0.0.1/v1/dinov3/manifest",
        "http://127.0.0.1/v1/dinov3/manifest?runtime=cpu",
    ]

    payloads["/v1/dinov3/manifest?runtime=cpu"]["files"] = [
        {"path": "DINOv3/Neri1.2preview.neri.json", "size": 2, "sha256": "b" * 64}
    ]
    with pytest.raises(DinoV3DistributionError, match="invalid_dinov3_distribution_path"):
        client.fetch_manifest(runtime="cpu")


def test_dual_runtime_cloud_install_and_status(experiment, tmp_path):
    from system.dinov3.component import dinov3_component_status, install_dinov3_component
    from system.dinov3.distribution import DinoV3DistributionClient

    gpu_manifest = _make_gpu_component_dir(experiment, tmp_path)
    cpu_dir = tmp_path / "cpu_source"
    _make_litert_component_dir(experiment, cpu_dir)

    old = FakeStore({})
    config = DistributionConfig(
        state_dir=tmp_path / "state",
        openlist_token="test",
        requests_per_minute=1200,
        dinov3_root=gpu_manifest.parent,
        dinov3_cpu_root=cpu_dir,
    )
    with patch("server.model_distribution.app.OpenListModelStore", return_value=old):
        app = create_app(config=config)
    with TestClient(app) as http:
        client = DinoV3DistributionClient("http://127.0.0.1/v1", opener=_LoopbackOpener(http))
        cpu_root = tmp_path / "installed_cpu"
        cpu_status = install_dinov3_component(root=cpu_root, distribution_client=client, runtime_choice="cpu")
        assert cpu_status["installed"] and cpu_status["healthy"]
        assert cpu_status["encoder_backend"] == "litert"
        assert cpu_status["runtime_choice"] == "cpu"
        assert cpu_status["resolved_runtime_choice"] == "cpu"
        assert (cpu_root / "Neri1.2preview.neri.json").is_file()
        assert not (cpu_root / "cpu").exists()

        gpu_root = tmp_path / "installed_gpu"
        gpu_status = install_dinov3_component(root=gpu_root, distribution_client=client, runtime_choice="gpu")
        assert gpu_status["installed"] and gpu_status["healthy"]
        assert gpu_status["encoder_backend"] == "pytorch"
        assert gpu_status["runtime_choice"] == "gpu"
        assert gpu_status["resolved_runtime_choice"] == "gpu"

        # Tampering LiteRT weights makes component unhealthy.
        (cpu_root / "assets" / "encoder.tflite").write_bytes(b"corrupt")
        broken = dinov3_component_status(root=cpu_root)
        assert broken["installed"] and not broken["healthy"]
        assert broken["encoder_backend"] is None


def test_seed_dir_litert_and_shared_learning_survive_runtime_switch(experiment, tmp_path, monkeypatch):
    from system.backend import dinov3_learning as learning
    from system.dinov3.component import (
        dinov3_component_status,
        install_dinov3_component,
        remove_dinov3_component,
    )
    from system.dinov3 import shared_learning as shared

    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "learning_state"))
    gpu_manifest = _make_gpu_component_dir(experiment, tmp_path)
    gpu_payload = read_manifest(gpu_manifest)
    gpu_payload["shared_learning"] = True
    gpu_manifest.write_text(json.dumps(gpu_payload), encoding="utf-8")

    cpu_dir = tmp_path / "cpu_seed"
    _make_litert_component_dir(experiment, cpu_dir, shared_learning=True)

    component_root = tmp_path / "res" / "model" / "DINOv3"
    install_dinov3_component(root=component_root, seed_dir=gpu_manifest.parent)
    gpu_status = dinov3_component_status(root=component_root)
    assert gpu_status["healthy"] and gpu_status["encoder_backend"] == "pytorch"
    assert gpu_status["resolved_runtime_choice"] == "gpu"

    installed_gpu = component_root / gpu_status["manifest"]
    photo = tmp_path / "bird.jpg"
    _write_image(photo, 120)

    def fake_encoder_factory(manifest_path, payload, **kwargs):
        def encode(crops, **_kw):
            out = np.zeros((len(crops), 768), dtype=np.float32)
            out[:, 2] = 1.0
            return out
        return SimpleNamespace(encode=encode, device="cpu", use_fp16=False)

    monkeypatch.setattr("system.dinov3.runtime.load_manifest_encoder", fake_encoder_factory)

    with threadpool_limits(limits=1):
        stage(installed_gpu, photo, "C", "gpu-confirm")
        learning.update_model(installed_gpu)
        family_id = shared.family_id(installed_gpu)
        assert learning.learning_status(installed_gpu)["learned_count"] == 1

        # Switch component from GPU PyTorch to CPU LiteRT in-place.
        install_dinov3_component(root=component_root, seed_dir=cpu_dir, runtime_choice="cpu")
        cpu_status = dinov3_component_status(root=component_root)
        assert cpu_status["healthy"] and cpu_status["encoder_backend"] == "litert"
        assert cpu_status["runtime_choice"] == "cpu"
        assert cpu_status["resolved_runtime_choice"] == "cpu"
        assert not (component_root / "encoder-code").exists()

        installed_cpu = component_root / cpu_status["manifest"]
        assert shared.family_id(installed_cpu) == family_id
        assert learning.learning_status(installed_cpu)["learned_count"] == 1
        photo.unlink()
        from system.dinov3.runtime import load_dinov3_model
        assert "C" in load_dinov3_model(installed_cpu).classifier.classes

        # Removing the component preserves user learning archives.
        remove_dinov3_component(root=component_root)
        assert shared.family_manifest(learning._state_root(), family_id).is_file()
        assert (learning._state_root() / "dinov3_learning.sqlite3").is_file()


def test_environment_api_runtime_choice_contract(monkeypatch):
    from system.backend import dinov3_environment as env_api
    from system.backend.main import app

    calls = []

    def fake_start(env_choice="自动检测", package_source="auto", runtime_choice="auto"):
        calls.append((env_choice, package_source, runtime_choice))
        return {
            "operation": "install_dinov3",
            "state": "starting",
            "message": "ok",
            "progress": 0,
            "runtime_choice": runtime_choice,
        }

    monkeypatch.setattr(env_api, "start_dinov3_install", fake_start)
    monkeypatch.setattr(env_api, "schedule_backend_shutdown", lambda: None)
    monkeypatch.setattr(
        env_api,
        "dinov3_component_status",
        lambda: {
            "installed": True,
            "healthy": True,
            "architecture": "DINOv3 ViT-B/16",
            "component_version": 1,
            "display_name": "Neri1.2preview",
            "class_count": 2,
            "feature_dim": 768,
            "manifest": "Neri1.2preview-litert.neri.json",
            "message": "ok",
            "encoder_backend": "litert",
            "runtime_choice": "auto",
            "resolved_runtime_choice": "cpu",
        },
    )

    with TestClient(app) as client:
        status_res = client.get("/api/environment/dinov3-status")
        assert status_res.status_code == 200
        assert status_res.json()["encoder_backend"] == "litert"
        assert status_res.json()["runtime_choice"] == "auto"
        assert status_res.json()["resolved_runtime_choice"] == "cpu"

        default_res = client.post("/api/environment/install-dinov3", json={})
        assert default_res.status_code == 202
        cpu_res = client.post("/api/environment/install-dinov3", json={"runtime_choice": "cpu"})
        assert cpu_res.status_code == 202
        gpu_res = client.post("/api/environment/install-dinov3", json={"runtime_choice": "gpu"})
        assert gpu_res.status_code == 202

        for bad in ("cuda", "CPU", "", None):
            bad_res = client.post("/api/environment/install-dinov3", json={"runtime_choice": bad})
            assert bad_res.status_code == 422

    assert calls == [
        ("自动检测", "auto", "auto"),
        ("自动检测", "auto", "cpu"),
        ("自动检测", "auto", "gpu"),
    ]


def test_maintenance_cpu_skips_torch_and_passes_cli_runtime_choice(tmp_path, monkeypatch):
    from system.backend import dinov3_maintenance as worker
    from system.backend import maintenance

    monkeypatch.setattr(maintenance, "maintenance_status_path", lambda: tmp_path / "status.json")
    monkeypatch.setattr(maintenance, "maintenance_log_path", lambda: tmp_path / "maintenance.log")
    monkeypatch.setattr(maintenance, "read_maintenance_status", lambda: {})
    monkeypatch.setattr(maintenance, "_wait_for_process_exit", lambda _: None)
    monkeypatch.setattr(maintenance, "_wait_for_port_release", lambda: None)
    monkeypatch.setattr(maintenance, "_start_backend", lambda: None)
    monkeypatch.setattr(maintenance, "toolkit_python", lambda: Path(sys.executable))
    monkeypatch.setattr(maintenance, "resolve_package_source", lambda _: ("official", "official", "https://pypi.org/simple"))

    import sys

    popen_cmds = []
    monkeypatch.setattr(worker.subprocess, "Popen", lambda cmd, **kw: popen_cmds.append(cmd) or SimpleNamespace(pid=123))
    worker.start_dinov3_install("自动检测", "official", "cpu")
    assert "--runtime-choice" in popen_cmds[0]
    idx = popen_cmds[0].index("--runtime-choice")
    assert popen_cmds[0][idx + 1] == "cpu"

    with pytest.raises(ValueError):
        worker.start_dinov3_install(runtime_choice="cuda")

    torch_calls = []
    install_calls = []
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: None if name == "torch" else object())
    monkeypatch.setattr(maintenance, "_run_pytorch_install", lambda *a, **k: torch_calls.append((a, k)))
    monkeypatch.setattr(worker, "_install_dino_python_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(worker, "install_dinov3_component", lambda **kw: install_calls.append(kw) or {"healthy": True})

    assert worker._run(["--operation", "install_dinov3", "--backend-pid", "0", "--runtime-choice", "cpu"]) == 0
    assert torch_calls == []
    assert install_calls[-1]["runtime_choice"] == "cpu"
    assert install_calls[-1]["resolved_runtime_choice"] == "cpu"

    # Auto mode without GPU skips torch; with GPU installs torch.
    monkeypatch.setattr("system.dinov3.runtime_selection.detect_supported_gpu", lambda: False)
    assert worker._run(["--operation", "install_dinov3", "--backend-pid", "0", "--runtime-choice", "auto"]) == 0
    assert torch_calls == []
    assert install_calls[-1]["runtime_choice"] == "auto"
    assert install_calls[-1]["resolved_runtime_choice"] == "cpu"

    monkeypatch.setattr("system.dinov3.runtime_selection.detect_supported_gpu", lambda: True)
    assert worker._run(["--operation", "install_dinov3", "--backend-pid", "0", "--runtime-choice", "auto"]) == 0
    assert len(torch_calls) == 1
    assert install_calls[-1]["runtime_choice"] == "auto"
    assert install_calls[-1]["resolved_runtime_choice"] == "gpu"

    with pytest.raises(SystemExit):
        worker._run(["--operation", "install_dinov3", "--backend-pid", "0", "--runtime-choice", "invalid"])


def test_auto_gpu_repairs_cpu_only_torch_and_legacy_cpu_env_choice(tmp_path, monkeypatch):
    from system.backend import dinov3_maintenance as worker
    from system.backend import maintenance

    monkeypatch.setattr(maintenance, "project_root", lambda: tmp_path)
    monkeypatch.setattr(maintenance, "maintenance_status_path", lambda: tmp_path / "status.json")
    monkeypatch.setattr(maintenance, "maintenance_log_path", lambda: tmp_path / "maintenance.log")
    monkeypatch.setattr(maintenance, "_wait_for_process_exit", lambda _: None)
    monkeypatch.setattr(maintenance, "_wait_for_port_release", lambda: None)
    monkeypatch.setattr(maintenance, "_start_backend", lambda: None)
    monkeypatch.setattr(maintenance, "XPU_ENABLED", False)
    monkeypatch.setattr(
        maintenance,
        "_detect_nvidia_cuda_pytorch_index",
        lambda: ("CUDA 12.8", "https://download.pytorch.org/whl/cu128"),
    )
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(worker, "_installed_torch_supports_gpu", lambda: False)

    pytorch_installs = []
    monkeypatch.setattr(
        maintenance,
        "_run_pytorch_install",
        lambda env_choice, pkg_src, **kw: pytorch_installs.append((env_choice, pkg_src)),
    )
    monkeypatch.setattr(worker, "_install_dino_python_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(
        worker,
        "install_dinov3_component",
        lambda **kw: {"healthy": True, "manifest": "Neri1.2preview.neri.json"},
    )

    rc = worker._run(
        [
            "--operation",
            "install_dinov3",
            "--backend-pid",
            "0",
            "--env-choice",
            "CPU Only",
            "--runtime-choice",
            "auto",
        ]
    )
    assert rc == 0
    assert pytorch_installs == [("自动检测", "auto")]


def test_gpu_repair_probe_exits_before_install_without_loading_parent_torch(monkeypatch):
    import builtins
    import sys
    from system.backend import dinov3_maintenance as worker, maintenance

    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.setattr(worker.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(maintenance, 'toolkit_python', lambda: Path('toolkit/python.exe'))
    monkeypatch.setattr(maintenance, 'resolve_pytorch_install_plan',
                        lambda _: {'index_url': 'https://download.pytorch.org/whl/cu128'})
    real_import = builtins.__import__
    parent_imports = []
    events = []

    def import_guard(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            parent_imports.append(name)
            raise AssertionError('Torch DLLs must not be loaded by the maintenance process')
        return real_import(name, *args, **kwargs)

    def child_probe(command, **kwargs):
        assert Path(command[0]) == Path('toolkit/python.exe')
        assert '-c' in command and 'import torch' in command[-1]
        assert kwargs['timeout'] > 0
        events.extend(['probe-started', 'probe-exited'])
        return SimpleNamespace(returncode=0, stdout='{"gpu_available":false}')

    def install(*args, **kwargs):
        assert events == ['probe-started', 'probe-exited']
        events.append('install')

    monkeypatch.setattr(builtins, '__import__', import_guard)
    monkeypatch.setattr(worker.subprocess, 'run', child_probe)
    monkeypatch.setattr(maintenance, '_run_pytorch_install', install)
    monkeypatch.setattr(maintenance, '_write_status', lambda *_: None)
    worker._ensure_gpu_pytorch_environment('自动检测', 'auto')
    assert parent_imports == []
    assert events == ['probe-started', 'probe-exited', 'install']


@pytest.mark.parametrize('returncode,stdout,expected', [
    (0, '{"gpu_available":true}', True),
    (0, '{"gpu_available":false}', False),
    (1, '', False),
    (0, 'invalid output', False),
])
def test_gpu_probe_child_result_is_checked(monkeypatch, returncode, stdout, expected):
    from system.backend import dinov3_maintenance as worker
    monkeypatch.setattr(worker.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=returncode, stdout=stdout))
    assert worker._installed_torch_supports_gpu() is expected


def test_maintenance_migrates_only_same_family_selected_classification_model(
    experiment, tmp_path, monkeypatch
):
    from system.backend import dinov3_maintenance as worker
    from system.backend import maintenance
    from system.dinov3.model import DinoV3RidgeWithinModel
    from system.dinov3.shared_learning import PREVIEW_BASE, PREVIEW_ENCODER

    original, _model, _raw, _cal = experiment
    monkeypatch.setattr(maintenance, "project_root", lambda: tmp_path)
    monkeypatch.setattr(maintenance, "maintenance_status_path", lambda: tmp_path / "status.json")
    monkeypatch.setattr(maintenance, "maintenance_log_path", lambda: tmp_path / "maintenance.log")
    monkeypatch.setattr(maintenance, "_wait_for_process_exit", lambda _: None)
    monkeypatch.setattr(maintenance, "_wait_for_port_release", lambda: None)
    monkeypatch.setattr(maintenance, "_start_backend", lambda: None)
    monkeypatch.setattr(worker, "_install_dino_python_dependencies", lambda *a, **k: None)

    dinov3_root = tmp_path / "res" / "model" / "DINOv3"
    monkeypatch.setattr(worker, "_component_root", lambda: dinov3_root)
    monkeypatch.setattr("system.dinov3.component._component_root", lambda root=None: dinov3_root if root is None else Path(root).resolve())

    cpu_seed = tmp_path / "cpu_seed"
    new_manifest = _make_litert_component_dir(experiment, cpu_seed, shared_learning=True)
    orig_payload = json.loads(new_manifest.read_text(encoding="utf-8"))

    old_bundled_dir = tmp_path / "res" / "model" / "cls" / "bundled"
    old_bundled_manifest = _make_litert_component_dir(experiment, old_bundled_dir, shared_learning=True)
    old_bundled_renamed = old_bundled_dir / "Neri1.2preview-litert.neri.json"
    old_bundled_manifest.replace(old_bundled_renamed)

    # Other family model must NEVER be overwritten.
    other_dir = tmp_path / "res" / "model" / "cls" / "user"
    other_dir.mkdir(parents=True)
    other_model = DinoV3RidgeWithinModel.load(resolve_manifest_asset(new_manifest, orig_payload["checkpoint"]))
    other_model.metadata["other_family"] = True
    other_model.save(other_dir / "other.neri-model")
    other_payload = dict(
        json.loads(new_manifest.read_text(encoding="utf-8")),
        checkpoint="other.neri-model",
        checkpoint_sha256=sha256(other_dir / "other.neri-model"),
        calibration="calibration.npz",
    )
    shutil.copy2(cpu_seed / "assets" / "calibration.npz", other_dir / "calibration.npz")
    shutil.copy2(cpu_seed / "assets" / "encoder.tflite", other_dir / "encoder.tflite")
    other_payload["encoder_weights"] = "encoder.tflite"
    other_payload["calibration_sha256"] = sha256(other_dir / "calibration.npz")
    other_manifest = other_dir / "other.neri.json"
    other_manifest.write_text(json.dumps(other_payload), encoding="utf-8")

    def run_install():
        from system.dinov3.component import install_dinov3_component

        monkeypatch.setattr(
            worker,
            "install_dinov3_component",
            lambda **kw: install_dinov3_component(root=dinov3_root, seed_dir=cpu_seed, runtime_choice="cpu"),
        )
        assert worker._run(["--operation", "install_dinov3", "--backend-pid", "0", "--runtime-choice", "cpu"]) == 0

    settings_path = tmp_path / "temp" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Existing old bundled manifest of same family -> migrates to DINOv3/Neri1.2preview.neri.json
    settings_path.write_text(json.dumps({"selected_classification_model": str(old_bundled_renamed)}), encoding="utf-8")
    run_install()
    updated = json.loads(settings_path.read_text(encoding="utf-8"))
    assert Path(updated["selected_classification_model"]).resolve() == (dinov3_root / "Neri1.2preview.neri.json").resolve()

    # 2. Different-family model -> preserved untouched
    settings_path.write_text(json.dumps({"selected_classification_model": str(other_manifest)}), encoding="utf-8")
    run_install()
    preserved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert preserved["selected_classification_model"] == str(other_manifest)

    # 3. Legacy missing bundled preview path -> migrates only when preview family matches
    old_bundled_renamed.unlink()
    settings_path.write_text(json.dumps({"selected_classification_model": str(old_bundled_renamed)}), encoding="utf-8")
    monkeypatch.setattr("system.dinov3.shared_learning.PREVIEW_ENCODER", orig_payload["encoder_sha256"])
    monkeypatch.setattr("system.dinov3.shared_learning.PREVIEW_BASE", orig_payload["checkpoint_sha256"])
    run_install()
    migrated_missing = json.loads(settings_path.read_text(encoding="utf-8"))
    assert Path(migrated_missing["selected_classification_model"]).resolve() == (dinov3_root / "Neri1.2preview.neri.json").resolve()
    assert PREVIEW_ENCODER and PREVIEW_BASE


def test_cpu_runtime_installs_missing_ai_edge_litert_without_torch_and_gpu_skips_it(
    tmp_path, monkeypatch
):
    from system.backend import dinov3_maintenance as worker
    from system.backend import maintenance

    monkeypatch.setattr(maintenance, "project_root", lambda: tmp_path)
    monkeypatch.setattr(maintenance, "maintenance_status_path", lambda: tmp_path / "status.json")
    monkeypatch.setattr(maintenance, "maintenance_log_path", lambda: tmp_path / "maintenance.log")
    monkeypatch.setattr(maintenance, "toolkit_python", lambda: Path(__import__("sys").executable))
    monkeypatch.setattr(
        maintenance,
        "resolve_package_source",
        lambda _src: ("tsinghua", "清华源", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    )
    monkeypatch.setattr(maintenance, "_wait_for_process_exit", lambda _: None)
    monkeypatch.setattr(maintenance, "_wait_for_port_release", lambda: None)
    monkeypatch.setattr(maintenance, "_start_backend", lambda: None)
    monkeypatch.setattr(
        worker.importlib.util,
        "find_spec",
        lambda name: None if name in {"ai_edge_litert", "torch"} else object(),
    )

    pip_commands = []
    monkeypatch.setattr(
        maintenance,
        "_run_commands",
        lambda commands, **kw: pip_commands.extend(commands),
    )
    pytorch_installs = []
    monkeypatch.setattr(
        maintenance,
        "_run_pytorch_install",
        lambda *a, **k: pytorch_installs.append((a, k)),
    )
    monkeypatch.setattr(
        worker,
        "install_dinov3_component",
        lambda **kw: {"healthy": True, "manifest": "Neri1.2preview.neri.json"},
    )

    # CPU branch installs missing ai-edge-litert via configured pip source and skips torch.
    assert (
        worker._run(
            [
                "--operation",
                "install_dinov3",
                "--backend-pid",
                "0",
                "--package-source",
                "tsinghua",
                "--runtime-choice",
                "cpu",
            ]
        )
        == 0
    )
    assert pytorch_installs == []
    assert len(pip_commands) == 1
    assert "ai-edge-litert==2.3.0" in pip_commands[0]
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in pip_commands[0]

    # GPU branch does not force-install ai-edge-litert when missing.
    pip_commands.clear()
    assert (
        worker._run(
            [
                "--operation",
                "install_dinov3",
                "--backend-pid",
                "0",
                "--package-source",
                "tsinghua",
                "--runtime-choice",
                "gpu",
            ]
        )
        == 0
    )
    assert len(pytorch_installs) == 1
    assert pip_commands == []
