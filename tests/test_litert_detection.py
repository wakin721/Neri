"""LiteRT detection fallback used when PyTorch is not installed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from system.dino.litert_detector import LiteRtDetector, read_model_classes
from system.dino.litert_image_processor import LiteRtImageProcessor
from system.model_sync.catalog import discover_models
from system.model_sync.layout import ModelLayout


class FakeInterpreter:
    def __init__(self, output, *, input_shape=(1, 640, 640, 3)):
        self.output = np.asarray(output, dtype=np.float32)
        self.input_shape = input_shape
        self.input = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [
            {
                "shape": np.asarray(self.input_shape),
                "dtype": np.float32,
                "index": 0,
                "quantization": (0.0, 0),
            }
        ]

    def get_output_details(self):
        return [
            {
                "shape": np.asarray(self.output.shape),
                "dtype": np.float32,
                "index": 1,
                "quantization": (0.0, 0),
            }
        ]

    def set_tensor(self, _index, tensor):
        self.input = tensor

    def invoke(self):
        return None

    def get_tensor(self, _index):
        return self.output


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "detector.tflite"
    path.write_bytes(b"fake-tflite")
    (tmp_path / "detector.labels.txt").write_text(
        "animal,0\nperson,1\nvehicle,2\n", encoding="utf-8"
    )
    return path


def test_sidecar_labels_are_read_without_loading_litert(tmp_path):
    path = _model_file(tmp_path)
    assert read_model_classes(path, class_count=3) == {
        0: "animal",
        1: "person",
        2: "vehicle",
    }


def test_end2end_detector_decodes_normalized_boxes(tmp_path):
    output = np.asarray(
        [
            [
                [0.25, 0.25, 0.75, 0.75, 0.9, 0],
                [0.10, 0.10, 0.20, 0.20, 0.8, 1],
            ]
        ],
        dtype=np.float32,
    )
    detector = LiteRtDetector(
        _model_file(tmp_path),
        interpreter_factory=lambda **_kwargs: FakeInterpreter(output),
    )
    rgb = np.full((100, 100, 3), 127, dtype=np.uint8)
    detections = detector.predict(rgb)

    assert [item.class_id for item in detections] == [0, 1]
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].x1 == pytest.approx(25.0, abs=1e-3)
    assert detections[0].y1 == pytest.approx(25.0, abs=1e-3)
    assert detections[0].x2 == pytest.approx(75.0, abs=1e-3)
    assert detections[0].y2 == pytest.approx(75.0, abs=1e-3)


def test_image_processor_matches_backend_detection_contract(tmp_path):
    output = np.asarray(
        [[[0.25, 0.25, 0.75, 0.75, 0.9, 0], [0.1, 0.1, 0.2, 0.2, 0.8, 1]]],
        dtype=np.float32,
    )
    model = _model_file(tmp_path)
    image = tmp_path / "animal.png"
    import cv2

    cv2.imwrite(str(image), np.full((100, 100, 3), 127, dtype=np.uint8))
    processor = LiteRtImageProcessor(
        str(model),
        interpreter_factory=lambda **_kwargs: FakeInterpreter(output),
    )
    result = processor.detect_batch_species([str(image)], conf=0.5, iou=0.5)

    assert result[0]["物种名称"] == "动物,人"
    assert result[0]["物种数量"] == "1,1"
    detection = result[0]["detect_results"][0]
    assert len(detection.boxes) == 2
    assert detection.names[0] == "animal"
    assert detection.boxes.cls.tolist() == [0, 1]
    assert detection.boxes[0].xyxy.tolist()[0][2] == pytest.approx(75.0, abs=1e-3)


def test_catalog_exposes_tflite_detector_as_litert(tmp_path):
    detect_user = tmp_path / "detect" / "user"
    detect_sync = tmp_path / "detect" / "sync"
    cls_user = tmp_path / "cls" / "user"
    cls_sync = tmp_path / "cls" / "sync"
    for directory in (detect_user, detect_sync, cls_user, cls_sync):
        directory.mkdir(parents=True)
    (detect_sync / "camera.tflite").write_bytes(b"fake")
    layout = ModelLayout(
        resource_root=tmp_path,
        root=tmp_path / "model",
        detect_user=detect_user,
        detect_sync=detect_sync,
        cls_user=cls_user,
        cls_sync=cls_sync,
        dinov3_root=tmp_path / "DINOv3",
        tracker=tmp_path / "tracker.yaml",
        state_file=tmp_path / ".sync-state.json",
    )

    models = discover_models(layout, "detect")
    assert [model.backend for model in models] == ["litert"]
    assert models[0].supports_video_fast is True
    assert models[0].supports_video_all is False


def test_backend_loads_litert_model_and_rejects_pt_without_torch(
    tmp_path, monkeypatch
):
    from system.backend import services

    tflite = tmp_path / "detector.tflite"
    tflite.write_bytes(b"fake")
    created = []

    class FakeProcessor:
        def __init__(self, model_path):
            created.append(model_path)

    monkeypatch.setattr(
        "system.dino.litert_image_processor.LiteRtImageProcessor",
        FakeProcessor,
    )
    detector = services._load_detector(str(tflite))
    assert isinstance(detector, FakeProcessor)
    assert created == [str(tflite)]

    pt = tmp_path / "detector.pt"
    pt.write_bytes(b"fake")
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else original(name),
    )
    with pytest.raises(RuntimeError, match="PyTorch"):
        services._load_detector(str(pt))


def test_settings_marks_litert_detection_ready_without_torch(monkeypatch):
    from system.backend import main_core_legacy as core
    from system.backend.models import ModelInfo

    model = ModelInfo(
        name="camera.tflite",
        path="/models/camera.tflite",
        kind="detect",
        backend="litert",
        supports_video_fast=True,
        supports_video_all=False,
    )
    monkeypatch.setattr(core, "list_available_models", lambda: [model])
    monkeypatch.setattr(core, "list_available_classification_models", lambda: [])
    monkeypatch.setattr(core, "missing_yolo_dependencies", lambda: ["torch"])
    monkeypatch.setattr(core, "detect_gpu_available", lambda: False)
    monkeypatch.setattr(core.settings_manager, "load_settings", lambda: {})
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            None
            if name in {"torch", "torchvision", "ultralytics"}
            else object()
            if name in {"ai_edge_litert", "cv2", "numpy"}
            else original(name)
        ),
    )

    response = core.settings()
    assert response.pytorch_available is False
    assert response.litert_available is True
    assert response.detection_ready is True
    assert response.selected_model == "/models/camera.tflite"
    assert response.selected_classification_model is None
