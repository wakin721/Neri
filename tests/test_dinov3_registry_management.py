from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from system.backend import dinov3_registry_service
from system.dinov3.registry import RegistryEntryNotFound, SpeciesRegistry


FP = "c" * 64
BASE = datetime(2026, 9, 10, 8, 0, 0)


def _vector() -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[0] = 1.0
    return value


def test_registry_delete_cascades_events_and_prototypes(tmp_path: Path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-a",
        captured_at=BASE,
        source_path=str(tmp_path / "frame.jpg"),
    )
    assert hasattr(registry, "delete"), "SpeciesRegistry.delete is required"
    registry.delete(entry.id)

    with pytest.raises(RegistryEntryNotFound):
        registry.get(entry.id)
    assert registry._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert registry._conn.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0] == 0
    registry.close()


def test_registry_event_keeps_crop_metadata_and_renders_square_example(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[10:70, 20:60] = (10, 80, 220)
    assert cv2.imwrite(str(image_path), image)

    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-a",
        captured_at=BASE,
        source_path=str(image_path),
        bbox=(20.0, 10.0, 60.0, 70.0),
        frame_index=None,
        timestamp_seconds=None,
    )
    event = registry.list_events(entry.id)[0]

    assert event["id"] > 0
    assert event["bbox"] == [20.0, 10.0, 60.0, 70.0]
    assert event["has_example"] is True
    assert hasattr(
        dinov3_registry_service, "render_registry_example"
    ), "render_registry_example is required"

    encoded = dinov3_registry_service.render_registry_example(
        registry, entry.id, event["id"]
    )
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[0] == decoded.shape[1]
    assert decoded.shape[0] == 320
    registry.close()


def test_registry_video_event_renders_requested_frame_as_square_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    frame = np.zeros((90, 140, 3), dtype=np.uint8)
    frame[10:60, 30:70] = (20, 120, 240)
    calls: list[tuple[int, float]] = []

    class FakeCapture:
        def __init__(self, path: str) -> None:
            assert Path(path) == video_path.resolve()

        def set(self, prop: int, value: float) -> bool:
            calls.append((prop, value))
            return True

        def read(self):
            return True, frame.copy()

        def release(self) -> None:
            pass

    monkeypatch.setattr(dinov3_registry_service.cv2, "VideoCapture", FakeCapture)

    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-a",
        captured_at=BASE,
        source_path=str(video_path),
        bbox=(30.0, 10.0, 70.0, 60.0),
        frame_index=17,
        timestamp_seconds=0.68,
    )
    event = registry.list_events(entry.id)[0]

    encoded = dinov3_registry_service.render_registry_example(
        registry, entry.id, event["id"]
    )
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape[0] == decoded.shape[1]
    assert (cv2.CAP_PROP_POS_FRAMES, 17.0) in calls
    registry.close()


def test_registry_example_is_unavailable_for_legacy_event_without_bbox(tmp_path: Path) -> None:
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=FP)
    entry = registry.record_unknown(
        _vector(),
        camera_id="cam-a",
        captured_at=BASE,
        source_path=str(tmp_path / "missing.jpg"),
    )
    event = registry.list_events(entry.id)[0]
    assert event.get("has_example") is False
    assert hasattr(
        dinov3_registry_service, "render_registry_example"
    ), "render_registry_example is required"
    with pytest.raises(FileNotFoundError):
        dinov3_registry_service.render_registry_example(registry, entry.id, event["id"])
    registry.close()
