from types import SimpleNamespace

import pytest

from system.backend import services
from system.backend.models import DetectionItem
from system.dinov3.display import model_display_name
from system.dinov3.processor import DinoV3CropCollector, finalize_photo_job


@pytest.mark.parametrize("name", ["Neri1.1", "Neri1.2preview", "自定义物种模型"])
def test_validation_messages_identify_the_selected_model(monkeypatch, name):
    monkeypatch.setattr(services, "_dino_manifest_payload", lambda _: {
        "backend": "dinov3", "display_name": name,
    })
    with pytest.raises(ValueError) as error:
        services._validate_dino_job_options(None, "model.neri.json", "skip")
    assert str(error.value) == f"{name} 分类模型必须同时选择探测模型。"
    with pytest.raises(ValueError) as error:
        services._validate_dino_job_options("detector.pt", "model.neri.json", "all")
    assert str(error.value) == f"{name} 暂不支持完整视频追踪，请使用快速识别或跳过视频。"


@pytest.mark.parametrize("manifest", [None, {}, {"display_name": None}, {"display_name": " "}])
def test_missing_model_identity_uses_neutral_name(manifest):
    assert model_display_name(manifest) == "联合模型"


@pytest.mark.parametrize("name", ["Neri1.1", "Neri1.2preview", None])
def test_empty_sequence_summary_uses_runtime_model_identity(tmp_path, name):
    path = tmp_path / "empty.jpg"
    item = DetectionItem(filename=path.name, path=str(path), file_type="jpg", detection_data={
        "detection_completed": True, "detection_box_count": 0,
        "detection_box_indices": [], "检测框": [],
    })
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", display_name=name),
        _dinov3_collector=DinoV3CropCollector(),
    )
    updates = []
    results = finalize_photo_job(detector, [item], tmp_path, mutate_job=lambda **kw: updates.append(kw))
    assert results[0].detection_data["sequence_completed"] is True
    assert updates[-1]["message"].startswith(f"{name or '联合模型'}序列聚合完成")
