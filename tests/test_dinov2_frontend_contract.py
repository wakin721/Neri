from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_active_flutter_api_client_uses_dinov2_routes_only():
    text = _read("frontend/lib/src/api_client.dart")
    assert "/api/environment/dinov2-status" in text
    assert "/api/environment/install-dinov2" in text
    assert "/api/environment/remove-dinov2" in text
    assert "/api/dinov2/registry" in text
    assert "/api/dinov2/feedback" in text
    assert "/api/dinov3" not in text
    assert "DinoV3" not in text


def test_active_flutter_model_info_identifies_dinov2():
    text = _read("frontend/lib/src/models/settings.dart")
    assert "bool get isDinoV2 => backend == 'dinov2';" in text
    assert "isDinoV3" not in text


def test_dinov2_startup_check_uses_new_component_status():
    text = _read("frontend/lib/src/dinov2_startup_check.dart")
    assert "fetchDinoV2ComponentStatus" in text
    assert "DinoV2StartupCheck" in text
    assert "DINOv3" not in text


def test_registry_dialog_matches_backend_initial_registration_conditions():
    text = _read("frontend/lib/src/widgets/dinov2_registry_dialog.dart")
    assert "_conditionRow('≥4 个独立事件', conditions['events'] == true)" in text
    assert "conditions['cameras']" not in text
    assert "≥5 个独立事件" not in text
