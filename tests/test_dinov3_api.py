from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def _write_model_bundle(tmp_path: Path) -> Path:
    encoder = tmp_path / "encoder.pth"
    encoder.write_bytes(b"encoder")
    encoder_hash = hashlib.sha256(encoder.read_bytes()).hexdigest()
    checkpoint = tmp_path / "head.pt"
    payload = {
        "backbone": "dinov3_vitb16",
        "feature_dim": 768,
        "classes": ["species-a", "species-b"],
        "head_state": {
            "weight": torch.zeros((2, 768)),
            "bias": torch.zeros(2),
        },
        "prototypes": torch.stack(
            [
                torch.nn.functional.one_hot(torch.tensor(0), 768),
                torch.nn.functional.one_hot(torch.tensor(1), 768),
            ]
        ).float(),
        # Keep this generic API fixture outside the formal duplicate guard.
        "threshold": 1.1,
        "encoder_weights": "encoder.pth",
        "encoder_sha256": encoder_hash,
        "preprocessing": "letterbox224_imagenet",
        "event_aggregation": "mean_l2_normalized_crop_embeddings",
    }
    torch.save(payload, checkpoint)
    manifest = tmp_path / "head.neri.json"
    manifest.write_text(
        '{"backend":"dinov3","checkpoint":"head.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )
    return manifest


def _write_multi_prototype_bundle(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "multi_prototype.pt"
    torch.save(make_multi_prototype_payload(threshold=0.31), checkpoint)
    manifest = tmp_path / "multi_prototype.neri.json"
    manifest.write_text(
        '{"backend":"dinov3","checkpoint":"multi_prototype.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )
    return manifest


def _vector(index=0):
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("NERI_DINOV3_STATE_DIR", str(tmp_path / "state"))
    from system.dinov3.api import dinov3_registry_router

    app = FastAPI()
    app.include_router(dinov3_registry_router())
    return TestClient(app)


def _seed_registration(manifest: Path, *, event_count=5, set_identity=True):
    from system.backend.dinov3_registry_service import open_registry_for_model

    registry = open_registry_for_model(str(manifest))
    try:
        entry = None
        base = datetime(2026, 9, 9, 0, 0, 0)
        for index in range(event_count):
            if entry is None:
                entry = registry.record_unknown(
                    _vector(),
                    camera_id=f"cam-{index % 2}",
                    captured_at=base + timedelta(hours=index),
                    source_path=f"/survey/cam-{index % 2}/{index}.jpg",
                )
            else:
                entry = registry.record_observation(
                    entry.id,
                    _vector(),
                    camera_id=f"cam-{index % 2}",
                    captured_at=base + timedelta(hours=index),
                    source_path=f"/survey/cam-{index % 2}/{index}.jpg",
                )
        if set_identity:
            entry = registry.set_identity(
                entry.id,
                common_name="豹猫",
                scientific_name="Prionailurus bengalensis",
            )
        return entry.id
    finally:
        registry.close()


def test_registry_list_detail_events_identity_and_register(monkeypatch, tmp_path):
    manifest = _write_model_bundle(tmp_path)
    client = _client(monkeypatch, tmp_path)
    registration_id = _seed_registration(manifest)

    listed = client.get(
        "/api/dinov3/registry",
        params={"classification_model_path": str(manifest)},
    )
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == registration_id
    assert listed.json()[0]["can_register"] is True

    detail = client.get(
        f"/api/dinov3/registry/{registration_id}",
        params={"classification_model_path": str(manifest)},
    )
    assert detail.status_code == 200
    assert detail.json()["common_name"] == "豹猫"

    events = client.get(
        f"/api/dinov3/registry/{registration_id}/events",
        params={"classification_model_path": str(manifest)},
    )
    assert events.status_code == 200
    assert len(events.json()) == 5
    assert "embedding" not in events.json()[0]

    identity = client.patch(
        f"/api/dinov3/registry/{registration_id}/identity",
        json={
            "classification_model_path": str(manifest),
            "common_name": "豹猫（测试）",
            "scientific_name": "Prionailurus bengalensis",
        },
    )
    assert identity.status_code == 200
    assert identity.json()["common_name"] == "豹猫（测试）"

    registered = client.post(
        f"/api/dinov3/registry/{registration_id}/register",
        json={"classification_model_path": str(manifest)},
    )
    assert registered.status_code == 200
    assert registered.json()["status"] == "provisional"


def test_register_api_blocks_duplicate_species_matching_formal_bank(monkeypatch, tmp_path):
    manifest = _write_multi_prototype_bundle(tmp_path)
    client = _client(monkeypatch, tmp_path)
    registration_id = _seed_registration(manifest, event_count=5)

    response = client.post(
        f"/api/dinov3/registry/{registration_id}/register",
        json={"classification_model_path": str(manifest)},
    )

    assert response.status_code == 409
    assert "species" in response.json()["detail"].lower()
    assert "A" in response.json()["detail"]


def test_registry_api_maps_not_found_invalid_model_and_unmet_conditions(monkeypatch, tmp_path):
    manifest = _write_model_bundle(tmp_path)
    client = _client(monkeypatch, tmp_path)
    registration_id = _seed_registration(manifest, event_count=1, set_identity=False)

    missing = client.get(
        "/api/dinov3/registry/9999",
        params={"classification_model_path": str(manifest)},
    )
    assert missing.status_code == 404

    blocked = client.post(
        f"/api/dinov3/registry/{registration_id}/register",
        json={"classification_model_path": str(manifest)},
    )
    assert blocked.status_code == 409

    invalid = tmp_path / "not-dino.pt"
    invalid.write_bytes(b"x")
    bad = client.get(
        "/api/dinov3/registry",
        params={"classification_model_path": str(invalid)},
    )
    assert bad.status_code == 400
