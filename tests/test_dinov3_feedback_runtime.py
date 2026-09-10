from __future__ import annotations

import sqlite3

import numpy as np
import pytest
import torch

from system.dinov3.checkpoint import load_checkpoint, validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.feedback import HumanFeedbackStore, feedback_path_for_registry
from system.dinov3.prototype_bank import PrototypeBank, PrototypeRecord
from system.dinov3.registry import SpeciesRegistry
from system.dinov3.runtime import load_dinov3_model
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def _vector(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


class _BankProvider:
    def __init__(self, bank: PrototypeBank):
        self.bank = bank

    def prototype_bank(self, feature_center):
        assert np.asarray(feature_center).shape == (768,)
        return self.bank


def _write_runtime_files(tmp_path):
    checkpoint_path = tmp_path / "head.pt"
    torch.save(
        make_multi_prototype_payload(
            classes=("A", "B"),
            threshold=0.8,
        ),
        checkpoint_path,
    )
    manifest_path = tmp_path / "head.neri.json"
    manifest_path.write_text(
        '{"backend":"dinov3","checkpoint":"head.pt","architecture":"dinov3_vitb16","feature_dim":768}',
        encoding="utf-8",
    )
    return checkpoint_path, manifest_path


def test_classifier_composes_checkpoint_feedback_and_registry_banks():
    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(threshold=0.8)
    )
    feedback = _BankProvider(
        PrototypeBank(
            formal=(
                PrototypeRecord("A", _vector(10), "feedback", registration_status="confirmed"),
            ),
            provisional=(
                PrototypeRecord("B", _vector(11), "feedback", registration_status="provisional"),
            ),
        )
    )
    registry = _BankProvider(
        PrototypeBank(
            formal=(
                PrototypeRecord(
                    "新物种",
                    _vector(12),
                    "overlay",
                    registry_id=7,
                    registration_status="confirmed",
                ),
            ),
            provisional=(
                PrototypeRecord(
                    "候选物种",
                    _vector(13),
                    "overlay",
                    registry_id=8,
                    registration_status="provisional",
                ),
            ),
        )
    )

    classifier = DinoV3Classifier(
        checkpoint,
        feedback=feedback,
        registry=registry,
    )
    bank = classifier._effective_bank()

    assert len(bank.formal) == len(checkpoint.prototypes) + 2
    assert [record.source for record in bank.formal[-2:]] == ["feedback", "overlay"]
    assert [record.source for record in bank.provisional] == ["feedback", "overlay"]


def test_confirmed_feedback_overlay_can_make_formal_prediction():
    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(threshold=0.8)
    )
    feedback = _BankProvider(
        PrototypeBank(
            formal=(
                PrototypeRecord("A", _vector(10), "feedback", registration_status="confirmed"),
            )
        )
    )
    prediction = DinoV3Classifier(
        checkpoint,
        feedback=feedback,
    ).classify_features(_vector(10)[None, :])[0]

    assert prediction.species == "A"
    assert prediction.accepted is True
    assert prediction.source == "feedback"
    assert prediction.registration_status == "confirmed"


def test_provisional_feedback_overlay_is_assistive_only():
    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(threshold=0.8)
    )
    feedback = _BankProvider(
        PrototypeBank(
            formal=(),
            provisional=(
                PrototypeRecord("A", _vector(10), "feedback", registration_status="provisional"),
            ),
        )
    )
    prediction = DinoV3Classifier(
        checkpoint,
        feedback=feedback,
    ).classify_features(_vector(10)[None, :])[0]

    assert prediction.species == "A"
    assert prediction.accepted is False
    assert prediction.assistive_match is True
    assert prediction.source == "feedback"


def test_runtime_creates_sibling_feedback_store_and_owns_it(tmp_path):
    checkpoint_path, manifest_path = _write_runtime_files(tmp_path)
    expected_fingerprint = load_checkpoint(checkpoint_path).fingerprint

    runtime = load_dinov3_model(
        manifest_path,
        state_root=tmp_path / "state",
        encoder_factory=lambda checkpoint, **kwargs: object(),
    )

    assert runtime.registry.model_fingerprint == expected_fingerprint
    assert runtime.feedback.model_fingerprint == expected_fingerprint
    assert runtime.feedback.path == feedback_path_for_registry(runtime.registry.path)
    assert runtime.classifier.feedback is runtime.feedback
    assert runtime.classifier.registry is runtime.registry
    assert runtime.owns_registry is True
    assert runtime.owns_feedback is True

    runtime.close()
    with pytest.raises(sqlite3.ProgrammingError):
        runtime.feedback._conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        runtime.registry._conn.execute("SELECT 1")


def test_runtime_rejects_injected_feedback_from_other_fingerprint(tmp_path):
    checkpoint_path, manifest_path = _write_runtime_files(tmp_path)
    fingerprint = load_checkpoint(checkpoint_path).fingerprint
    registry = SpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint=fingerprint,
    )
    feedback = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint="f" * 64,
        checkpoint_classes=("A", "B"),
        threshold=0.8,
    )

    with pytest.raises(ValueError, match="feedback.*fingerprint|Feedback.*fingerprint"):
        load_dinov3_model(
            manifest_path,
            registry=registry,
            feedback=feedback,
            encoder_factory=lambda checkpoint, **kwargs: object(),
        )

    feedback.close()
    registry.close()
