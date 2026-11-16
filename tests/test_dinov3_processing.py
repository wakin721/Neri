"""Unit and integration tests for DINOv3 context and sequence processing.

Covers:
- Causal history indexing and deduplication (FastCausalHistoryIndex)
- Content hash deduplication (renamed byte-identical media)
- Different camera isolation and missing timestamp independence
- Feature adaptation with frozen history (adapt_with_history parity with adapt_numpy)
- Crop collector deduplication, path discard on retry, and lifecycle clear
- Job finalizer cross-batch same-event consistency
- Rejected box retention as Unknown with bounded scores and threshold metadata
- Sequence pending/completed flags
- Job cancellation guards before and after classification
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.backend.models_legacy import DetectionItem
from system.dinov3.context import (
    FastCausalHistoryIndex,
    HistoryConfig,
    adapt_numpy,
    adapt_with_history,
    camera_id_for_path,
    group_job_events,
    parse_capture_time,
    unit,
)
from system.dinov3.processor import (
    DinoV3CropCollector,
    DinoV3CropRecord,
    DinoV3SequenceFinalizationError,
    _image_file_digest,
    finalize_photo_job,
)


@dataclass
class MockPrediction:
    species: str
    accepted: bool
    best_known_species: str
    known_score: float
    threshold: float
    embedding: np.ndarray
    ridge_score: float = 0.8
    within_score: float = 0.7
    candidates: tuple = ()
    confidence: float = 0.75

    def as_candidate(self, *, detection_confidence=None):
        return {
            "name": self.best_known_species,
            "conf": self.confidence,
            "raw_cls_conf": self.confidence,
            "raw_det_conf": detection_confidence,
            "accepted": self.accepted,
            "known_score": self.known_score,
            "threshold": self.threshold,
            "best_known_species": self.best_known_species,
            "predicted_species": self.best_known_species,
            "ridge_score": self.ridge_score,
            "within_score": self.within_score,
        }


def test_context_causal_history_and_camera_isolation():
    """Verify causal filtering, short_term=0 sequence rules, and camera isolation."""
    base_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        # Camera 1, Seq 1: two items
        dict(camera_id="cam1", seq_id="seq1", datetime=base_time, content_hash="hash_a"),
        dict(camera_id="cam1", seq_id="seq1", datetime=base_time + timedelta(minutes=5), content_hash="hash_b"),
        # Camera 1, Seq 2: later event (sees Seq 1)
        dict(camera_id="cam1", seq_id="seq2", datetime=base_time + timedelta(hours=2), content_hash="hash_c"),
        # Camera 1, Seq 3: still later event (sees Seq 1 and Seq 2)
        dict(camera_id="cam1", seq_id="seq3", datetime=base_time + timedelta(hours=6), content_hash="hash_d"),
        # Camera 2, Seq 4: different camera (must not see Camera 1)
        dict(camera_id="cam2", seq_id="seq4", datetime=base_time + timedelta(hours=3), content_hash="hash_e"),
    ]

    cfg = HistoryConfig(days=7.0, max_history=64, short_term=0, seed=20261105)
    index = FastCausalHistoryIndex(records, config=cfg)
    batch = index.build()
    history = batch.indices

    assert history.shape == (5, 64)
    # Row 0 (earliest in seq1): no history
    assert (history[0] == -1).all()
    # Row 1 (in seq1, short_term=0): same sequence has no admittance
    assert (history[1] == -1).all()
    # Row 2 (seq2): should select visible past from seq1
    assert history[2, 0] in (0, 1)
    assert history[2, 1] == -1
    # Row 3 (seq3): should select visible past from seq1 and seq2 (<=1 per sequence)
    selected_row3 = [idx for idx in history[3] if idx >= 0]
    assert len(selected_row3) == 2
    assert 2 in selected_row3
    assert any(i in selected_row3 for i in (0, 1))
    # Row 4 (camera 2): must NOT see any events from camera 1
    assert (history[4] == -1).all()


def test_context_content_hash_deduplication():
    """Verify that identical content_hash across different paths prevents attending to self."""
    base_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        # Camera 1, Seq 1, Image A
        dict(camera_id="cam1", seq_id="seq1", datetime=base_time, content_hash="identical_hash_1"),
        # Camera 1, Seq 2, Image B (renamed byte-identical duplicate of Image A)
        dict(camera_id="cam1", seq_id="seq2", datetime=base_time + timedelta(hours=1), content_hash="identical_hash_1"),
    ]
    cfg = HistoryConfig(days=7.0, max_history=64, short_term=0, seed=20261105)
    index = FastCausalHistoryIndex(records, config=cfg)
    history = index.build().indices

    # Query on row 1 shares content_hash with row 0: row 0 must be deduplicated away
    assert (history[1] == -1).all()


def test_context_missing_timestamp_handling():
    """Verify missing timestamps produce independent sequences with no history."""
    base_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = [
        dict(camera_id="cam1", seq_id="seq1", datetime=base_time, content_hash="c1"),
        dict(camera_id="cam1", seq_id="seq2", datetime=None, content_hash="c2"),  # missing time
        dict(camera_id="cam1", seq_id="seq3", datetime=base_time + timedelta(hours=2), content_hash="c3"),
    ]
    cfg = HistoryConfig(days=7.0, max_history=64, short_term=0, seed=20261105)
    index = FastCausalHistoryIndex(records, config=cfg)
    history = index.build().indices

    # Missing timestamp query gets no history
    assert (history[1] == -1).all()
    # Row 2 (seq3) can see row 0, but can NEVER see row 1
    selected_row2 = [idx for idx in history[2] if idx >= 0]
    assert 1 not in selected_row2
    assert 0 in selected_row2


def test_adapt_with_history_numerical_parity():
    """Verify adapt_with_history preserves unit norms, cold/warm history, and chunking."""
    np.random.seed(20261107)
    raw = np.random.randn(6, 768).astype(np.float32)
    raw = unit(raw)

    history = np.full((6, 64), -1, dtype=np.int64)
    # Row 2 attends to row 0
    history[2, 0] = 0
    # Row 3 attends to row 0 and 1
    history[3, 0] = 0
    history[3, 1] = 1

    adapted = adapt_with_history(raw, history, temperature=0.1, alpha=0.1, chunk_size=2)
    assert adapted.shape == raw.shape
    assert adapted.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(adapted, axis=1), 1.0, atol=1e-5)

    # Cold rows (0, 1, 4, 5) unchanged
    for idx in (0, 1, 4, 5):
        np.testing.assert_allclose(adapted[idx], raw[idx], atol=1e-6)

    # Warm rows modified
    assert float(np.max(np.abs(adapted[2] - raw[2]))) > 1e-4
    assert float(np.max(np.abs(adapted[3] - raw[3]))) > 1e-4

    # Check alias
    adapted_alias = adapt_numpy(raw, history, temperature=0.1, alpha=0.1)
    np.testing.assert_allclose(adapted, adapted_alias, atol=1e-7)


def test_group_job_events_30min_threshold():
    """Verify 30-min gap clustering, camera separation, and missing time isolation."""
    t0 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    records = [
        dict(camera_id="c1", path="img1.jpg", datetime=t0, content_id="1"),
        dict(camera_id="c1", path="img2.jpg", datetime=t0 + timedelta(minutes=15), content_id="2"),  # same event
        dict(camera_id="c1", path="img3.jpg", datetime=t0 + timedelta(minutes=46), content_id="3"),  # new event (31m gap)
        dict(camera_id="c2", path="img4.jpg", datetime=t0 + timedelta(minutes=5), content_id="4"),   # different cam
        dict(camera_id="c1", path="img5.jpg", datetime=None, content_id="5"),                         # missing time
    ]

    grouped = group_job_events(records, gap_seconds=1800)
    by_content = {g["content_id"]: g for g in grouped}

    # Items 1 and 2 share the same event sequence
    assert by_content["1"]["sequence_id"] == by_content["2"]["sequence_id"]
    # Item 3 has a different sequence
    assert by_content["3"]["sequence_id"] != by_content["1"]["sequence_id"]
    # Item 4 is in camera 2
    assert by_content["4"]["camera_id"] == "c2"
    assert "event:c2:" in by_content["4"]["sequence_id"]
    # Item 5 is missing timestamp
    assert by_content["5"]["timestamp_missing"] is True
    assert "missing:" in by_content["5"]["sequence_id"]


def test_collector_deduplication_and_discard():
    """Verify DinoV3CropCollector deduplicates same box_index and supports discard_paths on retry."""
    collector = DinoV3CropCollector()
    dummy_emb = np.zeros(768, dtype=np.float32)

    # Record box 0 and box 1 for image A
    collector.record_crop(DinoV3CropRecord("path/to/a.jpg", 0, (0, 0, 10, 10), 0.9, dummy_emb))
    collector.record_crop(DinoV3CropRecord("path/to/a.jpg", 1, (10, 10, 20, 20), 0.8, dummy_emb))
    assert len(collector.get_crops_for_path("path/to/a.jpg")) == 2

    # OOM retry on image A with updated confidence for box 0 (must NOT duplicate)
    collector.record_crop(DinoV3CropRecord("path/to/a.jpg", 0, (0, 0, 10, 10), 0.95, dummy_emb))
    crops_a = collector.get_crops_for_path("path/to/a.jpg")
    assert len(crops_a) == 2
    assert crops_a[0].detection_confidence == 0.95

    # Record for image B
    collector.record_crop(DinoV3CropRecord("path/to/b.jpg", 0, (0, 0, 10, 10), 0.7, dummy_emb))
    assert len(collector.get_all_crops()) == 3

    # Batch retry discarding image A
    collector.discard_paths(["path/to/a.jpg"])
    assert len(collector.get_crops_for_path("path/to/a.jpg")) == 0
    assert len(collector.get_crops_for_path("path/to/b.jpg")) == 1

    collector.clear()
    assert len(collector.get_all_crops()) == 0


def test_finalize_photo_job_sequence_aggregation(tmp_path):
    """Verify finalizer re-evaluates across batches, clears sequence_pending, sets sequence_completed, and cleans collector."""
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector.dinov3_classifier.display_name = "Neri1.2preview"
    detector._dinov3_collector = collector

    # Setup two mock photos in tmp_path
    file1 = tmp_path / "img1.jpg"
    file2 = tmp_path / "img2.jpg"
    file1.write_bytes(b"image_bytes_1")
    file2.write_bytes(b"image_bytes_2")

    emb1 = np.ones(768, dtype=np.float32) / np.sqrt(768)
    emb2 = np.ones(768, dtype=np.float32) / np.sqrt(768)

    collector.record_crop(DinoV3CropRecord(str(file1), 0, (10, 10, 50, 50), 0.9, emb1))
    collector.record_crop(DinoV3CropRecord(str(file2), 0, (20, 20, 60, 60), 0.85, emb2))

    pred_accepted = MockPrediction(
        species="Roe deer", accepted=True, best_known_species="Roe deer",
        known_score=2.5, threshold=1.0, embedding=emb1, confidence=0.82,
    )
    pred_rejected = MockPrediction(
        species="Wild boar", accepted=False, best_known_species="Wild boar",
        known_score=-0.5, threshold=1.0, embedding=emb2, confidence=0.18,
    )
    detector.dinov3_classifier.classify_features.return_value = [pred_accepted, pred_rejected]

    items = [
        DetectionItem(
            filename="img1.jpg", path=str(file1), file_type="image",
            date_taken="2026-11-01 10:00:00", detection_boxes=[{}],
            detection_data={"物种名称": "Roe deer", "sequence_pending": True},
        ),
        DetectionItem(
            filename="img2.jpg", path=str(file2), file_type="image",
            date_taken="2026-11-01 10:05:00", detection_boxes=[{}],
            detection_data={"物种名称": "Unknown", "sequence_pending": True},
        ),
    ]

    saved_batches = []
    def mock_save(batch, root):
        saved_batches.append(batch)

    summary_updates = []
    final_results = finalize_photo_job(
        detector, items, tmp_path, save_batch_fn=mock_save,
        mutate_job=lambda **kw: summary_updates.append(kw),
    )
    assert summary_updates[-1]["message"].startswith("Neri1.2preview序列聚合完成")

    assert len(final_results) == 2
    # Item 1: accepted
    d1 = final_results[0].detection_data
    assert d1["物种名称"] == "Roe deer"
    assert d1["sequence_completed"] is True
    assert d1["sequence_pending"] is False
    assert d1["sequence_status"] == "complete"
    box1 = d1["检测框"][0]
    assert box1["物种"] == "Roe deer"
    assert box1["accepted"] is True
    assert box1["sequence_status"] == "complete"
    assert box1["候选项"][0]["sequence_pending"] is False
    assert box1["候选项"][0]["sequence_status"] == "complete"

    # Item 2: rejected box retained as Unknown with bounded confidence and candidate threshold metadata
    d2 = final_results[1].detection_data
    assert d2["物种名称"] == "Unknown"
    assert d2["sequence_completed"] is True
    assert d2["sequence_pending"] is False
    assert d2["sequence_status"] == "complete"
    box2 = d2["检测框"][0]
    assert box2["物种"] == "Unknown"
    assert box2["accepted"] is False
    assert box2["known_score"] == -0.5
    assert box2["threshold"] == 1.0
    assert box2["best_known_species"] == "Wild boar"
    assert box2["置信度"] == 0.18
    assert box2["候选项"][0]["name"] == "Unknown"
    assert box2["候选项"][0]["best_known_species"] == "Wild boar"
    assert box2["候选项"][0]["sequence_pending"] is False
    assert box2["候选项"][0]["sequence_status"] == "complete"

    # Collector must be cleared after finalization
    assert len(collector.get_all_crops()) == 0
    # Saved batch must be called
    assert len(saved_batches) == 1


def test_finalize_photo_job_partial_summary_keeps_result_pending(tmp_path):
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector

    image_path = tmp_path / "partial.jpg"
    image_path.write_bytes(b"partial sequence")
    embedding = np.ones(768, dtype=np.float32) / np.sqrt(768)
    collector.record_crop(
        DinoV3CropRecord(
            str(image_path),
            0,
            (10, 10, 50, 50),
            0.9,
            embedding,
        )
    )
    detector.dinov3_classifier.classify_features.return_value = [
        MockPrediction(
            species="Roe deer",
            accepted=True,
            best_known_species="Roe deer",
            known_score=2.5,
            threshold=1.0,
            embedding=embedding,
            confidence=0.82,
        )
    ]
    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
        date_taken="2026-11-01 10:00:00",
    )

    partial = finalize_photo_job(
        detector,
        [item],
        tmp_path,
        is_cancelled=lambda: True,
        complete_sequence=False,
        incomplete_reason="cancelled",
        total_photos=3,
    )

    assert detector.dinov3_classifier.classify_features.called
    data = partial[0].detection_data
    assert data["sequence_completed"] is False
    assert data["sequence_pending"] is True
    assert data["sequence_status"] == "partial"
    assert data["sequence_reason"] == "cancelled"
    assert data["sequence_processed_count"] == 1
    assert data["sequence_total_count"] == 3
    box = data["检测框"][0]
    assert box["species" if "species" in box else "物种"] == "Roe deer"
    assert box["sequence_pending"] is True
    assert box["sequence_status"] == "partial"
    assert box["候选项"][0]["sequence_pending"] is True
    assert box["候选项"][0]["sequence_status"] == "partial"


@pytest.mark.parametrize("collector_state", ["missing", "empty"])
def test_finalize_photo_job_never_claims_completion_without_crops(
    tmp_path,
    collector_state,
):
    image_path = tmp_path / "empty.jpg"
    image_path.write_bytes(b"no collected crops")
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = (
        None if collector_state == "missing" else DinoV3CropCollector()
    )
    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
    )

    with pytest.raises(DinoV3SequenceFinalizationError):
        finalize_photo_job(detector, [item], tmp_path, complete_sequence=True)


def test_finalize_photo_job_completes_successfully_detected_empty_photos(tmp_path):
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector
    items = []
    for index in range(2):
        image_path = tmp_path / f"empty-{index}.jpg"
        image_path.write_bytes(b"successfully detected empty photo")
        items.append(
            DetectionItem(
                filename=image_path.name,
                path=str(image_path),
                file_type="image",
                detection_data={
                    "检测框": [],
                    "detection_completed": True,
                    "detection_box_count": 0,
                    "detection_box_indices": [],
                },
            )
        )
    saved_batches = []

    completed = finalize_photo_job(
        detector,
        items,
        tmp_path,
        save_batch_fn=lambda batch, root: saved_batches.append(batch),
        complete_sequence=True,
    )

    detector.dinov3_classifier.classify_features.assert_not_called()
    assert len(completed) == 2
    assert len(saved_batches) == 1
    for item in completed:
        assert item.detection_data["sequence_completed"] is True
        assert item.detection_data["sequence_pending"] is False
        assert item.detection_data["sequence_status"] == "complete"
    assert collector.get_all_crops() == []


@pytest.mark.parametrize(
    "detection_data",
    [
        {
            "检测框": [],
            "detection_completed": False,
            "detection_box_count": 0,
            "detection_box_indices": [],
        },
        {
            "检测框": [],
            "detection_completed": True,
            "detection_box_count": 1,
            "detection_box_indices": [],
        },
    ],
)
def test_finalize_photo_job_rejects_unconfirmed_or_inconsistent_empty_detection(
    tmp_path,
    detection_data,
):
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector
    image_path = tmp_path / "unconfirmed-empty.jpg"
    image_path.write_bytes(b"empty output without valid completion evidence")
    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
        detection_data=detection_data,
    )

    with pytest.raises(DinoV3SequenceFinalizationError):
        finalize_photo_job(detector, [item], tmp_path, complete_sequence=True)
    detector.dinov3_classifier.classify_features.assert_not_called()


@pytest.mark.parametrize("collected_indices", [(0,), (2,)])
def test_finalize_photo_job_rejects_partial_or_unexpected_crop_collection(
    tmp_path,
    collected_indices,
):
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector
    image_path = tmp_path / "multi-box.jpg"
    image_path.write_bytes(b"two detected boxes")
    embedding = np.ones(768, dtype=np.float32) / np.sqrt(768)
    for box_index in collected_indices:
        collector.record_crop(
            DinoV3CropRecord(
                str(image_path),
                box_index,
                (10, 10, 50, 50),
                0.9,
                embedding,
            )
        )
    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
        detection_data={
            "检测框": [
                {"边界框": [10, 10, 50, 50]},
                {"边界框": [60, 60, 90, 90]},
            ],
            "detection_completed": True,
            "detection_box_count": 2,
            "detection_box_indices": [0, 1],
        },
    )

    with pytest.raises(DinoV3SequenceFinalizationError, match="crop collection is incomplete"):
        finalize_photo_job(detector, [item], tmp_path, complete_sequence=True)
    detector.dinov3_classifier.classify_features.assert_not_called()


def test_finalize_photo_job_cancellation_guards(tmp_path):
    """Verify cancellation before and after classification skips database writes."""
    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = MagicMock()
    detector.dinov3_classifier.backend = "dinov3"
    detector._dinov3_collector = collector

    file1 = tmp_path / "img1.jpg"
    file1.write_bytes(b"test")
    collector.record_crop(DinoV3CropRecord(str(file1), 0, (0, 0, 10, 10), 0.9, np.zeros(768, dtype=np.float32)))
    item = DetectionItem(filename="img1.jpg", path=str(file1), file_type="image", date_taken="2026-11-01 10:00:00")

    save_mock = MagicMock()

    # 1. Cancelled before start
    res = finalize_photo_job(detector, [item], tmp_path, is_cancelled=lambda: True, save_batch_fn=save_mock)
    detector.dinov3_classifier.classify_features.assert_not_called()
    save_mock.assert_not_called()

    # 2. Cancelled after classification
    detector.dinov3_classifier.classify_features.return_value = [
        MockPrediction(species="Deer", accepted=True, best_known_species="Deer", known_score=1.0, threshold=0.5, embedding=np.zeros(768, dtype=np.float32))
    ]
    cancel_state = [False]
    def check_cancel():
        # Become cancelled during/after classification
        if detector.dinov3_classifier.classify_features.called:
            return True
        return False

    collector.record_crop(DinoV3CropRecord(str(file1), 0, (0, 0, 10, 10), 0.9, np.zeros(768, dtype=np.float32)))
    res2 = finalize_photo_job(detector, [item], tmp_path, is_cancelled=check_cancel, save_batch_fn=save_mock)
    save_mock.assert_not_called()
    assert len(collector.get_all_crops()) == 0


def test_group_job_events_separates_same_photo_boxes_and_links_tracks():
    """Verify same-photo multi-target boxes never share a track_id while sharing event_id."""
    t0 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    emb_squirrel_1 = np.zeros(16, dtype=np.float32)
    emb_squirrel_1[0] = 1.0
    emb_squirrel_2 = np.zeros(16, dtype=np.float32)
    emb_squirrel_2[0] = 0.98
    emb_squirrel_2[1] = 0.20

    emb_log_1 = np.zeros(16, dtype=np.float32)
    emb_log_1[2] = 1.0
    emb_log_2 = np.zeros(16, dtype=np.float32)
    emb_log_2[2] = 0.99
    emb_log_2[3] = 0.14

    records = [
        dict(
            camera_id="c1",
            path="IM_00052.JPG",
            box_index=0,
            bbox=[100, 100, 220, 200],
            raw_embedding=emb_squirrel_1,
            datetime=t0,
            content_id="sha52:0",
        ),
        dict(
            camera_id="c1",
            path="IM_00052.JPG",
            box_index=1,
            bbox=[600, 400, 760, 520],
            raw_embedding=emb_log_1,
            datetime=t0,
            content_id="sha52:1",
        ),
        dict(
            camera_id="c1",
            path="IM_00053.JPG",
            box_index=0,
            bbox=[110, 105, 230, 205],
            raw_embedding=emb_squirrel_2,
            datetime=t0 + timedelta(seconds=2),
            content_id="sha53:0",
        ),
        dict(
            camera_id="c1",
            path="IM_00053.JPG",
            box_index=1,
            bbox=[602, 401, 761, 521],
            raw_embedding=emb_log_2,
            datetime=t0 + timedelta(seconds=2),
            content_id="sha53:1",
        ),
        dict(
            camera_id="c1",
            path="IM_MISSING.JPG",
            box_index=0,
            bbox=[10, 10, 50, 50],
            raw_embedding=emb_squirrel_1,
            datetime=None,
            content_id="shamiss:0",
        ),
        dict(
            camera_id="c1",
            path="IM_MISSING.JPG",
            box_index=1,
            bbox=[300, 300, 400, 400],
            raw_embedding=emb_log_1,
            datetime=None,
            content_id="shamiss:1",
        ),
    ]

    grouped = group_job_events(records, gap_seconds=1800)
    by_content = {g["content_id"]: g for g in grouped}

    # All timed boxes in the burst share the same 30-min event_id for causal history exclusion
    assert by_content["sha52:0"]["event_id"] == by_content["sha52:1"]["event_id"]
    assert by_content["sha52:0"]["event_id"] == by_content["sha53:0"]["event_id"]

    # Same-photo boxes never share a track_id / sequence_id
    assert by_content["sha52:0"]["sequence_id"] != by_content["sha52:1"]["sequence_id"]
    assert by_content["sha53:0"]["sequence_id"] != by_content["sha53:1"]["sequence_id"]

    # Cross-frame target linking connects squirrel-to-squirrel and log-to-log
    assert by_content["sha52:0"]["sequence_id"] == by_content["sha53:0"]["sequence_id"]
    assert by_content["sha52:1"]["sequence_id"] == by_content["sha53:1"]["sequence_id"]

    # Missing-timestamp multi-box photo also separates track_id per box
    assert by_content["shamiss:0"]["event_id"] == by_content["shamiss:1"]["event_id"]
    assert by_content["shamiss:0"]["sequence_id"] != by_content["shamiss:1"]["sequence_id"]


def test_finalize_photo_job_separates_multi_target_scores_and_acceptance(tmp_path):
    """Verify IM_00052-style multi-target photo saves distinct per-box scores and rejects background box."""
    from system.dinov3.classifier import DinoV3Classifier
    from system.dinov3.model import DinoV3RidgeWithinModel

    rng = np.random.default_rng(20261107)
    dim = 768
    c_squirrel = np.zeros(dim, dtype=np.float32)
    c_squirrel[0] = 1.0
    c_badger = np.zeros(dim, dtype=np.float32)
    c_badger[1] = 1.0
    c_log = np.zeros(dim, dtype=np.float32)
    c_log[2] = 1.0

    bank_feats = []
    bank_labels = []
    bank_cams = []
    bank_seqs = []
    for sp_name, center in (("岩松鼠", c_squirrel), ("猪獾", c_badger)):
        for i in range(16):
            v = center + 0.04 * rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            bank_feats.append(v)
            bank_labels.append(sp_name)
            bank_cams.append(f"cam_{i % 4}")
            bank_seqs.append(f"seq_{sp_name}_{i // 2}")

    model = DinoV3RidgeWithinModel(ridge=0.01)
    model.fit(
        np.stack(bank_feats, axis=0),
        bank_labels,
        bank_cams,
        bank_seqs,
        origin=np.zeros(dim, dtype=np.float64),
        transform=np.eye(dim, dtype=np.float64),
    )

    cal_feats = list(bank_feats[:16])
    cal_labels = list(bank_labels[:16])
    cal_cams = [f"cal_cam_{i % 4}" for i in range(16)]
    cal_seqs = [f"cal_seq_k_{i // 2}" for i in range(16)]
    for i in range(12):
        v = c_log + 0.05 * rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        cal_feats.append(v)
        cal_labels.append("倒木背景")
        cal_cams.append(f"cal_cam_{i % 3}")
        cal_seqs.append(f"cal_seq_u_{i // 2}")

    model.calibrate(
        np.stack(cal_feats, axis=0),
        cal_labels,
        cal_cams,
        cal_seqs,
        target_recall=0.90,
    )

    collector = DinoV3CropCollector()
    detector = MagicMock()
    detector.dinov3_classifier = DinoV3Classifier(model)
    detector._dinov3_collector = collector

    cam_dir = tmp_path / "CamA"
    cam_dir.mkdir(parents=True)
    img52 = cam_dir / "IM_00052.JPG"
    img53 = cam_dir / "IM_00053.JPG"
    img52.write_bytes(b"img_00052_bytes")
    img53.write_bytes(b"img_00053_bytes")

    q_squirrel_1 = c_squirrel + 0.02 * rng.standard_normal(dim).astype(np.float32)
    q_squirrel_1 /= np.linalg.norm(q_squirrel_1)
    q_log_1 = c_log + 0.02 * rng.standard_normal(dim).astype(np.float32)
    q_log_1 /= np.linalg.norm(q_log_1)
    q_squirrel_2 = c_squirrel + 0.02 * rng.standard_normal(dim).astype(np.float32)
    q_squirrel_2 /= np.linalg.norm(q_squirrel_2)
    q_log_2 = c_log + 0.02 * rng.standard_normal(dim).astype(np.float32)
    q_log_2 /= np.linalg.norm(q_log_2)

    collector.record_crop(DinoV3CropRecord(str(img52), 0, (100, 100, 240, 220), 0.72168, q_squirrel_1))
    collector.record_crop(DinoV3CropRecord(str(img52), 1, (580, 380, 760, 520), 0.58545, q_log_1))
    collector.record_crop(DinoV3CropRecord(str(img53), 0, (108, 104, 248, 224), 0.74100, q_squirrel_2))
    collector.record_crop(DinoV3CropRecord(str(img53), 1, (582, 381, 761, 521), 0.57900, q_log_2))

    items = [
        DetectionItem(filename="IM_00052.JPG", path=str(img52), file_type="image", date_taken="2026-11-01 10:00:00"),
        DetectionItem(filename="IM_00053.JPG", path=str(img53), file_type="image", date_taken="2026-11-01 10:00:02"),
    ]

    updated = finalize_photo_job(detector, items, tmp_path)
    boxes_52 = updated[0].detection_boxes
    assert len(boxes_52) == 2

    animal_box, log_box = boxes_52[0], boxes_52[1]
    # Animal box is accepted as 岩松鼠; fallen log box in the same photo is rejected as Unknown
    assert animal_box["物种"] == "岩松鼠"
    assert animal_box["accepted"] is True
    assert animal_box["box_accepted"] is True
    assert log_box["物种"] == "Unknown"
    assert log_box["accepted"] is False
    assert log_box["box_accepted"] is False

    # Scores and confidences are strictly distinct between the two boxes in the same photo
    assert animal_box["置信度"] != log_box["置信度"]
    assert animal_box["置信度"] > 0.5 > log_box["置信度"]
    assert animal_box["known_score"] != log_box["known_score"]
    assert animal_box["box_known_score"] != log_box["box_known_score"]
    assert animal_box["sequence_known_score"] != log_box["sequence_known_score"]
    assert animal_box["track_id"] != log_box["track_id"]
    assert animal_box["event_id"] == log_box["event_id"]
