"""A shared event decision must not overwrite each frame's class evidence."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov3.classifier import DinoV3Classifier


def classifier():
    classes = np.asarray(["A", "B", "C"])
    frame = np.asarray([[2., 0., -1.], [.4, .7, -1.]])
    fused = np.asarray([[1., .5, -1.], [1., .9, -1.]])
    def predict(raw, cameras, sequences, **kwargs):
        assert len(raw) == 2
        return dict(predictions=np.asarray(["A", "A"]), accepted=np.asarray([True, True]),
                    known_score=np.asarray([3., 3.]), threshold=1.,
                    sequence_known_score=np.asarray([3., 3.]), sequence_threshold=1.,
                    sequence_accepted=np.asarray([True, True]), classification_scores=frame,
                    fused_scores=fused, ridge_scores=frame, within_scores=frame,
                    gate_mode="sequence", resolved_tracks=np.asarray(sequences))
    return DinoV3Classifier(SimpleNamespace(classes=classes, predict=predict, inference_recipe="experiment_full"))


def features():
    x = np.zeros((2, 768), np.float32)
    x[0, 0] = x[1, 1] = 1
    return x


def test_frame_support_uses_final_candidate_and_keeps_shared_gate():
    predictions = classifier().classify_features(features(),camera_ids=["c"]*2,sequence_ids=["e"]*2)
    a,b = predictions
    assert a.known_score == b.known_score == 3
    assert a.sequence_confidence == b.sequence_confidence
    assert a.confidence == pytest.approx(1/(1+math.exp(-2)))
    # Sequence selected A, but this particular frame favors B. Its evidence
    # must be below .5 instead of reporting its own top-two margin as positive.
    assert b.frame_classification_margin == pytest.approx(-.3)
    assert b.confidence == pytest.approx(1/(1+math.exp(.3)))
    assert a.confidence != b.confidence
    assert a.accepted and b.accepted
    candidate = b.as_candidate()
    assert candidate["confidence_kind"] == "sigmoid_frame_class_margin"
    assert candidate["conf"] == candidate["frame_classification_confidence"] == b.confidence
    assert candidate["sequence_confidence"] == b.sequence_confidence
    assert candidate["accepted"] is True


def test_finalizer_persists_per_frame_display_and_event_decision(tmp_path):
    from system.backend.models_legacy import DetectionItem
    from system.dinov3.processor import DinoV3CropCollector,DinoV3CropRecord,finalize_photo_job
    photo = tmp_path/"two-objects.jpg"
    photo.write_bytes(b"provided embeddings avoid image decoding")
    collector = DinoV3CropCollector()
    for i,x in enumerate(features()):
        collector.record_crop(DinoV3CropRecord(str(photo),i,(i*30,0,i*30+20,20),.9,x))
    detector = SimpleNamespace(dinov3_classifier=classifier(),_dinov3_collector=collector)
    item = DetectionItem(filename=photo.name,path=str(photo),file_type="image",date_taken="2026-11-14 10:00:00")
    result = finalize_photo_job(detector,[item],tmp_path)
    a,b = result[0].detection_boxes
    assert a["置信度"] != b["置信度"]
    assert b["置信度"] < .5 and b["物种"] == "A" and b["accepted"] is True
    assert a["sequence_known_score"] == b["sequence_known_score"] == 3
    assert a["sequence_confidence"] == b["sequence_confidence"]
    for box in (a,b):
        assert box["confidence_kind"] == "sigmoid_frame_class_margin"
        assert box["frame_classification_confidence"] == box["置信度"]
        assert box["候选项"][0]["conf"] == box["置信度"]
