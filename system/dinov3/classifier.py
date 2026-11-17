"""Bridge the joint model's classification and rejection into Neri candidates."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .evidence import validate_evidence_weights


@dataclass(frozen=True)
class DinoV3Prediction:
    species: str
    accepted: bool
    best_known_species: str
    known_score: float
    threshold: float
    embedding: np.ndarray
    candidates: tuple
    ridge_score: float
    within_score: float
    source: str = "dinov3_ridge_within"
    registry_id: int | None = None
    registration_status: str | None = None
    assistive_match: bool = False
    box_known_score: float | None = None
    box_threshold: float | None = None
    box_accepted: bool | None = None
    sequence_known_score: float | None = None
    sequence_threshold: float | None = None
    sequence_accepted: bool | None = None
    raw_ridge_score: float | None = None
    ridge_margin: float | None = None
    fused_ridge_margin: float | None = None
    within_margin: float | None = None
    whitened_norm: float | None = None
    track_id: str | None = None
    gate_mode: str = "box_and_sequence"
    inference_recipe: str = "production_dual"
    frame_classification_score: float | None = None
    frame_classification_margin: float | None = None

    @property
    def confidence_kind(self):
        return ("sigmoid_frame_class_margin" if self.gate_mode == "sequence"
                and self.frame_classification_margin is not None else "sigmoid_gate_margin")

    @property
    def confidence(self):
        # The experimental gate is shared by an event. Display each frame's
        # support for the final candidate instead; this is not a probability.
        margin = (self.frame_classification_margin if self.confidence_kind == "sigmoid_frame_class_margin"
                  else self.known_score - self.threshold)
        delta = max(-700.0, min(700.0, margin))
        return 1.0 / (1.0 + math.exp(-delta))

    @property
    def box_confidence(self):
        score = self.box_known_score if self.box_known_score is not None else self.known_score
        thresh = self.box_threshold if self.box_threshold is not None else self.threshold
        delta = max(-700.0, min(700.0, score - thresh))
        return 1.0 / (1.0 + math.exp(-delta))

    @property
    def sequence_confidence(self):
        score = self.sequence_known_score if self.sequence_known_score is not None else self.known_score
        thresh = self.sequence_threshold if self.sequence_threshold is not None else self.threshold
        delta = max(-700.0, min(700.0, score - thresh))
        return 1.0 / (1.0 + math.exp(-delta))

    def as_candidate(self, *, detection_confidence=None):
        box_score = self.box_known_score if self.box_known_score is not None else self.known_score
        box_thresh = self.box_threshold if self.box_threshold is not None else self.threshold
        box_acc = self.box_accepted if self.box_accepted is not None else self.accepted
        seq_score = self.sequence_known_score if self.sequence_known_score is not None else self.known_score
        seq_thresh = self.sequence_threshold if self.sequence_threshold is not None else self.threshold
        seq_acc = self.sequence_accepted if self.sequence_accepted is not None else self.accepted
        raw_ridge = self.raw_ridge_score if self.raw_ridge_score is not None else self.ridge_score
        if self.gate_mode == "sequence":
            box_score = box_thresh = box_acc = None
        return dict(name=self.best_known_species, conf=self.confidence,
                    raw_cls_conf=self.confidence, raw_det_conf=detection_confidence,
                    accepted=self.accepted, known_score=self.known_score,
                    threshold=self.threshold, score_threshold=self.threshold,
                    box_known_score=box_score, box_threshold=box_thresh,
                    box_accepted=box_acc, box_confidence=None if self.gate_mode == "sequence" else self.box_confidence,
                    sequence_known_score=seq_score, sequence_threshold=seq_thresh,
                    sequence_accepted=seq_acc, sequence_confidence=self.sequence_confidence,
                    best_known_species=self.best_known_species,
                    predicted_species=self.best_known_species,
                    ridge_score=self.ridge_score, raw_ridge_score=raw_ridge,
                    ridge_margin=self.ridge_margin, fused_ridge_margin=self.fused_ridge_margin,
                    within_score=self.within_score, within_margin=self.within_margin,
                    whitened_norm=self.whitened_norm, track_id=self.track_id,
                    source=self.source, gate_mode=self.gate_mode, inference_recipe=self.inference_recipe,
                    classification_score=self.ridge_score, confidence_kind=self.confidence_kind,
                    frame_classification_score=self.frame_classification_score,
                    frame_classification_margin=self.frame_classification_margin,
                    frame_classification_confidence=self.confidence if self.confidence_kind == "sigmoid_frame_class_margin" else None,
                    registry_id=None, registration_status=None, assistive_match=False)


class DinoV3Classifier:
    backend = "dinov3"

    def __init__(self, model, *, encoder=None):
        self.model = model
        self.checkpoint = model
        self.encoder = encoder
        self.feedback = None
        self.registry = None

    @property
    def classes(self):
        return tuple(map(str, self.model.classes))

    @property
    def names(self):
        return dict(enumerate(self.classes))

    @property
    def rejection_metadata(self):
        return dict(mode="dinov3_ridge_within", gate="top75_radial08",
                    inference_recipe=getattr(self.model, "inference_recipe", "production_dual"),
                    gate_mode="sequence" if getattr(self.model, "inference_recipe", None) == "experiment_full" else "box_and_sequence",
                    sequence_aggregation="complete_sequence", calibrated=self.model.calibration is not None)

    def classify_features(self, features, *, camera_ids=None, sequence_ids=None,
                          content_ids=None, site_ids=None, adapted_features=None,
                          evidence_weights=None):
        raw = np.asarray(features, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 768 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("DINOv3 features must be finite [N, 768]")
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("DINOv3 features must be nonzero")
        weights = (
            validate_evidence_weights(evidence_weights, len(raw))
            if evidence_weights is not None
            else None
        )
        # Encoder output and cached supports already use the frozen float32
        # normalization. A second projection would perturb radial calibration.
        if (camera_ids is None) != (sequence_ids is None):
            raise ValueError("Supply both camera_ids and sequence_ids")
        cameras = camera_ids if camera_ids is not None else ["single"] * len(raw)
        sequences = sequence_ids if sequence_ids is not None else [f"single:{i}" for i in range(len(raw))]
        if weights is None:
            result = self.model.predict(raw, cameras, sequences, contents=content_ids,
                                        sites=site_ids, adapted_features=adapted_features)
        else:
            result = self.model.predict(raw, cameras, sequences, contents=content_ids,
                                        sites=site_ids, adapted_features=adapted_features,
                                        evidence_weights=weights)
        classes = np.asarray(self.model.classes)
        box_scores = result.get("box_known_score", result["known_score"])
        box_thresh = float(result.get("box_threshold", result["threshold"]))
        box_acc = result.get("box_accepted", result["accepted"])
        seq_scores = result.get("sequence_known_score", result["known_score"])
        seq_thresh = float(result.get("sequence_threshold", result["threshold"]))
        seq_acc = result.get("sequence_accepted", result["accepted"])
        within_margins = result.get("within_margins")
        whitened_norms = result.get("whitened_norms")
        resolved_tracks = result.get("resolved_tracks")
        frame_scores = result.get("classification_scores")
        output = []
        for i, species in enumerate(result["predictions"]):
            ranking = np.argsort(result["fused_scores"][i])[::-1]
            candidates = tuple(
                dict(
                    name=str(classes[j]),
                    ridge_score=float(result["fused_scores"][i, j]),
                    raw_ridge_score=float(result["ridge_scores"][i, j]),
                    within_score=float(result["within_scores"][i, j]),
                )
                for j in ranking[:5]
            )
            winner = int(np.flatnonzero(classes == species)[0])
            frame_score = frame_margin = None
            if result.get("gate_mode") == "sequence" and frame_scores is not None:
                frame_row = np.asarray(frame_scores[i], dtype=np.float64)
                frame_score = float(frame_row[winner])
                other = np.delete(frame_row, winner)
                frame_margin = frame_score - float(other.max()) if len(other) else 0.0
            raw_row = result["ridge_scores"][i]
            fused_row = result["fused_scores"][i]
            if len(classes) >= 2:
                raw_top2 = np.sort(raw_row)[-2:]
                fused_top2 = np.sort(fused_row)[-2:]
                raw_r_margin = float(raw_top2[1] - raw_top2[0])
                fused_r_margin = float(fused_top2[1] - fused_top2[0])
            else:
                raw_r_margin = float(raw_row[winner])
                fused_r_margin = float(fused_row[winner])
            embedding = raw[i].copy()
            embedding.setflags(write=False)
            output.append(DinoV3Prediction(
                species=str(species), best_known_species=str(species),
                accepted=bool(result["accepted"][i]), known_score=float(result["known_score"][i]),
                threshold=float(result["threshold"]), embedding=embedding, candidates=candidates,
                ridge_score=float(result["fused_scores"][i, winner]),
                within_score=float(result["within_scores"][i, winner]),
                box_known_score=None if result.get("gate_mode") == "sequence" else float(box_scores[i]),
                box_threshold=None if result.get("gate_mode") == "sequence" else box_thresh,
                box_accepted=None if result.get("gate_mode") == "sequence" else bool(box_acc[i]),
                sequence_known_score=float(seq_scores[i]),
                sequence_threshold=seq_thresh,
                sequence_accepted=bool(seq_acc[i]),
                raw_ridge_score=float(result["ridge_scores"][i, winner]),
                ridge_margin=raw_r_margin,
                fused_ridge_margin=fused_r_margin,
                within_margin=float(within_margins[i]) if within_margins is not None else None,
                whitened_norm=float(whitened_norms[i]) if whitened_norms is not None else None,
                track_id=str(resolved_tracks[i]) if resolved_tracks is not None else str(sequences[i]),
                gate_mode=result.get("gate_mode", "box_and_sequence"),
                inference_recipe=getattr(self.model, "inference_recipe", "production_dual"),
                frame_classification_score=frame_score, frame_classification_margin=frame_margin))
        return output

    def classify_crops(self, crops, *, array_color="rgb", **kwargs):
        if self.encoder is None:
            raise RuntimeError("DINOv3 encoder is not loaded")
        return self.classify_features(self.encoder.encode(crops, array_color=array_color), **kwargs)

    def classify_event(self, crops, *, array_color="rgb"):
        return self.classify_crops(crops, array_color=array_color,
                                   camera_ids=["event"] * len(crops), sequence_ids=["event"] * len(crops))
