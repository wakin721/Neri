"""Reviewed linear-head classification and open-set prototype rejection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from .checkpoint import DINO_FEATURE_DIM, DinoV3Checkpoint
from .preprocess import ImageInput

def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array=np.asarray(values,dtype=np.float32); norms=np.linalg.norm(array,axis=1,keepdims=True)
    if np.any(norms<=0) or not np.isfinite(norms).all(): raise ValueError("Embedding rows must be finite and non-zero")
    return array/norms

def aggregate_event_embeddings(crop_embeddings: np.ndarray) -> np.ndarray:
    array=np.asarray(crop_embeddings,dtype=np.float32)
    if array.ndim!=2 or array.shape[1]!=DINO_FEATURE_DIM or len(array)==0: raise ValueError("Expected crop embeddings with shape (N, 768)")
    return _normalize_rows(_normalize_rows(array).mean(axis=0,keepdims=True))[0]

@dataclass(frozen=True)
class DinoV3Observation:
    result_index: int
    box_index: int
    embedding: np.ndarray
    accepted: bool
    species: str
    source: str
    registry_id: int | None
    registration_status: str | None
    known_score: float
    threshold: float
    detection_confidence: float

    def __post_init__(self):
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.shape != (DINO_FEATURE_DIM,):
            raise ValueError("Expected observation embedding with shape (768,)")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class DinoV3Prediction:
    species: str; accepted: bool; best_known_species: str; head_species: str; prototype_species: str
    head_prototype_consistent: bool; known_score: float; threshold: float
    candidates: tuple[dict[str,Any],...]; embedding: np.ndarray
    source: str="checkpoint"; registry_id: int|None=None; registration_status: str|None=None
    def as_candidate(self,*,detection_confidence=None):
        return {"name":self.species,"conf":self.known_score,"raw_cls_conf":self.known_score,
                "raw_det_conf":detection_confidence,"known_score":self.known_score,"threshold":self.threshold,
                "accepted":self.accepted,"head_species":self.head_species,
                "prototype_species":self.prototype_species,"head_prototype_consistent":self.head_prototype_consistent,
                "source":self.source,"registry_id":self.registry_id,"registration_status":self.registration_status}

class DinoV3Classifier:
    def __init__(self,checkpoint: DinoV3Checkpoint,*,encoder=None,registry=None):
        self.checkpoint=checkpoint; self.encoder=encoder; self.registry=registry
        self._weight=checkpoint.head_weight.numpy().astype(np.float32,copy=False)
        self._bias=checkpoint.head_bias.numpy().astype(np.float32,copy=False)
        self._prototypes=_normalize_rows(checkpoint.prototypes.numpy().astype(np.float32,copy=False))
        self.names={i:name for i,name in enumerate(checkpoint.classes)}; self.backend="dinov3"
    @property
    def classes(self): return self.checkpoint.classes
    def classify_features(self,features: np.ndarray) -> list[DinoV3Prediction]:
        array=np.asarray(features,dtype=np.float32)
        if array.ndim!=2 or array.shape[1]!=DINO_FEATURE_DIM: raise ValueError("Expected event features with shape (N, 768)")
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array,axis=1),1.0,atol=1e-5):
            raise ValueError("Expected finite L2-normalized event features")
        logits=array@self._weight.T+self._bias; sims=array@self._prototypes.T
        heads=logits.argmax(axis=1); nearest=sims.argmax(axis=1); out=[]
        for row,(hi,pi) in enumerate(zip(heads,nearest)):
            hi=int(hi); pi=int(pi); score=float(sims[row,pi]); consistent=hi==pi
            accepted=score>=self.checkpoint.threshold and consistent
            head=self.checkpoint.classes[hi]; proto=self.checkpoint.classes[pi]
            order=np.argsort(logits[row])[::-1][:3]
            candidates=tuple({"name":self.checkpoint.classes[int(i)],"logit":float(logits[row,int(i)]),
                              "prototype_score":float(sims[row,int(i)])} for i in order)
            prediction=DinoV3Prediction(head if accepted else "Unknown",accepted,proto,head,proto,consistent,score,
                                        self.checkpoint.threshold,candidates,array[row].copy())
            if not accepted and self.registry is not None:
                matched=self.registry.match(array[row])
                if matched is not None:
                    prediction=DinoV3Prediction(matched["display_name"],bool(matched["accepted"]),proto,head,proto,consistent,
                                                float(matched["score"]),float(matched["threshold"]),candidates,array[row].copy(),
                                                "registry",int(matched["id"]),str(matched["status"]))
            out.append(prediction)
        return out
    def classify_crops(self,crops: Sequence[ImageInput],*,array_color="rgb"):
        if self.encoder is None: raise RuntimeError("DINOv3 encoder is not loaded")
        return self.classify_features(self.encoder.encode(crops,array_color=array_color))
    def classify_event(self,crops: Sequence[ImageInput],*,array_color="rgb"):
        if self.encoder is None: raise RuntimeError("DINOv3 encoder is not loaded")
        event=aggregate_event_embeddings(self.encoder.encode(crops,array_color=array_color))
        return self.classify_features(event[None,:])[0]
