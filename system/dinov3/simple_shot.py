"""SimpleShot prototype helpers for dynamically registered species."""
from __future__ import annotations
import hashlib, random
from typing import Sequence, TypeVar
import numpy as np
T=TypeVar("T")
def normalize_embedding(value: np.ndarray)->np.ndarray:
    array=np.asarray(value,dtype=np.float32)
    if array.ndim!=1 or array.shape[0]!=768 or not np.isfinite(array).all(): raise ValueError("Expected a finite 768-dimensional embedding")
    norm=float(np.linalg.norm(array))
    if norm<=0: raise ValueError("Embedding must be non-zero")
    return array/norm
def build_prototype(embeddings: np.ndarray)->np.ndarray:
    array=np.asarray(embeddings,dtype=np.float32)
    if array.ndim!=2 or array.shape[1]!=768 or len(array)==0: raise ValueError("Expected embeddings with shape (N, 768)")
    return normalize_embedding(np.stack([normalize_embedding(row) for row in array]).mean(axis=0))
def cosine_similarity(embedding: np.ndarray,prototypes: np.ndarray)->np.ndarray:
    vector=normalize_embedding(embedding); matrix=np.asarray(prototypes,dtype=np.float32)
    if matrix.ndim!=2 or matrix.shape[1]!=768: raise ValueError("Expected prototypes with shape (N, 768)")
    return np.stack([normalize_embedding(row) for row in matrix])@vector
def deterministic_event_sample(values: Sequence[T],*,count:int,seed_material:str)->list[T]:
    items=list(values)
    if count<=0 or not items:return []
    rnd=random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8],"big")); idx=list(range(len(items))); rnd.shuffle(idx)
    return [items[i] for i in sorted(idx[:min(count,len(items))])]

def deterministic_two_means(embeddings: np.ndarray,*,iterations:int=12):
    array=np.asarray(embeddings,dtype=np.float32)
    if array.ndim!=2 or array.shape[1]!=768 or len(array)<8:return None
    normalized=np.stack([normalize_embedding(row) for row in array]); first=0; second=int(np.argmin(normalized@normalized[first])); centroids=np.stack([normalized[first],normalized[second]])
    labels=np.zeros(len(normalized),dtype=np.int64)
    for _ in range(max(1,iterations)):
        labels=np.argmax(normalized@centroids.T,axis=1)
        if min(np.bincount(labels,minlength=2))<4:return None
        nxt=np.stack([build_prototype(normalized[labels==i]) for i in range(2)])
        if np.allclose(nxt,centroids,atol=1e-6):centroids=nxt;break
        centroids=nxt
    if float(centroids[0]@centroids[1])>0.98:return None
    return centroids,labels
