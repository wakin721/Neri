from contextlib import closing
from types import SimpleNamespace

import pytest
import torch

from system.adaptive_batch import AcceleratorMemory, AdaptiveBatchSize
from system.backend import inference_batching
from system.backend.models import ProcessingOptions


GIB = 1024**3
BEFORE = AcceleratorMemory(14 * GIB, 16 * GIB, GIB, GIB)
AFTER = AcceleratorMemory(12 * GIB, 16 * GIB, GIB, 2 * GIB)


def test_tuner_ignores_warmup_and_keeps_faster_larger_batches():
    tuner = AdaptiveBatchSize(2, maximum=8, enabled=True, label="test")
    tuner.observe(2, 10.0, BEFORE, AFTER)
    for _ in range(3):
        tuner.observe(2, 0.02, BEFORE, AFTER)
    assert tuner.size == 4
    for _ in range(3):
        tuner.observe(4, 0.03, BEFORE, AFTER)
    assert tuner.size == 8
    for _ in range(10):
        tuner.observe(8, 0.05, BEFORE, AFTER)
    assert tuner.size == 8


def test_tuner_reverts_slower_trial_and_does_not_oscillate():
    tuner = AdaptiveBatchSize(2, maximum=32, enabled=True, label="test")
    for _ in range(4):
        tuner.observe(2, 0.02, BEFORE, AFTER)
    assert tuner.size == 4
    for _ in range(3):
        tuner.observe(4, 0.08, BEFORE, AFTER)
    assert tuner.size == 2
    for _ in range(20):
        tuner.observe(2, 0.01, BEFORE, AFTER)
    assert tuner.size == 2


@pytest.mark.parametrize("condition", ["memory", "unknown", "partial", "manual"])
def test_tuner_does_not_grow_without_safe_representative_samples(condition):
    tuner = AdaptiveBatchSize(2, maximum=32, enabled=condition != "manual", label="test")
    after = AcceleratorMemory(100 * 1024**2, 16 * GIB, GIB, 2 * GIB) if condition == "memory" else AFTER
    if condition == "unknown":
        after = None
    for _ in range(12):
        tuner.observe(1 if condition == "partial" else 2, 0.02, BEFORE, after)
    assert tuner.size == 2


def test_outer_inference_changes_batch_size_without_reordering(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(inference_batching, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(inference_batching, "_automatic_batch_size", lambda _: 2)
    monkeypatch.setattr(inference_batching, "accelerator_memory", lambda: AFTER)
    calls = []

    def detect(paths, **kwargs):
        calls.append(len(paths))
        clock[0] += len(paths) * 0.01
        return [{"path": path} for path in paths]

    paths = [str(index) for index in range(36)]
    batcher = inference_batching.InferenceBatcher(ProcessingOptions())
    batches = list(batcher.detect(SimpleNamespace(detect_batch_species=detect), paths))
    assert calls[:4] == [2] * 4
    assert 4 in calls and 8 in calls
    assert [row["path"] for batch in batches for row in batch.detections] == paths


def test_photo_prefetch_adopts_new_size_with_bounded_lookahead():
    size = [1]
    paths = list(range(30))
    with closing(inference_batching.prefetched_path_batches(paths, lambda: size[0])) as stream:
        first, _ = next(stream)
        assert first == [0]
        size[0] = 4
        batches = [first] + [batch for batch, _ in stream]
    assert [item for batch in batches for item in batch] == paths
    assert any(len(batch) == 4 for batch in batches)
    assert sum(len(batch) == 1 for batch in batches) <= 4


def test_runtime_gc_is_periodic_and_oom_or_final_cleanup_is_immediate(monkeypatch):
    from system import image_processor_legacy as legacy

    now, collected = [100.0], []
    monkeypatch.setattr(legacy.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(legacy.gc, "collect", lambda: collected.append(now[0]))
    monkeypatch.setattr(legacy.torch.cuda, "is_available", lambda: False)
    processor = legacy.ImageProcessor.__new__(legacy.ImageProcessor)
    for _ in range(100):
        processor.cleanup_runtime_cache(force_gc=False)
    assert not collected
    now[0] += 60
    processor.cleanup_runtime_cache(force_gc=False)
    assert len(collected) == 1
    processor.cleanup_runtime_cache(clear_cuda_cache=True, force_gc=False)
    processor.cleanup_runtime_cache()
    assert len(collected) == 3
