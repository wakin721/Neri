"""CUDA parity, cancellation and recovery for the production history adapter."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.dinov3 import history_acceleration as acceleration
from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history, unit
from system.dinov3.model import DinoV3RidgeWithinModel


def history_input(count=1025, dimensions=64):
    rng = np.random.default_rng(1511)
    raw = rng.normal(size=(count, dimensions)).astype(np.float32)
    raw *= rng.uniform(.1, 5, size=(count, 1)).astype(np.float32)
    raw[3] = 0
    history = rng.integers(0, count, size=(count, 64), dtype=np.int64)
    history[rng.random(history.shape) < .2] = -1
    history[::17] = -1
    return raw, history


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_optional_torch_missing_does_not_break_cpu_fallback(monkeypatch, device):
    raw, history = history_input()
    monkeypatch.setitem(sys.modules, "torch", None)
    expected = adapt_with_history(raw, history)
    actual = acceleration.adapt_history_for_job(raw, history, device=device)
    np.testing.assert_array_equal(actual, expected)


def test_small_and_cold_jobs_do_not_attempt_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CUDA should not be initialized")

    monkeypatch.setattr(acceleration, "_adapt_with_history_cuda", unexpected)
    raw, history = history_input(count=30)
    np.testing.assert_array_equal(acceleration.adapt_history_for_job(raw, history, device="cuda"),
                                  adapt_with_history(raw, history))
    raw, history = history_input()
    history[:] = -1
    actual = acceleration.adapt_history_for_job(raw, history, device="cuda")
    np.testing.assert_array_equal(actual, raw)
    assert not np.shares_memory(actual, raw)


@pytest.mark.parametrize("invalid", ["shape", "index", "nan", "temperature"])
def test_invalid_inputs_fail_before_cuda(monkeypatch, invalid):
    raw, history = history_input()
    kwargs = {}
    if invalid == "shape":
        history = history[:-1]
    elif invalid == "index":
        history[0, 0] = len(raw)
    elif invalid == "nan":
        raw[0, 0] = np.nan
    else:
        kwargs["temperature"] = 0
    monkeypatch.setattr(acceleration, "_adapt_with_history_cuda", lambda *a, **k: pytest.fail("invalid CUDA input"))
    with pytest.raises(ValueError):
        acceleration.adapt_history_for_job(raw, history, device="cuda", **kwargs)


def test_failed_cuda_recomputes_all_rows_and_keeps_progress_monotone(monkeypatch):
    raw, history = history_input()
    progress = []

    def fail(*args, progress_fn, **kwargs):
        progress_fn(512, len(raw))
        raise acceleration._CudaHistoryUnavailable("simulated failure after one chunk")

    monkeypatch.setattr(acceleration, "_adapt_with_history_cuda", fail)
    actual = acceleration.adapt_history_for_job(raw, history, device="cuda",
                                               progress_fn=lambda done, total: progress.append((done, total)))
    np.testing.assert_array_equal(actual, adapt_with_history(raw, history))
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert progress[-1] == (len(raw), len(raw))


def test_cancellation_is_never_retried_as_a_cuda_failure(monkeypatch):
    raw, history = history_input()

    class Cancelled(RuntimeError):
        pass

    def report(done, total):
        if done:
            raise Cancelled()

    def run(*args, progress_fn, **kwargs):
        progress_fn(512, len(raw))

    monkeypatch.setattr(acceleration, "_adapt_with_history_cuda", run)
    monkeypatch.setattr(acceleration, "adapt_with_history", lambda *a, **k: pytest.fail("cancelled job retried"))
    with pytest.raises(Cancelled):
        acceleration.adapt_history_for_job(raw, history, device="cuda", progress_fn=report)


def test_cuda_matches_cpu_under_ambient_fp16_and_restores_precision(cuda_torch):
    torch = cuda_torch
    raw, history = history_input(dimensions=768)
    expected = adapt_with_history(raw, history)
    previous = torch.get_float32_matmul_precision()
    progress = []
    try:
        torch.set_float32_matmul_precision("medium")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = acceleration.adapt_history_for_job(
                raw, history, device="cuda:0", progress_fn=lambda done, total: progress.append((done, total)),
            )
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-7)
    np.testing.assert_array_equal(actual[::17], raw[::17])
    assert progress[-1] == (len(raw), len(raw))


@pytest.mark.parametrize("recover", [True, False])
def test_cuda_oom_shrinks_chunks_or_falls_back(cuda_torch, monkeypatch, recover):
    torch = cuda_torch
    raw, history = history_input()
    original = acceleration._cuda_chunk
    chunks = []

    def run(torch_module, values, normalized, hh, begin, temperature, alpha):
        chunks.append(len(hh))
        if not recover or len(hh) > 128:
            raise torch.OutOfMemoryError("simulated allocation failure")
        return original(torch_module, values, normalized, hh, begin, temperature, alpha)

    monkeypatch.setattr(acceleration, "_cuda_chunk", run)
    actual = acceleration.adapt_history_for_job(raw, history, device="cuda")
    expected = adapt_with_history(raw, history)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-7)
    assert chunks[:3] == [512, 256, 128]
    if not recover:
        assert chunks[-1] == 32
        np.testing.assert_array_equal(actual, expected)


def test_low_vram_falls_back_before_upload(cuda_torch, monkeypatch):
    raw, history = history_input()
    monkeypatch.setattr(cuda_torch.cuda, "mem_get_info", lambda *a: (1, 1))
    actual = acceleration.adapt_history_for_job(raw, history, device="cuda")
    np.testing.assert_array_equal(actual, adapt_with_history(raw, history))


def test_real_cuda_cancellation_restores_precision_without_cpu_retry(cuda_torch, monkeypatch):
    raw, history = history_input()
    precision = cuda_torch.get_float32_matmul_precision()

    class Cancelled(RuntimeError):
        pass

    def progress(done, total):
        if done:
            raise Cancelled("cancel after the first completed CUDA chunk")

    monkeypatch.setattr(acceleration, "adapt_with_history", lambda *a, **k: pytest.fail("cancel retried on CPU"))
    with pytest.raises(Cancelled):
        acceleration.adapt_history_for_job(raw, history, device="cuda", progress_fn=progress)
    assert cuda_torch.get_float32_matmul_precision() == precision


def test_photo_finalizer_passes_encoder_device_to_history_adapter(monkeypatch, tmp_path):
    from system.backend.models import DetectionItem
    from system.dinov3 import processor
    from system.dinov3.classifier import DinoV3Prediction

    path = tmp_path / "photo.jpg"
    path.write_bytes(b"fixture")
    raw = unit(np.ones(768, np.float32))
    collector = processor.DinoV3CropCollector()
    collector.record_crop(processor.DinoV3CropRecord(str(path), 0, (0, 0, 10, 10), .9, raw))
    prediction = DinoV3Prediction("A", True, "A", 1., .1, raw, (), .8, .7)
    classifier = SimpleNamespace(backend="dinov3", encoder=SimpleNamespace(device="cuda:1"),
                                 classify_features=lambda *a, **k: [prediction])
    detector = SimpleNamespace(dinov3_classifier=classifier, _dinov3_collector=collector)
    devices = []

    def adapt(raw, history, *, device, **kwargs):
        devices.append(device)
        return adapt_with_history(raw, history, **kwargs)

    monkeypatch.setattr(processor, "adapt_history_for_job", adapt)
    item = DetectionItem(filename=path.name, path=str(path), file_type="image",
                         detection_data={"detection_completed": True, "detection_box_indices": [0]})
    result = processor.finalize_photo_job(detector, [item], tmp_path)
    assert devices == ["cuda:1"]
    assert result[0].detection_data["sequence_completed"] is True


def make_model(recipe, dimensions=64):
    rng = np.random.default_rng(512)
    centers = np.eye(dimensions, dtype=np.float32)[:3]
    training = unit(np.repeat(centers, 16, axis=0) + rng.normal(0, .02, (48, dimensions)).astype(np.float32))
    labels = np.repeat(["A", "B", "C"], 16)
    transform = np.diag(np.linspace(.5, 2, dimensions))
    model = DinoV3RidgeWithinModel(inference_recipe=recipe).fit(
        training, labels, [f"cam{i%3}" for i in range(48)], [f"s{i//2}" for i in range(48)],
        origin=np.zeros(dimensions), transform=transform,
    )
    unknown = unit(rng.normal(size=(24, dimensions)).astype(np.float32))
    calibration = np.r_[training, unknown]
    model.calibrate(calibration, np.r_[labels, ["U"] * 24], ["cal"] * 72,
                    [f"seq{i}" for i in range(72)])
    return model, training


@pytest.mark.parametrize("recipe", ["production_dual", "experiment_full"])
def test_reused_geometry_matches_original_evidence_and_builds_bank_once(monkeypatch, recipe):
    with threadpool_limits(limits=1):
        model, raw = make_model(recipe)
        adapted = raw * np.linspace(.5, 2, len(raw), dtype=np.float32)[:, None]
        expected_within = model._compute_within_scores(raw, adapted_features=adapted)
        expected_ridge = model.scores(raw, adapted_features=adapted)
        expected_norm = np.linalg.norm((np.asarray(adapted, np.float64)-model.origin_) @ model.transform_, axis=1)
        original_bank = model.within_updater.bank
        calls = []

        def bank():
            calls.append(1)
            return original_bank()

        monkeypatch.setattr(model.within_updater, "bank", bank)
        within, ridge, norm, _, _ = model._prediction_evidence(raw, adapted)
        np.testing.assert_array_equal(within, expected_within)
        np.testing.assert_array_equal(ridge, expected_ridge)
        np.testing.assert_array_equal(norm, expected_norm)
        calls.clear()
        model.predict(raw, ["cam"]*len(raw), [f"s{i//4}" for i in range(len(raw))], adapted_features=adapted)
        assert calls == [1]


@pytest.mark.parametrize("recipe", ["production_dual", "experiment_full"])
def test_cuda_preserves_classification_tracks_and_both_gates(cuda_torch, recipe):
    rng = np.random.default_rng(2077)
    with threadpool_limits(limits=1):
        model, training = make_model(recipe)
        count = 640
        raw = unit(training[np.arange(count) % len(training)] + rng.normal(0, .02, (count, 64)).astype(np.float32))
        raw[count//2:] = unit(rng.normal(size=(count//2, 64)).astype(np.float32))
        records = [dict(camera_id="cam", seq_id=f"event{i//16}",
                        datetime=1700000000+(i//16)*3600+i%16, content_hash=str(i)) for i in range(count)]
        history = FastCausalHistoryIndex(records).build()
        cpu = adapt_with_history(raw, history)
        gpu = acceleration.adapt_history_for_job(raw, history, device="cuda")
        metadata = dict(cameras=["cam"]*count, sequences=[f"event{i//16}" for i in range(count)],
                        contents=[f"frame{i//2}:{i%2}" for i in range(count)])
        expected = model.predict(raw, adapted_features=cpu, **metadata)
        actual = model.predict(raw, adapted_features=gpu, **metadata)
    for field in ("predictions", "accepted", "resolved_tracks", "box_accepted", "sequence_accepted"):
        if field in expected:
            np.testing.assert_array_equal(actual[field], expected[field])
    assert expected["accepted"].any() and not expected["accepted"].all()
    np.testing.assert_allclose(actual["known_score"], expected["known_score"], rtol=1e-4, atol=1e-4)


def test_batch_log_captures_sequence_backend_and_timing(tmp_path):
    import logging
    from system.backend.services_legacy import _start_batch_log_session

    session = _start_batch_log_session("sequence-test", log_dir=tmp_path)
    assert session is not None
    try:
        acceleration.adapt_history_for_job(np.ones((2, 16), np.float32), np.array([[-1], [0]]))
        logging.getLogger("system.dinov3.processor").info("DINOv3 sequence timing: test-stage")
    finally:
        session.close()
    text = session.path.read_text(encoding="utf-8")
    assert "history backend=cpu" in text and "sequence timing: test-stage" in text
