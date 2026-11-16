"""Compare pre-optimization Git sources and integrated CPU/CUDA aggregation.

Run from any directory: python scripts/benchmark_sequence_speed.py --repeats 3
Synthetic inputs and scratch databases only; no model or source media is changed.
CUDA history adaptation uses the same adapter as production photo finalization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gc
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from system import detection_db
from system.backend.job_state_store import JobStateStore
from system.backend.models import DetectionItem, JobState, JobSummary
from system.dinov3 import context, processor
from system.dinov3.history_acceleration import adapt_history_for_job


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def baseline(path, revision):
    source = git("show", f"{revision}:{path}")
    module = types.ModuleType("_sequence_before_" + Path(path).stem)
    module.__file__ = str(ROOT / path)
    module.__package__ = "system.dinov3" if "/dinov3/" in path else "system.backend"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def time_call(function):
    gc.collect()
    start = time.perf_counter()
    value = function()
    return time.perf_counter() - start, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/benchmarks/sequence-speed-20261115.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    before_context = baseline("system/dinov3/context.py", args.baseline)
    before_db = baseline("system/detection_db.py", args.baseline)
    before_store = baseline("system/backend/job_state_store.py", args.baseline)
    rng = np.random.default_rng(20261115)
    report = dict(baseline_commit=git("rev-parse", args.baseline).strip(),
                  repeats=args.repeats, seed=20261115,
                  environment=dict(python=sys.version, executable=sys.executable,
                                   platform=platform.platform(), numpy=np.__version__,
                                   sqlite=sqlite3.sqlite_version, cpu_threads=4),
                  source_sha256={}, results=[], limitations=[
                      "Synthetic workloads, not a complete inference job or real photo dataset.",
                      "Local SQLite files and OS-cached generated PNG files; no cold-disk or SMB measurement.",
                      "CUDA results measure the production history adapter, not complete model inference.",
                      "CUDA timings include feature/index uploads and result download; initialization is warmed up.",
                  ])
    for path in ("system/dinov3/context.py", "system/dinov3/processor.py",
                 "system/detection_db.py", "system/backend/job_state_store.py"):
        report["source_sha256"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

    def record(name, old_times, new_times, **metadata):
        old, new = statistics.median(old_times), statistics.median(new_times)
        row = dict(name=name, before_seconds=old, after_seconds=new,
                   speedup=old / new, reduction_percent=100 * (1 - new / old),
                   before_runs=old_times, after_runs=new_times, **metadata)
        report["results"].append(row)
        print(f"{name}: {old:.6f}s -> {new:.6f}s ({old/new:.2f}x)", flush=True)

    def compare(name, old, new, check, **metadata):
        times = [[], []]
        for trial in range(args.repeats):
            values = [None, None]
            for side in ((0, 1) if trial % 2 == 0 else (1, 0)):
                elapsed, values[side] = time_call((old, new)[side])
                times[side].append(elapsed)
            check(*values)
        record(name, *times, output_parity=True, **metadata)

    with threadpool_limits(limits=4):
        report["environment"]["threadpools"] = threadpool_info()
        for count in (1000, 4000):
            records = [dict(camera_id="camera", seq_id=f"event-{i//100}",
                            datetime=1700000000+(i//100)*3600+i%100, content_hash=str(i))
                       for i in range(count)]
            compare(f"history_{count}",
                    lambda: before_context.FastCausalHistoryIndex(records).build().indices,
                    lambda: context.FastCausalHistoryIndex(records).build().indices,
                    np.testing.assert_array_equal, includes_index_construction=True,
                    crops=count, events=count // 100)

        vectors = rng.normal(size=(3, 768)).astype(np.float32)
        records = [dict(camera_id="camera", path=f"{frame}.jpg", box_index=box,
                        bbox=[box*100, 0, box*100+50, 50], date_taken=1700000000+frame,
                        raw_embedding=vectors[box]) for frame in range(100) for box in range(3)]

        def check_groups(a, b):
            fields = ("path", "box_index", "event_id", "track_id", "sequence_id", "datetime")
            assert [tuple(r[f] for f in fields) for r in a] == [tuple(r[f] for f in fields) for r in b]

        compare("group_300_crops", lambda: before_context.group_job_events(records),
                lambda: context.group_job_events(records), check_groups, frames=100, crops=300, dimensions=768)

        raw = rng.normal(size=(4000, 768)).astype(np.float32)
        history = rng.integers(0, len(raw), size=(len(raw), 64), dtype=np.int64)
        history[rng.random(history.shape) < .1] = -1
        parity = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        compare("adapt_4000", lambda: before_context.adapt_with_history(raw, history),
                lambda: context.adapt_with_history(raw, history), parity, crops=4000, dimensions=768, history=64)

        scratch_parent = ROOT / "temp"
        scratch_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sequence-speed-", dir=scratch_parent) as temporary:
            scratch = Path(temporary).resolve()
            assert scratch.is_relative_to(scratch_parent.resolve())
            boxes = [dict(**{"物种": "deer", "边界框": [12, 24, 120, 240]},
                          confidence=.83, accepted=True,
                          **{"候选项": [dict(name=f"species-{j}", conf=.8-j*.05,
                                            known_score=.5, threshold=.2, source="dinov3_ridge_within")
                                       for j in range(5)]}) for _ in range(3)]
            pending = dict(**{"检测框": boxes}, sequence_completed=False, sequence_pending=True, sequence_status="pending")
            completed = {**pending, "sequence_completed": True, "sequence_pending": False, "sequence_status": "complete"}
            initial = [DetectionItem(filename=f"{i}.jpg", path=f"/fixture/{i}.jpg", file_type="image",
                                     detection_boxes=boxes, detection_data=pending) for i in range(4000)]
            finished = [item.model_copy(update={"detection_data": completed}) for item in initial]

            for changed in ("all", "one_percent"):
                times = [[], []]
                for trial in range(args.repeats):
                    outputs = [None, None]
                    for side in ((0, 1) if trial % 2 == 0 else (1, 0)):
                        cls = (before_store.JobStateStore, JobStateStore)[side]
                        store = cls(scratch / f"state-{changed}-{trial}-{side}.json")
                        job = JobSummary(id="fixture", state=JobState.RUNNING, input_dir=str(scratch),
                                         created_at="2026-11-15T00:00:00Z", updated_at="2026-11-15T00:00:00Z", results=initial)
                        store.save({job.id: job}, {})
                        results = finished if changed == "all" else [
                            finished[i] if i % 100 == 0 else item.model_copy() for i, item in enumerate(initial)]
                        job = job.model_copy(update={"state": JobState.COMPLETED, "results": results})
                        elapsed, _ = time_call(lambda: store.save({job.id: job}, {}))
                        times[side].append(elapsed)
                        outputs[side] = store.load()
                    assert outputs[0] == outputs[1]
                record(f"job_store_4000_{changed}_changed", *times, output_parity=True,
                       crops=12000, photos=4000, changed_fraction=1 if changed == "all" else .01,
                       all_result_objects_replaced=True, initial_checkpoint_excluded=True)

            grouped = [[(str(i), f"{i}.jpg", completed) for i in range(group, 4096, 16)] for group in range(16)]
            times = [[], []]
            for trial in range(args.repeats):
                outputs = [None, None]
                for side in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    module = (before_db, detection_db)[side]
                    folder = scratch / f"media-{trial}-{side}"
                    folder.mkdir()
                    paths = [str(folder / f"{i}.sqlite3") for i in range(16)]
                    for path, rows in zip(paths, grouped):
                        module.init_db(path)
                        module.upsert_detections_bulk(path, [(base, name, pending) for base, name, _ in rows])
                        module.upsert_validation_bulk(path, [(name, False) for _, name, _ in rows])

                    def save_media():
                        for chunk in range(0, 256, 32):
                            for path, rows in zip(paths, grouped):
                                rows = rows[chunk:chunk+32]
                                validations = [(name, False) for _, name, _ in rows]
                                if side == 0:
                                    module.init_db(path)
                                    module.upsert_detections_bulk(path, rows)
                                    module.upsert_validation_bulk(path, validations)
                                else:
                                    module.upsert_media_batch(path, rows, validations)

                    elapsed, _ = time_call(save_media)
                    times[side].append(elapsed)
                    outputs[side] = []
                    for path in paths:
                        with closing(sqlite3.connect(path)) as conn:
                            rows = conn.execute("SELECT base_name,image_filename,detection_json FROM detections ORDER BY base_name").fetchall()
                            validations = conn.execute("SELECT * FROM validation ORDER BY image_filename").fetchall()
                        outputs[side].append(([(base, name, json.loads(data)) for base, name, data in rows], validations))
                assert outputs[0] == outputs[1]
            record("media_db_4096_16_directories", *times, output_parity=True, photos=4096,
                   directories=16, chunk_size=512, before_connections=384, after_connections=128,
                   filesystem_checks_excluded=True, initial_detections_excluded=True)

            import cv2
            photos = []
            for i in range(64):
                image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
                path = scratch / f"photo-{i}.png"
                path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
                photos.append(path)

            def decode_before():
                with ThreadPoolExecutor(max_workers=8) as executor:
                    images = list(executor.map(lambda p: cv2.imread(str(p)), photos))
                return images, [processor._image_file_digest(p) for p in photos]

            def decode_after():
                collector = processor.DinoV3CropCollector()
                with ThreadPoolExecutor(max_workers=8) as executor:
                    images = list(executor.map(lambda p: processor.read_image_with_digest(p, collector), photos))
                return images, [collector.image_digest(p) for p in photos]

            def check_decode(a, b):
                assert a[1] == b[1]
                for old, new in zip(a[0], b[0]):
                    np.testing.assert_array_equal(old, new)

            compare("decode_and_hash_64_png", decode_before, decode_after, check_decode,
                    photos=64, bytes=sum(p.stat().st_size for p in photos), decode_workers=8,
                    includes_initial_hash_cost=True, os_cache="warm")

        # Measure the actual optional backend used by photo finalization.
        try:
            import torch
        except ImportError:
            report["cuda_probe"] = "PyTorch unavailable"
        else:
            report["environment"]["torch"] = torch.__version__
            if not torch.cuda.is_available():
                report["cuda_probe"] = "CUDA unavailable"
            else:
                report["environment"].update(gpu=torch.cuda.get_device_name(),
                    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    tf32=torch.backends.cuda.matmul.allow_tf32)
                torch.set_num_threads(4)

                def cuda_adapt(features, indices):
                    return adapt_history_for_job(features, indices, device="cuda")

                cuda_adapt(raw[:512], np.zeros((512, 64), dtype=np.int64))
                for count in (1000, 10000):
                    features = context.unit(rng.normal(size=(count, 768)).astype(np.float32))
                    indices = rng.integers(0, count, size=(count, 64), dtype=np.int64)
                    indices[rng.random(indices.shape) < .1] = -1
                    indices[0] = -1
                    errors = []

                    def check_cuda(a, b):
                        np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-7)
                        errors.append(float(np.max(np.abs(a-b))))

                    compare(f"cuda_integrated_adapt_{count}",
                            lambda: context.adapt_with_history(features, indices),
                            lambda: cuda_adapt(features, indices), check_cuda,
                            production_backend=True, baseline="optimized_CPU", dtype="float32",
                            cpu_chunk=128, cuda_chunk=512, includes_transfers=True,
                            parity_tolerance=dict(rtol=2e-5, atol=3e-7))
                    report["results"][-1]["max_absolute_error"] = max(errors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
