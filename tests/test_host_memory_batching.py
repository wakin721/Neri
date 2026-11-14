from contextlib import closing
from types import SimpleNamespace
import struct
import threading
import weakref
import zlib

import numpy as np
import pytest

from system.host_memory import GIB, MIB, HostMemory, HostMemoryPressure, PhotoMemoryGuard
from system.backend.inference_batching import prefetched_path_batches, InferenceBatcher
from system.backend.models import ProcessingOptions


def header_png(path, width, height):
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name+data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b''))
    return path


def ample():
    return HostMemory(48*GIB, 24*GIB, 64*GIB)


def test_actual_48_megapixel_headers_limit_decode_before_pixels_are_loaded(tmp_path, monkeypatch):
    from PIL import Image
    path = header_png(tmp_path/'large.png', 9248, 5200)
    monkeypatch.setattr(Image.Image, 'load', lambda *a, **kw: pytest.fail('must not decode pixels'))
    guard = PhotoMemoryGuard(ample)
    assert guard.estimate(path) == 9248*5200*12
    count, cost = guard.choose([path]*32, 32, leave_lookahead=True)
    assert 1 <= count < 32 and cost <= guard.budget()
    ahead, ahead_cost = guard.choose([path]*32, 32, reserved=cost, optional=True)
    assert cost+ahead_cost <= min(8*GIB, int(48*GIB*.15))
    assert ahead < 32
    assert guard.decode_workers([path]*count) < 8
    small = header_png(tmp_path/'small.png', 1280, 720)
    assert guard.choose([small]*32, 32)[0] == 32


def test_commit_exhaustion_blocks_native_decode_even_with_free_physical_ram():
    guard = PhotoMemoryGuard(lambda: HostMemory(48*GIB, 24*GIB, 57*MIB))
    with pytest.raises(HostMemoryPressure, match='解码前停止'):
        list(prefetched_path_batches(['image.jpg'], lambda:32,
              lambda paths:pytest.fail('unsafe decoder called'), memory_guard=guard))


def test_oversized_header_cannot_bypass_guard_via_unknown_header_fallback(tmp_path):
    path=header_png(tmp_path/'oversized.png',100000,100000)
    with pytest.raises(HostMemoryPressure,match='像素数'):
        PhotoMemoryGuard(ample).choose([path],32)


def test_unknown_headers_use_single_photo_and_missing_memory_has_conservative_fallback(tmp_path):
    guard = PhotoMemoryGuard(lambda: None)
    paths = [tmp_path/f'{i}.unknown' for i in range(4)]
    assert guard.choose(paths, 32)[0] == 1
    assert guard.choose(paths, 32, reserved=GIB, optional=True) == (0, 0)


def test_mixed_sizes_split_before_large_photo_without_skipping_paths(tmp_path):
    small = header_png(tmp_path/'small.png', 1280, 720)
    large = header_png(tmp_path/'large.png', 9248, 5200)
    state = [ample()]
    guard = PhotoMemoryGuard(lambda:state[0])
    paths = [small]*6 + [large]*20 + [small]*6
    seen = []
    def preload(batch):
        seen.append(list(batch))
        return list(range(len(batch))), [], []
    with closing(prefetched_path_batches(paths, lambda:32, preload, memory_guard=guard)) as batches:
        out = []
        for group, future in batches:
            out.extend(group)
            assert future.done()
            future = None
    assert out == paths
    assert len(seen) > 1
    assert all(len(batch) < 32 for batch in seen)


def test_decode_allocation_failures_shrink_and_retry_without_loss():
    guard = PhotoMemoryGuard(ample)
    guard.estimate = lambda path:4*MIB
    calls = []
    def preload(paths):
        calls.append(list(paths))
        if len(paths)>2:
            raise MemoryError('synthetic native allocation failure')
        return list(range(len(paths))), [], []
    paths = [str(i) for i in range(8)]
    batches = list(prefetched_path_batches(paths, lambda:8, preload, memory_guard=guard))
    assert [len(batch) for batch in calls[:3]] == [8,4,2]
    assert [path for group, _ in batches for path in group] == paths
    assert all(len(group)==2 for group, _ in batches)


def test_only_current_and_one_lookahead_own_decoded_arrays():
    guard = PhotoMemoryGuard(ample)
    guard.estimate = lambda path:4*MIB
    lock = threading.Lock()
    live, peak = [0], [0]
    def released():
        with lock:
            live[0]-=1
    def frame():
        value=np.zeros((8,8,3), np.uint8)
        with lock:
            live[0]+=1
            peak[0]=max(peak[0],live[0])
        weakref.finalize(value, released)
        return value
    def preload(paths):
        return [0], [frame()], [frame()]
    with closing(prefetched_path_batches(list(map(str,range(20))),lambda:1,preload,
                                        memory_guard=guard)) as stream:
        for _, future in stream:
            assert future.result()[0]==[0]
            future=None
    assert peak[0]<=4
    assert live[0]==0


def test_queued_batch_rechecks_pressure_and_refuses_single_image():
    state=[ample()]
    guard=PhotoMemoryGuard(lambda:state[0])
    guard.estimate=lambda path:4*MIB
    def preload(paths):
        state[0]=HostMemory(48*GIB,24*GIB,57*MIB)
        return [0], [], []
    with pytest.raises(HostMemoryPressure):
        list(prefetched_path_batches(['one'],lambda:1,preload,memory_guard=guard))


def test_cancelled_prefetch_does_not_start_decoding():
    def cancelled():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        list(prefetched_path_batches(['one'],lambda:1,lambda paths:pytest.fail('decoded'),
                                    memory_guard=PhotoMemoryGuard(ample),check_cancelled=cancelled))


def test_cpu_allocation_error_during_inference_reduces_auto_batch(monkeypatch):
    from system.backend import inference_batching
    monkeypatch.setattr(inference_batching,'_automatic_batch_size',lambda options:8)
    calls=[]
    def detect(paths,**kwargs):
        calls.append(len(paths))
        if len(paths)>2:
            raise RuntimeError("OpenCV: Insufficient memory (Failed to allocate 100 bytes)")
        return [{'path':path} for path in paths]
    batcher=InferenceBatcher(ProcessingOptions())
    output=list(batcher.detect(SimpleNamespace(detect_batch_species=detect),list(map(str,range(8)))))
    assert calls==[8,4,2,2,2,2]
    assert [x['path'] for batch in output for x in batch.detections]==list(map(str,range(8)))


def test_image_predictor_releases_arrays_without_unloading_models():
    from system.image_processor_legacy import ImageProcessor
    processor=ImageProcessor(None)
    model=SimpleNamespace(predictor=SimpleNamespace(results=[object()],batch=object(),dataset=object()),weights=object())
    processor.model=model
    processor.release_batch_images()
    assert processor.model is model and processor.model.weights is model.weights
    assert model.predictor.results==[] and model.predictor.batch is None and model.predictor.dataset is None


def test_decoder_and_crop_do_not_silently_swallow_allocation_errors(monkeypatch):
    from system import image_processor_legacy as legacy
    processor=legacy.ImageProcessor(None)
    monkeypatch.setattr(legacy.cv2,'imread',lambda path: (_ for _ in ()).throw(MemoryError('decode')))
    with pytest.raises(MemoryError):
        processor.preload_batch_data(['fixture.jpg'])
    monkeypatch.setattr(legacy.cv2,'copyMakeBorder',lambda *a,**kw: (_ for _ in ()).throw(MemoryError('crop')))
    box=SimpleNamespace(xyxy=np.asarray([[1,1,7,3]],dtype=float))
    with pytest.raises(MemoryError):
        processor._crop_single_box((0,0,box,np.zeros((8,8,3),np.uint8)))


def test_low_memory_job_preserves_progress_and_resumes_without_reprocessing(monkeypatch, tmp_path):
    from system.backend import services, inference_batching
    from system.backend.models import CreateJobRequest, DetectionItem, JobSummary, JobState
    legacy=services._legacy
    for module in (services,legacy):
        monkeypatch.setattr(module,'job_state_path',lambda:tmp_path/'state.json')
    photos=[tmp_path/'a.jpg',tmp_path/'b.jpg']
    for path in photos: path.write_bytes(b'fixture')
    state=[ample()]
    batcher=InferenceBatcher(ProcessingOptions(batch_size=1))
    batcher.host_guard=PhotoMemoryGuard(lambda:state[0])
    monkeypatch.setattr(legacy,'InferenceBatcher',lambda opts:batcher)
    monkeypatch.setattr(legacy,'_start_batch_log_session',lambda *a:None)
    monkeypatch.setattr(legacy,'_resolve_job_inputs',lambda *a,**kw:(tmp_path,photos))
    monkeypatch.setattr(legacy,'load_media_indexes',lambda *a,**kw:({},{}))
    monkeypatch.setattr(legacy,'_build_metadata_item',lambda p:DetectionItem(path=str(p),filename=p.name,file_type='jpg'))
    detector=SimpleNamespace(preload_batch_data=lambda paths:([],[],[]),_photo_memory_guard=None,
                             cleanup_runtime_cache=lambda **kw:None)
    monkeypatch.setattr(legacy,'_load_detector',lambda *a:detector)
    calls=[]
    def detect(detector,paths,items,*args,**kw):
        calls.extend(paths)
        if len(calls)==1: state[0]=HostMemory(48*GIB,24*GIB,57*MIB)
        return items
    monkeypatch.setattr(legacy,'_detect_image_batch',detect)
    request=CreateJobRequest(input_dir=str(tmp_path),options=ProcessingOptions(
        enable_detection=True,model_path='detector.pt',batch_size=1,video_mode='skip'))
    manager=services.ProcessingJobManager()
    now=services.utc_now()
    manager._jobs['ram']=JobSummary(id='ram',state=JobState.QUEUED,input_dir=str(tmp_path),created_at=now,updated_at=now)
    manager._job_requests['ram']=request
    try:
        manager._run_job('ram',request,[])
        failed=manager.get_job('ram')
        assert failed.state==JobState.FAILED and failed.processed==1
        assert failed.message=='内存不足，释放内存后可继续任务'
        assert [item.path for item in failed.results]==[str(photos[0])]
        state[0]=ample()
        monkeypatch.setattr(manager._executor,'submit',lambda fn,*args:fn(*args))
        manager.resume_job('ram')
        completed=manager.get_job('ram')
        assert completed.state==JobState.COMPLETED and completed.processed==2
        assert calls==photos
    finally:
        manager._executor.shutdown()
