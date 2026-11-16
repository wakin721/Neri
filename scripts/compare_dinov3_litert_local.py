"""Re-encode the frozen local E split and compare LiteRT to original DINOv3.

Does not train, recalibrate, modify source photographs, or install models.
Outputs are resumable and include per-crop decisions and paired disagreements.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.compare_local_dinov3_protocols import metrics
from scripts.export_dinov3_litert import LiteRtDinoV3Encoder, _preprocess_nhwc_batch
from system.dino.preprocess import preprocess_image
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.context import FastCausalHistoryIndex, adapt_with_history
from system.dinov3.encoder import DinoV3Encoder, sha256
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.runtime import read_manifest, resolve_manifest_asset


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf8')
    temporary.replace(path)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def timed(fn, repeats=3):
    fn()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return dict(seconds=durations, median_seconds=statistics.median(durations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=ROOT/'runs/local_protocol_comparison_20261114')
    parser.add_argument('--manifest', type=Path, default=ROOT/'res/model/cls/user/dinov3-experiment-local.neri.json')
    parser.add_argument('--litert', type=Path, default=ROOT/'runs/Neri1.2preview-litert')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/Neri1.2preview-litert-local-full-20261115')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--speed-count', type=int, default=64)
    parser.add_argument('--stage', choices=['all', 'extract', 'evaluate', 'speed'], default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    payload = read_manifest(args.manifest)
    lite_payload = json.loads((args.litert/'Neri1.2preview-litert.json').read_text('utf8'))
    checkpoint = resolve_manifest_asset(args.manifest, payload['checkpoint'])
    assert sha256(checkpoint) == payload['checkpoint_sha256'] == lite_payload['checkpoint_sha256']
    assert sha256(args.litert/'model.neri-model') == payload['checkpoint_sha256']
    for name, field in [('litert_fp32', 'encoder_fp32'), ('litert_fp16_weights', 'encoder_fp16_weights')]:
        assert sha256(args.litert/lite_payload[field]) == lite_payload[field+'_sha256']
    split = json.loads((args.comparison/'partition.json').read_text('utf8'))
    rows_file = args.comparison/'current_reviewed.jsonl'
    assert sha256(rows_file) == split['manifest_sha256']
    rows_all = [json.loads(line) for line in rows_file.read_text('utf8').splitlines()]
    indices = np.array(sorted(int(k) for k, v in split['assignments'].items()
                              if v in {'evaluation', 'evaluation_unknown'}), dtype=np.int64)
    rows = [rows_all[int(i)] for i in indices]
    paths = [r['file'] for r in rows]
    truth = np.array([r['species'] for r in rows])
    cameras = np.array([r['camera'] for r in rows])
    sequences = np.array([r['sequence'] for r in rows])
    contents = np.array([r['sha256'] for r in rows])
    records = [dict(camera_component=r['camera'], seq_id=r['sequence'], datetime=r['datetime'],
                    content_hash=r['source_sha256'] or r['sha256']) for r in rows]
    signature = dict(partition_sha256=sha256(args.comparison/'partition.json'),
                     manifest_sha256=sha256(args.manifest), rows_sha256=sha256(rows_file),
                     lite_manifest_sha256=sha256(args.litert/'Neri1.2preview-litert.json'),
                     script_sha256=sha256(__file__), rows=len(rows), batch_size=args.batch_size,
                     threads=args.threads, source_sha256={p:sha256(ROOT/p) for p in (
                         'system/dinov3/encoder.py','system/dino/preprocess.py','system/dinov3/context.py',
                         'system/dinov3/model.py','system/dinov3/experiment.py','system/dinov3/classifier.py',
                         'scripts/export_dinov3_litert.py')})
    identity = args.output/'identity.json'
    if identity.exists():
        old = json.loads(identity.read_text('utf8'))
        assert old == signature, 'Run configuration changed; choose a new output directory'
    else:
        write(identity, signature)
    report_path = args.output/'report.json'
    report = json.loads(report_path.read_text('utf8')) if report_path.exists() else {}
    report.update(identity=signature, environment=dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
        torch=torch.__version__, numpy=np.__version__, litert=importlib.metadata.version('ai-edge-litert'),
        cpu_threads=args.threads, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        protocol=dict(split='fixed camera-held-out E', calibration='original frozen; never refit',
                      history='strictly past, same-camera, E-only; rebuild independently per encoder',
                      aggregation='complete original events, crop hashes for fusion deduplication',
                      scope='existing animal crops, no detector, UI, SQLite or automatic learning',
                      speed='warm runs, OS cache uncontrolled; one backend at a time'))
    write(report_path, report)
    variants = ['pytorch_original', 'pytorch_gpu', 'litert_fp32', 'litert_fp16_weights']
    if not torch.cuda.is_available():
        variants.remove('pytorch_gpu')

    def make_encoder(variant, device=None, batch_size=None):
        size = batch_size or args.batch_size
        if variant.startswith('pytorch'):
            return DinoV3Encoder(resolve_manifest_asset(args.manifest, payload['encoder_weights']),
                resolve_manifest_asset(args.manifest, payload['encoder_code']), payload['encoder_sha256'],
                device=device or ('cuda' if variant=='pytorch_gpu' else 'cpu'), batch_size=size)
        field = 'encoder_fp32' if variant == 'litert_fp32' else 'encoder_fp16_weights'
        return LiteRtDinoV3Encoder(args.litert/lite_payload[field], num_threads=args.threads, batch_size=size)

    def forward(encoder, data):
        if isinstance(encoder, DinoV3Encoder):
            return encoder.encode_preprocessed(data)
        return encoder.encode_nhwc(data)

    if args.stage in {'all', 'extract'}:
        if not report.get('input_content_verified'):
            log(f'Auditing content of {len(rows)} real E crops')
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, actual_hash in enumerate(pool.map(sha256, paths)):
                    assert actual_hash==contents[i], f'Source crop changed: {paths[i]}'
                    if i%512==0 or i+1==len(rows):
                        log(f'Content audit: {i+1}/{len(rows)}')
        report['input_content_verified'] = True
        input_cache=None
        cache_info_path=args.output/'input_cache.json'
        if cache_info_path.exists():
            cache_info=json.loads(cache_info_path.read_text('utf8'))
            assert cache_info['completed']==len(rows) and cache_info['rows_sha256']==signature['rows_sha256']
            assert cache_info['partition_sha256']==signature['partition_sha256']
            input_cache=np.load(args.output/'preprocessed_nhwc.npy',mmap_mode='r')
            assert input_cache.shape==(len(rows),224,224,3) and input_cache.dtype==np.float32
            report['input_cache']=cache_info
            report['protocol']['extraction_input']='shared bitwise-verified preprocessed cache; CPU prefix decoded before cache was prepared'
        write(report_path,report)
        for variant in variants:
            progress_path = args.output/(variant+'_progress.json')
            progress = json.loads(progress_path.read_text('utf8')) if progress_path.exists() else dict(completed=0, preprocess_seconds=0., encoder_seconds=0.)
            feature_path = args.output/(variant+'_features.npy')
            if progress['completed'] == len(rows):
                log(f'{variant}: reusing complete verified feature output')
                continue
            start = time.perf_counter()
            log(f'{variant}: loading encoder')
            encoder = make_encoder(variant)
            initialization = time.perf_counter() - start
            preprocessor = preprocess_image if isinstance(encoder, DinoV3Encoder) else lambda p: _preprocess_nhwc_batch([p])[0]
            # Warm the actual batch shape before measured extraction.
            warm = [preprocessor(p) for p in paths[:args.batch_size]]
            forward(encoder, warm if isinstance(encoder, DinoV3Encoder) else np.stack(warm))
            out = np.lib.format.open_memmap(feature_path, mode='r+' if feature_path.exists() else 'w+',
                                            dtype=np.float32, shape=(len(rows), 768))
            with threadpool_limits(limits=args.threads), ThreadPoolExecutor(max_workers=4) as pool:
                for begin in range(progress['completed'], len(rows), 128):
                    stop = min(begin+128, len(rows))
                    start = time.perf_counter()
                    if input_cache is not None:
                        chunk=np.array(input_cache[begin:stop],copy=True)
                        data=list(torch.from_numpy(chunk.transpose(0,3,1,2))) if isinstance(encoder,DinoV3Encoder) else chunk
                    else:
                        data = list(pool.map(preprocessor, paths[begin:stop]))
                        if not isinstance(encoder, DinoV3Encoder):
                            data = np.stack(data)
                    progress['preprocess_seconds'] += time.perf_counter()-start
                    start = time.perf_counter()
                    encoded = forward(encoder, data)
                    progress['encoder_seconds'] += time.perf_counter()-start
                    assert encoded.shape == (stop-begin, 768) and np.isfinite(encoded).all()
                    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-5)
                    out[begin:stop] = encoded
                    out.flush()
                    progress.update(completed=stop, initialization_seconds=initialization,
                                    device=getattr(encoder, 'device', 'cpu'))
                    write(progress_path, progress)
                    if begin//128 % 8 == 0 or stop == len(rows):
                        log(f"{variant}: {stop}/{len(rows)}, encoder {progress['encoder_seconds']:.1f}s, decode/preprocess {progress['preprocess_seconds']:.1f}s")
            report.setdefault('full_extraction', {})[variant] = progress
            write(report_path, report)
            del encoder, out, warm, data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.stage in {'all', 'evaluate'}:
        model = DinoV3RidgeWithinModel.load(checkpoint)
        classifier = DinoV3Classifier(model)
        start = time.perf_counter()
        history = FastCausalHistoryIndex(records).build()
        report['history_index_seconds'] = time.perf_counter()-start
        predictions = {}
        with threadpool_limits(limits=4):
            for variant in variants:
                features = np.load(args.output/(variant+'_features.npy'), allow_pickle=False)
                progress = json.loads((args.output/(variant+'_progress.json')).read_text('utf8'))
                assert progress['completed'] == len(rows)
                start = time.perf_counter()
                adapted = adapt_with_history(features, history)
                history_time = time.perf_counter()-start
                start = time.perf_counter()
                pred = classifier.classify_features(features, camera_ids=cameras, sequence_ids=sequences,
                                                    content_ids=contents, adapted_features=adapted)
                classify_time = time.perf_counter()-start
                species = np.array([p.best_known_species for p in pred])
                accepted = np.array([p.accepted for p in pred])
                scores = np.array([p.known_score for p in pred])
                predictions[variant] = dict(species=species, accepted=accepted, scores=scores, features=features)
                np.savez_compressed(args.output/(variant+'_decisions.npz'), indices=indices,
                                    species=species, accepted=accepted, known_score=scores, truth=truth)
                entry = dict(metrics=metrics(truth,species,accepted,model.classes), history_seconds=history_time,
                             classification_seconds=classify_time, threshold=float(model.threshold),
                             events=len(set(zip(cameras.tolist(),sequences.tolist()))))
                if variant != 'pytorch_original':
                    base = predictions['pytorch_original']
                    diff = np.abs(features-base['features'])
                    cos = (features.astype(np.float64)*base['features']).sum(1)
                    changes = (species != base['species']) | (accepted != base['accepted'])
                    score_diff = np.abs(scores-base['scores'])
                    changed_events = set(zip(cameras[changes].tolist(), sequences[changes].tolist()))
                    entry['vs_original'] = dict(species_changes=int((species!=base['species']).sum()),
                        acceptance_changes=int((accepted!=base['accepted']).sum()),
                        final_output_changes=int(((np.where(accepted,species,'__unknown__')) !=
                                                  np.where(base['accepted'],base['species'],'__unknown__')).sum()),
                        changed_events=len(changed_events), max_feature_abs_diff=float(diff.max()),
                        mean_feature_abs_diff=float(diff.mean()), min_cosine=float(cos.min()),
                        median_cosine=float(np.median(cos)), max_known_score_diff=float(score_diff.max()),
                        p99_known_score_diff=float(np.quantile(score_diff,.99)))
                    for group_name, mask in [('known',np.isin(truth,model.classes)),('unknown',~np.isin(truth,model.classes))]:
                        event_keys=set(zip(cameras[mask].tolist(),sequences[mask].tolist()))
                        entry['vs_original'][group_name]=dict(images=int(mask.sum()),events=len(event_keys),
                            species_changes=int(((species!=base['species'])&mask).sum()),
                            acceptance_changes=int(((accepted!=base['accepted'])&mask).sum()),
                            changed_events=len(set(zip(cameras[changes&mask].tolist(),sequences[changes&mask].tolist()))))
                    changed_rows = []
                    for i in np.flatnonzero(changes):
                        changed_rows.append(dict(row_index=int(indices[i]), file=paths[i], truth=truth[i],
                            camera=cameras[i], sequence=sequences[i], original_species=base['species'][i],
                            lite_species=species[i], original_accepted=bool(base['accepted'][i]),
                            lite_accepted=bool(accepted[i]), original_score=float(base['scores'][i]),
                            lite_score=float(scores[i]), threshold=float(model.threshold)))
                    write(args.output/(variant+'_disagreements.json'),changed_rows)
                report.setdefault('quality',{})[variant]=entry
                write(report_path,report)
                log(f'{variant} quality: {json.dumps(entry,ensure_ascii=False)}')
                del pred, adapted
            frozen_path = args.comparison/'camera_split/experiment_full.npz'
            frozen_hashes = json.loads((args.comparison/'camera_split/decisions_frozen.json').read_text('utf8'))
            assert sha256(frozen_path) == frozen_hashes['files'][frozen_path.name]
            with np.load(frozen_path,allow_pickle=False) as frozen:
                np.testing.assert_array_equal(indices,frozen['e_indices'])
                base=predictions['pytorch_original']
                report['original_vs_frozen_reference']=dict(species_changes=int((base['species']!=frozen['e_pred']).sum()),
                    acceptance_changes=int((base['accepted']!=frozen['e_accepted']).sum()),
                    max_score_diff=float(abs(base['scores']-frozen['e_score']).max()))
            write(report_path,report)
        for variant in variants:
            progress_path=args.output/(variant+'_progress.json')
            if progress_path.exists():
                report.setdefault('full_extraction',{})[variant]=json.loads(progress_path.read_text('utf8'))
        write(report_path,report)

    if args.stage in {'all','speed'}:
        # Deterministic species-balanced real-image sample, independent of outcomes.
        rng=np.random.default_rng(20261115)
        sample=[]
        groups=[list(rng.permutation(np.flatnonzero(truth==species))) for species in np.unique(truth)]
        while len(sample)<min(args.speed_count,len(rows)):
            for group in groups:
                if group and len(sample)<args.speed_count:
                    sample.append(int(group.pop()))
        sample=np.array(sample)
        np.save(args.output/'speed_sample_indices.npy',indices[sample])
        sample_paths=[paths[i] for i in sample]
        prepared_nhwc=_preprocess_nhwc_batch(sample_paths)
        prepared_nchw=[torch.from_numpy(x.transpose(2,0,1).copy()) for x in prepared_nhwc]
        baseline_cpu=None
        for variant, device in [('pytorch_cpu','cpu'),('pytorch_gpu','cuda'),('litert_fp32','cpu'),('litert_fp16_weights','cpu')]:
            if device=='cuda' and not torch.cuda.is_available():
                continue
            encoder=make_encoder(variant,device=device)
            data=prepared_nchw if variant.startswith('pytorch') else prepared_nhwc
            entry=dict(images=len(sample), input='same preprocessed real crops; includes CPU/GPU transfers', batches={})
            with threadpool_limits(limits=args.threads):
                for batch_size in (1,8,32):
                    encoder.batch_size=batch_size
                    measurement=timed(lambda:forward(encoder,data))
                    measurement['images_per_second']=len(sample)/measurement['median_seconds']
                    measurement['ms_per_image']=1000*measurement['median_seconds']/len(sample)
                    entry['batches'][str(batch_size)]=measurement
                    log(f"Speed {variant} batch={batch_size}: {measurement['images_per_second']:.2f} crops/s")
                encoded=forward(encoder,data)
            if variant=='pytorch_cpu':
                baseline_cpu=encoded
            elif baseline_cpu is not None:
                entry['vs_cpu_features']=dict(max_abs_diff=float(abs(encoded-baseline_cpu).max()),
                    min_cosine=float((encoded.astype(np.float64)*baseline_cpu).sum(1).min()))
            report.setdefault('speed',{})[variant]=entry
            write(report_path,report)
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log(f'DONE {report_path}')


if __name__=='__main__':
    main()
