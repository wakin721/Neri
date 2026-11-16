"""Run the shipped detector, classifier and complete photo-event path without torch."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def verify(root, photos, output):
    root=Path(root).resolve(strict=True)
    photos=[Path(p).resolve(strict=True) for p in photos]
    if importlib.util.find_spec('torch') is not None:
        raise RuntimeError('Use the actual pruned portable Python runtime for this check')
    os.chdir(root)
    sys.path.insert(0,str(root))
    from system.backend import services
    from system.backend.models import DetectionItem
    from system.dinov3.processor import finalize_photo_job,ensure_dinov3_sequences_completed
    from system.model_sync.catalog import discover_models
    from system.model_sync.layout import get_model_layout
    layout=get_model_layout(root/'res')
    classifier=next(m for m in discover_models(layout,'cls')
        if m.source=='bundled' and m.encoder_backend=='litert')
    detect=next(m for m in discover_models(layout,'detect') if m.source=='bundled' and m.backend=='litert')
    detector=services._load_detector(detect.path,classifier.path)
    try:
        raw=detector.detect_batch_species([str(p) for p in photos],conf=.1,use_fp16=True)
        count=len(detector._dinov3_collector.get_all_crops())
        if not count:
            raise AssertionError('No animal boxes found; supply at least one clear animal photograph')
        items=[DetectionItem(filename=p.name,path=str(p),file_type=p.suffix.lstrip('.'),
            date_taken='2026-11-15T08:00:00',detection_data=services._legacy._serialize_detector_output(detector,d))
            for p,d in zip(photos,raw)]
        input_root=Path(os.path.commonpath([str(p.parent) for p in photos]))
        finalized=finalize_photo_job(detector,items,input_root)
        ensure_dinov3_sequences_completed(finalized)
        assert len(finalized)==len(photos)
        assert len(detector.dino_classifier.classes)==45
        assert detector.dino_classifier.encoder.device=='cpu'
        assert not detector.dino_classifier.encoder.use_fp16
        assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
        report=dict(passed=True,photos=len(photos),classified_crops=count,classes=45,
            classifier=classifier.name,encoder_backend='litert',device='cpu',precision='fp32',
            complete_photo_events=True,torch_installed=False,torch_imported=False,
            results=[dict(file=Path(item.path).name,species=item.species,
                          sequence_completed=item.detection_data['sequence_completed'],
                          boxes=len(item.detection_data.get('检测框',[]))) for item in finalized])
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return report
    finally:
        detector.dino_runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application-root',type=Path,required=True)
    parser.add_argument('--photos',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.application_root,args.photos,args.output.resolve())
