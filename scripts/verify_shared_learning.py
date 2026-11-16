"""Exercise real CPU->CUDA->CPU learning in a dedicated, disposable state root."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-manifest',type=Path,required=True)
    parser.add_argument('--gpu-manifest',type=Path,required=True)
    parser.add_argument('--photos',type=Path,nargs=2,required=True)
    parser.add_argument('--state-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    state=args.state_dir.resolve()
    if state.exists():
        raise FileExistsError('Use a fresh isolated state directory')
    state.mkdir(parents=True)
    os.environ['NERI_DINO_STATE_DIR']=str(state)
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo))
    from PIL import Image
    from system.backend import dinov3_learning as learning
    from system.dinov3.runtime import load_dinov3_model
    from system.dinov3.observation_support import make_dinov3_observation_id
    from system.dinov3.shared_learning import family_id
    cpu,gpu=args.cpu_manifest.resolve(),args.gpu_manifest.resolve()
    assert family_id(cpu) and family_id(cpu)==family_id(gpu)
    originals={path:path.read_bytes() for path in (cpu,gpu)}
    report=dict(passed=False,steps=[],family_id=family_id(cpu))
    started=time.perf_counter()
    def log(stage,**data):
        row=dict(stage=stage,elapsed_seconds=time.perf_counter()-started,**data)
        report['steps'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    labels=['共享学习验证甲','共享学习验证乙']
    photos=[]
    for index,source in enumerate(args.photos):
        target=state/'inputs'/str(index)/source.name
        target.parent.mkdir(parents=True)
        shutil.copy2(source,target)
        photos.append(target)
    def stage(manifest,photo,label,operation):
        with Image.open(photo) as image:
            width,height=image.size
        bbox=[0,0,width,height]
        box={'物种':label,'边界框':bbox,'observation_id':make_dinov3_observation_id(photo,0,bbox)}
        learning.stage_feedback(manifest,state/'inputs',photo,box,'update',operation)
    stage(cpu,photos[0],labels[0],'cpu-confirm')
    learning.update_model(cpu)
    assert 'torch' not in sys.modules
    runtime=load_dinov3_model(cpu)
    assert labels[0] in runtime.classifier.classes
    log('cpu_learning',classes=len(runtime.classifier.classes),device=runtime.classifier.encoder.device,torch_imported=False)
    runtime.close()
    runtime=load_dinov3_model(gpu)
    assert labels[0] in runtime.classifier.classes and runtime.classifier.encoder.device=='cuda'
    log('gpu_inherits_cpu',classes=len(runtime.classifier.classes),device=runtime.classifier.encoder.device)
    runtime.close()
    photos[0].unlink()
    stage(gpu,photos[1],labels[1],'gpu-confirm')
    learning.update_model(gpu)
    runtime=load_dinov3_model(cpu)
    assert set(labels)<=set(runtime.classifier.classes)
    log('cpu_inherits_gpu',classes=len(runtime.classifier.classes),device=runtime.classifier.encoder.device)
    runtime.close()
    learning.revert_staged_feedback(gpu,'cpu-confirm')
    learning.update_model(cpu)
    runtime=load_dinov3_model(gpu)
    assert labels[0] not in runtime.classifier.classes and labels[1] in runtime.classifier.classes
    log('cross_backend_undo_without_source_photo',classes=len(runtime.classifier.classes),device=runtime.classifier.encoder.device)
    runtime.close()
    command="""import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from system.dinov3.runtime import load_dinov3_model;r=load_dinov3_model(sys.argv[2]);assert sys.argv[3] in r.classifier.classes;assert sys.argv[4] not in r.classifier.classes;print(json.dumps({'passed':True,'classes':len(r.classifier.classes),'device':r.classifier.encoder.device}));r.close()"""
    resumed=subprocess.run([sys.executable,'-I','-X','utf8','-c',command,str(repo),str(gpu),labels[1],labels[0]],
        capture_output=True,text=True,encoding='utf8',timeout=60)
    assert resumed.returncode==0,(resumed.stdout,resumed.stderr)
    log('fresh_process_restart',**json.loads(resumed.stdout))
    assert all(path.read_bytes()==content for path,content in originals.items())
    report.update(passed=True,source_manifests_unchanged=True,cpu_gpu_shared=True,
        cross_backend_undo=True,restart_verified=True,isolated_state=str(state),
        elapsed_seconds=time.perf_counter()-started)
    args.output.resolve().parent.mkdir(parents=True,exist_ok=True)
    args.output.resolve().write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
    print('SHARED LEARNING VERIFIED',flush=True)


if __name__=='__main__':
    main()
