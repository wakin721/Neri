"""Cache canonical E inputs once; check Torch/NumPy normalization bitwise."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from system.dino.preprocess import _as_rgb_image,letterbox_image,IMAGENET_MEAN,IMAGENET_STD
from system.dinov3.encoder import sha256


def main():
    torch.set_num_threads(1)
    run=ROOT/'runs/Neri1.2preview-litert-local-full-20261115'
    source=ROOT/'runs/local_protocol_comparison_20261114'
    rows=[json.loads(l) for l in (source/'current_reviewed.jsonl').read_text('utf8').splitlines()]
    split=json.loads((source/'partition.json').read_text('utf8'))
    indices=sorted(int(k) for k,v in split['assignments'].items() if v in {'evaluation','evaluation_unknown'})
    paths=[rows[i]['file'] for i in indices]
    info_path=run/'input_cache.json'
    info=json.loads(info_path.read_text('utf8')) if info_path.exists() else dict(completed=0,seconds=0.,
        rows=len(paths),rows_sha256=sha256(source/'current_reviewed.jsonl'),
        partition_sha256=sha256(source/'partition.json'),preprocessing_bitwise_equal=True,
        preflight_original_functions_exact_images=16)
    assert info['rows_sha256']==sha256(source/'current_reviewed.jsonl')
    assert info['partition_sha256']==sha256(source/'partition.json')
    output=run/'preprocessed_nhwc.npy'
    cache=np.lib.format.open_memmap(output,mode='r+' if output.exists() else 'w+',dtype=np.float32,
                                    shape=(len(paths),224,224,3))
    mean=np.array(IMAGENET_MEAN,np.float32)
    std=np.array(IMAGENET_STD,np.float32)
    torch_mean=torch.tensor(IMAGENET_MEAN,dtype=torch.float32)[:,None,None]
    torch_std=torch.tensor(IMAGENET_STD,dtype=torch.float32)[:,None,None]
    def prepare(path):
        image=letterbox_image(_as_rgb_image(path,array_color='rgb'))
        pixels=np.asarray(image,dtype=np.float32)/255.
        numpy_input=(pixels-mean)/std
        torch_input=((torch.from_numpy(pixels.copy()).permute(2,0,1)-torch_mean)/torch_std).numpy().transpose(1,2,0)
        assert np.array_equal(numpy_input,torch_input),path
        return numpy_input
    with ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(info['completed'],len(paths),128):
            start=time.perf_counter()
            stop=min(begin+128,len(paths))
            cache[begin:stop]=np.stack(list(pool.map(prepare,paths[begin:stop])))
            cache.flush()
            info['completed']=stop
            info['seconds']+=time.perf_counter()-start
            temporary=info_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(info,indent=2)+'\n','utf8')
            temporary.replace(info_path)
            if begin//128%8==0 or stop==len(paths):
                print(time.strftime('%H:%M:%S'),f'Input cache: {stop}/{len(paths)}, {info["seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
