"""Four independent GPU shards; exact 432p CPU preprocess overlaps official VAE."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

from data import Data432, JOB, MANIFEST, SHAPE, sha, write_json
from cache_syncam_384 import acquire_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();assert 0<=a.shard<4
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_grad_enabled(False)
    data=Data432(); tasks=[json.loads(line) for line in (JOB/'tasks.jsonl').read_text().splitlines()]
    tasks=[r for r in tasks if r['shard']==a.shard]
    if a.limit:tasks=tasks[:a.limit]
    info=json.loads((JOB/'preparation.json').read_text())
    assert info['manifest_sha256']==data.manifest_sha and info['tasks_sha256']==sha(JOB/'tasks.jsonl')
    statuspath=JOB/f'cache_worker{a.shard}.json'
    def status(stage,**kw):
        item=dict(stage=stage,shard=a.shard,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),updated_unix=time.time(),**kw)
        write_json(statuspath,item)
        print(json.dumps(item),flush=True)
    status('acquiring_gpu')
    acquire_gpu()
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    assert sha(data.manifest['vae_weight'])==data.manifest['vae_sha256']
    tokenizer=Wan2pt1VAEInterface(vae_pth=data.manifest['vae_weight'],load_mean_std=False,chunk_duration=81,temporal_window=16)
    pending=[]; reused=0
    for row in tasks:
        if data.cache_path(row['scene'],row['camera']).exists():
            data.latent_cpu(row['scene'],row['camera'],wait=False);reused+=1
        else:pending.append(row)
    encoded=0;begin=time.monotonic();seconds=[]
    logpath=JOB/f'cache_shard{a.shard}.jsonl'
    with ThreadPoolExecutor(max_workers=1) as pool,logpath.open('a') as log:
        future=pool.submit(data.read_rgb,pending[0]['scene'],pending[0]['camera']) if pending else None
        for i,row in enumerate(pending):
            start=time.perf_counter();rgb=future.result()
            future=pool.submit(data.read_rgb,pending[i+1]['scene'],pending[i+1]['camera']) if i+1<len(pending) else None
            video=torch.from_numpy(rgb.copy()).permute(3,0,1,2)[None].cuda().to(torch.bfloat16)/127.5-1
            with torch.inference_mode(): z=tokenizer.encode(video)
            assert list(z.shape)==[1,*SHAPE] and torch.isfinite(z).all()
            latent=z[0].to(torch.bfloat16).cpu().contiguous()
            target=data.cache_path(row['scene'],row['camera']);target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_suffix('.tmp')
            save_file({'latent':latent},str(tmp),metadata=data.metadata(row['scene'],row['camera']))
            tmp.replace(target)
            if encoded==0:
                with torch.inference_mode(): recon=tokenizer.decode(latent[None].cuda()).float().cpu().clamp(-1,1)
                original=torch.from_numpy(rgb.copy()).permute(3,0,1,2)[None].float()/127.5-1
                frame_mse=((recon-original)/2).square().mean(dim=(0,1,3,4))
                psnr=float((-10*torch.log10(frame_mse.clamp_min(1e-12))).mean())
                assert recon.shape==original.shape and psnr>20
                from reconstruct_tokenizer import write_video
                rr=np.rint(((recon[0].permute(1,2,3,0).numpy()+1)/2).clip(0,1)*255).astype(np.uint8)
                write_video(JOB/f'codec_shard{a.shard}_reference_reconstruction.mp4',np.concatenate([rgb,rr],axis=2),15)
                write_json(JOB/f'codec_shard{a.shard}_validation.json',dict(status='passed',scene=row['scene'],camera=row['camera'],
                    reconstruction_psnr_db=psnr,latent_shape=list(latent.shape),video_shape=list(original.shape),
                    source_sha256=data.scenes[row['scene']]['videos'][row['camera']]['source_sha256'],cache_sha256=sha(target),
                    vae_sha256=data.manifest['vae_sha256'],preprocess_id=data.manifest['preprocess_id']))
                del recon,original,rr
            tokenizer.clear_cache();del video,z,latent,rgb
            encoded+=1;elapsed=time.perf_counter()-start;seconds.append(elapsed)
            record=dict(**row,seconds=elapsed,path=str(target),cache_sha256=sha(target),updated_unix=time.time())
            log.write(json.dumps(record)+'\n');log.flush()
            status('encoding',encoded=encoded,reused=reused,total=len(tasks),scene=row['scene'],camera=row['camera'],phase=row['phase'],
                   seconds=elapsed,mean_last20_seconds=sum(seconds[-20:])/len(seconds[-20:]),
                   wall_seconds=time.monotonic()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    status('completed',encoded=encoded,reused=reused,total=len(tasks))


if __name__=='__main__':
    try: main()
    except BaseException:
        traceback.print_exc();raise
