"""Bounded reference-text queue; exact cached video/geometry inputs, no training RNG.

The original data/text modules stay pinned. Queue entries are disposable, keyed by
absolute optimizer step and rank, and validated against the immutable manifest.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from data import Data432, JOB, MANIFEST, select, sha, write_json

QUEUE = Path(os.environ.get('PLENOPTIC_TEXT_QUEUE',str(JOB/'text_queue_v1'))).resolve()
QUEUE.relative_to(JOB)
VERSION = 'exact_reference_text_queue_v1'


def encoder_signature():
    code = Path(__file__).parent
    paths = [code/'text.py', code/'data.py',
             code.parent/'plenoptic_recovery/caption_recovery.py',
             Path('/root/wenfy/SVD/scripts/plenoptic/basic_full_data.py'),
             Path('/root/wenfy/SVD/results/text_encoder/weights.json')]
    return hashlib.sha256(json.dumps({str(p):sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()


def queue_path(step, rank):
    return QUEUE / f'step_{step:05d}' / f'rank{rank}.safetensors'


def metadata(data, step, rank, scene, reference):
    return dict(version=VERSION, step=str(step), rank=str(rank), scene=scene,
                reference=reference, source_sha256=data.scenes[scene]['videos'][reference]['source_sha256'],
                manifest_sha256=data.manifest_sha, preprocess_id=data.manifest['preprocess_id'],
                encoder_signature=encoder_signature())


def publish(data, encoder, step, rank, scene, reference):
    path=queue_path(step,rank)
    caption=encoder.caption(data,scene,reference)
    text=encoder.embedding(caption).cpu().contiguous()
    assert text.shape==(1,512,100352) and text.dtype==torch.bfloat16 and torch.isfinite(text).all()
    meta=metadata(data,step,rank,scene,reference)
    meta['caption_sha256']=hashlib.sha256(caption.encode()).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    save_file({'text':text},str(tmp),metadata=meta);tmp.replace(path)
    return text


def read_text(data, step, rank, scene, reference, wait=True):
    path=queue_path(step,rank);started=time.monotonic()
    while not path.exists():
        if not wait:raise FileNotFoundError(path)
        write_json(JOB/'text_wait'/f'rank{rank}.json',dict(step=step,rank=rank,path=str(path),
                   waiting_seconds=time.monotonic()-started,updated_unix=time.time()))
        if time.monotonic()-started>300:raise TimeoutError(f'Text queue stalled: {path}')
        time.sleep(.05)
    expected=metadata(data,step,rank,scene,reference)
    caption_record=json.loads((JOB/'captions'/scene/f'{reference}.json').read_text())
    expected['caption_sha256']=hashlib.sha256(caption_record['caption'].encode()).hexdigest()
    with safe_open(str(path),framework='pt',device='cpu') as h:
        if h.metadata()!=expected:raise ValueError(f'Text queue metadata mismatch: {path}')
        assert list(h.get_slice('text').get_shape())==[1,512,100352]
    text=load_file(str(path))['text']
    assert text.dtype==torch.bfloat16 and torch.isfinite(text).all()
    return text


class PreparedInputs:
    """One CPU worker owns its Data432, preventing last_latent_sources races."""
    def __init__(self, manifest=MANIFEST):
        self.data=Data432(manifest)

    def prepare(self, step, rank, world, k1_steps, seed):
        data=self.data
        scene,cameras,epoch=select(data.train,step,rank,world,k1_steps,seed)
        data.last_latent_sources=[]
        zs=[data.latent_cpu(scene,c) for c in cameras]
        text=read_text(data,step,rank,scene,cameras[0])
        data.camera_arrays(scene)  # Read/validate pose JSON ahead; geometry stays in original function.
        return dict(step=step,scene=scene,cameras=cameras,epoch=epoch,
                    sources=list(data.last_latent_sources),zs=[z.pin_memory() for z in zs],
                    text=text.pin_memory())

    @staticmethod
    def materialize(data, prepared):
        scene=prepared['scene'];cameras=prepared['cameras']
        zs=[z[None].to(device='cuda',dtype=torch.float32,non_blocking=True) for z in prepared['zs']]
        data.last_latent_sources=prepared['sources']
        return (torch.stack(zs[:-1],dim=1),zs[-1],data.camera(scene,cameras[:-1],cameras[-1]),
                prepared['text'].cuda(non_blocking=True))
