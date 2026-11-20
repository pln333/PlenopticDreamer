"""Pinned 432x768 raw-video preprocessing, official VAE cache, camera conditions."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from context_layout import ContextLayout
from syncam_data import center_crop_parameters, scene_cameras, transform_frame

LAB = Path(__file__).resolve().parents[2]
SVD = Path('/root/wenfy/SVD')
JOB = LAB / 'results/basic432_20261118'
MANIFEST = JOB / 'dataset.json'
SEED = 20261117
SHAPE = [16, 21, 54, 96]
PREPROCESS = 'crop1280x720_top280_resize768x432_cv2area_cosmos_wan21_tw16_bf16_v1'
ROOTS = {'multicam': Path('/data/nvs/recammaster/datasets/MultiCamVideo-extracted/MultiCamVideo-Dataset'),
         'syncam': Path('/data/datasets/nvs/SynCamVideo-Dataset')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


_PERMUTATIONS = {}


def select(train, step, rank, world=4, k1_steps=10000, seed=SEED):
    index = step * world + rank; epoch = index // len(train)
    key = (id(train),epoch,seed)
    if key not in _PERMUTATIONS:
        order = train.copy(); random.Random(seed + epoch).shuffle(order)
        _PERMUTATIONS[key] = order
    order = _PERMUTATIONS[key]
    scene = order[index % len(order)]
    k = 1 if step < k1_steps else 2
    cameras = random.Random(seed + index + 999).sample([f'cam{i:02d}' for i in range(1,11)], k + 1)
    return scene, cameras, epoch


class Data432:
    def __init__(self, path=MANIFEST):
        self.manifest_path = Path(path); self.manifest = json.loads(self.manifest_path.read_text())
        assert self.manifest['preprocess_id'] == PREPROCESS and self.manifest['latent_shape'] == SHAPE
        self.manifest_sha = sha(self.manifest_path)
        self.scenes = self.manifest['scenes']
        self.train = sorted(s for s,r in self.scenes.items() if r['split']=='train')
        self.val = sorted(s for s,r in self.scenes.items() if r['split']=='val')
        self.last_latent_sources = []

    def root(self, scene): return ROOTS[self.scenes[scene]['dataset']]

    def source(self, scene, camera): return self.root(scene) / self.scenes[scene]['videos'][camera]['source']

    def cache_path(self, scene, camera): return JOB / 'latents' / scene / f'{camera}.safetensors'

    def metadata(self, scene, camera):
        row = self.scenes[scene]; entry = row['videos'][camera]
        return dict(scene=scene, camera=camera, source_sha256=entry['source_sha256'],
                    source_camera_sha256=row['camera_sha256'], manifest_sha256=self.manifest_sha,
                    preprocess_id=PREPROCESS, vae_sha256=self.manifest['vae_sha256'],
                    source_kind='real_video_vae_encoding', layout='CTHW')

    def read_rgb(self, scene, camera):
        entry = self.scenes[scene]['videos'][camera]; source = self.source(scene,camera)
        assert source.stat().st_size == entry['source_bytes'] and sha(source) == entry['source_sha256']
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS,2])
        config = self.manifest['preprocessing']; frames = []
        try:
            for frame in range(81):
                ok, bgr = cap.read()
                if not ok: raise ValueError(f'Cannot decode frame {frame}: {source}')
                frames.append(cv2.cvtColor(transform_frame(bgr,config), cv2.COLOR_BGR2RGB))
        finally: cap.release()
        return np.stack(frames)

    def latent_cpu(self, scene, camera, wait=True):
        path = self.cache_path(scene,camera); started = time.monotonic()
        while not path.exists():
            if not wait: raise FileNotFoundError(path)
            rank = os.environ.get('RANK','0')
            write_json(JOB / 'cache_wait' / f'rank{rank}.json', dict(scene=scene,camera=camera,path=str(path),
                       waiting_seconds=time.monotonic()-started,updated_unix=time.time()))
            if time.monotonic()-started > 86400: raise TimeoutError(f'Cache unavailable for 24h: {path}')
            time.sleep(2)
        with safe_open(str(path),framework='pt',device='cpu') as h:
            assert h.metadata() == self.metadata(scene,camera)
            assert list(h.get_slice('latent').get_shape()) == SHAPE
        z = load_file(str(path))['latent']
        assert z.dtype == torch.bfloat16 and torch.isfinite(z).all()
        self.last_latent_sources.append(dict(camera=camera,mode='432_cache',source_sha256=self.scenes[scene]['videos'][camera]['source_sha256']))
        return z

    def latent(self,scene,camera): return self.latent_cpu(scene,camera)[None].float().cuda()

    @lru_cache(maxsize=32)
    def camera_arrays(self,scene):
        row = self.scenes[scene]; path = self.root(scene) / row['camera_source']
        assert sha(path) == row['camera_sha256']
        _,frames,c2w,w2c = scene_cameras(path)
        assert len(frames)==81 and np.isfinite(c2w).all() and np.max(np.abs(c2w@w2c-np.eye(4)))<1e-6
        _,original,affine,_ = center_crop_parameters(1280,1280)
        original[:2,:2] *= row['focal_mm']/24
        return c2w,affine@original

    def camera(self,scene,references,target):
        from cosmos_transfer2._src.predict2.camera.utils import convert_camera_to_plucker_rays
        poses,k = self.camera_arrays(scene)
        ids=[int(c[3:])-1 for c in [*references,target]]; selected=poses[ids][:,::4]
        local=np.linalg.inv(selected[0,0])
        w2c=torch.from_numpy(np.linalg.inv(local[None,None]@selected)).float().cuda()
        ks=torch.from_numpy(k).float().cuda().expand(len(ids),21,3,3)
        rays=[convert_camera_to_plucker_rays(w2c[:,i],ks[:,i],(432,768),out_dtype=torch.float32) for i in range(21)]
        return ContextLayout(len(references),21).pack_cameras(torch.stack(rays,dim=1)[None])

    def sample(self,scene,references,target,text_encoder):
        assert scene in self.train and len(set([*references,target]))==len(references)+1
        self.last_latent_sources=[]
        zs=[self.latent(scene,c) for c in [*references,target]]
        caption=text_encoder.caption(self,scene,references[0])
        return torch.stack(zs[:-1],dim=1),zs[-1],self.camera(scene,references,target),text_encoder.embedding(caption)


def prepare():
    JOB.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():
        print('Existing immutable manifest',sha(MANIFEST)); return
    oldpath=SVD/'results/basic_cached384_20261117/dataset.json'; old=json.loads(oldpath.read_text())
    groups={}
    for scene,row in old['scenes'].items():
        groups[scene]={k:row[k] for k in ['dataset','scene_id','focal_mm','split','camera_source','camera_sha256']}
        groups[scene]['videos']={c:{k:e[k] for k in ['source','source_sha256','source_bytes']} for c,e in row['videos'].items()}
    weights=json.loads((SVD/'results/tokenizer_reconstruction/tokenizer_weight.json').read_text())
    weight=SVD/weights['local_path']; assert sha(weight)==weights['sha256']
    config,_,_,_=center_crop_parameters(1280,1280)
    manifest=dict(scenes=groups,protected_scene_ids=old['protected_scene_ids'],width=768,height=432,frames=81,
                  latent_shape=SHAPE,preprocess_id=PREPROCESS,preprocessing=config,vae_weight=str(weight),vae_sha256=weights['sha256'],
                  tokenizer='Cosmos Wan2pt1VAEInterface; load_mean_std=False; temporal_window=16; chunk_duration=81',
                  split_source_sha256=sha(oldpath),sampler='Uniform shuffled full mixed scene list, independent of cache readiness; wait without substituting samples',
                  seed=SEED,k1_steps=10000,k2_steps=4000)
    write_json(MANIFEST,manifest)
    train=sorted(s for s,r in groups.items() if r['split']=='train')
    tasks=[]; seen=set()
    def add(scene,camera,phase):
        key=(scene,camera)
        if key not in seen: seen.add(key); tasks.append(dict(scene=scene,camera=camera,phase=phase))
    # Exact samples for two-step K1/K2 pipeline verification before formal training.
    for step in range(2):
        for rank in range(4):
            scene,cameras,_=select(train,step,rank,k1_steps=1)
            for c in cameras:add(scene,c,'smoke')
    # Initial held-out gallery: cache its references and targets before bulk work.
    for kind in ['multicam','syncam']:
        scenes=sorted(s for s,r in groups.items() if r['split']=='val' and r['dataset']==kind and r['focal_mm']==24)
        for scene in scenes[:2]:
            for c in ['cam02','cam03','cam06']:add(scene,c,'validation')
    # Rank assignment is independent of extraction shard; deterministic first-use priority.
    for step in range(14000):
        for rank in range(4):
            scene,cameras,_=select(train,step,rank)
            for c in cameras:add(scene,c,'training_first_use')
    for scene in sorted(groups):
        for c in sorted(groups[scene]['videos']):add(scene,c,'remaining_corpus')
    assert len(tasks)==170000
    with (JOB/'tasks.jsonl').open('x') as f:
        for i,t in enumerate(tasks):f.write(json.dumps(dict(index=i,shard=i%4,**t))+'\n')
    write_json(JOB/'preparation.json',dict(manifest_sha256=sha(MANIFEST),tasks_sha256=sha(JOB/'tasks.jsonl'),
               groups=len(groups),train_groups=len(train),val_groups=len(groups)-len(train),videos=len(tasks),
               expected_latent_payload_bytes=len(tasks)*int(np.prod(SHAPE))*2,
               current_workspace_outputs_only=True))
    print(json.dumps(json.loads((JOB/'preparation.json').read_text()),indent=2))


if __name__=='__main__': prepare()
