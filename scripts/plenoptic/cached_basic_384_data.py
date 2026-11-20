"""Mixed full-corpus Basic data, sibling-compatible cached/online Wan encoding."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from basic_full_data import TextEncoder
from cache_syncam_384 import (MULTICAM, MULTIDATA, DATA, CACHE, JOB as CACHE_JOB,
                             PREPROCESS, SHAPE, make_codec, encode, contract, sha, atomic_json)
from cache_syncam_384_prefetch import memoize_frame_counts
from context_layout import ContextLayout
from multiscene_data import ROOT
from syncam_data import scene_cameras

DIRECTORY = ROOT/'results/basic_cached384_20261117'
MANIFEST = DIRECTORY/'dataset.json'


def build():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError('Keep the pinned manifest; do not overwrite')
    old=json.loads((ROOT/'results/multiscene_expanded100_data/dataset.json').read_text())
    protected={s for s,r in old['scenes'].items() if r['split']=='val'}
    for split in ['validation','test']:
        protected.update(Path(s).name for s in json.loads(Path(f'/data/nvs/recammaster/runs/v2v_full_20261114/{split}_scenes.json').read_text()))
    groups={}
    multi_source=Path('/data/nvs/recammaster/datasets/MultiCamVideo-download/extracted_manifest.jsonl')
    multi_cache=Path('/data/nvs/recammaster/runs/v2v_full_20261114/cache_manifest_verified.jsonl')
    cache_records={r['source']:r for r in map(json.loads,multi_cache.read_text().splitlines())}
    manifests={
        'multicam':[(r['path'].removeprefix('MultiCamVideo-Dataset/'),r['sha256'],r['size']) for r in map(json.loads,multi_source.read_text().splitlines()) if r['path'].startswith('MultiCamVideo-Dataset/train/')],
        'syncam':[(r['path'],r['sha256'],r['bytes']) for r in json.loads((DATA/'source_manifest.json').read_text())['files'] if r['path'].startswith('train/')],
    }
    for kind,entries in manifests.items():
        bypath={p:(h,size) for p,h,size in entries}
        for p,h,size in entries:
            path=Path(p)
            if path.suffix!='.mp4':continue
            folder=path.parent.parent; key=f'{kind}/{folder.relative_to("train")}'
            camera_rel=str(folder/'cameras/camera_extrinsics.json')
            assert camera_rel in bypath
            if key not in groups:
                scene_id=folder.name
                focal=float(re.match(r'f([\d.]+)_',folder.parent.name).group(1))
                groups[key]=dict(dataset=kind,scene_id=scene_id,focal_mm=focal,
                                 split='val' if scene_id in protected else 'train',
                                 camera_source=camera_rel,camera_sha256=bypath[camera_rel][0],videos={})
            info=dict(source=p,source_sha256=h,source_bytes=size)
            if kind=='multicam':
                r=cache_records[str(path.relative_to('train'))]
                info.update(cache_sha256=r['sha256'],cache_bytes=r['bytes'])
            groups[key]['videos'][path.stem]=info
    assert len(groups)==17000
    for row in groups.values():assert set(row['videos'])=={f'cam{i:02d}' for i in range(1,11)}
    info=dict(scenes=groups,protected_scene_ids=sorted(protected),
              source_manifests={str(p):sha(p) for p in [multi_source,multi_cache,DATA/'source_manifest.json',CACHE_JOB/'contract.json']},
              width=672,height=384,frames=81,latent_shape=SHAPE,
              sampler='Uniform shuffled full mixed scene-group list; MultiCam:SynCam approximately 4:1; independent of cache availability.',
              geometry='Exact sibling resize 1280→672 square then vertical center crop144; K=affine@K_original, actual focal variants.',
              captions='On-demand automatic Cosmos-Reason annotations of first reference camera frames0/40/80; strings cached with source hashes. No target video captioning, no T5 embedding reuse.',
              fallback='Missing SynCam latent is computed with the validated exact sibling codec. Never skip to another scene or substitute a generated latent.')
    atomic_json(MANIFEST,info)
    print('DATASET',len(groups),'train',sum(r['split']=='train' for r in groups.values()),'val',sum(r['split']=='val' for r in groups.values()),flush=True)


class CachedBasicData:
    def __init__(self,path=MANIFEST):
        self.manifest_path=Path(path);self.manifest=json.loads(self.manifest_path.read_text())
        self.scenes=self.manifest['scenes'];self.train=sorted(k for k,r in self.scenes.items() if r['split']=='train');self.val=sorted(k for k,r in self.scenes.items() if r['split']=='val')
        self.codec=self.reader=None
        self.last_latent_sources=[]

    def root(self,scene):return MULTIDATA if self.scenes[scene]['dataset']=='multicam' else DATA

    def cache_path(self,scene,camera):
        row=self.scenes[scene];source=Path(row['videos'][camera]['source'])
        return (MULTICAM/'latents'/source.relative_to('train').with_suffix('.pt') if row['dataset']=='multicam' else CACHE/'latents'/source.with_suffix('.pt'))

    def codec_reader(self):
        if self.codec is None:
            self.codec,self.reader=make_codec(contract());memoize_frame_counts()
        return self.codec,self.reader

    def latent(self,scene,camera):
        row=self.scenes[scene];entry=row['videos'][camera];path=self.cache_path(scene,camera)
        if path.exists():
            if 'cache_sha256' in entry and sha(path)!=entry['cache_sha256']:raise ValueError('Sibling cache hash changed: '+str(path))
            d=torch.load(path,map_location='cpu',weights_only=True)
            assert d['source_sha256']==entry['source_sha256'] and d['preprocess_id']==PREPROCESS
            expected=entry['source'].removeprefix('train/') if row['dataset']=='multicam' else entry['source']
            assert d['source']==expected
            z=d['latents'];mode='existing_cache'
        else:
            if row['dataset']!='syncam':raise FileNotFoundError(path)
            source=self.root(scene)/entry['source']
            assert source.stat().st_size==entry['source_bytes'] and sha(source)==entry['source_sha256']
            codec,reader=self.codec_reader();z,_=encode(codec,reader,source);mode='online_missing_syncam_cache'
            # Extraction workers own the shared output. Avoid competing writes.
        assert list(z.shape)==SHAPE and z.dtype==torch.bfloat16 and torch.isfinite(z).all()
        self.last_latent_sources.append(dict(camera=camera,mode=mode,source_sha256=entry['source_sha256']))
        return z[None].float().cuda()

    @lru_cache(maxsize=32)
    def camera_arrays(self,scene):
        row=self.scenes[scene];path=self.root(scene)/row['camera_source']
        assert sha(path)==row['camera_sha256']
        names,frames,c2w,w2c=scene_cameras(path)
        r=c2w[...,:3,:3]
        assert len(frames)==81 and np.isfinite(c2w).all()
        assert np.max(np.abs(r.swapaxes(-1,-2)@r-np.eye(3)))<1e-4 and np.max(np.abs(np.linalg.det(r)-1))<1e-4
        assert np.max(np.abs(c2w@w2c-np.eye(4)))<1e-6
        focal=row['focal_mm'];k=np.array([[1280*focal/23.76,0,640],[0,1280*focal/23.76,640],[0,0,1]])
        affine=np.array([[.525,0,0],[0,.525,-144],[0,0,1]])
        return c2w,affine@k

    def camera(self,scene,references,target):
        from cosmos_transfer2._src.predict2.camera.utils import convert_camera_to_plucker_rays
        poses,k=self.camera_arrays(scene);ids=[int(c[3:])-1 for c in [*references,target]]
        selected=poses[ids][:,::4];local=np.linalg.inv(selected[0,0])
        w2c=torch.from_numpy(np.linalg.inv(local[None,None]@selected)).float().cuda()
        ks=torch.from_numpy(k).float().cuda().expand(len(ids),21,3,3)
        rays=[convert_camera_to_plucker_rays(w2c[:,i],ks[:,i],(384,672),out_dtype=torch.float32) for i in range(21)]
        return ContextLayout(len(references),21).pack_cameras(torch.stack(rays,dim=1)[None])

    def sample(self,scene,references,target,text_encoder):
        assert scene in self.train and len(set([*references,target]))==len(references)+1
        self.last_latent_sources=[]
        zs=[self.latent(scene,c) for c in [*references,target]]
        caption=text_encoder.caption(self,scene,references[0])
        return torch.stack(zs[:-1],dim=1),zs[-1],self.camera(scene,references,target),text_encoder.embedding(caption)


class ReferenceTextEncoder(TextEncoder):
    @torch.no_grad()
    def caption(self,data,scene,reference):
        from PIL import Image
        import imageio
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        from prepare_expanded import concise_caption
        row=data.scenes[scene];entry=row['videos'][reference]
        path=DIRECTORY/'captions'/scene/f'{reference}.json'
        if path.exists():
            saved=json.loads(path.read_text());assert saved['source_sha256']==entry['source_sha256']
            return saved['caption']
        source=data.root(scene)/entry['source'];assert sha(source)==entry['source_sha256']
        reader=imageio.get_reader(str(source));images=[]
        try:
            for frame_id in [0,40,80]:
                im=Image.fromarray(reader.get_data(frame_id));w,h=im.size;scale=max(672/w,384/h)
                im=TF.resize(im,(round(h*scale),round(w*scale)),interpolation=InterpolationMode.BILINEAR)
                im=TF.center_crop(im,(384,672));images.append(im.resize((336,192)))
        finally:reader.close()
        prompt='These images show the beginning, middle and end of one video. Write a short factual English caption, at most 60 words, describing the visible subject, movement and surroundings. Do not invent hidden objects or actions. Return only complete caption sentences, without reasoning.'
        messages=[{'role':'user','content':[*[{'type':'image'} for _ in images],{'type':'text','text':prompt}]}]
        conversation=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[conversation],images=images,return_tensors='pt').to('cuda')
        for budget in [192,384,512]:
            result=self.model.generate(**inputs,do_sample=False,max_new_tokens=budget,use_cache=True)
            answer=self.processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
            try:caption=concise_caption(answer);break
            except ValueError:
                if budget==512:raise
        atomic_json(path,dict(caption=caption,raw=answer,source_sha256=entry['source_sha256'],source_video=str(source),
                              scene=scene,reference_camera=reference,frames=[0,40,80],prompt=prompt,
                              source='automatic Cosmos-Reason1-7B annotation, not human reviewed or author-provided',generated_unix=time.time()))
        return caption


if __name__=='__main__':build()
