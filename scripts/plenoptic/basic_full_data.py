"""Full Basic source index and online video/text adapters; no disk embedding cache."""
import json,re
from pathlib import Path
import cv2,numpy as np,torch
from context_layout import ContextLayout
from multiscene_data import ROOT,digest,write_json
from syncam_data import scene_cameras,center_crop_parameters,read_rgb_frames,transform_frame
DIRECTORY=ROOT/'experiments/plenoptic/data/basic_full'
MANIFEST=DIRECTORY/'dataset.json'

def build_manifest():
    old=json.loads((ROOT/'results/multiscene_expanded100_data/dataset.json').read_text())
    protected={s for s,r in old['scenes'].items() if r['split']=='val'}
    rows={}; sources={}
    for kind in ['syncam','multicam']:
        directory=DIRECTORY/kind
        source=json.loads((directory/'source_manifest.json').read_text())
        if not source['complete']: raise ValueError('incomplete download')
        hashes={f['path']:f for f in source['files']}; sources[kind]=dict(repo=source['repo'],revision=source['revision'],manifest_sha256=digest(directory/'source_manifest.json'))
        groups=sorted(directory.glob('train/f*/scene*/cameras/camera_extrinsics.json'))
        expected=3400 if kind=='syncam' else 13600
        if len(groups)!=expected: raise ValueError(f'{kind}: expected {expected} groups, got {len(groups)}')
        for i,camera_path in enumerate(groups):
            folder=camera_path.parent.parent; focal=float(re.match(r'f([\d.]+)_',folder.parent.name).group(1))
            names,frames,c2w,_=scene_cameras(camera_path)
            if len(frames)!=81 or not np.isfinite(c2w).all(): raise ValueError('invalid camera')
            rotations=c2w[...,:3,:3]
            if max(np.abs(rotations.swapaxes(-1,-2)@rotations-np.eye(3)).max(),np.abs(np.linalg.det(rotations)-1).max())>1e-4: raise ValueError('invalid rotations')
            config,k0,affine,k=center_crop_parameters(1280,1280)
            k0[:2,:2]*=focal/24; k=affine@k0
            videos={}; video_hashes={}
            for name in names:
                path=folder/'videos'/f'{name}.mp4'; record=hashes[str(path.relative_to(directory))]
                if path.stat().st_size!=record['bytes']: raise ValueError('source size changed')
                cap=cv2.VideoCapture(str(path)); shape=[cap.get(v) for v in [cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT]]; cap.release()
                if not np.allclose(shape,[81,15,1280,1280],atol=1e-4): raise ValueError('video metadata '+str(path))
                videos[name]=str(path.relative_to(ROOT)); video_hashes[name]=record['sha256']
            scene=folder.name; key=f'{kind}/{folder.parent.name}/{scene}'
            output=DIRECTORY/'cameras'/kind/folder.parent.name/f'{scene}.npz'; output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output,c2w=c2w,K_processed=k,K_original=k0,pixel_affine=affine)
            rows[key]=dict(scene_id=scene,dataset=kind,focal_mm=focal,split='val' if scene in protected else 'train',videos=videos,video_sha256=video_hashes,camera_npz=str(output.relative_to(ROOT)),source_camera=str(camera_path.relative_to(ROOT)),source_camera_sha256=digest(camera_path),camera_sha256=digest(output),preprocessing=config,num_frames=81,fps=15,dynamic=bool(np.max(np.abs(c2w-c2w[:,:1]))>1e-6))
            if i%100==0: print('INDEX',kind,i,len(groups),flush=True)
    write_json(MANIFEST,dict(sources=sources,protected_scene_ids=sorted(protected),scenes=rows,frame_policy='81 original frames; paper reports 93-frame generated videos; no repeated or invented frames',split_policy='hold out the 12 historical validation scene IDs across both datasets and every focal variant'))
    print('MANIFEST',len(rows),sum(r['split']=='train' for r in rows.values()),flush=True)

class OnlineData:
    def __init__(self,path=MANIFEST):
        self.manifest_path=Path(path); self.manifest=json.loads(self.manifest_path.read_text()); self.scenes=self.manifest['scenes']
        self.train=sorted(s for s,r in self.scenes.items() if r['split']=='train')
        self.val=sorted(s for s,r in self.scenes.items() if r['split']=='val')
    def read_rgb(self,scene,camera):
        row=self.scenes[scene]; path=ROOT/row['videos'][camera]
        if 'video_sha256' in row and digest(path)!=row['video_sha256'][camera]: raise ValueError('changed video')
        capture=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        frames=[]
        try:
            for i in range(81):
                ok,frame=capture.read()
                if not ok: raise ValueError('cannot decode '+str(path))
                frames.append(cv2.cvtColor(transform_frame(frame,row['preprocessing']),cv2.COLOR_BGR2RGB))
        finally: capture.release()
        return np.stack(frames).transpose(0,3,1,2)

    def read_group(self,scene,cameras):
        return [self.read_rgb(scene,camera) for camera in cameras]

    def encode(self,tokenizer,scene,camera,rgb=None):
        row=self.scenes[scene]; path=ROOT/row['videos'][camera]
        # Check bytes against the pinned source manifest before this video enters training.
        if rgb is None: rgb=self.read_rgb(scene,camera)
        x=torch.from_numpy(rgb.copy()).permute(1,0,2,3)[None].cuda().to(torch.bfloat16)/127.5-1
        with torch.inference_mode(): z=tokenizer.encode(x).float()
        tokenizer.clear_cache()
        if z.shape!=(1,16,21,54,96) or not torch.isfinite(z).all(): raise ValueError('bad online latent')
        return z
    def camera(self,scene,references,target):
        from cosmos_transfer2._src.predict2.camera.utils import convert_camera_to_plucker_rays
        row=self.scenes[scene]
        if 'camera_sha256' in row and digest(ROOT/row['camera_npz'])!=row['camera_sha256']: raise ValueError('changed camera cache')
        with np.load(ROOT/row['camera_npz']) as handle: poses=handle['c2w']; intrinsics=handle['K_processed']
        ids=[int(c[3:])-1 for c in [*references,target]]
        # Causal VAE has RGB frame0 followed by groups of four. Use frame0/4/.../80
        # as camera representatives. This local temporal sampling choice is documented.
        poses=poses[ids][:,::4]
        local=np.linalg.inv(poses[0,0]); w2c=torch.from_numpy(np.linalg.inv(local[None,None]@poses)).float()
        ks=torch.from_numpy(intrinsics).float().expand(len(ids),21,3,3)
        # Construct one temporal frame at a time to avoid full RGB raymap memory peaks.
        output=[convert_camera_to_plucker_rays(w2c[:,i].cuda(),ks[:,i].cuda(),(432,768),out_dtype=torch.float32) for i in range(21)]
        return ContextLayout(len(references),21).pack_cameras(torch.stack(output,dim=1)[None])

class TextEncoder:
    def __init__(self):
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        weights=json.loads((ROOT/'results/text_encoder/weights.json').read_text()); path=ROOT/weights['local_path']
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=56*56,max_pixels=384*384)
        self.model,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',low_cpu_mem_usage=True,output_loading_info=True)
        if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']): raise ValueError(str(info))
        self.model=self.model.cuda().eval().requires_grad_(False)
    @torch.no_grad()
    def embedding(self,caption):
        messages=[{'role':'system','content':[{'type':'text','text':'You are a helpful assistant who will provide prompts to an image generator.'}]},{'role':'user','content':[{'type':'text','text':caption}]}]
        conversation=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,add_vision_id=False)
        ids=self.processor(text=[conversation],return_tensors='pt')['input_ids'][0].tolist()
        if len(ids)>512: raise ValueError('caption exceeds text window')
        tokens=torch.tensor([ids+[self.processor.tokenizer.pad_token_id]*(512-len(ids))],device='cuda')
        result=self.model.model(input_ids=tokens,attention_mask=None,output_hidden_states=True,use_cache=False,return_dict=True)
        layers=[(h-h.mean(dim=-1,keepdim=True))/(h.std(dim=-1,keepdim=True)+1e-8) for h in result.hidden_states[1:]]
        return torch.cat(layers,dim=-1).detach()

if __name__=='__main__': build_manifest()
