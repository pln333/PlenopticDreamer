"""Isolated full-resolution K1/K2 compute/checkpoint comparison on one cache GPU.

All variants start from identical trained weights/FP32 optimizer states and use
identical fixed real inputs, noise, sigma and clean-prefix masks. No formal run
outputs are modified. Store hashes of every trainable tensor after each update.
"""
import argparse
import gc
import hashlib
import json
import time

import torch
from safetensors.torch import load_file

from camera_backbone import load_camera_backbone, use_wan_mixed_precision
from context_layout import make_flow_inputs, flow_loss
from data import Data432, JOB, select, write_json
from text import ReferenceTextEncoder


def checkpoint(model,every):
    if every==0:return
    from cosmos_transfer2._src.predict2.camera.networks.dit_multiview_camera_ar import SACConfig
    from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts
    def context():
        def policy(ctx,func,*args,**kwargs):
            heavy=('scaled_dot_product' in str(func) or 'flash_attn' in str(func))
            return CheckpointPolicy.MUST_SAVE if heavy else CheckpointPolicy.PREFER_RECOMPUTE
        return create_selective_checkpoint_contexts(policy)
    config=SACConfig(mode='block_wise',every_n_blocks=every);config.get_context_fn=lambda:context
    model.enable_selective_checkpoint(config,model.blocks)


def fingerprint(named):
    h=hashlib.sha256()
    for n,p in named:
        h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--k',type=int,required=True,choices=[1,2])
    p.add_argument('--every',type=int,required=True,choices=[0,1,2,4]);p.add_argument('--iterations',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(0)
    out=JOB/'acceleration';out.mkdir(exist_ok=True);output=out/f'compute_k{a.k}_every{a.every}.json'
    # Use smoke selections, whose caches are guaranteed to exist for K1 and K2.
    data=Data432();s,c,_=select(data.train,a.k-1,a.k-1,k1_steps=1)
    encoder=ReferenceTextEncoder()
    reference,gt,rays,text=data.sample(s,c[:-1],c[-1],encoder)
    del encoder;gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(617);model,loading=load_camera_backbone()
    state=load_file(str(JOB/'train/step_00100/trainable_weights.safetensors'))
    model.load_state_dict(state,strict=False);del state
    use_wan_mixed_precision(model);model=model.cuda().train();checkpoint(model,a.every)
    named=[(n.replace('._checkpoint_wrapped_module',''),p) for n,p in model.named_parameters() if p.requires_grad]
    masters=[torch.nn.Parameter(p.detach().float().clone()) for n,p in named]
    optimizer=torch.optim.AdamW(masters,lr=2e-5,betas=(.9,.999),weight_decay=.01,fused=True)
    padding=torch.zeros(1,1,54,96,device='cuda');rows=[]
    try:
        for i in range(a.iterations):
            torch.manual_seed(172+i);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            noise=torch.randn((1,16,21*(a.k+1),54,96),device='cuda')
            sigma=torch.tensor([.75],device='cuda');inputs=make_flow_inputs(reference,gt,sigma,noise,target_prefix_frames=6 if i%2 else 0)
            mask=inputs['clean_mask'].expand(-1,-1,-1,54,96).contiguous();temporal=inputs['clean_mask'][:,0,:,0,0]
            times=temporal*.1+(1-temporal)*(sigma*1000)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(x_B_C_T_H_W=inputs['model_input'].to(torch.bfloat16),timesteps_B_T=times,crossattn_emb=text,
                                 condition_video_input_mask_B_C_T_H_W=mask,padding_mask=padding,camera=rays)
            loss=flow_loss(prediction,inputs['target_velocity'],inputs['clean_mask']);assert torch.isfinite(loss)
            loss.backward();torch.cuda.synchronize();backward_end=time.perf_counter()
            for (n,p),m in zip(named,masters):
                assert p.grad is not None;m.grad=p.grad.detach().float()
            norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for (n,p),m in zip(named,masters):p.copy_(m.to(p.dtype))
            optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);torch.cuda.synchronize();end=time.perf_counter()
            row=dict(iteration=i,loss=loss.item(),gradient_norm=norm.item(),forward_backward_seconds=backward_end-start,
                     compute_seconds=end-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,trainable_sha256=fingerprint(named))
            rows.append(row);print(json.dumps(row),flush=True)
            del prediction,loss,inputs,mask,temporal,times,noise
        result=dict(status='passed',k=a.k,checkpoint_every=a.every,rows=rows,
                    mean_compute_seconds=sum(x['compute_seconds'] for x in rows[1:])/len(rows[1:]),
                    scope='single GPU alongside cache extraction; timing excludes input and four-DP communication')
    except torch.OutOfMemoryError as e:
        result=dict(status='oom',k=a.k,checkpoint_every=a.every,rows=rows,error=str(e))
    write_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
