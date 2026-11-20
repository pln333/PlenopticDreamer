"""Four-replica mixed full-corpus Basic curriculum at 432x768 with official-tokenizer caches.

DP (not context parallelism), batch one per rank. New source-derived camera layers.
Trainable gradients explicitly averaged after backward, with DDP overlap disabled.
"""
import argparse,json,os,random,time,traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file
from data import Data432 as CachedBasicData, MANIFEST, JOB, select
from text import ReferenceTextEncoder
from multiscene_data import ROOT,digest,write_json
from camera_backbone import load_camera_backbone,use_wan_mixed_precision
from context_layout import make_flow_inputs,flow_loss


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x

def gpu_yield_probe(local):
    torch.cuda.set_device(local)
    x=torch.randn(4096,4096,device='cuda',dtype=torch.float16)
    start=time.monotonic()
    while time.monotonic()-start<90:
        y=x@x; torch.cuda.synchronize()
        if torch.cuda.mem_get_info()[0]>68*1024**3: break
    free=torch.cuda.mem_get_info()[0]; del x,y; torch.cuda.empty_cache()
    if free<68*1024**3: raise RuntimeError('vLLM has not yielded enough free memory')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',default=str(MANIFEST))
    p.add_argument('--output',required=True); p.add_argument('--k1-steps',type=int,default=10000); p.add_argument('--k2-steps',type=int,default=4000)
    p.add_argument('--smoke-mixed',action='store_true'); p.add_argument('--save-every',type=int,default=500); p.add_argument('--seed',type=int,default=20261117); p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-mode',default='block_wise'); p.add_argument('--checkpoint-every',type=int,default=1)
    p.add_argument('--benchmark',action='store_true'); p.add_argument('--cached-smoke',action='store_true'); p.add_argument('--warmup-gpu',action='store_true'); p.add_argument('--stop-after',type=int); p.add_argument('--prefetch',action='store_true')
    a=p.parse_args(); local=int(os.environ.get('LOCAL_RANK','0')); rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    torch.set_num_threads(6); torch.manual_seed(a.seed); random.seed(a.seed)
    if a.warmup_gpu: gpu_yield_probe(local)
    else: torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15),device_id=torch.device('cuda',local))
    out=(ROOT/a.output).resolve(); out.relative_to(Path('/root/wenfy/QAnchor4D_D4RT_test')); out.mkdir(parents=True,exist_ok=True)
    logpath=out/f'rank{rank}.jsonl'
    if logpath.exists() and not a.resume: raise FileExistsError('preserve existing run; explicit resume needed')
    data=CachedBasicData(ROOT/a.manifest); manifest_hash=digest(data.manifest_path)
    if not a.cached_smoke:
        if world!=4 and not a.benchmark: raise ValueError('formal batch requires four ranks')
    model,loading=load_camera_backbone()
    use_wan_mixed_precision(model); model=model.cuda().train()
    from cosmos_transfer2._src.predict2.camera.networks.dit_multiview_camera_ar import SACConfig
    from cosmos_transfer2._src.predict2.schedulers.rectified_flow import RectifiedFlow
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    if a.checkpoint_mode=='attention_only':
        from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts
        def attention_context():
            def policy(ctx,func,*args,**kwargs):
                heavy=('scaled_dot_product' in str(func) or 'flash_attn' in str(func))
                return CheckpointPolicy.MUST_SAVE if heavy else CheckpointPolicy.PREFER_RECOMPUTE
            return create_selective_checkpoint_contexts(policy)
        sac=SACConfig(mode='block_wise',every_n_blocks=a.checkpoint_every)
        sac.get_context_fn=lambda:attention_context
    else: sac=SACConfig(mode=a.checkpoint_mode,every_n_blocks=a.checkpoint_every)
    model.enable_selective_checkpoint(sac,model.blocks)
    named=[(n.replace('._checkpoint_wrapped_module',''),p) for n,p in model.named_parameters() if p.requires_grad]
    # GPU FP32 master weights/moments replace the old per-step CPU transfers.
    masters=[torch.nn.Parameter(param.detach().float().clone()) for _,param in named]
    optimizer=torch.optim.AdamW(masters,lr=2e-5,betas=(.9,.999),weight_decay=.01,fused=True)
    rf=RectifiedFlow(None,train_time_distribution='logitnormal',shift=5,train_time_weight_method='reweighting',device=torch.device('cuda',local),dtype=torch.float32)
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64)
    if a.cached_smoke: raise ValueError('Use --smoke-mixed --benchmark for this full-corpus loader')
    text_encoder=ReferenceTextEncoder()
    probes={g:next((n,param) for n,param in named if g in n) for g in ['self_attn','cam_encoder']}
    before={g:param.detach().float().clone() for g,(n,param) in probes.items()}
    frozen_name,frozen=next((n,param) for n,param in model.named_parameters() if not param.requires_grad and param.numel()>1000)
    frozen_before=frozen.detach().clone()
    torch.manual_seed(a.seed+rank+100); random.seed(a.seed+rank+100)
    padding=torch.zeros(1,1,54,96,device='cuda'); start_step=0
    recipe=dict(world=world,dp=world,cp=1,batch_per_rank=1,effective_batch=world,k1_steps=a.k1_steps,k2_steps=a.k2_steps,lr=2e-5,adam_betas=[.9,.999],weight_decay=.01,grad_clip=1.,checkpoint_mode=a.checkpoint_mode,checkpoint_every=a.checkpoint_every,manifest_sha256=manifest_hash,base_sha256=loading['base_sha256'],seed=a.seed,online='none; deterministic future samples drive extraction priority; wait for exact 432 cache',text='on-demand first-reference captions plus Cosmos embeddings',prefetch=False,frames=81,resolution=[432,768],smoke_mixed=a.smoke_mixed,trainable_rule='self_attn or cam_encoder',optimizer='GPU fused AdamW; FP32 masters and moments; same optimizer across curriculum',initialization='released Cosmos-Predict2.5-2B base plus locally initialized missing camera layers; no old finetuned weights',source_sha256={str(path):digest(path) for path in [Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('text.py'),ROOT/'scripts/plenoptic/camera_backbone.py',ROOT/'scripts/plenoptic/context_layout.py']})
    recipe['recovery_runtime_sha256']={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('entry.py'),Path(__file__).with_name('text.py')]}
    recipe['gradient_synchronization']='DDP no_sync forward/backward; explicit synchronous bucket mean after backward; stable parameter order'
    if a.resume:
        state=torch.load(out/'training_state.pt',map_location='cpu',weights_only=False)
        saved_recipe=dict(state['recipe']); saved_recipe.setdefault('prefetch',False); saved_recipe.setdefault('recovery_runtime_sha256',recipe['recovery_runtime_sha256'])
        saved_recipe.setdefault('gradient_synchronization',recipe['gradient_synchronization'])
        if saved_recipe!=recipe or state['names']!=[n for n,p in named]: raise ValueError('resume recipe differs')
        start_step=state['step']
        with torch.no_grad():
            for (_,param),master,saved in zip(named,masters,state['masters']): master.copy_(saved); param.copy_(master.to(param.dtype))
        optimizer.load_state_dict(state['optimizer'])
        # Reconcile only an explicitly documented suffix beyond the last atomic save.
        existing=[json.loads(line) for line in logpath.read_text().splitlines()]
        if len(existing)!=start_step: raise ValueError('history has unsaved suffix; explicit recovery audit required')
        rs=torch.load(out/f'step_{start_step:05d}'/f'rng_rank{rank}.pt',map_location='cpu',weights_only=False)
        if rs['step']!=start_step: raise ValueError('rank RNG checkpoint differs')
        torch.set_rng_state(rs['torch']); torch.cuda.set_rng_state(rs['cuda']); random.setstate(rs['python'])
    elif rank==0: write_json(out/'recipe.json',recipe)

    before={g:param.detach().float().clone() for g,(n,param) in probes.items()}

    def save(step,k):
        ck=out/f'step_{step:05d}'; ck.mkdir(exist_ok=True)
        dist.barrier()
        tmp=ck/f'rng_rank{rank}.pt.tmp'; torch.save(dict(step=step,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate()),tmp); tmp.replace(ck/f'rng_rank{rank}.pt')
        if rank==0:
            ck=out/f'step_{step:05d}'; ck.mkdir(exist_ok=True)
            path=ck/'trainable_weights.safetensors'
            save_file({n:param.detach().cpu().contiguous() for n,param in named},str(path.with_suffix('.tmp')),metadata=dict(base_sha256=loading['base_sha256'],context_references=str(k),local_training_steps=str(step),manifest_sha256=manifest_hash,scope='local mixed full-corpus four-DP 432x768 Basic k1/k2 reconstruction; official Wan codec; 81 source frames, not official author checkpoint'))
            path.with_suffix('.tmp').replace(path)
            write_json(ck/'checkpoint_manifest.json',dict(sha256=digest(path),checkpoint=str(path),local_steps=step,manifest_sha256=manifest_hash))
            state=dict(step=step,recipe=recipe,names=[n for n,p in named],masters=[p.detach().cpu() for p in masters],optimizer=cpu_tree(optimizer.state_dict()))
            torch.save(state,out/'training_state.pt.tmp'); (out/'training_state.pt.tmp').replace(out/'training_state.pt')
            write_json(out/'status.json',dict(status='running',saved_step=step,k=k,recipe=recipe,updated_unix=time.time()))
        dist.barrier()
    if not a.benchmark and not a.resume: save(0,1)
    total=a.k1_steps+a.k2_steps; permutation=[]; current_epoch=-1
    def selection(step):
        return select(data.train,step,rank,world,a.k1_steps,a.seed)
    with logpath.open('a') as log:
        for step in range(start_step,total):
            k=1 if step<a.k1_steps else 2
            scene,selected,epoch=selection(step);refs,target=selected[:-1],selected[-1]
            dist.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            reference,gt,rays,text=data.sample(scene,refs,target,text_encoder)
            torch.cuda.synchronize(); loaded=time.perf_counter()
            noise=torch.randn((1,16,21*(k+1),54,96),device='cuda')
            times=rf.get_discrete_timestamp(rf.sample_train_time(1),{'device':'cuda','dtype':torch.float32})
            sigma=rf.get_sigmas(times,{'device':'cuda','dtype':torch.float32})
            prefix=6 if random.random()<.45 else 0; dropped=random.random()<.2
            inputs=make_flow_inputs(reference,gt,sigma,noise,target_prefix_frames=prefix)
            mask=inputs['clean_mask'].expand(-1,-1,-1,54,96).contiguous(); temporal=inputs['clean_mask'][:,0,:,0,0]
            model_times=temporal*.1+(1-temporal)*(sigma*1000)
            with wrapped.no_sync():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=wrapped(x_B_C_T_H_W=inputs['model_input'].to(torch.bfloat16),timesteps_B_T=model_times,crossattn_emb=torch.zeros_like(text) if dropped else text,condition_video_input_mask_B_C_T_H_W=mask,padding_mask=padding,camera=rays)
                loss=flow_loss(prediction,inputs['target_velocity'],inputs['clean_mask'])
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward()
            torch.cuda.synchronize()
            # Keep every collective in the same parameter order on every rank.
            # Preserve gradient dtype, average across DP, then use FP32 masters.
            from torch._utils import _flatten_dense_tensors,_unflatten_dense_tensors
            buckets={}
            for name,param in named:
                if param.grad is None: raise RuntimeError('missing gradient before explicit sync: '+name)
                buckets.setdefault(param.grad.dtype,[]).append(param.grad)
            for dtype,grads in buckets.items():
                chunk=[];size=0
                for grad in grads+[None]:
                    if chunk and (grad is None or size+grad.numel()*grad.element_size()>64*1024**2):
                        flat=_flatten_dense_tensors(chunk)
                        dist.all_reduce(flat,op=dist.ReduceOp.SUM)
                        flat.div_(world)
                        for destination,value in zip(chunk,_unflatten_dense_tensors(flat,chunk)): destination.copy_(value)
                        chunk=[];size=0
                    if grad is not None:
                        chunk.append(grad);size+=grad.numel()*grad.element_size()
            torch.cuda.synchronize()
            for (_,param),master in zip(named,masters):
                if param.grad is None: raise RuntimeError('missing gradient')
                master.grad=param.grad.detach().float()
            norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True); optimizer.step()
            with torch.no_grad():
                for (_,param),master in zip(named,masters): param.copy_(master.to(param.dtype))
            optimizer.zero_grad(set_to_none=True); wrapped.zero_grad(set_to_none=True); torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            entry=dict(step=step+1,k=k,rank=rank,dataset=data.scenes[scene]['dataset'],latent_sources=data.last_latent_sources,scene=scene,references=refs,target=target,epoch=epoch,loss=loss.item(),sigma=sigma.item(),prefix=prefix,text_dropped=dropped,gradient_norm=norm.item(),load_seconds=loaded-start,seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            log.write(json.dumps(entry)+'\n'); log.flush(); print(json.dumps(entry),flush=True)
            if rank==0: write_json(out/'live.json',dict(status='training',step=step+1,k=k,updated_unix=time.time(),rank0=entry))
            del reference,gt,rays,text,noise,inputs,mask,prediction,loss,temporal,model_times
            if not a.benchmark and ((step+1)%a.save_every==0 or step+1 in [a.k1_steps,total] or step+1==a.stop_after): save(step+1,k)
            if step==start_step or step+1==total or (step+1)%100==0:
                # Ensure all replicas really update both trainable groups and stay synchronized.
                updates={g:(param.detach().float()-before[g]).abs().max().item() for g,(n,param) in probes.items()}
                if any(v<=0 for v in updates.values()) or not torch.equal(frozen,frozen_before): raise RuntimeError('freeze/update invariant failed')
                fingerprint=torch.stack([param.detach().float().sum() for _,param in probes.values()]); gathered=[torch.empty_like(fingerprint) for _ in range(world)]; dist.all_gather(gathered,fingerprint)
                if any(not torch.equal(gathered[0],x) for x in gathered): raise RuntimeError('replica parameters differ')
                import hashlib
                checksum=hashlib.sha256()
                for name,param in named:
                    checksum.update(name.encode());checksum.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                hashes=[None]*world;dist.all_gather_object(hashes,checksum.hexdigest())
                if len(set(hashes))!=1: raise RuntimeError('full trainable tensor checksum differs across ranks')
                write_json(out/f'checks_rank{rank}.json',dict(step=step+1,updates=updates,frozen_unchanged=True,replicas_agree=True,full_trainable_sha256=hashes[0],frozen_probe=frozen_name))
            if a.stop_after and step+1>=a.stop_after:
                if rank==0: write_json(out/'status.json',dict(status='bounded_stop',step=step+1,recipe=recipe))
                dist.destroy_process_group(); return
    if rank==0: write_json(out/'status.json',dict(status='completed',step=total,recipe=recipe,updated_unix=time.time()))
    dist.destroy_process_group()

if __name__=='__main__': main()
