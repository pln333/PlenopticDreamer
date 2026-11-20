"""Multi-scene paired camera ablations with automatic full development evaluation."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from data import Data432, JOB, sha, write_json
from context_layout import ContextLayout, make_flow_inputs, flow_loss
from multiview_inference import load_inference_backbone
from camera_temporal_adapter import attach, verify

LAB = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT = JOB / 'a_continue_batch40_20261120'
BASE = JOB / 'multiscene_camera_abc_20261120/A/train/step_01000/trainable_weights.safetensors'
PARENT_STATE = BASE.parent / 'training_state.pt'
DEV = JOB / 'development32_checkpoint_comparison'
SEED = 20261120
GPUS = dict(A=[0,1,2,3], B=[4,5], C=[6,7], guidance=[0,1])


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def protocol():
    p = json.loads((OUT/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items(): assert sha(Path(path))==digest, path
    assert sha(OUT/'schedule.json') == p['schedule_sha256']
    return p


def status(arm,stage,**kw):
    write_json(OUT/arm/'status.json',dict(stage=stage,arm=arm,pid=os.getpid(),updated_unix=time.time(),**kw))


def prepare():
    from cosmos_transfer2._src.predict2.schedulers.rectified_flow import RectifiedFlow
    assert not (OUT/'protocol.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    checks=verify(); data=Data432()
    train=data.train.copy(); random.Random(SEED).shuffle(train); chosen=train[:2000]
    assert len(chosen)==2000 and len(set(chosen))==2000
    protected={s.rsplit('/',1)[1] for s in data.val}
    assert not any(s.rsplit('/',1)[1] in protected for s in chosen)
    gen=torch.Generator().manual_seed(SEED+1)
    rf=RectifiedFlow(None,train_time_distribution='logitnormal',shift=5,
        train_time_weight_method='reweighting',device=torch.device('cpu'),dtype=torch.float32)
    normal=torch.randn(2000,generator=gen)
    sigmas=rf.get_sigmas(rf.get_discrete_timestamp(normal.sigmoid(),{}),{'device':'cpu','dtype':torch.float32}).tolist()
    rng=random.Random(SEED+2)
    high={rank:set(rng.sample(range(1000),200)) for rank in range(2)}
    rows=[]; missing=[]; existing_captions=0
    for i,scene in enumerate(chosen):
        step,rank=divmod(i,2); ref,target=rng.sample([f'cam{j:02d}' for j in range(1,11)],2)
        for c in [ref,target]:
            if not data.cache_path(scene,c).exists():missing.append(str(data.cache_path(scene,c)))
        if (JOB/'captions'/scene/f'{ref}.json').exists():existing_captions+=1
        rows.append(dict(update=step+1,rank=rank,scene=scene,reference=ref,target=target,
            dataset=data.scenes[scene]['dataset'],noise_seed=SEED+100000+i,
            text_dropped=rng.random()<.2,used_high_noise=step in high[rank],
            sigma=rng.uniform(.98,.9999) if step in high[rank] else sigmas[i]))
    assert not missing, f'Missing pinned video caches: {missing[:5]}'
    write_json(OUT/'schedule.json',rows)
    deps=[SCRIPT, SCRIPT.with_name('overnight_camera_abc_evaluate.py'),
          SCRIPT.with_name('camera_temporal_adapter.py'),SCRIPT.with_name('data.py'),SCRIPT.with_name('text.py'),
          SCRIPT.with_name('evaluate_development32.py'),SCRIPT.with_name('evaluate_camera_pose432.py')]
    import camera_backbone,context_layout,multiview_inference
    deps += [Path(m.__file__) for m in [camera_backbone,context_layout,multiview_inference]]
    p=dict(version='multiscene_camera_abc_v1',seed=SEED,parent_checkpoint=str(BASE),parent_sha256=sha(BASE),
        parent_formal_step=10000,updates=1000,save_updates=[250,500,1000],world_size=2,
        batch_per_rank=1,effective_batch=2,reference_count=1,learning_rate=2e-5,
        optimizer='Fresh FP32-master fused AdamW for ALL arms; weights-only continuation, not original optimizer resumption',
        betas=[.9,.999],weight_decay=.01,grad_clip=1.,
        high_noise_fraction_per_rank=.2,high_noise_sigma_range=[.98,.9999],base_noise='upstream shift5 logitnormal',
        target_gt_prefix=0,text_drop_probability=.2,frames=81,resolution=[432,768],
        trainable_rule='Original self_attn/cam_encoder including related norms/modulations; B/C also train geometry adapter',
        architecture=dict(A='Unchanged camera network',
            B='Shared pure-ray residual: 1536→64→diagonal temporal attention→1536; zero output; per-frame only',
            C='Same parameters and sinusoidal positions as B; full attention within each 21-frame view; zero output'),
        adapter_placement='Add residual to patchified camera rays BEFORE all existing learned cam_encoder projections; existing trained projections retained',
        adapter_initialization='B/C same seed, parameters and positional encoding; only attention mask differs',
        selection='Uniform shuffled full training-group list, first 2000 groups; independent of cache/caption readiness',
        training_groups=2000,training_scene_ids=len({s.rsplit('/',1)[1] for s in chosen}),
        dataset_counts={k:sum(r['dataset']==k for r in rows) for k in ['multicam','syncam']},
        existing_captions=existing_captions,caption_policy='Reference only; shared existing caption or generate once under per-reference lock; text embedding online per rank, no new full text cache',
        validation_scene_id_overlap=[],manifest_sha256=data.manifest_sha,
        schedule_sha256=sha(OUT/'schedule.json'),validation_protocol=str(DEV/'protocol.json'),
        validation_protocol_sha256=sha(DEV/'protocol.json'),evaluation_guidances=[0,1.5],
        evaluation_cases_per_guidance=64,evaluation_seed=456,evaluation_steps=35,evaluation_shift=5,
        auto_evaluation='After successful 1000-step save and training process exit: two independent evaluation shards on the same two GPUs; videos/PSNR/SSIM/full81frame VGGT/FVD, g0 and g1.5',
        guidance_baselines='GPU0/1 evaluate original 10K single and original11900 dual at g1.5; g0 baseline retained from completed development32',
        gpus=GPUS,module_checks=checks,source_sha256={str(f):sha(f) for f in deps},
        scope='CameraCtrl-inspired method ablation; not official PlenopticDreamer checkpoint reproduction',
        controlled_training_changes_vs_formal='All three arms use high-noise mixture, no target prefix, fresh optimizer, DP2 batch2. A/B/C comparisons isolate architecture, not these common changes.')
    write_json(OUT/'protocol.json',p)
    write_json(OUT/'module_checks.json',checks)
    sources=OUT/'sources';sources.mkdir(exist_ok=True)
    for i,f in enumerate(deps):shutil.copyfile(f,sources/f'{i}_{f.name}')
    for arm in ['A','B','C','guidance']:(OUT/arm).mkdir(exist_ok=True)
    (OUT/'README.md').write_text('''# 多场景相机编码 A/B/C

三组由单参考10K同权重初始化，2000个训练场景配置、1000步、两卡DP、有效batch2，固定同一份样本/噪声/文本丢弃计划。32个验证scene ID及全部焦距版本隔离。每rank恰好20%高噪声，目标无干净前缀，原self-attention/camera参数继续训练。优化器统一重新初始化，未声称恢复10K旧优化器。

A原结构；B新增逐帧纯几何残差；C与B同参数、同位置编码，但允许视角内时间attention。B/C输出层零初始化，保留已训练相机投影；模块从不读取视频latent，时间attention不跨视角。现有小规模实验模块复用，属于方法改动。

GPU2/3=A，4/5=B，6/7=C；GPU0/1补测原10K/11900的guidance1.5。每组250/500/1000步保存；训练进程退出后自动在原两卡各评32个验证案例，最终每组g0/g1.5各64视频，统一文本/目标/种子/采样器。继续自动跑PSNR、SSIM、全81帧VGGT位姿、FVD与GT校准，分MultiCam动态/SynCam静态汇报。评测器与论文动态ViPE不同；TransErr同时保留均值和求和，FVD不直接与论文比较。

进度看各组status.json、train/rank*.json、evaluation/workers/*/status.json。所有后台子任务有日志及失败状态，发生失败会停止本组后续阶段；不会把缺失结果当完成。各组最终evaluation/index.html展示视频，evaluation/REPORT.md与SUMMARY.json展示指标。
''')
    print(json.dumps({k:p[k] for k in ['training_groups','training_scene_ids','dataset_counts','existing_captions','updates','gpus']},indent=2),flush=True)


def acquire(local):
    torch.cuda.set_device(local)
    probe=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)
    start=time.monotonic(); last=-10
    while torch.cuda.mem_get_info()[0] < 68*2**30:
        y=probe@probe;torch.cuda.synchronize()
        if time.monotonic()-last>10:
            print(json.dumps(dict(stage='waiting_vllm_yield',local_rank=local,free_gib=torch.cuda.mem_get_info()[0]/2**30)),flush=True);last=time.monotonic()
        if time.monotonic()-start>180:raise RuntimeError('vLLM has not yielded enough GPU memory')
    del probe
    if 'y' in locals():del y
    torch.cuda.empty_cache()


def sync_gradients(named,world):
    from torch._utils import _flatten_dense_tensors,_unflatten_dense_tensors
    buckets={}
    for name,p in named:
        if p.grad is None:raise RuntimeError('Missing gradient: '+name)
        buckets.setdefault(p.grad.dtype,[]).append(p.grad)
    for grads in buckets.values():
        chunk=[];size=0
        for grad in grads+[None]:
            if chunk and (grad is None or size+grad.numel()*grad.element_size()>64*1024**2):
                flat=_flatten_dense_tensors(chunk);dist.all_reduce(flat);flat.div_(world)
                for dest,value in zip(chunk,_unflatten_dense_tensors(flat,chunk)):dest.copy_(value)
                chunk=[];size=0
            if grad is not None:chunk.append(grad);size+=grad.numel()*grad.element_size()


def train(arm,limit=1000,smoke=False):
    p=protocol(); local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==4 and arm=='A'
    torch.set_num_threads(4);torch.manual_seed(SEED);acquire(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=5),device_id=torch.device('cuda',local))
    dest=OUT/arm/('smoke_train' if smoke else 'train');dest.mkdir(exist_ok=True)
    start=int(os.environ.get('CONTINUE_START','0')); limit=int(os.environ.get('CONTINUE_END',str(limit)))
    logpath=dest/f'rank{rank}.jsonl'
    write_json(dest/f'rank{rank}.json',dict(stage='loading',pid=os.getpid(),rank=rank,updated_unix=time.time()))
    model,loading=load_inference_backbone(BASE)
    torch.manual_seed(SEED+50)
    if arm!='A':attach(model,temporal=arm=='C')
    for name,param in model.named_parameters():param.requires_grad_('self_attn' in name or 'cam_encoder' in name)
    model.train()
    from cosmos_transfer2._src.predict2.camera.networks.dit_multiview_camera_ar import SACConfig
    from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts
    def contexts():
        def policy(ctx,func,*args,**kwargs):
            return CheckpointPolicy.MUST_SAVE if 'scaled_dot_product' in str(func) or 'flash_attn' in str(func) else CheckpointPolicy.PREFER_RECOMPUTE
        return create_selective_checkpoint_contexts(policy)
    sac=SACConfig(mode='block_wise',every_n_blocks=1);sac.get_context_fn=lambda:contexts
    model.enable_selective_checkpoint(sac,model.blocks)
    named=[(n.replace('._checkpoint_wrapped_module',''),v) for n,v in model.named_parameters() if v.requires_grad]
    original=load_file(str(BASE))
    assert all(torch.equal(v.detach().cpu(),original[n]) for n,v in named if n in original)
    assert set(original)=={n for n,v in named if not n.startswith('cam_encoder_adapter.')}
    del original
    masters=[torch.nn.Parameter(v.detach().float().clone()) for _,v in named]
    optimizer=torch.optim.AdamW(masters,lr=2e-5,betas=(.9,.999),weight_decay=.01,fused=True)
    state_path=PARENT_STATE if start==0 else dest/f'step_{start:05d}/training_state.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    assert state['names']==[n for n,v in named]
    assert state['update']==(1000 if start==0 else start)
    for (name,v),m,saved in zip(named,masters,state['masters']):
        m.data.copy_(saved.to(m.device));v.data.copy_(m.to(v.dtype))
    optimizer.load_state_dict(state['optimizer'])
    del state

    from text import ReferenceTextEncoder
    text_encoder=ReferenceTextEncoder()
    data=Data432();layout=ContextLayout(1,21)
    source_rows=json.loads((OUT/'schedule.json').read_text())
    def make_row(update, micro):
        index=(update-1)*40+rank*10+micro
        epoch,offset=divmod(index,len(source_rows))
        order=list(range(len(source_rows)));random.Random(SEED+20000+epoch).shuffle(order)
        row=source_rows[order[offset]].copy()
        row.update(update=update,rank=rank,micro=micro,epoch=epoch,noise_seed=SEED+500000+index)
        return row

    frozen_name,frozen=next((n,v) for n,v in model.named_parameters() if not v.requires_grad and v.numel()>1000)
    frozen_initial=frozen.detach().clone()
    probes={g:next((n,v) for n,v in named if g in n) for g in ['self_attn','cam_encoder']}
    initial={g:v.detach().clone() for g,(n,v) in probes.items()}
    adapter_sha=tensor_sha(torch.cat([v.detach().flatten() for v in model.cam_encoder_adapter.parameters()])) if arm!='A' else None
    write_json(dest/f'initialization_rank{rank}.json',dict(parent_sha256=loading['finetuned_sha256'],
        original_parameters_identical=True,adapter_initial_sha256=adapter_sha,
        adapter_parameters=sum(v.numel() for v in model.cam_encoder_adapter.parameters()) if arm!='A' else 0,
        trainable_parameters=sum(v.numel() for _,v in named),fresh_optimizer=True))
    clean=layout.clean_mask(1,'cuda',target_prefix_frames=0)
    mask=clean.expand(-1,-1,-1,54,96).contiguous();padding=torch.zeros(1,1,54,96,device='cuda')
    assert torch.count_nonzero(mask[:,:,layout.target_slice])==0

    def save(update):
        folder=dest/f'step_{update:05d}';folder.mkdir(exist_ok=True)
        torch.save(dict(update=update,rank=rank,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder/f'rng_rank{rank}.pt')
        dist.barrier()
        if rank==0:
            path=folder/'trainable_weights.safetensors'
            save_file({n:v.detach().cpu().contiguous() for n,v in named if not n.startswith('cam_encoder_adapter.')},str(path.with_suffix('.tmp')),
                metadata=dict(base_sha256=loading['base_sha256'],context_references='1',local_training_steps=str(11000+update),
                    parent_sha256=p['parent_sha256'],arm=arm,scope='multiscene camera A/B/C method ablation'))
            path.with_suffix('.tmp').replace(path)
            manifest=dict(sha256=sha(path),local_steps=update,parent_step=11000,arm=arm,parent_sha256=p['parent_sha256'])
            if arm!='A':
                sidecar=folder/'camera_adapter.safetensors'
                save_file({n:v.detach().cpu().contiguous() for n,v in model.cam_encoder_adapter.state_dict().items()},str(sidecar),metadata=dict(arm=arm,temporal=str(arm=='C')))
                manifest['adapter_sha256']=sha(sidecar)
            write_json(folder/'checkpoint_manifest.json',manifest)
            if update%100==0 or smoke:
                state=dict(update=update,arm=arm,names=[n for n,v in named],masters=[v.detach().cpu() for v in masters],
                    optimizer=cpu_tree(optimizer.state_dict()),protocol_sha256=sha(OUT/'protocol.json'))
                torch.save(state,folder/'training_state.pt.tmp');(folder/'training_state.pt.tmp').replace(folder/'training_state.pt')
                del state
            write_json(dest/'checkpoint_status.json',dict(saved_update=update,checkpoint=str(path),updated_unix=time.time()))
        dist.barrier()

    with logpath.open('a') as log:
        for update in range(start+1,limit+1):
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for micro in range(10):
                row=make_row(update,micro)
                scene=row['scene'];refs=[row['reference']];target=row['target']
                data.last_latent_sources=[]
                reference=data.latent(scene,refs[0])[:,None];gt=data.latent(scene,target);camera=data.camera(scene,refs,target)
                locks=OUT/'caption_locks';locks.mkdir(exist_ok=True)
                lockpath=locks/(hashlib.sha256((scene+'/'+refs[0]).encode()).hexdigest()+'.lock')
                with lockpath.open('a') as caption_lock:
                    fcntl.flock(caption_lock,fcntl.LOCK_EX)
                    caption=text_encoder.caption(data,scene,refs[0])
                text=text_encoder.embedding(caption)
                text_hash=tensor_sha(text)
                sigma=torch.tensor([row['sigma']],device='cuda',dtype=torch.float32)
                noise=torch.randn((1,16,42,54,96),device='cuda',generator=torch.Generator(device='cuda').manual_seed(row['noise_seed']))
                inputs=make_flow_inputs(reference,gt,sigma,noise,target_prefix_frames=0)
                times=clean[:,0,:,0,0]*.1+(1-clean[:,0,:,0,0])*(sigma*1000)
                if update==1 and arm!='A':
                    with torch.no_grad():assert torch.equal(model.cam_encoder_adapter(camera),camera)
                loaded=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(x_B_C_T_H_W=inputs['model_input'].to(torch.bfloat16),timesteps_B_T=times,
                        crossattn_emb=torch.zeros_like(text) if row['text_dropped'] else text,
                        condition_video_input_mask_B_C_T_H_W=mask,padding_mask=padding,camera=camera)
                loss=flow_loss(prediction,inputs['target_velocity'],clean)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/10).backward()
            sync_gradients(named,world)
            adapter_out_grad=float(model.cam_encoder_adapter.out.weight.grad.float().norm()) if arm!='A' else None
            if update==1 and arm!='A':assert adapter_out_grad>0
            for (_,v),m in zip(named,masters):m.grad=v.grad.detach().float()
            norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                for (_,v),m in zip(named,masters):v.copy_(m.to(v.dtype))
            optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);torch.cuda.synchronize()
            record=dict(**row,arm=arm,loss=loss.item(),gradient_norm=norm.item(),text_sha256=text_hash,
                caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),adapter_out_gradient=adapter_out_grad,
                seconds=time.perf_counter()-started,load_seconds=loaded-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            write_json(dest/f'rank{rank}.json',dict(stage='training',update=update,total=limit,record=record,pid=os.getpid(),updated_unix=time.time()))
            del reference,gt,camera,text,noise,inputs,times,prediction,loss
            if update==start+1 or update%100==0:
                assert torch.equal(frozen,frozen_initial)
                changes={g:float((v.detach().float()-initial[g].float()).abs().max()) for g,(n,v) in probes.items()}
                assert all(x>0 for x in changes.values())
                digest=hashlib.sha256()
                for n,v in named:digest.update(n.encode());digest.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                hashes=[None]*world;dist.all_gather_object(hashes,digest.hexdigest());assert len(set(hashes))==1
                write_json(dest/f'checks_rank{rank}.json',dict(update=update,replicas_agree=True,full_trainable_sha256=hashes[0],
                    frozen_probe=frozen_name,frozen_unchanged=True,parameter_changes=changes))
            if update%100==0 or smoke and update==limit:save(update)
    write_json(dest/f'rank{rank}.json',dict(stage='completed',update=limit,total=limit,pid=os.getpid(),updated_unix=time.time()))
    dist.barrier();dist.destroy_process_group()


def supervise(arm):
    p=protocol();folder=OUT/arm
    lock=(folder/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS[arm])),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    records=[]
    if arm!='guidance':
        status(arm,'training')
        # bootstrap dispatches --distributed into torchrun using this exact venv.
        command=[str(LAB/'run.sh'),str(SCRIPT.with_name('bootstrap.py')),str(SCRIPT),'--distributed',arm]
        with (folder/'train.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            records.append(dict(stage='train',pid=child.pid,command=command));write_json(folder/'launch.json',dict(workers=records))
            if child.wait()!=0:raise RuntimeError(f'{arm} training failed; inspect train.log')
        for rank in range(2):assert json.loads((folder/f'train/rank{rank}.json').read_text())['stage']=='completed'
        assert (folder/'train/step_01000/checkpoint_manifest.json').exists()
    status(arm,'evaluating')
    children=[]
    for shard,gpu in enumerate(GPUS[arm]):
        command=[str(LAB/'run.sh'),str(SCRIPT.with_name('bootstrap.py')),str(SCRIPT.with_name('overnight_camera_abc_evaluate.py')),
            '--arm',arm,'--shard',str(shard)]
        with (folder/f'evaluate_shard{shard}.log').open('x') as log:
            child=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);records.append(dict(stage='evaluate',shard=shard,gpu=gpu,pid=child.pid,command=command))
    write_json(folder/'launch.json',dict(workers=records))
    while any(c.poll() is None for c in children):time.sleep(5)
    if any(c.returncode!=0 for c in children):raise RuntimeError(f'{arm} final evaluation failed')
    status(arm,'completed',train_updates=1000 if arm!='guidance' else None)
    from overnight_camera_abc_evaluate import report
    report(arm)


def launch():
    p=protocol();assert not (OUT/'launch.json').exists()
    workers=[]
    for arm in ['guidance','A','B','C']:
        command=[str(LAB/'run.sh'),str(SCRIPT.with_name('bootstrap.py')),str(SCRIPT),'--supervise',arm]
        with (OUT/arm/'supervisor.log').open('x') as log:
            child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS[arm])),
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(arm=arm,gpus=GPUS[arm],pid=child.pid,command=command))
        write_json(OUT/'launch.json',dict(workers=workers,started_unix=time.time()))
    print(json.dumps(workers,indent=2),flush=True)


def distributed(arm,smoke=False):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        str(SCRIPT.with_name('bootstrap.py')),str(SCRIPT),'--train',arm]
    if smoke:command+=['--smoke']
    os.execv(sys.executable,command)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--train',choices=['A','B','C']);parser.add_argument('--distributed',choices=['A','B','C'])
    parser.add_argument('--supervise',choices=['A','B','C','guidance']);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    try:
        if a.prepare:prepare()
        elif a.launch:launch()
        elif a.supervise:supervise(a.supervise)
        elif a.distributed:distributed(a.distributed,a.smoke)
        elif a.train:train(a.train,limit=2 if a.smoke else 1000,smoke=a.smoke)
        else:parser.error('Choose action')
    except BaseException as error:
        arm=a.supervise or a.train
        if arm:status(arm,'failed',error=repr(error),traceback=traceback.format_exc())
        raise
