"""Automatic paired g0/g1.5 held-out evaluation of overnight camera ablations."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

import evaluate_development32 as engine
import multiview_inference as inference
from data import Data432, sha, write_json
from diagnose_11900 import acquire_device
from camera_temporal_adapter import attach
from paper_metric_math import frechet_features
import overnight_camera_abc as campaign

SCRIPT=Path(__file__).resolve()
ORIGINAL_PIPELINE=inference.CameraVideoInference
ORIGINAL_SAMPLE=inference.sample_target
ACTIVE_GUIDANCE=0.


class Pipeline(ORIGINAL_PIPELINE):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__(checkpoint,device=device)
        folder=Path(checkpoint).parent
        sidecar=folder/'camera_adapter.safetensors'
        if sidecar.exists():
            manifest=json.loads((folder/'checkpoint_manifest.json').read_text())
            assert manifest['arm'] in ['B','C']
            assert sha(sidecar)==manifest['adapter_sha256']
            attach(self.model,temporal=manifest['arm']=='C')
            self.model.cam_encoder_adapter.load_state_dict(load_file(str(sidecar)),strict=True)
            assert torch.count_nonzero(self.model.cam_encoder_adapter.out.weight)>0
            self.model.eval().requires_grad_(False)
        self.guidance=ACTIVE_GUIDANCE

    def predict_velocity(self,**kwargs):
        conditional=super().predict_velocity(**kwargs)
        if self.guidance==0:return conditional
        unconditional=super().predict_velocity(**(kwargs|dict(crossattn_emb=torch.zeros_like(kwargs['crossattn_emb']))))
        return conditional+self.guidance*(conditional-unconditional)


def sample(*args,**kwargs):
    z,info=ORIGINAL_SAMPLE(*args,**kwargs)
    info.update(guidance=ACTIVE_GUIDANCE,cfg_formula='conditional + g*(conditional - zero_text)',
                text_condition_present=True,unconditional_reference_camera='same as conditional')
    return z,info


def root_for(arm):return campaign.OUT/arm/'evaluations'/f"step_{int(os.environ['CONTINUE_EVAL_UPDATE']):05d}"


def setup(arm):
    cp=campaign.protocol();root=root_for(arm);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'setup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    p=json.loads((campaign.DEV/'protocol.json').read_text())
    if arm=='guidance':
        arms=[dict(a,guidance=1.5,name=a['name']+'_g1p5') for a in p['arms'] if a['name'] in ['single_10000','dual_11900']]
    else:
        ck=campaign.OUT/arm/'train'/f"step_{int(os.environ['CONTINUE_EVAL_UPDATE']):05d}"/'trainable_weights.safetensors'
        manifest=json.loads((ck.parent/'checkpoint_manifest.json').read_text())
        assert sha(ck)==manifest['sha256']
        arms=[dict(name=f'{arm}_{gname}',step=11000,checkpoint=str(ck),checkpoint_sha256=manifest['sha256'],
                   references=['cam02'],guidance=g,checkpoint_metadata=dict(arm=arm,training_updates=1000))
              for gname,g in [('g0',0.)]]
    for c in p['cases']:c['worker']=p['scene_ids'].index(c['scene_id'])%2
    p.update(arms=arms,version='overnight_abc_evaluation_v1',guidance='per-arm 0 or 1.5',
        cases_per_arm=64,generated_videos=128,allowed_gpus=campaign.GPUS[arm],
        arm=arm,training_protocol_sha256=sha(campaign.OUT/'protocol.json'),
        original_development_protocol_sha256=sha(campaign.DEV/'protocol.json'),
        reference_count='1 for ABC; original 11900 baseline uses 2',
        source_sha256=p['source_sha256']|cp['source_sha256'])
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:write_json(path,p)
    conditions=root/'conditions'
    if not conditions.exists():conditions.symlink_to(campaign.DEV/'conditions',target_is_directory=True)
    return p


def status(root,shard,stage,**kw):
    write_json(root/f'workers/{shard}/status.json',dict(stage=stage,shard=shard,pid=os.getpid(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),updated_unix=time.time(),**kw))


def fvd(data,p,selected,root,shard):
    assert sha(engine.I3D)==p['fvd']['weights_sha256']
    detector=torch.jit.load(str(engine.I3D),map_location='cuda').eval()
    for case in selected:
        for name in ['real_target','reference_copy',*[a['name'] for a in p['arms']]]:
            path=root/'fvd'/name/case['scene']/'features.json'
            if path.exists():continue
            status(root,shard,'fvd_features',candidate=name,scene=case['scene'])
            if name=='real_target':pixels=data.read_rgb(case['scene'],'cam06')
            elif name=='reference_copy':pixels=data.read_rgb(case['scene'],'cam02')
            else:pixels=engine.video(engine.destination(name,case)/'generated.mp4')
            rows=[]
            for start in [0,32]:
                x=torch.from_numpy(pixels[start:start+49].copy()).permute(3,0,1,2)[None].contiguous().cuda().float()
                assert x.is_contiguous()
                value=detector(x,rescale=True,resize=True,return_features=True)
                assert value.shape==(1,400) and torch.isfinite(value).all()
                rows.append(value[0].cpu().tolist());del x,value
            write_json(path,dict(case=case,candidate=name,features=rows));del pixels
    del detector;gc.collect();torch.cuda.empty_cache()


def report(arm):
    root=root_for(arm)
    if not (root/'protocol.json').exists():return
    engine.OUT=root;engine.report()
    lock=(root/'report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    p=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'SUMMARY.json').read_text())
    states={str(i):json.loads((root/f'workers/{i}/status.json').read_text()) for i in range(2)
            if (root/f'workers/{i}/status.json').exists()}
    complete=len(states)==2 and all(s['stage']=='completed' for s in states.values())
    if complete:
        for name,s in summary['groups']['all'].items():
            assert s['pose_cases']==64 and s['fvd_cases']==64
            if name not in ['real_target','reference_copy']:assert s['pixel_cases']==64
    summary.update(status='completed' if complete else 'partial',workers=states)
    write_json(root/'SUMMARY.json',summary)
    body=(root/'REPORT.md').read_text()
    body=body.replace('单参考5K/8K/9K/10K可比较训练时长；11900经过双参考课程，不能当作单参考多训练1900步。',
        '本轮为相机结构A/B/C或原权重guidance对照；参见各组protocol.json。')
    body=body.replace('所有组guidance0、35步、seed456；旧guidance7结果不混用。',
        '各组guidance按名称为0或1.5，35步、seed456；只清空文字构造无条件分支，相机和参考不变。')
    if complete:body=body.replace('状态：等待/进行中','状态：完成')
    (root/'REPORT.md').write_text(body)
    # Independent continuation output; preserve original A/B/C campaign reports.


def campaign_report():
    root=campaign.OUT
    lock=(root/'findings.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    base=json.loads((campaign.DEV/'SUMMARY.json').read_text())
    groups={k:{n:v for n,v in values.items() if n in ['single_10000','dual_11900']}
            for k,values in base['groups'].items()}
    completed=[]
    for arm in ['guidance','A','B','C']:
        file=root_for(arm)/'SUMMARY.json'
        if not file.exists():continue
        s=json.loads(file.read_text())
        for domain in groups:
            groups[domain].update({n:v for n,v in s['groups'][domain].items() if n not in ['real_target','reference_copy']})
        if s['status']=='completed':completed.append(arm)
    input_audit={}
    histories={}
    for arm in ['A','B','C']:
        for rank in range(2):
            file=root/arm/f'train/rank{rank}.jsonl'
            if file.exists():histories[arm,rank]=[json.loads(l) for l in file.read_text().splitlines()]
    if len(histories)==6:
        compared=min(map(len,histories.values()))
        for rank in range(2):
            for i in range(compared):
                rows=[histories[a,rank][i] for a in ['A','B','C']]
                keys=['update','scene','reference','target','sigma','noise_seed','text_dropped','caption_sha256','text_sha256']
                if not all(all(r[k]==rows[0][k] for k in keys) for r in rows):
                    raise RuntimeError(f'A/B/C training inputs differ at rank{rank}, row{i}')
        input_audit=dict(paired_updates_checked_per_rank=compared,all_scenes_cameras_sigmas_noise_text_identical=True)
    write_json(root/'SUMMARY.json',dict(status='completed' if len(completed)==4 else 'partial',
        completed_groups=completed,groups=groups,training_input_audit=input_audit,updated_unix=time.time()))
    lines=['# 多场景相机A/B/C：自动汇总','',
        '由10K单参考权重出发，每组1000步、DP2有效batch2，训练样本/噪声/文本一致；原10K和11900的g0来自上一轮固定开发评测。新结果按g0/g1.5分别比较，静态/动态分开报告。','']
    for domain,values in groups.items():
        lines += [f'## {domain}','','| 组别 | 视频数 | PSNR↑ | SSIM↑ | TransErr均值m²↓ | RotErr均值°↓ | FVD↓ |','|---|---:|---:|---:|---:|---:|---:|']
        def fmt(v):return '—' if v is None else f'{v:.4f}'
        for name,v in values.items():
            lines.append(f"| {name} | {v['pixel_cases']}/{v['expected_cases']} | {fmt(v['psnr_db'])} | {fmt(v['ssim'])} | {fmt(v['trans_err_mean_m2'])} | {fmt(v['rot_err_mean_deg'])} | {fmt(v['fvd'])} |")
        lines.append('')
    lines += ['部分结果的覆盖可能不同，等完整同一案例再作结论。位姿为VGGT逐帧诊断，TransErr同时保存均值/求和；不是论文动态ViPE，FVD真实分布也不同。','']
    for a in ['guidance','A','B','C']:lines.append(f'- {a}：[视频]({a}/evaluation/index.html) / [指标]({a}/evaluation/REPORT.md)')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(arm,shard):
    global ACTIVE_GUIDANCE
    p=setup(arm);root=root_for(arm);engine.OUT=root
    folder=root/f'workers/{shard}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=[c for c in p['cases'] if c['worker']==shard];assert len(selected)==32
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.set_grad_enabled(False)
    engine.status=lambda worker,stage,**kw:status(root,shard,stage,**kw)
    inference.CameraVideoInference=Pipeline;inference.sample_target=sample
    status(root,shard,'acquiring_gpu');acquire_device(50)
    data=Data432();engine.conditions(data,selected)
    for a in p['arms']:
        ACTIVE_GUIDANCE=a['guidance']
        engine.generate(data,p,selected,a,shard)
        engine.decode(data,p,selected,a,shard)
        report(arm)
    engine.pose(data,p,selected,shard)
    fvd(data,p,selected,root,shard)
    status(root,shard,'completed',cases=32)
    report(arm)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A','B','C','guidance'],required=True)
    parser.add_argument('--shard',type=int,choices=[0,1]);parser.add_argument('--report',action='store_true');a=parser.parse_args()
    try:
        if a.report:report(a.arm)
        elif a.shard is not None:run(a.arm,a.shard)
        else:parser.error('Choose shard or report')
    except BaseException as error:
        if a.shard is not None:status(root_for(a.arm),a.shard,'failed',error=repr(error),traceback=traceback.format_exc())
        raise
