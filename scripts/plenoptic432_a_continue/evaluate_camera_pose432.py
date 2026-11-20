"""VGGT camera diagnostics, with scale anchored by the two actual input cameras.

Dynamic per-frame diagnostic, explicitly not the paper's ViPE evaluation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from data import Data432,JOB,sha,write_json
from diagnose_11900 import acquire_device

LAB=Path(__file__).resolve().parents[2]
SVD=Path('/root/wenfy/SVD')
OUT=JOB/'diagnosis_11900/camera_pose_B600'
MOTION=JOB/'diagnosis_11900/B600_motion_separation'
FIT=JOB/'diagnosis_11900/learnability_continue600/B/evaluation_00600'
NEW=JOB/'diagnosis_11900/B600_new_cameras'
SCENE='multicam/f24_aperture5/scene2316'
WEIGHTS=Path('/root/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors')
REFS=['cam10','cam03']


def homogeneous(x):
    x=np.asarray(x,dtype=np.float64)
    result=np.broadcast_to(np.eye(4),(*x.shape[:-2],4,4)).copy();result[...,:3,:]=x[...,:3,:]
    return result


def pose_scores(predicted,requested,reference_predicted,reference_requested):
    p,g,rp,rg=map(lambda x:np.asarray(x,np.float64),(predicted,requested,reference_predicted,reference_requested))
    assert p.shape==g.shape==rp.shape==rg.shape and p.shape[1:]==(4,4)
    assert all(np.isfinite(x).all() for x in [p,g,rp,rg])
    for x in [p,g,rp,rg]:
        rotations=x[:,:3,:3]
        assert np.allclose(rotations@rotations.transpose(0,2,1),np.eye(3),atol=5e-4)
        assert np.allclose(np.linalg.det(rotations),1,atol=5e-4)
    baseline=rg[:,:3,3];estimated=rp[:,:3,3]
    valid=np.linalg.norm(baseline,axis=-1)>.05
    denominator=np.square(estimated[valid]).sum()
    raw_scale=float(np.sum(baseline[valid]*estimated[valid])/denominator) if denominator>1e-12 else None
    scale=raw_scale if raw_scale is not None and raw_scale>0 else None
    angle=np.arccos(np.clip((np.einsum('nij,nij->n',p[:,:3,:3],g[:,:3,:3])-1)/2,-1,1))
    result=dict(frames=len(p),scale_from_input_references=scale,scale_fit_frames=int(valid.sum()),
        scale_status='ok' if scale is not None else 'invalid_or_nonpositive_reference_scale',
        rot_err_mean_rad=float(angle.mean()),rot_err_sum_rad=float(angle.sum()),
        rot_err_mean_deg=float(np.degrees(angle.mean())),rot_err_end_deg=float(np.degrees(angle[-1])),
        per_frame_rot_err_rad=angle.tolist())
    if scale is not None:
        squared=np.square(scale*p[:,:3,3]-g[:,:3,3]).sum(1)
        ref_squared=np.square(scale*estimated-baseline).sum(1)
        result.update(trans_err_mean_m2=float(squared.mean()),trans_err_sum_m2=float(squared.sum()),
            position_rmse_m=float(np.sqrt(squared.mean())),position_mean_distance_m=float(np.sqrt(squared).mean()),
            position_end_distance_m=float(np.sqrt(squared[-1])),per_frame_trans_err_m2=squared.tolist(),
            input_reference_translation_rmse_m=float(np.sqrt(ref_squared.mean())))
    else:
        result.update(trans_err_mean_m2=None,trans_err_sum_m2=None,position_rmse_m=None,
            position_mean_distance_m=None,position_end_distance_m=None,per_frame_trans_err_m2=None)
    return result


def selfcheck():
    from scipy.spatial.transform import Rotation
    n=5;g=np.repeat(np.eye(4)[None],n,axis=0);g[:,0,3]=np.arange(n)*.1
    r=g.copy();r[:,1,3]=1
    p=g.copy();rp=r.copy();p[:,:3,3]/=2;rp[:,:3,3]/=2
    s=pose_scores(p,g,rp,r)
    assert abs(s['scale_from_input_references']-2)<1e-10 and s['position_rmse_m']<1e-10
    # Target amplitude error must survive alignment based on input cameras.
    p[:,:3,3]*=2;s=pose_scores(p,g,rp,r);assert s['position_rmse_m']>.2
    p[:,:3,:3]=Rotation.from_euler('y',10,degrees=True).as_matrix()
    s=pose_scores(p,g,rp,r);assert abs(s['rot_err_mean_deg']-10)<1e-6
    # Inverting a direction must not be silently repaired by a negative scale.
    rp[:,:3,3]*=-1;s=pose_scores(p,g,rp,r);assert s['scale_from_input_references'] is None
    print('Pose math: scale gauge, amplitude error, known rotation, invalid scale checks passed.',flush=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    selfcheck();data=Data432();poses,k=data.camera_arrays(SCENE)
    tasks=[]
    for target in [f'cam{i:02d}' for i in range(1,10) if i!=3]:
        root=FIT if target in ['cam01','cam02'] else NEW
        for seed in [456,789]:
            path=root/(f'{target}_seed{seed}_true' if root==FIT else f'{target}_seed{seed}')/'generated.mp4'
            assert path.exists();tasks.append(dict(name=f'B600_{target}_seed{seed}',target=target,seed=seed,video=str(path),video_sha256=sha(path),group='B600'))
    for index,name in [(0,'half_yaw'),(1,'half_pitch')]:
        for seed in [456,789]:
            root=MOTION/f'rotation_worker{index}/{name}_seed{seed}'
            tasks.append(dict(name=f'B600_{name}_seed{seed}',target=None,seed=seed,video=str(root/'generated.mp4'),
                video_sha256=sha(root/'generated.mp4'),requested_c2w=str(root/'requested_c2w.npy'),
                requested_c2w_sha256=sha(root/'requested_c2w.npy'),group='B600_new_rotation'))
    p=dict(estimator='Official VGGT-1B, three-view [cam10, cam03, candidate] independently at each synchronized frame',
        source_sha256=sha(Path(__file__)),weights=str(WEIGHTS),weights_sha256=sha(WEIGHTS),
        vggt_source=str(SVD/'vendor/vggt_metrics'),scene=SCENE,references=REFS,frames=list(range(81)),
        independent_batch_size=2,manifest_sha256=data.manifest_sha,tasks=tasks,
        scale='One positive least-squares scale per candidate video, fit ONLY from estimated cam03-from-cam10 and known INPUT reference baseline across frames >5cm. Never fit candidate scale to target GT.',
        framewise_gauge_warning='VGGT estimates each triplet independently, so gauge can vary with frame; one shared scale may still leave estimator noise. GT calibration and reference-fit residual are reported.',
        transformation='OpenCV relative w2c E_candidate @ inv(E_cam10), requested inv(c2w_target) @ c2w_cam10',
        reductions=['translation squared m2: mean and sum','position RMSE m','rotation rad: mean and sum','rotation degrees: mean and end'],
        controls=['real target video','copy cam10 video'],
        scope='Local VGGT per-frame pose diagnostic for moving cameras, NOT paper dynamic ViPE protocol; no direct comparison to paper table.',
        allowed_gpus=[0,1,2,3])
    write_json(OUT/'protocol.json',p);snap=OUT/'sources';snap.mkdir();shutil.copyfile(__file__,snap/Path(__file__).name)
    print(json.dumps(dict(tasks=len(tasks),weights=p['weights'],scope=p['scope']),indent=2),flush=True)


def read_video(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    value=np.stack(frames);assert value.shape==(81,432,768,3),value.shape
    return value


def images(*frames):
    result=[]
    for frame in frames:
        height=round(frame.shape[0]*518/frame.shape[1]/14)*14
        resized=Image.fromarray(frame).resize((518,height),Image.Resampling.BICUBIC)
        result.append(torch.from_numpy(np.asarray(resized).copy()).permute(2,0,1).float()/255)
    return torch.stack(result)


def run_task(task,model,refs,poses,data,p):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dest=OUT/'cases'/task['name'];dest.mkdir(parents=True,exist_ok=True)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,task=task['name'],pid=os.getpid(),updated_unix=time.time(),**kw))
    if task.get('control')=='real':video=data.read_rgb(SCENE,task['target'])
    elif task.get('control')=='copy':video=refs[0]
    else:
        assert sha(task['video'])==task['video_sha256'];video=read_video(task['video'])
    if task.get('requested_c2w'):
        assert sha(task['requested_c2w'])==task['requested_c2w_sha256'];requested=np.load(task['requested_c2w'])
    else:requested=poses[int(task['target'][3:])-1]
    target_gt=np.linalg.inv(requested)@poses[9]
    ref_gt=np.linalg.inv(poses[2])@poses[9]
    write_json(dest/'inputs.json',task)
    record_path=dest/'poses.json';rows=json.loads(record_path.read_text())['frames'] if record_path.exists() else []
    done={r['frame'] for r in rows};remaining=[f for f in p['frames'] if f not in done]
    for begin in range(0,len(remaining),p['independent_batch_size']):
        selected=remaining[begin:begin+p['independent_batch_size']]
        batch=torch.stack([images(refs[0][f],refs[1][f],video[f]) for f in selected]).cuda()
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.bfloat16):tokens,_=model.aggregator(batch)
            with torch.autocast('cuda',enabled=False):
                enc=model.camera_head([tokens[-1].float()])[-1]
                extrinsic,intrinsic=pose_encoding_to_extri_intri(enc,batch.shape[-2:])
        estimates=homogeneous(extrinsic.float().cpu().numpy())
        for index,frame in enumerate(selected):
            relative=estimates[index]@np.linalg.inv(estimates[index,0])
            assert np.isfinite(relative).all()
            rows.append(dict(frame=frame,predicted_target=relative[2].tolist(),requested_target=target_gt[frame].tolist(),
                predicted_input_reference=relative[1].tolist(),requested_input_reference=ref_gt[frame].tolist()))
        write_json(record_path,dict(status='partial',frames=rows));status('estimating',frames_done=len(rows),frames_total=81)
        del batch,tokens,enc,extrinsic,intrinsic
    rows.sort(key=lambda r:r['frame']);assert [r['frame'] for r in rows]==p['frames']
    score=pose_scores(*[[r[key] for r in rows] for key in ['predicted_target','requested_target','predicted_input_reference','requested_input_reference']])
    write_json(record_path,dict(status='completed',frames=rows))
    write_json(dest/'metrics.json',dict(status='completed',task=task,scores=score,protocol_source_sha256=p['source_sha256']))
    print(json.dumps(dict(task=task['name'],scores={k:v for k,v in score.items() if not k.startswith('per_frame')})),flush=True)
    del video;gc.collect();torch.cuda.empty_cache()


def report():
    records=[];lines=[]
    for path in sorted((OUT/'cases').glob('*/metrics.json')):
        row=json.loads(path.read_text());records.append(row);s=row['scores']
        trans=f"{s['position_rmse_m']:.3f}" if s['position_rmse_m'] is not None else 'NA'
        mse=f"{s['trans_err_mean_m2']:.4f}" if s['trans_err_mean_m2'] is not None else 'NA'
        lines.append(f"| {row['task']['name']} | {trans} | {mse} | {s['rot_err_mean_deg']:.2f} | {s['rot_err_mean_rad']:.4f} |")
    write_json(OUT/'summary.json',dict(cases=records,completed=len(records)))
    body='''# 相机位姿误差：VGGT逐帧诊断

每个同步时刻独立输入[真实cam10、真实cam03、候选目标]，从官方VGGT相机头估计相对位姿。整段视频只用两个已知输入参考之间的平移来标定一个正尺度，生成目标的平移不参与尺度拟合。GT真实目标与复制cam10均作校准。全81帧，位置RMSE单位m，TransErr同时保留逐帧均值和求和（m²），RotErr保留rad与度。

这套是本地动态逐帧诊断，**不是论文动态轨迹使用的ViPE协议**，不能直接与论文表格作差。VGGT本身的尺度/几何估计可能随帧波动，需同时看真实GT校准误差、输入参考拟合残差与视频。不能把所有误差归于生成网络；无效尺度记NA，不填零。

| 视频/候选 | 位置RMSE m↓ | TransErr均值 m²↓ | 旋转均值 °↓ | RotErr均值 rad↓ |
|---|---|---|---|---|
'''+ '\n'.join(lines)+'\n\n完整逐帧预测、请求、尺度与求和在cases/各视频/poses.json、metrics.json。新半幅旋转的请求是真实相机矩阵的旋转插值，像素GT重映射不参与VGGT位姿估计。\n'
    (OUT/'REPORT.md').write_text(body)


def run(watch=False):
    p=json.loads((OUT/'protocol.json').read_text());assert p['source_sha256']==sha(Path(__file__))
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.set_grad_enabled(False)
    write_json(OUT/'status.json',dict(stage='acquiring_gpu',pid=os.getpid(),updated_unix=time.time()));acquire_device(32)
    assert sha(WEIGHTS)==p['weights_sha256'];data=Data432();assert data.manifest_sha==p['manifest_sha256']
    sys.path.insert(0,p['vggt_source']);from vggt.models.vggt import VGGT
    model=VGGT();model.load_state_dict(load_file(str(WEIGHTS)),strict=True);model.eval().cuda()
    poses,k=data.camera_arrays(SCENE);refs=[data.read_rgb(SCENE,c) for c in REFS]
    targets=sorted({t['target'] for t in p['tasks'] if t['target']})
    # Estimate GT and copy controls first, including actual input-reference residuals.
    tasks=[dict(name=f'{control}_{target}',target=target,control=control,group='calibration') for target in targets for control in ['real','copy']]+p['tasks']
    def consume(items):
        for task in items:
            if not (OUT/'cases'/task['name']/'metrics.json').exists():run_task(task,model,refs,poses,data,p)
            report()
    consume(tasks)
    if watch:
        for update in [100,200]:
            evaluation=MOTION/f'translation/evaluation_{update:05d}'
            while not (evaluation/'results.json').exists():
                status_path=MOTION/'status.json'
                if status_path.exists() and json.loads(status_path.read_text()).get('stage')=='failed':raise RuntimeError('Motion diagnosis failed; no queued pose evaluation')
                launch=json.loads((MOTION/'launch.json').read_text());pid=launch['supervisor_pid'];proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or not proc.read_bytes():raise RuntimeError('Motion supervisor exited without queued output')
                write_json(OUT/'status.json',dict(stage='waiting_for_generation',update=update,pid=os.getpid(),updated_unix=time.time()))
                time.sleep(5)
            generated=[]
            for target in ['cam05','cam01','cam02']:
                for seed in [456,789]:
                    path=evaluation/f'{target}_seed{seed}/generated.mp4'
                    generated.append(dict(name=f'B600plus{update}_{target}_seed{seed}',target=target,seed=seed,video=str(path),video_sha256=sha(path),group=f'B600plus{update}'))
            write_json(OUT/f'queued_{update}_inputs.json',generated);consume(generated)
    report();write_json(OUT/'status.json',dict(stage='completed',pid=os.getpid(),updated_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--selfcheck',action='store_true');args=parser.parse_args()
    try:
        if args.prepare:prepare()
        elif args.selfcheck:selfcheck()
        elif args.run:run(args.watch)
        else:parser.error('Choose action')
    except BaseException as error:
        if args.run:write_json(OUT/'status.json',dict(stage='failed',error=repr(error),pid=os.getpid(),updated_unix=time.time()))
        raise
