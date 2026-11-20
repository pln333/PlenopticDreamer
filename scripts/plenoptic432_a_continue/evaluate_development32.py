"""Fixed scene-disjoint development evaluation; detached GPU0-3 queue.

No generation interface accepts target pixels/latents. Pose evaluation uses the
same offline auxiliary real view for all arms to calibrate translation scale.
"""
import argparse
import csv
import gc
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from data import Data432, JOB, sha, write_json
from diagnose_11900 import acquire_device, tensor_sha
from evaluate_multiscene import pixel_metrics
from reconstruct_tokenizer import write_video
from evaluate_camera_pose432 import WEIGHTS, homogeneous, images, pose_scores

LAB = Path(__file__).resolve().parents[2]
OUT = JOB / 'development32_checkpoint_comparison'
SCRIPT = Path(__file__).resolve()
REFS = ['cam02', 'cam03']
TARGET = 'cam06'
I3D = Path('/root/wenfy/SVD/weights/paper_metrics/i3d_torchscript.pt')
ARM_SPECS = [('single_10000', 10000, 1), ('dual_11900', 11900, 2),
             ('single_08000', 8000, 1), ('single_09000', 9000, 1),
             ('single_05000', 5000, 1), ('dual11900_with_one_reference', 11900, 1)]


def read_json(path):
    return json.loads(Path(path).read_text())


def status(worker, stage, **kw):
    write_json(OUT / f'workers/{worker}/status.json', dict(
        stage=stage, pid=os.getpid(), gpu=worker, updated_unix=time.time(), **kw))


def protocol():
    p = read_json(OUT / 'protocol.json')
    for path, digest in p['source_sha256'].items():
        assert sha(Path(path)) == digest, f'Source changed: {path}'
    assert sha(JOB / 'dataset.json') == p['manifest_sha256']
    return p


def destination(arm, case):
    return OUT / 'arms' / arm / case['scene']


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Existing campaign is immutable'
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data432()
    ids = sorted({s.rsplit('/', 1)[1] for s in data.val}, key=lambda x: int(x[5:]))
    assert len(ids) == 32
    # The same IDs are protected across focal settings AND dataset versions.
    assert not any(s.rsplit('/', 1)[1] in ids for s in data.train)
    cases = []
    missing_refs = []
    for index, sid in enumerate(ids):
        for kind in ['multicam', 'syncam']:
            scene = f'{kind}/f24_aperture5/{sid}'
            assert scene in data.val
            poses, k = data.camera_arrays(scene)
            expected = 775.7575757575756
            assert np.allclose([k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                               [expected, expected, 384, 216])
            reference_baseline = np.linalg.norm(poses[2, :, :3, 3] - poses[1, :, :3, 3], axis=1)
            assert (reference_baseline > .05).any(), 'Offline scale baseline unavailable'
            for camera in [*REFS, TARGET]:
                path = data.source(scene, camera)
                assert path.stat().st_size == data.scenes[scene]['videos'][camera]['source_bytes']
            for camera in REFS:
                if not data.cache_path(scene, camera).exists():
                    missing_refs.append(dict(scene=scene, camera=camera))
            cases.append(dict(scene=scene, scene_id=sid, dataset=kind,
                              worker=index % 4, target=TARGET, seed=456,
                              intrinsics_pixels=k.tolist(),
                              auxiliary_reference_baseline_min_m=float(reference_baseline.min()),
                              source_sha256={c: data.scenes[scene]['videos'][c]['source_sha256']
                                             for c in [*REFS, TARGET]},
                              camera_source_sha256=data.scenes[scene]['camera_sha256']))
    overlaps = []
    history = []
    for rank in range(4):
        log = JOB / f'train/rank{rank}.jsonl'
        history.append(dict(path=str(log), sha256=sha(log)))
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row['scene'].rsplit('/', 1)[1] in ids:
                overlaps.append(dict(rank=rank, step=row['step'], scene=row['scene']))
    assert not overlaps, overlaps
    arms = []
    for name, step, count in ARM_SPECS:
        ck = JOB / f'train/step_{step:05d}/trainable_weights.safetensors'
        with safe_open(str(ck), framework='pt', device='cpu') as handle:
            metadata = handle.metadata()
        digest = sha(ck)
        assert read_json(ck.parent / 'checkpoint_manifest.json')['sha256'] == digest
        assert metadata['context_references'] == ('1' if step <= 10000 else '2')
        arms.append(dict(name=name, step=step, checkpoint=str(ck), checkpoint_sha256=digest,
                         checkpoint_metadata=metadata, references=REFS[:count]))
    gates = [JOB / 'balanced_motion_B600_900/status.json',
             *[JOB / f'balanced_motion_ablation900/{v}/status.json'
               for v in ['lower_lr', 'camera_only', 'slow_self_attn']]]
    assert all(p.exists() for p in gates)
    dependencies = [SCRIPT, Path(__file__).with_name('data.py'),
                    Path(__file__).with_name('text.py'),
                    Path(__file__).with_name('evaluate_camera_pose432.py'),
                    Path('/root/wenfy/SVD/scripts/plenoptic/multiview_inference.py'),
                    Path('/root/wenfy/SVD/scripts/plenoptic/paper_metric_math.py'),
                    LAB / 'scripts/plenoptic/camera_backbone.py',
                    LAB / 'scripts/plenoptic/context_layout.py']
    # The local camera modules are resolved by bootstrap; pin the resolved files.
    import camera_backbone, context_layout
    dependencies[-2:] = [Path(camera_backbone.__file__), Path(context_layout.__file__)]
    p = dict(version='development32_v1', scene_ids=ids, unique_scene_ids=32,
             focal_mm=24, cases=cases, cases_per_arm=64, arms=arms,
             generated_videos=64 * len(arms), manifest_sha256=data.manifest_sha,
             frames=81, resolution=[432, 768], steps=35, shift=5, guidance=0,
             target_gt_prefix=0, text='Same reference-cam02-only caption/embedding for all arms',
             pose=dict(estimator='Official VGGT-1B synchronized three-view frame triplets',
                       weights=str(WEIGHTS), weights_sha256=sha(WEIGHTS), frames=list(range(81)),
                       independent_batch_size=2, offline_calibration_views=REFS,
                       scale='One positive video-wide scale from real cam02/cam03 baseline ONLY; never target GT alignment',
                       auxiliary_view='cam03 is offline evaluator calibration for ALL arms; generator sees it ONLY in dual_11900',
                       warning='VGGT frame triplets are not the paper dynamic ViPE protocol. Report GT calibration and invalid scales.'),
             fvd=dict(weights=str(I3D), weights_sha256=sha(I3D), window_frames=49,
                      starts=[0, 32], statistical_unit='Overlapping windows; 32 scene-ID clusters, 64 dataset cases, not 128 independent scenes',
                      real_population='Paired real targets; not paper Basic in-the-wild proxy'),
             source_sha256={str(path): sha(path) for path in dependencies},
             training_history=history, training_scene_id_overlap=overlaps,
             gate_paths=[str(g) for g in gates], allowed_gpus=[0, 1, 2, 3],
             caveats=['Development split, repeatedly inspected; not blind official benchmark.',
                      'All focal variants and both dataset versions protected by scene ID.',
                      '11900 differs in curriculum/reference count, so is not single-reference training-duration continuation.',
                      '11900 with one reference is a post-dual-curriculum diagnostic, not a single-reference-trained checkpoint.',
                      'One generation seed; results are paired but do not estimate sampling variance.',
                      'Fixed cam02 to cam06 pair; does not cover every available target trajectory.',
                      'Pixel, pose and FVD metrics use saved MP4 pixels consistently.'])
    write_json(OUT / 'protocol.json', p)
    write_json(OUT / 'preflight.json', dict(status='passed', training_scene_id_overlap=overlaps,
               cases=len(cases), unique_scene_ids=len(ids), arms=len(arms),
               missing_reference_caches=missing_refs,
               cache_policy='Encode missing reference videos with pinned official VAE after GPU gate, never substitute data'))
    snap = OUT / 'sources'; snap.mkdir(exist_ok=True)
    for index, path in enumerate(dependencies):
        shutil.copyfile(path, snap / f'{index}_{path.name}')
    (OUT / 'README.md').write_text('''# 32 个保留场景的固定开发评测

32 个 scene ID，各取 MultiCam/SynCam 的 24mm 版本，共 64 个配对目标。每组 seed456、81 帧、432×768、35 步、shift5、guidance0。单参考 cam02，双参考 cam02/cam03，目标 cam06，文字仅由 cam02 生成。所有训练日志及同 ID 的不同焦距均已检查隔离。

比较单参考训练 5K/8K/9K/10K 与双参考续训至 11900；另给 11900 输入一个参考，以区分权重变化与增加参考。后者不是单参考训练到 11900 的权重。

GPU0–3 各负责 8 个 scene ID 的两个版本，等待本卡原实验完成训练和最终评测后自动运行。本任务不占 GPU4–7、不修改旧结果，不复用 guidance7 生成。

生成不读取目标 RGB/latent。保存新 latent、生成视频、参考/GT/生成拼接视频、PSNR/SSIM。位姿指标采用全81帧 VGGT，每帧独立输入真实 cam02、真实 cam03、候选目标；仅用两个真实辅助视角间的已知基线标定正平移尺度，目标 GT 不参与拟合。对单参考组，cam03 仅用于离线评估，不输入生成器。GT 与复制参考同样评测。该协议不同于论文动态 ViPE。

FVD 使用已有 StyleGAN-V I3D，81 帧取 [0:49]、[32:81] 两窗口。按数据集分开汇总，小样本与相关窗口的限制保留。所有指标从保存的 MP4 计算。报告 scene ID 配对变化，32 个 ID 是统计分组单位，不能把两个数据集版本或窗口当成独立场景。

实时状态见 workers/*/status.json，阶段性结果见 SUMMARY.json、REPORT.md；最终展示见 index.html。本轮是开发验证，不是作者官方盲测。
''')
    report()
    print(json.dumps(dict(preflight='passed', cases=len(cases), arms=len(arms),
                          videos=p['generated_videos'], missing_reference_caches=len(missing_refs))), flush=True)


def conditions(data, selected):
    missing = [(case['scene'], c) for case in selected for c in REFS
               if not data.cache_path(case['scene'], c).exists()]
    if missing:
        from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        vae = Wan2pt1VAEInterface(vae_pth=data.manifest['vae_weight'], load_mean_std=False,
                                  chunk_duration=81, temporal_window=16)
        for scene, camera in missing:
            rgb = data.read_rgb(scene, camera)
            x = torch.from_numpy(rgb.copy()).permute(3, 0, 1, 2)[None].cuda().to(torch.bfloat16) / 127.5 - 1
            z = vae.encode(x)[0].to(torch.bfloat16).cpu().contiguous()
            assert list(z.shape) == [16, 21, 54, 96] and torch.isfinite(z).all()
            path = data.cache_path(scene, camera); path.parent.mkdir(parents=True, exist_ok=True)
            save_file({'latent': z}, str(path.with_suffix('.tmp')), metadata=data.metadata(scene, camera))
            path.with_suffix('.tmp').replace(path)
            vae.clear_cache(); del rgb, x, z
        del vae; gc.collect(); torch.cuda.empty_cache()
    encoder = None
    from text import ReferenceTextEncoder
    for case in selected:
        path = OUT / 'conditions' / case['scene'] / 'cam02.safetensors'
        if path.exists():
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                assert handle.metadata()['source_sha256'] == case['source_sha256']['cam02']
            continue
        if encoder is None:
            encoder = ReferenceTextEncoder()
        caption = encoder.caption(data, case['scene'], 'cam02')
        text = encoder.embedding(caption).cpu().contiguous()
        assert text.shape == (1, 512, 100352) and torch.isfinite(text).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({'text': text}, str(path.with_suffix('.tmp')), metadata=dict(
            source_sha256=case['source_sha256']['cam02'], caption=caption,
            preprocess_id=data.manifest['preprocess_id']))
        path.with_suffix('.tmp').replace(path)
        print('Condition prepared', case['scene'], flush=True)
        del text
    del encoder; gc.collect(); torch.cuda.empty_cache()


def video(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    value = np.asarray(frames)
    assert value.shape == (81, 432, 768, 3), (str(path), value.shape)
    return value


def generate(data, p, selected, arm, worker):
    from multiview_inference import CameraVideoInference, sample_target
    from context_layout import ContextLayout
    refs = arm['references']
    assert sha(Path(arm['checkpoint'])) == arm['checkpoint_sha256']
    todo = [c for c in selected if not (destination(arm['name'], c) / 'sampling.json').exists()]
    if not todo: return
    pipeline = CameraVideoInference(arm['checkpoint'])
    for case in todo:
        scene = case['scene']; dest = destination(arm['name'], case); dest.mkdir(parents=True, exist_ok=True)
        reference = torch.stack([data.latent(scene, c) for c in refs], dim=1)
        camera = data.camera(scene, refs, TARGET)
        text_path = OUT / 'conditions' / scene / 'cam02.safetensors'
        text = load_file(str(text_path))['text'].cuda()
        layout = ContextLayout(len(refs), 21)
        observed = {}
        def predict(**kw):
            if not observed:
                mask = kw['condition_video_input_mask_B_C_T_H_W']
                assert torch.count_nonzero(mask[:, :, layout.target_slice]) == 0
                assert torch.equal(kw['camera'], camera)
                observed.update(target_gt_supplied=False, target_clean_prefix=0,
                                camera_sha256=tensor_sha(camera), reference_latent_sha256=tensor_sha(reference))
            return pipeline.predict_velocity(**kw)
        def progress(row):
            status(worker, 'generating', arm=arm['name'], scene=scene, solver_step=row['step'])
        z, sampling = sample_target(predict, reference, camera, text,
                                    steps=p['steps'], shift=p['shift'], seed=case['seed'], progress=progress)
        assert z.shape == (1, 16, 21, 54, 96) and torch.isfinite(z).all()
        save_file({'latent': z.cpu().contiguous()}, str(dest / 'generated_latent.tmp'), metadata=dict(
            source_kind='generated_not_gt', checkpoint_sha256=arm['checkpoint_sha256']))
        (dest / 'generated_latent.tmp').replace(dest / 'generated_latent.safetensors')
        write_json(dest / 'sampling.json', dict(sampling, arm=arm, case=case, observed=observed,
                    text_sha256=sha(text_path), protocol_sha256=sha(OUT / 'protocol.json')))
        print('Generated', arm['name'], scene, flush=True)
        del reference, camera, text, z
        gc.collect(); torch.cuda.empty_cache()
    del pipeline; gc.collect(); torch.cuda.empty_cache()


def decode(data, p, selected, arm, worker):
    todo = [c for c in selected if not (destination(arm['name'], c) / 'metrics.json').exists()]
    if not todo: return
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    vae = Wan2pt1VAEInterface(vae_pth=data.manifest['vae_weight'], load_mean_std=False,
                              chunk_duration=81, temporal_window=16)
    for case in todo:
        dest = destination(arm['name'], case)
        status(worker, 'decoding_scoring', arm=arm['name'], scene=case['scene'])
        z = load_file(str(dest / 'generated_latent.safetensors'))['latent'].cuda().to(torch.bfloat16)
        rgb = vae.decode(z)
        assert rgb.shape == (1, 3, 81, 432, 768) and torch.isfinite(rgb).all()
        pixels = np.rint(((rgb[0].float().cpu().permute(1, 2, 3, 0).numpy() + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
        write_video(dest / 'generated.mp4', pixels, 15)
        generated = video(dest / 'generated.mp4')
        # Target is first loaded after generation is finished, exclusively for scoring.
        gt = data.read_rgb(case['scene'], TARGET)
        reference = data.read_rgb(case['scene'], 'cam02')
        scores = pixel_metrics(generated.astype(np.float32) / 255, gt.astype(np.float32) / 255)
        copy_scores = pixel_metrics(reference.astype(np.float32) / 255, gt.astype(np.float32) / 255)
        panel = np.concatenate([reference, gt, generated], axis=2)
        write_video(dest / 'reference_target_generated.mp4', panel, 15)
        sheet = np.concatenate([panel[t] for t in [0, 40, 80]], axis=0)
        cv2.imwrite(str(dest / 'comparison.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        write_json(dest / 'metrics.json', dict(status='completed', arm=arm['name'], case=case,
            pixel_metrics=scores, reference_copy=copy_scores, generated_video_sha256=sha(dest / 'generated.mp4'),
            checkpoint_sha256=arm['checkpoint_sha256'], target_pixels_used_for_sampling=False,
            pixel_population='saved MP4, full 81 frames'))
        print('Scored', arm['name'], case['scene'], scores, flush=True)
        del z, rgb, pixels, generated, gt, reference, panel, sheet
        vae.clear_cache(); gc.collect(); torch.cuda.empty_cache()
    del vae; gc.collect(); torch.cuda.empty_cache()


def pose(data, p, selected, worker):
    sys.path.insert(0, '/root/wenfy/SVD/vendor/vggt_metrics')
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    assert sha(WEIGHTS) == p['pose']['weights_sha256']
    # Load every released tensor strictly; only call aggregator/camera_head below.
    # Disabling heads in the constructor would reject the full released state.
    model = VGGT()
    model.load_state_dict(load_file(str(WEIGHTS)), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    for case in selected:
        scene = case['scene']; poses, _ = data.camera_arrays(scene)
        refs = [data.read_rgb(scene, c) for c in REFS]
        target_gt = np.linalg.inv(poses[5]) @ poses[1]
        ref_gt = np.linalg.inv(poses[2]) @ poses[1]
        candidates = [('real_target', None), ('reference_copy', None)] + [(a['name'], a) for a in p['arms']]
        for name, arm in candidates:
            dest = OUT / 'pose' / name / scene; dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'metrics.json').exists(): continue
            status(worker, 'pose', candidate=name, scene=scene)
            if name == 'real_target': candidate = data.read_rgb(scene, TARGET)
            elif name == 'reference_copy': candidate = refs[0]
            else:
                source = destination(name, case) / 'generated.mp4'
                assert sha(source) == read_json(source.parent / 'metrics.json')['generated_video_sha256']
                candidate = video(source)
            source_digest = case['source_sha256'][TARGET] if name == 'real_target' else (
                case['source_sha256']['cam02'] if name == 'reference_copy' else sha(source))
            record_path = dest / 'poses.json'
            existing = read_json(record_path) if record_path.exists() else dict(source_sha256=source_digest, frames=[])
            assert existing['source_sha256'] == source_digest
            rows = existing['frames']; done = {r['frame'] for r in rows}
            remaining = [f for f in range(81) if f not in done]
            for begin in range(0, len(remaining), 2):
                chosen = remaining[begin:begin + 2]
                batch = torch.stack([images(refs[0][f], refs[1][f], candidate[f]) for f in chosen]).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16): tokens, _ = model.aggregator(batch)
                with torch.autocast('cuda', enabled=False):
                    enc = model.camera_head([tokens[-1].float()])[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(enc, batch.shape[-2:])
                estimates = homogeneous(extrinsic.float().cpu().numpy())
                for i, frame in enumerate(chosen):
                    relative = estimates[i] @ np.linalg.inv(estimates[i, 0])
                    rows.append(dict(frame=frame, predicted_target=relative[2].tolist(),
                        requested_target=target_gt[frame].tolist(), predicted_input_reference=relative[1].tolist(),
                        requested_input_reference=ref_gt[frame].tolist()))
                write_json(record_path, dict(source_sha256=source_digest, frames=rows))
                status(worker, 'pose', candidate=name, scene=scene, frames_done=len(rows))
                del batch, tokens, enc, extrinsic, intrinsic
            rows.sort(key=lambda r: r['frame']); assert [r['frame'] for r in rows] == list(range(81))
            scores = pose_scores(*[[r[key] for r in rows] for key in [
                'predicted_target', 'requested_target', 'predicted_input_reference', 'requested_input_reference']])
            write_json(dest / 'metrics.json', dict(status='completed', case=case, candidate=name, scores=scores,
                       source_sha256=source_digest, protocol_sha256=sha(OUT / 'protocol.json')))
            print('Pose', name, scene, scores['position_rmse_m'], scores['rot_err_mean_deg'], flush=True)
            del candidate; gc.collect(); torch.cuda.empty_cache()
        del refs
    del model; gc.collect(); torch.cuda.empty_cache()


def fvd_features(data, p, selected, worker):
    assert sha(I3D) == p['fvd']['weights_sha256']
    detector = torch.jit.load(str(I3D), map_location='cuda').eval()
    for case in selected:
        for name in ['real_target', 'reference_copy', *[a['name'] for a in p['arms']]]:
            path = OUT / 'fvd' / name / case['scene'] / 'features.json'
            if path.exists(): continue
            status(worker, 'fvd_features', candidate=name, scene=case['scene'])
            if name == 'real_target': pixels = data.read_rgb(case['scene'], TARGET)
            elif name == 'reference_copy': pixels = data.read_rgb(case['scene'], 'cam02')
            else: pixels = video(destination(name, case) / 'generated.mp4')
            values = []
            for start in p['fvd']['starts']:
                x = torch.from_numpy(pixels[start:start + 49].copy()).permute(3, 0, 1, 2)[None].cuda().float()
                value = detector(x, rescale=True, resize=True, return_features=True)
                assert value.shape == (1, 400) and torch.isfinite(value).all()
                values.append(value[0].cpu().tolist()); del x, value
            write_json(path, dict(case=case, candidate=name, features=values))
            del pixels
    del detector; gc.collect(); torch.cuda.empty_cache()


def run_worker(worker):
    import fcntl
    folder = OUT / f'workers/{worker}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(worker)
    p = protocol(); selected = [c for c in p['cases'] if c['worker'] == worker]
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.set_grad_enabled(False)
    status(worker, 'acquiring_gpu'); acquire_device(50)
    data = Data432(); status(worker, 'preparing_conditions')
    conditions(data, selected)
    for arm in p['arms']:
        generate(data, p, selected, arm, worker)
        decode(data, p, selected, arm, worker)
        report(worker)
    pose(data, p, selected, worker)
    fvd_features(data, p, selected, worker)
    status(worker, 'completed', cases=len(selected), arms=len(p['arms']))
    report(worker)


def mean_valid(values):
    x = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def report(worker=None):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'report.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    p = read_json(OUT / 'protocol.json'); records = []
    for arm in p['arms']:
        for case in p['cases']:
            path = destination(arm['name'], case) / 'metrics.json'
            if not path.exists(): continue
            row = read_json(path)
            pose_path = OUT / 'pose' / arm['name'] / case['scene'] / 'metrics.json'
            row['pose'] = read_json(pose_path)['scores'] if pose_path.exists() else None
            records.append(row)
    groups = {}
    from paper_metric_math import frechet_features
    for dataset in ['all', 'multicam', 'syncam']:
        expected = [c for c in p['cases'] if dataset == 'all' or c['dataset'] == dataset]
        groups[dataset] = {}
        for name in [*[a['name'] for a in p['arms']], 'reference_copy', 'real_target']:
            chosen = [r for r in records if r['arm'] == name and (dataset == 'all' or r['case']['dataset'] == dataset)]
            pos = [read_json(OUT / 'pose' / name / c['scene'] / 'metrics.json')['scores']
                   for c in expected if (OUT / 'pose' / name / c['scene'] / 'metrics.json').exists()]
            pixel = [r['pixel_metrics'] for r in chosen]
            if name == 'reference_copy':
                pixel = [r['reference_copy'] for r in records if r['arm'] == 'single_10000'
                         and (dataset == 'all' or r['case']['dataset'] == dataset)]
            summary = dict(pixel_cases=len(pixel), expected_cases=len(expected),
                psnr_db=mean_valid([x['psnr_mean_db'] for x in pixel]),
                ssim=mean_valid([x['ssim_mean'] for x in pixel]), pose_cases=len(pos),
                pose_valid_translation_cases=sum(x['position_rmse_m'] is not None for x in pos),
                position_rmse_m_mean=mean_valid([x['position_rmse_m'] for x in pos]),
                trans_err_mean_m2=mean_valid([x['trans_err_mean_m2'] for x in pos]),
                rot_err_mean_deg=mean_valid([x['rot_err_mean_deg'] for x in pos]))
            real, generated = [], []
            for c in expected:
                r = OUT / 'fvd/real_target' / c['scene'] / 'features.json'
                g = OUT / 'fvd' / name / c['scene'] / 'features.json'
                if r.exists() and g.exists():
                    real.extend(read_json(r)['features']); generated.extend(read_json(g)['features'])
            summary['fvd_cases'] = len(real) // 2
            # Do not publish an FVD value computed on a changing subset.
            summary['fvd'] = frechet_features(np.asarray(real), np.asarray(generated)) if len(real) == 2 * len(expected) else None
            groups[dataset][name] = summary
    comparisons = [('single_08000', 'single_09000'), ('single_09000', 'single_10000'),
                   ('single_05000', 'single_10000'), ('single_10000', 'dual_11900'),
                   ('single_10000', 'dual11900_with_one_reference'),
                   ('dual11900_with_one_reference', 'dual_11900')]
    paired = []
    lookup = {(r['arm'], r['case']['scene']): r for r in records}
    for before, after in comparisons:
        per_id = {}
        for case in p['cases']:
            a = lookup.get((before, case['scene'])); b = lookup.get((after, case['scene']))
            if not a or not b: continue
            per_id.setdefault(case['scene_id'], []).append(
                b['pixel_metrics']['psnr_mean_db'] - a['pixel_metrics']['psnr_mean_db'])
        clusters = np.array([np.mean(x) for x in per_id.values()])
        ci = None
        if len(clusters) == 32 and all(len(x) == 2 for x in per_id.values()):
            rng = np.random.default_rng(20261119)
            ci = np.quantile(clusters[rng.integers(0, 32, (2000, 32))].mean(1), [.025, .975]).tolist()
        paired.append(dict(before=before, after=after, scene_id_clusters=len(clusters),
            psnr_delta_db=mean_valid(clusters.tolist()), psnr_delta_95pct_cluster_bootstrap=ci,
            note='Two dataset versions averaged within scene ID; seed variance not estimated'))
    states = {str(i): read_json(OUT / f'workers/{i}/status.json')
              for i in range(4) if (OUT / f'workers/{i}/status.json').exists()}
    complete = len(states) == 4 and all(s['stage'] == 'completed' for s in states.values())
    write_json(OUT / 'SUMMARY.json', dict(status='completed' if complete else 'partial',
        updated_unix=time.time(), groups=groups, paired_comparisons=paired, workers=states,
        cases=records, protocol_sha256=sha(OUT / 'protocol.json')))
    lines = ['# 固定32场景开发评测', '',
        f"状态：{'完成' if complete else '等待/进行中'}。32 个 scene ID × 两个数据集版本，共64案例/组；不是64个独立场景。", '',
        '单参考5K/8K/9K/10K可比较训练时长；11900经过双参考课程，不能当作单参考多训练1900步。', '',
        '所有组guidance0、35步、seed456；旧guidance7结果不混用。位姿为VGGT离线诊断，cam03对单参考仅用于评测尺度标定，不参与生成；与论文动态ViPE不同。', '']
    for dataset, values in groups.items():
        lines += [f'## {dataset}', '', '| 权重/输入 | 视频 | PSNR dB↑ | SSIM↑ | 位姿数/有效平移数 | 位置RMSE均值 m↓ | TransErr m²↓ | RotErr °↓ | FVD↓ |',
                  '|---|---|---|---|---|---|---|---|---|']
        def fmt(x): return '—' if x is None else f'{x:.4f}'
        for name, s in values.items():
            lines.append(f"| {name} | {s['pixel_cases']}/{s['expected_cases']} | {fmt(s['psnr_db'])} | {fmt(s['ssim'])} | {s['pose_cases']}/{s['pose_valid_translation_cases']} | {fmt(s['position_rmse_m_mean'])} | {fmt(s['trans_err_mean_m2'])} | {fmt(s['rot_err_mean_deg'])} | {fmt(s['fvd'])} |")
        lines.append('')
    lines += ['部分结果的样本可能尚不一致，正式比较应等待同一批案例完整。FVD仅在组内案例全部完成后发布。GT校准误差必须一并解释。', '',
              '协议、训练隔离审计、逐帧位姿、配对差值及scene-ID聚类区间见protocol.json、preflight.json、SUMMARY.json与pose/。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    cards = []
    for case in p['cases']:
        sections = []
        for arm in p['arms']:
            dest = destination(arm['name'], case)
            if not (dest / 'metrics.json').exists(): continue
            score = read_json(dest / 'metrics.json')['pixel_metrics']
            link = (dest / 'reference_target_generated.mp4').relative_to(OUT)
            sections.append(f'<h3>{html.escape(arm["name"])}: {score["psnr_mean_db"]:.2f} dB / SSIM {score["ssim_mean"]:.3f}</h3><video controls preload="none" width="1100" src="{link}"></video>')
        if sections: cards.append(f'<details><summary>{html.escape(case["scene"])}</summary>'+''.join(sections)+'</details>')
    (OUT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Development32 comparison</title><h1>32 scene IDs: checkpoint comparison</h1><p>Panels: reference / GT / generated. Same seed, target, sampler; development validation.</p>'+''.join(cards))
    fields = ['arm', 'scene', 'dataset', 'psnr_db', 'ssim', 'position_rmse_m', 'trans_err_mean_m2', 'rot_err_mean_deg']
    with (OUT / 'per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            pose_row = r['pose'] or {}
            writer.writerow(dict(arm=r['arm'], scene=r['case']['scene'], dataset=r['case']['dataset'],
                psnr_db=r['pixel_metrics']['psnr_mean_db'], ssim=r['pixel_metrics']['ssim_mean'],
                **{key: pose_row.get(key) for key in fields[5:]}))


def queue_worker(worker):
    import fcntl
    p = protocol(); folder = OUT / f'workers/{worker}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = Path(p['gate_paths'][worker])
    while True:
        state = read_json(gate)
        if state.get('stage') == 'completed': break
        if state.get('stage') == 'failed': raise RuntimeError(f'Existing GPU{worker} experiment failed: {gate}')
        status(worker, 'queued', wait_for=str(gate), existing_stage=state.get('stage'))
        time.sleep(10)
    command = [str(LAB / 'run.sh'), str(SCRIPT.with_name('bootstrap.py')), str(SCRIPT), '--worker', str(worker)]
    with (folder / 'worker.log').open('a') as log:
        child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker)),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / 'launch.json', dict(queue_pid=os.getpid(), worker_pid=child.pid,
                    gpu=worker, command=command, started_unix=time.time()))
        code = child.wait()
    if code: raise RuntimeError(f'Worker {worker} failed, exit={code}; see {folder / "worker.log"}')
    report(worker)


def launch():
    protocol()
    assert not (OUT / 'launch.json').exists(), 'Queue already launched'
    children = []
    for worker in range(4):
        folder = OUT / f'workers/{worker}'; folder.mkdir(parents=True, exist_ok=True)
        command = [str(LAB / 'run.sh'), str(SCRIPT.with_name('bootstrap.py')),
                   str(SCRIPT), '--queue', str(worker)]
        with (folder / 'queue.log').open('x') as log:
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker),
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(dict(gpu=worker, queue_pid=child.pid, command=command,
                             wait_for=read_json(OUT / 'protocol.json')['gate_paths'][worker]))
        write_json(OUT / 'launch.json', dict(started_unix=time.time(), queues=children))
    print(json.dumps(dict(stage='queued', queues=children), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--queue', type=int, choices=range(4))
    parser.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    worker_id = args.worker if args.worker is not None else args.queue
    try:
        if args.prepare: prepare()
        elif args.launch: launch()
        elif args.queue is not None: queue_worker(args.queue)
        elif args.worker is not None: run_worker(args.worker)
        elif args.report: report()
        else: parser.error('Choose action')
    except BaseException as error:
        if worker_id is not None:
            status(worker_id, 'failed', error=repr(error), traceback=traceback.format_exc())
        raise
