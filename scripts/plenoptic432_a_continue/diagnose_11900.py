"""Controlled two-reference trajectory interventions; never train or supply target GT to sampling."""
import argparse
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from data import Data432, JOB, sha, write_json
from context_layout import ContextLayout
from audit_dynamic_camera import motion

OUT = JOB / 'diagnosis_11900' / 'trajectory_controls'
CHECKPOINT = JOB / 'train/step_11900/trainable_weights.safetensors'
MODES = ['true', 'frozen', 'reversed']


def acquire_device(required_gib):
    """Submit a small real workload so the opportunistic vLLM service yields."""
    torch.cuda.set_device(0)
    probe = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
    start = time.monotonic(); last_notice = -10
    while torch.cuda.mem_get_info()[0] < required_gib*1024**3:
        result = probe @ probe
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        if elapsed-last_notice >= 10:
            print(json.dumps(dict(stage='waiting_for_gpu_yield', free_gib=torch.cuda.mem_get_info()[0]/2**30, seconds=elapsed)), flush=True)
            last_notice = elapsed
        if elapsed > 180: raise RuntimeError('Opportunistic service has not yielded enough GPU memory')
    del probe
    if 'result' in locals(): del result
    torch.cuda.empty_cache()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cases():
    return [dict(name='heldout', scene='multicam/f24_aperture5/scene1130', references=['cam02', 'cam03'], target='cam06', split='val'),
            dict(name='trained_pair', scene='multicam/f24_aperture5/scene1340', references=['cam06', 'cam07'], target='cam02', split='train')]


def tasks():
    return [dict(worker=i, **case, seed=seed) for i, (case, seed) in
            enumerate((case, seed) for case in cases() for seed in [456, 789])]


def background(video):
    """Apparent motion in a fixed top/right region, not a recovered 3D pose."""
    sift = cv2.SIFT_create(nfeatures=7000)
    mask = np.full(video.shape[1:3], 255, dtype=np.uint8)
    mask[60:, :560] = 0
    first_key, first_desc = sift.detectAndCompute(cv2.cvtColor(video[0], cv2.COLOR_RGB2GRAY), mask)
    rows = []
    for frame in [20, 40, 60, 80]:
        key, desc = sift.detectAndCompute(cv2.cvtColor(video[frame], cv2.COLOR_RGB2GRAY), mask)
        row = dict(frame=frame)
        if first_desc is None or desc is None:
            rows.append(row | dict(status='no_features')); continue
        matches = [a for pair in cv2.BFMatcher().knnMatch(first_desc, desc, k=2)
                   if len(pair) == 2 for a, b in [pair] if a.distance < .7*b.distance]
        if len(matches) < 12:
            rows.append(row | dict(status='insufficient_matches', matches=len(matches))); continue
        p = np.float32([first_key[m.queryIdx].pt for m in matches])
        q = np.float32([key[m.trainIdx].pt for m in matches])
        _, valid = cv2.findHomography(p, q, cv2.RANSAC, 2.)
        if valid is None or int(valid.sum()) < 12:
            rows.append(row | dict(status='insufficient_inliers', matches=len(matches), inliers=int(valid.sum()) if valid is not None else 0)); continue
        difference = q[valid[:, 0].astype(bool)] - p[valid[:, 0].astype(bool)]
        rows.append(row | dict(status='ok', matches=len(matches), inliers=len(difference),
                    median_displacement_px=float(np.median(np.linalg.norm(difference, axis=1))),
                    median_dx_px=float(np.median(difference[:, 0])), median_dy_px=float(np.median(difference[:, 1]))))
    return rows


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data432()
    protocol = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
                    manifest_sha256=data.manifest_sha, tasks=tasks(), modes=MODES,
                    steps=35, shift=5, guidance=0, target_gt_prefix=0,
                    frames=81, resolution=[432, 768], scope='four controlled development cases, not an official benchmark',
                    source_sha256=sha(Path(__file__)))
    with safe_open(str(CHECKPOINT), framework='pt', device='cpu') as handle:
        assert handle.metadata()['context_references'] == '2'
    training = []
    for rank in range(4):
        training.extend(json.loads(line) for line in (JOB/'train'/f'rank{rank}.jsonl').read_text().splitlines())
    encoder = None
    from text import ReferenceTextEncoder
    for case in cases():
        assert data.scenes[case['scene']]['split'] == case['split']
        poses, intrinsics = data.camera_arrays(case['scene'])
        target_motion = motion(poses[int(case['target'][3:])-1])
        assert target_motion['max_distance_m'] > .01 or target_motion['max_rotation_deg'] > 1
        seen = [dict(step=r['step'], rank=r['rank'], k=r['k']) for r in training
                if r['scene'] == case['scene'] and r['references'] == case['references'] and r['target'] == case['target']]
        assert bool(seen) == (case['split'] == 'train')
        dest = OUT/'conditions'/case['name']; dest.mkdir(parents=True, exist_ok=True)
        caption_path = JOB/'captions'/case['scene']/f"{case['references'][0]}.json"
        caption = json.loads(caption_path.read_text())
        assert caption['source_sha256'] == data.scenes[case['scene']]['videos'][case['references'][0]]['source_sha256']
        text_path = dest/'text.safetensors'
        if not text_path.exists():
            if encoder is None:
                acquire_device(20)
                encoder = ReferenceTextEncoder()
            embedding = encoder.embedding(caption['caption']).cpu().contiguous()
            assert embedding.shape == (1, 512, 100352) and torch.isfinite(embedding).all()
            save_file({'text': embedding}, str(text_path), metadata=dict(caption_sha256=sha(caption_path)))
            del embedding
        write_json(dest/'inputs.json', dict(**case, exact_training_occurrences=seen,
                    intrinsics_pixels=intrinsics.tolist(), target_motion=target_motion,
                    caption=caption['caption'], caption_source=str(caption_path), text_sha256=sha(text_path)))
    if (OUT/'protocol.json').exists(): assert json.loads((OUT/'protocol.json').read_text()) == protocol
    write_json(OUT/'protocol.json', protocol)
    print(json.dumps(protocol, indent=2), flush=True)


def worker(index):
    from multiview_inference import CameraVideoInference, sample_target
    from reconstruct_tokenizer import write_video
    from evaluate_multiscene import pixel_metrics
    torch.set_num_threads(4); cv2.setNumThreads(2); torch.cuda.set_device(0); torch.set_grad_enabled(False)
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert protocol['source_sha256'] == sha(Path(__file__)) and protocol['checkpoint_sha256'] == sha(CHECKPOINT)
    task = protocol['tasks'][index]
    dest = OUT/f"worker{index}_{task['name']}_seed{task['seed']}"; dest.mkdir(exist_ok=True)
    def status(stage, **kwargs):
        write_json(dest/'status.json', dict(stage=stage, pid=os.getpid(), worker=index, updated_unix=time.time(), **kwargs))
    status('acquiring_gpu'); acquire_device(50); status('loading')
    data = Data432(); layout = ContextLayout(2, 21)
    references = torch.stack([data.latent(task['scene'], c) for c in task['references']], dim=1)
    rays = data.camera(task['scene'], task['references'], task['target'])
    text = load_file(str(OUT/'conditions'/task['name']/'text.safetensors'))['text'].cuda()
    pipeline = CameraVideoInference(CHECKPOINT)
    report = dict(task=task, checkpoint_sha256=protocol['checkpoint_sha256'], guidance=0,
                  target_gt_supplied_to_sampler=False, reference_sha256=tensor_sha(references),
                  text_sha256=tensor_sha(text), input_audits={})
    noise_hashes = []
    for mode in MODES:
        case_dir = dest/mode; case_dir.mkdir(exist_ok=True)
        camera = rays.clone()
        if mode == 'frozen': camera[:, layout.target_slice] = rays[:, layout.target_slice.start:layout.target_slice.start+1]
        if mode == 'reversed': camera[:, layout.target_slice] = rays[:, layout.target_slice].flip(1)
        assert torch.equal(camera[:, :layout.target_slice.start], rays[:, :layout.target_slice.start])
        observed = {}
        def model_hook(module, args, kwargs):
            if observed.get('model'): return
            mask = kwargs['condition_video_input_mask_B_C_T_H_W']
            assert torch.count_nonzero(mask[:, :, layout.target_slice]) == 0
            assert torch.all(mask[:, :, :layout.target_slice.start] == 1)
            actual_x = kwargs['x_B_C_T_H_W']
            expected_refs = torch.cat(list(references.unbind(1)), dim=2).to(actual_x.dtype)
            assert torch.equal(actual_x[:, :, :42], expected_refs)
            observed['model'] = dict(packed_shape=list(actual_x.shape), mask_shape=list(mask.shape),
                   target_clean_elements=0, target_timesteps=kwargs['timesteps_B_T'][:, layout.target_slice].cpu().tolist(),
                   reference_timesteps=kwargs['timesteps_B_T'][:, :42].cpu().tolist(), clean_reference_input_matches=True)
        def camera_hook(module, args):
            if observed.get('camera'): return
            actual = args[0]; assert torch.equal(actual, camera)
            target = actual[:, layout.target_slice]
            unique = len({tensor_sha(frame) for frame in target[0]})
            assert unique == (1 if mode == 'frozen' else 21)
            observed['camera'] = dict(shape=list(actual.shape), sha256=tensor_sha(actual), unique_target_frames=unique,
                target_rms_change_from_first=(target-target[:, :1]).square().mean(dim=(0, 2, 3, 4)).sqrt().cpu().tolist())
        h1 = pipeline.model.register_forward_pre_hook(model_hook, with_kwargs=True)
        h2 = pipeline.model.blocks[0].cam_encoder.register_forward_pre_hook(camera_hook)
        def progress(row):
            status('sampling', mode=mode, solver_step=row['step'], solver_steps=35)
            print(json.dumps(dict(worker=index, mode=mode, **row)), flush=True)
        try:
            generated, sampling = sample_target(pipeline.predict_velocity, references, camera, text,
                              steps=35, shift=5, seed=task['seed'], progress=progress)
        finally:
            h1.remove(); h2.remove()
        assert observed.get('model') and observed.get('camera')
        noise_hashes.append(sampling['initial_noise_sha256'])
        report['input_audits'][mode] = observed
        sampling['guidance'] = 0
        save_file({'latent': generated.cpu().contiguous()}, str(case_dir/'generated_latent.safetensors'),
                  metadata=dict(checkpoint_sha256=protocol['checkpoint_sha256'], condition=mode, source_kind='free_generation_no_target_gt'))
        write_json(case_dir/'sampling.json', sampling)
        del generated, camera; gc.collect(); torch.cuda.empty_cache()
    assert len(set(noise_hashes)) == 1
    report['same_initial_noise_all_modes'] = True
    write_json(dest/'input_audit.json', report)
    del pipeline, references, rays, text; gc.collect(); torch.cuda.empty_cache()
    status('decoding')
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    vae = Wan2pt1VAEInterface(vae_pth=data.manifest['vae_weight'], load_mean_std=False, chunk_duration=81, temporal_window=16)
    # Target pixels are first read here, after every generated latent is saved.
    gt = data.read_rgb(task['scene'], task['target'])
    videos = {'gt': gt}
    for mode in MODES:
        decoded = vae.decode(load_file(str(dest/mode/'generated_latent.safetensors'))['latent'].cuda().to(torch.bfloat16))
        pixels = ((decoded[0].float().cpu().permute(1, 2, 3, 0).numpy()+1)/2).clip(0, 1)
        videos[mode] = np.rint(pixels*255).astype(np.uint8)
        write_video(dest/mode/'generated.mp4', videos[mode], 15)
        if mode == 'true': report['true_trajectory_pixel_metrics'] = pixel_metrics(pixels, gt.astype(np.float32)/255)
        del decoded, pixels; vae.clear_cache(); gc.collect(); torch.cuda.empty_cache()
    report['background_motion_scope'] = 'fixed top/right SIFT+homography mask, apparent motion only, not TransErr/RotErr; invalid matches remain missing'
    report['background_motion'] = {name: background(video) for name, video in videos.items()}
    true_z = load_file(str(dest/'true/generated_latent.safetensors'))['latent'].float()
    report['intervention_response'] = {}
    for mode in ['frozen', 'reversed']:
        other = load_file(str(dest/mode/'generated_latent.safetensors'))['latent'].float()
        report['intervention_response'][mode] = dict(latent_rms_difference=float((other-true_z).square().mean().sqrt()),
            rgb_mean_abs_difference_0_1=float(np.abs(videos[mode].astype(np.float32)-videos['true'].astype(np.float32)).mean()/255))
    panel = np.concatenate([videos[name].copy() for name in ['gt', *MODES]], axis=2)
    for frame in panel:
        for col, label in enumerate(['GT (true trajectory)', 'Generated: true', 'Generated: frozen', 'Generated: reversed']):
            cv2.rectangle(frame, (col*768, 0), ((col+1)*768, 28), (0, 0, 0), -1)
            cv2.putText(frame, label, (col*768+10, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    write_video(dest/'gt_true_frozen_reversed.mp4', panel, 15)
    cv2.imwrite(str(dest/'comparison.jpg'), cv2.cvtColor(np.concatenate([panel[t] for t in [0, 40, 80]], axis=0), cv2.COLOR_RGB2BGR))
    report.update(status='completed', completed_unix=time.time())
    write_json(dest/'results.json', report); status('completed')
    print('COMPLETED', dest, flush=True)


def report_all():
    cards = []; reports = []
    for task in tasks():
        folder = OUT/f"worker{task['worker']}_{task['name']}_seed{task['seed']}"
        path = folder/'results.json'
        if not path.exists(): continue
        result = json.loads(path.read_text()); reports.append(result)
        score = result['true_trajectory_pixel_metrics']
        cards.append(f'<h2>{html.escape(task["name"])} seed {task["seed"]}</h2><p>{html.escape(task["scene"])}; refs {task["references"]}, target {task["target"]}; true PSNR {score["psnr_mean_db"]:.2f} dB, SSIM {score["ssim_mean"]:.3f}</p><video controls preload="metadata" style="width:100%" src="{folder.name}/gt_true_frozen_reversed.mp4"></video>')
    write_json(OUT/'summary.json', dict(status='completed' if len(reports)==4 else 'partial', completed=len(reports), total=4, cases=reports,
               caveat='camera intervention sensitivity does not establish correct tracking; reversed/frozen requests have no corresponding dynamic-action GT video'))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><body style="max-width:1800px;margin:auto;font-family:sans-serif"><h1>11900 two-reference trajectory diagnosis</h1><p>GT / true / frozen / reversed. g=0, 35 steps, no target GT prefix. Fixed noise/reference/text within each comparison. Four development cases, not an official benchmark.</p>'+''.join(cards)+'</body>')
    print('REPORT', len(reports), '/4', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int)
    parser.add_argument('--report', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.report: report_all()
    elif args.worker is not None:
        try: worker(args.worker)
        except BaseException as error:
            task = tasks()[args.worker]; dest = OUT/f"worker{args.worker}_{task['name']}_seed{task['seed']}"
            write_json(dest/'status.json', dict(stage='failed', error=repr(error), updated_unix=time.time())); raise
    else: parser.error('Choose --prepare, --worker, or --report')
