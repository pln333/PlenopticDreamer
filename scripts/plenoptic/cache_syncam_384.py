"""SynCam cache using the sibling run's exact reader and standalone Wan VAE.

Only a new SynCam subdirectory is written in the shared cache. Logs, source
snapshots and control files stay in SVD. No training is launched here.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / 'results/syncam_cache_384_20261117'
DATA = Path('/data/datasets/nvs/SynCamVideo-Dataset')
CACHE = Path('/home/pln/3dvision/nvs/recammaster/cache/syncam_384x672_wan21_v1')
MULTICAM = Path('/home/pln/3dvision/nvs/recammaster/cache/v2v_empty_384x672_v1')
MULTIDATA = Path('/data/nvs/recammaster/datasets/MultiCamVideo-extracted/MultiCamVideo-Dataset')
VENDOR = ROOT / 'vendor/recammaster_cache'
PREPROCESS = 'official_81_384_672_tiled34x34_stride18x16_bf16_v1'
SHAPE = [16, 21, 48, 84]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract():
    c = json.loads((JOB / 'contract.json').read_text())
    if sha(JOB / 'sources.jsonl') != c['sources_sha256']:
        raise ValueError('Source index changed')
    for name, expected in c['code_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Code changed: ' + name)
    return c


def prepare():
    JOB.mkdir(parents=True, exist_ok=True)
    if (JOB / 'contract.json').exists():
        print('EXISTING_CONTRACT', contract()['videos'], flush=True)
        return
    manifest = json.loads((DATA / 'source_manifest.json').read_text())
    assert manifest['complete']
    entries = {r['path']: r for r in manifest['files']}
    rows = []
    for path, r in sorted(entries.items()):
        if not path.endswith('.mp4'):
            continue
        p = Path(path)
        assert p.parts[0] in ('train', 'val') and p.parent.name == 'videos'
        camera_path = str(p.parent.parent / 'cameras/camera_extrinsics.json')
        assert camera_path in entries
        rows.append(dict(source=path, source_sha256=r['sha256'], source_bytes=r['bytes'],
                         source_split=p.parts[0], scene=str(p.parent.parent),
                         camera=p.stem, camera_source=camera_path,
                         camera_source_sha256=entries[camera_path]['sha256']))
    assert len(rows) == 34010
    index = JOB / 'sources.jsonl'
    with index.open('x') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    # The original reader is called through load_video(absolute_path), never
    # __getitem__. Its metadata only initializes the exact preprocessing.
    with (JOB / 'reader_metadata.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'text'])
        w.writerow(['unused.mp4', ''])
    weight = json.loads((ROOT / 'results/tokenizer_reconstruction/tokenizer_weight.json').read_text())
    weight_path = ROOT / weight['local_path']
    assert sha(weight_path) == weight['sha256']
    code = [Path(__file__), VENDOR / 'wan_video_vae.py', VENDOR / 'video_dataset.py']
    c = dict(dataset='SynCamVideo', repo=manifest['repo'], revision=manifest['revision'],
             source_manifest_sha256=sha(DATA / 'source_manifest.json'),
             sources_sha256=sha(index), data_root=str(DATA), cache_root=str(CACHE),
             videos=len(rows), source_train_videos=34000, source_val_videos=10,
             shape=SHAPE, dtype='bfloat16', preprocess_id=PREPROCESS,
             weight=str(weight_path), vae_sha256=weight['sha256'],
             code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
             source_snapshot=json.loads((VENDOR / 'provenance.json').read_text()),
             preprocessing=dict(reader='sibling TextVideoDataset.load_video; PIL bilinear resize then torchvision center crop',
                                rgb_shape=[3, 81, 384, 672], frames=list(range(81)),
                                rgb_normalization='float32 ToTensor/Normalize, then bfloat16',
                                original_shape=[1280, 1280], resized_shape=[672, 672],
                                crop_left=0, crop_top=144, width=672, height=384,
                                pixel_affine=[[0.525, 0, 0], [0, 0.525, -144], [0, 0, 1]]),
             training_split='Not assigned by cache extraction. Preserve source train/val and hold out historical validation scene IDs across both datasets when building the training index.',
             text='Video VAE only. No T5 empty embedding; Cosmos text features prepared separately.')
    atomic_json(JOB / 'contract.json', c)
    print(json.dumps(c, indent=2), flush=True)


def acquire_gpu():
    """Submit work to vLLM-occupied GPUs without stopping any process."""
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    probe = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
    started = time.monotonic()
    # Small real workload lets the opportunistic service observe demand.
    while torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        result = probe @ probe
        torch.cuda.synchronize()
        if time.monotonic() - started > 120:
            raise RuntimeError('GPU has not yielded 18 GiB; no other process was stopped')
    del probe
    torch.cuda.empty_cache()


def make_codec(c):
    import torch
    module = load_module('sibling_wan_vae', VENDOR / 'wan_video_vae.py')
    # Avoid per-tile progress bars in long-running logs; codec code is unchanged.
    module.tqdm = lambda iterable, **kwargs: iterable
    if sha(c['weight']) != c['vae_sha256']:
        raise ValueError('VAE weight hash differs')
    # mean/std are ordinary tensors in this upstream class, so constructing
    # under a meta device would leave those normalization tensors on meta.
    codec = module.WanVideoVAE().to(dtype=torch.bfloat16)
    state = torch.load(c['weight'], map_location='cpu', weights_only=True)
    state = codec.state_dict_converter().from_civitai(state)
    codec.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True, assign=True)
    codec = codec.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    reader_module = load_module('sibling_video_dataset', VENDOR / 'video_dataset.py')
    reader = reader_module.TextVideoDataset(str(DATA), str(JOB / 'reader_metadata.csv'), height=384, width=672)
    return codec, reader


def encode(codec, reader, source):
    import torch
    video = reader.load_video(str(source))
    if video is None or list(video.shape) != [3, 81, 384, 672]:
        raise ValueError('Bad decoded video: ' + str(source))
    # Matches pipe.encode_video(video.unsqueeze(0).to(cuda,bfloat16)).
    with torch.inference_mode():
        z = codec.encode(video[None].to('cuda', dtype=torch.bfloat16), device='cuda',
                         tiled=True, tile_size=(34, 34), tile_stride=(18, 16))[0].cpu()
    if list(z.shape) != SHAPE or z.dtype != torch.bfloat16 or not torch.isfinite(z).all():
        raise ValueError('Invalid latent: ' + str(source))
    return z, video


def validate():
    import numpy as np
    import torch
    c = contract()
    acquire_gpu()
    codec, reader = make_codec(c)
    comparisons = []
    for rel in ['f18_aperture10/scene1/videos/cam01.mp4',
                'f35_aperture2.4/scene473/videos/cam06.mp4']:
        source = MULTIDATA / 'train' / rel
        cached = torch.load(MULTICAM / 'latents' / Path(rel).with_suffix('.pt'),
                            map_location='cpu', weights_only=True)
        assert sha(source) == cached['source_sha256']
        start = time.perf_counter()
        z, video = encode(codec, reader, source)
        diff = z.float() - cached['latents'].float()
        item = dict(source=rel, seconds=time.perf_counter() - start,
                    exact_equal=torch.equal(z, cached['latents']),
                    mean_abs=float(diff.abs().mean()), max_abs=float(diff.abs().max()),
                    relative_rmse=float(diff.square().mean().sqrt() / cached['latents'].float().square().mean().sqrt()))
        if item['mean_abs'] > .02 or item['relative_rmse'] > .03:
            raise ValueError('Sibling cache parity failed: ' + str(item))
        comparisons.append(item)
        print('SIBLING_PARITY', json.dumps(item), flush=True)
    # Independently decode a sibling latent with the Cosmos tokenizer to check
    # the normalization and channel semantics used by our generation model.
    del codec
    torch.cuda.empty_cache()
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    decoder = Wan2pt1VAEInterface(vae_pth=c['weight'], load_mean_std=False,
                                 chunk_duration=81, temporal_window=16)
    with torch.inference_mode():
        recon = decoder.decode(cached['latents'][None].to('cuda')).float().cpu().clamp(-1, 1)
    assert recon.shape == video[None].shape and torch.isfinite(recon).all()
    frame_mse = ((recon[0] - video) / 2).square().mean(dim=(0, 2, 3))
    psnr = -10 * torch.log10(frame_mse.clamp_min(1e-12))
    if psnr.mean() < 20:
        raise ValueError('Cosmos cached-latent reconstruction below 20 dB')
    # Save one actual comparison video for review.
    from reconstruct_tokenizer import write_video
    gt = ((video.permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    rgb = ((recon[0].permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    write_video(JOB / 'sibling_cache_cosmos_decode_comparison.mp4', np.concatenate([gt, rgb], axis=2), 15)
    result = dict(status='passed', contract_sha256=sha(JOB / 'contract.json'),
                  sibling_parity=comparisons, cosmos_decode_psnr_mean_db=float(psnr.mean()),
                  cosmos_decode_psnr_per_frame_db=psnr.tolist(), validated_unix=time.time(),
                  note='Cache codec/reader parity and Cosmos decoder compatibility, not generation quality.')
    atomic_json(JOB / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def worker(shard, shards):
    import shutil
    import torch
    c = contract()
    gate = json.loads((JOB / 'validation.json').read_text())
    assert gate['status'] == 'passed' and gate['contract_sha256'] == sha(JOB / 'contract.json')
    lock = (JOB / f'worker_{shard}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [json.loads(s) for s in (JOB / 'sources.jsonl').read_text().splitlines()][shard::shards]
    CACHE.mkdir(parents=True, exist_ok=True)
    acquire_gpu()
    codec, reader = make_codec(c)
    created = reused = 0
    started = time.monotonic()
    statepath = JOB / f'worker_{shard}.json'

    def read(r):
        p = DATA / r['source']
        if p.stat().st_size != r['source_bytes'] or sha(p) != r['source_sha256']:
            raise ValueError('Original video hash differs: ' + str(p))
        return r

    with (JOB / f'journal_{shard}.jsonl').open('a') as journal:
        for i, r in enumerate(rows):
            tick = time.monotonic()
            source = DATA / r['source']
            dest = CACHE / 'latents' / Path(r['source']).with_suffix('.pt')
            was_reused = dest.exists()
            if shutil.disk_usage(CACHE).free < 30 * 1024**3:
                raise RuntimeError('Cache disk has less than 30 GiB free')
            if was_reused:
                v = torch.load(dest, map_location='cpu', weights_only=True)
                assert v['source_sha256'] == r['source_sha256'] and v['preprocess_id'] == PREPROCESS
                assert v['dataset'] == 'SynCamVideo' and list(v['latents'].shape) == SHAPE
                assert v['vae_sha256'] == c['vae_sha256'] and v['latents'].dtype == torch.bfloat16
                assert torch.isfinite(v['latents']).all()
                reused += 1
            else:
                read(r)
                z, _ = encode(codec, reader, source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(f'.{os.getpid()}.part')
                torch.save(dict(latents=z, dataset='SynCamVideo', source=r['source'],
                                source_sha256=r['source_sha256'], preprocess_id=PREPROCESS,
                                source_split=r['source_split'], vae_sha256=c['vae_sha256'],
                                contract_sha256=gate['contract_sha256'],
                                source_camera_sha256=r['camera_source_sha256']), tmp)
                # Read back before publishing the file.
                check = torch.load(tmp, map_location='cpu', weights_only=True)
                assert torch.equal(check['latents'], z)
                tmp.replace(dest)
                created += 1
            entry = dict(index=shard + shards*i, source=r['source'], cache=str(dest),
                         cache_sha256=sha(dest), bytes=dest.stat().st_size,
                         seconds=time.monotonic()-tick, reused=was_reused,
                         time=time.time())
            journal.write(json.dumps(entry)+'\n')
            journal.flush()
            elapsed = time.monotonic()-started
            status = dict(status='completed' if i+1 == len(rows) else 'running', pid=os.getpid(),
                          shard=shard, shards=shards, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          completed=i+1, total=len(rows), created=created, reused=reused,
                          elapsed_seconds=elapsed, seconds_per_video=elapsed/(i+1),
                          remaining_seconds=(len(rows)-i-1)*elapsed/(i+1),
                          last_source=r['source'], updated_unix=time.time())
            atomic_json(statepath, status)
            if i < 3 or (i+1) % 25 == 0 or i+1 == len(rows):
                print(json.dumps(status), flush=True)


def supervise():
    c = contract()
    gate = json.loads((JOB / 'validation.json').read_text())
    assert gate['status'] == 'passed' and gate['contract_sha256'] == sha(JOB / 'contract.json')
    lock = (JOB / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    CACHE.mkdir(parents=True, exist_ok=True)
    contract_dst = CACHE / 'cache_contract.json'
    if contract_dst.exists():
        assert contract_dst.read_bytes() == (JOB / 'contract.json').read_bytes()
    else:
        atomic_json(contract_dst, c)
    children = []
    logs = []
    for shard, gpu in enumerate([4, 5, 6, 7]):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        log = (JOB / f'worker_{shard}.log').open('a')
        logs.append(log)
        cmd = [str(ROOT/'run.sh'), '-u', str(Path(__file__)), 'worker', '--shard', str(shard), '--shards', '4']
        children.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    atomic_json(JOB / 'launch.json', dict(supervisor_pid=os.getpid(), workers=[dict(pid=p.pid, gpu=g) for p,g in zip(children,[4,5,6,7])], started_unix=time.time()))
    while True:
        exits = [p.poll() for p in children]
        states = [json.loads((JOB/f'worker_{i}.json').read_text()) for i in range(4) if (JOB/f'worker_{i}.json').exists()]
        failed = any(x not in (None, 0) for x in exits)
        done = all(x is not None for x in exits)
        completed = sum(s['completed'] for s in states)
        status = 'completed' if done and not failed and completed == c['videos'] else ('failed' if failed else 'running')
        atomic_json(JOB/'status.json', dict(status=status, total=c['videos'], completed=completed,
                                          progress=completed/c['videos'], cache_root=str(CACHE),
                                          workers=states, worker_exit_codes=exits, updated_unix=time.time()))
        if done:
            if status != 'completed':
                raise RuntimeError('At least one worker failed; preserve successful shards and inspect logs')
            # All workers verify each published tensor and append file hashes.
            count = 0
            seen = set()
            with (JOB/'manifest_verified.jsonl').open('w') as out:
                for i in range(4):
                    for line in (JOB/f'journal_{i}.jsonl').read_text().splitlines():
                        row = json.loads(line)
                        if row['source'] not in seen:
                            out.write(json.dumps(row)+'\n'); seen.add(row['source']); count += 1
            assert count == c['videos']
            atomic_json(CACHE/'complete.json', dict(status='completed', files=count,
                                                  contract_sha256=sha(JOB/'contract.json'),
                                                  manifest_sha256=sha(JOB/'manifest_verified.jsonl'),
                                                  manifest=str(JOB/'manifest_verified.jsonl'),
                                                  completed_unix=time.time()))
            print('COMPLETE', count, flush=True)
            return
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['prepare', 'validate', 'worker', 'supervise'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=4)
    a = ap.parse_args()
    if a.mode == 'worker':
        assert 0 <= a.shard < a.shards
    try:
        if a.mode == 'prepare': prepare()
        elif a.mode == 'validate': validate()
        elif a.mode == 'worker': worker(a.shard, a.shards)
        else: supervise()
    except Exception as e:
        atomic_json(JOB / f'error_{a.mode}_{a.shard}.json', dict(error=str(e), traceback=traceback.format_exc(), pid=os.getpid(), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
