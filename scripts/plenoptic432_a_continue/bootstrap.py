"""Existing SVD environment, local outputs/caches, local 432p module namespace."""
import os
import json
import time
from pathlib import Path
import runpy
import sys

LAB=Path(__file__).resolve().parents[2]
SVD=Path('/root/wenfy/SVD')
PYTHON=SVD/'.venv/bin/python'
# Explicit local routing applies only to future evaluation subprocesses.
# Running training processes and all other scripts retain their assigned GPUs.
routing=LAB/'results/basic432_20261118/gpu_routing.json'
if len(sys.argv)>1 and routing.exists():
    route=json.loads(routing.read_text())
    target_path=Path(sys.argv[1]).resolve()
    if str(target_path)==route['script'] and route['required_argument'] in sys.argv[2:]:
        current=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if current in route['mapping']:
            os.environ['CUDA_VISIBLE_DEVICES']=route['mapping'][current]
            print(f"Local evaluation GPU routing: {current} -> {os.environ['CUDA_VISIBLE_DEVICES']}",flush=True)
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
        gate=route.get('wait_for_training_completion',{}).get(gpu)
        if gate:
            last_notice=0
            while True:
                training=json.loads(Path(gate).read_text())
                if training.get('stage')=='completed':
                    # Status is written just before the training process returns.
                    launch=json.loads((Path(gate).parent.parent/'launch.json').read_text())
                    name='train_'+Path(gate).parent.name
                    pid=next(item['pid'] for item in launch['workers'] if item['name']==name)
                    proc=Path(f'/proc/{pid}/cmdline')
                    if not proc.exists() or not proc.read_bytes():break
                if training.get('stage')=='failed':raise RuntimeError('Training failed; inspect before queued evaluation')
                if time.monotonic()-last_notice>30:
                    print(f"Waiting for GPU{gpu} training to exit before evaluation; update={training.get('update')}",flush=True)
                    last_notice=time.monotonic()
                time.sleep(2)
policy_path=LAB/'results/basic432_20261118/gpu_policy.json'
if policy_path.exists():
    policy=json.loads(policy_path.read_text())
    allowed={str(index) for index in policy['allowed_gpus']}
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(index) for index in policy['allowed_gpus'])
    selected={value.strip() for value in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if value.strip()}
    if not selected.issubset(allowed):
        raise RuntimeError(f"GPU assignment {selected} violates the user's allowed GPUs {allowed}")
paths=[LAB/'runtime',Path(__file__).resolve().parent,LAB,SVD/'scripts/plenoptic',SVD,
       SVD/'vendor/cosmos-transfer2.5',SVD/'vendor/cosmos-transfer2.5/packages/cosmos-oss',SVD/'vendor/cosmos-transfer2.5/packages/cosmos-cuda']
if Path(sys.prefix).resolve()!=PYTHON.parent.parent.resolve():
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,paths)),UV_CACHE_DIR=str(LAB/'.cache/uv'),
             IMAGINAIRE_OUTPUT_ROOT=str(LAB/'results/basic432_20261118/cosmos'))
    os.execve(str(PYTHON),[str(PYTHON),'-B',str(Path(__file__).resolve()),*sys.argv[1:]],env)
sys.path[:0]=list(map(str,paths))
target=Path(sys.argv[1]).resolve();target.relative_to(LAB)
sys.argv=[str(target),*sys.argv[2:]]
runpy.run_path(str(target),run_name='__main__')
