import os,sys,time,subprocess,json,traceback,fcntl
from pathlib import Path
from data import write_json
LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'results/basic432_20261118/a_continue_batch40_20261120/A'
BOOT=LAB/'scripts/plenoptic432_a_continue/bootstrap.py'
TRAIN=LAB/'scripts/plenoptic432_a_continue/overnight_camera_abc.py'
EVAL=LAB/'scripts/plenoptic432_a_continue/overnight_camera_abc_evaluate.py'
def status(stage,**args):write_json(ROOT/'continuation_status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**args))
def run(cmd,env,log):
    with log.open('a') as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rc=p.wait()
    if rc:raise RuntimeError(f'process failed with status {rc}; see {log}')
def main():
    lock=(ROOT/'continue_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN')
    base=[str(LAB/'run.sh'),str(BOOT)]
    status('smoke_training')
    run(base+[str(TRAIN),'--distributed','A','--smoke'],dict(env,CONTINUE_START='0',CONTINUE_END='1'),ROOT/'smoke.log')
    for rank in range(4):assert json.loads((ROOT/f'smoke_train/rank{rank}.json').read_text())['stage']=='completed'
    for end in range(100,1001,100):
        start=end-100
        status('training',start=start,end=end)
        run(base+[str(TRAIN),'--distributed','A'],dict(env,CONTINUE_START=str(start),CONTINUE_END=str(end)),ROOT/f'train_{end:05d}.log')
        for rank in range(4):
            state=json.loads((ROOT/f'train/rank{rank}.json').read_text());assert state['stage']=='completed' and state['update']==end
        assert (ROOT/f'train/step_{end:05d}/training_state.pt').exists()
        status('evaluating',update=end)
        children=[]
        for shard,gpu in enumerate([0,1]):
            with (ROOT/f'evaluate_{end:05d}_shard{shard}.log').open('a') as f:
                children.append(subprocess.Popen(base+[str(EVAL),'--arm','A','--shard',str(shard)],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),CONTINUE_EVAL_UPDATE=str(end)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        codes=[p.wait() for p in children]
        if any(codes):raise RuntimeError(f'evaluation failed at {end}: {codes}')
        summary=json.loads((ROOT/f'evaluations/step_{end:05d}/SUMMARY.json').read_text())
        assert summary['status']=='completed'
    status('completed',update=1000)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
