"""Measure exact three-frame caption reading without changing the running job."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from data import Data432, JOB, sha, write_json
from syncam_data import transform_frame

cv2.setNumThreads(1)
data=Data432()
rows=[json.loads(s) for s in (JOB/'train/rank0.jsonl').read_text().splitlines()][-6:]
reports=[]
for i,row in enumerate(rows):
    scene=row['scene'];camera=row['references'][0]
    def sparse():
        entry=data.scenes[scene]['videos'][camera];source=data.source(scene,camera)
        assert source.stat().st_size==entry['source_bytes'] and sha(source)==entry['source_sha256']
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        frames=[]
        try:
            for frame_id in [0,40,80]:
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
                ok,frame=cap.read();assert ok
                frames.append(cv2.cvtColor(transform_frame(frame,data.manifest['preprocessing']),cv2.COLOR_BGR2RGB))
        finally:cap.release()
        return np.stack(frames)
    funcs={'full':lambda:data.read_rgb(scene,camera)[[0,40,80]],'sparse':sparse}
    order=['full','sparse'] if i%2==0 else ['sparse','full']
    times={};outputs={}
    for mode in order:
        start=time.perf_counter();outputs[mode]=funcs[mode]();times[mode]=time.perf_counter()-start
    equal=np.array_equal(outputs['full'],outputs['sparse']);assert equal
    reports.append(dict(scene=scene,camera=camera,exact_pixel_equal=equal,**times))
result=dict(status='passed',scope='CPU-only six-video paired timing; running trainer unchanged',cases=reports,
            mean_full_seconds=float(np.mean([r['full'] for r in reports])),
            mean_sparse_seconds=float(np.mean([r['sparse'] for r in reports])))
write_json(JOB/'caption_read_benchmark.json',result)
print(json.dumps(result,indent=2))
