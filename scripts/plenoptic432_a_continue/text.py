"""Reference-only Cosmos captions and embeddings for the exact 432p crop."""
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image

from data import JOB, write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plenoptic_recovery'))
from caption_recovery import generate_caption, POLICY
from basic_full_data import TextEncoder


class ReferenceTextEncoder(TextEncoder):
    @torch.no_grad()
    def caption(self,data,scene,reference):
        entry=data.scenes[scene]['videos'][reference]
        path=JOB/'captions'/scene/f'{reference}.json'
        if path.exists():
            record=json.loads(path.read_text())
            assert record['source_sha256']==entry['source_sha256'] and record['preprocess_id']==data.manifest['preprocess_id']
            return record['caption']
        # Caption only reference frames; target content never enters this path.
        rgb=data.read_rgb(scene,reference)
        images=[Image.fromarray(rgb[t]).resize((384,216)) for t in [0,40,80]]
        def generate(prompt,budget):
            messages=[{'role':'user','content':[*[{'type':'image'} for _ in images],{'type':'text','text':prompt}]}]
            conversation=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[conversation],images=images,return_tensors='pt').to('cuda')
            result=self.model.generate(**inputs,do_sample=False,max_new_tokens=budget,use_cache=True)
            return self.processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
        caption,attempts=generate_caption(generate)
        write_json(path,dict(scene=scene,reference_camera=reference,caption=caption,attempts=attempts,frames=[0,40,80],
                   source_sha256=entry['source_sha256'],preprocess_id=data.manifest['preprocess_id'],
                   policy=POLICY,text_condition_omitted=not bool(caption),generated_unix=time.time(),
                   source='automatic Cosmos-Reason1-7B reference-only annotation, not author/human captions'))
        return caption

    @torch.no_grad()
    def embedding(self,caption):
        if not caption:return torch.zeros((1,512,100352),device='cuda',dtype=torch.bfloat16)
        return super().embedding(caption)
