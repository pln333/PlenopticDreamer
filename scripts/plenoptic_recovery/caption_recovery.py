"""Auditable caption retries; omit unavailable text without dropping video samples."""
import json
import os
from pathlib import Path
import re
import time

import torch
import cached_basic_384_data as data_module
from prepare_expanded import concise_caption
from cache_syncam_384 import atomic_json, sha

LAB = Path(__file__).resolve().parents[2]
JOB = LAB / 'results/basic_cached384_recovery_20261118'
OLD = Path('/root/wenfy/SVD/results/basic_cached384_20261117')
POLICY = 'bounded_caption_retry_then_zero_text_v1'
PROMPT = ('These images show the beginning, middle and end of one video. Write a short factual English caption, '
          'at most 60 words, describing the visible subject, movement and surroundings. Do not invent hidden '
          'objects or actions. Return only complete caption sentences, without reasoning.')
RETRY = ('Describe only what is visible in these three video frames. Reply with one short English sentence '
         'of at most 30 words, ending with a period. Do not explain or reason.')


def parse_caption(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Empty caption')
    if len(re.findall(r'<think>', raw)) != len(re.findall(r'</think>', raw)):
        raise ValueError('Unfinished reasoning')
    caption = concise_caption(raw)
    if not re.search('[A-Za-z]', caption):
        raise ValueError('No English caption text')
    return caption


def generate_caption(generate):
    attempts = []
    for prompt, budget in [(PROMPT, 192), (PROMPT, 384), (PROMPT, 512), (RETRY, 192)]:
        raw = generate(prompt, budget)
        try:
            caption = parse_caption(raw)
            attempts.append(dict(prompt=prompt, budget=budget, raw=raw, status='accepted'))
            return caption, attempts
        except ValueError as error:
            attempts.append(dict(prompt=prompt, budget=budget, raw=raw, status='rejected', error=str(error)))
    # No fabricated scene description or target-derived text is substituted.
    return '', attempts


class ReferenceTextEncoder(data_module.ReferenceTextEncoder):
    @torch.no_grad()
    def caption(self, data, scene, reference):
        from PIL import Image
        import imageio
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        row = data.scenes[scene]; entry = row['videos'][reference]
        path = JOB / 'captions' / scene / f'{reference}.json'
        for cached in [path, OLD / 'captions' / scene / f'{reference}.json']:
            if cached.exists():
                record = json.loads(cached.read_text())
                assert record['source_sha256'] == entry['source_sha256']
                assert record['scene'] == scene and record['reference_camera'] == reference
                if record['caption']:
                    return record['caption']
                assert record.get('text_condition_omitted') and record.get('policy') == POLICY
                return ''
        source = data.root(scene) / entry['source']
        assert sha(source) == entry['source_sha256']
        reader = imageio.get_reader(str(source)); images = []
        try:
            for frame in [0, 40, 80]:
                im = Image.fromarray(reader.get_data(frame)); w, h = im.size
                scale = max(672 / w, 384 / h)
                im = TF.resize(im, (round(h * scale), round(w * scale)), interpolation=InterpolationMode.BILINEAR)
                im = TF.center_crop(im, (384, 672))
                images.append(im.resize((336, 192)))
        finally:
            reader.close()

        def generate(prompt, budget):
            messages = [{'role': 'user', 'content': [*[{'type': 'image'} for _ in images], {'type': 'text', 'text': prompt}]}]
            conversation = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[conversation], images=images, return_tensors='pt').to('cuda')
            result = self.model.generate(**inputs, do_sample=False, max_new_tokens=budget, use_cache=True)
            return self.processor.batch_decode(result[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        caption, attempts = generate_caption(generate)
        record = dict(caption=caption, attempts=attempts, source_sha256=entry['source_sha256'],
                      source_video=str(source), scene=scene, reference_camera=reference, frames=[0, 40, 80],
                      policy=POLICY, text_condition_omitted=not bool(caption), generated_unix=time.time(),
                      source='automatic Cosmos-Reason1-7B annotation; not human reviewed or author-provided')
        atomic_json(path, record)
        if len(attempts) > 1 or not caption:
            print(json.dumps(dict(event='caption_recovery', scene=scene, reference=reference,
                                  attempts=len(attempts), text_condition_omitted=not bool(caption)), ensure_ascii=False), flush=True)
        return caption

    @torch.no_grad()
    def embedding(self, caption):
        if not caption:
            return torch.zeros((1, 512, 100352), device='cuda', dtype=torch.bfloat16)
        return super().embedding(caption)


def install():
    data_module.ReferenceTextEncoder = ReferenceTextEncoder
