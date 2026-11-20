"""Build an immutable 100-train/12-validation SynCamVideo dataset.

Keep the original 12/4 split and real caches intact. New captions are explicitly
automatic annotations from real cam02 frames, not dataset-provided labels.
"""
import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from multiscene_data import DATA, ROOT, PilotData, digest, write_json
from syncam_data import center_crop_parameters, read_rgb_frames, scene_cameras


def cached_latent(path, metadata):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return (handle.metadata() == metadata
                and tuple(handle.get_slice("latent").get_shape()) == (1, 16, 21, 54, 96))


def concise_caption(raw):
    """Keep complete opening sentences, retaining the raw generation for audit."""
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    clean = re.sub(r"</?answer>", "", clean).strip()
    if "<think>" in clean:
        raise ValueError("Unfinished reasoning in caption output")
    sentences = re.findall(r".+?[.!?](?:\s+|$)", clean, flags=re.DOTALL)
    selected = []
    for sentence in sentences:
        if len(selected) >= 3 or len(" ".join(selected+[sentence]).split()) > 80:
            break
        selected.append(sentence.strip())
    if not selected:
        raise ValueError("Caption generation has no complete bounded sentence")
    return " ".join(selected)


def audit(raw, output):
    source = json.loads((raw / "source_manifest.json").read_text())
    if not source["complete"] or len(source["scenes"]) != 112:
        raise ValueError("Need 112 complete real-video scene groups")
    for item in source["files"]:
        if digest(raw / item["path"]) != item["sha256"]:
            raise ValueError(f"Raw source checksum mismatch: {item['path']}")
    old = PilotData()
    old.verify()
    if not set(old.scenes) <= set(source["scenes"]):
        raise ValueError("Expanded selection must contain all 16 historical scenes")
    new = sorted(set(source["scenes"]) - set(old.scenes),
                 key=lambda scene: hashlib.sha256(f"expanded100-v1/{scene}".encode()).hexdigest())
    validation = set(old.val + new[-8:])
    rows = {}
    camera_dir = output / "cameras"
    camera_dir.mkdir(exist_ok=True)
    images_dir = output / "caption_frames"
    images_dir.mkdir(exist_ok=True)
    from audit_syncam import epipolar_check
    geometries = []
    for scene in source["scenes"]:
        if scene in old.scenes:
            rows[scene] = dict(old.scenes[scene])
            # Confirm the larger extraction contains identical old videos.
            for camera, path in rows[scene]["videos"].items():
                if digest(ROOT / path) != digest(raw / scene / "videos" / f"{camera}.mp4"):
                    raise ValueError("Historical and expanded source videos disagree")
            continue
        folder = raw / scene
        names, frames, c2w, w2c = scene_cameras(folder / "cameras/camera_extrinsics.json")
        config, original_k, affine, k = center_crop_parameters(1280, 1280)
        rotations = c2w[..., :3, :3]
        errors = dict(rotation=float(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)).max()),
                      determinant=float(np.abs(np.linalg.det(rotations) - 1).max()),
                      inverse=float(np.abs(c2w @ w2c - np.eye(4)).max()),
                      static=float(np.abs(c2w - c2w[:, :1]).max()))
        if len(frames) != 81 or not np.isfinite(c2w).all() or max(errors.values()) > 1e-4:
            raise ValueError(f"Invalid camera geometry: {scene}, {errors}")
        camera_path = camera_dir / f"{scene}.npz"
        if not camera_path.exists():
            np.savez_compressed(camera_path, c2w=c2w, w2c=w2c, K_original=original_k,
                                K_processed=k, pixel_affine=affine, frame_indices=np.arange(81))
        else:
            with np.load(camera_path) as saved:
                if not np.array_equal(saved["c2w"], c2w) or not np.array_equal(saved["K_processed"], k):
                    raise ValueError("Existing camera cache disagrees with source")
        videos = {}
        for camera in names:
            path = folder / "videos" / f"{camera}.mp4"
            cap = cv2.VideoCapture(str(path))
            values = [cap.get(key) for key in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS,
                                              cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
            cap.release()
            if not np.allclose(values, [81, 15, 1280, 1280], atol=1e-4):
                raise ValueError(f"Unexpected video metadata: {path}, {values}")
            videos[camera] = str(path.relative_to(ROOT))
        review_paths = []
        rgb = read_rgb_frames(ROOT / videos["cam02"], config, [0, 40, 80]).transpose(0, 2, 3, 1)
        for frame, pixels in zip((0, 40, 80), rgb):
            path = images_dir / f"{scene}_cam02_f{frame:02d}.jpg"
            if not cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(pixels, (384, 216)), cv2.COLOR_RGB2BGR)):
                raise RuntimeError("Failed to save caption review frame")
            review_paths.append(str(path.relative_to(ROOT)))
        # Known-camera check, never fit F to predictions. Failure to match is
        # retained as diagnostic evidence, not incorrectly treated as zero.
        for target in ("cam03", "cam06"):
            target_rgb = read_rgb_frames(ROOT / videos[target], config, [0]).transpose(0, 2, 3, 1)[0]
            score, _ = epipolar_check(rgb[0], target_rgb, c2w[1, 0], c2w[int(target[3:])-1, 0], k)
            geometries.append(dict(scene=scene, reference="cam02", target=target, **score))
        rows[scene] = dict(split="val" if scene in validation else "train", videos=videos,
                           camera_npz=str(camera_path.relative_to(ROOT)), preprocessing=config,
                           num_frames=81, fps=15, caption_frames=review_paths, camera_audit=errors)
        print(f"audited {scene} ({len(rows)}/112), split={rows[scene]['split']}", flush=True)
    counts = {split: sum(row["split"] == split for row in rows.values()) for split in ("train", "val")}
    if counts != {"train": 100, "val": 12}:
        raise ValueError(f"Incorrect split sizes: {counts}")
    write_json(output / "rows.json", dict(scenes=rows, split_counts=counts,
               protected_validation_scenes=sorted(validation), historical_validation_scenes=old.val,
               new_validation_scenes=sorted(validation-set(old.val)),
               source_manifest=str((raw / "source_manifest.json").relative_to(ROOT)),
               source_manifest_sha256=digest(raw / "source_manifest.json"),
               selection="first 112 complete archive scenes; preserve historical split; hash-select 8 new validation scenes",
               limitations="Static synthetic cameras; archive-order subset, not a representative sample of the full dataset"))
    write_json(output / "geometry_audit.json", dict(camera_source="annotated real-data cameras",
               calibration_checks=geometries, scope="two known-camera SIFT checks per new scene, RGB frame zero"))
    print(f"AUDIT COMPLETE: {counts}", flush=True)


def latents(output):
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    rows = json.loads((output / "rows.json").read_text())
    record = json.loads((ROOT / "results/tokenizer_reconstruction/tokenizer_weight.json").read_text())
    weight = ROOT / record["local_path"]
    if digest(weight) != record["sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    tokenizer = Wan2pt1VAEInterface(vae_pth=str(weight), load_mean_std=False, chunk_duration=81, temporal_window=16)
    report = {**rows, "tokenizer_sha256": record["sha256"], "scenes": {}}
    count = 0
    for scene, row in rows["scenes"].items():
        destination = output / "latents" / scene
        destination.mkdir(parents=True, exist_ok=True)
        entries = {}
        camera_sha = digest(ROOT / row["camera_npz"])
        for camera, relative in sorted(row["videos"].items()):
            source = ROOT / relative
            metadata = {"scene": scene, "camera": camera, "layout": "BCTHW", "source_kind": "real_video_vae_encoding",
                        "tokenizer_sha256": record["sha256"], "source_video_sha256": digest(source),
                        "source_camera_sha256": camera_sha, "preprocessing": json.dumps(row["preprocessing"], sort_keys=True),
                        "rgb_frames": "81", "encoder_temporal_window": "16"}
            path = destination / f"{camera}.safetensors"
            inherited = row.get("latents", {}).get(camera)
            if inherited:
                path = ROOT / inherited["path"]
                if digest(path) != inherited["sha256"]:
                    raise ValueError("Inherited real latent changed")
            if path.exists():
                if not cached_latent(path, metadata):
                    raise ValueError(f"Cache provenance differs: {path}")
            else:
                begin = time.monotonic()
                rgb = read_rgb_frames(source, row["preprocessing"], list(range(81)))
                x = torch.from_numpy(rgb.copy()).permute(1, 0, 2, 3)[None].cuda().to(torch.bfloat16) / 127.5 - 1
                with torch.inference_mode():
                    latent = tokenizer.encode(x).float().cpu().contiguous()
                if latent.shape != (1, 16, 21, 54, 96) or not torch.isfinite(latent).all():
                    raise ValueError(f"Invalid real encoding: {scene}/{camera}")
                temporary = path.with_suffix(".tmp")
                save_file({"latent": latent}, str(temporary), metadata=metadata)
                temporary.replace(path)
                print(f"encoded {scene}/{camera} {time.monotonic()-begin:.2f}s", flush=True)
                del rgb, x, latent
                tokenizer.clear_cache()
            entries[camera] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            count += 1
        report["scenes"][scene] = dict(row, latents=entries)
        write_json(output / "latent_index.partial.json", report)
        write_json(output / "latent_progress.json", dict(completed_videos=count, total_videos=1120,
                   completed_scenes=len(report["scenes"]), updated_unix=time.time()))
        gc.collect()
    write_json(output / "latent_index.json", report)


def text(output):
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    rows = json.loads((output / "rows.json").read_text())["scenes"]
    weights = json.loads((ROOT / "results/text_encoder/weights.json").read_text())
    path = ROOT / weights["local_path"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, min_pixels=56*56, max_pixels=384*384)
    model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
        low_cpu_mem_usage=True, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Reason encoder load mismatch: {loading}")
    model = model.cuda().eval().requires_grad_(False)
    output.joinpath("text").mkdir(exist_ok=True)
    caption_path = output / "captions.json"
    captions = json.loads(caption_path.read_text()) if caption_path.exists() else dict(
        source="Historical annotations retained; new captions automatically generated by Cosmos-Reason1-7B from real cam02 frames 0/40/80; not human-reviewed or source-dataset captions",
        revision=weights["revision"], scenes={})
    report = {}
    prompt = ("These three images are frames at the beginning, middle and end of the same video. "
              "Write one factual video caption in English, at most 60 words. Describe the visible person, clothing, "
              "observable movement and surroundings. Be conservative about ambiguous motion. "
              "Do not invent actions or hidden objects. Return only the caption, without reasoning.")
    for scene, row in rows.items():
        if row.get("text"):
            report[scene] = row["text"]
            captions["scenes"][scene] = dict(caption=row["caption"], source="historical pilot annotation")
            write_json(caption_path, captions)
            continue
        if scene not in captions["scenes"]:
            images = [Image.open(ROOT / image_path).convert("RGB") for image_path in row["caption_frames"]]
            messages = [{"role": "user", "content": [*[{"type": "image"} for _ in images], {"type": "text", "text": prompt}]}]
            conversation = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[conversation], images=images, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=192, use_cache=True)
            answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            caption = concise_caption(answer)
            captions["scenes"][scene] = dict(caption=caption, raw_generation=answer,
                extraction="first up to three complete sentences, at most 80 words; incomplete trailing sentence discarded",
                source="automatic real-frame annotation; not human reviewed",
                frames=[dict(path=p, sha256=digest(ROOT/p)) for p in row["caption_frames"]], prompt=prompt)
            write_json(caption_path, captions)
            print(f"captioned {scene}: {caption}", flush=True)
            del images, inputs, generated
        elif "raw_generation" not in captions["scenes"][scene]:
            # Explicit migration of captions from the interrupted first attempt.
            entry = captions["scenes"][scene]
            entry["raw_generation"] = entry["caption"]
            entry["caption"] = concise_caption(entry["raw_generation"])
            entry["extraction"] = "first up to three complete sentences, at most 80 words; incomplete trailing sentence discarded"
            write_json(caption_path, captions)
        caption = captions["scenes"][scene]["caption"]
        target = output / "text" / f"{scene}.safetensors"
        metadata = dict(scene=scene, caption=caption, revision=weights["revision"],
                        source=captions["scenes"][scene].get("embedding_source",
                          "Cosmos-Reason1-7B automated real cam02 frame review 0/40/80; not human reviewed"))
        if target.exists():
            with safe_open(str(target), framework="pt", device="cpu") as handle:
                previous = handle.metadata()
                valid = previous == metadata and tuple(handle.get_slice("embedding").get_shape()) == (1, 512, 100352)
            if not valid:
                if (previous.get("scene") != scene or previous.get("revision") != weights["revision"]
                        or previous.get("caption") != captions["scenes"][scene].get("raw_generation")):
                    raise ValueError("Existing text embedding provenance mismatch")
                preserved = target.with_suffix(".before_concise.safetensors")
                if preserved.exists():
                    raise FileExistsError("Preserved pre-migration text embedding already exists")
                target.rename(preserved)
        if not target.exists():
            messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}]},
                        {"role": "user", "content": [{"type": "text", "text": caption}]}]
            conversation = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_vision_id=False)
            ids = processor(text=[conversation], return_tensors="pt")["input_ids"][0].tolist()
            if len(ids) > 512:
                raise ValueError("Caption exceeds released text context")
            tokens = torch.tensor([ids + [processor.tokenizer.pad_token_id]*(512-len(ids))], device="cuda")
            with torch.inference_mode():
                result = model.model(input_ids=tokens, attention_mask=None, output_hidden_states=True, use_cache=False, return_dict=True)
                layers = [(h-h.mean(dim=-1, keepdim=True))/(h.std(dim=-1, keepdim=True)+1e-8) for h in result.hidden_states[1:]]
                embedding = torch.cat(layers, dim=-1).cpu().contiguous()
            if embedding.shape != (1, 512, 100352) or not torch.isfinite(embedding).all():
                raise ValueError("Invalid text embedding")
            temporary = target.with_suffix(".tmp")
            save_file({"embedding": embedding}, str(temporary), metadata=metadata)
            temporary.replace(target)
            del result, layers, embedding, tokens
        report[scene] = dict(path=str(target.relative_to(ROOT)), sha256=digest(target), revision=weights["revision"])
        write_json(output / "text_index.partial.json", report)
        write_json(output / "text_progress.json", dict(completed_scenes=len(report), total_scenes=112, updated_unix=time.time()))
        torch.cuda.empty_cache()
    write_json(output / "text_index.json", report)
    write_json(caption_path, captions)


def validate(output):
    dataset = json.loads((output / "latent_index.json").read_text())
    text_index = json.loads((output / "text_index.json").read_text())
    captions = json.loads((output / "captions.json").read_text())
    for scene, row in dataset["scenes"].items():
        row.update(text=text_index[scene], caption=captions["scenes"][scene]["caption"])
    dataset["scope"] = "100 train / 12 validation scene groups; 1120 real videos; static synthetic cameras; k=1"
    write_json(output / "dataset.json", dataset)
    data = PilotData(output, verify=False)
    report = data.verify()
    for scene, row in data.scenes.items():
        for camera in row["latents"]:
            if not torch.isfinite(data.latent(scene, camera)).all():
                raise ValueError("Nonfinite real latent")
        if not torch.isfinite(data.text(scene)).all():
            raise ValueError("Nonfinite text embedding")
    report.update(train_scene_count=100, validation_scene_count=12,
                  captions="new automatic real-frame captions; unreviewed", dynamic_camera_training=False)
    write_json(output / "verification.json", report)
    (output / "README.md").write_text(
        "# Expanded SynCamVideo training data\n\n100 training and 12 disjoint validation scene groups; ten synchronized videos each. "
        "The original 12/4 split remains fixed; eight new validation groups were selected by a recorded hash rule. "
        "Historical real latent/text caches are reused read-only after provenance checks. New videos are independently "
        "encoded by the verified Wan tokenizer at 81 frames, 768x432. New captions are automatic Cosmos-Reason1-7B "
        "annotations of real cam02 frames 0/40/80, not author captions or human-reviewed labels.\n\n"
        "This is an archive-order subset with static cameras. It does not implement the paper's full data mix, "
        "progressive multi-reference schedule, retrieval, or second stage. Original four validation groups have "
        "already informed development; report them separately from the eight new validation groups.\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "latents", "text", "validate"))
    parser.add_argument("--raw", default="experiments/plenoptic/data/syncam_expanded112")
    parser.add_argument("--output", default="results/multiscene_expanded100_data")
    args = parser.parse_args()
    output, raw = [(ROOT/p).resolve() for p in (args.output, args.raw)]
    output.relative_to(ROOT)
    raw.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    if args.stage == "audit":
        audit(raw, output)
    else:
        {"latents": latents, "text": text, "validate": validate}[args.stage](output)


if __name__ == "__main__":
    main()
