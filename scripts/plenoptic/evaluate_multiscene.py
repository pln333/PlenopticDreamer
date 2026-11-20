"""Fixed train/held-out scene generation, reference-copy baseline and controls."""
import argparse
import gc
import json
import time

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from skimage.metrics import structural_similarity

from ablate_camera import features, geometry_score
from multiscene_data import DATA, ROOT, PilotData, digest, write_json
from multiview_inference import CameraVideoInference
from reconstruct_tokenizer import write_video
from syncam_data import read_rgb_frames


def cases(data, controls):
    plan = [dict(scene=s, reference="cam02", target="cam06", role="held_out_scene") for s in data.val]
    for scene in ("scene473", "scene1195"):
        plan.append(dict(scene=scene, reference="cam02", target="cam01", role="training_scene"))
    if controls:
        plan += [dict(scene="scene473", reference="cam02", target="cam02", role="identity_control"),
                 dict(scene="scene473", reference="cam01", target="cam02", role="reverse_pair_control")]
    return plan


def pixel_metrics(generated, target):
    mse = ((generated - target) ** 2).mean((1, 2, 3))
    return {"psnr_mean_db": float((-10 * np.log10(np.maximum(mse, 1e-15))).mean()),
            "ssim_mean": float(np.mean([structural_similarity(g, t, channel_axis=-1, data_range=1.) for g, t in zip(generated, target)]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA.relative_to(ROOT)))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT)
    out.mkdir(exist_ok=True)
    if (out / "evaluation.json").exists():
        raise FileExistsError("Completed evaluation already exists")
    data = PilotData(ROOT / args.data)
    plan = cases(data, args.controls)
    write_json(out / "plan.json", {"cases": plan, "sampling_seed": 456, "sampling_steps": args.steps,
               "manifest_sha256": digest(data.manifest_path), "target_pixels_used_for_sampling": False,
               "scope": f"{len(data.val)} held-out scenes, 2 training scenes, one sampling seed; no paper-scale claims"})
    pipeline = CameraVideoInference(ROOT / args.checkpoint)
    checkpoint_sha = pipeline.loading["finetuned_sha256"]
    samples = []
    for case in plan:
        scene, ref, target = case["scene"], case["reference"], case["target"]
        name = f"{scene}_{ref}_{target}"
        destination = out / name
        destination.mkdir(exist_ok=True)
        print(f"sampling {name} role={case['role']}", flush=True)
        reference, rays, text = data.conditions(scene, ref, target, "cuda")
        latent, sampling = pipeline.generate_latent(reference, rays, text, seed=456, steps=args.steps, shift=5)
        save_file({"latent": latent.cpu().contiguous()}, str(destination / "generated_latent.safetensors"),
                  metadata={"source_kind": "generated_prediction_not_ground_truth", "checkpoint_sha256": checkpoint_sha,
                            "scene": scene, "reference": ref, "target": target})
        write_json(destination / "sampling.json", sampling)
        samples.append((case, destination, sampling))
        del reference, rays, text, latent
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    weights = json.loads((ROOT / "results/tokenizer_reconstruction/tokenizer_weight.json").read_text())
    if digest(ROOT / weights["local_path"]) != weights["sha256"]:
        raise ValueError("Tokenizer changed")
    tokenizer = Wan2pt1VAEInterface(vae_pth=str(ROOT / weights["local_path"]), load_mean_std=False, chunk_duration=81, temporal_window=16)
    reports = []
    for case, destination, sampling in samples:
        scene, ref, target = case["scene"], case["reference"], case["target"]
        row = data.scenes[scene]
        latent = load_file(str(destination / "generated_latent.safetensors"))["latent"]
        with torch.inference_mode():
            decoded = tokenizer.decode(latent.cuda().to(torch.bfloat16))
        if decoded.shape != (1, 3, 81, 432, 768) or not torch.isfinite(decoded).all():
            raise ValueError("Invalid decoded generation")
        pixels = ((decoded[0].float().cpu().permute(1, 2, 3, 0).numpy() + 1) / 2).clip(0, 1)
        rgb = np.rint(pixels * 255).astype(np.uint8)
        # First target RGB decoding occurs after all generation has finished.
        gt = read_rgb_frames(ROOT / row["videos"][target], row["preprocessing"], list(range(81))).transpose(0, 2, 3, 1)
        reference = read_rgb_frames(ROOT / row["videos"][ref], row["preprocessing"], list(range(81))).transpose(0, 2, 3, 1)
        gt_float = gt.astype(np.float32) / 255
        score = dict(case, **pixel_metrics(pixels, gt_float), reference_copy=pixel_metrics(reference.astype(np.float32) / 255, gt_float),
                     checkpoint_sha256=checkpoint_sha, target_pixels_used_for_sampling=False,
                     initial_noise_sha256=sampling["initial_noise_sha256"], clean_reference_max_change=sampling["clean_reference_max_change"])
        poses, k = data.camera_arrays(scene)
        local = np.linalg.inv(poses[int(ref[3:]) - 1])
        w2c = np.linalg.inv(local @ poses[int(target[3:]) - 1])
        r, t = w2c[:3, :3], w2c[:3, 3]
        tx, ty, tz = t
        skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        fundamental = None if ref == target else np.linalg.inv(k).T @ skew @ r @ np.linalg.inv(k)
        geometry, real_geometry, tiles = [], [], []
        for frame in (0, 40, 80):
            ref_features = features(reference[frame])
            geometry.append(geometry_score(ref_features, rgb[frame], fundamental) | {"frame": frame})
            real_geometry.append(geometry_score(ref_features, gt[frame], fundamental) | {"frame": frame})
            parts = []
            for label, clip in ((f"reference {ref}", reference), (f"GT {target}", gt), ("generated", rgb)):
                tile = cv2.cvtColor(cv2.resize(clip[frame], (384, 216)), cv2.COLOR_RGB2BGR)
                cv2.putText(tile, f"{scene} {label} f{frame}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 255), 1)
                parts.append(tile)
            tiles.append(np.concatenate(parts, axis=1))
        score.update(generated_geometry=geometry, real_calibration_baseline=real_geometry)
        measured = [x["median_sampson_px"] for x in geometry if x.get("status") == "known_camera_sampson_measured"]
        score["median_sampson_px"] = float(np.median(measured)) if measured else None
        score["geometry_valid_frames"] = len(measured)
        score["psnr_minus_reference_copy_db"] = score["psnr_mean_db"] - score["reference_copy"]["psnr_mean_db"]
        write_video(destination / "generated.mp4", rgb, row["fps"])
        write_video(destination / "reference_gt_generated.mp4", np.concatenate([reference, gt, rgb], axis=2), row["fps"])
        cv2.imwrite(str(destination / "comparison.jpg"), np.concatenate(tiles, axis=0))
        write_json(destination / "metrics.json", score)
        reports.append(score)
        print(f"evaluated {scene} {ref}->{target}: PSNR={score['psnr_mean_db']:.3f}, SSIM={score['ssim_mean']:.4f}, copy_delta={score['psnr_minus_reference_copy_db']:+.3f}", flush=True)
        del decoded, pixels, rgb, reference, gt, gt_float
        tokenizer.clear_cache()
        torch.cuda.empty_cache()
    summary = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha, "status": "completed",
               "cases": reports, "sampling_seed": 456, "steps": args.steps,
               "limitations": [f"One seed and {len(data.val)} held-out scenes; not paper-scale evaluation.",
                               "Known-F SIFT error depends on matching success and is not camera pose estimation.",
                               "Static synthetic cameras only; camera holdout is distinct from scene holdout."]}
    for role in ("training_scene", "held_out_scene"):
        subset = [x for x in reports if x["role"] == role]
        summary[role] = {metric: float(np.mean([x[metric] for x in subset])) for metric in ("psnr_mean_db", "ssim_mean", "psnr_minus_reference_copy_db")}
    historical = set(data.manifest.get("historical_validation_scenes", data.val))
    for label, selected in (("historical_validation", historical), ("new_validation", set(data.val) - historical)):
        subset = [x for x in reports if x["role"] == "held_out_scene" and x["scene"] in selected]
        if subset:
            summary[label] = {metric: float(np.mean([x[metric] for x in subset])) for metric in
                              ("psnr_mean_db", "ssim_mean", "psnr_minus_reference_copy_db")}
            summary[label]["scene_count"] = len(subset)
    baseline = out.parent / "eval_00000/evaluation.json"
    if out.name != "eval_00000" and baseline.exists():
        before = json.loads(baseline.read_text())
        for role in ("training_scene", "held_out_scene"):
            summary[role]["psnr_delta_from_initialization_db"] = summary[role]["psnr_mean_db"] - before[role]["psnr_mean_db"]
            summary[role]["ssim_delta_from_initialization"] = summary[role]["ssim_mean"] - before[role]["ssim_mean"]
    write_json(out / "evaluation.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
