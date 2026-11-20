"""Reconstruct real SynCam clips through the unmodified official tokenizer."""
import argparse
import gc
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import save_file
from skimage.metrics import structural_similarity

from syncam_data import read_rgb_frames

ROOT = Path(__file__).resolve().parents[2]


def write_video(path, rgb, fps):
    frames, height, width, _ = rgb.shape
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in rgb:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--cameras", nargs="+", choices=[f"cam{i:02d}" for i in range(1, 11)],
                        help="Encode only these cameras; outputs stay in --output")
    parser.add_argument("--temporal-window", type=int, default=16,
                        help="RGB frames per encoder chunk; inherited Cosmos training config uses 16")
    parser.add_argument("--output", default="results/tokenizer_reconstruction")
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4 or not 1 <= args.views <= 10:
        parser.error("Use 4n+1 RGB frames and 1..10 views")
    if args.temporal_window < 4 or args.temporal_window % 4:
        parser.error("Encoder temporal window must be a positive multiple of 4")
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "reconstruction.json").exists() or any(out.glob("*_latent.safetensors")):
        raise FileExistsError("Keep completed encodings intact; choose a fresh output directory")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA device")
    torch.cuda.set_device(0)
    from cosmos_transfer2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    weight_info = json.loads((ROOT / "results/tokenizer_reconstruction/tokenizer_weight.json").read_text())
    weight = ROOT / weight_info["local_path"]
    if hashlib.sha256(weight.read_bytes()).hexdigest() != weight_info["sha256"]:
        raise RuntimeError("Tokenizer weight changed after download verification")
    print(f"Loading official tokenizer on {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = Wan2pt1VAEInterface(vae_pth=str(weight), load_mean_std=False, chunk_duration=args.frames,
                                  temporal_window=args.temporal_window)
    row = json.loads((ROOT / "results/plenoptic_data_audit/train.jsonl").read_text().splitlines()[0])
    base = ROOT / "results/plenoptic_data_audit"
    if args.frames > row["num_frames"]:
        raise ValueError("Requested frames exceed source video; no artificial frame padding is allowed")
    indices = list(range(args.frames))
    samples = []
    cameras = args.cameras or sorted(row["videos"])[:args.views]
    if len(set(cameras)) != len(cameras) or any(camera not in row["videos"] for camera in cameras):
        parser.error("Camera list must contain distinct cameras from the selected scene")
    for camera in cameras:
        torch.cuda.reset_peak_memory_stats()
        rgb = read_rgb_frames(base / row["videos"][camera], row["preprocessing"], indices)
        original = rgb.transpose(0, 2, 3, 1).copy()
        x = torch.from_numpy(rgb.copy()).permute(1, 0, 2, 3).unsqueeze(0).cuda().to(torch.bfloat16) / 127.5 - 1
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            latent = tokenizer.encode(x)
            torch.cuda.synchronize()
            encode_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            recon = tokenizer.decode(latent)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - begin
        expected = (1, 16, 1 + (args.frames - 1) // 4, 54, 96)
        if tuple(latent.shape) != expected or recon.shape != x.shape:
            raise RuntimeError(f"Unexpected codec shapes: {latent.shape}, {recon.shape}")
        if not torch.isfinite(latent).all() or not torch.isfinite(recon).all():
            raise RuntimeError("Nonfinite tokenizer output")
        raw_recon = ((recon[0].float().cpu().permute(1, 2, 3, 0).numpy() + 1) / 2)
        reconstruction_float = np.clip(raw_recon, 0, 1)
        original_float = original.astype(np.float32) / 255
        mse = np.mean((reconstruction_float - original_float) ** 2, axis=(1, 2, 3))
        psnr = -10 * np.log10(np.maximum(mse, 1e-15))
        ssim = [float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
                for a, b in zip(original_float, reconstruction_float)]
        reconstruction = np.round(reconstruction_float * 255).astype(np.uint8)
        z = latent.detach().float().cpu()
        source_video = (base / row["videos"][camera]).resolve()
        camera_file = (base / row["camera_npz"]).resolve()
        save_file({"latent": z}, str(out / f"{row['scene']}_{camera}_latent.safetensors"),
                  metadata={"scene": row["scene"], "camera": camera, "tokenizer_sha256": weight_info["sha256"],
                            "layout": "BCTHW", "normalization": "official Wan channel mean/std; load_mean_std=False",
                            "source_kind": "real_video_vae_encoding",
                            "source_video_sha256": hashlib.sha256(source_video.read_bytes()).hexdigest(),
                            "source_camera_sha256": hashlib.sha256(camera_file.read_bytes()).hexdigest(),
                            "preprocessing": json.dumps(row["preprocessing"], sort_keys=True),
                            "rgb_frames": str(args.frames), "encoder_temporal_window": str(args.temporal_window)})
        stem = f"{row['scene']}_{camera}"
        write_video(out / f"{stem}_original.mp4", original, row["fps"])
        write_video(out / f"{stem}_reconstruction.mp4", reconstruction, row["fps"])
        write_video(out / f"{stem}_comparison.mp4", np.concatenate([original, reconstruction], axis=2), row["fps"])
        tiles = []
        for index in sorted(set([0, args.frames // 2, args.frames - 1])):
            left, right = [cv2.resize(image[index], (384, 216)) for image in [original, reconstruction]]
            tile = cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR)
            cv2.putText(tile, f"{camera} frame {index}: original | reconstruction", (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            tiles.append(tile)
        cv2.imwrite(str(out / f"{stem}_comparison.jpg"), np.concatenate(tiles, axis=0))
        sample = {"scene": row["scene"], "camera": camera, "rgb_shape": list(x.shape),
                  "latent_shape": list(latent.shape), "reconstruction_shape": list(recon.shape),
                  "latent_dtype": str(latent.dtype), "latent_mean": float(z.mean()), "latent_std": float(z.std()),
                  "latent_channel_means": z.mean(dim=(0, 2, 3, 4)).tolist(),
                  "latent_channel_stds": z.std(dim=(0, 2, 3, 4)).tolist(),
                  "psnr_mean_db": float(psnr.mean()), "psnr_per_frame_db": psnr.tolist(),
                  "ssim_mean": float(np.mean(ssim)), "ssim_per_frame": ssim,
                  "raw_reconstruction_out_of_range_fraction": float(np.mean((raw_recon < 0) | (raw_recon > 1))),
                  "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                  "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "comparison_video": f"{stem}_comparison.mp4"}
        samples.append(sample)
        print(json.dumps({k: v for k, v in sample.items() if not isinstance(v, list)}), flush=True)
        del x, latent, recon, z, raw_recon, original_float, reconstruction_float
        tokenizer.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
    report = {"status": "passed_reconstruction_only_not_novel_view_generation",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "samples": samples,
              "tokenizer_weight": weight_info,
              "implementation": "unmodified cosmos_transfer2._src.predict2.tokenizers.wan2pt1.Wan2pt1VAEInterface",
              "metric_definition": "PSNR/SSIM on decoded float RGB clamped to [0,1], before MP4 compression",
              "spatial_compression": 8, "temporal_compression": 4, "latent_channels": 16,
              "encoder_temporal_window": args.temporal_window,
              "pixel_to_latent_frames": "1+(F-1)//4; experiment uses F=4n+1",
              "normalization": "RGB/127.5-1; built-in 16-channel Wan latent mean/std; no dataset-estimated extra normalization"}
    (out / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {out / 'reconstruction.json'}", flush=True)


if __name__ == "__main__":
    main()
