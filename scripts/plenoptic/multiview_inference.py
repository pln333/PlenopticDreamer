"""Local camera-video sampling adapter; not the unavailable official wrapper.

Reference latents are always clean, only target latents enter the solver. No
target RGB/latent ground truth is accepted by this inference interface. Supports
the progressive 1..4 reference layouts; current GPU validation uses one.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from camera_backbone import load_camera_backbone, use_wan_mixed_precision
from context_layout import ContextLayout


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inference_backbone(checkpoint, device="cuda"):
    """Load base plus exactly the complete trainable subset, with provenance."""
    checkpoint = Path(checkpoint).resolve()
    model, report = load_camera_backbone()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("base_sha256") != report["base_sha256"]:
        raise ValueError("Adapter was not saved for this base checkpoint")
    manifest_path = checkpoint.parent / "checkpoint_manifest.json"
    digest = sha256_file(checkpoint)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != digest:
            raise ValueError("Finetuned checkpoint differs from its recorded hash")
    state = load_file(str(checkpoint), device="cpu")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(f"Trainable key mismatch: missing={expected - set(state)}, extra={set(state) - expected}")
    current = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in state.items():
            if value.shape != current[key].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapter tensor: {key}")
            # Retain FP32 modulation tensors saved by the training adapter.
            current[key].data = value
    fp32 = use_wan_mixed_precision(model)
    model = model.eval().requires_grad_(False).to(device)
    report.update({"finetuned_checkpoint": str(checkpoint), "finetuned_sha256": digest,
                   "finetuned_tensors_loaded": len(state), "checkpoint_metadata": metadata,
                   "fp32_modules": fp32})
    return model, report


@torch.inference_mode()
def sample_target(predict_velocity, references, camera, text, *, steps=35, shift=5.0, seed=1,
                  conditional_frame_timestep=0.1, progress=None):
    """Sample BCTHW target from N(0,I), using upstream AR UniPC scheduling.

    predict_velocity consumes the packed network kwargs and returns packed
    velocities. No CFG: one conditional prediction per step, equivalent to
    guidance=0 in the released AR formula cond+g*(cond-uncond).
    """
    from cosmos_transfer2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if references.ndim != 6 or references.shape[2] != 16 or steps < 1 or shift <= 0:
        raise ValueError("Expected BKCTHW reference latents with 16 channels, positive steps/shift")
    batch, refs, channels, frames, height, width = references.shape
    if height % 2 or width % 2:
        raise ValueError("Latent H/W must be divisible by DiT patch size 2")
    layout = ContextLayout(refs, frames)
    expected_camera = (batch, (refs + 1) * frames, height // 2, width // 2, 1536)
    if tuple(camera.shape) != expected_camera or tuple(text.shape) != (batch, 512, 100352):
        raise ValueError(f"Condition shape mismatch: camera={camera.shape}, text={text.shape}")
    if any(t.device != references.device for t in [camera, text]):
        raise ValueError("Conditions and references must use the same device")
    if not all(torch.isfinite(t).all() for t in [references, camera, text]):
        raise ValueError("Nonfinite input conditions")
    references = references.detach().float().clone()
    initial_references = references.clone()
    generator = torch.Generator(device=references.device).manual_seed(seed)
    target = torch.randn((batch, channels, frames, height, width), generator=generator,
                         device=references.device, dtype=torch.float32)
    initial_noise_sha = hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    # Matches the released AR call. Generic parent inference optionally selects
    # Karras sigmas; AR uses shifted linear sigmas, with no Karras option passed.
    scheduler.set_timesteps(steps, device=references.device, shift=shift)
    clean = layout.clean_mask(batch, references.device)
    mask = clean.expand(-1, -1, -1, height, width).contiguous()
    time_mask = clean[:, 0, :, 0, 0]
    padding = torch.zeros(batch, 1, height, width, device=references.device)
    history = []
    for index, timestep in enumerate(scheduler.timesteps):
        begin = time.perf_counter()
        packed = layout.pack_latents(references, target)
        model_times = time_mask * conditional_frame_timestep + (1 - time_mask) * timestep.float()
        prediction = predict_velocity(x_B_C_T_H_W=packed, timesteps_B_T=model_times,
                                      crossattn_emb=text, condition_video_input_mask_B_C_T_H_W=mask,
                                      padding_mask=padding, camera=camera)
        if prediction.shape != packed.shape or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Invalid network prediction at step {index}")
        velocity = prediction[:, :, layout.target_slice].float()
        target = scheduler.step(velocity, timestep, target, return_dict=False, generator=generator)[0]
        if not torch.isfinite(target).all():
            raise RuntimeError(f"Nonfinite sampled target at step {index}")
        entry = {"step": index + 1, "timestep": int(timestep.item()),
                 "sigma": float(scheduler.sigmas[index]), "latent_mean": target.mean().item(),
                 "latent_std": target.std().item(), "velocity_rms": velocity.square().mean().sqrt().item(),
                 "seconds": time.perf_counter() - begin}
        history.append(entry)
        if progress is not None:
            progress(entry)
    if not torch.equal(references, initial_references):
        raise RuntimeError("Reference latents changed during sampling")
    return target, {"sampler": "unmodified upstream FlowUniPCMultistepScheduler, order=2",
                    "schedule": "AR shifted linear sigmas; no Karras", "shift": shift, "steps": steps,
                    "initialization": "standard Gaussian, target ground truth never supplied",
                    "initial_noise_sha256": initial_noise_sha, "seed": seed,
                    "guidance": "conditional only; AR guidance=0; no unconditional branch",
                    "clean_reference_max_change": 0.0, "reference_count": refs,
                    "packed_order": layout.order, "target_slot": layout.target_slot,
                    "history": history, "final_sigma": float(scheduler.sigmas[-1]),
                    "initial_sigma": float(scheduler.sigmas[0]),
                    "conditional_frame_timestep": conditional_frame_timestep}


class CameraVideoInference:
    def __init__(self, checkpoint, device="cuda"):
        self.model, self.loading = load_inference_backbone(checkpoint, device=device)

    def predict_velocity(self, **kwargs):
        kwargs["x_B_C_T_H_W"] = kwargs["x_B_C_T_H_W"].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs).float()

    def generate_latent(self, references, camera, text, **kwargs):
        return sample_target(self.predict_velocity, references, camera, text, **kwargs)
