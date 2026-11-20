"""Build released Camera DiT and strictly audit the permitted base-weight differences."""
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def use_wan_mixed_precision(model):
    """Match explicit FP32 regions in the released network under BF16 autocast."""
    modules = []
    for name, module in model.named_modules():
        if name in {"t_embedder", "t_embedding_norm", "final_layer"} or ".adaln_modulation_" in name:
            module.float()
            modules.append(name)
    return modules


def load_camera_backbone(device="cpu"):
    from cosmos_transfer2._src.predict2.camera.networks.dit_multiview_camera_ar import (
        CameraARMiniTrainDITwithConditionalMask, SACConfig,
    )
    record = json.loads((ROOT / "results/backbone_preflight/backbone_weight.json").read_text())
    state = torch.load(ROOT / record["local_path"], map_location="cpu", weights_only=True, mmap=True)
    if not all(k.startswith("net.") for k in state):
        raise ValueError("Unexpected non-network checkpoint entries")
    state = {k.removeprefix("net."): v for k, v in state.items()}
    # TE runtime metadata is not a learned parameter. Leave local TE runtime
    # state at its installed version rather than transplant a serialized blob.
    runtime_state = [k for k in state if k.endswith("._extra_state")]
    state = {k: v for k, v in state.items() if k not in runtime_state}
    kwargs = dict(max_img_h=128, max_img_w=128, max_frames=128,
                  in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                  model_channels=2048, num_blocks=28, num_heads=16,
                  concat_padding_mask=True, pos_emb_cls="rope3d", pos_emb_learnable=True,
                  use_adaln_lora=True, adaln_lora_dim=256, atten_backend="minimal_a2a",
                  extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
                  rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
                  rope_t_extrapolation_ratio=1.0, use_crossattn_projection=True,
                  crossattn_proj_in_channels=100352, crossattn_emb_channels=1024,
                  use_wan_fp32_strategy=True, sac_config=SACConfig(mode="none"), timestep_scale=0.001)
    # Meta construction avoids allocating/initializing an extra 8 GB of random
    # backbone weights. Missing camera parameters are explicitly initialized.
    with torch.device("meta"):
        model = CameraARMiniTrainDITwithConditionalMask(**kwargs)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [{"key": k, "model": list(expected[k].shape), "checkpoint": list(state[k].shape)}
                  for k in state.keys() & expected.keys()
                  if isinstance(expected[k], torch.Tensor) and expected[k].shape != state[k].shape]
    allowed_camera = {f"blocks.{i}.cam_encoder.weight" for i in range(28)}
    illegal_missing = [k for k in missing if k not in allowed_camera and not k.endswith("._extra_state")]
    if illegal_missing or unexpected or mismatched:
        raise RuntimeError(f"Unapproved base-weight differences: missing={illegal_missing}, unexpected={unexpected}, shapes={mismatched}")
    camera_parameters = []
    for key in sorted(allowed_camera):
        if key in state:
            raise ValueError("Base checkpoint unexpectedly already contains camera finetuning weights")
        value = torch.empty(expected[key].shape, dtype=torch.bfloat16)
        std = 1 / math.sqrt(2048)
        torch.nn.init.trunc_normal_(value, std=std, a=-3 * std, b=3 * std)
        state[key] = value
        camera_parameters.append(key)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys or any(not k.endswith("._extra_state") for k in result.missing_keys):
        raise RuntimeError(f"Unexpected load_state_dict result: {result}")
    # Keep geometric position buffers in float32, as in their constructor.
    model.pos_embedder.seq = torch.arange(128, dtype=torch.float32)
    model.pos_embedder.dim_spatial_range = model.pos_embedder.dim_spatial_range.float()
    model.pos_embedder.dim_temporal_range = model.pos_embedder.dim_temporal_range.float()
    model = model.to(device)
    for name, parameter in model.named_parameters():
        # Upstream uses substring matching on module names: this includes
        # layer_norm_self_attn and adaln_modulation_self_attn as well as attention.
        should_train = "self_attn" in name or "cam_encoder" in name
        if parameter.requires_grad != should_train:
            raise RuntimeError(f"Parameter freeze rule mismatch: {name}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Uninitialized meta parameter remains")
    report = {"base_sha256": record["sha256"], "checkpoint_tensors_loaded": len(state) - len(camera_parameters),
              "new_camera_parameter_keys": camera_parameters,
              "te_runtime_metadata_not_transplanted": runtime_state,
              "unapproved_missing_keys": illegal_missing, "unexpected_keys": unexpected,
              "shape_mismatches": mismatched,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "trainable_rule": "upstream substring match: self_attn or cam_encoder (including self-attention norms/modulation)",
              "trainable_parameter_keys": [n for n, p in model.named_parameters() if p.requires_grad],
              "model_kwargs": kwargs | {"sac_config": "none"},
              "position_buffer_policy": "seq derived from base length 128, range buffers float32",
              "status": "base_loaded_with_new_untrained_camera_layers"}
    return model, report


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model, report = load_camera_backbone()
    output = ROOT / "results/backbone_preflight/load_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
