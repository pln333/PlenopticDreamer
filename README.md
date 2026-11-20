# PlenopticDreamer reproduction: training code snapshot

Research snapshot for reviewing the implementation and comparing training speed. This is an independent, incomplete reproduction, not the authors' official PlenopticDreamer implementation or weights. The snapshot preserves the experiment scripts as run, including machine-specific paths; it is provided for inspection and is not a portable, turnkey training package.

## Current A continuation

- Entry: `scripts/plenoptic432_a_continue/overnight_camera_abc.py --train A`, launched through `bootstrap.py` with `torch.distributed.run --standalone --nproc_per_node=4`.
- Four data-parallel ranks; microbatch 1 per GPU, 10 sequential accumulation iterations, effective global batch 40. The configuration field `batch_per_rank=10` denotes the accumulated batch, not ten samples in one forward pass.
- One clean reference plus one noisy target; 81 frames at 432 × 768, latent `[16,21,54,96]` per view. Latents are precomputed; text embeddings are computed online. Caption generation can be used when a cached caption is absent.
- BF16 autocast, FP32 optimizer masters, AdamW resumed from A-1000 with learning rate 2e-5; self-attention and camera-encoder parameters train. 20% samples have sigma sampled in [0.98, 0.9999]; text is dropped with 20% probability.
- Training pauses for evaluation after every 100 optimizer updates. A recent example was roughly 65 seconds per optimizer update, excluding separate evaluation; this is not a controlled comparison with another project. Checkpoint loading, hardware, data readiness, and validation can change total wall time.
- The current `load_seconds` log field is misnamed: for accumulation it includes compute from the previous microbatches. Do not interpret it as disk I/O time.

## Code map

- `scripts/plenoptic432_a_continue/overnight_camera_abc.py`: actual optimization loop, gradient accumulation and synchronization.
- `scripts/plenoptic432_a_continue/data.py`, `text.py`: latent/camera loading and text path.
- `scripts/plenoptic/context_layout.py`: reference/target layout, flow input and masked loss.
- `scripts/plenoptic/camera_backbone.py`, `multiview_inference.py`: model loading and sampling.
- `scripts/plenoptic432_a_continue/camera_temporal_adapter.py`: optional B/C camera ablation; A does not use this adapter.
- Evaluation files are included to show the evaluation path, but other dependencies, data manifests, model weights and external metrics repositories are not bundled. Absolute paths and saved protocol hashes must be adapted for a new environment.

## Upstream and provenance

The backbone depends on NVIDIA [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), local upstream revision `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`. It also requires Cosmos model weights and other external packages. This snapshot does not include upstream repository source, datasets, captions, cached latents, checkpoints, environment credentials or generated videos. The upstream Apache 2.0 license is included for reference; third-party dependencies and weights have their own licenses and access terms.

The upstream PlenopticDreamer-specific training and inference components were not fully available to us. Our camera, context, data, and training adaptations must not be assumed identical to the authors' implementation.

## Review configuration

See [速度对照说明](docs/速度对照说明.md) for a comparison checklist and implementation pointers. `configs/a_continue_batch40.json` records the actual continuation configuration with microbatch and accumulation explicitly separated. `configs/source_sha256.json` hashes the included Python snapshot. Historical `train.py`, `train_fast.py` and prefetch/benchmark helpers are included for context; they are not the current continuation entry.
