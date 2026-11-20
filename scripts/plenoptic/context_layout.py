"""Variable-context layout and flow inputs for reconstruction of the training pipeline.

These helpers are not yet integrated into the upstream AR model. K=4 preserves
its [ref0, ref1, target, ref2, ref3] order; smaller K enables progressive training.
The caller supplies sigma after the chosen scheduler's time sampling/shift.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContextLayout:
    num_references: int
    frames_per_view: int

    def __post_init__(self):
        if not 1 <= self.num_references <= 4 or self.frames_per_view < 1:
            raise ValueError("Expected 1..4 references and positive latent frames per view")

    @property
    def target_slot(self):
        return min(2, self.num_references)

    @property
    def order(self):
        """Indices into source-order [ref0,...,refK-1,target]."""
        refs = list(range(self.num_references))
        return refs[:self.target_slot] + [self.num_references] + refs[self.target_slot:]

    @property
    def target_slice(self):
        begin = self.target_slot * self.frames_per_view
        return slice(begin, begin + self.frames_per_view)

    def pack_latents(self, references, target):
        """[B,K,C,F,H,W] and [B,C,F,H,W] -> concatenated [B,C,(K+1)F,H,W]."""
        if references.ndim != 6 or target.ndim != 5:
            raise ValueError("Invalid latent ranks")
        if references.shape[1] != self.num_references or references.shape[3] != self.frames_per_view:
            raise ValueError("Reference shape disagrees with layout")
        if references.shape[:1] + references.shape[2:] != target.shape:
            raise ValueError("Reference and target latent shapes differ")
        if references.dtype != target.dtype or references.device != target.device:
            raise ValueError("Reference and target dtype/device must agree")
        chunks = list(references.unbind(1)) + [target]
        return torch.cat([chunks[i] for i in self.order], dim=2)

    def pack_cameras(self, cameras):
        """[B,K+1,F,...], source order -> [B,(K+1)F,...], model order."""
        if cameras.shape[1:3] != (self.num_references + 1, self.frames_per_view):
            raise ValueError("Camera view/frame axes disagree with latent layout")
        return cameras[:, self.order].flatten(1, 2)

    def clean_mask(self, batch, device, dtype=torch.float32, target_prefix_frames=0):
        if not 0 <= target_prefix_frames < self.frames_per_view:
            raise ValueError("Target prefix must leave at least one predicted frame")
        mask = torch.ones(batch, 1, (self.num_references + 1) * self.frames_per_view, 1, 1,
                          device=device, dtype=dtype)
        mask[:, :, self.target_slice] = 0
        begin = self.target_slice.start
        mask[:, :, begin:begin + target_prefix_frames] = 1
        return mask


def make_flow_inputs(references, target, sigma, epsilon, *, target_prefix_frames=0):
    """Make clean conditioning + noisy target; no scheduler assumptions beyond sigma in [0,1]."""
    layout = ContextLayout(references.shape[1], target.shape[2])
    x0 = layout.pack_latents(references, target)
    if epsilon.shape != x0.shape:
        raise ValueError("epsilon must have the full packed shape")
    sigma = torch.as_tensor(sigma, device=x0.device, dtype=x0.dtype)
    if sigma.shape != (x0.shape[0],) or not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
        raise ValueError("sigma must contain one finite value in [0,1] per batch item")
    sigma = sigma[:, None, None, None, None]
    mask = layout.clean_mask(x0.shape[0], x0.device, x0.dtype, target_prefix_frames)
    noisy = (1 - sigma) * x0 + sigma * epsilon
    return {"layout": layout, "clean_mask": mask,
            "model_input": mask * x0 + (1 - mask) * noisy,
            "target_velocity": epsilon - x0}


def flow_loss(prediction, target_velocity, clean_mask, time_weights=None):
    """Preserve upstream full-packed-tensor mean (clean regions contribute zero).

    Loss scale depends on context count, as in upstream. Do not silently switch
    to a target-only denominator when comparing stage learning curves.
    """
    if prediction.shape != target_velocity.shape or prediction.ndim != 5:
        raise ValueError("Prediction and velocity must be matching BCTHW tensors")
    per_instance = ((prediction.float() - target_velocity.float()).square() * (1 - clean_mask)).mean((1, 2, 3, 4))
    if time_weights is not None:
        if time_weights.shape != per_instance.shape:
            raise ValueError("Expected one time weight per batch item")
        per_instance = per_instance * time_weights
    return per_instance.mean()
