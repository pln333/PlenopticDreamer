"""Explicit, independently checkable math for the local paper-metric evaluation.

These functions do not load generation models or fit camera rotations to GT.
"""
import numpy as np


def frechet_features(real, generated):
    """StyleGAN-V population covariance FVD, using an exact low-rank identity.

    tr(sqrt(Cr Cg)) = nuclear_norm(Xc Yc.T) / sqrt(nr ng).
    Working in sample space avoids unstable 400x400 square roots for 12 videos.
    """
    x, y = np.asarray(real, np.float64), np.asarray(generated, np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("FVD needs at least two finite, equally sized feature vectors per population")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite FVD features")
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    trace_x = np.square(xc).sum() / len(x)
    trace_y = np.square(yc).sum() / len(y)
    cross_trace = np.linalg.svd(xc @ yc.T, compute_uv=False).sum() / np.sqrt(len(x) * len(y))
    score = np.square(mx - my).sum() + trace_x + trace_y - 2 * cross_trace
    if score < -1e-7 * max(trace_x + trace_y, 1):
        raise ArithmeticError(f"Negative squared Frechet distance: {score}")
    return float(max(score, 0))


def homogeneous(extrinsic):
    x = np.asarray(extrinsic, np.float64)
    if x.shape[-2:] == (4, 4):
        return x.copy()
    if x.shape[-2:] != (3, 4):
        raise ValueError("Expected world-to-camera extrinsics")
    out = np.broadcast_to(np.eye(4), (*x.shape[:-2], 4, 4)).copy()
    out[..., :3, :] = x
    return out


def relative_w2c(extrinsics):
    """Two estimated cameras -> target-from-reference; invariant to world gauge."""
    e = homogeneous(extrinsics)
    if e.shape != (2, 4, 4) or not np.isfinite(e).all():
        raise ValueError("Expected two finite camera extrinsics")
    return e[1] @ np.linalg.inv(e[0])


def pose_errors(predicted, ground_truth):
    """One nonnegative scale per video; rotations remain completely unaligned.

    Return both the paper's literal per-video sums and our frame-count-independent
    means. The paper does not release enough code to establish identical reduction
    or scale alignment, so neither is labeled an exact paper reproduction.
    """
    p, g = np.asarray(predicted, np.float64), np.asarray(ground_truth, np.float64)
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (4, 4) or not len(p):
        raise ValueError("Pose arrays must have identical [frames,4,4] shape")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Non-finite pose estimates")
    for rotations in (p[:, :3, :3], g[:, :3, :3]):
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=5e-4):
            raise ValueError("Camera rotation is not orthogonal")
        if not np.allclose(np.linalg.det(rotations), 1, atol=5e-4):
            raise ValueError("Improper camera rotation")
    pt, gt = p[:, :3, 3], g[:, :3, 3]
    denominator = np.square(pt).sum()
    raw_scale = float((pt * gt).sum() / denominator) if denominator > 1e-12 else None
    scale = max(raw_scale, 0.) if raw_scale is not None else 0.
    translation = np.square(scale * pt - gt).sum(1)
    cos_angle = (np.einsum("nij,nij->n", p[:, :3, :3], g[:, :3, :3]) - 1) / 2
    rotation = np.arccos(np.clip(cos_angle, -1, 1))
    return {
        "frames": len(p), "translation_scale": scale,
        "scale_at_nonnegative_boundary": raw_scale is None or raw_scale <= 0,
        "trans_err_mean_m2": float(translation.mean()),
        "trans_err_sum_m2": float(translation.sum()),
        "rot_err_mean_rad": float(rotation.mean()),
        "rot_err_sum_rad": float(rotation.sum()),
        "rot_err_mean_deg": float(np.degrees(rotation.mean())),
        "per_frame_trans_err_m2": translation.tolist(),
        "per_frame_rot_err_rad": rotation.tolist(),
    }


def interval_starts(frame_count, interval=49):
    """Consecutive intervals covering every frame; last interval overlaps if needed."""
    if frame_count < interval or interval < 1:
        raise ValueError("Video shorter than the requested FVD interval")
    starts = list(range(0, frame_count - interval + 1, interval))
    if starts[-1] != frame_count - interval:
        starts.append(frame_count - interval)
    return starts


def roma_counts(certainty, height=432, width=768, threshold=.5):
    """Separate directions: never sum a symmetric Hx2W grid as HxW pixels."""
    c = np.asarray(certainty)
    if c.shape != (height, width * 2) or not np.isfinite(c).all():
        raise ValueError(f"Unexpected RoMa certainty grid {c.shape}")
    forward = int((c[:, :width] >= threshold).sum())
    backward = int((c[:, width:] >= threshold).sum())
    return {"matched_pixels_forward": forward, "matched_pixels_backward": backward,
            "matched_pixels_bidirectional_mean": (forward + backward) / 2}
