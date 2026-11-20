"""SynCamVideo pixel/camera adapter. Does not import Cosmos or load a model."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def opencv_c2w(matrix_string):
    values = [float(x) for x in FLOAT.findall(matrix_string)]
    if len(values) != 16:
        raise ValueError(f"Expected 16 matrix entries, got {len(values)}")
    # Matches the source dataset's ReCamMaster training reader: transpose the
    # UE row-vector transform, reorder forward/right/up, flip the image y axis.
    pose = np.asarray(values, dtype=np.float64).reshape(4, 4).T
    pose = pose[:, [1, 2, 0, 3]].copy()
    pose[:3, 1] *= -1
    pose[:3, 3] /= 100.0  # centimeters -> meters, as in the official reader
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous camera matrix")
    return pose


def scene_cameras(path):
    raw = json.loads(Path(path).read_text())
    frames = sorted(raw, key=lambda name: int(name.removeprefix("frame")))
    if frames != [f"frame{i}" for i in range(len(frames))]:
        raise ValueError("Camera frame indices are not contiguous from zero")
    names = [f"cam{i:02d}" for i in range(1, 11)]
    poses = np.asarray([[opencv_c2w(raw[frame][name]) for frame in frames] for name in names])
    # Canonical world = cam01/frame0's OpenCV coordinates. Relative transforms
    # remove the world/camera handedness mismatch of the original UE basis.
    origin_inverse = np.linalg.inv(poses[0, 0])
    c2w = origin_inverse[None, None] @ poses
    return names, frames, c2w, np.linalg.inv(c2w)


def center_crop_parameters(source_width, source_height, width=768, height=432):
    ratio = width / height
    if source_width / source_height >= ratio:
        crop_height = source_height
        crop_width = int(round(crop_height * ratio))
    else:
        crop_width = source_width
        crop_height = int(round(crop_width / ratio))
    left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2
    sx, sy = width / crop_width, height / crop_height
    affine = np.asarray([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]], dtype=np.float64)
    # SynCamVideo's official dataset card: 24 mm focal, 23.76 mm square sensor.
    original_k = np.asarray([[source_width * 24 / 23.76, 0, source_width / 2],
                             [0, source_height * 24 / 23.76, source_height / 2], [0, 0, 1]])
    config = {"source_width": source_width, "source_height": source_height,
              "crop_left": left, "crop_top": top, "crop_width": crop_width,
              "crop_height": crop_height, "width": width, "height": height}
    return config, original_k, affine, affine @ original_k


def transform_frame(bgr, config):
    if bgr.shape[:2] != (config["source_height"], config["source_width"]):
        raise ValueError("Decoded frame dimensions differ from the camera preprocessing")
    y, x = config["crop_top"], config["crop_left"]
    image = bgr[y:y + config["crop_height"], x:x + config["crop_width"]]
    return cv2.resize(image, (config["width"], config["height"]), interpolation=cv2.INTER_AREA)


def read_rgb_frames(path, config, frame_indices):
    capture = cv2.VideoCapture(str(path))
    output = []
    next_index = 0
    try:
        for index in frame_indices:
            if index != next_index:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError(f"Could not seek frame {index} in {path}")
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index} from {path}")
            next_index = index + 1
            output.append(cv2.cvtColor(transform_frame(frame, config), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(output).transpose(0, 3, 1, 2)  # [F,C,H,W], uint8


class SynCamPilotDataset:
    """A CPU sample reader for data verification, before model-specific adaptation.

    Frame indices are RGB indices, never presumed to be VAE latent indices.
    No captions are fabricated; consumers must handle missing captions explicitly.
    """
    def __init__(self, manifest, num_references=1, frame_indices=None, require_caption=False):
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not 1 <= num_references <= 4:
            raise ValueError("Pilot supports 1 to 4 reference views")
        self.num_references = num_references
        self.frame_indices = frame_indices
        self.require_caption = require_caption

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.require_caption and not row.get("caption"):
            raise ValueError("Caption is missing; this pilot is not ready for caption-conditioned training")
        base = self.manifest.parent
        camera_path = (base / row["camera_npz"]).resolve()
        with np.load(camera_path) as data:
            c2w, k = data["c2w"], data["K_processed"]
        indices = list(range(row["num_frames"])) if self.frame_indices is None else list(self.frame_indices)
        if not indices or any(i < 0 or i >= row["num_frames"] for i in indices):
            raise ValueError("Invalid requested RGB frame indices")
        target = row["target_camera"]
        references = row["reference_candidates"][:self.num_references]
        def load(name):
            return read_rgb_frames(base / row["videos"][name], row["preprocessing"], indices)
        camera_indices = [int(name.removeprefix("cam")) - 1 for name in references + [target]]
        # Re-anchor each sample to its first reference at its first selected frame.
        local = np.linalg.inv(c2w[camera_indices[0], indices[0]])
        selected_c2w = local[None, None] @ c2w[camera_indices][:, indices]
        return {"scene": row["scene"], "reference_cameras": references, "target_camera": target,
                "reference_video_rgb_uint8": np.stack([load(name) for name in references]),
                "target_video_rgb_uint8": load(target), "c2w": selected_c2w,
                "w2c": np.linalg.inv(selected_c2w), "K": k, "frame_indices": indices,
                "caption": row.get("caption"), "fps": row["fps"]}
