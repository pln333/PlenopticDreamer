"""Verified real-video cache and static-camera k=1 conditions for the pilot."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/multiscene_pilot_data"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


class PilotData:
    def __init__(self, directory=DATA, verify=True):
        self.directory = Path(directory).resolve()
        self.directory.relative_to(ROOT)
        self.manifest_path = self.directory / "dataset.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.scenes = self.manifest["scenes"]
        self.train = sorted(s for s, row in self.scenes.items() if row["split"] == "train")
        self.val = sorted(s for s, row in self.scenes.items() if row["split"] == "val")
        counts = self.manifest.get("split_counts", {"train": 12, "val": 4})
        if (len(self.train) != counts["train"] or len(self.val) != counts["val"]
                or not self.train or not self.val or set(self.train) & set(self.val)
                or any(row["split"] not in {"train", "val"} for row in self.scenes.values())):
            raise ValueError("Scene splits disagree with the declared nonempty, disjoint counts")
        protected = set(self.manifest.get("protected_validation_scenes", []))
        if not protected <= set(self.val):
            raise ValueError("A protected validation scene was removed or assigned to training")
        if verify:
            self.verify()

    def verify(self):
        expected_tokenizer = self.manifest["tokenizer_sha256"]
        for scene, row in self.scenes.items():
            if set(row["latents"]) != {f"cam{i:02d}" for i in range(1, 11)}:
                raise ValueError(f"Incomplete views for {scene}")
            camera_sha = digest(ROOT / row["camera_npz"])
            for camera, entry in row["latents"].items():
                path = ROOT / entry["path"]
                if digest(path) != entry["sha256"]:
                    raise ValueError(f"Latent checksum mismatch: {path}")
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                    shape = tuple(handle.get_slice("latent").get_shape())
                expected = {"scene": scene, "camera": camera, "layout": "BCTHW",
                            "source_kind": "real_video_vae_encoding", "tokenizer_sha256": expected_tokenizer,
                            "source_video_sha256": digest(ROOT / row["videos"][camera]),
                            "source_camera_sha256": camera_sha,
                            "preprocessing": json.dumps(row["preprocessing"], sort_keys=True)}
                if shape != (1, 16, 21, 54, 96) or any(metadata.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Invalid real GT provenance: {path}")
            text_entry = row["text"]
            if digest(ROOT / text_entry["path"]) != text_entry["sha256"]:
                raise ValueError(f"Text checksum mismatch: {scene}")
            with safe_open(str(ROOT / text_entry["path"]), framework="pt", device="cpu") as handle:
                if tuple(handle.get_slice("embedding").get_shape()) != (1, 512, 100352):
                    raise ValueError(f"Invalid caption embedding: {scene}")
                metadata = handle.metadata() or {}
                if metadata.get("caption") != row["caption"] or metadata.get("scene") != scene:
                    raise ValueError(f"Caption provenance mismatch: {scene}")
            with np.load(ROOT / row["camera_npz"]) as data:
                poses = data["c2w"]
                if poses.shape != (10, 81, 4, 4) or not np.allclose(poses, poses[:, :1], atol=1e-6):
                    raise ValueError("Pilot supports audited static cameras only")
        return {"status": "verified_real_sources_and_disjoint_scenes", "train_scenes": self.train,
                "val_scenes": self.val, "videos": sum(len(row["latents"]) for row in self.scenes.values()),
                "manifest_sha256": digest(self.manifest_path)}

    @lru_cache(maxsize=24)
    def latent(self, scene, camera):
        return load_file(str(ROOT / self.scenes[scene]["latents"][camera]["path"]))["latent"].float()

    @lru_cache(maxsize=16)
    def text(self, scene):
        return load_file(str(ROOT / self.scenes[scene]["text"]["path"]))["embedding"]

    @lru_cache(maxsize=16)
    def camera_arrays(self, scene):
        with np.load(ROOT / self.scenes[scene]["camera_npz"]) as data:
            return data["c2w"][:, 0].copy(), data["K_processed"].copy()

    def camera(self, scene, reference, target, device):
        from cosmos_transfer2._src.predict2.camera.utils import convert_camera_to_plucker_rays
        poses, intrinsics = self.camera_arrays(scene)
        ids = [int(name[3:]) - 1 for name in (reference, target)]
        local = np.linalg.inv(poses[ids[0]])
        w2c = torch.from_numpy(np.linalg.inv(local[None] @ poses[ids])).float()[None]
        k = torch.from_numpy(intrinsics).float().expand(1, 2, 3, 3)
        # Static poses: compute two ray frames, then expand each to 21 latent times.
        rays = convert_camera_to_plucker_rays(w2c, k, (432, 768), out_dtype=torch.float32)
        return rays.to(device).repeat_interleave(21, dim=1)

    def conditions(self, scene, reference, target, device):
        # Deliberately does not read target pixels or target latent.
        return (self.latent(scene, reference).to(device)[:, None],
                self.camera(scene, reference, target, device), self.text(scene).to(device))

    def training_sample(self, scene, reference, target, device):
        if scene not in self.train or reference == target:
            raise ValueError("Training requires a training scene and distinct cameras")
        reference_z, rays, text = self.conditions(scene, reference, target, device)
        return reference_z, self.latent(scene, target).to(device), rays, text
