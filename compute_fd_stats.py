"""Compute FD-Loss reference statistics (mu, sigma) from target-domain CT images.

Supports:
    1. Multi-Target Mode (default):
       Scans the training split and computes reference statistics (mu, sigma)
       for all 5 active target domains in a single pass:
           - Full_B30f
           - Full_B45f
           - Full_B60f
           - Quarter_B30f
           - Quarter_B45f
       Saves all statistics to a single .npz file (e.g. data/fd_stats/target_5domains_dinov2b14.npz).

    2. Single-Domain Mode:
       If --target-dose or --target-kernel is explicitly given, computes statistics
       for only that specific domain and saves standard 'mu' and 'sigma' keys.

Supports single-GPU and multi-GPU (torchrun) execution.

Usage (Multi-target, recommended):
    # Single GPU
    python compute_fd_stats.py \\
        --dataset-dir /path/to/dataset \\
        --split       train \\
        --dino-model  dinov2_vitb14 \\
        --output      data/fd_stats/target_5domains_dinov2b14.npz

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=2 compute_fd_stats.py \\
        --dataset-dir /path/to/dataset \\
        --split       train \\
        --dino-model  dinov2_vitb14 \\
        --output      data/fd_stats/target_5domains_dinov2b14.npz
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import pydicom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fd_stats")

# Kernel alias map (same as dataset.py)
_KERNEL_ALIASES: dict[str, list[str]] = {
    "B30f": ["Br36d"],
    "B45f": ["Br49d"],
    "B60f": ["Br59d"],
}

# The 5 standard target domains across the 9 multi-domain CT conversion tasks
DEFAULT_TARGET_DOMAINS = [
    ("Full", "B30f"),
    ("Full", "B45f"),
    ("Full", "B60f"),
    ("Quarter", "B30f"),
    ("Quarter", "B45f"),
]


# ---------------------------------------------------------------------------
# Dataset: Multi-target CT slice collection
# ---------------------------------------------------------------------------

class MultiTargetSliceDataset(Dataset):
    """Dataset of target-domain DICOM CT slices for multiple target domains.

    Collects slices for the specified target domains and yields
    (image_tensor, domain_idx).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        target_domains: list[tuple[str, str]] | None = None,
        image_size: int = 512,
    ):
        self.image_size = image_size
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        if target_domains is None:
            target_domains = list(DEFAULT_TARGET_DOMAINS)

        self.domain_keys = [f"{d}_{k}" for d, k in target_domains]
        self.domain_to_id = {k: i for i, k in enumerate(self.domain_keys)}

        self.samples: list[tuple[str, int]] = []
        domain_counts: dict[str, int] = {}

        for dose, kernel in target_domains:
            key = f"{dose}_{kernel}"
            dom_id = self.domain_to_id[key]
            valid_kernels = {kernel} | set(_KERNEL_ALIASES.get(kernel, []))
            dcm_files: list[str] = []

            for root, dirs, fnames in os.walk(split_dir):
                root_path = Path(root)
                parts = root_path.parts
                if dose in parts and root_path.name in valid_kernels:
                    for fname in fnames:
                        if fname.lower().endswith(".dcm"):
                            dcm_files.append(os.path.join(root, fname))

            dcm_files = sorted(list(set(dcm_files)))
            domain_counts[key] = len(dcm_files)
            for fpath in dcm_files:
                self.samples.append((fpath, dom_id))

        logger.info(f"Scanning {split_dir} for {len(self.domain_keys)} target domains:")
        for key in self.domain_keys:
            logger.info(f"  [{key:12s}] : {domain_counts.get(key, 0):,} slices")
        logger.info(f"Total slices across all target domains: {len(self.samples):,}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No DICOM files found in {split_dir} matching target domains!")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fpath, dom_id = self.samples[idx]
        try:
            dcm = pydicom.dcmread(fpath, force=True)
            img = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            img = img * slope + intercept
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}. Returning zeros.")
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        img = np.clip(img, -1024.0, 3071.0)
        img = (img + 1024.0) / (3071.0 + 1024.0)  # → [0, 1]
        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor, dom_id


# ---------------------------------------------------------------------------
# Dataset: Single-domain CT slice collection (backward-compatible)
# ---------------------------------------------------------------------------

class SingleTargetSliceDataset(Dataset):
    """Dataset of CT slices for a single specified target domain."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        target_dose: str | None = None,
        target_kernel: str | None = None,
        image_size: int = 512,
    ):
        self.image_size = image_size
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        logger.info(
            f"Scanning {split_dir} (dose={target_dose or 'ALL'}, kernel={target_kernel or 'ALL'}) ..."
        )
        self.files = self._find_dcm_files(split_dir, target_dose, target_kernel)
        if len(self.files) == 0:
            raise RuntimeError(
                f"No DICOM files found in {split_dir} matching dose={target_dose}, kernel={target_kernel}."
            )
        logger.info(f"Found {len(self.files):,} DICOM files.")

    @staticmethod
    def _find_dcm_files(
        split_dir: Path,
        target_dose: str | None,
        target_kernel: str | None,
    ) -> list[str]:
        files: list[str] = []
        for root, dirs, fnames in os.walk(split_dir):
            root_path = Path(root)
            parts = root_path.parts
            if target_dose is not None and target_dose not in parts:
                continue
            if target_kernel is not None:
                valid_kernels = {target_kernel} | set(_KERNEL_ALIASES.get(target_kernel, []))
                if root_path.name not in valid_kernels:
                    continue
            for fname in fnames:
                if fname.lower().endswith(".dcm"):
                    files.append(os.path.join(root, fname))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fpath = self.files[idx]
        try:
            dcm = pydicom.dcmread(fpath, force=True)
            img = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            img = img * slope + intercept
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}. Returning zeros.")
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        img = np.clip(img, -1024.0, 3071.0)
        img = (img + 1024.0) / (3071.0 + 1024.0)  # → [0, 1]
        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor


# ---------------------------------------------------------------------------
# DINOv2 CLS token extractor
# ---------------------------------------------------------------------------

class DINOv2CLSExtractor(torch.nn.Module):
    """Frozen DINOv2 encoder that returns CLS-token features."""

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_name: str = "dinov2_vitb14", base_patch_size: int = 16):
        super().__init__()
        logger.info(f"Loading {model_name} from torch.hub ...")
        try:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", model_name, trust_repo=True
            )
        except Exception as e:
            logger.warning(f"Failed ({e}). Forcing reload ...")
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", model_name,
                trust_repo=True, force_reload=True,
            )
        self.encoder.eval().requires_grad_(False)
        self.base_patch_size = base_patch_size

        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.feat_dim: int = {
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }.get(model_name, 768)
        logger.info(f"DINOv2 loaded: feat_dim={self.feat_dim}")

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        b, c, h, w = x.shape
        target_h = int(14 * h / self.base_patch_size)
        target_w = int(14 * w / self.base_patch_size)
        x = F.interpolate(x, (target_h, target_w), mode="bicubic", align_corners=False)
        x_norm = (x - self.mean.to(x)) / self.std.to(x)
        feats = self.encoder.forward_features(x_norm)
        return feats["x_norm_clstoken"].float()


# ---------------------------------------------------------------------------
# Distributed setup helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed: rank={rank}/{world_size}, local_rank={local_rank}")
    else:
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)
    return rank, world_size


def teardown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Feature extraction loops
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_stats_single(
    extractor: DINOv2CLSExtractor,
    loader: DataLoader,
    feat_dim: int,
    rank: int,
    world_size: int,
    max_images: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    device = next(extractor.parameters()).device
    feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    count = 0

    for batch in loader:
        if max_images is not None and count >= max_images:
            break
        imgs = batch.to(device, non_blocking=True)
        cls = extractor(imgs).double()
        feat_sum.add_(cls.sum(0))
        feat_outer.addmm_(cls.T, cls)
        count += cls.shape[0]
        if rank == 0 and count % 5000 < loader.batch_size:
            logger.info(f"  Processed {count:,} slices ...")

    if world_size > 1:
        dist.reduce(feat_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(feat_outer, dst=0, op=dist.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        count = int(count_t.item())

    if rank == 0:
        s_np = feat_sum.cpu().numpy()
        S_np = feat_outer.cpu().numpy()
        mu = s_np / count
        sigma = (S_np - np.outer(s_np, s_np) / count) / (count - 1)
        return mu, sigma, count

    return None, None, count


@torch.inference_mode()
def extract_stats_multi(
    extractor: DINOv2CLSExtractor,
    loader: DataLoader,
    num_domains: int,
    feat_dim: int,
    rank: int,
    world_size: int,
    max_images: int | None = None,
) -> tuple[dict[int, np.ndarray] | None, dict[int, np.ndarray] | None, dict[int, int]]:
    """Accumulate statistics partitioned by domain index."""
    device = next(extractor.parameters()).device
    feat_sums = {
        i: torch.zeros(feat_dim, dtype=torch.float64, device=device)
        for i in range(num_domains)
    }
    feat_outers = {
        i: torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
        for i in range(num_domains)
    }
    counts = {i: 0 for i in range(num_domains)}
    total_processed = 0

    for imgs, dom_ids in loader:
        if max_images is not None and total_processed >= max_images:
            break
        imgs = imgs.to(device, non_blocking=True)
        dom_ids = dom_ids.to(device, non_blocking=True)
        cls = extractor(imgs).double()

        for i in range(num_domains):
            mask = (dom_ids == i)
            if mask.any():
                sub_cls = cls[mask]
                feat_sums[i].add_(sub_cls.sum(0))
                feat_outers[i].addmm_(sub_cls.T, sub_cls)
                counts[i] += int(sub_cls.shape[0])

        total_processed += cls.shape[0]
        if rank == 0 and total_processed % 5000 < loader.batch_size:
            logger.info(f"  Processed {total_processed:,} slices ...")

    if world_size > 1:
        for i in range(num_domains):
            dist.reduce(feat_sums[i], dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(feat_outers[i], dst=0, op=dist.ReduceOp.SUM)
            c_tensor = torch.tensor([counts[i]], dtype=torch.long, device=device)
            dist.reduce(c_tensor, dst=0, op=dist.ReduceOp.SUM)
            counts[i] = int(c_tensor.item())

    if rank == 0:
        mu_dict: dict[int, np.ndarray] = {}
        sigma_dict: dict[int, np.ndarray] = {}
        for i in range(num_domains):
            cnt = counts[i]
            if cnt > 1:
                s_np = feat_sums[i].cpu().numpy()
                S_np = feat_outers[i].cpu().numpy()
                mu_dict[i] = s_np / cnt
                sigma_dict[i] = (S_np - np.outer(s_np, s_np) / cnt) / (cnt - 1)
            else:
                logger.warning(f"Domain {i} had insufficient slices ({cnt}).")
                mu_dict[i] = np.zeros(feat_dim, dtype=np.float64)
                sigma_dict[i] = np.eye(feat_dim, dtype=np.float64) * 1e-4
        return mu_dict, sigma_dict, counts

    return None, None, counts


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute FD-Loss reference statistics from target-domain CT images."
    )
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Root dataset directory (contains train/, valid/, test/).")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to use (default: train).")
    p.add_argument("--target-dose", type=str, default=None,
                   help="Filter by dose attribute (e.g. 'Full'). If None and target-kernel is None, multi-target mode is used.")
    p.add_argument("--target-kernel", type=str, default=None,
                   help="Filter by kernel attribute (e.g. 'B30f'). If None and target-dose is None, multi-target mode is used.")
    p.add_argument("--image-size", type=int, default=512,
                   help="CT image spatial size (default: 512).")
    p.add_argument("--dino-model", type=str, default="dinov2_vitb14",
                   choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                   help="DINOv2 model variant (default: dinov2_vitb14).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size per GPU (default: 64).")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader worker processes (default: 8).")
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional cap on total images processed.")
    p.add_argument("--output", type=str,
                   default="data/fd_stats/target_5domains_dinov2b14.npz",
                   help="Output .npz path.")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size = setup_distributed()

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    is_multi_target = (args.target_dose is None and args.target_kernel is None)

    device = torch.device("cuda")
    extractor = DINOv2CLSExtractor(
        model_name=args.dino_model,
        base_patch_size=16,
    ).to(device)

    max_per_rank = (
        (args.max_images + world_size - 1) // world_size
        if args.max_images is not None else None
    )

    if is_multi_target:
        dataset = MultiTargetSliceDataset(
            dataset_dir=args.dataset_dir,
            split=args.split,
            target_domains=DEFAULT_TARGET_DOMAINS,
            image_size=args.image_size,
        )
        sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        logger.info(
            f"Extracting multi-target features ({len(dataset.domain_keys)} domains): "
            f"{len(dataset):,} slices total, {world_size} GPU(s), batch_size={args.batch_size} ..."
        )

        mu_dict, sigma_dict, counts = extract_stats_multi(
            extractor, loader,
            num_domains=len(dataset.domain_keys),
            feat_dim=extractor.feat_dim,
            rank=rank,
            world_size=world_size,
            max_images=max_per_rank,
        )

        if rank == 0 and mu_dict is not None and sigma_dict is not None:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_dict: dict[str, np.ndarray] = {
                "target_domain_keys": np.array(dataset.domain_keys),
            }
            for i, key in enumerate(dataset.domain_keys):
                save_dict[f"mu_{key}"] = mu_dict[i]
                save_dict[f"sigma_{key}"] = sigma_dict[i]

            np.savez(str(out_path), **save_dict)
            logger.info(
                f"\n=======================================================\n"
                f"Successfully saved multi-target reference stats → {out_path}\n"
                f"Model: {args.dino_model} (feat_dim={extractor.feat_dim})\n"
                f"Domains computed:"
            )
            for i, key in enumerate(dataset.domain_keys):
                logger.info(f"  [{key:12s}] : {counts[i]:,} slices")
            logger.info("=======================================================")

    else:
        dataset = SingleTargetSliceDataset(
            dataset_dir=args.dataset_dir,
            split=args.split,
            target_dose=args.target_dose,
            target_kernel=args.target_kernel,
            image_size=args.image_size,
        )
        sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        logger.info(
            f"Extracting single-target features: {len(dataset):,} slices total, "
            f"{world_size} GPU(s), batch_size={args.batch_size} ..."
        )

        mu, sigma, count = extract_stats_single(
            extractor, loader,
            feat_dim=extractor.feat_dim,
            rank=rank,
            world_size=world_size,
            max_images=max_per_rank,
        )

        if rank == 0 and mu is not None and sigma is not None:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(out_path), mu=mu, sigma=sigma)
            logger.info(
                f"Saved single-domain reference stats → {out_path}\n"
                f"  n={count:,}, feat_dim={mu.shape[0]}, model={args.dino_model}, "
                f"dose={args.target_dose or 'ALL'}, kernel={args.target_kernel or 'ALL'}"
            )

    teardown_distributed()


if __name__ == "__main__":
    main()
