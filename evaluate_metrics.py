"""
Evaluation script for computing image quality metrics between target and reconstructed images.

Metrics computed:
    - PSNR (Peak Signal-to-Noise Ratio)  — per slice, aggregated per task
    - SSIM (Structural Similarity Index)  — per slice, aggregated per task
    - LPIPS (Learned Perceptual Image Patch Similarity) — per slice, aggregated per task
    - FID (Frechet Inception Distance) — per task (all slices pooled)
    - FRD (Frechet Radiomics Distance) — per task (all slices pooled)

Directory structure expected:
    results/{experiment_name}/samples_{config}/{task_name}/{group_name}/{patient_id}/{NNNNN_target.png, NNNNN_target_recon.png}

Usage:
    python evaluate_metrics.py \
        --results_dir rests \
        --experiment_name multidomain_PixelDiT-L_feat0.5_RTX3090 \
        --samples_dir samples_nfe2_clip_iter150000
"""

import os
import argparse
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3

# Pixel-level metrics
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Perceptual metric
import lpips

from scipy.linalg import sqrtm
from frd_score import compute_frd


# ============================================================================
# Image loading utilities
# ============================================================================
def load_image_np(path):
    """Load an image file and return it as a numpy array normalized to [0, 1] (grayscale)."""
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float64) / 255.0


def load_image_tensor(path, device="cpu"):
    """Load an image file and return it as a torch tensor for LPIPS ([-1, 1] range, 3-channel)."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize(0.5, 0.5),  # [-1, 1]
    ])
    return transform(img).unsqueeze(0).to(device)


# ============================================================================
# Inception v3 feature extractor for FID computation
# ============================================================================
class InceptionFeatureExtractor:
    """Extract features from Inception v3 for FID computation."""
    
    def __init__(self, device="cpu", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        
        # For FID: Inception v3 without final FC layer -> 2048-dim features
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def get_features(self, image_paths):
        """Extract 2048-dim features from all images."""
        features = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(self.preprocess(img))
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                feat = self.model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)


# ============================================================================
# FID computation
# ============================================================================
def compute_fid(features_real, features_fake):
    """
    Compute Frechet Inception Distance between two sets of features.
    FID = ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2 * sqrt(C_r * C_f))
    """
    mu_r = np.mean(features_real, axis=0)
    mu_f = np.mean(features_fake, axis=0)
    sigma_r = np.cov(features_real, rowvar=False)
    sigma_f = np.cov(features_fake, rowvar=False)
    
    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    
    # Handle numerical instability (imaginary components from sqrtm)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


# IS code removed


# ============================================================================
# Main evaluation pipeline
# ============================================================================
def evaluate(args):
    base_path = os.path.join(args.results_dir, args.experiment_name, args.samples_dir)
    
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Samples directory not found: {base_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Initialize LPIPS model (AlexNet backbone)
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    
    # Initialize Inception feature extractor for FID
    print("[INFO] Loading Inception v3 for FID computation...")
    inception_extractor = InceptionFeatureExtractor(device=device, batch_size=args.batch_size)
    
    # Discover all tasks
    task_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    # Filter tasks if specified
    if args.tasks:
        task_dirs = [t for t in task_dirs if t in args.tasks]
    
    print(f"[INFO] Found {len(task_dirs)} tasks: {task_dirs}")
    
    # Storage for results
    all_slice_records = []  # Per-slice PSNR, SSIM, LPIPS
    all_task_records = []   # Per-task aggregated metrics
    
    for task_name in task_dirs:
        task_path = os.path.join(base_path, task_name)
        group_dirs = sorted([d for d in os.listdir(task_path)
                             if os.path.isdir(os.path.join(task_path, d))])
        
        total_patients = 0
        for group_name in group_dirs:
            group_path = os.path.join(task_path, group_name)
            patient_dirs = sorted([d for d in os.listdir(group_path)
                                   if os.path.isdir(os.path.join(group_path, d))])
            total_patients += len(patient_dirs)
        
        print(f"\n{'='*80}")
        print(f"[TASK] {task_name} — {total_patients} patients across {len(group_dirs)} groups")
        print(f"{'='*80}")
        
        # Collect all target/recon paths for FID/FRD (task-level)
        task_source_paths = []
        task_target_paths = []
        task_recon_paths = []
        task_psnr_list = []
        task_ssim_list = []
        task_lpips_list = []
        
        for group_name in group_dirs:
            group_path = os.path.join(task_path, group_name)
            patient_dirs = sorted([d for d in os.listdir(group_path)
                                   if os.path.isdir(os.path.join(group_path, d))])
            
            for patient_id in tqdm(patient_dirs, desc=f"  {task_name} / {group_name}"):
                patient_path = os.path.join(group_path, patient_id)
                target_files = sorted(glob.glob(os.path.join(patient_path, "*_target.png")))
                
                for target_file in target_files:
                    basename = os.path.basename(target_file)
                    slice_id = basename.replace("_target.png", "")
                    recon_file = os.path.join(patient_path, f"{slice_id}_target_recon.png")
                    
                    if not os.path.exists(recon_file):
                        print(f"  [WARN] Missing recon file: {recon_file}")
                        continue
                    
                    # --- PSNR & SSIM (numpy, grayscale) ---
                    img_target = load_image_np(target_file)
                    img_recon = load_image_np(recon_file)
                    
                    psnr_val = compute_psnr(img_target, img_recon, data_range=1.0)
                    ssim_val = compute_ssim(img_target, img_recon, data_range=1.0)
                    
                    # --- LPIPS (torch, RGB, [-1, 1]) ---
                    tensor_target = load_image_tensor(target_file, device)
                    tensor_recon = load_image_tensor(recon_file, device)
                    with torch.no_grad():
                        lpips_val = lpips_model(tensor_target, tensor_recon).item()
                    
                    # Record slice-level metrics
                    all_slice_records.append({
                        "task": task_name,
                        "group": group_name,
                        "patient": patient_id,
                        "slice_id": slice_id,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "lpips": lpips_val,
                    })
                    
                    task_psnr_list.append(psnr_val)
                    task_ssim_list.append(ssim_val)
                    task_lpips_list.append(lpips_val)
                    task_source_paths.append(os.path.join(patient_path, f"{slice_id}_source.png"))
                    task_target_paths.append(target_file)
                    task_recon_paths.append(recon_file)
        
        # --- Task-level FID & FRD (target vs recon, all slices pooled) ---
        n_slices = len(task_target_paths)
        print(f"  [INFO] Computing task-level FID & FRD over {n_slices} slices...")
        
        if n_slices >= 2:
            feat_target = inception_extractor.get_features(task_target_paths)
            feat_recon = inception_extractor.get_features(task_recon_paths)
            task_fid = compute_fid(feat_target, feat_recon)
            
            print(f"  [INFO] Computing task-level FRD...")
            task_frd = compute_frd([task_target_paths, task_recon_paths], frd_version='v1')
        else:
            task_fid = float("nan")
            task_frd = float("nan")
        
        # Store task-level summary
        all_task_records.append({
            "task": task_name,
            "num_groups": len(group_dirs),
            "num_patients": total_patients,
            "num_slices": n_slices,
            "psnr_mean": np.mean(task_psnr_list),
            "psnr_std": np.std(task_psnr_list),
            "ssim_mean": np.mean(task_ssim_list),
            "ssim_std": np.std(task_ssim_list),
            "lpips_mean": np.mean(task_lpips_list),
            "lpips_std": np.std(task_lpips_list),
            "fid": task_fid,
            "frd": task_frd,
        })
        
        print(f"  [RESULT] PSNR={np.mean(task_psnr_list):.3f}±{np.std(task_psnr_list):.3f}  "
              f"SSIM={np.mean(task_ssim_list):.3f}±{np.std(task_ssim_list):.3f}  "
              f"LPIPS={np.mean(task_lpips_list):.3f}±{np.std(task_lpips_list):.3f}  "
              f"FID={task_fid:.3f}  FRD={task_frd:.3f}")
              
        # --- Qualitative Visual Inspection (Save Grid) ---
        # 1. Fixed slices for Liver & Lung (Please fill in the slice IDs)
        # LIVER_SLICE_IDS = []  # e.g., ["00153", "00154", "00155"]
        # LUNG_SLICE_IDS = []   # e.g., ["00085", "00086", "00087"]
        
        # def save_slice_grid(indices, title, filename, windowing=None):
        #     if not len(indices): return
        #     n_show = len(indices)
        #     fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
        #     if n_show == 1:
        #         axes = np.expand_dims(axes, axis=0)
            
        #     fig.suptitle(f"{title}: {task_name}", fontsize=16, fontweight="bold")
            
        #     for i, idx in enumerate(indices):
        #         src_path = task_source_paths[idx]
        #         tgt_path = task_target_paths[idx]
        #         rec_path = task_recon_paths[idx]
                
        #         # Load images (normalized to [0, 1])
        #         img_src = load_image_np(src_path) if os.path.exists(src_path) else np.zeros((256, 256))
        #         img_tgt = load_image_np(tgt_path) if os.path.exists(tgt_path) else np.zeros((256, 256))
        #         img_rec = load_image_np(rec_path) if os.path.exists(rec_path) else np.zeros((256, 256))
                
        #         kwargs = {"cmap": "gray"}
        #         if windowing == "abdomen":
        #             # Abdomen: WW=350, WL=50  -> HU range [-125, 225]
        #             # Map to [0,1] using (HU + 1024) / 4095
        #             kwargs["vmin"] = (-125 + 1024) / 4095.0
        #             kwargs["vmax"] = (225 + 1024) / 4095.0
        #         elif windowing == "lung":
        #             # Lung: WW=1500, WL=-750 -> HU range [-1500, 0]
        #             # Note: original data clipped at -1024
        #             kwargs["vmin"] = (-1500 + 1024) / 4095.0
        #             kwargs["vmax"] = (0 + 1024) / 4095.0
                
        #         axes[i, 0].imshow(img_src, **kwargs)
        #         axes[i, 0].set_title(f"Source (LDCT)\n{os.path.basename(src_path)}\nPSNR: {task_psnr_list[idx]:.2f}")
        #         axes[i, 0].axis("off")
                
        #         axes[i, 1].imshow(img_tgt, **kwargs)
        #         axes[i, 1].set_title(f"Target (NDCT GT)\n{os.path.basename(tgt_path)}")
        #         axes[i, 1].axis("off")
                
        #         axes[i, 2].imshow(img_rec, **kwargs)
        #         axes[i, 2].set_title(f"Reconstruction\n{os.path.basename(rec_path)}")
        #         axes[i, 2].axis("off")
                
        #     plt.tight_layout()
            
        #     output_dir = os.path.join(base_path, "evaluation_results")
        #     os.makedirs(output_dir, exist_ok=True)
        #     plot_path = os.path.join(output_dir, filename)
        #     plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        #     plt.close(fig)
        #     print(f"  [SAVED] {title} grid: {plot_path}")

        # # Plot Liver and Lung fixed slices
        # liver_indices = [i for i, p in enumerate(task_target_paths) if any(sid in p for sid in LIVER_SLICE_IDS)]
        # lung_indices = [i for i, p in enumerate(task_target_paths) if any(sid in p for sid in LUNG_SLICE_IDS)]
        
        # save_slice_grid(liver_indices, "Liver Slices (Abdomen Window)", f"{task_name}_liver_grid.png", windowing="abdomen")
        # save_slice_grid(lung_indices, "Lung Slices (Lung Window)", f"{task_name}_lung_grid.png", windowing="lung")
        
        # # Plot Top-5 Best and Worst by PSNR
        # if n_slices >= 5:
        #     sorted_indices = np.argsort(task_psnr_list)  # Ascending order (worst first)
        #     worst_5_indices = sorted_indices[:5]
        #     best_5_indices = sorted_indices[-5:][::-1]   # Descending (best first)
            
        #     save_slice_grid(best_5_indices, "Top-5 Best Slices (by PSNR)", f"{task_name}_top5_best_grid.png")
        #     save_slice_grid(worst_5_indices, "Top-5 Worst Slices (by PSNR)", f"{task_name}_top5_worst_grid.png")
    
    # ========================================================================
    # Save results to CSV
    # ========================================================================
    output_dir = os.path.join(base_path, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Slice-level results (per-slice PSNR, SSIM, LPIPS)
    df_slice = pd.DataFrame(all_slice_records)
    slice_csv = os.path.join(output_dir, "metrics_slice_level.csv")
    df_slice.to_csv(slice_csv, index=False)
    print(f"\n[SAVED] Slice-level results: {slice_csv}")
    
    # 2. Task-level summary (mean±std of slice metrics + FID + FRD)
    df_task = pd.DataFrame(all_task_records)
    task_csv = os.path.join(output_dir, "metrics_task_level.csv")
    df_task.to_csv(task_csv, index=False)
    print(f"[SAVED] Task-level results: {task_csv}")
    
    # ========================================================================
    # Print summary table
    # ========================================================================
    print(f"\n{'='*130}")
    print("SUMMARY (Slice-level mean ± std, Task-level FID & FRD)")
    print(f"{'='*130}")
    print(f"{'Task':<35} {'#Slices':>8} {'PSNR':>18} {'SSIM':>18} {'LPIPS':>18} {'FID':>10} {'FRD':>10}")
    print(f"{'-'*130}")
    for rec in all_task_records:
        print(f"{rec['task']:<35} "
              f"{rec['num_slices']:>8d} "
              f"{rec['psnr_mean']:>8.3f}±{rec['psnr_std']:<8.3f} "
              f"{rec['ssim_mean']:>8.3f}±{rec['ssim_std']:<8.3f} "
              f"{rec['lpips_mean']:>8.3f}±{rec['lpips_std']:<8.3f} "
              f"{rec['fid']:>10.3f} "
              f"{rec['frd']:>10.3f}")
    
    print(f"\n[DONE] All results saved to: {output_dir}")
    return df_slice, df_task


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate image quality metrics (PSNR, SSIM, LPIPS, FID, FRD)"
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of the experiment directory")
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Name of the samples directory (e.g., samples_nfe2_clip_iter150000)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for Inception feature extraction")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all)")
    
    args = parser.parse_args()
    evaluate(args)
