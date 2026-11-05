# I2SB + PixelDiT — Multi-Domain CT Restoration

> **Universal LDCT-to-NDCT restoration** using Image-to-Image Schrödinger Bridge (I2SB) with a dual-level PixelDiT backbone, noise-adaptive multi-domain conditioning, and Target-Conditioned Representation Fréchet Loss (FD-Loss).

---

## Overview

This repository implements a universal framework for low-dose CT (LDCT) restoration that performs **kernel conversion** and **denoising** simultaneously within a single unified network. Instead of training separate models per conversion pair, a single model is conditioned on source and target domain attributes (dose level + reconstruction kernel) and input noise statistics, enabling flexible many-to-many domain conversion.

### Key Components

| Component | Description |
|---|---|
| **I2SB** | Image-to-Image Schrödinger Bridge — probabilistic generative model for paired, non-Gaussian boundary image translation |
| **PixelDiT** | Dual-level DiT backbone: patch-level context stream (`AugmentedDiTBlock`) + pixel-level detail stream (`PiTBlock`) |
| **NoiseAdaptiveMultiDomainEmbedder** | Learns conditioning embeddings for dose & kernel attributes with cross-domain interactions and Haar-wavelet input noise statistics |
| **Target-Conditioned FD-Loss** | Differentiable Representation Fréchet Loss in DINOv2 feature space partitioned across 5 distinct target domains, aligning distribution without compromising PSNR/SSIM |

---

## Architecture

```
Input: x_t (noisy state) + x1 (source CT image, channel-concatenated)
         │
         ├─ PixelTokenEmbedder  → pixel-level feature stream
         │      │
         │      └─ pixel_depth × PiTBlock  (high-frequency & edge detail)
         │
         ├─ PatchTokenEmbedder  → patch-level token stream
         │      │
         │      └─ patch_depth × AugmentedDiTBlock  (long-range context reasoning)
         │              │
         │              └─ Conditioning via AdaLN-Zero:
         │                   TimestepConditioner(t) + NoiseAdaptiveMultiDomainEmbedder(domain, noise)
         │
         └─ FinalLayer → output prediction (denoised direction)
```

### Multi-Domain Conditioning Modes (`--embedding-mode`)

| Mode | Description |
|---|---|
| `base` | Base embeddings: `src_dose` + `src_kernel` + `trg_dose` + `trg_kernel` (4 lookup tables, element-wise sum) |
| `base_cross` | `base` + joint cross-domain interaction embeddings for `(src_dose, src_kernel)` and `(trg_dose, trg_kernel)` pairs, fused via MLP |
| `base_cross_noise` | `base_cross` + 2D Haar wavelet noise statistics vector (20-dim) capturing subband spatial variances, fused via MLP |

### PixelDiT Model Sizes

| Size | `patch_depth` | `pixel_depth` | `hidden_size` | `num_groups` | `pixel_hidden_size` | Active in Experiments |
|---|---|---|---|---|---|---|
| Base | 12 | 2 | 768 | 12 | 16 | Phase 1 baseline |
| Large | 22 | 4 | 1024 | 16 | 16 | **Default (Step 1–4)** |
| XLarge | 26 | 4 | 1152 | 16 | 16 | Scaled configuration |

---

## Domain Attributes & Tasks

```
Dose   : Quarter (0),  Full (1)
Kernel : B30f (0),  B45f (1),  B60f (2)

Kernel aliases (clinically equivalent):
  B30f ≡ Br36d (Soft/Standard)  |  B45f ≡ Br49d (Medium)  |  B60f ≡ Br59d (Sharp/High-frequency)
```

### 9 Default Conversion Tasks & 5 Target Domains

The 9 paired tasks map directly into **5 distinct target domains**:

| Source → Target | Task Type | Target Domain ($T_k$) | Characteristics |
|---|---|---|---|
| Quarter,B45f → Quarter,B30f | Kernel conversion | `Quarter_B30f` | Smooth kernel, Quarter noise texture |
| Quarter,B60f → Quarter,B30f | Kernel conversion | `Quarter_B30f` | Smooth kernel, Quarter noise texture |
| Quarter,B60f → Quarter,B45f | Kernel conversion | `Quarter_B45f` | Medium kernel, Quarter noise texture |
| Full,B45f → Full,B30f | Kernel conversion | `Full_B30f` | Standard soft kernel, low noise (NDCT) |
| Full,B60f → Full,B30f | Kernel conversion | `Full_B30f` | Standard soft kernel, low noise (NDCT) |
| Quarter,B30f → Full,B30f | Denoising | `Full_B30f` | Standard soft kernel, low noise (NDCT) |
| Full,B60f → Full,B45f | Kernel conversion | `Full_B45f` | Medium kernel, low noise (NDCT) |
| Quarter,B45f → Full,B45f | Denoising | `Full_B45f` | Medium kernel, low noise (NDCT) |
| Quarter,B60f → Full,B60f | Denoising | `Full_B60f` | Sharp kernel, low noise (NDCT) |

---

## Target-Conditioned FD-Loss

### Why FD-Loss over REPA?
In medical CT restoration (especially image-to-image translation), **PSNR and SSIM are paramount** for clinical diagnostic accuracy:
* **REPA (Representation Alignment)** forces denoiser intermediate patch features to align with frozen DINOv2 patch features via point-wise cosine similarity. While it improves perceptual and semantic metrics (LPIPS/FID), it causes slight pixel-level HU distortion, **leading to degraded PSNR and SSIM**.
* **FD-Loss (Representation Fréchet Loss)** aligns the global feature distribution ($\mu, \Sigma$) of generated CT slices with target reference statistics without imposing point-wise penalties on individual pixels. This preserves pixel-level I2SB reconstruction accuracy while effectively matching domain-specific textures and MTF characteristics.

### Target-Conditioned Routing
A single global statistic would cause conflicting gradients between Quarter-dose tasks (noise should remain) and Full-dose tasks (noise should be removed), or between smooth B30f and sharp B60f kernels.
Our **Target-Conditioned FD-Loss** maintains 5 separate `FeatureQueue` instances:
$$\mathcal{L}_{FD} = \frac{1}{|K_{batch}|} \sum_{k \in K_{batch}} \frac{\text{FD}\left((\mu_{gen}^{(k)}, \Sigma_{gen}^{(k)}), (\mu_{ref}^{(k)}, \Sigma_{ref}^{(k)})\right)}{\text{FD} + \epsilon}$$
* **EMA Stats Tracking**: Tracks running first and second moments (`--fd-ema-beta 0.999`) without storing large image buffers, maintaining high stability even with small microbatches.
* **Fast Trace Computation**: Computes $\Sigma_{ref}^{1/2}$ once at initialization (`--fd-eigvalsh`), providing $\sim 8\times$ faster eigenvalue calculations during backpropagation.

---

## Dataset Directory Structure

```
{dataset_dir}/
├── train/
│   └── {group}/
│       └── {dose}/
│           └── {patient}/
│               └── {kernel}/
│                   └── *.dcm
├── valid/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

**Example:**
```
dataset/
├── train/
│   └── WBCT_Chest/
│       ├── Quarter/
│       │   └── Patient001/
│       │       ├── B30f/  *.dcm
│       │       ├── B45f/  *.dcm
│       │       └── B60f/  *.dcm
│       └── Full/
│           └── Patient001/
│               └── B30f/  *.dcm
```

---

## Installation

```bash
# Install base requirements
pip install --upgrade typing_extensions
pip install -r requirements.txt

# Install DICOM & image I/O dependencies
pip install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64
pip install pylibjpeg==1.3.0 pylibjpeg-libjpeg==1.3.0
pip install pylibjpeg-openjpeg==1.2.0 pylibjpeg-rle==1.2.0
pip install pydicom==2.1.0 numpy==1.22.3
```

> **Environment**: Python 3.10, PyTorch 2.1.2, CUDA 11.x / 12.x

---

## Progressive Ablation Study (4-Step Ladder)

The ablation study isolates the contribution of each architectural innovation systematically:

| Step | Model Config | `--embedding-mode` | Loss Formulation | Description |
|---|---|---|---|---|
| **Step 1** | PixelDiT-Large | `base` | $\mathcal{L}_{I2SB}$ | Baseline multi-domain model with independent dose & kernel table embeddings |
| **Step 2** | PixelDiT-Large | `base_cross` | $\mathcal{L}_{I2SB}$ | Adds pairwise `(dose × kernel)` cross-domain interaction embeddings |
| **Step 3** | PixelDiT-Large | `base_cross_noise` | $\mathcal{L}_{I2SB}$ | Adds Haar-wavelet noise statistics vector (20-dim) to condition on degradation level |
| **Step 4** | PixelDiT-Large | `base_cross_noise` | $\mathcal{L}_{I2SB}$ + $\lambda_{FD}$ $\mathcal{L}_{FD}$ | Adds Target-Conditioned FD-Loss across 5 target domains (EMA mode, `--fd-eigvalsh`) |

---

## Step-by-Step Training Guide

### Step 0: [Prerequisite] Compute Reference Statistics (Run ONCE)

Before training Step 4, extract and save reference feature distributions $(\mu_{ref}, \Sigma_{ref})$ across the 5 target domains:

```bash
# Single GPU
python compute_fd_stats.py \
    --dataset-dir /workspace/changyong/07.LDCT2NDCT/data/changyong_dataset \
    --split       train \
    --dino-model  dinov2_vitb14 \
    --output      data/fd_stats/target_5domains_dinov2b14.npz

# Multi-GPU (torchrun)
torchrun --nproc_per_node=2 compute_fd_stats.py \
    --dataset-dir /workspace/changyong/07.LDCT2NDCT/data/changyong_dataset \
    --split       train \
    --dino-model  dinov2_vitb14 \
    --output      data/fd_stats/target_5domains_dinov2b14.npz
```

---

### Step 1 to 4: Training Commands

All training steps can be managed via `scripts/run_train.sh`:

```bash
bash scripts/run_train.sh
```

#### Step 4 Execution (Target-Conditioned FD-Loss)

```bash
# Fine-tune from Step 3 checkpoint (Recommended)
python train.py \
    --dataset-dir        /workspace/changyong/07.LDCT2NDCT/data/changyong_dataset \
    --corrupt            multi_domain \
    --num-dose-attrs     2 \
    --num-kernel-attrs   3 \
    --n-gpu-per-node     2 \
    --master-port        4574 \
    --clip-denoise \
    --cond-x1 \
    --batch-size         64 \
    --microbatch         1 \
    --lr                 1e-4 \
    --lr-gamma           0.99 \
    --lr-step            1000 \
    --ema                0.99 \
    --hidden-size        1024 \
    --pixel-hidden-size  16 \
    --patch-depth        22 \
    --pixel-depth        4 \
    --num-groups         16 \
    --patch-size         16 \
    --embedding-mode     base_cross_noise \
    --noise-feat-dim     20 \
    --feat-loss-weight   0.0 \
    --dino-model         dinov2_vitb14 \
    --fd-loss-weight     1.0 \
    --fd-loss-interval   1 \
    --fd-stats-path      data/fd_stats/target_5domains_dinov2b14.npz \
    --fd-ema-beta        0.999 \
    --fd-eigvalsh \
    --fd-norm-eps        0.01 \
    --ckpt               step3_i2sb_base_cross_noise_RTX3090 \
    --start-itr          200000 \
    --num-itr            250000 \
    --name               step4_i2sb_fd_base_cross_noise_RTX3090
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--embedding-mode` | `base_cross_noise` | Domain conditioning mode (`base`, `base_cross`, `base_cross_noise`) |
| `--noise-feat-dim` | `20` | Dimension of input Haar wavelet noise feature |
| `--feat-loss-weight` | `0.0` | REPA loss weight (set to `0.0` to preserve PSNR/SSIM) |
| `--fd-loss-weight` | `0.0` | Weight for Target-Conditioned FD-Loss (`1.0` in Step 4) |
| `--fd-stats-path` | `None` | Path to reference statistics `.npz` file |
| `--fd-ema-beta` | `0.999` | Exponential moving average decay for FD statistics tracking |
| `--fd-eigvalsh` | `False` | Enable precomputed $\Sigma_{ref}^{1/2}$ for $\sim 8\times$ faster eigenvalue trace computation |
| `--fd-norm-eps` | `0.01` | Normalization epsilon: $\mathcal{L}_{FD} = \text{FD} / (\text{FD} + \epsilon) \in [0, 1)$ |
| `--hidden-size` | `1024` | PixDiT patch stream context dimension (Large: 1024) |
| `--patch-depth` | `22` | Number of patch-level `AugmentedDiTBlock` layers (Large: 22) |
| `--pixel-depth` | `4` | Number of pixel-level `PiTBlock` layers (Large: 4) |
| `--batch-size` | `64` | Global batch size across GPUs |
| `--microbatch` | `1` | Microbatch size per GPU step (gradient accumulation) |

---

## Sampling / Inference

Execute DDPM sampling on the evaluation/test split:

```bash
bash scripts/run_sample.sh
```

**Manual execution example:**
```bash
python sample.py \
    --dataset-dir    /workspace/changyong/07.LDCT2NDCT/data/changyong_dataset \
    --ckpt           step4_i2sb_fd_base_cross_noise_RTX3090 \
    --load-itr       250000 \
    --pair-defs      "Quarter,B45f:Quarter,B30f" "Quarter,B60f:Quarter,B30f" "Quarter,B60f:Quarter,B45f" \
                     "Full,B45f:Full,B30f" "Full,B60f:Full,B30f" "Full,B60f:Full,B45f" \
                     "Quarter,B30f:Full,B30f" "Quarter,B45f:Full,B45f" "Quarter,B60f:Full,B60f" \
    --n-gpu-per-node 1 \
    --batch-size     1 \
    --nfe            2 \
    --clip-denoise
```

### Output Structure

```
results/{ckpt_name}/
└── samples_corrected_nfe{N}_clip_iter{M}/
    └── {task_name}/           # e.g., Quarter_B30f_to_Full_B30f
        └── {group_name}/      # e.g., WBCT_Chest
            └── {patient_id}/
                ├── {slice}_source.png        # Source image (LDCT / Sharp)
                ├── {slice}_target.png        # Ground-truth target (NDCT / Soft)
                └── {slice}_target_recon.png  # Restored reconstruction
```

---

## Quantitative Evaluation

Evaluate image quality metrics across all 9 restoration tasks:

```bash
python evaluate_metrics.py \
    --results_dir     results \
    --experiment_name step4_i2sb_fd_base_cross_noise_RTX3090 \
    --samples_dir     samples_corrected_nfe2_clip_iter250000
```

### Metrics Computed

| Metric | Level | Description |
|---|---|---|
| **PSNR** | Per-slice & per-task | Peak Signal-to-Noise Ratio (dB) in normalized HU space |
| **SSIM** | Per-slice & per-task | Structural Similarity Index |
| **FID** | Per-task (pooled) | Fréchet Inception Distance evaluating perceptual feature distribution |
| **FRD** | Per-task (pooled) | Fréchet Radiomics Distance evaluating clinically relevant CT texture & shape features |

Outputs are automatically saved to `results/{experiment_name}/{samples_dir}/evaluation_results/`:
* `metrics_slice_level.csv`: Detailed slice-by-slice metric records.
* `metrics_task_level.csv`: Summary aggregated mean and standard deviation per task.

---

## File Structure

```
i2sb_pixeldit_multidomain/
│
├── train.py                     # Main training entry point
├── sample.py                    # Multi-domain sampling & inference entry point
├── evaluate_metrics.py          # Metric evaluation script (PSNR, SSIM, FID, FRD)
├── compute_fd_stats.py          # 5-domain reference statistics precomputation
├── distributed_util.py          # Multi-GPU / DDP utility module
├── logger.py                    # Logging utility (TensorBoard / WandB)
├── requirements.txt
│
├── i2sb/                        # Core I2SB framework
│   ├── runner.py                # Training loop, evaluation, checkpointing & FD routing
│   ├── fd_loss.py               # FeatureQueue, MultiDomainFeatureQueue, differentiable FD loss
│   ├── network.py               # PixDiTNet: I2SB model wrapper
│   ├── diffusion.py             # Schrödinger bridge forward process & sampling steps
│   ├── ckpt_util.py             # Checkpoint I/O utilities
│   └── util.py                  # Dataloader and optimizer setup helpers
│
├── pixdit_core/                 # PixelDiT architecture
│   ├── pixeldit_c2i.py          # Dual-stream PixelDiT backbone
│   └── modules.py               # AugmentedDiTBlock, PiTBlock, NoiseAdaptiveMultiDomainEmbedder
│
├── dataset/
│   └── dataset.py               # Multi-domain CT dataset loader & pair resolution
│
└── scripts/
    ├── run_train.sh             # Training execution script (Step 1 to Step 4)
    └── run_sample.sh            # Sampling execution script
```

---

## Cluster / SBATCH Execution

Cluster batch scripts are located under:
```
h:/SBATCH_Workspace/LDCT2NDCT/i2sb_pixeldit_multidomain/
├── train.sh                     # SBATCH script launching training container
├── sample.sh                    # SBATCH script launching sampling container
└── compute_metrics.sh           # SBATCH script for running offline evaluations
```

Submit jobs via SLURM:
```bash
sbatch train.sh
sbatch sample.sh
sbatch compute_metrics.sh
```
