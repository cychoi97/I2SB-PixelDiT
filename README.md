# I2SB + PixelDiT — Multi-Domain CT Restoration

> **Universal LDCT-to-NDCT restoration** using Image-to-Image Schrödinger Bridge (I2SB) with a dual-level PixelDiT backbone and noise-adaptive multi-domain conditioning.

---

## Overview

This repository implements a universal model for low-dose CT (LDCT) restoration that performs **kernel conversion** and **denoising** simultaneously within a single network. Instead of training a separate model per conversion pair, a single model is conditioned on the source and target domain attributes (dose level + reconstruction kernel), enabling flexible many-to-many domain conversion.

### Key Components

| Component | Description |
|---|---|
| **I2SB** | Image-to-Image Schrödinger Bridge — probabilistic generative model for paired image translation |
| **PixelDiT** | Dual-level DiT backbone: patch-level context stream (AugmentedDiTBlock) + pixel-level detail stream (PiTBlock) |
| **NoiseAdaptiveMultiDomainEmbedder** | Learns conditioning embeddings for dose & kernel attributes with three embedding modes |
| **REPA Loss** | DINOv2-based representation alignment loss for improved feature-level learning |

---

## Architecture

```
Input: x_t (noisy image) + x1 (source/conditioned image, channel-concatenated)
         │
         ├─ PixelTokenEmbedder  → pixel-level feature stream
         │      │
         │      └─ pixel_depth × PiTBlock  (pixel-level detail)
         │
         ├─ PatchTokenEmbedder  → patch-level token stream
         │      │
         │      └─ patch_depth × AugmentedDiTBlock  (context reasoning)
         │              │
         │              └─ Conditioning via AdaLN:
         │                   TimestepConditioner(t) + NoiseAdaptiveMultiDomainEmbedder(domain)
         │
         └─ FinalLayer → output prediction (noise / denoised direction)
```

### Multi-Domain Conditioning Modes (`--embedding-mode`)

| Mode | Description |
|---|---|
| `base` | Base embeddings: `src_dose` + `src_kernel` + `trg_dose` + `trg_kernel` (4 tables, element-wise sum) |
| `base_cross` | `base` + cross-domain embedding for `src(dose × kernel)` and `trg(dose × kernel)` pair index, fused via MLP |
| `base_cross_noise` | `base_cross` + Haar-wavelet-based noise statistics vector (20-dim), fused via MLP |

### PixelDiT Model Sizes

| Size | `patch_depth` | `pixel_depth` | `hidden_size` | `num_groups` | `pixel_hidden_size` |
|---|---|---|---|---|---|
| Base | 12 | 2 | 768 | 12 | 16 |
| Medium | 18 | 4 | 896 | 14 | 16 |
| Large | 22 | 4 | 1024 | 16 | 16 |
| XLarge | 26 | 4 | 1152 | 16 | 16 |

---

## Domain Attributes

```
Dose   : Quarter (0),  Full (1)
Kernel : B30f (0),  B45f (1),  B60f (2)

Kernel aliases (clinically equivalent):
  B30f ≡ Br36d  |  B45f ≡ Br49d  |  B60f ≡ Br59d
```

**Default Conversion Pairs (9 tasks):**

| Source → Target | Task Type |
|---|---|
| Quarter,B45f → Quarter,B30f | Kernel conversion |
| Quarter,B60f → Quarter,B30f | Kernel conversion |
| Quarter,B60f → Quarter,B45f | Kernel conversion |
| Full,B45f → Full,B30f | Kernel conversion |
| Full,B60f → Full,B30f | Kernel conversion |
| Full,B60f → Full,B45f | Kernel conversion |
| Quarter,B30f → Full,B30f | Denoising |
| Quarter,B45f → Full,B45f | Denoising |
| Quarter,B60f → Full,B60f | Denoising |

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
changyong_dataset/
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
pip install opencv-python==4.5.5.64
pip install opencv-python-headless==4.5.5.64
pip install pylibjpeg==1.3.0 pylibjpeg-libjpeg==1.3.0
pip install pylibjpeg-openjpeg==1.2.0 pylibjpeg-rle==1.2.0
pip install pydicom==2.1.0
pip install numpy==1.22.3
```

> **Environment**: Python 3.10, PyTorch 2.1.2, CUDA 11.x+

---

## Training

### Quick Start (via shell script)

```bash
bash scripts/run_train.sh
```

### Manual Training Command

```bash
python train.py \
    --dataset-dir    /path/to/dataset \
    --corrupt        multi_domain \
    --num-dose-attrs 2 \
    --num-kernel-attrs 3 \
    --n-gpu-per-node 2 \
    --master-port    4572 \
    --cond-x1 \
    --clip-denoise \
    --batch-size     64 \
    --microbatch     4 \
    --num-itr        200000 \
    --lr             1e-4 \
    --lr-gamma       0.99 \
    --lr-step        1000 \
    --ema            0.99 \
    --hidden-size    768 \
    --patch-depth    12 \
    --pixel-depth    2 \
    --num-groups     12 \
    --patch-size     16 \
    --embedding-mode base \
    --feat-loss-weight 0.0 \
    --name           my_experiment
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--embedding-mode` | `base_cross_noise` | Domain conditioning mode (`base`, `base_cross`, `base_cross_noise`) |
| `--feat-loss-weight` | `0.5` | REPA feature alignment loss weight (0 = disabled) |
| `--dino-model` | `dinov2_vitl14` | DINOv2 variant for REPA (`dinov2_vitb14`, `dinov2_vitl14`) |
| `--align-layer` | `8` | Patch block index (1-based) to extract features for REPA |
| `--noise-feat-dim` | `20` | Noise statistics feature dimension (used in `base_cross_noise` mode) |
| `--batch-size` | `16` | Total batch size across all GPUs |
| `--microbatch` | `1` | Per-step batch size (gradient accumulation) |
| `--num-itr` | `200000` | Total training iterations |
| `--ckpt` | `None` | Checkpoint name to resume from |
| `--start-itr` | `0` | Starting iteration for resumed checkpoint |

### Dataset Dry-Run

Before training, verify dataset paths and file counts:

```bash
python train.py \
    --dataset-dir /path/to/dataset \
    --name dry_run \
    --dry-run
```

---

## Ablation Study Plan

The ablation study follows a **4-step progressive ladder** to isolate each component's contribution:

| Step | Config | `--embedding-mode` | REPA | Key Change |
|---|---|---|---|---|
| 1 | Baseline | `base` | ✗ | I2SB + PixelDiT + base embedding only |
| 2 | + REPA | `base` | ✓ | Add DINOv2 feature alignment loss |
| 3 | + Cross embedding | `base_cross` | ✓ | Add dose×kernel cross-domain embedding |
| 4 | + Noise-adaptive | `base_cross_noise` | ✓ | Add input noise statistics conditioning |

---

## Sampling

### Quick Start (via shell script)

```bash
bash scripts/run_sample.sh
```

### Manual Sampling Command

```bash
python sample.py \
    --dataset-dir /path/to/dataset \
    --ckpt        my_experiment \
    --load-itr    200000 \
    --pair-defs   "Quarter,B45f:Quarter,B30f" "Quarter,B30f:Full,B30f" \
    --n-gpu-per-node 1 \
    --batch-size  1 \
    --nfe         10 \
    --clip-denoise
```

### Key Sampling Arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | — | Checkpoint name (folder under `results/`) |
| `--load-itr` | `150000` | Checkpoint iteration to load |
| `--nfe` | — | Number of function evaluations (DDPM steps); defaults to `interval - 1` |
| `--pair-defs` | — | Conversion pairs to sample; uses all 9 default pairs if omitted |
| `--clip-denoise` | — | Clamp predictions to [-1, 1] at each step |
| `--embedding-mode` | `base_cross_noise` | Must match the training config |

### Output Structure

```
results/{ckpt_name}/
└── samples_corrected_nfe{N}_clip_iter{M}/
    └── {task_name}/           # e.g. Quarter_B45f_to_Quarter_B30f
        └── {group_name}/      # e.g. WBCT_Chest
            └── {patient_id}/
                ├── {slice}_source.png        # source image
                ├── {slice}_target.png        # ground-truth target
                └── {slice}_target_recon.png  # reconstruction
```

---

## Evaluation

After sampling, compute quantitative metrics (PSNR, SSIM, LPIPS, FID, FRD):

```bash
python evaluate_metrics.py \
    --results_dir      results \
    --experiment_name  my_experiment \
    --samples_dir      samples_corrected_nfe10_clip_iter200000
```

### Metrics

| Metric | Level | Description |
|---|---|---|
| **PSNR** | Per-slice | Peak Signal-to-Noise Ratio (dB), computed in HU space |
| **SSIM** | Per-slice | Structural Similarity Index |
| **LPIPS** | Per-slice | Learned Perceptual Image Patch Similarity (VGG) |
| **FID** | Per-task | Fréchet Inception Distance |
| **FRD** | Per-task | Fréchet Radiomics Distance |

---

## File Structure

```
i2sb_pixeldit_multidomain/
│
├── train.py                    # Training entry point
├── sample.py                   # Sampling/inference entry point
├── evaluate_metrics.py         # Offline metric evaluation
├── distributed_util.py         # Multi-GPU distributed utilities
├── logger.py                   # TensorBoard/WandB logging wrapper
├── requirements.txt
│
├── i2sb/                       # Core I2SB framework
│   ├── runner.py               # Training loop, evaluation, DDPM sampling
│   ├── network.py              # PixDiTNet: I2SB network wrapper
│   ├── diffusion.py            # I2SB forward process & DDPM sampling
│   ├── ckpt_util.py            # Checkpoint loading utilities
│   └── util.py                 # Miscellaneous utilities
│
├── pixdit_core/                # PixelDiT backbone
│   ├── pixeldit_c2i.py         # PixDiT model (patch + pixel streams)
│   └── modules.py              # Building blocks:
│                               #   AugmentedDiTBlock, PiTBlock,
│                               #   NoiseAdaptiveMultiDomainEmbedder,
│                               #   TimestepConditioner, FinalLayer, ...
│
├── dataset/
│   └── dataset.py              # CTDataset, domain registry, pair utilities
│
├── corruption/                 # (Legacy) corruption modules
│
└── scripts/
    ├── run_train.sh            # Ablation study training scripts
    └── run_sample.sh           # Sampling scripts
```

---

## SBATCH / Cluster Execution

```
h:/SBATCH_Workspace/LDCT2NDCT/i2sb_pixeldit_multidomain/
├── train.sh                    # SBATCH wrapper for training
└── sample.sh                   # SBATCH wrapper for sampling
```

Submit training job:
```bash
sbatch train.sh
```

The SBATCH wrapper launches a Docker container and executes `scripts/run_train.sh` internally. Edit the active `python train.py` command in `scripts/run_train.sh` to select the desired ablation step.

---

## References

- **I2SB**: Liu et al., "I²SB: Image-to-Image Schrödinger Bridge" (ICML 2023)
- **PixelDiT**: Luo et al., "Pixel-level and Patch-level DiT for High-resolution Image Generation" (2024)
- **REPA**: Stabilizing representation learning for diffusion models via DINOv2 feature alignment
- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
