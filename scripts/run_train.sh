#!/bin/bash
# ============================================================
# i2sb_v2_pixeldit: Full Multi-Domain CT Restoration
# Architecture : PixDiT (hidden=768, depth=12, patch_size=16)
# Conditioning : NoiseAdaptiveMultiDomainEmbedder (embedding_mode)
# Extra loss   : DINOv2 feature matching (REPA-style cosine)
#
# Domains:
#   dose   : [Quarter (0), Full (1)]
#   kernel : [B30f (0), B45f (1), B60f (2)]
#
# Default pairs (full multi-domain conditioning):
#   (Quarter,B45f) → (Quarter,B30f)   kernel conversion
#   (Quarter,B60f) → (Quarter,B30f)   kernel conversion
#   (Quarter,B60f) → (Quarter,B45f)   kernel conversion
#   (Quarter,B30f) → (Full,B30f)      denoising
#   (Quarter,B45f) → (Full,B45f)      densising
#   (Quarter,B60f) → (Full,B60f)      densising
#
# PixelDiT architecture:
#   patch-level depth N: 12 (base), 22 (large), 26 (xlarge)
#   pixel-level depth M: 2 (base), 4 (large), 4 (xlarge)
#   hidden-size D: 768 (base), 1024 (large), 1152 (xlarge)
#   num groups G: 12 (base), 16 (large), 16 (xlarge)
#   pixel hidden size D_pix: 16 (base), 16 (large), 16 (xlarge)
#   patch size: 16 (base), 16 (large), 16 (xlarge)
# ============================================================

GPU_NAME="RTX3090"
DATASET_DIR="/workspace/changyong/07.LDCT2NDCT/data/changyong_dataset"

# ---------------------------------------------------------------
# DRY-RUN: Verify dataset paths and pair counts before training.
# ---------------------------------------------------------------
# python train.py \
#     --dataset-dir ${DATASET_DIR} \
#     --name dry_run_multidomain \
#     --dry-run

# ================================================================
# Ablation Study Training Scripts (3-step ladder: I2SB only)
# ================================================================
# Common shared arguments for all ablation steps:
ABLATION_COMMON="\
    --dataset-dir        ${DATASET_DIR} \
    --corrupt            multi_domain \
    --num-dose-attrs     2 \
    --num-kernel-attrs   3 \
    --n-gpu-per-node     2 \
    --clip-denoise \
    --cond-x1 \
    --batch-size         64 \
    --microbatch         1 \
    --num-itr            200000 \
    --lr                 1e-4 \
    --lr-gamma           0.99 \
    --lr-step            1000 \
    --ema                0.99 \
    --hidden-size        1024 \
    --pixel-hidden-size  16 \
    --patch-depth        22 \
    --pixel-depth        4 \
    --num-groups         16 \
    --patch-size         16"

# ---------------------------------------------------------------
# Step 1: Baseline (I2SB + PixelDiT + base embedding only)
# ---------------------------------------------------------------
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4571 \
#     --embedding-mode     base \
#     --feat-loss-weight   0.0 \
#     --name               step1_PixelDiT-L_i2sb_base_${GPU_NAME} \
#     --ckpt               step1_PixelDiT-L_i2sb_base_${GPU_NAME} \
#     --start-itr          60000

# ---------------------------------------------------------------
# Step 2: + Cross embedding (without REPA loss)
# ---------------------------------------------------------------
python train.py \
    ${ABLATION_COMMON} \
    --master-port        4572 \
    --embedding-mode     base_cross \
    --feat-loss-weight   0.0 \
    --dino-model         dinov2_vitb14 \
    --name               step2_PixelDiT-L_i2sb_base_cross_${GPU_NAME} \
    --ckpt               step2_PixelDiT-L_i2sb_base_cross_${GPU_NAME} \
    --start-itr          60000

# ---------------------------------------------------------------
# Step 3: + Noise adaptive embedding (I2SB + base_cross_noise)
# ---------------------------------------------------------------
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4573 \
#     --embedding-mode     base_cross_noise \
#     --noise-feat-dim     20 \
#     --feat-loss-weight   0.0 \
#     --dino-model         dinov2_vitb14 \
#     --name               step3_i2sb_base_cross_noise_${GPU_NAME} \
#     --ckpt               step3_i2sb_base_cross_noise_${GPU_NAME} \
#     --start-itr          160000

# ================================================================
# [PREREQUISITE] Compute FD-Loss reference statistics (run ONCE)
# ================================================================
# Compute reference statistics for all 5 target domains in a single pass:
#   - Full_B30f, Full_B45f, Full_B60f, Quarter_B30f, Quarter_B45f
#
# Single GPU:
# python compute_fd_stats.py \
#     --dataset-dir ${DATASET_DIR} \
#     --split       train \
#     --dino-model  dinov2_vitb14 \
#     --output      data/fd_stats/target_5domains_dinov2b14.npz

# Multi-GPU (torchrun):
#   torchrun --nproc_per_node=2 compute_fd_stats.py \
#       --dataset-dir ${DATASET_DIR} \
#       --split       train \
#       --dino-model  dinov2_vitb14 \
#       --output      data/fd_stats/target_5domains_dinov2b14.npz

# ---------------------------------------------------------------
# Step 4: + Target-Conditioned FD-Loss (I2SB + base_cross_noise, NO REPA)
#   - REPA is disabled (--feat-loss-weight 0.0) to preserve PSNR/SSIM
#   - Target-Conditioned FD-Loss aligns each sample to its specific target domain
#   - 5 target domains (Full_B30f, Full_B45f, Full_B60f, Quarter_B30f, Quarter_B45f)
#   - EMA stats mode (fd_ema_beta=0.999 as in official paper)
#   - --fd-eigvalsh: ~8x faster covariance trace computation
# ---------------------------------------------------------------
# Option A: Train from scratch with FD-Loss
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4574 \
#     --embedding-mode     base_cross_noise \
#     --noise-feat-dim     20 \
#     --feat-loss-weight   0.0 \
#     --dino-model         dinov2_vitb14 \
#     --fd-loss-weight     1.0 \
#     --fd-loss-interval   1 \
#     --fd-stats-path      data/fd_stats/target_5domains_dinov2b14.npz \
#     --fd-ema-beta        0.999 \
#     --fd-eigvalsh \
#     --fd-norm-eps        0.01 \
#     --name               step4_i2sb_fd_base_cross_noise_${GPU_NAME}

# Option B: Fine-tune / Resume from Step 3 checkpoint (recommended)
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4574 \
#     --embedding-mode     base_cross_noise \
#     --noise-feat-dim     20 \
#     --feat-loss-weight   0.0 \
#     --dino-model         dinov2_vitb14 \
#     --fd-loss-weight     1.0 \
#     --fd-loss-interval   1 \
#     --fd-stats-path      data/fd_stats/target_5domains_dinov2b14.npz \
#     --fd-ema-beta        0.999 \
#     --fd-eigvalsh \
#     --fd-norm-eps        0.01 \
#     --ckpt               step3_i2sb_base_cross_noise_${GPU_NAME} \
#     --start-itr          200000 \
#     --num-itr            250000 \
#     --name               step4_i2sb_fd_base_cross_noise_${GPU_NAME}
