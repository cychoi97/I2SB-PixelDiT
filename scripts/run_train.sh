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
#
# PixelDiT architecture:
#   patch-level depth N: 12 (base), 18 (medium), 22 (large), 26 (xlarge)
#   pixel-level depth M: 2 (base), 4 (medium), 4 (large), 4 (xlarge)
#   hidden-size D: 768 (base), 896 (medium), 1024 (large), 1152 (xlarge)
#   num groups G: 12 (base), 14 (medium), 16 (large), 16 (xlarge)
#   pixel hidden size D_pix: 16 (base), 16 (medium), 16 (large), 16 (xlarge)
#   patch size: 16 (base), 16 (medium), 16 (large), 16 (xlarge)
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
    --microbatch         4 \
    --num-itr            200000 \
    --lr                 1e-4 \
    --lr-gamma           0.99 \
    --lr-step            1000 \
    --ema                0.99 \
    --hidden-size        768 \
    --pixel-hidden-size  16 \
    --patch-depth        12 \
    --pixel-depth        2 \
    --num-groups         12 \
    --patch-size         16"

# ---------------------------------------------------------------
# Step 1: Baseline (I2SB + PixelDiT + base embedding only)
# ---------------------------------------------------------------
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4571 \
#     --embedding-mode     base \
#     --feat-loss-weight   0.0 \
#     --name               step1_i2sb_base_${GPU_NAME} \
#     --ckpt               step1_i2sb_base_${GPU_NAME} \
#     --start-itr          175000

# ---------------------------------------------------------------
# Step 2: + REPA loss (I2SB + DINOv2 feature alignment)
# ---------------------------------------------------------------
python train.py \
    ${ABLATION_COMMON} \
    --master-port        4572 \
    --embedding-mode     base \
    --feat-loss-weight   0.5 \
    --feat-loss-interval 1 \
    --dino-model         dinov2_vitb14 \
    --align-layer        8 \
    --name               step2_i2sb_repa_base_${GPU_NAME} \
    --ckpt               step2_i2sb_repa_base_${GPU_NAME} \
    --start-itr          185000

# ---------------------------------------------------------------
# Step 3: + Cross embedding (I2SB + REPA + base_cross embedding)
# ---------------------------------------------------------------
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4573 \
#     --embedding-mode     base_cross \
#     --feat-loss-weight   0.5 \
#     --feat-loss-interval 1 \
#     --dino-model         dinov2_vitb14 \
#     --align-layer        8 \
#     --name               step3_i2sb_repa_base_cross_${GPU_NAME} \
#     --ckpt               step3_i2sb_repa_base_cross_${GPU_NAME} \
#     --start-itr          180000

# ---------------------------------------------------------------
# Step 4: + Noise adaptive embedding (I2SB + REPA + base_cross_noise)
# ---------------------------------------------------------------
# python train.py \
#     ${ABLATION_COMMON} \
#     --master-port        4574 \
#     --embedding-mode     base_cross_noise \
#     --noise-feat-dim     20 \
#     --feat-loss-weight   0.5 \
#     --feat-loss-interval 1 \
#     --dino-model         dinov2_vitb14 \
#     --align-layer        8 \
#     --name               step4_i2sb_repa_base_cross_noise_${GPU_NAME} \
#     --ckpt               step4_i2sb_repa_base_cross_noise_${GPU_NAME} \
#     --start-itr          180000
