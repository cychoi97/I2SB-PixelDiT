#!/bin/bash
# ============================================================
# Multi-Domain CT Restoration — Sampling Script (I2SB DDPM)
#
# Runs DDPM sampling on the test split using a trained
# multi-domain checkpoint.
#
# Pair format: "SrcDose,SrcKernel:TrgDose,TrgKernel"
# ============================================================

DATASET_DIR="/workspace/changyong/07.LDCT2NDCT/data/changyong_dataset"
CKPT_NAME="step2_i2sb_repa_base_RTX3090"
LOAD_ITR=200000
NFE=10

# ---------------------------------------------------------------
# Sample: all 9 default pairs (I2SB DDPM)
# ---------------------------------------------------------------
python sample.py \
    --dataset-dir   ${DATASET_DIR} \
    --ckpt          ${CKPT_NAME} \
    --load-itr      ${LOAD_ITR} \
    --pair-defs     "Quarter,B45f:Quarter,B30f" "Quarter,B60f:Quarter,B30f" "Quarter,B60f:Quarter,B45f" "Full,B45f:Full,B30f" "Full,B60f:Full,B30f" "Full,B60f:Full,B45f" "Quarter,B30f:Full,B30f" "Quarter,B45f:Full,B45f" "Quarter,B60f:Full,B60f" \
    --n-gpu-per-node 1 \
    --batch-size    1 \
    --nfe           ${NFE} \
    --clip-denoise

# ---------------------------------------------------------------
# Sample: kernel conversion only
# ---------------------------------------------------------------
# python sample.py \
#     --dataset-dir   ${DATASET_DIR} \
#     --ckpt          ${CKPT_NAME} \
#     --load-itr      ${LOAD_ITR} \
#     --pair-defs     "Quarter,B45f:Quarter,B30f" "Quarter,B60f:Quarter,B30f" \
#     --n-gpu-per-node 1 \
#     --batch-size    1 \
#     --nfe           ${NFE} \
#     --clip-denoise

# ---------------------------------------------------------------
# Sample: denoising only
# ---------------------------------------------------------------
# python sample.py \
#     --dataset-dir   ${DATASET_DIR} \
#     --ckpt          ${CKPT_NAME} \
#     --load-itr      ${LOAD_ITR} \
#     --pair-defs     "Quarter,B30f:Full,B30f" \
#     --n-gpu-per-node 1 \
#     --batch-size    1 \
#     --nfe           ${NFE} \
#     --clip-denoise
