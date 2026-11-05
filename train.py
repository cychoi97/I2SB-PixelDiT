from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import dataset
from i2sb import Runner
from dataset.dataset import (
    DOSE_ATTRS,
    KERNEL_ATTRS,
    DOSE_ID_TO_NAME,
    KERNEL_ID_TO_NAME,
    NUM_DOSE_ATTRS,
    NUM_KERNEL_ATTRS,
    KERNEL_ALIAS_MAP,
    DEFAULT_PAIRS,
    resolve_pair_defs,
    pair_label,
)

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--name",               type=str,   default=None,               help="experiment ID")
    parser.add_argument("--ckpt",               type=str,   default=None,               help="resumed checkpoint name")
    parser.add_argument("--gpu",                type=int,   default=None,               help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node",     type=int,   default=1,                  help="number of gpu on each node")
    parser.add_argument("--master-address",     type=str,   default='localhost',        help="address for master")
    parser.add_argument("--master-port",        type=str,   default='6020',             help="port for master")
    parser.add_argument("--node-rank",          type=int,   default=0,                  help="the index of node")
    parser.add_argument("--num-proc-node",      type=int,   default=1,                  help="The number of nodes in multi node env")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",         type=int,   default=512)
    parser.add_argument("--in-channels",        type=int,   default=1)
    parser.add_argument("--t0",                 type=float, default=1e-4,               help="sigma start time in network parametrization")
    parser.add_argument("--T",                  type=float, default=1.,                 help="sigma end time in network parametrization")
    parser.add_argument("--interval",           type=int,   default=1000,               help="number of interval")
    parser.add_argument("--beta-max",           type=float, default=0.3,                help="max diffusion for the diffusion model")
    parser.add_argument("--ot-ode",             action="store_true",                    help="use OT-ODE model")
    parser.add_argument("--clip-denoise",       action="store_true",                    help="clamp predicted image to [-1,1] at each step")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",            action="store_true",                    help="conditional the network on degraded images")

    # --------------- multi-domain settings ---------------
    parser.add_argument("--corrupt",            type=str,   default='multi_domain',     help="task type: use 'multi_domain' for the universal model")
    parser.add_argument("--pair-defs",          type=str,   default=None, nargs='+',    help="Conversion pairs: 'SrcDose,SrcKernel:TrgDose,TrgKernel'. If omitted, all DEFAULT_PAIRS are used.", metavar="PAIR")
    parser.add_argument("--num-dose-attrs",     type=int,   default=NUM_DOSE_ATTRS,     help=f"Number of dose attributes (default: {NUM_DOSE_ATTRS})")
    parser.add_argument("--num-kernel-attrs",   type=int,   default=NUM_KERNEL_ATTRS,   help=f"Number of kernel attributes (default: {NUM_KERNEL_ATTRS})")
    parser.add_argument("--noise-feat-dim",     type=int,   default=20,                 help="Dimension of noise statistics feature")
    parser.add_argument("--embedding-mode",     type=str,   default="base_cross_noise", choices=["base", "base_cross", "base_cross_noise"],
                        help="Select embedding mode")

    # --------------- PixDiT architecture ---------------
    parser.add_argument("--hidden-size",        type=int,   default=1152,               help="PixDiT patch-level context dim")
    parser.add_argument("--pixel-hidden-size",  type=int,   default=16,                 help="PixDiT pixel-level feature dim")
    parser.add_argument("--patch-depth",        type=int,   default=12,                 help="Number of AugmentedDiTBlock layers")
    parser.add_argument("--pixel-depth",        type=int,   default=2,                  help="Number of PiTBlock layers")
    parser.add_argument("--num-groups",         type=int,   default=16,                 help="Attention heads (must divide hidden-size)")
    parser.add_argument("--patch-size",         type=int,   default=16,                 help="Pixel patch size for dual-level pipeline")

    # --------------- REPA feature alignment loss ---------------
    parser.add_argument("--feat-loss-weight",   type=float, default=0.5,                help="Weight for REPA feature alignment loss. Set 0 to disable.")
    parser.add_argument("--feat-loss-interval", type=int,   default=1,                  help="Compute feature loss every N iterations.")
    parser.add_argument("--dino-model",         type=str,   default="dinov2_vitb14",    help="DINOv2 model variant (shared by REPA and FD-Loss).")
    parser.add_argument("--align-layer",        type=int,   default=8,                  help="Patch block layer index (1-based) for REPA alignment.")

    # --------------- FD-Loss: distribution matching in DINOv2 feature space ---------------
    parser.add_argument("--fd-loss-weight",     type=float, default=0.0,
                        help="Weight for FD-Loss (Fréchet Distance distribution matching). Set 0 to disable.")
    parser.add_argument("--fd-loss-interval",   type=int,   default=1,
                        help="Compute FD-Loss every N iterations (default: 1).")
    parser.add_argument("--fd-stats-path",      type=str,   default=None,
                        help="Path to reference statistics .npz (computed by compute_fd_stats.py). Required when --fd-loss-weight > 0.")
    parser.add_argument("--fd-queue-size",      type=int,   default=5000,
                        help="Feature queue size for FD covariance estimation when fd_ema_beta=0 (default: 5000).")
    parser.add_argument("--fd-ema-beta",        type=float, default=0.999,
                        help="EMA decay for FD stats (default: 0.999 as in FD-Loss paper; set 0.0 to use queue).")
    parser.add_argument("--fd-eigvalsh",        action="store_true",
                        help="Use fast eigvalsh mode (~8x faster); requires precomputing σ_ref^{1/2}.")
    parser.add_argument("--fd-norm-eps",        type=float, default=0.01,
                        help="Epsilon for FD-Loss normalisation: loss = FD / (FD + eps) → [0, 1) (default: 0.01).")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",         type=int,   default=16)
    parser.add_argument("--microbatch",         type=int,   default=1,                  help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--start-itr",          type=int,   default=0,                  help="resumed checkpoint iteration")
    parser.add_argument("--num-itr",            type=int,   default=200000,             help="training iteration")
    parser.add_argument("--lr",                 type=float, default=1e-4,               help="learning rate")
    parser.add_argument("--lr-gamma",           type=float, default=0.99,               help="learning rate decay ratio")
    parser.add_argument("--lr-step",            type=int,   default=1000,               help="learning rate decay step size")
    parser.add_argument("--l2-norm",            type=float, default=0.0)
    parser.add_argument("--ema",                type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",        type=Path,  default="/data",            help="path to dataset")
    parser.add_argument("--log-dir",            type=Path,  default=".log",             help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",         type=str,   default=None,               help="log writer: tensorboard, wandb, or None")
    parser.add_argument("--wandb-api-key",      type=str,   default=None,               help="unique API key of your W&B account")
    parser.add_argument("--wandb-user",         type=str,   default=None,               help="user name of your W&B account")

    # --------------- dry-run ---------------
    parser.add_argument("--dry-run",            action="store_true",                    help="scan dataset paths and report file counts per pair, then exit")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device = 'cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16    = False

    # Validate pair definitions.
    resolved = resolve_pair_defs(opt.pair_defs)
    opt._resolved_pairs = resolved

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / f"{opt.start_itr:0>7}.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    assert opt.batch_size % opt.microbatch == 0, (
        f"{opt.batch_size} is not divisible by {opt.microbatch}!"
    )
    return opt


# ---------------------------------------------------------------------------
# Dry-run: inspect data availability before training
# ---------------------------------------------------------------------------

def dry_run(opt):
    """Scan dataset paths and report paired file counts per pair/split."""
    import glob as _glob

    resolved = resolve_pair_defs(opt.pair_defs)

    print("=" * 70)
    print("  DRY-RUN MODE  —  no model will be loaded or trained")
    print("=" * 70)
    print(f"  dataset-dir     : {opt.dataset_dir}")
    print(f"  num-dose-attrs  : {opt.num_dose_attrs}")
    print(f"  num-kernel-attrs: {opt.num_kernel_attrs}")
    print(f"  pairs           : {[pair_label(*p) for p in resolved]}")
    print()

    # Print attribute registries.
    print("  Dose Attributes:")
    for name, idx in sorted(DOSE_ATTRS.items(), key=lambda x: x[1]):
        print(f"    [{idx}] {name}")
    print("  Kernel Attributes:")
    for name, idx in sorted(KERNEL_ATTRS.items(), key=lambda x: x[1]):
        aliases = KERNEL_ALIAS_MAP.get(name, [])
        alias_str = f"  (aliases: {aliases})" if aliases else ""
        print(f"    [{idx}] {name}{alias_str}")
    print()

    splits = ['train', 'valid', 'test']

    for split in splits:
        split_dir = opt.dataset_dir / split
        if not split_dir.exists():
            print(f"  [{split:5s}]  directory not found: {split_dir}")
            continue

        print(f"  [{split}]")
        grand_total = 0

        for src_dose, src_kernel, trg_dose, trg_kernel in resolved:
            plabel = pair_label(src_dose, src_kernel, trg_dose, trg_kernel)

            # Determine the task type for display.
            if src_dose == trg_dose and src_kernel != trg_kernel:
                task_type = "kernel"
            elif src_dose != trg_dose and src_kernel == trg_kernel:
                task_type = "denoise"
            else:
                task_type = "cross"

            print(f"    Pair: {plabel}  (task={task_type})")

            # Search for source directories (including kernel aliases).
            src_kernel_folders = [src_kernel] + KERNEL_ALIAS_MAP.get(src_kernel, [])
            trg_kernel_folders = [trg_kernel] + KERNEL_ALIAS_MAP.get(trg_kernel, [])

            pair_total = 0

            for sk_folder in src_kernel_folders:
                pattern = os.path.join(str(split_dir), '*', src_dose, '*', sk_folder)
                src_dirs = sorted(_glob.glob(pattern))

                src_count, matched, missing = 0, 0, 0
                for src_dir in src_dirs:
                    parent = os.path.dirname(src_dir)
                    found = False
                    for tk_folder in trg_kernel_folders:
                        if src_dose == trg_dose:
                            trg_dir = os.path.join(parent, tk_folder)
                        else:
                            trg_parent = parent.replace(
                                os.sep + src_dose + os.sep,
                                os.sep + trg_dose + os.sep,
                            )
                            if trg_parent == parent:
                                trg_parent = parent.replace(
                                    '/' + src_dose + '/',
                                    '/' + trg_dose + '/',
                                )
                            trg_dir = os.path.join(trg_parent, tk_folder)

                        if os.path.isdir(trg_dir):
                            sf = sorted(f for f in _glob.glob(os.path.join(src_dir, '*.dcm'))
                                        if f.endswith('.dcm'))
                            tf = sorted(f for f in _glob.glob(os.path.join(trg_dir, '*.dcm'))
                                        if f.endswith('.dcm'))
                            src_count += len(sf)
                            matched   += min(len(sf), len(tf))
                            found = True
                            break  # found target, skip other aliases

                    if not found:
                        missing += 1

                pair_total += matched
                print(f"      ({sk_folder})  dirs={len(src_dirs):,}  "
                      f"src_files={src_count:,}  pairs={matched:,}  "
                      f"missing_trg={missing}")
                if src_dirs and matched == 0:
                    print(f"      [WARNING] dirs found but no pairs formed!")
                elif not src_dirs:
                    print(f"      [WARNING] no source dirs found!")

            print(f"      Pair total: {pair_total:,}")
            grand_total += pair_total

        print(f"    {'─' * 50}")
        print(f"    Grand total ({split}): {grand_total:,} pairs")
        print()

    print("=" * 70)
    print("  Dry-run complete. Exiting.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("    Multi-Domain Image-to-Image Schrodinger Bridge      ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    resolved = getattr(opt, '_resolved_pairs', resolve_pair_defs(opt.pair_defs))
    log.info(f"Active pairs   : {[pair_label(*p) for p in resolved]}")
    log.info(f"Dose attrs     : {DOSE_ATTRS}")
    log.info(f"Kernel attrs   : {KERNEL_ATTRS}")

    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    train_dataset = dataset.CTDataset(opt, log, mode='train')
    val_dataset   = dataset.CTDataset(opt, log, mode='valid')

    run = Runner(opt, log)
    run.train(opt, train_dataset, val_dataset)
    log.info("Finish!")


if __name__ == '__main__':
    opt = create_training_options()

    if opt.dry_run:
        dry_run(opt)
        sys.exit(0)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank  = rank
            global_rank     = rank + opt.node_rank * opt.n_gpu_per_node
            global_size     = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print(
                'Node rank %d, local proc %d, global proc %d, global_size %d'
                % (opt.node_rank, rank, global_rank, global_size)
            )
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank  = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
