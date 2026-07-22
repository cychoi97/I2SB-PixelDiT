import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner
from dataset import dataset
from i2sb import ckpt_util
from dataset.dataset import (
    DOSE_ATTRS,
    KERNEL_ATTRS,
    DOSE_ID_TO_NAME,
    KERNEL_ID_TO_NAME,
    NUM_DOSE_ATTRS,
    NUM_KERNEL_ATTRS,
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


def build_subset_per_gpu(opt, full_dataset, log):
    n_data = len(full_dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    total_idx    = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu  = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(
        f"[Dataset] Add {n_dump} data to the end to be divided by {n_gpu}. "
        f"Total length={len(total_idx)}!"
    )

    indices = idx_per_gpu.tolist()
    subset  = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset


def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [s.cpu() for s in gathered_samples]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)


def get_recon_imgs_fn(opt, nfe):
    recon_imgs_fn = RESULT_DIR / opt.ckpt / "samples_corrected_nfe{}{}_iter{}".format(
        nfe, "_clip" if opt.clip_denoise else "", opt.load_itr
    )
    os.makedirs(recon_imgs_fn, exist_ok=True)
    return recon_imgs_fn


def compute_batch(ckpt_opt, out):
    """Unpack a batch from the data loader.

    The multi-domain dataset returns:
        (clean_img, corrupt_img, src_dose, src_kernel, trg_dose, trg_kernel)
    """
    global opt
    clean_img, corrupt_img, src_dose, src_kernel, trg_dose, trg_kernel = out

    x1         = corrupt_img.to(opt.device)
    src_dose   = src_dose.to(opt.device)
    src_kernel = src_kernel.to(opt.device)
    trg_dose   = trg_dose.to(opt.device)
    trg_kernel = trg_kernel.to(opt.device)
    cond       = x1.detach() if ckpt_opt.cond_x1 else None

    return clean_img, corrupt_img, x1, cond, src_dose, src_kernel, trg_dose, trg_kernel


@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    nfe      = opt.nfe or ckpt_opt.interval - 1

    # Resolve conversion pairs: prefer command-line override.
    if opt.pair_defs is not None:
        ckpt_opt.pair_defs = opt.pair_defs

    # Propagate fields needed by dataset / runner.
    ckpt_opt.dataset_dir = opt.dataset_dir
    ckpt_opt.image_size  = opt.image_size

    # Ensure domain attr counts are set.
    if not hasattr(ckpt_opt, 'num_dose_attrs') or ckpt_opt.num_dose_attrs is None:
        ckpt_opt.num_dose_attrs = NUM_DOSE_ATTRS
    if not hasattr(ckpt_opt, 'num_kernel_attrs') or ckpt_opt.num_kernel_attrs is None:
        ckpt_opt.num_kernel_attrs = NUM_KERNEL_ATTRS

    # Build test dataset.
    val_dataset = dataset.CTDataset(ckpt_opt, log, mode='test')
    n_samples   = len(val_dataset)
    log.info(f"[Sample] Total test samples: {n_samples}")

    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(
        subset_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
    )

    runner = Runner(ckpt_opt, log, save_opt=False)

    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    num = 0

    for loader_itr, out in enumerate(val_loader):
        (clean_img, corrupt_img, x1, cond,
         src_dose, src_kernel, trg_dose, trg_kernel) = compute_batch(ckpt_opt, out)

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1,
            cond=cond,
            src_dose=src_dose, src_kernel=src_kernel,
            trg_dose=trg_dose, trg_kernel=trg_kernel,
            clip_denoise=opt.clip_denoise,
            nfe=nfe,
            verbose=(opt.n_gpu_per_node == 1),
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape == clean_img.shape

        for i in range(corrupt_img.size(0)):
            sd = DOSE_ID_TO_NAME.get(int(src_dose[i].item()), "?")
            sk = KERNEL_ID_TO_NAME.get(int(src_kernel[i].item()), "?")
            td = DOSE_ID_TO_NAME.get(int(trg_dose[i].item()), "?")
            tk = KERNEL_ID_TO_NAME.get(int(trg_kernel[i].item()), "?")

            # Determine task name as the conversion direction
            task_name = f"{sd}_{sk}_to_{td}_{tk}"

            # Retrieve the patient name and slice name from the dataset
            global_idx = loader_itr * opt.batch_size + i
            patient_name = "unknown_patient"
            group_name = "unknown_group"
            slice_name = f"{global_idx:05}"

            if global_idx < len(subset_dataset):
                dataset_idx = subset_dataset.indices[global_idx]
                src_path = val_dataset.pairs[dataset_idx][0]
                src_p = Path(src_path)
                slice_name = src_p.stem
                if len(src_p.parts) >= 5:
                    patient_name = src_p.parent.parent.name
                try:
                    rel_p = src_p.relative_to(val_dataset.dataset_dir)
                    group_name = rel_p.parts[0]
                except ValueError:
                    if len(src_p.parts) >= 6:
                        group_name = src_p.parent.parent.parent.parent.name

            # Group images into task, group, and patient directories
            save_dir = recon_imgs_fn / task_name / group_name / patient_name
            os.makedirs(save_dir, exist_ok=True)

            tu.save_image(
                (corrupt_img[i:i+1, ...] + 1) / 2,
                save_dir / f"{slice_name}_source.png",
            )
            tu.save_image(
                (clean_img[i:i+1, ...] + 1) / 2,
                save_dir / f"{slice_name}_target.png",
            )
            tu.save_image(
                (recon_img[i:i+1, ...] + 1) / 2,
                save_dir / f"{slice_name}_target_recon.png",
            )
            log.info(f"Saved output images! pair=({sd},{sk})→({td},{tk}) paths: {save_dir}/{slice_name}")

        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        log.info(
            f"Max memory allocated: {torch.cuda.max_memory_allocated(opt.device)/2**30:.2f} GB"
        )
        dist.barrier()

    del runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr}, recon_imgs_fn / "recon.pt")
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,  default=0)
    parser.add_argument("--n-gpu-per-node",     type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address",     type=str,  default='localhost', help="address for master")
    parser.add_argument("--master-port",        type=str,  default='6020',      help="port for master")
    parser.add_argument("--node-rank",          type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",      type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",         type=int,  default=512)
    parser.add_argument("--dataset-dir",        type=Path, default="./data",    help="path to dataset")
    parser.add_argument("--pair-defs",          type=str,  nargs='+', default=None, metavar="PAIR",
                        help="Conversion pairs: 'SrcDose,SrcKernel:TrgDose,TrgKernel'. Overrides checkpoint.")

    # sample
    parser.add_argument("--load-itr",           type=int,  default=150000)
    parser.add_argument("--batch-size",         type=int,  default=1)
    parser.add_argument("--ckpt",               type=str,  default=None,        help="checkpoint name to sample from")
    parser.add_argument("--nfe",                type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",       action="store_true",            help="clamp predicted image to [-1,1] at each step")
    parser.add_argument("--use-fp16",           action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--noise-feat-dim",     type=int,  default=20,          help="Dimension of noise statistics feature")
    parser.add_argument("--embedding-mode",     type=str,  default="base_cross_noise", choices=["base", "base_cross", "base_cross_noise"], help="Select embedding mode")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    set_seed(opt.seed)

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
            p = Process(
                target=dist_util.init_processes,
                args=(global_rank, global_size, main, opt),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank  = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
