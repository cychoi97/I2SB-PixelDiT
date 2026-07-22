# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch
from argparse import Namespace

from pathlib import Path
from easydict import EasyDict as edict

from ipdb import set_trace as debug


def build_ckpt_option(opt, log, ckpt_path):
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    log.info(f"Loaded options from {opt_pkl_path}!")

    overwrite_keys = ["use_fp16", "device"]
    for k in overwrite_keys:
        assert hasattr(opt, k)
        setattr(ckpt_opt, k, getattr(opt, k))

    ckpt_opt.load = ckpt_path / f"{opt.load_itr:07}.pt"
    return ckpt_opt
