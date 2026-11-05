import os
import copy
import math
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

import distributed_util as dist_util

from . import util
from .network import PixDiTNet
from .diffusion import Diffusion
from .fd_loss import (
    FeatureQueue,
    MultiDomainFeatureQueue,
    compute_frechet_distance_loss,
    load_fd_reference_stats,
    precompute_sigma_ref_sqrt,
    DEFAULT_TARGET_DOMAINS,
    DEFAULT_TARGET_DOMAIN_KEYS,
)
from dataset.dataset import (
    DOSE_ATTRS,
    KERNEL_ATTRS,
    DOSE_ID_TO_NAME,
    KERNEL_ID_TO_NAME,
    NUM_DOSE_ATTRS,
    NUM_KERNEL_ATTRS,
    DEFAULT_PAIRS,
    pair_label,
)

from ipdb import set_trace as debug

# ImageNet normalisation constants used by DINOv2.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_optimizer_sched(opt, net, log, repa_proj=None):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict}!")

    if repa_proj is not None:
        optimizer.add_param_group({'params': repa_proj.parameters(), 'lr': opt.lr})
        log.info(f"[Opt] Added repa_proj parameter group to optimizer!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
            except Exception as e:
                log.warning(f"[Opt] Could not load optimizer checkpoint ({e}). Starting with a fresh optimizer state.")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            try:
                sched.load_state_dict(checkpoint["sched"])
                log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
            except Exception as e:
                log.warning(f"[Opt] Could not load lr sched checkpoint ({e}).")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def all_cat_cpu(opt, log, t):
    if not opt.distributed:
        return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()


# ---------------------------------------------------------------------------
# Frozen DINOv2 encoder for REPA alignment loss.
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 for REPA-style representation alignment.

    Matches the original PixelDiT REPATrainer implementation:
      - Resize input so that each model patch_size maps to DINOv2's 14×14.
      - Return ``x_norm_patchtokens`` (normalized patch token features).

    Input contract:
        x : [B, C_img, H, W]  float32 in [0, 1]  (clean image, already scaled).
    """

    def __init__(self, model_name: str = 'dinov2_vitb14', base_patch_size: int = 16):
        super().__init__()
        try:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                trust_repo=True,
            )
        except Exception as e:
            print(f"Failed to load DINOv2 from cache ({e}). Forcing reload...")
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                trust_repo=True,
                force_reload=True,
            )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.base_patch_size = base_patch_size

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features.

        Args:
            x : [B, C, H, W] in [-1, 1].

        Returns:
            [B, N_patches, feat_dim] patch token features.
        """
        x01 = (x + 1.0) / 2.0
        if x01.shape[1] == 1:
            x01 = x01.repeat(1, 3, 1, 1)
        b, c, h, w = x01.shape
        # Resize so that each base_patch_size maps to 14 pixels (DINOv2 patch).
        target_h = int(14 * h / self.base_patch_size)
        target_w = int(14 * w / self.base_patch_size)
        x01 = F.interpolate(x01, (target_h, target_w), mode='bicubic', align_corners=False)
        x_norm = (x01 - self.mean.to(x01)) / self.std.to(x01)
        features = self.encoder.forward_features(x_norm)
        return features['x_norm_patchtokens'].to(torch.float32)


# ---------------------------------------------------------------------------
# REPA Projector MLP — maps denoiser hidden features to DINOv2 feature space.
# ---------------------------------------------------------------------------

class REPAProjector(nn.Module):
    """Learnable MLP that projects denoiser intermediate features
    into the DINOv2 representation space for cosine alignment.

    Architecture follows the original REPATrainer:
        Linear → SiLU → Linear → SiLU → Linear
    """

    def __init__(self, denoiser_dim: int, hidden_dim: int = 256, encoder_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(denoiser_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Noise Statistics Extraction
# ---------------------------------------------------------------------------

def _haar_wavelet_2d(x: torch.Tensor):
    """Single-level 2D Haar wavelet decomposition via tensor slicing.

    Args:
        x: [B, 1, H, W] tensor (H and W must be even).

    Returns:
        LL, LH, HL, HH: each [B, 1, H//2, W//2].
    """
    r0 = x[:, :, 0::2, :]  # even rows
    r1 = x[:, :, 1::2, :]  # odd rows
    LL = (r0[:, :, :, 0::2] + r0[:, :, :, 1::2] + r1[:, :, :, 0::2] + r1[:, :, :, 1::2]) * 0.5
    LH = (r0[:, :, :, 0::2] + r0[:, :, :, 1::2] - r1[:, :, :, 0::2] - r1[:, :, :, 1::2]) * 0.5
    HL = (r0[:, :, :, 0::2] - r0[:, :, :, 1::2] + r1[:, :, :, 0::2] - r1[:, :, :, 1::2]) * 0.5
    HH = (r0[:, :, :, 0::2] - r0[:, :, :, 1::2] - r1[:, :, :, 0::2] + r1[:, :, :, 1::2]) * 0.5
    return LL, LH, HL, HH


def compute_noise_stats(x: torch.Tensor, patch_size: int = 16, target_dim: int = 20) -> torch.Tensor:
    """Extracts a noise statistics vector from input tensor x.

    20 non-redundant features organized into 5 categories:

    A. Global statistics (4):
        g_mean, g_std, skewness, kurtosis
    B. Spatial noise variation (3):
        p_mean_std, p_std_mean, p_std_std
    C. Haar Wavelet Level 1 (7):
        LH1_std, HL1_std, HH1_std, HH1_mad_sigma,
        HH1_to_LL1_energy, directional_aniso, HH1_kurtosis
    D. Haar Wavelet Level 2 (4):
        LH2_std, HL2_std, HH2_std, HH2_to_HH1_ratio
    E. Cross-domain ratios (2):
        noise_consistency, HH1_skewness

    Args:
        x: [B, C, H, W] input image tensor.
        patch_size: patch size for spatial statistics.
        target_dim: output dimension (padded/clipped to match).

    Returns:
        [B, target_dim] noise statistics tensor.
    """
    B, C, H, W = x.shape
    device = x.device
    eps = 1e-8

    # Convert to grayscale
    x_gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # === A. Global statistics (4) ===
    g_mean = x_gray.mean(dim=[2, 3])  # [B, 1]
    g_std = x_gray.std(dim=[2, 3], unbiased=False)  # [B, 1]

    x_flat = x_gray.view(B, -1)  # [B, H*W]
    x_centered = x_flat - g_mean
    variance = g_std ** 2 + eps
    skewness = (x_centered ** 3).mean(dim=1, keepdim=True) / (variance ** 1.5)
    kurtosis = (x_centered ** 4).mean(dim=1, keepdim=True) / (variance ** 2)

    # === B. Spatial noise variation (3) ===
    patches = F.unfold(x_gray, kernel_size=patch_size, stride=patch_size)  # [B, P*P, N]
    patch_means = patches.mean(dim=1)  # [B, N]
    patch_stds = patches.std(dim=1, unbiased=False)  # [B, N]

    p_mean_std = patch_means.std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    p_std_mean = patch_stds.mean(dim=1, keepdim=True)  # [B, 1]
    p_std_std = patch_stds.std(dim=1, keepdim=True, unbiased=False)  # [B, 1]

    # === Haar Wavelet Decomposition ===
    # Ensure even dimensions for Haar transform
    h_even = (H // 2) * 2
    w_even = (W // 2) * 2
    x_even = x_gray[:, :, :h_even, :w_even]

    # Level 1
    LL1, LH1, HL1, HH1 = _haar_wavelet_2d(x_even)

    # Level 2 (apply to LL1, ensure even dimensions again)
    h2 = (LL1.shape[2] // 2) * 2
    w2 = (LL1.shape[3] // 2) * 2
    LL1_even = LL1[:, :, :h2, :w2]
    LL2, LH2, HL2, HH2 = _haar_wavelet_2d(LL1_even)

    # === C. Haar Wavelet Level 1 (7) ===
    LH1_std = LH1.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    HL1_std = HL1.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    HH1_std = HH1.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]

    # MAD-based robust noise estimate: sigma = median(|HH|) / 0.6745
    HH1_flat = HH1.view(B, -1)
    HH1_abs = HH1_flat.abs()
    HH1_mad_sigma = (HH1_abs.median(dim=1, keepdim=True)[0] / 0.6745)  # [B, 1]

    # Noise-to-signal energy ratio
    LL1_energy = LL1.view(B, -1).pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    HH1_energy = HH1_flat.pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    HH1_to_LL1_energy = HH1_energy / (LL1_energy + eps)  # [B, 1]

    # Directional anisotropy: isotropic noise → LH ≈ HL → aniso ≈ 0
    LH1_energy = LH1.view(B, -1).pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    HL1_energy = HL1.view(B, -1).pow(2).mean(dim=1, keepdim=True)  # [B, 1]
    directional_aniso = (LH1_energy - HL1_energy).abs() / (LH1_energy + HL1_energy + eps)

    # HH1 kurtosis (excess kurtosis of diagonal detail coefficients)
    HH1_mean = HH1_flat.mean(dim=1, keepdim=True)
    HH1_centered = HH1_flat - HH1_mean
    HH1_var = HH1_centered.pow(2).mean(dim=1, keepdim=True) + eps
    HH1_kurtosis = HH1_centered.pow(4).mean(dim=1, keepdim=True) / (HH1_var ** 2)

    # === D. Haar Wavelet Level 2 (4) ===
    LH2_std = LH2.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    HL2_std = HL2.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
    HH2_std = HH2.view(B, -1).std(dim=1, keepdim=True, unbiased=False)  # [B, 1]

    # Cross-scale ratio: noise is scale-invariant, structure is not
    HH2_to_HH1_ratio = HH2_std / (HH1_std + eps)  # [B, 1]

    # === E. Cross-domain ratios (2) ===
    # Wavelet noise estimate vs global std agreement
    noise_consistency = HH1_mad_sigma / (g_std + eps)  # [B, 1]

    # HH1 skewness (non-Gaussian noise detection)
    HH1_skewness = HH1_centered.pow(3).mean(dim=1, keepdim=True) / (HH1_var ** 1.5)

    # === Concatenate all features ===
    feats = [
        # A. Global statistics (4)
        g_mean, g_std, skewness, kurtosis,
        # B. Spatial noise variation (3)
        p_mean_std, p_std_mean, p_std_std,
        # C. Haar Wavelet Level 1 (7)
        LH1_std, HL1_std, HH1_std, HH1_mad_sigma,
        HH1_to_LL1_energy, directional_aniso, HH1_kurtosis,
        # D. Haar Wavelet Level 2 (4)
        LH2_std, HL2_std, HH2_std, HH2_to_HH1_ratio,
        # E. Cross-domain ratios (2)
        noise_consistency, HH1_skewness,
    ]

    noise_stats = torch.cat(feats, dim=-1)  # [B, 20]

    # Pad or clip to match target_dim
    if noise_stats.shape[-1] < target_dim:
        pad_dim = target_dim - noise_stats.shape[-1]
        noise_stats = F.pad(noise_stats, (0, pad_dim), "constant", 0.0)
    elif noise_stats.shape[-1] > target_dim:
        noise_stats = noise_stats[:, :target_dim]

    return noise_stats


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner(object):
    """Multi-domain I2SB runner with PixDiT backbone + DINOv2 feature loss.

    Full multi-domain conditioning: every sample carries (dose, kernel)
    tuples for both source and target, enabling the network to condition
    on ALL domains regardless of which is being converted.

    Dataset returns:
        (clean, corrupt, src_dose_id, src_kernel_id, trg_dose_id, trg_kernel_id)
    """

    def __init__(self, opt, log, save_opt=True):
        super().__init__()

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        # ---- I2SB diffusion ----
        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval // 2], np.flip(betas[:opt.interval // 2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        self.net = PixDiTNet(
            log,
            noise_levels=noise_levels,
            in_channels=opt.in_channels,
            cond=opt.cond_x1,
            num_dose_attrs=getattr(opt, 'num_dose_attrs', NUM_DOSE_ATTRS),
            num_kernel_attrs=getattr(opt, 'num_kernel_attrs', NUM_KERNEL_ATTRS),
            hidden_size=getattr(opt, 'hidden_size', 384),
            pixel_hidden_size=getattr(opt, 'pixel_hidden_size', 64),
            patch_depth=getattr(opt, 'patch_depth', 6),
            pixel_depth=getattr(opt, 'pixel_depth', 4),
            num_groups=getattr(opt, 'num_groups', 8),
            patch_size=getattr(opt, 'patch_size', 2),
            noise_feat_dim=getattr(opt, 'noise_feat_dim', 20),
            embedding_mode=getattr(opt, 'embedding_mode', 'base_cross_noise'),
        )

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            missing_keys, unexpected_keys = self.net.load_state_dict(checkpoint['net'], strict=False)
            log.info(f"[Net] Loaded network ckpt: {opt.load}! Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            if "ema" in checkpoint:
                old_param_names = [name for name, _ in self.net.named_parameters() if name in checkpoint["net"]]
                if len(old_param_names) == len(checkpoint["ema"]["shadow_params"]):
                    ema_name_to_tensor = {
                        name: tensor for name, tensor in zip(old_param_names, checkpoint["ema"]["shadow_params"])
                    }
                    new_shadow_params = []
                    for name, param in self.net.named_parameters():
                        if name in ema_name_to_tensor:
                            new_shadow_params.append(ema_name_to_tensor[name].to(device=param.device, dtype=param.dtype))
                        else:
                            new_shadow_params.append(param.clone().detach())
                    self.ema.shadow_params = new_shadow_params
                    self.ema.decay = checkpoint["ema"].get("decay", self.ema.decay)
                    self.ema.num_updates = checkpoint["ema"].get("num_updates", self.ema.num_updates)
                    log.info(f"[Ema] Loaded ema ckpt manually: {opt.load}!")
                else:
                    log.warning("[Ema] Mismatch in parameter count between old model and EMA checkpoint. EMA initialized from net parameters.")
            else:
                log.warning("[Ema] No EMA dict found in checkpoint!")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.log = log

        # REPA: DINOv2 encoder (frozen) + learnable projector.
        self.dino = None
        self.repa_proj = None
        self.align_layer = getattr(opt, 'align_layer', 8)
        feat_w = getattr(opt, 'feat_loss_weight', 0.0)
        fd_w   = getattr(opt, 'fd_loss_weight',   0.0)

        # DINOv2 is shared by REPA and FD-Loss — load once if either is active.
        if feat_w > 0.0 or fd_w > 0.0:
            dino_model = getattr(opt, 'dino_model', 'dinov2_vitb14')
            base_patch_size = getattr(opt, 'patch_size', 16)
            # DINOv2 ViT-B/14 → 768, ViT-L/14 → 1024, ViT-g/14 → 1536
            encoder_dim = {'dinov2_vitb14': 768, 'dinov2_vitl14': 1024,
                           'dinov2_vitg14': 1536}.get(dino_model, 768)
            denoiser_dim = getattr(opt, 'hidden_size', 1024)
            try:
                self.dino = DINOv2Encoder(
                    model_name=dino_model, base_patch_size=base_patch_size,
                )
                self.dino.to(opt.device)
                if feat_w > 0.0:
                    self.repa_proj = REPAProjector(
                        denoiser_dim=denoiser_dim,
                        hidden_dim=256,
                        encoder_dim=encoder_dim,
                    )
                    self.repa_proj.to(opt.device)
                    log.info(
                        f"[REPA] Loaded frozen {dino_model} + projector "
                        f"(align_layer={self.align_layer}, weight={feat_w}, "
                        f"denoiser_dim={denoiser_dim}→encoder_dim={encoder_dim})"
                    )
                else:
                    log.info(
                        f"[DINOv2] Loaded frozen {dino_model} for FD-Loss "
                        f"(feat_dim={encoder_dim})"
                    )
            except Exception as e:
                log.warning(f"[REPA/FD] Failed to init DINOv2: {e}. Feature losses disabled.")
                self.dino = None
                self.repa_proj = None

        # ------------------------------------------------------------------
        # FD-Loss: FeatureQueue + reference statistics
        # ------------------------------------------------------------------
        self.fd_queue          = None
        self.fd_mu_ref         = None
        self.fd_sigma_ref      = None
        self.fd_ref_stats      = None
        self.fd_sigma_ref_sqrt = None
        self.is_multi_fd       = False

        if fd_w > 0.0:
            fd_stats_path = getattr(opt, 'fd_stats_path', None)
            fd_queue_size = getattr(opt, 'fd_queue_size', 5000)
            fd_ema_beta   = getattr(opt, 'fd_ema_beta', 0.999)
            dino_model    = getattr(opt, 'dino_model', 'dinov2_vitb14')
            fd_feat_dim   = {'dinov2_vitb14': 768, 'dinov2_vitl14': 1024,
                             'dinov2_vitg14': 1536}.get(dino_model, 768)

            if fd_stats_path is None:
                log.warning(
                    "[FD-Loss] fd_loss_weight > 0 but --fd-stats-path not provided. "
                    "FD-Loss will be disabled."
                )
            elif self.dino is None:
                log.warning(
                    "[FD-Loss] DINOv2 not loaded. FD-Loss will be disabled."
                )
            else:
                ref_stats = load_fd_reference_stats(
                    fd_stats_path, device=opt.device
                )

                if isinstance(ref_stats, dict):
                    # Multi-domain target-conditioned mode
                    self.is_multi_fd = True
                    self.fd_ref_stats = ref_stats
                    domain_keys = list(ref_stats.keys())
                    self.fd_queue = MultiDomainFeatureQueue(
                        domain_keys=domain_keys,
                        size=fd_queue_size,
                        feat_dim=fd_feat_dim,
                        online_accum=True,
                        ema_beta=fd_ema_beta,
                    ).to(opt.device)

                    if getattr(opt, 'fd_eigvalsh', False):
                        self.fd_sigma_ref_sqrt = precompute_sigma_ref_sqrt(
                            self.fd_ref_stats
                        )
                        log.info(
                            f"[FD-Loss] Precomputed σ_ref^{{1/2}} for {len(self.fd_ref_stats)} target domains."
                        )
                    log.info(
                        f"[FD-Loss] Initialized Target-Conditioned MultiDomainFeatureQueue: "
                        f"domains={domain_keys}, ema_beta={fd_ema_beta}, feat_dim={fd_feat_dim}"
                    )
                else:
                    # Single-domain fallback mode
                    self.is_multi_fd = False
                    self.fd_mu_ref, self.fd_sigma_ref = ref_stats
                    self.fd_queue = FeatureQueue(
                        size=fd_queue_size,
                        feat_dim=fd_feat_dim,
                        online_accum=True,
                        ema_beta=fd_ema_beta,
                    ).to(opt.device)

                    if getattr(opt, 'fd_eigvalsh', False):
                        self.fd_sigma_ref_sqrt = precompute_sigma_ref_sqrt(
                            self.fd_sigma_ref
                        )
                        log.info("[FD-Loss] Precomputed σ_ref^{1/2} for single-domain mode.")
                    log.info(
                        f"[FD-Loss] Initialized single-domain FeatureQueue: "
                        f"ema_beta={fd_ema_beta}, feat_dim={fd_feat_dim}"
                    )

                # Restore queue state from checkpoint if available
                if opt.load:
                    checkpoint = torch.load(opt.load, map_location="cpu")
                    if "fd_queue" in checkpoint:
                        try:
                            self.fd_queue.load_state_dict(checkpoint["fd_queue"])
                            self.fd_queue.to(opt.device)
                            log.info("[FD-Loss] Restored FeatureQueue state from checkpoint.")
                        except Exception as e:
                            log.warning(f"[FD-Loss] Could not restore fd_queue from checkpoint ({e}).")

    # ------------------------------------------------------------------
    # Label computation
    # ------------------------------------------------------------------

    def compute_label(self, step, x0, xt):
        """Eq 12 of I2SB paper."""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        return (xt - x0) / std_fwd

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1., 1.)
        return pred_x0

    # ------------------------------------------------------------------
    # FD-Loss helper: DINOv2 CLS-token extraction with gradient support
    # ------------------------------------------------------------------

    def _extract_fd_features(self, x_01: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 CLS-token features while preserving gradients.

        Unlike DINOv2Encoder.forward() (which is decorated with @torch.no_grad),
        this method directly calls the underlying encoder so that gradients
        can flow from the FD loss back through the predicted x0 to the network.
        DINOv2 parameters remain frozen (requires_grad=False); only the
        activations carry gradients.

        Args:
            x_01: [B, 1 or 3, H, W] float32 in [0, 1].

        Returns:
            [B, feat_dim] float32 CLS-token feature tensor (with grad).
        """
        # Ensure 3-channel input for DINOv2
        if x_01.shape[1] == 1:
            x_01 = x_01.repeat(1, 3, 1, 1)

        dino_enc = self.dino
        b, c, h, w = x_01.shape

        # Resize: each base_patch_size pixel → 14 DINOv2 pixels
        target_h = int(14 * h / dino_enc.base_patch_size)
        target_w = int(14 * w / dino_enc.base_patch_size)
        x_resized = F.interpolate(
            x_01, (target_h, target_w), mode="bicubic", align_corners=False
        )

        # ImageNet normalization (same as DINOv2Encoder)
        x_norm = (x_resized - dino_enc.mean.to(x_resized)) / dino_enc.std.to(x_resized)

        # Forward through encoder WITHOUT @torch.no_grad() — gradients preserved
        feats = dino_enc.encoder.forward_features(x_norm)
        cls_feat = feats["x_norm_clstoken"].to(torch.float32)  # [B, feat_dim]
        return cls_feat

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def sample_batch(self, opt, loader):
        """Draw one mini-batch from the multi-domain data loader.

        Returns:
            x0         : [B, C, H, W] clean target tensor.
            x1         : [B, C, H, W] degraded source tensor.
            cond       : x1 or None depending on cond_x1 flag.
            src_dose   : [B] long tensor on device.
            src_kernel : [B] long tensor on device.
            trg_dose   : [B] long tensor on device.
            trg_kernel : [B] long tensor on device.
        """
        clean_img, corrupt_img, src_dose, src_kernel, trg_dose, trg_kernel = next(loader)

        x0         = clean_img.detach().to(opt.device)
        x1         = corrupt_img.detach().to(opt.device)
        src_dose   = src_dose.to(opt.device)
        src_kernel = src_kernel.to(opt.device)
        trg_dose   = trg_dose.to(opt.device)
        trg_kernel = trg_kernel.to(opt.device)

        cond = x1.detach() if opt.cond_x1 else None

        assert x0.shape == x1.shape
        return x0, x1, cond, src_dose, src_kernel, trg_dose, trg_kernel

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        find_unused = getattr(opt, 'embedding_mode', 'base_cross_noise') != 'base_cross_noise'
        net = DDP(self.net, device_ids=[opt.device], find_unused_parameters=find_unused)
        ema = self.ema

        # Collect all trainable parameters: net + REPA projector.
        all_params = list(net.parameters())
        if self.repa_proj is not None:
            all_params += list(self.repa_proj.parameters())
        optimizer, sched = build_optimizer_sched(opt, net, log, repa_proj=self.repa_proj)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        net.train()

        feat_w        = getattr(opt, 'feat_loss_weight', 0.0)
        feat_interval = getattr(opt, 'feat_loss_interval', 1)

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        for it in range(opt.start_itr, opt.num_itr + 1):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, cond, src_dose, src_kernel, trg_dose, trg_kernel = \
                    self.sample_batch(opt, train_loader)

                # ===== compute noise statistics if embedding_mode includes noise =====
                noise_stats = None
                if getattr(opt, 'embedding_mode', 'base_cross_noise') == 'base_cross_noise':
                    noise_stats = compute_noise_stats(
                        x1,
                        patch_size=16,
                        target_dim=getattr(opt, 'noise_feat_dim', 20)
                    )

                # ===== I2SB diffusion training step =====
                step  = torch.randint(0, opt.interval, (x0.shape[0],))
                xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                # ===== REPA: register forward hook to capture intermediate features =====
                src_feature = []
                hook_handle = None
                if (self.dino is not None
                        and self.repa_proj is not None
                        and feat_w > 0.0
                        and (it % feat_interval == 0)):
                    def _forward_hook(module, input, output):
                        feat = output
                        if isinstance(feat, tuple):
                            feat = feat[0]
                        src_feature.append(feat)

                    # Access patch_blocks through DDP → PixDiTNet → PixDiT.
                    pixdit_model = net.module.model
                    hook_handle = pixdit_model.patch_blocks[
                        self.align_layer - 1
                    ].register_forward_hook(_forward_hook)

                pred = net(
                    xt, step, cond=cond,
                    src_dose=src_dose, src_kernel=src_kernel,
                    trg_dose=trg_dose, trg_kernel=trg_kernel,
                    noise_stats=noise_stats,
                )
                assert xt.shape == label.shape == pred.shape

                # I2SB: noise-prediction MSE loss
                loss_diff = F.mse_loss(pred, label)

                # ===== REPA alignment loss =====
                loss_feat = torch.tensor(0.0, device=opt.device)
                if hook_handle is not None:
                    hook_handle.remove()
                    if len(src_feature) > 0:
                        # Project denoiser features to DINOv2 space.
                        proj_feat = self.repa_proj(src_feature[0])  # [B, N, encoder_dim]

                        # DINOv2 features from clean target image.
                        with torch.no_grad():
                            dst_feat = self.dino(x0)  # [B, M, encoder_dim]

                        # Handle spatial resolution mismatch.
                        if dst_feat.shape[1] != proj_feat.shape[1]:
                            bsz, ls, ch = proj_feat.shape
                            ld = dst_feat.shape[1]
                            hs = int(ls ** 0.5)
                            hd = int(ld ** 0.5)
                            src_spatial = proj_feat.view(bsz, hs, hs, ch).permute(0, 3, 1, 2)
                            if ld < ls:
                                resized = F.adaptive_avg_pool2d(src_spatial, (hd, hd))
                            else:
                                resized = F.interpolate(
                                    src_spatial, size=(hd, hd),
                                    mode='bilinear', align_corners=False,
                                )
                            proj_feat = resized.permute(0, 2, 3, 1).reshape(bsz, ld, ch)

                        cos_sim = F.cosine_similarity(proj_feat, dst_feat, dim=-1)
                        loss_feat = (1.0 - cos_sim).mean()

                # ===== FD-Loss: distribution matching in DINOv2 CLS-token space =====
                loss_fd = torch.tensor(0.0, device=opt.device)
                fd_w_cur        = getattr(opt, 'fd_loss_weight',   0.0)
                fd_interval_cur = getattr(opt, 'fd_loss_interval', 1)
                fd_norm_eps     = getattr(opt, 'fd_norm_eps',       0.01)

                if (self.fd_queue is not None
                        and fd_w_cur > 0.0
                        and it % fd_interval_cur == 0):

                    # Compute predicted x0 (restored CT image) from current step.
                    # pred_x0 carries gradients via pred → net → model parameters.
                    pred_x0 = self.compute_pred_x0(
                        step, xt, pred.detach() if not xt.requires_grad else pred,
                        clip_denoise=opt.clip_denoise,
                    )
                    # Re-derive pred_x0 with gradient from pred
                    std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
                    pred_x0_grad = xt.detach() - std_fwd * pred  # gradient through pred
                    if opt.clip_denoise:
                        # clamp preserves gradient (straight-through-style)
                        pred_x0_grad = pred_x0_grad.clamp(-1.0, 1.0)

                    # Convert [-1, 1] → [0, 1] for DINOv2
                    pred_x0_01 = (pred_x0_grad + 1.0) / 2.0

                    # Extract DINOv2 CLS features WITH gradient flow
                    cls_feat = self._extract_fd_features(pred_x0_01)  # [B, D]

                    if getattr(self, 'is_multi_fd', False):
                        sample_domain_keys = [
                            f"{DOSE_ID_TO_NAME[int(td)]}_{KERNEL_ID_TO_NAME[int(tk)]}"
                            for td, tk in zip(trg_dose.cpu(), trg_kernel.cpu())
                        ]
                        loss_fd, _ = self.fd_queue.compute_target_fd_loss(
                            cls_feat=cls_feat,
                            sample_domain_keys=sample_domain_keys,
                            ref_stats=self.fd_ref_stats,
                            ref_sigma_sqrt=self.fd_sigma_ref_sqrt,
                            norm_eps=fd_norm_eps,
                        )
                    else:
                        # Single-domain fallback
                        mu_gen, sigma_gen = self.fd_queue.build_feats_stats(cls_feat)
                        loss_fd_raw = compute_frechet_distance_loss(
                            self.fd_mu_ref, self.fd_sigma_ref,
                            mu=mu_gen, sigma=sigma_gen,
                            sigma_ref_sqrt=self.fd_sigma_ref_sqrt,
                        )
                        loss_fd = loss_fd_raw / (loss_fd_raw.detach() + fd_norm_eps)

                loss = loss_diff + feat_w * loss_feat + fd_w_cur * loss_fd
                loss.backward()

                # Update FD queue with detached features AFTER backward
                if (self.fd_queue is not None
                        and fd_w_cur > 0.0
                        and it % fd_interval_cur == 0):
                    if getattr(self, 'is_multi_fd', False):
                        self.fd_queue.enqueue(cls_feat.detach(), sample_domain_keys)
                    else:
                        self.fd_queue.enqueue(cls_feat.detach())

            optimizer.step()
            ema.update()
            if sched is not None:
                sched.step()

            # -------- logging --------
            vram_alloc = torch.cuda.memory_allocated(opt.device) / (1024 ** 3)
            vram_reserved = torch.cuda.memory_reserved(opt.device) / (1024 ** 3)
            log.info(
                "train_it {}/{} | lr:{} | loss_diff:{} | loss_feat:{} | loss_fd:{} | loss:{} | VRAM:{:.2f}GB (reserved:{:.2f}GB)".format(
                    1 + it,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    "{:+.4f}".format(loss_diff.item()),
                    "{:+.4f}".format(loss_feat.item()),
                    "{:+.4f}".format(loss_fd.item()),
                    "{:+.4f}".format(loss.item()),
                    vram_alloc,
                    vram_reserved,
                )
            )
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss/total', loss.detach())
                self.writer.add_scalar(it, 'loss/diff',  loss_diff.detach())
                self.writer.add_scalar(it, 'loss/feat',  loss_feat.detach())
                self.writer.add_scalar(it, 'loss/fd',    loss_fd.detach())

            if it % 5000 == 0 or it == opt.num_itr:
                if opt.global_rank == 0:
                    ckpt_dict = {
                        "net":       self.net.state_dict(),
                        "ema":       ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched":     sched.state_dict() if sched is not None else sched,
                    }
                    if self.repa_proj is not None:
                        ckpt_dict["repa_proj"] = self.repa_proj.state_dict()
                    # Save FD queue state for warm-restart
                    if self.fd_queue is not None:
                        ckpt_dict["fd_queue"] = self.fd_queue.state_dict()
                    torch.save(ckpt_dict, opt.ckpt_path / f"{it:07}.pt")
                    log.info(f"Saved latest(it={it}) checkpoint to {opt.ckpt_path}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 5000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, opt.ckpt_path)
                net.train()

        self.writer.close()

    # ------------------------------------------------------------------
    # DDPM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddpm_sampling(
        self,
        opt,
        x1,
        cond=None,
        src_dose=None,
        src_kernel=None,
        trg_dose=None,
        trg_kernel=None,
        clip_denoise=False,
        nfe=None,
        log_count=10,
        verbose=True,
    ):
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe + 1)

        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        self.log.info(
            f"[DDPM Sampling] steps={opt.interval}, nfe={nfe}, log steps={log_steps}!"
        )

        x1 = x1.to(opt.device)
        if cond       is not None: cond       = cond.to(opt.device)
        if src_dose   is not None: src_dose   = src_dose.to(opt.device)
        if src_kernel is not None: src_kernel = src_kernel.to(opt.device)
        if trg_dose   is not None: trg_dose   = trg_dose.to(opt.device)
        if trg_kernel is not None: trg_kernel = trg_kernel.to(opt.device)

        noise_stats = None
        if getattr(opt, 'embedding_mode', 'base_cross_noise') == 'base_cross_noise':
            noise_stats = compute_noise_stats(
                x1,
                patch_size=16,
                target_dim=getattr(opt, 'noise_feat_dim', 20)
            )

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out  = self.net(
                    xt, step, cond=cond,
                    src_dose=src_dose, src_kernel=src_kernel,
                    trg_dose=trg_dose, trg_kernel=trg_kernel,
                    noise_stats=noise_stats,
                )
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, ot_ode=opt.ot_ode,
                log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)
        return xs, pred_x0

    # ------------------------------------------------------------------
    # Evaluation with per-pair metrics (framework-aware)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, ckpt_path):
        image_path = ckpt_path / 'valid_images'
        os.makedirs(image_path, exist_ok=True)

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        x0, x1, cond, src_dose, src_kernel, trg_dose, trg_kernel = \
            self.sample_batch(opt, val_loader)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, cond=cond,
            src_dose=src_dose, src_kernel=src_kernel,
            trg_dose=trg_dose, trg_kernel=trg_kernel,
            clip_denoise=opt.clip_denoise, nfe=2, verbose=(opt.global_rank == 0),
        )

        log.info("Collecting tensors ...")
        img_clean       = all_cat_cpu(opt, log, x0)
        img_corrupt     = all_cat_cpu(opt, log, x1)
        src_dose_cpu    = all_cat_cpu(opt, log, src_dose)
        src_kernel_cpu  = all_cat_cpu(opt, log, src_kernel)
        trg_dose_cpu    = all_cat_cpu(opt, log, trg_dose)
        trg_kernel_cpu  = all_cat_cpu(opt, log, trg_kernel)
        pred_x0s        = all_cat_cpu(opt, log, pred_x0s)
        xs              = all_cat_cpu(opt, log, xs)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img + 1) / 2, nrow=nrow))

        def log_metric(tag, score):
            self.writer.add_scalar(it, tag, score)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),       nrow=len_t)

        tu.save_image(
            tu.make_grid((img_clean + 1) / 2, nrow=10),
            os.path.join(image_path, f'{it:07}_target.png')
        )
        tu.save_image(
            tu.make_grid((img_corrupt + 1) / 2, nrow=10),
            os.path.join(image_path, f'{it:07}_source.png')
        )
        tu.save_image(
            tu.make_grid((img_recon + 1) / 2, nrow=10),
            os.path.join(image_path, f'{it:07}_target_recon.png')
        )
        tu.save_image(
            tu.make_grid((pred_x0s.reshape(-1, *xdim) + 1) / 2, nrow=len_t),
            os.path.join(image_path, f'{it:07}_pred_target_traj.png')
        )
        tu.save_image(
            tu.make_grid((xs.reshape(-1, *xdim) + 1) / 2, nrow=len_t),
            os.path.join(image_path, f'{it:07}_target_recon_traj.png')
        )

        # ---- metrics ----
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        all_psnr, all_ssim = [], []
        per_pair = {}  # keyed by (src_dose, src_kernel, trg_dose, trg_kernel)

        for i in range(img_clean.size(0)):
            clean_hu = (np.array(img_clean[i, 0, ...]) + 1) / 2 * 4095 - 1024
            recon_hu = (np.array(img_recon[i, 0, ...]) + 1) / 2 * 4095 - 1024

            psnr_val = float(psnr(clean_hu, recon_hu, data_range=4096))
            ssim_val = float(ssim(clean_hu, recon_hu, data_range=4096))

            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            sd = DOSE_ID_TO_NAME.get(int(src_dose_cpu[i].item()), "?")
            sk = KERNEL_ID_TO_NAME.get(int(src_kernel_cpu[i].item()), "?")
            td = DOSE_ID_TO_NAME.get(int(trg_dose_cpu[i].item()), "?")
            tk = KERNEL_ID_TO_NAME.get(int(trg_kernel_cpu[i].item()), "?")
            key = (sd, sk, td, tk)
            if key not in per_pair:
                per_pair[key] = {"psnr": [], "ssim": []}
            per_pair[key]["psnr"].append(psnr_val)
            per_pair[key]["ssim"].append(ssim_val)

        log.info("Logging metrics ...")
        log.info(f"[Overall] PSNR | Mean: {np.mean(all_psnr):.4f}, SD: {np.std(all_psnr):.4f}")
        log.info(f"[Overall] SSIM | Mean: {np.mean(all_ssim):.4f}, SD: {np.std(all_ssim):.4f}")
        log_metric("metric/Mean_PSNR", np.mean(all_psnr))
        log_metric("metric/Mean_SSIM", np.mean(all_ssim))

        for (sd, sk, td, tk), m in sorted(per_pair.items()):
            if not m["psnr"]:
                continue
            plabel = f"({sd},{sk})_to_({td},{tk})"
            log.info(
                f"[Pair: {plabel}] "
                f"PSNR={np.mean(m['psnr']):.4f}, "
                f"SSIM={np.mean(m['ssim']):.4f}  (n={len(m['psnr'])})"
            )
            log_metric(f"metric/{plabel}/PSNR", np.mean(m['psnr']))
            log_metric(f"metric/{plabel}/SSIM", np.mean(m['ssim']))

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
