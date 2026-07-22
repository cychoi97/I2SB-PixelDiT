import torch
import torch.nn as nn

from pixdit_core.pixeldit_c2i import PixDiT
from . import util

from ipdb import set_trace as debug


# ---------------------------------------------------------------------------
# PixDiTNet: I2SB network built on the PixDiT dual-level architecture.
#
# Architecture overview (PixDiT, c2i variant):
#   - Patch-level context stream  : patch_depth × AugmentedDiTBlock
#   - Pixel-level detail stream   : pixel_depth × PiTBlock
#   - Conditioning                : t_emb + y_emb via AdaLN
#     → Multi-domain : NoiseAdaptiveMultiDomainEmbedder(embedding_mode)
#       - 'base'             : src/trg dose + kernel base embeddings
#       - 'base_cross'       : base + cross-domain (src×trg) embedding
#       - 'base_cross_noise' : base + cross + noise-adaptive embedding
#     → Legacy class  : ClassEmbedder(task_id)
#
# cond_x1 support:
#   When cond_x1=True, the source image x1 is concatenated to x along the
#   channel dimension before the forward pass, so in_channels = C_img * 2.
#   A lightweight 1×1 conv projects the output back to C_img channels.
#
# Default lightweight configuration:
#   hidden_size=384, patch_depth=6
#   pixel_hidden_size=64, pixel_depth=4
#   num_groups=8 (attention heads), patch_size=2
# ---------------------------------------------------------------------------


class PixDiTNet(nn.Module):
    """Multi-domain conditioned I2SB network using PixDiT architecture.

    Args:
        log               : Logger instance.
        noise_levels      : 1-D float tensor [T] of noise levels.
        in_channels       : Image channels (1 for CT grayscale).
        cond              : If True, concatenate source image (x1) along channel dim.
        num_dose_attrs    : Dose domain attribute count.
        num_kernel_attrs  : Kernel domain attribute count.
        hidden_size       : Patch-level context dimension.
        pixel_hidden_size : Pixel-level feature dimension.
        patch_depth       : Number of AugmentedDiTBlock layers.
        pixel_depth       : Number of PiTBlock layers.
        num_groups        : Attention heads (must divide hidden_size).
        patch_size        : Pixel patch size for the dual-level pipeline.
        noise_feat_dim    : Dimension of the noise statistics feature vector.
        embedding_mode    : Conditioning mode for NoiseAdaptiveMultiDomainEmbedder.
                            One of 'base', 'base_cross', 'base_cross_noise'.
    """

    def __init__(
        self,
        log,
        noise_levels,
        in_channels: int = 1,
        cond: bool = False,
        num_dose_attrs: int = 2,
        num_kernel_attrs: int = 3,
        hidden_size: int = 384,
        pixel_hidden_size: int = 64,
        patch_depth: int = 6,
        pixel_depth: int = 4,
        num_groups: int = 8,
        patch_size: int = 2,
        noise_feat_dim: int = 20,
        embedding_mode: str = "base_cross_noise",
    ):
        super().__init__()

        self.img_channels = in_channels
        self.cond = cond
        self.noise_levels = noise_levels

        # When cond_x1=True, the network sees 2×in_channels as input.
        net_in_channels = in_channels * 2 if cond else in_channels

        # Build PixDiT with multi-domain conditioning.
        self.model = PixDiT(
            in_channels=net_in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            patch_size=patch_size,
            num_dose_attrs=num_dose_attrs,
            num_kernel_attrs=num_kernel_attrs,
            use_pixel_abs_pos=True,
            noise_feat_dim=noise_feat_dim,
            embedding_mode=embedding_mode,
        )

        # If cond=True, PixDiT outputs 2C channels (out_channels==in_channels).
        # Add a 1×1 conv to project back to C channels.
        if cond:
            self.out_proj = nn.Conv2d(net_in_channels, in_channels, kernel_size=1, bias=True)
            nn.init.ones_(self.out_proj.weight[:, :in_channels])
            nn.init.zeros_(self.out_proj.weight[:, in_channels:])
            nn.init.zeros_(self.out_proj.bias)
        else:
            self.out_proj = None

        param_count = util.count_parameters(self.model)
        log.info(
            f"[Net] Initialized PixDiTNet | "
            f"hidden={hidden_size}, patch_depth={patch_depth}, "
            f"pixel_hidden={pixel_hidden_size}, pixel_depth={pixel_depth}, "
            f"patch_size={patch_size}, num_groups={num_groups}, "
            f"dose_attrs={num_dose_attrs}, kernel_attrs={num_kernel_attrs}, "
            f"cond_x1={cond}, embedding_mode={embedding_mode}, "
            f"noise_feat_dim={noise_feat_dim} | Params={param_count:,}"
        )

    def forward(self, x, steps, cond=None,
                src_dose=None, src_kernel=None,
                trg_dose=None, trg_kernel=None,
                noise_stats=None):
        """Forward pass.

        Args:
            x          : [B, C, H, W] noisy image at current timestep.
            steps      : [B] integer step indices (into noise_levels).
            cond       : [B, C, H, W] source image; required when self.cond=True.
            src_dose   : [B] long tensor — source dose attribute index.
            src_kernel : [B] long tensor — source kernel attribute index.
            trg_dose   : [B] long tensor — target dose attribute index.
            trg_kernel : [B] long tensor — target kernel attribute index.
            noise_stats: [B, noise_feat_dim] float tensor — input image noise statistics.

        Returns:
            [B, C, H, W] network prediction (denoised direction).
        """
        # Convert step indices to noise-level float values.
        t = self.noise_levels[steps].detach()  # [B] float
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        x_in = torch.cat([x, cond], dim=1) if self.cond else x

        # PixDiT forward with multi-domain conditioning.
        out = self.model(
            x_in, t,
            src_dose=src_dose, src_kernel=src_kernel,
            trg_dose=trg_dose, trg_kernel=trg_kernel,
            noise_stats=noise_stats,
        )

        # Project back to original channel count when cond=True.
        if self.out_proj is not None:
            out = self.out_proj(out)

        return out
