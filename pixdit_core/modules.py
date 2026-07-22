import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# scaled_dot_product_attention requires PyTorch >= 2.0.
# Fall back to a manual implementation for older versions.
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    from torch.nn.functional import scaled_dot_product_attention
else:
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0):
        """Manual SDPA fallback for PyTorch < 2.0."""
        scale = query.size(-1) ** -0.5
        attn_weight = torch.matmul(query * scale, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout_p > 0.0:
            attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
        return torch.matmul(attn_weight, value)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class NoiseAdaptiveMultiDomainEmbedder(nn.Module):
    """Noise-Adaptive Multi-Domain Embedder.

    Keeps the original 4 embedding tables from MultiDomainEmbedder to maintain
    compatibility with old checkpoints, and adds:
    1. Cross embedding for dose x kernel combination to model non-linear domain interaction.
    2. Projection for input noise statistics.
    3. Fusion MLP to fuse all embeddings into the final hidden_size representation.

    Embedding modes (controlled by ``embedding_mode``):
      - 'base'             : base domain embeddings only (noise_stats ignored).
      - 'base_cross'       : base + cross-domain embedding (noise_stats ignored).
      - 'base_cross_noise' : base + cross + noise-adaptive embedding (noise_stats required).
    """

    def __init__(
        self,
        num_dose_attrs: int,
        num_kernel_attrs: int,
        hidden_size: int,
        noise_feat_dim: int = 20,
        embedding_mode: str = "base_cross_noise"
    ):
        super().__init__()

        self.embedding_mode = embedding_mode
        self.num_dose_attrs   = num_dose_attrs
        self.num_kernel_attrs = num_kernel_attrs
        self.hidden_size = hidden_size
        self.noise_feat_dim = noise_feat_dim

        # Base embeddings
        self.src_dose_emb   = nn.Embedding(num_dose_attrs, hidden_size)
        self.src_kernel_emb = nn.Embedding(num_kernel_attrs, hidden_size)
        self.trg_dose_emb   = nn.Embedding(num_dose_attrs, hidden_size)
        self.trg_kernel_emb = nn.Embedding(num_kernel_attrs, hidden_size)

        # Cross embeddings (Dose x Kernel combinations)
        num_cross_attrs = num_dose_attrs * num_kernel_attrs
        self.src_cross_emb = nn.Embedding(num_cross_attrs, hidden_size)
        self.trg_cross_emb = nn.Embedding(num_cross_attrs, hidden_size)

        # Noise statistics projection
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_feat_dim, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # Fusion MLP depending on embedding mode
        if self.embedding_mode == "base_cross_noise":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        elif self.embedding_mode == "base_cross":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, int(hidden_size)),
                nn.SiLU(),
                nn.Linear(int(hidden_size), hidden_size)
            )
        elif self.embedding_mode == "base":
            self.fusion_mlp = nn.Identity()
        else:
            raise ValueError(f"Unknown embedding mode: {self.embedding_mode}")

    def forward(
        self,
        src_dose: torch.Tensor,
        src_kernel: torch.Tensor,
        trg_dose: torch.Tensor,
        trg_kernel: torch.Tensor,
        noise_stats: torch.Tensor = None,
    ) -> torch.Tensor:
        # Base embedding (compatible with original MultiDomainEmbedder)
        base_emb = (self.src_dose_emb(src_dose)     + self.src_kernel_emb(src_kernel) +
                    self.trg_dose_emb(trg_dose)     + self.trg_kernel_emb(trg_kernel))

        if self.embedding_mode == "base":
            return base_emb

        # Cross embedding (Dose x Kernel combination)
        src_idx = src_dose * self.num_kernel_attrs + src_kernel
        trg_idx = trg_dose * self.num_kernel_attrs + trg_kernel
        cross_emb = self.src_cross_emb(src_idx) + self.trg_cross_emb(trg_idx)

        if self.embedding_mode == "base_cross":
            fused = torch.cat([base_emb, cross_emb], dim=-1)
            return self.fusion_mlp(fused)

        # Noise statistics embedding (only for 'base_cross_noise')
        assert noise_stats is not None, (
            "noise_stats must be provided when embedding_mode='base_cross_noise'"
        )
        noise_emb = self.noise_proj(noise_stats.to(base_emb.dtype))

        # Fusion
        fused = torch.cat([base_emb, cross_emb, noise_emb], dim=-1)
        output = self.fusion_mlp(fused)
        return output


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis

def precompute_freqs_cis_ex2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=1.0):
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, height*scale[0], width)
    y_pos = torch.linspace(0, width*scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

