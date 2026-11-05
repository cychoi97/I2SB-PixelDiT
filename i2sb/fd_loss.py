"""Lightweight FD-Loss module for I2SB CT restoration.

Adapted from "Representation Fréchet Loss for Visual Generation" (Yang et al., 2026).
https://arxiv.org/abs/2604.28190

Provides:
    - FeatureQueue      : circular buffer / EMA stats for stable covariance estimation
    - compute_frechet_distance_loss : differentiable FID (gradient-preserving)
    - precompute_sigma_ref_sqrt     : one-time σ_ref^{1/2} computation
    - load_fd_reference_stats       : load .npz reference statistics
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Queue — circular buffer with online accumulation & EMA stats support
# =============================================================================

class FeatureQueue(nn.Module):
    """Circular buffer or EMA stats tracker for FD computation.

    Supports EMA stats mode (default, ema_beta=0.999):
        Tracks running first and second moments (mu_ema, m2_ema) with exponential
        decay. Does not allocate a full feature buffer, saving VRAM.
        Formula:
            mu = beta * mu_ema + (1 - beta) * E[x]
            m2 = beta * m2_ema + (1 - beta) * E[x x^T]
            sigma = m2 - mu mu^T

    Supports online_accum mode (ema_beta=0.0):
        Maintains running sum/outer-product and updates incrementally as items
        are evicted from a circular buffer.

    Registered as buffers so they survive state_dict save/load and
    automatically move with .to(device) / .cuda().

    Args:
        size        : Number of feature vectors to keep (when ema_beta=0.0).
        feat_dim    : Dimensionality of each feature vector.
        online_accum: If True, use incremental sum/outer accumulation (for non-EMA).
        ema_beta    : EMA decay coefficient. If > 0.0, enables EMA stats mode (default: 0.999).
    """

    def __init__(
        self,
        size: int = 5000,
        feat_dim: int = 1024,
        online_accum: bool = True,
        ema_beta: float = 0.999,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.zeros(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("_filled", torch.zeros(1, dtype=torch.bool))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item()) if not self.ema_stats else 0

    @property
    def is_filled(self) -> bool:
        if self.ema_stats:
            return int(self._ema_count.item()) > 0
        return bool(self._filled.item())

    # ------------------------------------------------------------------
    # EMA Streaming Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        """Streaming accumulation for EMA initialization."""
        feats_d = feats.detach().float().double()
        if self.ema_stats:
            self.mu_ema.add_(feats_d.sum(0))
            self.m2_ema.addmm_(feats_d.T, feats_d)
            self._ema_count.add_(feats_d.shape[0])

    @torch.no_grad()
    def _finalize_streaming_init(self):
        """Normalize accumulated sums into moments (mu, E[xx^T])."""
        if self.ema_stats:
            count = int(self._ema_count.item())
            if count == 0:
                logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
                return
            self.mu_ema.div_(count)
            self.m2_ema.div_(count)
            logger.info(f"[FeatureQueue] EMA init done: {count} features (beta={self.ema_beta})")

    @torch.no_grad()
    def _init_accumulators(self):
        """Compute feat_sum_old and feat_outer_old for non-EMA online accum."""
        if not self.ema_stats:
            feats_d = self.feats.double()
            self.feat_sum_old.copy_(feats_d.sum(0))
            self.feat_outer_old.copy_(feats_d.T @ feats_d)

    # ------------------------------------------------------------------
    # Statistics computation (differentiable)
    # ------------------------------------------------------------------

    def build_feats_stats(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) with gradients flowing through new_feats.

        Dispatches to EMA mode if enabled (ema_stats=True); otherwise uses
        online accumulators / snapshot.

        Args:
            new_feats: [B, feat_dim] tensor (requires grad for FD backprop).

        Returns:
            mu    : [feat_dim] float64 mean vector.
            sigma : [feat_dim, feat_dim] float64 covariance matrix.
        """
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        B = new_d.shape[0]
        N = self.size

        if not self.is_filled or not self.online_accum:
            snap = self._snapshot(self.feats, new_feats)
            n_valid = min(N, self.pointer + B) if not self.is_filled else N
            snap_d = snap[:n_valid].double()
            if n_valid < 2:
                mu = new_d.mean(0)
                sigma = torch.eye(self.feat_dim, device=new_d.device, dtype=torch.float64) * 1e-4
                return mu, sigma
            mu = snap_d.mean(0)
            feats_c = snap_d - mu.detach()
            sigma = (feats_c.T @ feats_c) / (n_valid - 1)
            return mu, sigma

        evicted = self._get_evicted_feats(B).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / N
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / N) / (N - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) via EMA moments blended with new_feats (FD-Loss paper default)."""
        beta = self.ema_beta
        new_d = new_feats.double()
        B = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B

        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return new
        n = new.shape[0]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr : ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr : self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    # ------------------------------------------------------------------
    # Enqueue (detached — no gradient stored in moments / buffer)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        """Dequeue oldest entries or update running EMA moments.

        Args:
            new_feats: [B, feat_dim] tensor (will be detached).
        """
        n = new_feats.shape[0]
        new_det = new_feats.detach().float()

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            self._ema_count.add_(n)
            return

        if self.size == 0:
            return

        ptr = self.pointer
        if self.online_accum and self.is_filled:
            evicted = self._get_evicted_feats(n).double()
            new_d = new_det.double()
            self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
            self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)

        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr : self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]

        new_ptr = (ptr + n) % self.size
        self.ptr[0] = new_ptr

        if not self.is_filled and (ptr + n) >= self.size:
            self._filled[0] = True
            if self.online_accum:
                self._init_accumulators()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr : ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr : self.size], self.feats[: n - first]], dim=0)


# =============================================================================
# Multi-Domain Feature Queue for Target-Conditioned FD-Loss
# =============================================================================

DEFAULT_TARGET_DOMAINS = [
    ("Full", "B30f"),
    ("Full", "B45f"),
    ("Full", "B60f"),
    ("Quarter", "B30f"),
    ("Quarter", "B45f"),
]
DEFAULT_TARGET_DOMAIN_KEYS = [f"{d}_{k}" for d, k in DEFAULT_TARGET_DOMAINS]


class MultiDomainFeatureQueue(nn.Module):
    """Container managing multiple FeatureQueue instances partitioned by target domain.

    Supports multi-domain CT restoration where samples in a mini-batch belong to
    distinct target domains (e.g. Full_B30f, Full_B45f, Full_B60f, Quarter_B30f, Quarter_B45f).
    Maintains running moments for each target domain separately, preventing
    cross-domain distribution corruption (e.g., mixing smooth and sharp kernels).

    Args:
        domain_keys : List of string keys representing active target domains.
        size        : Buffer size per domain (for non-EMA mode).
        feat_dim    : Feature dimension (e.g. 768 for DINOv2-B).
        online_accum: Online accumulation flag.
        ema_beta    : EMA decay coefficient (default: 0.999).
    """

    def __init__(
        self,
        domain_keys: list[str],
        size: int = 5000,
        feat_dim: int = 768,
        online_accum: bool = True,
        ema_beta: float = 0.999,
    ):
        super().__init__()
        self.domain_keys = list(domain_keys)
        self.feat_dim = feat_dim
        self.queues = nn.ModuleDict({
            key: FeatureQueue(
                size=size,
                feat_dim=feat_dim,
                online_accum=online_accum,
                ema_beta=ema_beta,
            )
            for key in self.domain_keys
        })

    def get_queue(self, key: str) -> FeatureQueue | None:
        return self.queues.get(key, None)

    def compute_target_fd_loss(
        self,
        cls_feat: torch.Tensor,
        sample_domain_keys: list[str],
        ref_stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
        ref_sigma_sqrt: dict[str, torch.Tensor] | None = None,
        norm_eps: float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute target-conditioned Fréchet Distance loss partitioned by domain.

        Args:
            cls_feat: [B, feat_dim] DINOv2 CLS features (requires_grad for backprop).
            sample_domain_keys: List of length B with the target domain key for each sample.
            ref_stats: Dict mapping domain_key -> (mu_ref, sigma_ref).
            ref_sigma_sqrt: Dict mapping domain_key -> sigma_ref^{1/2} (if eigvalsh).
            norm_eps: Epsilon for FD normalization (default: 0.01).

        Returns:
            mean_loss: Scalar torch.Tensor with gradients flowing to cls_feat.
            loss_dict: Dict mapping domain_key -> detached loss float.
        """
        device = cls_feat.device
        domain_to_indices: dict[str, list[int]] = {}
        for idx, key in enumerate(sample_domain_keys):
            if key in self.queues and key in ref_stats:
                domain_to_indices.setdefault(key, []).append(idx)

        if not domain_to_indices:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        domain_losses = []
        loss_dict = {}

        for key, indices in domain_to_indices.items():
            sub_feats = cls_feat[indices]  # [B_k, feat_dim] (preserves autograd)
            queue = self.queues[key]
            mu_gen, sigma_gen = queue.build_feats_stats(sub_feats)

            mu_ref, sigma_ref = ref_stats[key]
            sigma_sqrt_k = ref_sigma_sqrt[key] if ref_sigma_sqrt is not None else None

            loss_raw = compute_frechet_distance_loss(
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                mu=mu_gen,
                sigma=sigma_gen,
                sigma_ref_sqrt=sigma_sqrt_k,
            )
            loss_norm = loss_raw / (loss_raw.detach() + norm_eps)
            domain_losses.append(loss_norm)
            loss_dict[key] = float(loss_norm.item())

        mean_loss = torch.stack(domain_losses).mean()
        return mean_loss, loss_dict

    @torch.no_grad()
    def enqueue(
        self,
        cls_feat: torch.Tensor,
        sample_domain_keys: list[str],
    ):
        """Enqueue detached features to their respective domain queues."""
        domain_to_indices: dict[str, list[int]] = {}
        for idx, key in enumerate(sample_domain_keys):
            if key in self.queues:
                domain_to_indices.setdefault(key, []).append(idx)

        for key, indices in domain_to_indices.items():
            sub_feats = cls_feat[indices].detach()
            self.queues[key].enqueue(sub_feats)


# =============================================================================
# Covariance helpers
# =============================================================================

def precompute_sigma_ref_sqrt(
    sigma_ref: torch.Tensor | dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Precompute σ_ref^{1/2} via eigendecomposition.

    Supports:
        - Single tensor: sigma_ref [D, D]
        - Dict of tensors: {key: sigma_ref}
        - Dict of tuples: {key: (mu_ref, sigma_ref)}
    """
    if isinstance(sigma_ref, dict):
        sqrt_dict = {}
        for k, v in sigma_ref.items():
            sig = v[1] if isinstance(v, tuple) else v
            sqrt_dict[k] = precompute_sigma_ref_sqrt(sig)
        return sqrt_dict

    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute tr(Σ) + tr(Σ_ref) - 2·tr(√(Σ·Σ_ref))."""
    if sigma_ref_sqrt is not None:
        M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return None
        eigvals = torch.linalg.eigvals(product).real
        eigvals = torch.clamp(eigvals, min=0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))

    return (
        torch.diagonal(sigma).sum()
        + torch.diagonal(sigma_ref).sum()
        - 2.0 * tr_covmean
    )


# =============================================================================
# Differentiable Fréchet Distance loss
# =============================================================================

def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable FID from raw features or pre-computed (mu, sigma)."""
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            return torch.tensor(
                1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True
            )
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu.detach()
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma.")

    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        return torch.tensor(1e6, device=mu.device, dtype=torch.float32)

    return (mean_term + trace_term).float()


# =============================================================================
# Reference statistics I/O
# =============================================================================

def load_fd_reference_stats(
    stats_path: str,
    device: str | torch.device = "cuda",
):
    """Load reference FID statistics from a .npz file.

    Supports:
        - Single-domain format: {'mu': ..., 'sigma': ...}
          Returns: (mu_ref, sigma_ref)
        - Multi-domain format: {'mu_<key>': ..., 'sigma_<key>': ...}
          Returns: dict of {key: (mu_ref, sigma_ref)}
    """
    ref = np.load(stats_path)

    # Check for single-domain file
    if "mu" in ref.files and "sigma" in ref.files:
        mu_ref = torch.tensor(ref["mu"], device=device, dtype=torch.float64)
        sigma_ref = torch.tensor(ref["sigma"], device=device, dtype=torch.float64)
        logger.info(
            f"[FD-Loss] Loaded single-domain reference stats from {stats_path} "
            f"(feat_dim={mu_ref.shape[0]})"
        )
        return mu_ref, sigma_ref

    # Check for multi-domain file
    stats_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    domain_keys = [k[3:] for k in ref.files if k.startswith("mu_")]
    if not domain_keys and "target_domain_keys" in ref.files:
        domain_keys = [str(k) for k in ref["target_domain_keys"]]

    for k in domain_keys:
        if f"mu_{k}" in ref.files and f"sigma_{k}" in ref.files:
            mu_k = torch.tensor(ref[f"mu_{k}"], device=device, dtype=torch.float64)
            sigma_k = torch.tensor(ref[f"sigma_{k}"], device=device, dtype=torch.float64)
            stats_dict[k] = (mu_k, sigma_k)
            logger.info(
                f"[FD-Loss] Loaded multi-domain reference stats for '{k}' "
                f"(feat_dim={mu_k.shape[0]})"
            )

    if not stats_dict:
        raise ValueError(
            f"No valid reference statistics found in {stats_path}. "
            f"Keys in .npz: {ref.files}"
        )

    logger.info(
        f"[FD-Loss] Loaded multi-domain reference stats with {len(stats_dict)} domains: "
        f"{list(stats_dict.keys())}"
    )
    return stats_dict

