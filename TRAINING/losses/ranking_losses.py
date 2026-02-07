# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Ranking Loss Functions for Cross-Sectional Training
====================================================

Loss functions that directly optimize ranking quality within each timestamp,
rather than pointwise prediction accuracy.

Loss Types:
- **Pairwise Logistic**: Winners should score higher than losers
- **Listwise Softmax**: Match score distribution to target distribution
- **Pointwise Percentile**: Simple MSE on CS percentile targets (baseline)

See .claude/plans/cs-ranking-phase3-losses.md for design details.
"""

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==============================================================================
# NUMERICAL STABILITY HELPERS
# ==============================================================================


def stable_log_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(sigmoid(x)).

    log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x)) = -softplus(-x)

    This avoids numerical issues when x is very large or very small.
    """
    return -F.softplus(-x)


def stable_log1p_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(1 + exp(x)).

    For large x: log(1 + exp(x)) ≈ x
    For small x: log(1 + exp(x)) ≈ exp(x)
    """
    return F.softplus(x)


# ==============================================================================
# PAIRWISE RANKING LOSS
# ==============================================================================


def pairwise_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    max_pairs: int = 100,
    margin: float = 0.0,
    min_symbols: int = 10,
) -> torch.Tensor:
    """
    Pairwise ranking loss: winners should score higher than losers.

    For each timestamp t in batch:
    1. Identify top_pct symbols as winners (high target)
    2. Identify bottom_pct symbols as losers (low target)
    3. Sample pairs (winner, loser)
    4. Loss = log(1 + exp(-(s_winner - s_loser - margin)))

    This is the **recommended** loss for cross-sectional ranking because:
    - Intuitive: "did you rank winners above losers?"
    - Scalable: O(k²) pairs, not O(M²)
    - Robust: sampling reduces variance from outliers

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M), NaN for missing
        mask: Valid symbol mask (B, M), 1.0 if present
        top_pct: Fraction of symbols to consider "winners"
        bottom_pct: Fraction of symbols to consider "losers"
        max_pairs: Maximum pairs to sample per timestamp
        margin: Margin for hinge-like behavior (0 = pure logistic)
        min_symbols: Minimum valid symbols to compute loss

    Returns:
        Scalar loss tensor (mean over all pairs)
    """
    B, M = scores.shape
    device = scores.device

    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for b in range(B):
        # Get valid symbols for this timestamp
        valid_mask = (mask[b] > 0.5) & (~torch.isnan(targets[b]))
        n_valid = valid_mask.sum().item()

        if n_valid < min_symbols:
            continue

        valid_targets = targets[b][valid_mask]
        valid_scores = scores[b][valid_mask]

        # Determine number of winners/losers
        k_top = max(1, int(n_valid * top_pct))
        k_bottom = max(1, int(n_valid * bottom_pct))

        # Get top (winners) and bottom (losers) indices
        _, top_idx = valid_targets.topk(k_top, largest=True)
        _, bottom_idx = valid_targets.topk(k_bottom, largest=False)

        # Sample pairs
        n_possible_pairs = k_top * k_bottom

        if n_possible_pairs <= max_pairs:
            # Use all pairs
            top_selected = torch.arange(k_top, device=device).repeat_interleave(k_bottom)
            bottom_selected = torch.arange(k_bottom, device=device).repeat(k_top)
        else:
            # Random sampling of pairs (deterministic within forward pass)
            pair_indices = torch.randperm(n_possible_pairs, device=device)[:max_pairs]
            top_selected = pair_indices // k_bottom
            bottom_selected = pair_indices % k_bottom

        # Get scores for selected pairs
        winner_scores = valid_scores[top_idx[top_selected]]
        loser_scores = valid_scores[bottom_idx[bottom_selected]]

        # Pairwise logistic loss: -log(sigmoid(s_w - s_l - margin))
        # = log(1 + exp(-(s_w - s_l - margin)))
        diff = winner_scores - loser_scores - margin
        pair_loss = stable_log1p_exp(-diff)

        total_loss = total_loss + pair_loss.sum()
        n_pairs += len(pair_loss)

    if n_pairs == 0:
        # Return zero loss with gradient connection
        return (scores * 0).sum()

    return total_loss / n_pairs


def pairwise_hinge_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
    max_pairs: int = 100,
    margin: float = 1.0,
    min_symbols: int = 10,
) -> torch.Tensor:
    """
    Pairwise hinge loss: max(0, margin - (s_winner - s_loser)).

    Similar to pairwise_logistic_loss but uses hinge loss instead of
    logistic loss. Gradients are zero when margin is satisfied.

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        top_pct: Fraction of winners
        bottom_pct: Fraction of losers
        max_pairs: Maximum pairs per timestamp
        margin: Required score difference
        min_symbols: Minimum valid symbols

    Returns:
        Scalar loss tensor
    """
    B, M = scores.shape
    device = scores.device

    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for b in range(B):
        valid_mask = (mask[b] > 0.5) & (~torch.isnan(targets[b]))
        n_valid = valid_mask.sum().item()

        if n_valid < min_symbols:
            continue

        valid_targets = targets[b][valid_mask]
        valid_scores = scores[b][valid_mask]

        k_top = max(1, int(n_valid * top_pct))
        k_bottom = max(1, int(n_valid * bottom_pct))

        _, top_idx = valid_targets.topk(k_top, largest=True)
        _, bottom_idx = valid_targets.topk(k_bottom, largest=False)

        n_possible_pairs = k_top * k_bottom

        if n_possible_pairs <= max_pairs:
            top_selected = torch.arange(k_top, device=device).repeat_interleave(k_bottom)
            bottom_selected = torch.arange(k_bottom, device=device).repeat(k_top)
        else:
            pair_indices = torch.randperm(n_possible_pairs, device=device)[:max_pairs]
            top_selected = pair_indices // k_bottom
            bottom_selected = pair_indices % k_bottom

        winner_scores = valid_scores[top_idx[top_selected]]
        loser_scores = valid_scores[bottom_idx[bottom_selected]]

        # Hinge loss
        diff = winner_scores - loser_scores
        pair_loss = F.relu(margin - diff)

        total_loss = total_loss + pair_loss.sum()
        n_pairs += len(pair_loss)

    if n_pairs == 0:
        return (scores * 0).sum()

    return total_loss / n_pairs


# ==============================================================================
# LISTWISE RANKING LOSS
# ==============================================================================


def listwise_softmax_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    min_symbols: int = 10,
) -> torch.Tensor:
    """
    Listwise ranking loss: match score distribution to target distribution.

    For each timestamp:
    - p = softmax(targets / temp)  -> Target distribution
    - q = softmax(scores / temp)   -> Predicted distribution
    - loss = cross_entropy(p, log_q) = -sum(p * log(q))

    This is equivalent to minimizing KL divergence from p to q (up to constant).

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        temperature: Softmax temperature (lower = sharper distribution)
        min_symbols: Minimum valid symbols

    Returns:
        Scalar loss tensor
    """
    B, M = scores.shape
    device = scores.device

    total_loss = torch.tensor(0.0, device=device)
    n_valid_batches = 0

    for b in range(B):
        valid_mask = (mask[b] > 0.5) & (~torch.isnan(targets[b]))
        n_valid = valid_mask.sum().item()

        if n_valid < min_symbols:
            continue

        valid_scores = scores[b][valid_mask]
        valid_targets = targets[b][valid_mask]

        # Convert to distributions
        p = F.softmax(valid_targets / temperature, dim=0)  # Target distribution
        log_q = F.log_softmax(valid_scores / temperature, dim=0)  # Log predicted

        # Cross-entropy loss: -sum(p * log(q))
        ce_loss = -torch.sum(p * log_q)

        total_loss = total_loss + ce_loss
        n_valid_batches += 1

    if n_valid_batches == 0:
        return (scores * 0).sum()

    return total_loss / n_valid_batches


def listwise_kl_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    min_symbols: int = 10,
) -> torch.Tensor:
    """
    Listwise KL divergence loss.

    KL(p || q) = sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
               = -H(p) + CE(p, q)

    Since H(p) is constant w.r.t. model parameters, this is equivalent to
    cross-entropy up to a constant. Provided for completeness.

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        temperature: Softmax temperature
        min_symbols: Minimum valid symbols

    Returns:
        Scalar loss tensor
    """
    B, M = scores.shape
    device = scores.device

    total_loss = torch.tensor(0.0, device=device)
    n_valid_batches = 0

    for b in range(B):
        valid_mask = (mask[b] > 0.5) & (~torch.isnan(targets[b]))
        n_valid = valid_mask.sum().item()

        if n_valid < min_symbols:
            continue

        valid_scores = scores[b][valid_mask]
        valid_targets = targets[b][valid_mask]

        p = F.softmax(valid_targets / temperature, dim=0)
        q = F.softmax(valid_scores / temperature, dim=0)

        # KL divergence with numerical stability
        kl = F.kl_div(q.log(), p, reduction="sum")

        total_loss = total_loss + kl
        n_valid_batches += 1

    if n_valid_batches == 0:
        return (scores * 0).sum()

    return total_loss / n_valid_batches


# ==============================================================================
# POINTWISE LOSS (BASELINE)
# ==============================================================================


def pointwise_mse_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    min_symbols: int = 1,
) -> torch.Tensor:
    """
    Simple MSE loss on CS percentile targets.

    Not a true ranking loss, but provides an aligned baseline for comparison.
    Uses the same CS-normalized targets as ranking losses.

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        min_symbols: Minimum valid symbols (not used, for API consistency)

    Returns:
        Scalar loss tensor
    """
    valid = (mask > 0.5) & (~torch.isnan(targets))

    if valid.sum() == 0:
        return (scores * 0).sum()

    return F.mse_loss(scores[valid], targets[valid])


def pointwise_huber_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    min_symbols: int = 1,
) -> torch.Tensor:
    """
    Huber loss on CS percentile targets.

    More robust to outliers than MSE.

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        delta: Threshold for quadratic vs linear
        min_symbols: Minimum valid symbols

    Returns:
        Scalar loss tensor
    """
    valid = (mask > 0.5) & (~torch.isnan(targets))

    if valid.sum() == 0:
        return (scores * 0).sum()

    return F.huber_loss(scores[valid], targets[valid], delta=delta)


# ==============================================================================
# COMBINED / HYBRID LOSSES
# ==============================================================================


def hybrid_pairwise_pointwise_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pairwise_weight: float = 0.8,
    pointwise_weight: float = 0.2,
    **pairwise_kwargs,
) -> torch.Tensor:
    """
    Hybrid loss combining pairwise ranking and pointwise regression.

    Useful for balancing ranking quality with absolute prediction accuracy.

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        pairwise_weight: Weight for pairwise loss
        pointwise_weight: Weight for pointwise loss
        **pairwise_kwargs: Additional args for pairwise loss

    Returns:
        Scalar loss tensor
    """
    pw_loss = pairwise_logistic_loss(scores, targets, mask, **pairwise_kwargs)
    pt_loss = pointwise_mse_loss(scores, targets, mask)

    return pairwise_weight * pw_loss + pointwise_weight * pt_loss


# ==============================================================================
# LOSS FACTORY
# ==============================================================================


class RankingLossType:
    """Available ranking loss types."""

    PAIRWISE_LOGISTIC = "pairwise_logistic"
    PAIRWISE_HINGE = "pairwise_hinge"
    LISTWISE_SOFTMAX = "listwise_softmax"
    LISTWISE_KL = "listwise_kl"
    POINTWISE_MSE = "pointwise_mse"
    POINTWISE_HUBER = "pointwise_huber"
    HYBRID = "hybrid"


def get_ranking_loss(
    loss_type: str,
    **kwargs,
) -> callable:
    """
    Factory function to get ranking loss by name.

    Args:
        loss_type: One of RankingLossType values
        **kwargs: Loss-specific configuration

    Returns:
        Loss function with signature (scores, targets, mask) -> loss

    Example:
        >>> loss_fn = get_ranking_loss("pairwise_logistic", top_pct=0.2, max_pairs=50)
        >>> loss = loss_fn(scores, targets, mask)
    """
    loss_map = {
        RankingLossType.PAIRWISE_LOGISTIC: pairwise_logistic_loss,
        RankingLossType.PAIRWISE_HINGE: pairwise_hinge_loss,
        RankingLossType.LISTWISE_SOFTMAX: listwise_softmax_loss,
        RankingLossType.LISTWISE_KL: listwise_kl_loss,
        RankingLossType.POINTWISE_MSE: pointwise_mse_loss,
        RankingLossType.POINTWISE_HUBER: pointwise_huber_loss,
        RankingLossType.HYBRID: hybrid_pairwise_pointwise_loss,
    }

    if loss_type not in loss_map:
        valid = list(loss_map.keys())
        raise ValueError(f"Unknown loss type: {loss_type}. Valid types: {valid}")

    loss_fn = loss_map[loss_type]

    # Return a partial function with kwargs baked in
    def configured_loss(scores, targets, mask):
        return loss_fn(scores, targets, mask, **kwargs)

    return configured_loss


class RankingLoss(nn.Module):
    """
    PyTorch Module wrapper for ranking losses.

    Useful when you need a nn.Module interface (e.g., for model composition).

    Example:
        >>> criterion = RankingLoss("pairwise_logistic", top_pct=0.2)
        >>> loss = criterion(scores, targets, mask)
    """

    def __init__(self, loss_type: str, **kwargs):
        """
        Initialize ranking loss module.

        Args:
            loss_type: One of RankingLossType values
            **kwargs: Loss-specific configuration
        """
        super().__init__()
        self.loss_type = loss_type
        self.kwargs = kwargs
        self._loss_fn = get_ranking_loss(loss_type, **kwargs)

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ranking loss.

        Args:
            scores: Model predictions (B, M)
            targets: CS percentile targets (B, M)
            mask: Valid symbol mask (B, M)

        Returns:
            Scalar loss tensor
        """
        return self._loss_fn(scores, targets, mask)

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, kwargs={self.kwargs}"
