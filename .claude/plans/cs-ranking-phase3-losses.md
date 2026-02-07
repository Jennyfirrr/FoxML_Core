# Phase 3: Ranking Loss Functions

**Parent**: `cross-sectional-ranking-objective.md`
**Status**: ✅ Complete
**Estimated Effort**: 2-3 hours

## Objective

Implement loss functions that directly optimize ranking quality within each timestamp, rather than pointwise prediction accuracy.

## Current State

```python
# Current: Pointwise MSE loss
loss = F.mse_loss(predictions, targets)  # Treats each sample independently
```

## Proposed Loss Functions

### Option A: Pairwise Logistic Loss (Recommended)

**Concept**: For pairs (i, j) where y_i > y_j, encourage s_i > s_j.

```python
# TRAINING/losses/ranking_losses.py

import torch
import torch.nn.functional as F


def pairwise_logistic_loss(
    scores: torch.Tensor,      # (B, M) model outputs
    targets: torch.Tensor,     # (B, M) CS percentile targets
    mask: torch.Tensor,        # (B, M) 1 if valid, 0 if missing
    top_pct: float = 0.2,      # Top 20% as "winners"
    bottom_pct: float = 0.2,   # Bottom 20% as "losers"
    max_pairs: int = 100,      # Max pairs per timestamp
    margin: float = 0.0,       # Optional margin for harder objective
) -> torch.Tensor:
    """
    Pairwise ranking loss: winners should score higher than losers.

    For each timestamp t in batch:
    1. Identify top_pct symbols as winners (high target)
    2. Identify bottom_pct symbols as losers (low target)
    3. Sample pairs (winner, loser)
    4. Loss = log(1 + exp(-(s_winner - s_loser - margin)))

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        top_pct: Fraction of symbols to consider "winners"
        bottom_pct: Fraction of symbols to consider "losers"
        max_pairs: Maximum pairs to sample per timestamp
        margin: Margin for hinge-like behavior

    Returns:
        Scalar loss tensor
    """
    B, M = scores.shape
    total_loss = 0.0
    n_pairs = 0

    for b in range(B):
        # Get valid symbols for this timestamp
        valid_mask = mask[b] > 0.5
        valid_targets = targets[b][valid_mask]
        valid_scores = scores[b][valid_mask]
        n_valid = valid_mask.sum().item()

        if n_valid < 10:  # Skip timestamps with too few symbols
            continue

        # Determine thresholds
        k_top = max(1, int(n_valid * top_pct))
        k_bottom = max(1, int(n_valid * bottom_pct))

        # Get top and bottom indices
        _, top_idx = valid_targets.topk(k_top)
        _, bottom_idx = valid_targets.topk(k_bottom, largest=False)

        # Sample pairs
        n_possible_pairs = k_top * k_bottom
        if n_possible_pairs > max_pairs:
            # Random sampling of pairs
            pair_indices = torch.randperm(n_possible_pairs)[:max_pairs]
            top_selected = pair_indices // k_bottom
            bottom_selected = pair_indices % k_bottom
        else:
            # All pairs
            top_selected = torch.arange(k_top).repeat_interleave(k_bottom)
            bottom_selected = torch.arange(k_bottom).repeat(k_top)

        winner_scores = valid_scores[top_idx[top_selected]]
        loser_scores = valid_scores[bottom_idx[bottom_selected]]

        # Pairwise logistic loss
        diff = winner_scores - loser_scores - margin
        pair_loss = torch.log(1 + torch.exp(-diff))

        total_loss += pair_loss.sum()
        n_pairs += len(pair_loss)

    if n_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return total_loss / n_pairs
```

### Option B: Listwise Softmax Loss

**Concept**: Treat targets as a distribution, match with score distribution.

```python
def listwise_softmax_loss(
    scores: torch.Tensor,      # (B, M)
    targets: torch.Tensor,     # (B, M) CS percentile targets
    mask: torch.Tensor,        # (B, M)
    temperature: float = 1.0,  # Softmax temperature
) -> torch.Tensor:
    """
    Listwise ranking loss: match score distribution to target distribution.

    For each timestamp:
    p = softmax(targets / temp)   # Target distribution
    q = softmax(scores / temp)    # Predicted distribution
    loss = KL(p || q) or cross_entropy(p, q)

    Args:
        scores: Model predictions (B, M)
        targets: CS percentile targets (B, M)
        mask: Valid symbol mask (B, M)
        temperature: Softmax temperature (lower = sharper)

    Returns:
        Scalar loss tensor
    """
    B, M = scores.shape
    total_loss = 0.0
    n_valid_batches = 0

    for b in range(B):
        valid_mask = mask[b] > 0.5
        if valid_mask.sum() < 10:
            continue

        valid_scores = scores[b][valid_mask]
        valid_targets = targets[b][valid_mask]

        # Convert to distributions
        p = F.softmax(valid_targets / temperature, dim=0)  # Target dist
        log_q = F.log_softmax(valid_scores / temperature, dim=0)  # Pred dist

        # Cross-entropy loss: -sum(p * log(q))
        ce_loss = -torch.sum(p * log_q)

        total_loss += ce_loss
        n_valid_batches += 1

    if n_valid_batches == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return total_loss / n_valid_batches
```

### Option C: LambdaRank (Advanced)

**Concept**: Weight pairs by how much swapping them would change NDCG.

```python
def lambda_rank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    LambdaRank: Pairwise loss weighted by NDCG delta.

    More sophisticated than basic pairwise, weights pairs by
    how much ranking quality would improve from a swap.

    Note: More complex to implement correctly. Consider using
    existing implementations (e.g., from allRank library).
    """
    # Implementation based on LambdaMART paper
    # Omitted for brevity - can use library or implement later
    raise NotImplementedError("Use allRank library or implement from paper")
```

### Option D: Pointwise on Percentile (Baseline)

```python
def pointwise_percentile_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Simple MSE on CS percentile targets.

    Not a true ranking loss, but aligned baseline.
    """
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return F.mse_loss(scores[valid], targets[valid])
```

## Comparison

| Loss | Pros | Cons | When to Use |
|------|------|------|-------------|
| **Pairwise Logistic** | Intuitive, scalable, robust | Sampling introduces variance | Default choice |
| **Listwise Softmax** | Full cross-section interaction | Memory for large M | Small-medium universe |
| **LambdaRank** | Theoretically optimal | Complex, slow | Final optimization |
| **Pointwise MSE** | Simple, fast | Not ranking-aligned | Baseline comparison |

## Gradient Considerations

### Pairwise Loss Gradients

```
∂L/∂s_winner = -sigmoid(-(s_w - s_l)) / n_pairs
∂L/∂s_loser  = +sigmoid(-(s_w - s_l)) / n_pairs

When winner >> loser: gradients → 0 (already correct)
When winner ≈ loser: gradients strongest (learning signal)
When winner << loser: gradients saturate at ±1 (wrong order)
```

### Numerical Stability

```python
# Stable log-sigmoid
def stable_log_sigmoid(x):
    """log(sigmoid(x)) = -softplus(-x)"""
    return -F.softplus(-x)

# In pairwise loss:
pair_loss = -stable_log_sigmoid(winner_scores - loser_scores - margin)
```

## Training Loop Integration

```python
# In model trainer

def train_epoch_ranking(model, dataloader, optimizer, loss_fn, loss_config):
    """
    Training loop for ranking objective.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        X = batch['X'].to(device)      # (B, M, L, F)
        y = batch['y'].to(device)      # (B, M)
        mask = batch['mask'].to(device) # (B, M)

        optimizer.zero_grad()

        # Forward: model processes each symbol's sequence
        # Output shape: (B, M) - one score per symbol
        scores = model(X, mask)

        # Compute ranking loss
        loss = loss_fn(scores, y, mask, **loss_config)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## Model Architecture Consideration

The model needs to output `(B, M)` scores from `(B, M, L, F)` input:

```python
class CrossSectionalSequenceModel(nn.Module):
    """
    Model that scores each symbol's sequence independently.

    Input: (B, M, L, F)
    Output: (B, M)
    """

    def __init__(self, seq_model: nn.Module):
        """
        Args:
            seq_model: Base sequence model (LSTM, Transformer, CNN1D)
                       Takes (batch, L, F) -> (batch, 1)
        """
        super().__init__()
        self.seq_model = seq_model

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, M, L, F)
            mask: (B, M)

        Returns:
            scores: (B, M)
        """
        B, M, L, F = X.shape

        # Flatten to process all sequences
        X_flat = X.view(B * M, L, F)  # (B*M, L, F)

        # Get scores
        scores_flat = self.seq_model(X_flat).squeeze(-1)  # (B*M,)

        # Reshape back
        scores = scores_flat.view(B, M)  # (B, M)

        # Zero out masked positions (optional, loss handles this)
        scores = scores * mask

        return scores
```

## Deliverables

1. [x] `TRAINING/losses/ranking_losses.py`:
   - `pairwise_logistic_loss()` ✅
   - `pairwise_hinge_loss()` ✅ (bonus)
   - `listwise_softmax_loss()` ✅
   - `listwise_kl_loss()` ✅ (bonus)
   - `pointwise_mse_loss()` ✅
   - `pointwise_huber_loss()` ✅ (bonus)
   - `hybrid_pairwise_pointwise_loss()` ✅ (bonus)

2. [x] `TRAINING/losses/__init__.py`:
   - Factory function `get_ranking_loss(name, config)` ✅
   - `RankingLoss` nn.Module wrapper ✅
   - `RankingLossType` enum ✅

3. [x] Unit tests in `tests/test_ranking_losses.py` - 34 tests passing ✅

4. [x] Model wrappers in `TRAINING/losses/cs_model_wrapper.py`:
   - `CrossSectionalSequenceModel` ✅
   - `CrossSectionalMLPModel` ✅ (bonus baseline)
   - `CrossSectionalAttentionModel` ✅ (bonus)

## Definition of Done

- [x] Pairwise loss implemented and tested
- [x] Listwise loss implemented and tested
- [x] Numerically stable (no NaN/Inf gradients) - uses `stable_log_sigmoid()`, `stable_log1p_exp()`
- [x] Handles missing symbols via mask
- [x] Integrates with existing model trainers (via factory and Module interface)

## Implementation Notes (2026-01-21)

**Files created:**
- `TRAINING/losses/ranking_losses.py` - 7 loss functions + factory + Module wrapper
- `TRAINING/losses/cs_model_wrapper.py` - 3 model wrappers
- `TRAINING/losses/__init__.py` - Package exports
- `tests/test_ranking_losses.py` - 34 unit tests

**Key design decisions:**
- Explicit for-loop over batch dimension for clarity and handling variable valid symbols
- Numerical stability via `F.softplus()` for log-sigmoid operations
- Both functional and nn.Module interfaces for flexibility
- Model wrapper flattens (B, M, L, F) -> (B*M, L, F) for base model processing

**Bonus features beyond original spec:**
- Hinge loss variant for pairwise
- KL divergence variant for listwise
- Huber loss variant for pointwise
- Hybrid loss combining pairwise + pointwise
- MLP baseline model for ablation studies
- Cross-symbol attention model for advanced use
