# Phase 4: Metrics and Evaluation

**Parent**: `cross-sectional-ranking-objective.md`
**Status**: ✅ COMPLETE
**Estimated Effort**: 1-2 hours
**Completed**: 2026-01-21

## Objective

Implement metrics that match the trading objective: **ranking quality within each timestamp**, not pointwise prediction accuracy.

## Current Metrics (Pointwise)

```python
# Current: Pointwise metrics
mse = mean((pred - actual)^2)
mae = mean(|pred - actual|)
auc = roc_auc_score(binary_label, pred)  # Mixes timestamps
```

**Problem**: These don't tell you "did the model rank symbols correctly at each t?"

## Proposed Metrics (Cross-Sectional)

### 1. Spearman IC (Information Coefficient) - Primary Metric

```python
# TRAINING/metrics/ranking_metrics.py

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List


def spearman_ic_per_timestamp(
    scores: np.ndarray,    # (T, M) model scores
    returns: np.ndarray,   # (T, M) actual forward returns
    mask: np.ndarray,      # (T, M) valid symbol mask
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation between scores and returns per timestamp.

    This is THE metric for cross-sectional ranking: do higher scores
    correspond to higher actual returns within each t?

    Returns:
        {
            'ic_mean': mean IC across timestamps,
            'ic_std': std of IC,
            'ic_ir': IC / std (information ratio),
            'ic_hit_rate': fraction of timestamps with positive IC,
            'ic_series': List of per-timestamp ICs,
        }
    """
    T, M = scores.shape
    ics = []

    for t in range(T):
        valid = mask[t] > 0.5
        if valid.sum() < 10:
            continue

        s = scores[t, valid]
        r = returns[t, valid]

        # Spearman correlation
        ic, _ = spearmanr(s, r)
        if not np.isnan(ic):
            ics.append(ic)

    if len(ics) == 0:
        return {'ic_mean': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'ic_hit_rate': 0.0}

    ics = np.array(ics)
    return {
        'ic_mean': float(np.mean(ics)),
        'ic_std': float(np.std(ics)),
        'ic_ir': float(np.mean(ics) / (np.std(ics) + 1e-8)),
        'ic_hit_rate': float(np.mean(ics > 0)),
        'ic_series': ics.tolist(),
    }
```

### 2. Top-Bottom Spread

```python
def top_bottom_spread(
    scores: np.ndarray,    # (T, M)
    returns: np.ndarray,   # (T, M) actual forward returns
    mask: np.ndarray,      # (T, M)
    top_pct: float = 0.1,  # Top decile
    bottom_pct: float = 0.1,  # Bottom decile
) -> Dict[str, float]:
    """
    Compute return spread between top and bottom ranked symbols.

    This simulates the actual trading strategy:
    - Long top decile, short bottom decile
    - Measure return difference

    Returns:
        {
            'spread_mean': mean (top_return - bottom_return),
            'spread_std': std of spread,
            'spread_sharpe': spread_mean / spread_std * sqrt(252),
            'top_return_mean': mean return of top decile,
            'bottom_return_mean': mean return of bottom decile,
        }
    """
    T, M = scores.shape
    spreads = []
    top_returns = []
    bottom_returns = []

    for t in range(T):
        valid = mask[t] > 0.5
        n_valid = valid.sum()
        if n_valid < 20:  # Need enough for deciles
            continue

        s = scores[t, valid]
        r = returns[t, valid]

        k_top = max(1, int(n_valid * top_pct))
        k_bottom = max(1, int(n_valid * bottom_pct))

        # Rank by score
        ranking = np.argsort(s)[::-1]  # Descending

        top_r = r[ranking[:k_top]].mean()
        bottom_r = r[ranking[-k_bottom:]].mean()

        spreads.append(top_r - bottom_r)
        top_returns.append(top_r)
        bottom_returns.append(bottom_r)

    if len(spreads) == 0:
        return {'spread_mean': 0.0, 'spread_std': 0.0, 'spread_sharpe': 0.0}

    spreads = np.array(spreads)
    # Assuming 5-min bars, ~78 bars/day, ~252 trading days
    annualization = np.sqrt(78 * 252)

    return {
        'spread_mean': float(np.mean(spreads)),
        'spread_std': float(np.std(spreads)),
        'spread_sharpe': float(np.mean(spreads) / (np.std(spreads) + 1e-8) * annualization),
        'top_return_mean': float(np.mean(top_returns)),
        'bottom_return_mean': float(np.mean(bottom_returns)),
        'spread_series': spreads.tolist(),
    }
```

### 3. Turnover (for Cost Analysis)

```python
def portfolio_turnover(
    scores: np.ndarray,    # (T, M)
    mask: np.ndarray,      # (T, M)
    top_pct: float = 0.1,
) -> Dict[str, float]:
    """
    Compute turnover of top decile portfolio over time.

    High turnover = high transaction costs.

    Returns:
        {
            'turnover_mean': average fraction of portfolio changed per period,
            'turnover_series': per-period turnover,
        }
    """
    T, M = scores.shape
    turnovers = []
    prev_top_set = None

    for t in range(T):
        valid = mask[t] > 0.5
        n_valid = valid.sum()
        if n_valid < 20:
            continue

        s = scores[t, valid]
        valid_indices = np.where(valid)[0]
        k_top = max(1, int(n_valid * top_pct))

        # Get top symbols
        ranking = np.argsort(s)[::-1]
        top_indices = set(valid_indices[ranking[:k_top]])

        if prev_top_set is not None:
            # Turnover = fraction not in previous top
            overlap = len(top_indices & prev_top_set)
            turnover = 1.0 - (overlap / len(top_indices))
            turnovers.append(turnover)

        prev_top_set = top_indices

    if len(turnovers) == 0:
        return {'turnover_mean': 0.0}

    return {
        'turnover_mean': float(np.mean(turnovers)),
        'turnover_series': turnovers,
    }
```

### 4. Cost-Adjusted Returns

```python
def cost_adjusted_spread(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
    cost_per_trade_bps: float = 5.0,  # 5 bps per trade
) -> Dict[str, float]:
    """
    Spread minus estimated transaction costs.

    Assumes cost_per_trade_bps per full turnover.
    """
    spread_result = top_bottom_spread(scores, returns, mask, top_pct, bottom_pct)
    turnover_result = portfolio_turnover(scores, mask, top_pct)

    # Cost = turnover * 2 (enter + exit) * cost_bps
    cost_per_period = turnover_result['turnover_mean'] * 2 * (cost_per_trade_bps / 10000)

    net_spread = spread_result['spread_mean'] - cost_per_period

    return {
        'gross_spread': spread_result['spread_mean'],
        'turnover': turnover_result['turnover_mean'],
        'cost_per_period': cost_per_period,
        'net_spread': net_spread,
    }
```

## Metric Comparison

| Metric | What It Measures | Trading Relevance |
|--------|------------------|-------------------|
| **Spearman IC** | Rank correlation per t | Core signal quality |
| **Top-Bottom Spread** | Return of long-short | Direct P&L proxy |
| **Turnover** | Portfolio churn | Transaction cost driver |
| **Net Spread** | Spread minus costs | Realistic profitability |

## Aggregated Report

```python
def compute_ranking_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    config: Dict = None,
) -> Dict[str, any]:
    """
    Compute full suite of ranking metrics.

    Returns comprehensive evaluation for model comparison.
    """
    config = config or {}
    top_pct = config.get('top_pct', 0.1)
    bottom_pct = config.get('bottom_pct', 0.1)
    cost_bps = config.get('cost_per_trade_bps', 5.0)

    ic_result = spearman_ic_per_timestamp(scores, returns, mask)
    spread_result = top_bottom_spread(scores, returns, mask, top_pct, bottom_pct)
    turnover_result = portfolio_turnover(scores, mask, top_pct)
    cost_result = cost_adjusted_spread(scores, returns, mask, top_pct, bottom_pct, cost_bps)

    return {
        # Primary metrics
        'spearman_ic': ic_result['ic_mean'],
        'ic_ir': ic_result['ic_ir'],
        'ic_hit_rate': ic_result['ic_hit_rate'],

        # Trading metrics
        'spread': spread_result['spread_mean'],
        'spread_sharpe': spread_result['spread_sharpe'],
        'net_spread': cost_result['net_spread'],

        # Cost metrics
        'turnover': turnover_result['turnover_mean'],
        'cost_per_period': cost_result['cost_per_period'],

        # Detailed results (for analysis)
        'details': {
            'ic': ic_result,
            'spread': spread_result,
            'turnover': turnover_result,
            'cost': cost_result,
        }
    }
```

## Visualization (Optional)

```python
def plot_ranking_metrics(metrics: Dict, save_path: Path = None):
    """
    Generate diagnostic plots for ranking evaluation.

    1. IC time series with rolling mean
    2. Spread time series
    3. Cumulative return of long-short
    4. Turnover distribution
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # IC time series
    ic_series = metrics['details']['ic']['ic_series']
    axes[0, 0].plot(ic_series, alpha=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title(f"Spearman IC (mean={metrics['spearman_ic']:.4f})")

    # Spread time series
    spread_series = metrics['details']['spread']['spread_series']
    axes[0, 1].plot(spread_series, alpha=0.5)
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title(f"Top-Bottom Spread (mean={metrics['spread']:.4f})")

    # Cumulative return
    cum_return = np.cumsum(spread_series)
    axes[1, 0].plot(cum_return)
    axes[1, 0].set_title("Cumulative Long-Short Return")

    # Turnover histogram
    turnover_series = metrics['details']['turnover']['turnover_series']
    axes[1, 1].hist(turnover_series, bins=30)
    axes[1, 1].set_title(f"Turnover Distribution (mean={metrics['turnover']:.2f})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
```

## Deliverables

1. [x] `TRAINING/models/specialized/metrics.py` (extended existing module):
   - `top_bottom_spread()` - simulates long top/short bottom strategy
   - `portfolio_turnover()` - measures portfolio churn
   - `cost_adjusted_spread()` - net returns after costs
   - `spearman_ic_matrix()` - IC for (T, M) matrix input
   - `compute_ranking_metrics()` - unified interface
   - `compute_ranking_metrics_from_flat()` - adapter for flattened data

2. [ ] `TRAINING/metrics/visualization.py` (DEFERRED):
   - `plot_ranking_metrics()` - Optional, can be added in Phase 5

3. [x] Unit tests in `tests/test_ranking_metrics.py` - 27 tests, all passing

## Definition of Done

- [x] All metrics implemented
- [x] Handles masked/missing data correctly
- [x] Returns both summary and detailed results
- [ ] Visualization generates useful diagnostic plots (DEFERRED to Phase 5)
- [ ] Integrated into training evaluation loop (Phase 5)

## Implementation Notes

**Key decision**: Extended `TRAINING/models/specialized/metrics.py` instead of creating
a new `TRAINING/metrics/` module. This keeps ranking metrics alongside the existing
`cs_metrics_by_time()` function and follows the codebase pattern.

**Compatibility**: Added `compute_ranking_metrics_from_flat()` to bridge the existing
flattened format `(y_true, y_pred, ts)` with the new (T, M) matrix format from
`CrossSectionalDataset`.
