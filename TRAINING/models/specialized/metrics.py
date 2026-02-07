# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Sectional Metrics for Ranking Models
===========================================

Metrics that evaluate ranking quality within each timestamp, aligned with the
actual trading objective: rank symbols, long top decile, short bottom decile.

Core metrics:
- **Spearman IC**: Rank correlation between predictions and returns per timestamp
- **Top-Bottom Spread**: Return spread between top and bottom ranked symbols
- **Portfolio Turnover**: How much the top portfolio changes over time
- **Cost-Adjusted Spread**: Net returns after transaction costs

See .claude/plans/cs-ranking-phase4-metrics.md for design details.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


"""Metrics functions for specialized models."""

def cs_metrics_by_time(y_true: np.ndarray, y_pred: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    """Calculate cross-sectional metrics per timestamp (true CS evaluation)."""
    try:
        from scipy.stats import spearmanr, pearsonr
        scipy_available = True
    except Exception:
        scipy_available = False
        
    ts = np.asarray(ts)
    ic_list, ric_list = [], []
    grp_sizes, grp_hits = [], []
    
    total_timestamps = len(np.unique(ts))
    skipped_timestamps = 0
    
    # Single pass through unique timestamps
    for t in np.unique(ts):
        m = (ts == t)
        if m.sum() <= 2:
            skipped_timestamps += 1
            continue
        y_t, pred_t = y_true[m], y_pred[m]
        
        # skip degenerate groups
        if np.std(y_t) < 1e-12 or np.std(pred_t) < 1e-12:
            skipped_timestamps += 1
            continue
        
        # Compute correlations
        if scipy_available:
            ic = pearsonr(y_t, pred_t)[0]
            ric = spearmanr(y_t, pred_t)[0]
        else:
            # Simple numpy fallback
            def _corr(a, b):
                if a.size < 2: return np.nan
                return float(np.corrcoef(a, b)[0,1])
            ic = _corr(y_t, pred_t)
            # Rank-IC fallback
            ric = _corr(y_t.argsort().argsort(), pred_t.argsort().argsort())
        
        if not np.isnan(ic): ic_list.append(ic)
        if not np.isnan(ric): ric_list.append(ric)
        
        # Hit rate per timestamp: majority vote on direction
        hit_rate_t = float(np.mean(np.sign(y_t) == np.sign(pred_t)))
        grp_sizes.append(m.sum())
        grp_hits.append(hit_rate_t)
    
    # Weight hit rate by group size
    hit_rate = float(np.average(grp_hits, weights=grp_sizes)) if grp_sizes else 0.0
    
    # Log fraction of skipped timestamps
    if total_timestamps > 0:
        skipped_fraction = skipped_timestamps / total_timestamps
        if skipped_fraction > 0.1:  # Log if more than 10% skipped
            logger.warning(f"⚠️  Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
        else:
            logger.info(f"📊 Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
    
    # Calculate IC_IR (Information Ratio)
    ic_arr = np.asarray(ic_list)
    ic_ir = float(ic_arr.mean() / (ic_arr.std(ddof=1) + 1e-12)) if ic_list else 0.0
    
    return {
        "mean_IC": float(np.mean(ic_list)) if ic_list else 0.0,
        "mean_RankIC": float(np.mean(ric_list)) if ric_list else 0.0,
        "IC_IR": ic_ir,
        "n_times": int(len(ic_list)),
        "hit_rate": hit_rate,
        "skipped_timestamps": skipped_timestamps,
        "total_timestamps": total_timestamps
    }


# ==============================================================================
# CROSS-SECTIONAL RANKING METRICS (Phase 4)
# ==============================================================================


def top_bottom_spread(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: Optional[np.ndarray] = None,
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
    min_symbols: int = 20,
    annualization_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute return spread between top and bottom ranked symbols.

    Simulates the actual trading strategy:
    - Long top decile (by predicted score)
    - Short bottom decile
    - Measure return difference

    Supports two input formats:
    1. Flattened: scores/returns (N,), requires `ts` for grouping
    2. Matrix: scores/returns (T, M) where T=timestamps, M=symbols

    Args:
        scores: Model predictions - (T, M) matrix or (N,) flattened
        returns: Actual forward returns - same shape as scores
        mask: Valid symbol mask (T, M) or None for matrix input.
              For flattened input, use NaN in returns to mark invalid.
        top_pct: Fraction of symbols to consider "top" (default: 10%)
        bottom_pct: Fraction of symbols to consider "bottom" (default: 10%)
        min_symbols: Minimum valid symbols per timestamp to include
        annualization_factor: Factor for Sharpe calculation. If None, uses
                              sqrt(78 * 252) assuming 5-min bars.

    Returns:
        Dict with:
            'spread_mean': Mean (top_return - bottom_return) per period
            'spread_std': Std of spread
            'spread_sharpe': Annualized Sharpe of spread
            'top_return_mean': Mean return of top decile
            'bottom_return_mean': Mean return of bottom decile
            'spread_series': List of per-period spreads (for plotting)
            'n_periods': Number of valid periods
    """
    scores = np.asarray(scores)
    returns = np.asarray(returns)

    # Handle 1D input (convert to 2D with single row)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        returns = returns.reshape(1, -1)
        if mask is not None:
            mask = np.asarray(mask).reshape(1, -1)

    T, M = scores.shape

    if mask is None:
        # Default mask: valid where both scores and returns are finite
        mask = np.isfinite(scores) & np.isfinite(returns)
    else:
        mask = np.asarray(mask) > 0.5

    # Default annualization: 5-min bars, 78 bars/day, 252 trading days
    if annualization_factor is None:
        annualization_factor = np.sqrt(78 * 252)

    spreads = []
    top_returns = []
    bottom_returns = []

    for t in range(T):
        valid = mask[t] & np.isfinite(scores[t]) & np.isfinite(returns[t])
        n_valid = int(valid.sum())

        if n_valid < min_symbols:
            continue

        s = scores[t, valid]
        r = returns[t, valid]

        k_top = max(1, int(n_valid * top_pct))
        k_bottom = max(1, int(n_valid * bottom_pct))

        # Rank by score (descending)
        ranking = np.argsort(s)[::-1]

        top_r = float(r[ranking[:k_top]].mean())
        bottom_r = float(r[ranking[-k_bottom:]].mean())

        spreads.append(top_r - bottom_r)
        top_returns.append(top_r)
        bottom_returns.append(bottom_r)

    if len(spreads) == 0:
        return {
            "spread_mean": 0.0,
            "spread_std": 0.0,
            "spread_sharpe": 0.0,
            "top_return_mean": 0.0,
            "bottom_return_mean": 0.0,
            "spread_series": [],
            "n_periods": 0,
        }

    spreads_arr = np.array(spreads)
    spread_mean = float(np.mean(spreads_arr))
    spread_std = float(np.std(spreads_arr, ddof=1)) if len(spreads_arr) > 1 else 0.0

    return {
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "spread_sharpe": float(spread_mean / (spread_std + 1e-10) * annualization_factor),
        "top_return_mean": float(np.mean(top_returns)),
        "bottom_return_mean": float(np.mean(bottom_returns)),
        "spread_series": spreads,
        "n_periods": len(spreads),
    }


def portfolio_turnover(
    scores: np.ndarray,
    mask: Optional[np.ndarray] = None,
    top_pct: float = 0.1,
    min_symbols: int = 20,
) -> Dict[str, Any]:
    """
    Compute turnover of top-ranked portfolio over time.

    High turnover = high transaction costs. This metric helps evaluate
    whether ranking signals are stable enough for practical trading.

    Args:
        scores: Model predictions (T, M) matrix
        mask: Valid symbol mask (T, M), or None to use non-NaN as valid
        top_pct: Fraction to consider "top" portfolio
        min_symbols: Minimum valid symbols per timestamp

    Returns:
        Dict with:
            'turnover_mean': Average fraction of portfolio changed per period
            'turnover_series': Per-period turnover values
            'n_periods': Number of periods with turnover computed
    """
    scores = np.asarray(scores)

    # Handle 1D input
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        if mask is not None:
            mask = np.asarray(mask).reshape(1, -1)

    T, M = scores.shape

    if mask is None:
        mask = np.isfinite(scores)
    else:
        mask = np.asarray(mask) > 0.5

    turnovers = []
    prev_top_set: Optional[set] = None

    for t in range(T):
        valid = mask[t] & np.isfinite(scores[t])
        n_valid = int(valid.sum())

        if n_valid < min_symbols:
            continue

        s = scores[t, valid]
        valid_indices = np.where(valid)[0]
        k_top = max(1, int(n_valid * top_pct))

        # Get top symbols by score
        ranking = np.argsort(s)[::-1]
        top_indices = set(valid_indices[ranking[:k_top]].tolist())

        if prev_top_set is not None:
            # Turnover = fraction NOT in previous top
            overlap = len(top_indices & prev_top_set)
            turnover = 1.0 - (overlap / len(top_indices))
            turnovers.append(turnover)

        prev_top_set = top_indices

    if len(turnovers) == 0:
        return {
            "turnover_mean": 0.0,
            "turnover_series": [],
            "n_periods": 0,
        }

    return {
        "turnover_mean": float(np.mean(turnovers)),
        "turnover_series": turnovers,
        "n_periods": len(turnovers),
    }


def cost_adjusted_spread(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: Optional[np.ndarray] = None,
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
    cost_per_trade_bps: float = 5.0,
    min_symbols: int = 20,
) -> Dict[str, Any]:
    """
    Compute spread minus estimated transaction costs.

    Combines top_bottom_spread and portfolio_turnover to estimate
    realistic profitability after costs.

    Args:
        scores: Model predictions (T, M) matrix
        returns: Actual forward returns (T, M) matrix
        mask: Valid symbol mask (T, M)
        top_pct: Fraction for top portfolio
        bottom_pct: Fraction for bottom portfolio
        cost_per_trade_bps: Cost per trade in basis points (default: 5 bps)
        min_symbols: Minimum valid symbols per timestamp

    Returns:
        Dict with:
            'gross_spread': Mean spread before costs
            'turnover': Mean portfolio turnover
            'cost_per_period': Estimated cost per period
            'net_spread': Spread after costs
            'cost_drag_pct': Percentage of gross spread lost to costs
    """
    spread_result = top_bottom_spread(
        scores, returns, mask, top_pct, bottom_pct, min_symbols
    )
    turnover_result = portfolio_turnover(scores, mask, top_pct, min_symbols)

    # Cost = turnover * 2 (enter + exit) * cost_bps
    # Applied to both long and short legs
    cost_per_period = (
        turnover_result["turnover_mean"] * 2 * 2 * (cost_per_trade_bps / 10000)
    )

    gross_spread = spread_result["spread_mean"]
    net_spread = gross_spread - cost_per_period

    # What fraction of gross spread is lost to costs?
    cost_drag_pct = (cost_per_period / (abs(gross_spread) + 1e-10)) * 100

    return {
        "gross_spread": gross_spread,
        "turnover": turnover_result["turnover_mean"],
        "cost_per_period": cost_per_period,
        "net_spread": net_spread,
        "cost_drag_pct": float(cost_drag_pct),
    }


def spearman_ic_matrix(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_symbols: int = 10,
) -> Dict[str, Any]:
    """
    Compute Spearman IC (rank correlation) per timestamp for matrix input.

    This is equivalent to `cs_metrics_by_time()` but accepts (T, M) matrix
    input directly, matching the CrossSectionalDataset output format.

    Args:
        scores: Model predictions (T, M) matrix
        returns: Actual forward returns (T, M) matrix
        mask: Valid symbol mask (T, M)
        min_symbols: Minimum valid symbols per timestamp

    Returns:
        Dict with:
            'ic_mean': Mean Spearman IC across timestamps
            'ic_std': Std of IC
            'ic_ir': IC / std (Information Ratio)
            'ic_hit_rate': Fraction of timestamps with positive IC
            'ic_series': List of per-timestamp ICs
            'n_times': Number of valid timestamps
    """
    try:
        from scipy.stats import spearmanr
        scipy_available = True
    except ImportError:
        scipy_available = False

    scores = np.asarray(scores)
    returns = np.asarray(returns)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        returns = returns.reshape(1, -1)
        if mask is not None:
            mask = np.asarray(mask).reshape(1, -1)

    T, M = scores.shape

    if mask is None:
        mask = np.isfinite(scores) & np.isfinite(returns)
    else:
        mask = np.asarray(mask) > 0.5

    ics = []

    for t in range(T):
        valid = mask[t] & np.isfinite(scores[t]) & np.isfinite(returns[t])
        n_valid = int(valid.sum())

        if n_valid < min_symbols:
            continue

        s = scores[t, valid]
        r = returns[t, valid]

        # Skip if no variance
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue

        if scipy_available:
            ic, _ = spearmanr(s, r)
        else:
            # Numpy fallback using rank correlation
            s_ranks = s.argsort().argsort()
            r_ranks = r.argsort().argsort()
            ic = float(np.corrcoef(s_ranks, r_ranks)[0, 1])

        if not np.isnan(ic):
            ics.append(ic)

    if len(ics) == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_hit_rate": 0.0,
            "ic_series": [],
            "n_times": 0,
        }

    ics_arr = np.array(ics)
    ic_mean = float(np.mean(ics_arr))
    ic_std = float(np.std(ics_arr, ddof=1)) if len(ics_arr) > 1 else 0.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": float(ic_mean / (ic_std + 1e-10)),
        "ic_hit_rate": float(np.mean(ics_arr > 0)),
        "ic_series": ics,
        "n_times": len(ics),
    }


def compute_ranking_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute full suite of cross-sectional ranking metrics.

    This is the main entry point for ranking evaluation. Returns comprehensive
    metrics for model comparison aligned with trading objectives.

    Args:
        scores: Model predictions (T, M) matrix or (N,) flattened with ts
        returns: Actual forward returns, same shape as scores
        mask: Valid symbol mask (T, M), or None
        config: Optional configuration:
            - top_pct: Fraction for top portfolio (default: 0.1)
            - bottom_pct: Fraction for bottom portfolio (default: 0.1)
            - cost_per_trade_bps: Transaction cost in bps (default: 5.0)
            - min_symbols: Minimum symbols per timestamp (default: 20)
            - annualization_factor: For Sharpe calculation (default: sqrt(78*252))

    Returns:
        Dict with primary and detailed metrics:
            Primary metrics (for quick comparison):
                'spearman_ic': Mean IC across timestamps
                'ic_ir': Information Ratio
                'ic_hit_rate': Fraction of positive IC timestamps
                'spread': Mean top-bottom spread
                'spread_sharpe': Annualized Sharpe of spread
                'net_spread': Spread after transaction costs
                'turnover': Mean portfolio turnover

            Detailed results (for analysis):
                'details': {
                    'ic': Full IC results dict
                    'spread': Full spread results dict
                    'turnover': Full turnover results dict
                    'cost': Full cost-adjusted results dict
                }
    """
    config = config or {}
    top_pct = config.get("top_pct", 0.1)
    bottom_pct = config.get("bottom_pct", 0.1)
    cost_bps = config.get("cost_per_trade_bps", 5.0)
    min_symbols = config.get("min_symbols", 20)
    annualization = config.get("annualization_factor", None)

    # Compute all metrics
    ic_result = spearman_ic_matrix(scores, returns, mask, min_symbols=min_symbols)
    spread_result = top_bottom_spread(
        scores, returns, mask, top_pct, bottom_pct, min_symbols, annualization
    )
    turnover_result = portfolio_turnover(scores, mask, top_pct, min_symbols)
    cost_result = cost_adjusted_spread(
        scores, returns, mask, top_pct, bottom_pct, cost_bps, min_symbols
    )

    return {
        # Primary metrics (flat for easy logging/comparison)
        "spearman_ic": ic_result["ic_mean"],
        "ic_ir": ic_result["ic_ir"],
        "ic_hit_rate": ic_result["ic_hit_rate"],
        "spread": spread_result["spread_mean"],
        "spread_sharpe": spread_result["spread_sharpe"],
        "net_spread": cost_result["net_spread"],
        "turnover": turnover_result["turnover_mean"],
        # Detailed results (for deeper analysis)
        "details": {
            "ic": ic_result,
            "spread": spread_result,
            "turnover": turnover_result,
            "cost": cost_result,
        },
    }


def compute_ranking_metrics_from_flat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ts: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute ranking metrics from flattened arrays with timestamp column.

    This is a convenience wrapper that converts the flattened format
    (used by existing cs_metrics_by_time) to matrix format for the
    full ranking metrics suite.

    Args:
        y_true: Actual returns, shape (N,)
        y_pred: Predicted scores, shape (N,)
        ts: Timestamp values, shape (N,)
        config: Optional metric configuration

    Returns:
        Same as compute_ranking_metrics()
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ts = np.asarray(ts)

    # Get unique timestamps in sorted order
    unique_ts = np.unique(ts)
    unique_ts = np.sort(unique_ts)
    T = len(unique_ts)

    # Find max symbols per timestamp
    ts_to_idx = {t: i for i, t in enumerate(unique_ts)}
    counts = np.zeros(T, dtype=int)
    for t_val in ts:
        counts[ts_to_idx[t_val]] += 1
    M = int(counts.max())

    # Build (T, M) matrices with NaN padding
    scores = np.full((T, M), np.nan, dtype=np.float64)
    returns = np.full((T, M), np.nan, dtype=np.float64)

    # Track position within each timestamp
    pos = np.zeros(T, dtype=int)

    for i in range(len(ts)):
        t_idx = ts_to_idx[ts[i]]
        m_idx = pos[t_idx]
        scores[t_idx, m_idx] = y_pred[i]
        returns[t_idx, m_idx] = y_true[i]
        pos[t_idx] += 1

    # Mask is where we have valid data
    mask = np.isfinite(scores) & np.isfinite(returns)

    return compute_ranking_metrics(scores, returns, mask, config)

