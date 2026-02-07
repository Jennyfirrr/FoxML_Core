# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Sectional Target Construction

Transforms raw forward returns into cross-sectionally normalized targets suitable for
ranking-based training. Supports three normalization methods:

1. **CS Percentile Rank** (recommended): Converts returns to percentile ranks [0, 1]
   - Bounded, robust, directly encodes ordering
   - Best for pairwise/listwise ranking losses

2. **CS Z-Score**: Robust z-score using median and MAD
   - Preserves magnitude information
   - Good when signal strength matters

3. **Vol-Scaled CS**: Volatility-adjusted cross-sectional demeaned returns
   - Risk-adjusted relative performance
   - Standard practice in cross-sectional equity

All functions operate per-timestamp to ensure no future data leakage.

Usage:
    from TRAINING.common.targets.cross_sectional import (
        compute_cs_percentile_target,
        compute_cs_zscore_target,
        compute_vol_scaled_cs_target,
        CrossSectionalTargetType,
    )

    # Percentile rank target (recommended)
    y = compute_cs_percentile_target(df, return_col='fwd_ret_5m')

See .claude/plans/cs-ranking-phase1-targets.md for design details.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from TRAINING.common.exceptions import DataIntegrityError

logger = logging.getLogger(__name__)


class CrossSectionalTargetType(str, Enum):
    """Target normalization type for cross-sectional ranking."""

    CS_PERCENTILE = "cs_percentile"
    CS_ZSCORE = "cs_zscore"
    VOL_SCALED = "vol_scaled"

    @classmethod
    def from_string(cls, value: str) -> "CrossSectionalTargetType":
        """
        Convert string to CrossSectionalTargetType enum.

        Args:
            value: String value

        Returns:
            CrossSectionalTargetType enum value

        Raises:
            ValueError: If value is not valid
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid = [t.value for t in cls]
            raise ValueError(
                f"Invalid cross_sectional target type: '{value}'. Must be one of: {valid}"
            )


def compute_cs_percentile_target(
    df: pd.DataFrame,
    return_col: str = "fwd_ret_5m",
    time_col: str = "ts",
    residualize: bool = True,
    winsorize_pct: Tuple[float, float] = (0.01, 0.99),
    min_symbols: int = 5,
) -> pd.Series:
    """
    Compute cross-sectional percentile rank target.

    For each timestamp t:
    1. Get forward returns for all symbols
    2. Optionally residualize (subtract cross-sectional mean)
    3. Convert to percentile rank (0, 1)
    4. Winsorize extremes

    Percentile ranking is **recommended** for ranking losses because:
    - Bounded [0, 1] - no outliers blowing up gradients
    - Directly encodes what we trade: ordering
    - Robust to fat tails without additional clipping

    Args:
        df: DataFrame with columns [time_col, 'symbol', return_col]
        return_col: Forward return column name
        time_col: Timestamp column name
        residualize: If True, subtract cross-sectional mean first
        winsorize_pct: (lower, upper) percentile bounds for clipping (default: 1%/99%)
        min_symbols: Minimum symbols required per timestamp (groups with fewer are NaN)

    Returns:
        Series of percentile targets [0, 1], same index as df.
        Returns NaN for timestamps with fewer than min_symbols.

    Raises:
        DataIntegrityError: If required columns are missing
    """
    _validate_dataframe(df, return_col, time_col)

    # Pre-allocate result array with same index as input
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)

    # Process each timestamp group
    for ts_value, group in df.groupby(time_col, sort=True):
        r = group[return_col].values.astype(np.float64)
        n_valid = np.sum(~np.isnan(r))

        # Skip timestamps with too few symbols
        if n_valid < min_symbols:
            continue

        # Step 1: Residualize (subtract cross-sectional mean)
        if residualize:
            r = r - np.nanmean(r)

        # Step 2: Rank to percentile using scipy.stats.rankdata
        # nan_policy='omit' assigns NaN values rank of NaN
        # Using average ranking for ties (standard behavior)
        ranks = scipy_stats.rankdata(r, nan_policy="omit")

        # Convert ranks to percentiles in (0, 1) range
        # Using (rank) / (n_valid + 1) to avoid exact 0 or 1
        percentiles = ranks / (n_valid + 1)

        # Handle NaN positions (rankdata returns NaN for NaN inputs)
        nan_mask = np.isnan(r)
        percentiles[nan_mask] = np.nan

        # Step 3: Winsorize (clip to percentile bounds)
        percentiles = np.clip(percentiles, winsorize_pct[0], winsorize_pct[1])

        # Assign to result
        result.loc[group.index] = percentiles

    if len(result) > 0:
        values = result.values
        logger.debug(
            f"Computed CS percentile targets: "
            f"mean={np.nanmean(values):.4f}, std={np.nanstd(values):.4f}, "
            f"nan_pct={np.isnan(values).mean():.2%}"
        )

    return result


def compute_cs_zscore_target(
    df: pd.DataFrame,
    return_col: str = "fwd_ret_5m",
    time_col: str = "ts",
    residualize: bool = True,
    winsorize_std: float = 3.0,
    min_symbols: int = 5,
    use_robust: bool = True,
) -> pd.Series:
    """
    Compute cross-sectional z-score target.

    For each timestamp t:
    1. Get forward returns for all symbols
    2. Optionally residualize (subtract mean)
    3. Compute z-score using median/MAD (robust) or mean/std
    4. Winsorize to [-winsorize_std, +winsorize_std]

    Z-score targets preserve magnitude information, which can be useful when
    the signal strength matters for position sizing.

    Args:
        df: DataFrame with columns [time_col, 'symbol', return_col]
        return_col: Forward return column name
        time_col: Timestamp column name
        residualize: If True, subtract cross-sectional mean first
        winsorize_std: Maximum absolute z-score (values beyond are clipped)
        min_symbols: Minimum symbols required per timestamp
        use_robust: If True, use median/MAD instead of mean/std (recommended)

    Returns:
        Series of z-score targets, same index as df.
        Returns NaN for timestamps with fewer than min_symbols.

    Raises:
        DataIntegrityError: If required columns are missing
    """
    _validate_dataframe(df, return_col, time_col)

    # MAD scale factor to make it comparable to std for normal distributions
    # For normal distribution: std ≈ 1.4826 * MAD
    MAD_SCALE = 1.4826

    # Pre-allocate result array with same index as input
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)

    # Process each timestamp group
    for ts_value, group in df.groupby(time_col, sort=True):
        r = group[return_col].values.astype(np.float64)
        n_valid = np.sum(~np.isnan(r))

        # Skip timestamps with too few symbols
        if n_valid < min_symbols:
            continue

        # Step 1: Residualize (subtract mean)
        if residualize:
            r = r - np.nanmean(r)

        # Step 2: Compute z-score
        if use_robust:
            # Robust z-score using median and MAD
            median = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - median))
            # Scale MAD to be comparable to std, add epsilon for numerical stability
            scale = mad * MAD_SCALE + 1e-10
            z = (r - median) / scale
        else:
            # Standard z-score using mean and std
            mean = np.nanmean(r)
            std = np.nanstd(r) + 1e-10
            z = (r - mean) / std

        # Step 3: Winsorize
        z = np.clip(z, -winsorize_std, winsorize_std)

        # Assign to result
        result.loc[group.index] = z

    if len(result) > 0:
        values = result.values
        logger.debug(
            f"Computed CS z-score targets: "
            f"mean={np.nanmean(values):.4f}, std={np.nanstd(values):.4f}, "
            f"nan_pct={np.isnan(values).mean():.2%}"
        )

    return result


def compute_vol_scaled_cs_target(
    df: pd.DataFrame,
    return_col: str = "fwd_ret_5m",
    time_col: str = "ts",
    vol_col: str = "rolling_vol_20",
    winsorize_pct: Tuple[float, float] = (0.01, 0.99),
    min_symbols: int = 5,
) -> pd.Series:
    """
    Compute volatility-scaled, cross-section demeaned target.

    For each timestamp t:
    1. Compute vol-scaled returns: r_scaled = r / vol
    2. Cross-sectionally demean: r_demeaned = r_scaled - mean_cs(r_scaled)
    3. Winsorize extremes

    This produces "how much did you beat the universe, per unit risk" - a
    standard measure in cross-sectional equity analysis.

    **Note**: This requires a pre-computed volatility column (e.g., rolling_vol_20).
    The volatility should be computed using past data only to avoid leakage.

    Args:
        df: DataFrame with columns [time_col, 'symbol', return_col, vol_col]
        return_col: Forward return column name
        time_col: Timestamp column name
        vol_col: Pre-computed volatility column name (must exist in df)
        winsorize_pct: (lower, upper) percentile for clipping
        min_symbols: Minimum symbols required per timestamp

    Returns:
        Series of vol-scaled CS-demeaned targets, same index as df.
        Returns NaN for timestamps with fewer than min_symbols.

    Raises:
        DataIntegrityError: If required columns are missing (including vol_col)
    """
    _validate_dataframe(df, return_col, time_col, vol_col=vol_col)

    # Pre-allocate result array with same index as input
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)

    # Process each timestamp group
    for ts_value, group in df.groupby(time_col, sort=True):
        r = group[return_col].values.astype(np.float64)
        vol = group[vol_col].values.astype(np.float64)
        n_valid = np.sum(~np.isnan(r) & ~np.isnan(vol) & (vol > 0))

        # Skip timestamps with too few symbols
        if n_valid < min_symbols:
            continue

        # Step 1: Vol-scale (add epsilon to avoid division by zero)
        r_scaled = r / (vol + 1e-10)

        # Step 2: Cross-sectional demean
        r_demeaned = r_scaled - np.nanmean(r_scaled)

        # Step 3: Winsorize using percentile bounds computed within group
        lower = np.nanpercentile(r_demeaned, winsorize_pct[0] * 100)
        upper = np.nanpercentile(r_demeaned, winsorize_pct[1] * 100)
        r_clipped = np.clip(r_demeaned, lower, upper)

        # Assign to result
        result.loc[group.index] = r_clipped

    if len(result) > 0:
        values = result.values
        logger.debug(
            f"Computed vol-scaled CS targets: "
            f"mean={np.nanmean(values):.4f}, std={np.nanstd(values):.4f}, "
            f"nan_pct={np.isnan(values).mean():.2%}"
        )

    return result


def compute_cs_target(
    df: pd.DataFrame,
    target_type: CrossSectionalTargetType | str = CrossSectionalTargetType.CS_PERCENTILE,
    return_col: str = "fwd_ret_5m",
    time_col: str = "ts",
    vol_col: Optional[str] = None,
    residualize: bool = True,
    winsorize_pct: Tuple[float, float] = (0.01, 0.99),
    winsorize_std: float = 3.0,
    min_symbols: int = 5,
) -> pd.Series:
    """
    Unified interface for computing cross-sectional targets.

    This is the main entry point for CS target construction. Use this function
    when the target type is determined by config.

    Args:
        df: DataFrame with required columns
        target_type: Target type (CrossSectionalTargetType enum or string)
        return_col: Forward return column name
        time_col: Timestamp column name
        vol_col: Volatility column (required for vol_scaled type)
        residualize: Whether to subtract cross-sectional mean (for percentile/zscore)
        winsorize_pct: Percentile bounds for winsorization (percentile and vol_scaled)
        winsorize_std: Z-score bounds for winsorization (zscore only)
        min_symbols: Minimum symbols per timestamp

    Returns:
        Series of CS-normalized targets

    Raises:
        DataIntegrityError: If required columns missing
        ValueError: If invalid target_type or vol_col missing for vol_scaled
    """
    if isinstance(target_type, str):
        target_type = CrossSectionalTargetType.from_string(target_type)

    if target_type == CrossSectionalTargetType.CS_PERCENTILE:
        return compute_cs_percentile_target(
            df=df,
            return_col=return_col,
            time_col=time_col,
            residualize=residualize,
            winsorize_pct=winsorize_pct,
            min_symbols=min_symbols,
        )
    elif target_type == CrossSectionalTargetType.CS_ZSCORE:
        return compute_cs_zscore_target(
            df=df,
            return_col=return_col,
            time_col=time_col,
            residualize=residualize,
            winsorize_std=winsorize_std,
            min_symbols=min_symbols,
        )
    elif target_type == CrossSectionalTargetType.VOL_SCALED:
        if vol_col is None:
            raise ValueError(
                "vol_col is required for vol_scaled target type. "
                "Provide pre-computed rolling volatility column."
            )
        return compute_vol_scaled_cs_target(
            df=df,
            return_col=return_col,
            time_col=time_col,
            vol_col=vol_col,
            winsorize_pct=winsorize_pct,
            min_symbols=min_symbols,
        )
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def _validate_dataframe(
    df: pd.DataFrame,
    return_col: str,
    time_col: str,
    vol_col: Optional[str] = None,
) -> None:
    """
    Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        return_col: Required return column
        time_col: Required timestamp column
        vol_col: Optional volatility column

    Raises:
        DataIntegrityError: If required columns are missing
    """
    required_cols = [time_col, return_col]
    if vol_col is not None:
        required_cols.append(vol_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DataIntegrityError(
            message=f"Missing required columns for CS target computation: {missing}",
            error_code="DATA_MISSING_REQUIRED",
            column=str(missing),
            context={
                "required_columns": required_cols,
                "available_columns": list(df.columns),
            },
        )

    # Warn if return column has many NaN values
    nan_pct = df[return_col].isna().mean()
    if nan_pct > 0.5:
        logger.warning(
            f"Return column '{return_col}' has {nan_pct:.1%} NaN values. "
            "This may result in many skipped timestamps."
        )
