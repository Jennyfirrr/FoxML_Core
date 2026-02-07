# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Polars-to-NumPy Conversion Helpers

Memory-efficient conversion from Polars DataFrames directly to sklearn-ready numpy arrays,
eliminating the Pandas intermediate step that causes memory spikes.

CRITICAL: The key to memory efficiency is IMMEDIATE RELEASE of DataFrames after extraction.
All functions that extract data should:
1. Select only needed columns first (reduces working set)
2. Extract to numpy
3. Explicitly delete and gc.collect() the source DataFrame

Usage:
    # Single-pass extraction (most memory efficient)
    X, y, symbols, time_vals, feature_names = extract_training_arrays(
        pl_df, feature_cols, target_col
    )
    # pl_df is released inside the function

Memory Impact:
    - Traditional: Polars (32GB) + Pandas (32GB) simultaneous = 64GB peak
    - This module: Polars (32GB) → numpy (32GB) sequential = 32GB peak
"""

import gc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def polars_to_sklearn_dense(
    pl_df: pl.DataFrame,
    feature_cols: List[str],
    imputation_strategy: str = "median",
    *,
    return_impute_values: bool = False,
) -> Tuple[np.ndarray, List[str], Optional[Dict[str, float]]]:
    """
    Convert Polars DataFrame directly to sklearn-ready numpy array.

    This function eliminates the Pandas intermediate step, reducing peak memory
    by avoiding simultaneous Polars + Pandas DataFrames in memory.

    Args:
        pl_df: Polars DataFrame containing features
        feature_cols: List of feature column names to extract
        imputation_strategy: Imputation strategy ("median", "mean", "zero")
        return_impute_values: If True, return the imputation values dict

    Returns:
        Tuple of:
            - X: numpy array (n_samples, n_features) with float32 dtype
            - final_features: List of feature names (may differ from input if columns dropped)
            - impute_values: Dict of {feature: impute_value} if return_impute_values=True, else None

    Raises:
        ValueError: If no valid features remain after filtering

    DETERMINISM:
        - Feature columns are always sorted before processing
        - Imputation uses deterministic Polars operations
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")

    # DETERMINISM: Sort feature columns to ensure consistent ordering
    sorted_features = sorted(feature_cols)

    # 1. Select only requested columns that exist in DataFrame
    available_cols = set(pl_df.columns)
    valid_features = [f for f in sorted_features if f in available_cols]

    if not valid_features:
        raise ValueError(
            f"None of the {len(feature_cols)} requested features exist in DataFrame. "
            f"Requested: {sorted_features[:10]}... "
            f"Available: {list(available_cols)[:10]}..."
        )

    missing = set(sorted_features) - set(valid_features)
    if missing:
        logger.warning(f"⚠️  {len(missing)} features not found in DataFrame: {sorted(missing)[:10]}...")

    # 2. Select features and cast to Float32 (handles non-numeric with strict=False)
    cast_exprs = []
    for col in valid_features:
        cast_exprs.append(
            pl.col(col).cast(pl.Float32, strict=False).alias(col)
        )

    selected_df = pl_df.select(cast_exprs)

    # 3. Replace inf/-inf with null (Polars way)
    replace_inf_exprs = []
    for col in valid_features:
        replace_inf_exprs.append(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    selected_df = selected_df.select(replace_inf_exprs)

    # 4. Drop columns that are entirely null
    null_counts = selected_df.select([
        pl.col(c).is_null().all().alias(c) for c in valid_features
    ]).row(0)

    all_null_cols = [
        valid_features[i] for i, is_all_null in enumerate(null_counts) if is_all_null
    ]

    if all_null_cols:
        logger.warning(f"🔧 Dropping {len(all_null_cols)} all-null columns: {all_null_cols[:10]}...")
        valid_features = [f for f in valid_features if f not in set(all_null_cols)]
        selected_df = selected_df.select(valid_features)

    if not valid_features:
        raise ValueError("All features became all-null after casting/inf replacement")

    # 5. Compute imputation values
    impute_values = None
    if imputation_strategy == "median":
        # Compute column medians
        impute_row = selected_df.select([
            pl.col(c).median().alias(c) for c in valid_features
        ]).row(0)
        impute_values = dict(zip(valid_features, impute_row))
    elif imputation_strategy == "mean":
        impute_row = selected_df.select([
            pl.col(c).mean().alias(c) for c in valid_features
        ]).row(0)
        impute_values = dict(zip(valid_features, impute_row))
    elif imputation_strategy == "zero":
        impute_values = {f: 0.0 for f in valid_features}
    else:
        raise ValueError(f"Unknown imputation strategy: {imputation_strategy}")

    # 6. Fill nulls with imputed values
    fill_exprs = []
    for col in valid_features:
        fill_val = impute_values.get(col, 0.0)
        # Handle case where median/mean is also null (all values were null)
        if fill_val is None or (isinstance(fill_val, float) and np.isnan(fill_val)):
            fill_val = 0.0
            impute_values[col] = 0.0
        # Use pl.lit() to ensure the fill value is properly typed
        fill_exprs.append(
            pl.col(col).fill_null(pl.lit(fill_val, dtype=pl.Float32)).alias(col)
        )

    filled_df = selected_df.select(fill_exprs)

    # 7. Extract to numpy
    # NOTE: Polars to_numpy() returns column-major by default, we need row-major for sklearn
    X = filled_df.to_numpy()

    # Ensure float32 dtype
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    # 8. Final safety check: replace any remaining NaN with column imputation values
    # This handles edge cases where Polars null handling differs from numpy NaN
    if np.any(np.isnan(X)):
        for i, col in enumerate(valid_features):
            col_data = X[:, i]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask):
                fill_val = impute_values.get(col, 0.0)
                if fill_val is None or np.isnan(fill_val):
                    fill_val = 0.0
                X[nan_mask, i] = fill_val

    logger.info(f"✅ Polars→numpy conversion: {X.shape[0]} samples × {X.shape[1]} features")

    if return_impute_values:
        return X, valid_features, impute_values
    else:
        return X, valid_features, None


def polars_cross_sectional_filter(
    pl_df: pl.DataFrame,
    time_col: str,
    min_cs: int,
) -> pl.DataFrame:
    """
    Filter Polars DataFrame to timestamps with at least min_cs symbols.

    Args:
        pl_df: Polars DataFrame with time and symbol columns
        time_col: Name of timestamp column
        min_cs: Minimum cross-sectional size required

    Returns:
        Filtered Polars DataFrame

    DETERMINISM:
        - Uses Polars filter expressions (deterministic)
    """
    if time_col not in pl_df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    return pl_df.filter(
        pl.len().over(time_col) >= min_cs
    )


def polars_cross_sectional_sample(
    pl_df: pl.DataFrame,
    time_col: str,
    max_samples: int,
    *,
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """
    Deterministic per-timestamp sampling in Polars.

    Args:
        pl_df: Polars DataFrame with time and symbol columns
        time_col: Name of timestamp column
        max_samples: Maximum samples per timestamp
        symbol_col: Name of symbol column

    Returns:
        Sampled Polars DataFrame (max_samples rows per timestamp)

    DETERMINISM:
        - Sorts by timestamp, then by symbol (alphabetical)
        - Takes head(max_samples) after sorting
        - Result is deterministic across runs
    """
    if time_col not in pl_df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    return (
        pl_df
        .sort([time_col, symbol_col])
        .group_by(time_col, maintain_order=True)
        .head(max_samples)
    )


def polars_extract_column_as_numpy(
    pl_df: pl.DataFrame,
    col_name: str,
    *,
    dtype: Optional[np.dtype] = None,
    replace_inf: bool = True,
) -> np.ndarray:
    """
    Extract a single column from Polars DataFrame as numpy array.

    Args:
        pl_df: Polars DataFrame
        col_name: Column name to extract
        dtype: Optional numpy dtype (default: float32 for numeric, object for strings)
        replace_inf: If True, replace inf/-inf with nan

    Returns:
        1D numpy array
    """
    if col_name not in pl_df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")

    # Replace infinities if requested (using select to apply expression)
    if replace_inf:
        col_dtype = pl_df.schema.get(col_name)
        if col_dtype in [pl.Float32, pl.Float64]:
            arr = pl_df.select(
                pl.when(pl.col(col_name).is_infinite())
                .then(None)
                .otherwise(pl.col(col_name))
                .alias(col_name)
            ).get_column(col_name).to_numpy()
        else:
            arr = pl_df.get_column(col_name).to_numpy()
    else:
        arr = pl_df.get_column(col_name).to_numpy()

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype == np.float64:
        # Default to float32 for memory efficiency
        arr = arr.astype(np.float32, copy=False)

    return arr


def polars_compute_row_nan_ratio(
    pl_df: pl.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Compute per-row NaN ratio for feature columns.

    Args:
        pl_df: Polars DataFrame
        feature_cols: List of feature column names

    Returns:
        1D numpy array of NaN ratios (0.0 to 1.0)
    """
    if not feature_cols:
        return np.zeros(len(pl_df), dtype=np.float32)

    # Filter to only columns that exist in the DataFrame
    valid_cols = [c for c in feature_cols if c in pl_df.columns]
    if not valid_cols:
        return np.zeros(len(pl_df), dtype=np.float32)

    # Simple approach: select the feature columns and compute null count
    # Using horizontal sum via struct + unnest
    feature_df = pl_df.select(valid_cols)

    # Convert to numpy and compute NaN ratio per row
    arr = feature_df.to_numpy()

    # Handle the case where arr might not be float (cast to float for isnan)
    if arr.dtype not in [np.float32, np.float64]:
        # For non-float arrays, check for None (which becomes nan after conversion)
        arr = arr.astype(np.float64)

    nan_counts = np.isnan(arr).sum(axis=1)
    return (nan_counts / len(valid_cols)).astype(np.float32)


def polars_valid_mask(
    pl_df: pl.DataFrame,
    target_col: str,
    feature_cols: List[str],
    max_nan_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute validity mask for rows (valid target and acceptable feature NaN ratio).

    Args:
        pl_df: Polars DataFrame
        target_col: Target column name
        feature_cols: List of feature column names
        max_nan_ratio: Maximum allowed NaN ratio per row (default 0.5 = 50%)

    Returns:
        Boolean numpy array (True = valid row)
    """
    # Target validity: not null and finite
    if target_col in pl_df.columns:
        target_valid = pl_df.select(
            (~pl.col(target_col).is_null() & pl.col(target_col).is_finite()).alias("_valid")
        ).get_column("_valid").to_numpy()
    else:
        target_valid = np.ones(len(pl_df), dtype=bool)

    # Feature validity: NaN ratio <= threshold
    nan_ratio = polars_compute_row_nan_ratio(pl_df, feature_cols)
    feature_valid = nan_ratio <= max_nan_ratio

    return target_valid & feature_valid
