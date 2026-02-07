# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Memory-Efficient Streaming Data Preparation

This module provides streaming data preparation that avoids loading all data
into memory at once. Key optimizations:

1. Stream from parquet files directly using Polars lazy evaluation
2. Process symbols incrementally, releasing memory after each
3. Pre-allocate numpy arrays instead of concat operations
4. Use memory-mapped arrays for very large datasets

Usage:
    from TRAINING.training_strategies.execution.data_preparation_streaming import (
        prepare_training_data_streaming
    )

    result = prepare_training_data_streaming(
        data_dir="/path/to/data",
        symbols=["AAPL", "MSFT", ...],
        target="fwd_ret_5m",
        feature_names=["feat1", "feat2", ...],
    )
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# DETERMINISM: Use sorted glob for filesystem enumeration
from TRAINING.common.utils.determinism_ordering import glob_sorted

logger = logging.getLogger(__name__)

# Check if Polars is available
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


def prepare_training_data_streaming(
    data_dir: str,
    symbols: List[str],
    target: str,
    feature_names: Optional[List[str]] = None,
    interval: str = "5m",
    min_cs: int = 10,
    max_samples_per_symbol: Optional[int] = None,
    max_total_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Memory-efficient streaming data preparation.

    Instead of loading all data into a giant DataFrame, this function:
    1. Scans parquet files lazily
    2. Processes one symbol at a time
    3. Appends to pre-allocated numpy arrays
    4. Releases memory after each symbol

    Args:
        data_dir: Directory containing parquet files
        symbols: List of symbols to load
        target: Target column name
        feature_names: List of feature columns (auto-detected if None)
        interval: Data interval (e.g., "5m")
        min_cs: Minimum cross-sectional samples per timestamp
        max_samples_per_symbol: Maximum rows per symbol
        max_total_samples: Maximum total rows across all symbols

    Returns:
        Dict with keys: X, y, timestamps, symbols, feature_names, metadata
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is required for streaming data preparation")

    logger.info(f"🚀 Streaming data preparation for {len(symbols)} symbols")
    logger.info(f"   Target: {target}")
    logger.info(f"   Max samples/symbol: {max_samples_per_symbol or 'unlimited'}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Phase 1: Scan first symbol to discover schema
    first_symbol = sorted(symbols)[0]
    symbol_path = data_path / f"symbol={first_symbol}"

    if not symbol_path.exists():
        # Try without partition
        parquet_files = glob_sorted(data_path, "*.parquet")
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        schema_file = parquet_files[0]
    else:
        schema_file = glob_sorted(symbol_path, "*.parquet")[0]

    # Get schema lazily
    schema_scan = pl.scan_parquet(schema_file)
    all_columns = schema_scan.columns

    # Auto-detect features if not provided
    if feature_names is None:
        exclude_prefixes = ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_']
        exclude_cols = {'symbol', 'timestamp', 'ts', 'time', 'datetime'}
        feature_names = [
            col for col in all_columns
            if col not in exclude_cols
            and not any(col.startswith(p) for p in exclude_prefixes)
        ]

    # Verify target exists
    if target not in all_columns:
        raise ValueError(f"Target '{target}' not found in columns: {all_columns[:20]}...")

    logger.info(f"   Features: {len(feature_names)}")

    # Phase 2: Count total rows (lazy, fast)
    total_rows = 0
    symbol_row_counts = {}

    for symbol in sorted(symbols):
        symbol_path = data_path / f"symbol={symbol}"
        if symbol_path.exists():
            parquet_file = glob_sorted(symbol_path, "*.parquet")[0]
        else:
            continue

        # Fast row count without loading data
        lazy_df = pl.scan_parquet(parquet_file)
        count = lazy_df.select(pl.count()).collect().item()

        if max_samples_per_symbol:
            count = min(count, max_samples_per_symbol)

        symbol_row_counts[symbol] = count
        total_rows += count

        if max_total_samples and total_rows >= max_total_samples:
            total_rows = max_total_samples
            break

    logger.info(f"   Total rows to process: {total_rows:,}")

    # Phase 3: Pre-allocate arrays
    n_features = len(feature_names)

    X = np.empty((total_rows, n_features), dtype=np.float32)
    y = np.empty(total_rows, dtype=np.float32)
    timestamps = np.empty(total_rows, dtype='datetime64[ns]')
    symbol_ids = np.empty(total_rows, dtype='<U10')  # Fixed-length string

    # Phase 4: Stream data from each symbol
    current_idx = 0
    time_col = None

    for symbol in sorted(symbols):
        if symbol not in symbol_row_counts:
            continue

        symbol_path = data_path / f"symbol={symbol}"
        parquet_file = glob_sorted(symbol_path, "*.parquet")[0]

        # Determine columns to load
        cols_to_load = feature_names + [target]

        # Add time column if present
        for tc in ['timestamp', 'ts', 'time', 'datetime']:
            if tc in all_columns:
                cols_to_load.append(tc)
                time_col = tc
                break

        # Load only needed columns
        lazy_df = pl.scan_parquet(parquet_file).select(cols_to_load)

        # Apply row limit if specified
        if max_samples_per_symbol:
            lazy_df = lazy_df.head(max_samples_per_symbol)

        # Collect (materialize) just this symbol
        df = lazy_df.collect()
        n_rows = len(df)

        if n_rows == 0:
            continue

        # Check if we'd exceed total limit
        if max_total_samples and current_idx + n_rows > max_total_samples:
            n_rows = max_total_samples - current_idx
            df = df.head(n_rows)

        # Extract to numpy (zero-copy where possible)
        end_idx = current_idx + n_rows

        # Features
        for i, feat in enumerate(feature_names):
            if feat in df.columns:
                X[current_idx:end_idx, i] = df[feat].to_numpy()
            else:
                X[current_idx:end_idx, i] = np.nan

        # Target
        y[current_idx:end_idx] = df[target].to_numpy()

        # Timestamps
        if time_col and time_col in df.columns:
            ts_values = df[time_col].to_numpy()
            timestamps[current_idx:end_idx] = ts_values

        # Symbol IDs
        symbol_ids[current_idx:end_idx] = symbol

        current_idx = end_idx

        # Release memory
        del df
        gc.collect()

        logger.debug(f"   Processed {symbol}: {n_rows:,} rows (total: {current_idx:,})")

        if max_total_samples and current_idx >= max_total_samples:
            break

    # Trim arrays if we got fewer rows than expected
    if current_idx < total_rows:
        X = X[:current_idx]
        y = y[:current_idx]
        timestamps = timestamps[:current_idx]
        symbol_ids = symbol_ids[:current_idx]

    logger.info(f"✅ Loaded {current_idx:,} samples with {n_features} features")
    logger.info(f"   Memory: X={X.nbytes / 1e9:.2f} GB, y={y.nbytes / 1e6:.1f} MB")

    # Phase 5: Apply min_cs filter (optional - may need cross-symbol view)
    if min_cs > 1 and time_col:
        # Count samples per timestamp
        unique_ts, ts_counts = np.unique(timestamps, return_counts=True)
        valid_ts = set(unique_ts[ts_counts >= min_cs])

        mask = np.array([t in valid_ts for t in timestamps])

        if mask.sum() < len(mask):
            X = X[mask]
            y = y[mask]
            timestamps = timestamps[mask]
            symbol_ids = symbol_ids[mask]
            logger.info(f"   After min_cs={min_cs} filter: {len(X):,} samples")

    return {
        'X': X,
        'y': y,
        'timestamps': timestamps,
        'symbols': symbol_ids,
        'feature_names': feature_names,
        'target': target,
        'metadata': {
            'n_symbols': len(symbol_row_counts),
            'n_samples': len(X),
            'n_features': len(feature_names),
            'time_column': time_col,
        }
    }


def estimate_memory_usage(
    data_dir: str,
    symbols: List[str],
    n_features: int = 500,
    max_samples_per_symbol: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate memory usage before loading data.

    Returns:
        Dict with memory estimates in GB
    """
    if not POLARS_AVAILABLE:
        return {'error': 'Polars not available'}

    data_path = Path(data_dir)
    total_rows = 0

    for symbol in symbols:
        symbol_path = data_path / f"symbol={symbol}"
        if not symbol_path.exists():
            continue

        parquet_file = glob_sorted(symbol_path, "*.parquet")[0]
        lazy_df = pl.scan_parquet(parquet_file)
        count = lazy_df.select(pl.count()).collect().item()

        if max_samples_per_symbol:
            count = min(count, max_samples_per_symbol)

        total_rows += count

    # Estimate memory (float32 = 4 bytes)
    x_memory_gb = (total_rows * n_features * 4) / 1e9
    y_memory_gb = (total_rows * 4) / 1e9
    ts_memory_gb = (total_rows * 8) / 1e9  # datetime64[ns]
    overhead_gb = 0.5  # Estimated overhead

    return {
        'total_rows': total_rows,
        'X_memory_gb': x_memory_gb,
        'y_memory_gb': y_memory_gb,
        'timestamps_memory_gb': ts_memory_gb,
        'overhead_gb': overhead_gb,
        'total_estimated_gb': x_memory_gb + y_memory_gb + ts_memory_gb + overhead_gb,
        'pandas_concat_estimated_gb': (total_rows * (n_features + 10) * 8 * 3) / 1e9,  # Why pandas is bad
    }
