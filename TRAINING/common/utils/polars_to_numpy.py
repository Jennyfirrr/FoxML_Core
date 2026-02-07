# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Polars-to-NumPy Column Extraction Helper

This module provides a helper for extracting single columns from Polars DataFrames
to numpy arrays. This is used to extract the target column early in the pipeline
before the main Polars → Pandas conversion.

NOTE: The main data conversion uses Polars → Pandas → numpy (via to_pandas()).
Direct Polars → numpy conversion for feature matrices was attempted but caused
higher memory usage than the Pandas path. See .claude/plans/polars-native-memory-optimization.md
for details on why this approach failed.
"""

import logging
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def polars_extract_column_as_numpy(
    pl_df: pl.DataFrame,
    col_name: str,
    *,
    dtype: Optional[np.dtype] = None,
    replace_inf: bool = True,
) -> np.ndarray:
    """
    Extract a single column from Polars DataFrame as numpy array.

    This is used to extract the target column from Polars before converting
    the full DataFrame to Pandas. Extracting a single column is memory-efficient
    and doesn't cause the issues seen with full feature matrix extraction.

    Args:
        pl_df: Polars DataFrame
        col_name: Column name to extract
        dtype: Optional numpy dtype (default: float32 for numeric, object for strings)
        replace_inf: If True, replace inf/-inf with nan

    Returns:
        1D numpy array

    Raises:
        ValueError: If column not found in DataFrame
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
