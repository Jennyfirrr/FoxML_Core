# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Unified data loader with schema reading and column projection.

This module provides a single source of truth for data loading across all
pipeline stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING). It supports:

1. Schema-only reading: Get column names without loading data (~1ms/file)
2. Column projection: Load only specified columns (5-10x memory reduction)
3. Per-target loading: Convenience method for target + features + metadata

Memory Impact:
    - Current: 25 symbols × 500 cols × 500k rows = 85GB peak
    - With projection: 25 symbols × 100 cols × 500k rows = 17GB peak (5x reduction)

Example:
    ```python
    from TRAINING.data.loading.unified_loader import UnifiedDataLoader

    # Initialize loader
    loader = UnifiedDataLoader(data_dir="/data/prices", interval="5m")

    # Read schema only (no data loading)
    schemas = loader.read_schema(symbols=["AAPL", "GOOGL"])
    print(f"AAPL columns: {schemas['AAPL'][:5]}...")

    # Get columns common to all symbols
    common_cols = loader.get_common_columns(symbols=["AAPL", "GOOGL"])

    # Load with column projection
    mtf_data = loader.load_data(
        symbols=["AAPL", "GOOGL"],
        columns=["close", "volume", "rsi_14", "macd_signal"],
        max_rows_per_symbol=100000
    )

    # Load for a specific target (includes metadata columns automatically)
    mtf_data = loader.load_for_target(
        symbols=["AAPL", "GOOGL"],
        target="fwd_ret_60m",
        features=["close", "volume", "rsi_14"],
        max_rows_per_symbol=100000
    )
    ```

SST Compliance:
    - Deterministic: sorted symbols, sorted columns
    - Config access: Uses get_cfg() for any config values
    - Path construction: Uses pathlib throughout
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import pyarrow.parquet as pq

from TRAINING.common.utils.determinism_ordering import glob_sorted, iterdir_sorted

logger = logging.getLogger(__name__)

# Use Polars by default (can be disabled via env var)
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"

# Metadata columns that are ALWAYS included (required for pipeline operations)
METADATA_COLUMNS = frozenset(["ts", "timestamp", "symbol"])


class UnifiedDataLoader:
    """Unified data loader with schema reading and column projection.

    Single source of truth for data loading across all pipeline stages.
    Supports schema-only reading (fast) and column projection (memory-efficient).

    Attributes:
        data_dir: Path to data directory
        interval: Data interval (e.g., "5m", "1h")
        use_polars: Whether to use Polars for loading (faster, recommended)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        interval: str = "5m",
        use_polars: Optional[bool] = None,
    ):
        """Initialize UnifiedDataLoader.

        Args:
            data_dir: Path to data directory containing parquet files
            interval: Data interval (e.g., "5m", "1h", "1d")
            use_polars: Use polars instead of pandas. Defaults to USE_POLARS env var.
        """
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.use_polars = use_polars if use_polars is not None else USE_POLARS

        # Cache for schema data (symbol -> column list)
        self._schema_cache: Dict[str, List[str]] = {}

    def _resolve_parquet_path(self, symbol: str) -> Optional[Path]:
        """Resolve parquet file path for a symbol.

        Supports multiple directory structures:
        1. New: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.parquet
        2. Legacy: {data_dir}/{symbol}_mtf.parquet
        3. Direct: {data_dir}/{symbol}.parquet

        Args:
            symbol: Symbol name

        Returns:
            Path to parquet file or None if not found
        """
        # Try new structure first
        new_path = (
            self.data_dir
            / f"interval={self.interval}"
            / f"symbol={symbol}"
            / f"{symbol}.parquet"
        )
        if new_path.exists():
            return new_path

        # Try direct path (without interval subdirectory)
        direct_path = self.data_dir / f"symbol={symbol}" / f"{symbol}.parquet"
        if direct_path.exists():
            return direct_path

        # Try legacy format
        legacy_path = self.data_dir / f"{symbol}_mtf.parquet"
        if legacy_path.exists():
            return legacy_path

        # Try simple format
        simple_path = self.data_dir / f"{symbol}.parquet"
        if simple_path.exists():
            return simple_path

        return None

    def _resolve_time_col(self, columns: List[str]) -> Optional[str]:
        """Resolve time column name from column list.

        Args:
            columns: List of column names

        Returns:
            Time column name or None if not found
        """
        for c in ("ts", "timestamp", "time", "datetime", "ts_pred"):
            if c in columns:
                return c
        return None

    def read_schema(
        self, symbols: List[str], use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """Read column names from parquet files without loading data.

        This is very fast (~1ms per file) as it only reads parquet metadata.
        Use this for pre-flight checks before loading data.

        Args:
            symbols: List of symbols to read schema from
            use_cache: Whether to use cached schema data

        Returns:
            Dictionary mapping symbol -> sorted list of column names

        Example:
            ```python
            schemas = loader.read_schema(["AAPL", "GOOGL"])
            print(f"AAPL has {len(schemas['AAPL'])} columns")
            ```
        """
        schemas: Dict[str, List[str]] = {}

        # DETERMINISTIC: Sort symbols for consistent iteration order
        for symbol in sorted(symbols):
            # Check cache first
            if use_cache and symbol in self._schema_cache:
                schemas[symbol] = self._schema_cache[symbol]
                continue

            # Resolve file path
            parquet_path = self._resolve_parquet_path(symbol)
            if parquet_path is None:
                logger.warning(f"File not found for {symbol}")
                continue

            try:
                # Read schema only (no data loading) - uses pyarrow metadata
                schema = pq.read_schema(parquet_path)
                column_names = sorted([field.name for field in schema])

                # Cache for future use
                self._schema_cache[symbol] = column_names
                schemas[symbol] = column_names

            except Exception as e:
                logger.error(f"Error reading schema for {symbol}: {e}")

        return schemas

    def get_common_columns(self, symbols: List[str]) -> List[str]:
        """Get columns present in ALL symbols (intersection).

        Useful for cross-sectional analysis where features must be common
        across all symbols.

        Args:
            symbols: List of symbols to check

        Returns:
            Sorted list of column names present in all symbols

        Example:
            ```python
            common = loader.get_common_columns(["AAPL", "GOOGL", "MSFT"])
            print(f"{len(common)} columns common to all symbols")
            ```
        """
        schemas = self.read_schema(symbols)

        if not schemas:
            return []

        # Start with first symbol's columns
        # DETERMINISTIC: Sort symbols first to ensure consistent first symbol
        sorted_symbols = sorted(schemas.keys())
        common = set(schemas[sorted_symbols[0]])

        # Intersect with remaining symbols
        for symbol in sorted_symbols[1:]:
            common &= set(schemas[symbol])

        # Return sorted for determinism
        return sorted(common)

    def load_data(
        self,
        symbols: List[str],
        columns: Optional[List[str]] = None,
        max_rows_per_symbol: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data with optional column projection.

        When columns is specified, only those columns are loaded from the
        parquet files, significantly reducing memory usage.

        Args:
            symbols: List of symbols to load
            columns: Optional list of columns to load. If None, loads all columns.
                    Metadata columns (ts, timestamp, symbol) are always included.
            max_rows_per_symbol: Optional row limit per symbol (most recent rows)

        Returns:
            Dictionary mapping symbol -> DataFrame

        Example:
            ```python
            # Load all columns (backward compatible)
            data = loader.load_data(["AAPL", "GOOGL"])

            # Load only specific columns (5-10x memory reduction)
            data = loader.load_data(
                ["AAPL", "GOOGL"],
                columns=["close", "volume", "rsi_14"],
                max_rows_per_symbol=100000
            )
            ```
        """
        mtf_data: Dict[str, pd.DataFrame] = {}

        # DETERMINISTIC: Sort symbols for consistent iteration order
        for symbol in sorted(symbols):
            parquet_path = self._resolve_parquet_path(symbol)
            if parquet_path is None:
                logger.warning(f"File not found for {symbol}")
                continue

            try:
                if columns is not None:
                    df = self._load_with_projection(
                        parquet_path, columns, max_rows_per_symbol
                    )
                else:
                    df = self._load_all_columns(parquet_path, max_rows_per_symbol)

                if df is not None and len(df) > 0:
                    mtf_data[symbol] = df
                    logger.debug(
                        f"Loaded {symbol}: {len(df):,} rows, {len(df.columns)} cols"
                    )
                else:
                    logger.warning(f"Empty data for {symbol}")

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        logger.info(
            f"Loaded {len(mtf_data)}/{len(symbols)} symbols, "
            f"columns={len(columns) if columns else 'all'}"
        )

        return mtf_data

    def load_for_target(
        self,
        symbols: List[str],
        target: str,
        features: List[str],
        max_rows_per_symbol: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for a specific target with its features.

        Convenience method that automatically includes:
        - Target column
        - Feature columns
        - Metadata columns (ts, timestamp, symbol)

        Args:
            symbols: List of symbols to load
            target: Target column name (e.g., "fwd_ret_60m")
            features: List of feature column names
            max_rows_per_symbol: Optional row limit per symbol

        Returns:
            Dictionary mapping symbol -> DataFrame

        Example:
            ```python
            mtf_data = loader.load_for_target(
                symbols=["AAPL", "GOOGL"],
                target="fwd_ret_60m",
                features=["close", "volume", "rsi_14", "macd_signal"],
                max_rows_per_symbol=100000
            )
            ```
        """
        # Build column list: target + features + metadata
        # Use set to avoid duplicates, then sort for determinism
        columns_to_load = set([target])
        columns_to_load.update(features)
        columns_to_load.update(METADATA_COLUMNS)

        # Sort for determinism
        columns = sorted(columns_to_load)

        logger.info(
            f"Loading for target='{target}': {len(features)} features + metadata "
            f"= {len(columns)} columns"
        )

        return self.load_data(
            symbols=symbols,
            columns=columns,
            max_rows_per_symbol=max_rows_per_symbol,
        )

    def _load_with_projection(
        self,
        parquet_path: Path,
        columns: List[str],
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load parquet with column projection.

        Args:
            parquet_path: Path to parquet file
            columns: List of columns to load
            max_rows: Optional row limit

        Returns:
            DataFrame with selected columns or None on error
        """
        if self.use_polars:
            return self._load_projection_polars(parquet_path, columns, max_rows)
        else:
            return self._load_projection_pandas(parquet_path, columns, max_rows)

    def _load_projection_polars(
        self,
        parquet_path: Path,
        columns: List[str],
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load with column projection using Polars (faster).

        Args:
            parquet_path: Path to parquet file
            columns: Columns to select
            max_rows: Optional row limit

        Returns:
            DataFrame (converted to pandas) or None
        """
        import polars as pl

        try:
            # Lazy scan - won't materialize until collect()
            lf = pl.scan_parquet(str(parquet_path))

            # Get available columns
            available_cols = set(lf.collect_schema().names())

            # Filter to columns that exist
            # DETERMINISTIC: Sort for consistent column order
            select_cols = sorted([c for c in columns if c in available_cols])

            if not select_cols:
                logger.warning(f"No requested columns found in {parquet_path}")
                return None

            # Log any missing columns
            missing = set(columns) - available_cols
            if missing:
                logger.debug(f"Columns not in file: {sorted(missing)}")

            # Select columns (column projection)
            lf = lf.select(select_cols)

            # Detect and convert time column
            # CRITICAL: Use time_unit='ns' because parquet files store timestamps as int64 nanoseconds
            # Without this, Polars defaults to microseconds and interprets ns values as us, causing
            # timestamps to be 1000x too large (e.g., year 47979 instead of 2016)
            time_col = self._resolve_time_col(select_cols)
            if time_col:
                lf = lf.with_columns(
                    pl.col(time_col).cast(pl.Datetime("ns"), strict=False).alias(time_col)
                ).drop_nulls([time_col])

            # Apply row limit (keep most recent rows)
            if max_rows:
                lf = lf.tail(max_rows)

            # Collect and convert to pandas
            df_pl = lf.collect(streaming=True)
            return df_pl.to_pandas(use_pyarrow_extension_array=False)

        except Exception as e:
            logger.error(f"Polars load failed for {parquet_path}: {e}")
            return None

    def _load_projection_pandas(
        self,
        parquet_path: Path,
        columns: List[str],
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load with column projection using pandas.

        Args:
            parquet_path: Path to parquet file
            columns: Columns to select
            max_rows: Optional row limit

        Returns:
            DataFrame or None
        """
        try:
            # Read schema first to check available columns
            schema = pq.read_schema(parquet_path)
            available_cols = set(field.name for field in schema)

            # Filter to columns that exist
            # DETERMINISTIC: Sort for consistent column order
            select_cols = sorted([c for c in columns if c in available_cols])

            if not select_cols:
                logger.warning(f"No requested columns found in {parquet_path}")
                return None

            # Load with column selection
            df = pd.read_parquet(parquet_path, columns=select_cols)

            # Apply row limit (keep most recent)
            if max_rows and len(df) > max_rows:
                df = df.tail(max_rows)

            return df

        except Exception as e:
            logger.error(f"Pandas load failed for {parquet_path}: {e}")
            return None

    def _load_all_columns(
        self,
        parquet_path: Path,
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load all columns from parquet (backward compatible).

        Args:
            parquet_path: Path to parquet file
            max_rows: Optional row limit

        Returns:
            DataFrame or None
        """
        if self.use_polars:
            return self._load_all_polars(parquet_path, max_rows)
        else:
            return self._load_all_pandas(parquet_path, max_rows)

    def _load_all_polars(
        self,
        parquet_path: Path,
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load all columns using Polars.

        Args:
            parquet_path: Path to parquet file
            max_rows: Optional row limit

        Returns:
            DataFrame (converted to pandas) or None
        """
        import polars as pl

        try:
            lf = pl.scan_parquet(str(parquet_path))

            # Detect and convert time column
            # CRITICAL: Use time_unit='ns' because parquet files store timestamps as int64 nanoseconds
            # Without this, Polars defaults to microseconds and interprets ns values as us, causing
            # timestamps to be 1000x too large (e.g., year 47979 instead of 2016)
            schema_cols = lf.collect_schema().names()
            time_col = self._resolve_time_col(schema_cols)
            if time_col:
                lf = lf.with_columns(
                    pl.col(time_col).cast(pl.Datetime("ns"), strict=False).alias(time_col)
                ).drop_nulls([time_col])

            # Apply row limit
            if max_rows:
                lf = lf.tail(max_rows)

            df_pl = lf.collect(streaming=True)
            return df_pl.to_pandas(use_pyarrow_extension_array=False)

        except Exception as e:
            logger.error(f"Polars load failed for {parquet_path}: {e}")
            return None

    def _load_all_pandas(
        self,
        parquet_path: Path,
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load all columns using pandas.

        Args:
            parquet_path: Path to parquet file
            max_rows: Optional row limit

        Returns:
            DataFrame or None
        """
        try:
            df = pd.read_parquet(parquet_path)

            if max_rows and len(df) > max_rows:
                df = df.tail(max_rows)

            return df

        except Exception as e:
            logger.error(f"Pandas load failed for {parquet_path}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the schema cache."""
        self._schema_cache.clear()

    def discover_symbols(self) -> List[str]:
        """Discover available symbols in data directory.

        Returns:
            Sorted list of symbol names
        """
        symbols: List[str] = []

        # Try new structure first
        interval_dir = self.data_dir / f"interval={self.interval}"
        if interval_dir.exists():
            for symbol_dir in iterdir_sorted(interval_dir):
                if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                    symbol = symbol_dir.name.replace("symbol=", "")
                    symbols.append(symbol)

        # Try direct symbol= directories
        for symbol_dir in iterdir_sorted(self.data_dir):
            if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                symbol = symbol_dir.name.replace("symbol=", "")
                if symbol not in symbols:
                    symbols.append(symbol)

        # Try legacy structure
        for parquet_file in glob_sorted(self.data_dir, "*_mtf.parquet"):
            symbol = parquet_file.stem.replace("_mtf", "")
            if symbol not in symbols:
                symbols.append(symbol)

        return sorted(set(symbols))


def get_memory_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes (RSS - Resident Set Size)
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.warning("psutil not available, memory tracking disabled")
        return 0.0


def release_data(
    mtf_data: Dict[str, pd.DataFrame],
    verify: bool = False,
    log_memory: bool = False,
) -> Optional[float]:
    """Release data and force garbage collection.

    Call this between targets to free memory before loading next target.

    Args:
        mtf_data: Dictionary of symbol -> DataFrame to release
        verify: If True, verify memory was actually released using psutil
        log_memory: If True, log memory usage before/after

    Returns:
        Memory freed in MB if verify=True, None otherwise

    Example:
        ```python
        for target in targets:
            mtf_data = loader.load_for_target(symbols, target, features)
            # ... train model ...
            freed = release_data(mtf_data, verify=True, log_memory=True)
            print(f"Freed {freed:.1f} MB")
        ```
    """
    mem_before = get_memory_mb() if (verify or log_memory) else 0.0

    # Count items before release
    n_symbols = len(mtf_data)
    n_rows = sum(len(df) for df in mtf_data.values() if df is not None) if mtf_data else 0

    # Release all DataFrames
    for key in list(mtf_data.keys()):
        del mtf_data[key]
    mtf_data.clear()

    # Force garbage collection (multiple passes for thorough cleanup)
    gc.collect(0)  # Generation 0
    gc.collect(1)  # Generation 1
    gc.collect(2)  # Generation 2 (full collection)

    mem_after = get_memory_mb() if (verify or log_memory) else 0.0
    freed_mb = mem_before - mem_after

    if log_memory:
        logger.info(
            f"🧹 Released {n_symbols} symbols ({n_rows:,} rows): "
            f"{mem_before:.1f}MB → {mem_after:.1f}MB (freed {freed_mb:.1f}MB)"
        )

    return freed_mb if verify else None


class MemoryTracker:
    """Track memory usage during data loading operations.

    Use this to verify memory is being properly managed and identify leaks.

    Example:
        ```python
        tracker = MemoryTracker()
        tracker.checkpoint("start")

        mtf_data = loader.load_for_target(symbols, target, features)
        tracker.checkpoint("after_load")

        # ... train model ...

        release_data(mtf_data)
        tracker.checkpoint("after_release")

        tracker.report()
        # Output:
        # Memory Report:
        #   start:         1024.5 MB
        #   after_load:    2048.3 MB (+1023.8 MB)
        #   after_release: 1030.2 MB (-1018.1 MB)
        ```
    """

    def __init__(self):
        """Initialize memory tracker."""
        self._checkpoints: List[tuple] = []
        self._start_mem = get_memory_mb()

    def checkpoint(self, name: str) -> float:
        """Record a memory checkpoint.

        Args:
            name: Name for this checkpoint

        Returns:
            Current memory usage in MB
        """
        mem = get_memory_mb()
        self._checkpoints.append((name, mem))
        return mem

    def report(self) -> Dict[str, Any]:
        """Generate memory report.

        Returns:
            Dictionary with checkpoint data and analysis
        """
        if not self._checkpoints:
            return {"error": "No checkpoints recorded"}

        report = {
            "checkpoints": [],
            "total_change_mb": 0.0,
            "peak_mb": 0.0,
        }

        prev_mem = self._start_mem
        peak_mem = self._start_mem

        for name, mem in self._checkpoints:
            delta = mem - prev_mem
            report["checkpoints"].append({
                "name": name,
                "memory_mb": mem,
                "delta_mb": delta,
            })
            prev_mem = mem
            peak_mem = max(peak_mem, mem)

        report["total_change_mb"] = self._checkpoints[-1][1] - self._start_mem
        report["peak_mb"] = peak_mem

        # Log report
        logger.info("📊 Memory Report:")
        for cp in report["checkpoints"]:
            delta_str = f"+{cp['delta_mb']:.1f}" if cp['delta_mb'] >= 0 else f"{cp['delta_mb']:.1f}"
            logger.info(f"  {cp['name']:20s}: {cp['memory_mb']:.1f} MB ({delta_str} MB)")

        logger.info(f"  Peak: {report['peak_mb']:.1f} MB")
        logger.info(f"  Net change: {report['total_change_mb']:.1f} MB")

        return report

    def verify_no_leak(self, tolerance_mb: float = 50.0) -> bool:
        """Verify memory returned to near-starting level.

        Args:
            tolerance_mb: Allowable memory increase in MB

        Returns:
            True if no leak detected (within tolerance)

        Raises:
            MemoryLeakError: If memory leak detected (above tolerance)
        """
        if not self._checkpoints:
            return True

        final_mem = self._checkpoints[-1][1]
        increase = final_mem - self._start_mem

        if increase > tolerance_mb:
            raise MemoryLeakError(
                f"Memory leak detected: started at {self._start_mem:.1f}MB, "
                f"ended at {final_mem:.1f}MB (increase of {increase:.1f}MB, "
                f"tolerance: {tolerance_mb:.1f}MB)"
            )

        return True


class MemoryLeakError(Exception):
    """Raised when a memory leak is detected."""
    pass


def streaming_concat(
    mtf_data: Dict[str, pd.DataFrame],
    symbol_column: str = "symbol",
    target_column: Optional[str] = None,
    use_float32: Optional[bool] = None,
    release_after_convert: bool = True,
) -> "pl.LazyFrame":
    """Convert mtf_data dict to a memory-efficient Polars LazyFrame.

    This is the DRY helper for all stages that need to combine symbol data.
    It converts each DataFrame to a lazy frame immediately, releasing memory
    as it goes, then returns a lazy concat that can be collected with streaming.

    This function is designed to work seamlessly with the existing data loading
    infrastructure:
    - Input: Output from UnifiedDataLoader.load_data() or load_for_target()
    - Output: Polars LazyFrame for streaming collection
    - Memory: Releases each DataFrame after conversion to minimize peak usage

    Args:
        mtf_data: Dictionary mapping symbol -> DataFrame
            (typically from UnifiedDataLoader.load_data() or load_for_target())
        symbol_column: Name for the symbol column to add (default: "symbol")
        target_column: If provided, skip symbols missing this column
        use_float32: Cast float64 columns to float32 for 50% memory reduction.
            If None, reads from config: intelligent_training.lazy_loading.use_float32 (default: True)
        release_after_convert: Release each DataFrame after converting to lazy frame

    Returns:
        Polars LazyFrame - call .collect(streaming=True) to materialize efficiently

    Example:
        ```python
        from TRAINING.data.loading import UnifiedDataLoader, streaming_concat

        # Load data using existing methods
        loader = UnifiedDataLoader(data_dir="/data/prices", interval="5m")
        mtf_data = loader.load_for_target(symbols, target, features)

        # Convert to streaming lazy frame (memory efficient)
        lf = streaming_concat(mtf_data, target_column=target)

        # Apply filters lazily (no memory allocation yet)
        lf = lf.filter(pl.len().over("ts") >= min_cs)

        # Collect with streaming mode (processes in chunks)
        combined_df = lf.collect(streaming=True).to_pandas()
        ```

    Memory Comparison (728 symbols × 75k rows × 100 cols):
        - pd.concat(all_data): ~132 GB peak (OOM on 128GB)
        - streaming_concat + collect(streaming=True): ~46 GB peak ✓

    SST Compliance:
        - Deterministic: processes symbols in sorted order
        - Reuses existing UnifiedDataLoader output format
        - Works with existing release_data() for additional cleanup
    """
    import polars as pl

    # Load config defaults if not explicitly specified
    if use_float32 is None:
        try:
            from CONFIG.config_loader import get_cfg
            use_float32 = get_cfg(
                "intelligent_training.lazy_loading.use_float32",
                default=True
            )
        except ImportError:
            use_float32 = True  # Default if config not available

    lfs: List[pl.LazyFrame] = []

    # DETERMINISTIC: Sort symbols for consistent processing order
    symbols_sorted = sorted(mtf_data.keys())

    for symbol in symbols_sorted:
        df = mtf_data[symbol]

        # Skip None entries (already released)
        if df is None:
            continue

        # Skip if target column required but missing
        if target_column and target_column not in df.columns:
            logger.debug(f"Skipping {symbol}: target '{target_column}' not found")
            if release_after_convert:
                mtf_data[symbol] = None
            continue

        # Convert to Polars lazy frame
        lf = pl.from_pandas(df).lazy()

        # Add symbol column
        lf = lf.with_columns(pl.lit(symbol).alias(symbol_column))

        # Cast to float32 for memory efficiency (50% reduction)
        if use_float32:
            # Get float64 columns (exclude metadata)
            float_cols = [
                c for c in df.select_dtypes(include=["float64"]).columns
                if c not in METADATA_COLUMNS
            ]
            if float_cols:
                lf = lf.with_columns([
                    pl.col(c).cast(pl.Float32, strict=False) for c in float_cols
                ])

        lfs.append(lf)

        # Release DataFrame immediately to free memory
        if release_after_convert:
            mtf_data[symbol] = None
            del df

    # Force garbage collection if we released data
    if release_after_convert:
        gc.collect()

    if not lfs:
        logger.warning("streaming_concat: No data to concat")
        return pl.LazyFrame()

    logger.info(
        f"📊 streaming_concat: {len(lfs)} symbols → lazy concat "
        f"(float32={use_float32}, released={release_after_convert})"
    )

    # Return lazy concat - no memory allocation until collect()
    return pl.concat(lfs, how="vertical_relaxed")


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================

def load_mtf_data(
    data_dir: str,
    symbols: List[str],
    interval: str = "5m",
    max_rows_per_symbol: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load MTF data for specified symbols (convenience function).

    This is the canonical SST function for loading MTF data. Use this instead of
    deprecated `_load_mtf_data_pandas` functions scattered across the codebase.

    Args:
        data_dir: Path to data directory
        symbols: List of symbols to load
        interval: Data interval (e.g., "5m", "1h")
        max_rows_per_symbol: Optional row limit per symbol (most recent rows)
        columns: Optional list of columns to load (memory optimization)

    Returns:
        Dictionary mapping symbol -> DataFrame

    Example:
        ```python
        from TRAINING.data.loading.unified_loader import load_mtf_data

        # Basic usage (loads all columns)
        mtf_data = load_mtf_data("/data/prices", ["AAPL", "GOOGL"], interval="5m")

        # With column projection (5-10x memory reduction)
        mtf_data = load_mtf_data(
            "/data/prices",
            ["AAPL", "GOOGL"],
            interval="5m",
            columns=["close", "volume", "rsi_14"],
            max_rows_per_symbol=100000
        )
        ```
    """
    loader = UnifiedDataLoader(data_dir, interval=interval)
    return loader.load_data(
        symbols=symbols,
        columns=columns,
        max_rows_per_symbol=max_rows_per_symbol,
    )
