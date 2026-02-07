# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Parquet file data loader.

Implements the DataLoader interface for loading data from parquet files.
Supports both Polars (default) and pandas backends for flexibility.

Directory structures supported:
    1. New: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.parquet
    2. Legacy: {data_dir}/{symbol}_mtf.parquet

Example:
    ```python
    from TRAINING.data.loading import get_loader

    loader = get_loader("parquet")
    result = loader.load("/data/prices", ["AAPL", "GOOGL"], interval="5m")
    for symbol, df in result.data.items():
        print(f"{symbol}: {len(df)} rows")
    ```
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from TRAINING.common.utils.determinism_ordering import glob_sorted, iterdir_sorted

from .interface import DataLoader, LoadResult, SchemaRequirement
from .schema import DEFAULT_SCHEMA, validate_dataframe

logger = logging.getLogger(__name__)

# Use Polars by default (can be disabled via env var)
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"


class ParquetLoader(DataLoader):
    """Load data from parquet files.

    Supports two directory structures:
        1. New: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.parquet
        2. Legacy: {data_dir}/{symbol}_mtf.parquet

    Uses Polars for efficient loading when available, with pandas fallback.

    Attributes:
        compression: Compression codec for parquet files
        use_polars: Whether to use Polars (faster) or pandas
        validate_schema: Whether to validate schema on load
    """

    def __init__(
        self,
        compression: str = "snappy",
        use_polars: Optional[bool] = None,
        validate_schema: bool = True,
        **kwargs,
    ):
        """Initialize ParquetLoader.

        Args:
            compression: Compression codec (snappy, gzip, zstd)
            use_polars: Use polars instead of pandas. Defaults to USE_POLARS env var.
            validate_schema: Whether to validate schema on load
            **kwargs: Additional options (ignored)
        """
        self.compression = compression
        self.use_polars = use_polars if use_polars is not None else USE_POLARS
        self._validate_schema = validate_schema

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        max_rows_per_symbol: Optional[int] = None,
        **kwargs,
    ) -> LoadResult:
        """Load parquet data for symbols.

        Args:
            source: Path to data directory
            symbols: List of symbols to load
            interval: Data interval (e.g., "5m", "1h")
            schema: Optional schema requirements for validation
            max_rows_per_symbol: Optional row limit per symbol
            **kwargs: Additional options

        Returns:
            LoadResult with loaded dataframes
        """
        data: Dict[str, pd.DataFrame] = {}
        symbols_loaded: List[str] = []
        symbols_failed: List[str] = []
        metadata: Dict[str, Any] = {
            "interval": interval,
            "source": source,
            "loader": "parquet",
            "use_polars": self.use_polars,
        }

        source_path = Path(source)
        schema = schema or DEFAULT_SCHEMA

        for symbol in symbols:
            try:
                df = self._load_symbol(
                    source_path, symbol, interval, max_rows_per_symbol
                )
                if df is None:
                    symbols_failed.append(symbol)
                    continue

                # Validate schema
                if self._validate_schema:
                    df = validate_dataframe(df, schema, strict=False, auto_fix=True)

                data[symbol] = df
                symbols_loaded.append(symbol)
                logger.info(f"Loaded {symbol}: {len(df):,} rows, {len(df.columns)} cols")

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                symbols_failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=sorted(symbols_loaded),
            symbols_failed=sorted(symbols_failed),
            metadata=metadata,
        )

    def _load_symbol(
        self,
        source_path: Path,
        symbol: str,
        interval: str,
        max_rows: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Load parquet data for a single symbol.

        Args:
            source_path: Path to data directory
            symbol: Symbol name
            interval: Data interval
            max_rows: Optional row limit

        Returns:
            DataFrame or None if not found
        """
        # Try new structure first
        new_path = source_path / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"
        legacy_path = source_path / f"{symbol}_mtf.parquet"

        file_path = new_path if new_path.exists() else legacy_path

        if not file_path.exists():
            logger.warning(f"File not found for {symbol} at {new_path} or {legacy_path}")
            return None

        if self.use_polars:
            return self._load_with_polars(file_path, max_rows)
        else:
            return self._load_with_pandas(file_path, max_rows)

    def _load_with_polars(
        self, file_path: Path, max_rows: Optional[int]
    ) -> pd.DataFrame:
        """Load parquet with Polars backend.

        Args:
            file_path: Path to parquet file
            max_rows: Optional row limit

        Returns:
            DataFrame (converted to pandas for compatibility)
        """
        import polars as pl

        # Lazy scan for efficiency
        lf = pl.scan_parquet(str(file_path))

        # Detect and standardize time column
        schema_cols = lf.collect_schema().names()
        tcol = self._resolve_time_col(schema_cols)

        # Convert time column to datetime (handles both string and datetime)
        # CRITICAL: Use time_unit='ns' because parquet files store timestamps as int64 nanoseconds
        # Without this, Polars defaults to microseconds and interprets ns values as us, causing
        # timestamps to be 1000x too large (e.g., year 47979 instead of 2016)
        lf = lf.with_columns(
            pl.col(tcol).cast(pl.Datetime("ns"), strict=False).alias(tcol)
        ).drop_nulls([tcol])

        # Apply row limit (keep most recent)
        if max_rows:
            lf = lf.tail(max_rows)

        # Collect and convert to pandas
        df = lf.collect(streaming=True)
        return df.to_pandas(use_pyarrow_extension_array=False)

    def _load_with_pandas(
        self, file_path: Path, max_rows: Optional[int]
    ) -> pd.DataFrame:
        """Load parquet with pandas backend.

        Args:
            file_path: Path to parquet file
            max_rows: Optional row limit

        Returns:
            DataFrame
        """
        df = pd.read_parquet(file_path)

        if max_rows:
            df = df.tail(max_rows)

        return df

    def _resolve_time_col(self, cols: List[str]) -> str:
        """Resolve time column name from column list.

        Args:
            cols: List of column names

        Returns:
            Time column name

        Raises:
            KeyError: If no time column found
        """
        for c in ("ts", "timestamp", "time", "datetime", "ts_pred"):
            if c in cols:
                return c
        raise KeyError(f"No time column found in {cols[:10]}")

    def validate_schema(
        self, df: pd.DataFrame, schema: SchemaRequirement
    ) -> bool:
        """Validate dataframe schema.

        Args:
            df: DataFrame to validate
            schema: Schema requirements

        Returns:
            True if valid
        """
        try:
            validate_dataframe(df, schema, strict=True, auto_fix=False)
            return True
        except Exception:
            return False

    def discover_symbols(
        self, source: str, interval: Optional[str] = None
    ) -> List[str]:
        """Discover symbols in source directory.

        Args:
            source: Path to data directory
            interval: Optional interval filter

        Returns:
            Sorted list of symbol names
        """
        source_path = Path(source)
        symbols: List[str] = []

        # Try new structure first
        if interval:
            interval_dir = source_path / f"interval={interval}"
            if interval_dir.exists():
                for symbol_dir in iterdir_sorted(interval_dir):
                    if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                        symbol = symbol_dir.name.replace("symbol=", "")
                        symbols.append(symbol)

        # Try legacy structure
        if not symbols:
            # Look for {symbol}_mtf.parquet files
            for parquet_file in glob_sorted(source_path, "*_mtf.parquet"):
                symbol = parquet_file.stem.replace("_mtf", "")
                symbols.append(symbol)

            # Also look for symbol= directories
            for symbol_dir in iterdir_sorted(source_path):
                if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                    symbol = symbol_dir.name.replace("symbol=", "")
                    if symbol not in symbols:
                        symbols.append(symbol)

        return sorted(set(symbols))

    def get_supported_intervals(self, source: str) -> List[str]:
        """Get available intervals in source.

        Args:
            source: Path to data directory

        Returns:
            Sorted list of interval strings
        """
        source_path = Path(source)
        intervals = []

        for interval_dir in iterdir_sorted(source_path):
            if interval_dir.is_dir() and interval_dir.name.startswith("interval="):
                interval = interval_dir.name.replace("interval=", "")
                intervals.append(interval)

        return sorted(intervals)

    def get_metadata(self, source: str, symbol: str) -> Dict[str, Any]:
        """Get metadata for a symbol.

        Args:
            source: Path to data directory
            symbol: Symbol name

        Returns:
            Dictionary with file info
        """
        source_path = Path(source)

        # Try to find the file
        for interval in self.get_supported_intervals(source) or ["5m"]:
            new_path = source_path / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"
            legacy_path = source_path / f"{symbol}_mtf.parquet"

            file_path = new_path if new_path.exists() else legacy_path

            if file_path.exists():
                stat = file_path.stat()
                return {
                    "path": str(file_path),
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                }

        return {}


# Register with the registry (do not auto-register here, let __init__.py handle it)
# This avoids circular import issues
