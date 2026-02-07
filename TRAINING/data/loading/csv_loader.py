# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""CSV file data loader.

Implements the DataLoader interface for loading data from CSV files.
Supports flexible directory structures and automatic date parsing.

Directory structures supported:
    1. Single file: {data_dir}/{symbol}.csv
    2. Interval subdirs: {data_dir}/{interval}/{symbol}.csv
    3. Hive-style: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.csv
    4. Symbol subdirs: {data_dir}/{symbol}/*.csv (concatenates multiple files)

Example:
    ```python
    from TRAINING.data.loading import get_loader

    loader = get_loader("csv")
    result = loader.load("/data/csv", ["AAPL", "GOOGL"], interval="5m")
    ```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from TRAINING.common.utils.determinism_ordering import glob_sorted, iterdir_sorted

from .interface import DataLoader, LoadResult, SchemaRequirement
from .schema import DEFAULT_SCHEMA, validate_dataframe

logger = logging.getLogger(__name__)


class CSVLoader(DataLoader):
    """Load data from CSV files.

    Supports flexible directory structures and various CSV formats.
    Automatically parses date columns and handles common variations.

    Attributes:
        delimiter: CSV field delimiter
        date_column: Name of date/time column
        date_format: Optional strptime format string
        encoding: File encoding
    """

    def __init__(
        self,
        delimiter: str = ",",
        date_column: str = "timestamp",
        date_format: Optional[str] = None,
        encoding: str = "utf-8",
        validate_schema: bool = True,
        **kwargs,
    ):
        """Initialize CSVLoader.

        Args:
            delimiter: CSV delimiter character
            date_column: Name of date/time column
            date_format: Optional strptime format string
            encoding: File encoding
            validate_schema: Whether to validate schema
            **kwargs: Additional options (ignored)
        """
        self.delimiter = delimiter
        self.date_column = date_column
        self.date_format = date_format
        self.encoding = encoding
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
        """Load CSV data for symbols.

        Args:
            source: Path to data directory
            symbols: List of symbols to load
            interval: Data interval (e.g., "5m", "1h")
            schema: Optional schema requirements
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
            "loader": "csv",
            "delimiter": self.delimiter,
        }

        source_path = Path(source)
        schema = schema or DEFAULT_SCHEMA

        for symbol in symbols:
            try:
                df = self._load_symbol(source_path, symbol, interval)
                if df is None:
                    symbols_failed.append(symbol)
                    continue

                # Parse dates
                df = self._parse_dates(df)

                # Apply row limit
                if max_rows_per_symbol:
                    df = df.tail(max_rows_per_symbol)

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
    ) -> Optional[pd.DataFrame]:
        """Load CSV for a single symbol.

        Args:
            source_path: Path to data directory
            symbol: Symbol name
            interval: Data interval

        Returns:
            DataFrame or None if not found
        """
        # Try various path patterns (in order of preference)
        patterns = [
            # Hive-style
            source_path / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.csv",
            # Interval subdirectory
            source_path / interval / f"{symbol}.csv",
            # Symbol subdirectory
            source_path / f"symbol={symbol}" / f"{symbol}.csv",
            # Direct file
            source_path / f"{symbol}.csv",
            # With _mtf suffix (matching parquet convention)
            source_path / f"{symbol}_mtf.csv",
        ]

        for csv_path in patterns:
            if csv_path.exists():
                logger.debug(f"Found {symbol} at {csv_path}")
                return pd.read_csv(
                    csv_path,
                    delimiter=self.delimiter,
                    encoding=self.encoding,
                )

        # Try directory with multiple CSVs (concatenate)
        multi_file_dirs = [
            source_path / symbol,
            source_path / f"symbol={symbol}",
        ]

        for symbol_dir in multi_file_dirs:
            if symbol_dir.exists() and symbol_dir.is_dir():
                csv_files = glob_sorted(symbol_dir, "*.csv")
                if csv_files:
                    logger.debug(f"Concatenating {len(csv_files)} files for {symbol}")
                    dfs = [
                        pd.read_csv(f, delimiter=self.delimiter, encoding=self.encoding)
                        for f in csv_files
                    ]
                    return pd.concat(dfs, ignore_index=True)

        logger.warning(f"No CSV file found for {symbol}")
        return None

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column in dataframe.

        Args:
            df: DataFrame with date column

        Returns:
            DataFrame with parsed dates
        """
        # Find date column (try configured name first, then common names)
        date_col = None
        candidates = [self.date_column, "timestamp", "ts", "time", "datetime", "date"]

        for col in candidates:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            logger.debug("No date column found to parse")
            return df

        # Parse dates
        try:
            df[date_col] = pd.to_datetime(
                df[date_col],
                format=self.date_format,
                errors="coerce",
            )
        except Exception as e:
            logger.warning(f"Could not parse date column '{date_col}': {e}")

        return df

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
        """Discover symbols from CSV files.

        Args:
            source: Path to data directory
            interval: Optional interval filter

        Returns:
            Sorted list of symbol names
        """
        source_path = Path(source)
        symbols: List[str] = []

        # Pattern 1: {source}/*.csv
        for csv_file in glob_sorted(source_path, "*.csv"):
            symbol = csv_file.stem
            # Remove common suffixes
            if symbol.endswith("_mtf"):
                symbol = symbol[:-4]
            symbols.append(symbol)

        # Pattern 2: {source}/{interval}/*.csv
        if interval:
            interval_dir = source_path / interval
            if interval_dir.exists():
                for csv_file in glob_sorted(interval_dir, "*.csv"):
                    symbol = csv_file.stem
                    if symbol not in symbols:
                        symbols.append(symbol)

        # Pattern 3: Hive-style {source}/interval={interval}/symbol={symbol}/
        if interval:
            interval_dir = source_path / f"interval={interval}"
            if interval_dir.exists():
                for symbol_dir in iterdir_sorted(interval_dir):
                    if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                        symbol = symbol_dir.name.replace("symbol=", "")
                        if symbol not in symbols:
                            symbols.append(symbol)

        # Pattern 4: {source}/symbol={symbol}/
        for symbol_dir in iterdir_sorted(source_path):
            if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                symbol = symbol_dir.name.replace("symbol=", "")
                if symbol not in symbols:
                    symbols.append(symbol)

        # Pattern 5: {source}/{symbol}/ (directory per symbol)
        for subdir in iterdir_sorted(source_path):
            if subdir.is_dir() and not subdir.name.startswith(("interval=", "symbol=", ".")):
                # Check if it contains CSVs
                csv_files = list(subdir.glob("*.csv"))
                if csv_files:
                    if subdir.name not in symbols:
                        symbols.append(subdir.name)

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

        # Look for interval= directories (Hive-style)
        for interval_dir in iterdir_sorted(source_path):
            if interval_dir.is_dir() and interval_dir.name.startswith("interval="):
                interval = interval_dir.name.replace("interval=", "")
                intervals.append(interval)

        # Look for common interval subdirectories
        common_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for interval in common_intervals:
            interval_dir = source_path / interval
            if interval_dir.exists() and interval_dir.is_dir():
                if interval not in intervals:
                    intervals.append(interval)

        return sorted(intervals)


# Registration happens in __init__.py to avoid circular imports
