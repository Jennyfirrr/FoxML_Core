# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Abstract interface for data loaders.

This module defines the DataLoader abstract base class and related types
that all data loaders must implement. This enables pluggable data sources
while maintaining a consistent interface for the pipeline.

Example:
    ```python
    from TRAINING.data.loading.interface import DataLoader, LoadResult

    class MyCustomLoader(DataLoader):
        def load(self, source, symbols, interval="5m", **kwargs):
            # Custom loading logic
            return LoadResult(data={...}, symbols_loaded=[...], ...)

        def validate_schema(self, df, schema):
            return True

        def discover_symbols(self, source, interval=None):
            return ["AAPL", "GOOGL"]
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd


@dataclass
class LoadResult:
    """Result of a data load operation.

    Attributes:
        data: Dictionary mapping symbol to DataFrame
        symbols_loaded: List of successfully loaded symbols
        symbols_failed: List of symbols that failed to load
        metadata: Loader-specific metadata (interval, source path, etc.)
    """

    data: Dict[str, pd.DataFrame]
    symbols_loaded: List[str]
    symbols_failed: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaRequirement:
    """Schema requirements for loaded data.

    Attributes:
        required_columns: Set of columns that must be present
        time_column: Name of the timestamp column
        index_column: Optional column to use as index
        dtypes: Optional mapping of column name to expected dtype
    """

    required_columns: Set[str] = field(default_factory=lambda: {"timestamp"})
    time_column: str = "timestamp"
    index_column: Optional[str] = None
    dtypes: Optional[Dict[str, str]] = None


class DataLoader(ABC):
    """Abstract base class for data loaders.

    All data loaders must implement this interface to be usable
    with the FoxML pipeline. The interface supports:
    - Loading data for specified symbols
    - Schema validation
    - Symbol discovery

    Implementations:
        - ParquetLoader: Loads parquet files (default)
        - CSVLoader: Loads CSV files

    Example:
        ```python
        from TRAINING.data.loading import get_loader

        # Use default loader from config
        loader = get_loader()
        result = loader.load("/data/prices", ["AAPL", "GOOGL"], interval="5m")

        # Use specific loader
        csv_loader = get_loader("csv")
        result = csv_loader.load("/data/csv", ["AAPL"])
        ```
    """

    @abstractmethod
    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        **kwargs,
    ) -> LoadResult:
        """Load data for specified symbols.

        Args:
            source: Path to data directory or connection string
            symbols: List of symbols to load
            interval: Data interval (e.g., "5m", "1h", "1d")
            schema: Optional schema requirements for validation
            **kwargs: Loader-specific options

        Returns:
            LoadResult with loaded dataframes and metadata
        """
        pass

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame, schema: SchemaRequirement) -> bool:
        """Validate dataframe against schema requirements.

        Args:
            df: DataFrame to validate
            schema: Schema requirements

        Returns:
            True if valid, False otherwise

        Raises:
            SchemaValidationError: If validation fails in strict mode
        """
        pass

    @abstractmethod
    def discover_symbols(
        self, source: str, interval: Optional[str] = None
    ) -> List[str]:
        """Discover available symbols in data source.

        Args:
            source: Path to data directory or connection string
            interval: Optional interval filter

        Returns:
            Sorted list of available symbol names
        """
        pass

    def get_supported_intervals(self, source: str) -> List[str]:
        """Get list of intervals available in source.

        Default implementation returns empty list (unknown).
        Override for loaders that can enumerate intervals.

        Args:
            source: Path to data directory or connection string

        Returns:
            Sorted list of interval strings
        """
        return []

    def get_metadata(self, source: str, symbol: str) -> Dict[str, Any]:
        """Get metadata for a specific symbol.

        Default implementation returns empty dict.
        Override for loaders that provide metadata.

        Args:
            source: Path to data directory or connection string
            symbol: Symbol to get metadata for

        Returns:
            Dictionary of metadata
        """
        return {}


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""

    pass
