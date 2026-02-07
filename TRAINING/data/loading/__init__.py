# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Data loading utilities with pluggable loader support.

This module provides a pluggable data loading system that supports
multiple data formats (parquet, CSV, custom) while maintaining
backward compatibility with existing code.

Quick Start:
    ```python
    from TRAINING.data.loading import get_loader, load_mtf_data

    # Option 1: Use pluggable loader system (recommended for new code)
    loader = get_loader("parquet")  # or "csv"
    result = loader.load("/data/prices", ["AAPL", "GOOGL"], interval="5m")

    # Option 2: Use backward-compatible function
    data = load_mtf_data("/data/prices", ["AAPL", "GOOGL"], interval="5m")
    ```

Custom Loaders:
    ```python
    from TRAINING.data.loading import DataLoader, register_loader, LoadResult

    class MyLoader(DataLoader):
        def load(self, source, symbols, interval="5m", **kwargs):
            # Custom loading logic
            return LoadResult(data={}, symbols_loaded=[], symbols_failed=[], metadata={})

        def validate_schema(self, df, schema):
            return True

        def discover_symbols(self, source, interval=None):
            return []

    register_loader("my_loader", MyLoader)
    ```
"""

# Import interface types first
from .interface import (
    DataLoader,
    LoadResult,
    SchemaRequirement,
    SchemaValidationError,
)

# Import schema utilities
from .schema import (
    DEFAULT_SCHEMA,
    check_schema,
    infer_schema,
    validate_dataframe,
)

# Import registry functions
from .registry import (
    clear_cache,
    get_loader,
    list_loaders,
    register_loader,
    unregister_loader,
)

# Import concrete loaders (this registers them)
from .parquet_loader import ParquetLoader
from .csv_loader import CSVLoader

# Import unified loader (new single source of truth)
from .unified_loader import (
    UnifiedDataLoader,
    release_data,
    METADATA_COLUMNS,
    MemoryTracker,
    MemoryLeakError,
    get_memory_mb,
    streaming_concat,
)

# Register built-in loaders
register_loader("parquet", ParquetLoader)
register_loader("csv", CSVLoader)

# Import backward-compatible functions
from .data_loader import (
    _load_mtf_data_pandas,
    _pick_one,
    _prepare_training_data_cross_sectional_pandas,
    cs_transform_live,
    load_mtf_data,
    prepare_training_data_cross_sectional,
    resolve_time_col,
    targets_for_interval,
)
from .data_utils import (
    collapse_identical_duplicate_columns,
    prepare_sequence_cs,
    strip_targets,
)

# For backward compatibility, create module-level aliases
data_loader = _load_mtf_data_pandas
data_utils = type(
    "DataUtils",
    (),
    {
        "strip_targets": strip_targets,
        "collapse_identical_duplicate_columns": collapse_identical_duplicate_columns,
    },
)()

__all__ = [
    # Interface types
    "DataLoader",
    "LoadResult",
    "SchemaRequirement",
    "SchemaValidationError",
    # Schema utilities
    "DEFAULT_SCHEMA",
    "validate_dataframe",
    "check_schema",
    "infer_schema",
    # Registry functions
    "register_loader",
    "get_loader",
    "list_loaders",
    "clear_cache",
    "unregister_loader",
    # Concrete loaders
    "ParquetLoader",
    "CSVLoader",
    # Unified loader (new single source of truth)
    "UnifiedDataLoader",
    "release_data",
    "METADATA_COLUMNS",
    "MemoryTracker",
    "MemoryLeakError",
    "get_memory_mb",
    "streaming_concat",
    # Backward-compatible functions
    "resolve_time_col",
    "_pick_one",
    "_load_mtf_data_pandas",
    "_prepare_training_data_cross_sectional_pandas",
    "load_mtf_data",
    "prepare_training_data_cross_sectional",
    "targets_for_interval",
    "cs_transform_live",
    # Data utilities
    "strip_targets",
    "collapse_identical_duplicate_columns",
    "prepare_sequence_cs",
    # Aliases
    "data_loader",
    "data_utils",
]
