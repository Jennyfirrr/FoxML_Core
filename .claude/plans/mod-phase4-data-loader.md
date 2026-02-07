# Phase 4: Pluggable Data Loader Interface

**Status**: ✅ Complete
**Priority**: P1 (Extensibility)
**Effort**: 12 hours
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)
**Completed**: 2026-01-19

---

## Quick Resume

```
STATUS: COMPLETE
FILES CREATED: interface.py, registry.py, schema.py, parquet_loader.py, csv_loader.py
FILES MODIFIED: data_loader.py, __init__.py
VERIFICATION: All imports work, all contract tests pass (same 5 pre-existing failures)
```

---

## Problem Statement

Currently, the data loading system is **parquet-only** with hardcoded assumptions:
- Users with CSV/SQL data must pre-convert to parquet
- No pluggable interface for custom data sources
- No schema validation for incoming data
- Symbol discovery assumes specific directory structure

**Goal**: Create a pluggable data loader architecture that:
1. Supports multiple formats (parquet, CSV, SQL, custom)
2. Provides schema validation
3. Maintains backward compatibility
4. Is easily extensible via config

---

## Current Architecture

### File: `TRAINING/data/loading/data_loader.py`

```python
# Current hardcoded approach
def load_mtf_data(data_dir: str, symbols: List[str], interval: str = "5m", ...):
    # Assumes: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.parquet
    # No pluggable interface
    # No format selection
```

### File: `TRAINING/orchestration/utils/symbol_discovery.py`

```python
# Current discovery
def discover_symbols_from_data_dir(data_dir: str) -> List[str]:
    # Hardcoded path patterns
    # No format-agnostic discovery
```

---

## Target Architecture

```
TRAINING/data/loading/
├── __init__.py              # Public API
├── interface.py             # NEW: DataLoader ABC + LoadResult type
├── registry.py              # NEW: Loader registry (name -> loader)
├── schema.py                # NEW: Schema validation
├── parquet_loader.py        # NEW: ParquetLoader (current logic)
├── csv_loader.py            # NEW: CSVLoader
├── sql_loader.py            # NEW: SQLLoader (optional)
├── data_loader.py           # MODIFIED: Thin wrapper using registry
└── utils.py                 # Shared utilities
```

### Config Integration

```yaml
# CONFIG/data/data_loading.yaml
data_loading:
  default_loader: parquet       # Default loader name
  loaders:
    parquet:
      class: TRAINING.data.loading.parquet_loader.ParquetLoader
      options:
        compression: snappy
    csv:
      class: TRAINING.data.loading.csv_loader.CSVLoader
      options:
        delimiter: ","
        date_column: timestamp
    custom:
      class: my_project.loaders.MyCustomLoader
      options: {}
```

### Experiment Config Override

```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  loader: csv                    # Override default
  data_dir: /path/to/csv/data
  loader_options:
    delimiter: ";"
    encoding: utf-8
```

---

## Implementation Plan

### Task 1: Create DataLoader Interface

**File**: `TRAINING/data/loading/interface.py`

```python
"""Abstract interface for data loaders."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import pandas as pd


@dataclass
class LoadResult:
    """Result of a data load operation."""
    data: Dict[str, pd.DataFrame]  # symbol -> dataframe
    symbols_loaded: List[str]
    symbols_failed: List[str]
    metadata: Dict[str, Any]  # loader-specific metadata


@dataclass
class SchemaRequirement:
    """Schema requirements for loaded data."""
    required_columns: Set[str]
    time_column: str = "timestamp"
    index_column: Optional[str] = None
    dtypes: Optional[Dict[str, str]] = None


class DataLoader(ABC):
    """Abstract base class for data loaders.

    All data loaders must implement this interface to be usable
    with the FoxML pipeline.

    Example:
        ```python
        class MyLoader(DataLoader):
            def load(self, source, symbols, interval, **kwargs):
                # Load data from custom source
                return LoadResult(data=..., symbols_loaded=..., ...)

            def validate_schema(self, df, schema):
                # Validate dataframe against schema
                return True

            def discover_symbols(self, source, interval):
                # Discover available symbols
                return ["AAPL", "GOOGL", ...]
        ```
    """

    @abstractmethod
    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        **kwargs
    ) -> LoadResult:
        """Load data for specified symbols.

        Args:
            source: Path to data directory or connection string
            symbols: List of symbols to load
            interval: Data interval (e.g., "5m", "1h")
            schema: Optional schema requirements for validation
            **kwargs: Loader-specific options

        Returns:
            LoadResult with loaded dataframes and metadata
        """
        pass

    @abstractmethod
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: SchemaRequirement
    ) -> bool:
        """Validate dataframe against schema requirements.

        Args:
            df: DataFrame to validate
            schema: Schema requirements

        Returns:
            True if valid, False otherwise

        Raises:
            SchemaValidationError: If validation fails and strict mode
        """
        pass

    @abstractmethod
    def discover_symbols(
        self,
        source: str,
        interval: Optional[str] = None
    ) -> List[str]:
        """Discover available symbols in data source.

        Args:
            source: Path to data directory or connection string
            interval: Optional interval filter

        Returns:
            List of available symbol names
        """
        pass

    def get_supported_intervals(self, source: str) -> List[str]:
        """Get list of intervals available in source.

        Default implementation returns empty list (unknown).
        Override for loaders that can enumerate intervals.
        """
        return []

    def get_metadata(self, source: str, symbol: str) -> Dict[str, Any]:
        """Get metadata for a specific symbol.

        Default implementation returns empty dict.
        Override for loaders that provide metadata.
        """
        return {}


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    pass
```

### Task 2: Create Loader Registry

**File**: `TRAINING/data/loading/registry.py`

```python
"""Registry for data loaders."""
from typing import Dict, Optional, Type
import importlib
import logging

from CONFIG.config_loader import get_cfg
from .interface import DataLoader

logger = logging.getLogger(__name__)

_LOADERS: Dict[str, Type[DataLoader]] = {}
_INSTANCES: Dict[str, DataLoader] = {}


def register_loader(name: str, loader_cls: Type[DataLoader]) -> None:
    """Register a data loader class.

    Args:
        name: Unique name for the loader (e.g., "parquet", "csv")
        loader_cls: DataLoader subclass

    Example:
        ```python
        from TRAINING.data.loading.registry import register_loader

        class MyLoader(DataLoader):
            ...

        register_loader("my_loader", MyLoader)
        ```
    """
    if not issubclass(loader_cls, DataLoader):
        raise TypeError(f"{loader_cls} must be a subclass of DataLoader")
    _LOADERS[name] = loader_cls
    logger.debug(f"Registered data loader: {name}")


def get_loader(
    name: Optional[str] = None,
    options: Optional[Dict] = None
) -> DataLoader:
    """Get a data loader instance by name.

    Args:
        name: Loader name. If None, uses default from config.
        options: Optional override options for the loader.

    Returns:
        DataLoader instance

    Raises:
        ValueError: If loader not found
    """
    if name is None:
        name = get_cfg("data_loading.default_loader", default="parquet")

    # Check cache first
    cache_key = f"{name}:{hash(frozenset((options or {}).items()))}"
    if cache_key in _INSTANCES:
        return _INSTANCES[cache_key]

    # Get loader class
    if name not in _LOADERS:
        # Try loading from config
        loader_config = get_cfg(f"data_loading.loaders.{name}", default=None)
        if loader_config and "class" in loader_config:
            _load_loader_from_config(name, loader_config)
        else:
            raise ValueError(
                f"Unknown data loader: {name}. "
                f"Available: {list(_LOADERS.keys())}"
            )

    loader_cls = _LOADERS[name]

    # Merge options: config defaults + runtime overrides
    config_options = get_cfg(f"data_loading.loaders.{name}.options", default={})
    merged_options = {**config_options, **(options or {})}

    # Create instance
    instance = loader_cls(**merged_options)
    _INSTANCES[cache_key] = instance
    return instance


def _load_loader_from_config(name: str, config: Dict) -> None:
    """Dynamically load a loader class from config."""
    class_path = config["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        loader_cls = getattr(module, class_name)
        register_loader(name, loader_cls)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load loader {name} from {class_path}: {e}")


def list_loaders() -> Dict[str, Type[DataLoader]]:
    """List all registered loaders."""
    return dict(_LOADERS)


def clear_cache() -> None:
    """Clear loader instance cache (for testing)."""
    _INSTANCES.clear()
```

### Task 3: Create Schema Validator

**File**: `TRAINING/data/loading/schema.py`

```python
"""Schema validation for loaded data."""
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import logging

from .interface import SchemaRequirement, SchemaValidationError

logger = logging.getLogger(__name__)


# Default schema for FoxML pipeline
DEFAULT_SCHEMA = SchemaRequirement(
    required_columns={"timestamp"},
    time_column="timestamp",
    index_column=None,
    dtypes=None
)

# Common time column names (auto-detection)
TIME_COLUMN_CANDIDATES = [
    "timestamp", "ts", "time", "datetime", "date",
    "Timestamp", "DateTime", "Date", "Time"
]


def validate_dataframe(
    df: pd.DataFrame,
    schema: SchemaRequirement,
    strict: bool = False,
    auto_fix: bool = True
) -> pd.DataFrame:
    """Validate and optionally fix dataframe schema.

    Args:
        df: DataFrame to validate
        schema: Schema requirements
        strict: If True, raise on validation failure
        auto_fix: If True, attempt to fix common issues

    Returns:
        Validated (and possibly fixed) DataFrame

    Raises:
        SchemaValidationError: If strict mode and validation fails
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check for time column
    time_col = schema.time_column
    if time_col not in df.columns:
        # Try auto-detection
        detected = None
        for candidate in TIME_COLUMN_CANDIDATES:
            if candidate in df.columns:
                detected = candidate
                break

        if detected:
            if auto_fix:
                df = df.rename(columns={detected: time_col})
                warnings.append(f"Renamed '{detected}' to '{time_col}'")
            else:
                errors.append(f"Time column '{time_col}' not found (detected: {detected})")
        else:
            errors.append(f"Time column '{time_col}' not found")

    # Check required columns
    missing = schema.required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check dtypes if specified
    if schema.dtypes:
        for col, expected_dtype in schema.dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not _dtype_compatible(actual_dtype, expected_dtype):
                    if auto_fix:
                        try:
                            df[col] = df[col].astype(expected_dtype)
                            warnings.append(f"Converted '{col}' from {actual_dtype} to {expected_dtype}")
                        except (ValueError, TypeError) as e:
                            errors.append(f"Cannot convert '{col}' to {expected_dtype}: {e}")
                    else:
                        errors.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")

    # Log warnings
    for warning in warnings:
        logger.warning(f"Schema auto-fix: {warning}")

    # Handle errors
    if errors:
        error_msg = f"Schema validation failed: {'; '.join(errors)}"
        if strict:
            raise SchemaValidationError(error_msg)
        else:
            logger.warning(error_msg)

    return df


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Check if dtypes are compatible."""
    # Normalize dtype strings
    actual = actual.lower()
    expected = expected.lower()

    # Direct match
    if actual == expected:
        return True

    # Numeric compatibility
    numeric_types = {"float64", "float32", "int64", "int32", "float", "int"}
    if actual in numeric_types and expected in numeric_types:
        return True

    # String compatibility
    string_types = {"object", "string", "str"}
    if actual in string_types and expected in string_types:
        return True

    return False


def infer_schema(df: pd.DataFrame) -> SchemaRequirement:
    """Infer schema from a dataframe.

    Useful for documenting existing data format.
    """
    # Find time column
    time_col = None
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in df.columns:
            time_col = candidate
            break

    return SchemaRequirement(
        required_columns=set(df.columns),
        time_column=time_col or "timestamp",
        dtypes={col: str(df[col].dtype) for col in df.columns}
    )
```

### Task 4: Implement ParquetLoader

**File**: `TRAINING/data/loading/parquet_loader.py`

```python
"""Parquet file data loader."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

from TRAINING.common.utils.iterdir_sorted import glob_sorted, iterdir_sorted
from .interface import DataLoader, LoadResult, SchemaRequirement
from .schema import validate_dataframe, DEFAULT_SCHEMA
from .registry import register_loader

logger = logging.getLogger(__name__)


class ParquetLoader(DataLoader):
    """Load data from parquet files.

    Supports two directory structures:
    1. New: {data_dir}/interval={interval}/symbol={symbol}/{symbol}.parquet
    2. Legacy: {data_dir}/symbol={symbol}/*.parquet

    Example:
        ```python
        loader = ParquetLoader()
        result = loader.load("/data/prices", ["AAPL", "GOOGL"], interval="5m")
        for symbol, df in result.data.items():
            print(f"{symbol}: {len(df)} rows")
        ```
    """

    def __init__(
        self,
        compression: str = "snappy",
        use_polars: bool = False,
        validate_schema: bool = True,
        **kwargs
    ):
        """Initialize ParquetLoader.

        Args:
            compression: Compression codec (snappy, gzip, zstd)
            use_polars: Use polars instead of pandas (faster)
            validate_schema: Whether to validate schema on load
        """
        self.compression = compression
        self.use_polars = use_polars
        self._validate_schema = validate_schema

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        **kwargs
    ) -> LoadResult:
        """Load parquet data for symbols."""
        data: Dict[str, pd.DataFrame] = {}
        symbols_loaded: List[str] = []
        symbols_failed: List[str] = []
        metadata: Dict[str, Any] = {"interval": interval, "source": source}

        source_path = Path(source)
        schema = schema or DEFAULT_SCHEMA

        for symbol in symbols:
            try:
                # Try new structure first
                parquet_path = source_path / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"

                if not parquet_path.exists():
                    # Try legacy structure
                    legacy_dir = source_path / f"symbol={symbol}"
                    if legacy_dir.exists():
                        parquet_files = glob_sorted(legacy_dir, "*.parquet")
                        if parquet_files:
                            parquet_path = parquet_files[0]

                if not parquet_path.exists():
                    logger.warning(f"No parquet file found for {symbol}")
                    symbols_failed.append(symbol)
                    continue

                # Load data
                if self.use_polars:
                    import polars as pl
                    df = pl.read_parquet(parquet_path).to_pandas()
                else:
                    df = pd.read_parquet(parquet_path)

                # Validate schema
                if self._validate_schema:
                    df = validate_dataframe(df, schema, strict=False, auto_fix=True)

                data[symbol] = df
                symbols_loaded.append(symbol)
                logger.debug(f"Loaded {symbol}: {len(df)} rows")

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                symbols_failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=symbols_loaded,
            symbols_failed=symbols_failed,
            metadata=metadata
        )

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: SchemaRequirement
    ) -> bool:
        """Validate dataframe schema."""
        try:
            validate_dataframe(df, schema, strict=True, auto_fix=False)
            return True
        except Exception:
            return False

    def discover_symbols(
        self,
        source: str,
        interval: Optional[str] = None
    ) -> List[str]:
        """Discover symbols in source directory."""
        source_path = Path(source)
        symbols: List[str] = []

        # Try new structure
        if interval:
            interval_dir = source_path / f"interval={interval}"
            if interval_dir.exists():
                for symbol_dir in iterdir_sorted(interval_dir):
                    if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                        symbol = symbol_dir.name.replace("symbol=", "")
                        symbols.append(symbol)

        # Try legacy structure
        if not symbols:
            for symbol_dir in iterdir_sorted(source_path):
                if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                    symbol = symbol_dir.name.replace("symbol=", "")
                    symbols.append(symbol)

        return sorted(symbols)

    def get_supported_intervals(self, source: str) -> List[str]:
        """Get available intervals in source."""
        source_path = Path(source)
        intervals = []

        for interval_dir in iterdir_sorted(source_path):
            if interval_dir.is_dir() and interval_dir.name.startswith("interval="):
                interval = interval_dir.name.replace("interval=", "")
                intervals.append(interval)

        return sorted(intervals)


# Register loader
register_loader("parquet", ParquetLoader)
```

### Task 5: Implement CSVLoader

**File**: `TRAINING/data/loading/csv_loader.py`

```python
"""CSV file data loader."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

from TRAINING.common.utils.iterdir_sorted import glob_sorted, iterdir_sorted
from .interface import DataLoader, LoadResult, SchemaRequirement
from .schema import validate_dataframe, DEFAULT_SCHEMA
from .registry import register_loader

logger = logging.getLogger(__name__)


class CSVLoader(DataLoader):
    """Load data from CSV files.

    Supports flexible directory structures:
    1. Single file per symbol: {data_dir}/{symbol}.csv
    2. Directory per symbol: {data_dir}/{symbol}/*.csv
    3. Interval structure: {data_dir}/{interval}/{symbol}.csv

    Example:
        ```python
        loader = CSVLoader(delimiter=",", date_column="timestamp")
        result = loader.load("/data/csv", ["AAPL", "GOOGL"])
        ```
    """

    def __init__(
        self,
        delimiter: str = ",",
        date_column: str = "timestamp",
        date_format: Optional[str] = None,
        encoding: str = "utf-8",
        validate_schema: bool = True,
        **kwargs
    ):
        """Initialize CSVLoader.

        Args:
            delimiter: CSV delimiter
            date_column: Name of date/time column
            date_format: Optional date format string
            encoding: File encoding
            validate_schema: Whether to validate schema
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
        **kwargs
    ) -> LoadResult:
        """Load CSV data for symbols."""
        data: Dict[str, pd.DataFrame] = {}
        symbols_loaded: List[str] = []
        symbols_failed: List[str] = []
        metadata: Dict[str, Any] = {"interval": interval, "source": source}

        source_path = Path(source)
        schema = schema or DEFAULT_SCHEMA

        for symbol in symbols:
            try:
                df = self._load_symbol(source_path, symbol, interval)
                if df is None:
                    symbols_failed.append(symbol)
                    continue

                # Parse dates
                if self.date_column in df.columns:
                    df[self.date_column] = pd.to_datetime(
                        df[self.date_column],
                        format=self.date_format,
                        errors='coerce'
                    )

                # Validate schema
                if self._validate_schema:
                    df = validate_dataframe(df, schema, strict=False, auto_fix=True)

                data[symbol] = df
                symbols_loaded.append(symbol)
                logger.debug(f"Loaded {symbol}: {len(df)} rows")

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                symbols_failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=symbols_loaded,
            symbols_failed=symbols_failed,
            metadata=metadata
        )

    def _load_symbol(
        self,
        source_path: Path,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Load CSV for a single symbol."""
        # Try various path patterns
        patterns = [
            source_path / f"{symbol}.csv",
            source_path / interval / f"{symbol}.csv",
            source_path / f"symbol={symbol}" / f"{symbol}.csv",
            source_path / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.csv",
        ]

        for csv_path in patterns:
            if csv_path.exists():
                return pd.read_csv(
                    csv_path,
                    delimiter=self.delimiter,
                    encoding=self.encoding
                )

        # Try directory with multiple CSVs
        symbol_dir = source_path / symbol
        if symbol_dir.exists():
            csv_files = glob_sorted(symbol_dir, "*.csv")
            if csv_files:
                dfs = [
                    pd.read_csv(f, delimiter=self.delimiter, encoding=self.encoding)
                    for f in csv_files
                ]
                return pd.concat(dfs, ignore_index=True)

        logger.warning(f"No CSV file found for {symbol}")
        return None

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: SchemaRequirement
    ) -> bool:
        """Validate dataframe schema."""
        try:
            validate_dataframe(df, schema, strict=True, auto_fix=False)
            return True
        except Exception:
            return False

    def discover_symbols(
        self,
        source: str,
        interval: Optional[str] = None
    ) -> List[str]:
        """Discover symbols from CSV files."""
        source_path = Path(source)
        symbols: List[str] = []

        # Pattern 1: {source}/*.csv
        for csv_file in glob_sorted(source_path, "*.csv"):
            symbols.append(csv_file.stem)

        # Pattern 2: {source}/{interval}/*.csv
        if interval:
            interval_dir = source_path / interval
            if interval_dir.exists():
                for csv_file in glob_sorted(interval_dir, "*.csv"):
                    symbols.append(csv_file.stem)

        # Pattern 3: {source}/symbol={symbol}/
        for symbol_dir in iterdir_sorted(source_path):
            if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                symbol = symbol_dir.name.replace("symbol=", "")
                symbols.append(symbol)

        return sorted(set(symbols))


# Register loader
register_loader("csv", CSVLoader)
```

### Task 6: Update data_loader.py Wrapper

**File**: `TRAINING/data/loading/data_loader.py`

```python
"""
Data loading module.

This is a backward-compatible wrapper around the pluggable loader system.
For new code, prefer using the registry directly:

    from TRAINING.data.loading.registry import get_loader

    loader = get_loader("parquet")  # or "csv", or custom
    result = loader.load(data_dir, symbols, interval)
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from CONFIG.config_loader import get_cfg
from .registry import get_loader
from .interface import SchemaRequirement

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
from .interface import DataLoader, LoadResult, SchemaValidationError
from .schema import SchemaRequirement, DEFAULT_SCHEMA, validate_dataframe
from .registry import register_loader, get_loader, list_loaders

__all__ = [
    # Interface
    "DataLoader",
    "LoadResult",
    "SchemaRequirement",
    "SchemaValidationError",
    # Schema
    "DEFAULT_SCHEMA",
    "validate_dataframe",
    # Registry
    "register_loader",
    "get_loader",
    "list_loaders",
    # Backward compat
    "load_mtf_data",
]


def load_mtf_data(
    data_dir: str,
    symbols: List[str],
    interval: str = "5m",
    max_rows_per_symbol: Optional[int] = None,
    loader_name: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """Load multi-timeframe data for symbols.

    This is a backward-compatible wrapper around the new loader system.

    Args:
        data_dir: Path to data directory
        symbols: List of symbols to load
        interval: Data interval (e.g., "5m", "1h")
        max_rows_per_symbol: Optional row limit per symbol
        loader_name: Optional loader name (default from config)
        **kwargs: Additional loader options

    Returns:
        Dict mapping symbol to DataFrame
    """
    # Get loader (from config or argument)
    loader_name = loader_name or get_cfg("data_loading.default_loader", default="parquet")
    loader = get_loader(loader_name)

    # Load data
    result = loader.load(
        source=data_dir,
        symbols=symbols,
        interval=interval,
        **kwargs
    )

    # Apply row limit if specified
    data = result.data
    if max_rows_per_symbol:
        data = {
            symbol: df.head(max_rows_per_symbol)
            for symbol, df in data.items()
        }

    # Log summary
    if result.symbols_failed:
        logger.warning(f"Failed to load {len(result.symbols_failed)} symbols: {result.symbols_failed[:5]}...")

    return data
```

---

## Checklist

### Task 1: Interface
- [ ] Create `TRAINING/data/loading/interface.py`
- [ ] Define `DataLoader` ABC
- [ ] Define `LoadResult` dataclass
- [ ] Define `SchemaRequirement` dataclass
- [ ] Add docstrings with examples

### Task 2: Registry
- [ ] Create `TRAINING/data/loading/registry.py`
- [ ] Implement `register_loader()`
- [ ] Implement `get_loader()`
- [ ] Add config-based dynamic loading
- [ ] Add instance caching

### Task 3: Schema Validation
- [ ] Create `TRAINING/data/loading/schema.py`
- [ ] Implement `validate_dataframe()`
- [ ] Implement time column auto-detection
- [ ] Implement dtype compatibility checking
- [ ] Implement `infer_schema()` helper

### Task 4: ParquetLoader
- [ ] Create `TRAINING/data/loading/parquet_loader.py`
- [ ] Migrate existing parquet loading logic
- [ ] Implement `discover_symbols()`
- [ ] Implement `get_supported_intervals()`
- [ ] Register with registry

### Task 5: CSVLoader
- [ ] Create `TRAINING/data/loading/csv_loader.py`
- [ ] Implement flexible path patterns
- [ ] Implement date parsing
- [ ] Implement multi-file concat
- [ ] Register with registry

### Task 6: Wrapper Update
- [ ] Update `TRAINING/data/loading/data_loader.py`
- [ ] Add backward-compatible `load_mtf_data()`
- [ ] Re-export public API
- [ ] Add deprecation warnings where appropriate

### Task 7: Config
- [ ] Create `CONFIG/data/data_loading.yaml`
- [ ] Add default_loader setting
- [ ] Add loader configurations
- [ ] Document loader options

### Task 8: Documentation
- [ ] Update `CONFIG/data/README.md`
- [ ] Create `DATA_LOADER_PLUGINS.md` in Phase 5

### Task 9: Tests
- [ ] Create `tests/test_data_loader_interface.py`
- [ ] Create `tests/test_parquet_loader.py`
- [ ] Create `tests/test_csv_loader.py`
- [ ] Create `tests/test_schema_validation.py`

---

## Testing Strategy

### Unit Tests
```python
# tests/test_data_loader_interface.py
def test_parquet_loader_implements_interface():
    from TRAINING.data.loading import ParquetLoader, DataLoader
    assert issubclass(ParquetLoader, DataLoader)

def test_csv_loader_implements_interface():
    from TRAINING.data.loading import CSVLoader, DataLoader
    assert issubclass(CSVLoader, DataLoader)

def test_registry_returns_correct_loader():
    from TRAINING.data.loading import get_loader
    loader = get_loader("parquet")
    assert loader.__class__.__name__ == "ParquetLoader"

def test_backward_compat_load_mtf_data():
    from TRAINING.data.loading import load_mtf_data
    # Should work exactly as before
    data = load_mtf_data("/path/to/data", ["TEST"], interval="5m")
```

### Integration Tests
```bash
# Test with real data
pytest tests/test_data_loading_integration.py -v

# Test CSV loading
python -c "
from TRAINING.data.loading import get_loader
loader = get_loader('csv')
result = loader.load('/tmp/test_csv', ['TEST'], interval='5m')
print(result)
"
```

---

## Migration Guide

### For Existing Code

No changes required - `load_mtf_data()` continues to work:
```python
# This still works
from TRAINING.data.loading.data_loader import load_mtf_data
data = load_mtf_data(data_dir, symbols, interval)
```

### For New Code

Use the registry for more control:
```python
from TRAINING.data.loading import get_loader

# Use default loader from config
loader = get_loader()
result = loader.load(data_dir, symbols, interval)

# Use specific loader
csv_loader = get_loader("csv")
result = csv_loader.load("/path/to/csv", symbols)

# Custom loader with options
loader = get_loader("csv", options={"delimiter": ";"})
```

### For Custom Loaders

Implement the interface and register:
```python
from TRAINING.data.loading import DataLoader, register_loader, LoadResult

class MyCustomLoader(DataLoader):
    def load(self, source, symbols, interval="5m", **kwargs):
        # Custom loading logic
        data = {}
        for symbol in symbols:
            df = my_custom_load(source, symbol)
            data[symbol] = df
        return LoadResult(data=data, symbols_loaded=list(data.keys()), ...)

    def validate_schema(self, df, schema):
        # Custom validation
        return True

    def discover_symbols(self, source, interval=None):
        # Custom discovery
        return ["SYM1", "SYM2"]

# Register
register_loader("my_custom", MyCustomLoader)

# Or via config
# CONFIG/experiments/my_experiment.yaml
# data:
#   loader: my_custom
#   loader_options:
#     my_option: value
```

---

## Success Criteria

- [ ] All existing tests pass (backward compatibility)
- [ ] ParquetLoader matches current behavior exactly
- [ ] CSVLoader can load CSV files with flexible paths
- [ ] Custom loaders can be registered at runtime
- [ ] Config can specify default and per-experiment loaders
- [ ] Schema validation catches common data issues
- [ ] Documentation explains extension pattern
