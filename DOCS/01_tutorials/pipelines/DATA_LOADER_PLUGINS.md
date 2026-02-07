# Writing Custom Data Loaders

This guide explains how to create custom data loaders for SQL databases, APIs, or any custom data source.

## Overview

The data loading system uses a pluggable architecture:
- **DataLoader** - Abstract base class defining the interface
- **Registry** - Manages loader registration and retrieval
- **Built-in loaders** - ParquetLoader and CSVLoader

You can add custom loaders for any data source.

## The DataLoader Interface

All loaders must implement three methods:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

from TRAINING.data.loading import (
    DataLoader,
    LoadResult,
    SchemaRequirement,
    register_loader,
)


class MyCustomLoader(DataLoader):
    """Custom data loader implementation."""

    def __init__(self, my_option: str = "default", **kwargs):
        """Initialize with custom options."""
        self.my_option = my_option

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
            source: Connection string, path, or identifier
            symbols: List of symbols to load
            interval: Data interval (e.g., "5m", "1h")
            schema: Optional schema requirements
            **kwargs: Additional options

        Returns:
            LoadResult with data, loaded/failed symbols, and metadata
        """
        data = {}
        loaded = []
        failed = []

        for symbol in symbols:
            try:
                df = self._load_symbol(source, symbol, interval)
                data[symbol] = df
                loaded.append(symbol)
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
                failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=sorted(loaded),
            symbols_failed=sorted(failed),
            metadata={"source": source, "interval": interval},
        )

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: SchemaRequirement,
    ) -> bool:
        """Validate dataframe against schema.

        Args:
            df: DataFrame to validate
            schema: Schema requirements

        Returns:
            True if valid
        """
        required = schema.required_columns
        return required.issubset(set(df.columns))

    def discover_symbols(
        self,
        source: str,
        interval: Optional[str] = None,
    ) -> List[str]:
        """Discover available symbols.

        Args:
            source: Data source identifier
            interval: Optional interval filter

        Returns:
            Sorted list of symbol names
        """
        # Implement symbol discovery logic
        return []

    def _load_symbol(
        self,
        source: str,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        """Load data for a single symbol."""
        # Your custom loading logic here
        raise NotImplementedError


# Register the loader
register_loader("my_custom", MyCustomLoader)
```

## Example: SQL Database Loader

```python
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional

from TRAINING.data.loading import (
    DataLoader,
    LoadResult,
    SchemaRequirement,
    register_loader,
)


class SQLLoader(DataLoader):
    """Load data from SQL database."""

    def __init__(
        self,
        connection_string: str = "",
        table_name: str = "prices",
        **kwargs,
    ):
        """Initialize SQL loader.

        Args:
            connection_string: SQLAlchemy connection string
            table_name: Table containing price data
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._engine = None

    @property
    def engine(self):
        """Lazy engine initialization."""
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        **kwargs,
    ) -> LoadResult:
        """Load data from SQL database."""
        # Use source as connection string if provided
        conn_string = source or self.connection_string
        engine = create_engine(conn_string)

        data = {}
        loaded = []
        failed = []

        for symbol in symbols:
            try:
                # Parameterized query for safety
                query = text(f"""
                    SELECT *
                    FROM {self.table_name}
                    WHERE symbol = :symbol
                    AND interval = :interval
                    ORDER BY timestamp
                """)

                df = pd.read_sql(
                    query,
                    engine,
                    params={"symbol": symbol, "interval": interval},
                )

                if len(df) > 0:
                    data[symbol] = df
                    loaded.append(symbol)
                else:
                    failed.append(symbol)

            except Exception as e:
                print(f"SQL error for {symbol}: {e}")
                failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=sorted(loaded),
            symbols_failed=sorted(failed),
            metadata={
                "source": conn_string,
                "table": self.table_name,
                "interval": interval,
            },
        )

    def validate_schema(self, df: pd.DataFrame, schema: SchemaRequirement) -> bool:
        """Validate dataframe schema."""
        from TRAINING.data.loading.schema import check_schema
        errors = check_schema(df, schema)
        return len(errors) == 0

    def discover_symbols(
        self,
        source: str,
        interval: Optional[str] = None,
    ) -> List[str]:
        """Discover symbols in database."""
        conn_string = source or self.connection_string
        engine = create_engine(conn_string)

        query = f"SELECT DISTINCT symbol FROM {self.table_name}"
        if interval:
            query += f" WHERE interval = '{interval}'"
        query += " ORDER BY symbol"

        result = pd.read_sql(query, engine)
        return result["symbol"].tolist()


# Register the loader
register_loader("sql", SQLLoader)
```

## Example: REST API Loader

```python
import pandas as pd
import requests
from typing import Dict, List, Optional

from TRAINING.data.loading import (
    DataLoader,
    LoadResult,
    register_loader,
)


class APILoader(DataLoader):
    """Load data from REST API."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.example.com",
        timeout: int = 30,
        **kwargs,
    ):
        """Initialize API loader.

        Args:
            api_key: API authentication key
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema=None,
        **kwargs,
    ) -> LoadResult:
        """Load data from API."""
        data = {}
        loaded = []
        failed = []

        # Use source as base_url if provided
        base_url = source or self.base_url

        for symbol in symbols:
            try:
                response = requests.get(
                    f"{base_url}/prices",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                # Parse response
                json_data = response.json()
                df = pd.DataFrame(json_data["data"])

                # Convert timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                data[symbol] = df
                loaded.append(symbol)

            except requests.RequestException as e:
                print(f"API error for {symbol}: {e}")
                failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=sorted(loaded),
            symbols_failed=sorted(failed),
            metadata={"source": base_url, "interval": interval},
        )

    def validate_schema(self, df, schema) -> bool:
        return "timestamp" in df.columns

    def discover_symbols(self, source: str, interval: Optional[str] = None) -> List[str]:
        """Discover symbols from API."""
        base_url = source or self.base_url

        try:
            response = requests.get(
                f"{base_url}/symbols",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return sorted(response.json()["symbols"])
        except Exception:
            return []


register_loader("api", APILoader)
```

## Example: HDF5 Loader

```python
import pandas as pd
from pathlib import Path
from typing import List, Optional

from TRAINING.data.loading import DataLoader, LoadResult, register_loader


class HDF5Loader(DataLoader):
    """Load data from HDF5 files."""

    def __init__(self, key_pattern: str = "/data/{symbol}", **kwargs):
        """Initialize HDF5 loader.

        Args:
            key_pattern: HDF5 key pattern with {symbol} placeholder
        """
        self.key_pattern = key_pattern

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema=None,
        **kwargs,
    ) -> LoadResult:
        """Load data from HDF5 file."""
        data = {}
        loaded = []
        failed = []

        with pd.HDFStore(source, mode="r") as store:
            for symbol in symbols:
                key = self.key_pattern.format(symbol=symbol, interval=interval)
                try:
                    if key in store:
                        df = store[key]
                        data[symbol] = df
                        loaded.append(symbol)
                    else:
                        failed.append(symbol)
                except Exception as e:
                    print(f"HDF5 error for {symbol}: {e}")
                    failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=sorted(loaded),
            symbols_failed=sorted(failed),
            metadata={"source": source},
        )

    def validate_schema(self, df, schema) -> bool:
        return "timestamp" in df.columns

    def discover_symbols(self, source: str, interval: Optional[str] = None) -> List[str]:
        """Discover symbols in HDF5 file."""
        symbols = []
        with pd.HDFStore(source, mode="r") as store:
            for key in store.keys():
                # Extract symbol from key
                parts = key.strip("/").split("/")
                if len(parts) >= 2:
                    symbols.append(parts[-1])
        return sorted(set(symbols))


register_loader("hdf5", HDF5Loader)
```

## Configuration

### Via Config File

Register loaders in the config system:

```yaml
# CONFIG/data/data_loading.yaml
data_loading:
  default_loader: parquet

  loaders:
    sql:
      class: TRAINING.data.loading.sql_loader.SQLLoader
      options:
        connection_string: postgresql://localhost/prices
        table_name: price_data

    api:
      class: my_project.loaders.APILoader
      options:
        base_url: https://api.mydata.com
        api_key: ${API_KEY}  # Environment variable

    hdf5:
      class: my_project.loaders.HDF5Loader
      options:
        key_pattern: /prices/{interval}/{symbol}
```

### Via Experiment Config

Override loader per experiment:

```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  loader: sql
  data_dir: postgresql://prod-db/prices  # Used as 'source'

loader_options:
  table_name: production_prices
```

### Programmatic Registration

```python
from TRAINING.data.loading import register_loader
from my_project.loaders import MyCustomLoader

# Register at module import
register_loader("my_custom", MyCustomLoader)

# Or conditionally
if HAS_SPECIAL_LIBRARY:
    register_loader("special", SpecialLoader)
```

## Testing Your Loader

### Unit Tests

```python
import pytest
import pandas as pd
from TRAINING.data.loading import DataLoader, LoadResult


class TestMyLoader:
    """Tests for custom loader."""

    def test_implements_interface(self):
        """Verify loader implements DataLoader."""
        from my_project.loaders import MyCustomLoader
        assert issubclass(MyCustomLoader, DataLoader)

    def test_load_returns_loadresult(self):
        """Verify load returns correct type."""
        from my_project.loaders import MyCustomLoader
        loader = MyCustomLoader()
        result = loader.load("/test/source", ["TEST"])
        assert isinstance(result, LoadResult)

    def test_load_single_symbol(self):
        """Test loading single symbol."""
        from my_project.loaders import MyCustomLoader
        loader = MyCustomLoader()
        result = loader.load("/test/source", ["AAPL"])

        # Check result structure
        assert "AAPL" in result.data or "AAPL" in result.symbols_failed
        assert isinstance(result.symbols_loaded, list)
        assert isinstance(result.symbols_failed, list)

    def test_discover_symbols(self):
        """Test symbol discovery."""
        from my_project.loaders import MyCustomLoader
        loader = MyCustomLoader()
        symbols = loader.discover_symbols("/test/source")
        assert isinstance(symbols, list)
        # Should be sorted
        assert symbols == sorted(symbols)

    def test_validate_schema(self):
        """Test schema validation."""
        from my_project.loaders import MyCustomLoader
        from TRAINING.data.loading import SchemaRequirement

        loader = MyCustomLoader()
        schema = SchemaRequirement(required_columns={"timestamp", "close"})

        # Valid dataframe
        df_valid = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "close": range(10),
        })
        assert loader.validate_schema(df_valid, schema)

        # Invalid dataframe
        df_invalid = pd.DataFrame({"other": range(10)})
        assert not loader.validate_schema(df_invalid, schema)
```

### Integration Test

```python
def test_loader_integration():
    """End-to-end loader test."""
    from TRAINING.data.loading import get_loader, register_loader
    from my_project.loaders import MyCustomLoader

    # Register
    register_loader("test_custom", MyCustomLoader)

    # Get via registry
    loader = get_loader("test_custom")
    assert isinstance(loader, MyCustomLoader)

    # Load data
    result = loader.load("/path/to/test/data", ["TEST_SYMBOL"])

    # Verify
    if result.symbols_loaded:
        df = result.data[result.symbols_loaded[0]]
        assert "timestamp" in df.columns
        assert len(df) > 0
```

## Best Practices

1. **Handle errors gracefully** - Return failed symbols, don't raise
2. **Sort output lists** - For determinism
3. **Lazy initialization** - Don't connect in `__init__`
4. **Support options via kwargs** - For flexibility
5. **Implement all methods** - Even if they return empty
6. **Add logging** - For debugging
7. **Document options** - In docstrings and README

## Troubleshooting

### "Loader not found"

The loader isn't registered when needed.

**Fix**: Ensure registration happens before use:
```python
# In __init__.py or early in startup
from my_project.loaders import MyLoader
register_loader("my_loader", MyLoader)
```

### "Circular import"

Registration causes import cycle.

**Fix**: Use lazy imports or register in a separate module:
```python
# loaders/__init__.py
def register_all():
    from .my_loader import MyLoader
    register_loader("my_loader", MyLoader)
```

### "Connection timeout"

Database or API is slow.

**Fix**: Implement connection pooling or increase timeout:
```python
def __init__(self, timeout: int = 60, pool_size: int = 5):
    self.engine = create_engine(
        conn_string,
        pool_size=pool_size,
        pool_timeout=timeout,
    )
```

## Next Steps

- [Adding Custom Datasets](./CUSTOM_DATASETS.md) - Data format requirements
- [Adding Custom Features](./CUSTOM_FEATURES.md) - Feature registration
