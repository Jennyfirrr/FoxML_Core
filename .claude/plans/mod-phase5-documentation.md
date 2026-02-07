# Phase 5: Custom Data Documentation

**Status**: ✅ Complete
**Priority**: P0 (User Enablement)
**Effort**: 8 hours
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)
**Depends On**: Phase 4 (Data Loader Interface)
**Completed**: 2026-01-19

---

## Quick Resume

```
STATUS: COMPLETE
FILES CREATED:
  - DOCS/01_tutorials/pipelines/CUSTOM_DATASETS.md
  - DOCS/01_tutorials/pipelines/CUSTOM_FEATURES.md
  - DOCS/01_tutorials/pipelines/DATA_LOADER_PLUGINS.md
```

---

## Problem Statement

Current state:
- **Zero documentation** for adding custom datasets
- **Incomplete documentation** for custom features
- **No guide** for writing custom data loaders
- Users with CSV/SQL data are stuck

**Goal**: Enable users to bring their own data without reading source code

---

## Files to Create

### 1. CUSTOM_DATASETS.md

**Location**: `DOCS/01_tutorials/pipelines/CUSTOM_DATASETS.md`

**Outline**:
```markdown
# Adding Custom Datasets

## Overview
- What the pipeline expects
- Supported formats (parquet, CSV, custom)

## Step 1: Prepare Your Data

### Required Columns
- `timestamp` (or configured time column)
- Price columns: `open`, `high`, `low`, `close`, `volume`
- Feature columns (your custom features)
- Target columns (if pre-computed)

### Data Types
| Column | Type | Notes |
|--------|------|-------|
| timestamp | datetime64[ns] | Required |
| open/high/low/close | float64 | Required |
| volume | float64/int64 | Optional |
| features | float64 | Your custom features |

### Example DataFrame
```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='5min'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'my_custom_feature': [...],
})
```

## Step 2: Directory Structure

### Option A: Parquet (Recommended)
```
data/
└── interval=5m/
    └── symbol=AAPL/
        └── AAPL.parquet
```

### Option B: CSV
```
data/
├── AAPL.csv
├── GOOGL.csv
└── MSFT.csv
```

### Option C: Per-Symbol Directories
```
data/
├── AAPL/
│   ├── 2024-01.csv
│   └── 2024-02.csv
└── GOOGL/
    ├── 2024-01.csv
    └── 2024-02.csv
```

## Step 3: Convert CSV to Parquet (Optional)

```python
import pandas as pd
from pathlib import Path

def convert_csv_to_parquet(csv_dir: str, output_dir: str, interval: str = "5m"):
    csv_path = Path(csv_dir)
    out_path = Path(output_dir)

    for csv_file in csv_path.glob("*.csv"):
        symbol = csv_file.stem
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])

        # Create output directory
        symbol_dir = out_path / f"interval={interval}" / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        df.to_parquet(symbol_dir / f"{symbol}.parquet", index=False)
        print(f"Converted {symbol}")

# Usage
convert_csv_to_parquet("/path/to/csv", "/path/to/parquet", interval="5m")
```

## Step 4: Create Experiment Config

```yaml
# CONFIG/experiments/my_custom_data.yaml
experiment:
  name: my_custom_data
  description: Experiment with custom dataset

data:
  data_dir: /path/to/your/data
  loader: parquet  # or "csv"
  interval: 5m
  symbols: []  # Empty = auto-discover

# Optional: loader-specific options
loader_options:
  delimiter: ","        # For CSV
  date_column: timestamp
```

## Step 5: Run Pipeline

```bash
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir my_experiment \
    --experiment-config CONFIG/experiments/my_custom_data.yaml
```

## Troubleshooting

### "No symbols found"
- Check directory structure matches expected pattern
- Verify parquet/CSV files exist
- Check file permissions

### "Missing required columns"
- Verify `timestamp` column exists
- Check column names are lowercase
- Validate date parsing

### "Schema validation failed"
- Run schema validator:
  ```python
  from TRAINING.data.loading import validate_dataframe, infer_schema
  schema = infer_schema(your_df)
  print(schema)
  ```

## Next Steps
- [Adding Custom Features](./CUSTOM_FEATURES.md)
- [Writing Custom Data Loaders](./DATA_LOADER_PLUGINS.md)
```

---

### 2. CUSTOM_FEATURES.md

**Location**: `DOCS/01_tutorials/pipelines/CUSTOM_FEATURES.md`

**Outline**:
```markdown
# Adding Custom Features

## Overview
- Feature registry system
- Leakage prevention
- Two registration methods

## Method 1: YAML Registry (Recommended)

### Add to feature_registry.yaml

```yaml
# CONFIG/data/feature_registry.yaml

# Add your feature family
feature_families:
  my_custom:
    pattern: ^my_custom_
    description: My custom indicator family
    default_lag_bars: 2
    default_allowed_horizons: [5, 10, 30, 60]

# Or add specific features
features:
  my_custom_momentum_10:
    source: derived
    lag_bars: 10
    lookback_minutes: 50  # For 5m data: 10 bars * 5 min
    allowed_horizons: [5, 10, 30]
    description: Custom momentum indicator with 10-bar lookback
```

### Leakage Prevention Rules
- `lag_bars` must be >= lookback of feature computation
- `allowed_horizons` should exclude horizons where feature could leak
- Rule of thumb: `horizon > lag_bars` to be safe

## Method 2: Python Registration (Advanced)

```python
from TRAINING.common.feature_registry import get_registry

registry = get_registry()

# Register single feature
registry.register_feature("my_feature", {
    "source": "derived",
    "lag_bars": 5,
    "lookback_minutes": 25,
    "allowed_horizons": [10, 30, 60],
    "description": "My custom feature"
})

# Register feature family
registry.register_family("my_family", {
    "pattern": r"^my_family_(\d+)$",
    "default_lag_bars": 2,
    "default_allowed_horizons": [5, 10, 30, 60]
})
```

## Computing Custom Features

### Option A: Pre-compute and Include in Data

```python
import pandas as pd

def compute_my_feature(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Compute custom feature."""
    return df['close'].rolling(lookback).mean() / df['close'] - 1

# Add to your data
df['my_custom_feature'] = compute_my_feature(df)
```

### Option B: Feature Transform Pipeline (Future)

```yaml
# CONFIG/data/feature_transforms.yaml
transforms:
  my_custom_feature:
    compute_fn: my_module.compute_my_feature
    args:
      lookback: 10
```

## Testing Your Feature

### Leakage Test

```python
from TRAINING.ranking.predictability.leakage_detection import detect_feature_leakage

# Test feature for leakage against target
result = detect_feature_leakage(
    df=your_df,
    feature_column="my_custom_feature",
    target_column="fwd_ret_5",
    target_horizon=5
)

if result.has_leakage:
    print(f"WARNING: Feature has leakage: {result.reason}")
```

### Registry Validation

```python
from TRAINING.common.feature_registry import get_registry

registry = get_registry()

# Check if feature is allowed for horizon
is_ok = registry.is_allowed("my_custom_feature", target_horizon=5)
print(f"Allowed for horizon 5: {is_ok}")

# Get feature metadata
metadata = registry.get_feature_metadata("my_custom_feature")
print(metadata)
```

## Best Practices

1. **Always specify lag_bars** - Prevents accidental leakage
2. **Use lookback_minutes** - Required for interval-agnostic features
3. **Test before training** - Run leakage detection on new features
4. **Document your features** - Add descriptions to registry

## Common Patterns

### Momentum Features
```yaml
my_momentum_10:
  source: derived
  lag_bars: 10
  allowed_horizons: [10, 30, 60]  # Exclude 5 (too close to lookback)
```

### Rolling Statistics
```yaml
my_rolling_std_20:
  source: derived
  lag_bars: 20
  allowed_horizons: [30, 60, 120]  # Safe margin from lookback
```

### Cross-Sectional Features
```yaml
my_relative_strength:
  source: cross_sectional
  lag_bars: 1
  allowed_horizons: [5, 10, 30, 60]
```
```

---

### 3. DATA_LOADER_PLUGINS.md

**Location**: `DOCS/01_tutorials/pipelines/DATA_LOADER_PLUGINS.md`

**Outline**:
```markdown
# Writing Custom Data Loaders

## Overview
- DataLoader interface
- Registration system
- Example implementations

## The DataLoader Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

from TRAINING.data.loading import (
    DataLoader,
    LoadResult,
    SchemaRequirement,
    register_loader
)

class MyCustomLoader(DataLoader):
    """Custom data loader implementation."""

    def __init__(self, my_option: str = "default", **kwargs):
        self.my_option = my_option

    def load(
        self,
        source: str,
        symbols: List[str],
        interval: str = "5m",
        schema: Optional[SchemaRequirement] = None,
        **kwargs
    ) -> LoadResult:
        """Load data from custom source."""
        data = {}
        loaded = []
        failed = []

        for symbol in symbols:
            try:
                df = self._load_symbol(source, symbol)
                data[symbol] = df
                loaded.append(symbol)
            except Exception as e:
                failed.append(symbol)

        return LoadResult(
            data=data,
            symbols_loaded=loaded,
            symbols_failed=failed,
            metadata={"source": source}
        )

    def validate_schema(self, df: pd.DataFrame, schema: SchemaRequirement) -> bool:
        # Implement validation
        return True

    def discover_symbols(self, source: str, interval: Optional[str] = None) -> List[str]:
        # Implement discovery
        return []

    def _load_symbol(self, source: str, symbol: str) -> pd.DataFrame:
        # Your custom loading logic
        ...

# Register the loader
register_loader("my_custom", MyCustomLoader)
```

## Example: SQL Database Loader

```python
import pandas as pd
from sqlalchemy import create_engine

class SQLLoader(DataLoader):
    def __init__(self, connection_string: str, table_name: str = "prices", **kwargs):
        self.engine = create_engine(connection_string)
        self.table_name = table_name

    def load(self, source, symbols, interval="5m", **kwargs):
        data = {}
        for symbol in symbols:
            query = f'''
                SELECT * FROM {self.table_name}
                WHERE symbol = '{symbol}'
                AND interval = '{interval}'
                ORDER BY timestamp
            '''
            df = pd.read_sql(query, self.engine)
            data[symbol] = df
        return LoadResult(data=data, symbols_loaded=list(data.keys()), ...)

    def discover_symbols(self, source, interval=None):
        query = f"SELECT DISTINCT symbol FROM {self.table_name}"
        result = pd.read_sql(query, self.engine)
        return result['symbol'].tolist()

register_loader("sql", SQLLoader)
```

## Example: API Loader

```python
import requests
import pandas as pd

class APILoader(DataLoader):
    def __init__(self, api_key: str, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url

    def load(self, source, symbols, interval="5m", **kwargs):
        data = {}
        for symbol in symbols:
            response = requests.get(
                f"{self.base_url}/prices",
                params={"symbol": symbol, "interval": interval},
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            df = pd.DataFrame(response.json()["data"])
            data[symbol] = df
        return LoadResult(data=data, ...)

register_loader("api", APILoader)
```

## Configuration

### Via Config File

```yaml
# CONFIG/data/data_loading.yaml
data_loading:
  loaders:
    my_custom:
      class: my_project.loaders.MyCustomLoader
      options:
        my_option: custom_value

    sql:
      class: TRAINING.data.loading.sql_loader.SQLLoader
      options:
        connection_string: postgresql://localhost/prices
        table_name: price_data
```

### Via Experiment Config

```yaml
# CONFIG/experiments/my_experiment.yaml
data:
  loader: sql
  loader_options:
    connection_string: postgresql://prod/prices
```

## Testing Your Loader

```python
import pytest
from TRAINING.data.loading import DataLoader

def test_my_loader_implements_interface():
    from my_project.loaders import MyCustomLoader
    assert issubclass(MyCustomLoader, DataLoader)

def test_my_loader_loads_data():
    from my_project.loaders import MyCustomLoader
    loader = MyCustomLoader()
    result = loader.load("/path/to/data", ["TEST"])
    assert "TEST" in result.data
    assert len(result.symbols_failed) == 0

def test_my_loader_discovers_symbols():
    from my_project.loaders import MyCustomLoader
    loader = MyCustomLoader()
    symbols = loader.discover_symbols("/path/to/data")
    assert len(symbols) > 0
```
```

---

## Checklist

### CUSTOM_DATASETS.md
- [ ] Write data format requirements section
- [ ] Document all directory structure options
- [ ] Add CSV to parquet conversion script
- [ ] Add experiment config examples
- [ ] Add troubleshooting section
- [ ] Review for accuracy

### CUSTOM_FEATURES.md
- [ ] Document YAML registration (recommended)
- [ ] Document Python registration (advanced)
- [ ] Add leakage prevention rules
- [ ] Add testing examples
- [ ] Add best practices
- [ ] Review for accuracy

### DATA_LOADER_PLUGINS.md
- [ ] Document DataLoader interface
- [ ] Add SQL loader example
- [ ] Add API loader example
- [ ] Document config integration
- [ ] Add testing examples
- [ ] Review for accuracy

### Integration
- [ ] Update CONFIG/data/README.md with links
- [ ] Update CLAUDE.md if needed
- [ ] Add cross-references between docs
- [ ] Run spell check

---

## Success Criteria

- [ ] New user can add parquet dataset following only CUSTOM_DATASETS.md
- [ ] New user can add CSV dataset following only CUSTOM_DATASETS.md
- [ ] New user can register custom feature following only CUSTOM_FEATURES.md
- [ ] Developer can write custom loader following only DATA_LOADER_PLUGINS.md
- [ ] All code examples are tested and working
