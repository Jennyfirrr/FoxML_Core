# Adding Custom Datasets

This guide explains how to add your own datasets to the FoxML pipeline.

## Overview

The pipeline supports multiple data formats:
- **Parquet** (recommended) - Fast, compressed, type-safe
- **CSV** - Simple, widely compatible
- **Custom loaders** - SQL, APIs, or any custom source

## Step 1: Prepare Your Data

### Required Columns

Your data must include a timestamp column. The pipeline auto-detects these names:
- `timestamp` (preferred)
- `ts`, `time`, `datetime`, `date`

### Recommended Columns

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | datetime64[ns] | Required - when the bar occurred |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | float64/int64 | Trading volume |

### Feature Columns

Add your features directly to the DataFrame. Name them descriptively:

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='5min'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    # Your custom features
    'sma_20': [...],
    'rsi_14': [...],
    'my_custom_signal': [...],
})
```

### Target Columns (Optional)

If you pre-compute targets, use the naming convention:
- `fwd_ret_{horizon}m` - Forward returns (e.g., `fwd_ret_5m`, `fwd_ret_30m`)
- `will_peak_{horizon}m` - Binary peak prediction
- `will_valley_{horizon}m` - Binary valley prediction

## Step 2: Directory Structure

### Option A: Parquet with Hive Partitioning (Recommended)

```
data/
└── interval=5m/
    ├── symbol=AAPL/
    │   └── AAPL.parquet
    ├── symbol=GOOGL/
    │   └── GOOGL.parquet
    └── symbol=MSFT/
        └── MSFT.parquet
```

This structure supports:
- Multiple intervals (`interval=1m`, `interval=5m`, `interval=1h`)
- Efficient symbol discovery
- Partition pruning for large datasets

### Option B: Flat Parquet Files

```
data/
├── AAPL_mtf.parquet
├── GOOGL_mtf.parquet
└── MSFT_mtf.parquet
```

Legacy structure, still supported.

### Option C: CSV Files

```
data/
├── AAPL.csv
├── GOOGL.csv
└── MSFT.csv
```

Or with interval subdirectories:

```
data/
└── 5m/
    ├── AAPL.csv
    ├── GOOGL.csv
    └── MSFT.csv
```

### Option D: Per-Symbol Directories (Multi-file)

```
data/
├── AAPL/
│   ├── 2024-01.csv
│   ├── 2024-02.csv
│   └── 2024-03.csv
└── GOOGL/
    ├── 2024-01.csv
    └── 2024-02.csv
```

CSVs are automatically concatenated per symbol.

## Step 3: Convert CSV to Parquet (Optional)

Parquet is 5-10x faster to load. Here's a conversion script:

```python
import pandas as pd
from pathlib import Path

def convert_csv_to_parquet(csv_dir: str, output_dir: str, interval: str = "5m"):
    """Convert CSV files to Hive-partitioned parquet."""
    csv_path = Path(csv_dir)
    out_path = Path(output_dir)

    for csv_file in sorted(csv_path.glob("*.csv")):
        symbol = csv_file.stem
        print(f"Converting {symbol}...")

        # Load CSV
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])

        # Create output directory
        symbol_dir = out_path / f"interval={interval}" / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        df.to_parquet(symbol_dir / f"{symbol}.parquet", index=False)
        print(f"  Saved {len(df):,} rows to {symbol_dir}")

# Usage
convert_csv_to_parquet(
    csv_dir="/path/to/csv/files",
    output_dir="/path/to/parquet/output",
    interval="5m"
)
```

## Step 4: Create Experiment Config

Create a YAML config to tell the pipeline about your data:

```yaml
# CONFIG/experiments/my_custom_data.yaml
experiment:
  name: my_custom_data
  description: Experiment with my custom dataset

data:
  data_dir: /path/to/your/data
  loader: parquet  # or "csv"
  interval: 5m
  symbols: []  # Empty = auto-discover all symbols

# Optional: Override loader options
loader_options:
  validate_schema: true
```

### CSV-Specific Options

```yaml
data:
  loader: csv

loader_options:
  delimiter: ","
  date_column: timestamp
  date_format: "%Y-%m-%d %H:%M:%S"  # Optional strptime format
  encoding: utf-8
```

## Step 5: Run the Pipeline

```bash
# Run with your custom config
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir my_experiment \
    --experiment-config CONFIG/experiments/my_custom_data.yaml

# For deterministic runs
bash bin/run_deterministic.sh \
    -m TRAINING.orchestration.intelligent_trainer \
    --output-dir my_experiment \
    --experiment-config CONFIG/experiments/my_custom_data.yaml
```

## Step 6: Validate Your Data (Optional)

Before running the full pipeline, validate your data:

```python
from TRAINING.data.loading import get_loader, infer_schema, validate_dataframe
import pandas as pd

# Load a sample
loader = get_loader("parquet")  # or "csv"
result = loader.load("/path/to/data", ["AAPL"], interval="5m")

if result.symbols_loaded:
    df = result.data["AAPL"]

    # Infer and print schema
    schema = infer_schema(df)
    print("Detected schema:")
    print(f"  Time column: {schema.time_column}")
    print(f"  Columns: {len(schema.required_columns)}")
    print(f"  Dtypes: {schema.dtypes}")

    # Validate
    errors = check_schema(df, schema)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Schema validation passed!")
else:
    print(f"Failed to load: {result.symbols_failed}")
```

## Troubleshooting

### "No symbols found"

**Cause**: Directory structure doesn't match expected pattern.

**Fix**:
1. Check that files exist in the expected locations
2. Verify file extensions (`.parquet` or `.csv`)
3. Check directory permissions
4. Try explicit symbol list in config:
   ```yaml
   data:
     symbols: ["AAPL", "GOOGL", "MSFT"]
   ```

### "Missing required columns"

**Cause**: Timestamp column not found or wrong name.

**Fix**:
1. Verify column names (case-sensitive)
2. Rename to `timestamp`:
   ```python
   df = df.rename(columns={"your_time_col": "timestamp"})
   ```
3. Or configure the loader:
   ```yaml
   loader_options:
     date_column: your_time_col
   ```

### "Schema validation failed"

**Cause**: Data types or required columns don't match.

**Fix**:
1. Run the validation script above
2. Convert types:
   ```python
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   df['close'] = df['close'].astype(float)
   ```

### "File not found"

**Cause**: Path mismatch between config and actual location.

**Fix**:
1. Use absolute paths in config
2. Check for typos in symbol names
3. Verify interval matches directory structure

## Performance Tips

1. **Use Parquet** - 5-10x faster than CSV
2. **Enable Polars** - Set `USE_POLARS=1` environment variable
3. **Limit symbols** - Start with a subset for testing
4. **Use SSD** - I/O is often the bottleneck

## Next Steps

- [Adding Custom Features](./CUSTOM_FEATURES.md) - Register your feature columns
- [Writing Custom Data Loaders](./DATA_LOADER_PLUGINS.md) - SQL, API, or custom sources
