# Adding Custom Features

This guide explains how to register custom features with the FoxML pipeline to ensure proper leakage prevention and horizon filtering.

## Overview

The feature registry system:
- Tracks feature metadata (lookback, lag, allowed horizons)
- Prevents data leakage during training
- Enables automatic feature filtering per target horizon

## Why Register Features?

Without registration, features may:
1. **Leak future information** into predictions
2. **Be used for inappropriate horizons**
3. **Cause silent prediction errors** in production

Registration ensures your features are safe to use.

## Method 1: YAML Registration (Recommended)

### Register a Feature Family

Feature families match multiple features by pattern:

```yaml
# CONFIG/data/feature_registry.yaml

feature_families:
  # Match all features starting with "my_custom_"
  my_custom:
    pattern: ^my_custom_
    description: My custom indicator family
    default_lag_bars: 2
    default_allowed_horizons: [5, 10, 30, 60]
    default_lookback_minutes: 10

  # Match momentum features with different lookbacks
  my_momentum:
    pattern: ^my_momentum_(\d+)$
    description: Custom momentum indicators
    default_lag_bars: 0  # Computed at bar close
    default_allowed_horizons: [5, 10, 30, 60, 120]
```

### Register Individual Features

For specific features with custom settings:

```yaml
# CONFIG/data/feature_registry.yaml

features:
  my_custom_signal:
    source: derived
    lag_bars: 5
    lookback_minutes: 25  # 5 bars * 5 min
    allowed_horizons: [10, 30, 60]  # Exclude 5 (too close to lookback)
    description: Custom signal with 5-bar lookback

  my_relative_strength:
    source: cross_sectional
    lag_bars: 1
    lookback_minutes: 5
    allowed_horizons: [5, 10, 30, 60]
    description: Cross-sectional relative strength
```

## Method 2: Python Registration (Advanced)

For dynamic feature registration:

```python
from TRAINING.common.feature_registry import get_registry

registry = get_registry()

# Register single feature
registry.register_feature("my_custom_feature", {
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
    "default_lookback_minutes": 10,
    "default_allowed_horizons": [5, 10, 30, 60]
})
```

## Leakage Prevention Rules

### Understanding lag_bars

`lag_bars` specifies how many bars to wait before using a feature:

```
Bar:        0    1    2    3    4    5
Feature:    X    X    X    ?    ?    ?    (computed using bars 0-2)
            └────────────┘
               lookback

If lag_bars=0:  Feature available at bar 3
If lag_bars=2:  Feature available at bar 5 (safe margin)
```

### Understanding allowed_horizons

`allowed_horizons` lists which prediction horizons can use this feature:

```python
# Feature with 20-bar lookback should NOT predict 5-bar horizon
# because the feature "knows" most of the prediction period
my_feature:
  lookback_minutes: 100  # 20 bars at 5m
  allowed_horizons: [30, 60, 120]  # NOT 5, 10
```

### Rule of Thumb

```
horizon_minutes > lookback_minutes / 2
```

If your feature looks back 20 minutes, don't use it to predict 5-minute returns.

## Computing Custom Features

### Option A: Pre-compute in Data

The simplest approach - compute features before saving:

```python
import pandas as pd
import numpy as np

def compute_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom features to dataframe."""
    df = df.copy()

    # Momentum feature
    df['my_momentum_10'] = df['close'].pct_change(10)

    # Volatility feature
    df['my_volatility_20'] = df['close'].rolling(20).std()

    # Custom signal
    df['my_custom_signal'] = (
        df['close'].rolling(5).mean() /
        df['close'].rolling(20).mean() - 1
    )

    return df

# Apply to your data
df = compute_custom_features(df)
df.to_parquet("data/interval=5m/symbol=AAPL/AAPL.parquet")
```

### Option B: Feature Transform Config (Future)

```yaml
# CONFIG/data/feature_transforms.yaml
transforms:
  my_momentum_10:
    function: pandas.DataFrame.pct_change
    args:
      periods: 10
    input_column: close
```

## Testing Your Features

### Test 1: Leakage Detection

```python
from TRAINING.ranking.predictability.leakage_detection import detect_feature_leakage

# Load your data
df = pd.read_parquet("your_data.parquet")

# Test feature for leakage
result = detect_feature_leakage(
    df=df,
    feature_column="my_custom_signal",
    target_column="fwd_ret_5m",
    target_horizon_minutes=5
)

if result.has_leakage:
    print(f"WARNING: Feature has leakage!")
    print(f"  Reason: {result.reason}")
    print(f"  Correlation: {result.correlation:.3f}")
else:
    print("Feature passed leakage test")
```

### Test 2: Registry Validation

```python
from TRAINING.common.feature_registry import get_registry

registry = get_registry()

# Check if feature is registered
metadata = registry.get_feature_metadata("my_custom_signal")
if metadata:
    print(f"Feature registered:")
    print(f"  Lag bars: {metadata.get('lag_bars')}")
    print(f"  Lookback: {metadata.get('lookback_minutes')} minutes")
    print(f"  Allowed horizons: {metadata.get('allowed_horizons')}")
else:
    print("Feature not registered - will use defaults")

# Check if allowed for specific horizon
is_allowed = registry.is_feature_allowed("my_custom_signal", target_horizon=5)
print(f"Allowed for 5-min horizon: {is_allowed}")
```

### Test 3: Full Validation Suite

```python
def validate_custom_features(df: pd.DataFrame, feature_cols: list):
    """Validate all custom features."""
    registry = get_registry()
    issues = []

    for feature in feature_cols:
        # Check registration
        meta = registry.get_feature_metadata(feature)
        if not meta:
            issues.append(f"{feature}: Not registered")
            continue

        # Check for NaN
        nan_pct = df[feature].isna().mean() * 100
        if nan_pct > 10:
            issues.append(f"{feature}: {nan_pct:.1f}% NaN values")

        # Check variance
        if df[feature].std() < 1e-10:
            issues.append(f"{feature}: Zero variance (constant)")

    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All features validated successfully")

# Usage
validate_custom_features(df, ["my_momentum_10", "my_custom_signal"])
```

## Common Feature Patterns

### Momentum Features

```yaml
my_momentum_5:
  source: derived
  lag_bars: 5
  lookback_minutes: 25
  allowed_horizons: [10, 30, 60]
  description: 5-bar momentum

my_momentum_20:
  source: derived
  lag_bars: 20
  lookback_minutes: 100
  allowed_horizons: [30, 60, 120]
  description: 20-bar momentum
```

### Rolling Statistics

```yaml
my_rolling_std_10:
  source: derived
  lag_bars: 10
  lookback_minutes: 50
  allowed_horizons: [15, 30, 60]
  description: 10-bar rolling volatility

my_rolling_mean_20:
  source: derived
  lag_bars: 20
  lookback_minutes: 100
  allowed_horizons: [30, 60, 120]
  description: 20-bar moving average deviation
```

### Cross-Sectional Features

Features computed across symbols at each timestamp:

```yaml
my_rank_momentum:
  source: cross_sectional
  lag_bars: 1  # Computed at bar close
  lookback_minutes: 5
  allowed_horizons: [5, 10, 30, 60]
  description: Cross-sectional momentum rank

my_sector_relative:
  source: cross_sectional
  lag_bars: 1
  lookback_minutes: 5
  allowed_horizons: [5, 10, 30, 60]
  description: Relative to sector average
```

### Lagged Features

Features that are intentionally delayed:

```yaml
my_signal_lag2:
  source: derived
  lag_bars: 2  # Explicit 2-bar lag
  lookback_minutes: 10
  allowed_horizons: [5, 10, 30, 60]
  description: Signal with 2-bar lag for safety margin
```

## Best Practices

1. **Always specify lag_bars** - Don't rely on defaults
2. **Use lookback_minutes** - Required for interval-agnostic features
3. **Be conservative with allowed_horizons** - Exclude borderline cases
4. **Test before training** - Run leakage detection
5. **Document features** - Future you will thank you
6. **Version your features** - `my_signal_v2` not just `my_signal`

## Troubleshooting

### "Feature not allowed for horizon"

Your feature's `allowed_horizons` doesn't include the target horizon.

**Fix**: Either add the horizon to `allowed_horizons` or use a different feature.

### "Feature has leakage"

The leakage detector found suspicious correlation.

**Fix**:
1. Increase `lag_bars`
2. Remove horizon from `allowed_horizons`
3. Review feature computation for future data usage

### "Feature not found in registry"

The feature name doesn't match any registered pattern.

**Fix**:
1. Register the feature explicitly
2. Or add a family pattern that matches

## Next Steps

- [Adding Custom Datasets](./CUSTOM_DATASETS.md) - Prepare your data
- [Writing Custom Data Loaders](./DATA_LOADER_PLUGINS.md) - Custom data sources
