# Feature Engineering

Skill for adding features safely without introducing data leakage.

## When to Use Features vs Raw OHLCV

| Approach | Use Case | Models |
|----------|----------|--------|
| **Engineered Features** (default) | Tree models, classic ML, interpretability needed | LightGBM, XGBoost, CatBoost, RandomForest |
| **Raw OHLCV Sequences** | Deep learning, letting model learn features | LSTM, Transformer, CNN1D |

**RAW_SEQUENCE mode** skips feature engineering entirely - the pipeline feeds raw OHLCV bars directly to sequence-capable neural models. Configure via:

```yaml
# CONFIG/experiments/my_experiment.yaml
intelligent_training:
  input_mode: RAW_SEQUENCE  # Skip feature engineering
  sequence_length: 60
```

The rest of this skill applies to `FEATURES` mode (the default).

## Core Principle: No Lookahead Bias

Features MUST NOT contain information from the future relative to the prediction target. This is the most critical requirement in financial ML.

**Example of leakage:**
- Target: `fwd_ret_10m` (10-minute forward return)
- Feature: `price_close` without lag
- Problem: `price_close` at time T includes information that contributes to the 10-minute return

## Feature Registry System

### Registry Location

| File | Purpose |
|------|---------|
| `CONFIG/data/feature_registry.yaml` | Feature family patterns |
| `CONFIG/feature_registry.yaml` | Individual feature overrides |
| `CONFIG/data/excluded_features.yaml` | Globally excluded features |

### Feature Family Definitions

```yaml
# CONFIG/data/feature_registry.yaml
feature_families:
  lagged_returns:
    pattern: ^ret_\d+$
    description: Lagged returns (ret_N where N is lag)
    default_lag_bars: null  # Inferred from pattern
    default_allowed_horizons: [1, 2, 3, 5, 12, 24, 60]

  technical_indicators:
    pattern: ^(rsi|sma|ema|macd|bb|atr|adx)_\d+$
    description: Technical indicators with numeric lookback
    default_lag_bars: null
    default_allowed_horizons: [1, 2, 3, 5, 12, 24, 60]

  # REJECTED patterns - automatically excluded
  rejected_future_returns:
    pattern: ^(ret_future_|fwd_ret_)
    description: "REJECTED: Future returns (leaky)"
    default_lag_bars: -1  # Indicates future data
    default_allowed_horizons: []
    rejected: true
```

### Individual Feature Overrides

```yaml
# CONFIG/feature_registry.yaml
features:
  my_custom_feature:
    lag_bars: 3             # Minimum bars of lag
    allowed_horizons: [5, 10, 30, 60]  # Safe for these targets
    source: "custom"
    description: "Custom indicator with 3-bar lookback"
```

## Leakage Prevention Rules

### Rule 1: lag_bars Requirement

Every feature must have `lag_bars` defined (explicitly or via family default).

| lag_bars | Meaning |
|----------|---------|
| 0 | Uses current bar only (safe for all horizons) |
| N > 0 | Uses data up to N bars ago (safe for horizons > N bars) |
| -1 | Uses future data (always leaky, auto-rejected) |
| null | Inferred from pattern (e.g., `ret_5` → lag_bars=5) |

### Rule 2: allowed_horizons Constraint

Features are only used for targets where the horizon is in `allowed_horizons`.

```yaml
# Feature safe only for 30m+ horizons
my_feature:
  lag_bars: 6  # Uses 6 bars of history
  allowed_horizons: [30, 60, 120]  # Only for 30m, 1h, 2h targets
```

### Rule 3: Automatic Rejection Patterns

These patterns are auto-rejected (always leaky):

| Pattern | Why Leaky |
|---------|-----------|
| `^(fwd_ret_|ret_future_)` | Forward-looking returns |
| `^(y_|target_)` | Target columns |
| `^(mfe|mdd)_` | Maximum favorable/adverse excursion |
| `^barrier_` | Barrier features |
| `^tth_` | Time-to-hit features |
| `^p_` | Prediction probabilities |

## Adding a New Feature

### Step 1: Determine Leakage Profile

Questions to answer:
1. Does the feature use any future data? → If yes, cannot use
2. What's the minimum lookback period? → Sets `lag_bars`
3. Which target horizons is it safe for? → Sets `allowed_horizons`

### Step 2: Add to Feature Registry

```yaml
# CONFIG/feature_registry.yaml
features:
  my_new_feature:
    lag_bars: 2
    allowed_horizons: [5, 10, 30, 60]
    source: "my_module"
    description: "2-period momentum indicator"
```

### Step 3: Implement Feature Calculation

```python
# DATA_PROCESSING/features/my_feature.py
import numpy as np


def compute_my_feature(data: np.ndarray, lookback: int = 2) -> np.ndarray:
    """
    Compute my feature with proper lag.

    SST: Uses lookback bars of history, safe for horizons > lookback.
    """
    # Shift by lookback to avoid lookahead
    feature = np.empty_like(data)
    feature[:lookback] = np.nan  # Not enough history
    feature[lookback:] = data[lookback:] - data[:-lookback]

    return feature
```

### Step 4: Test for Leakage

```python
def test_feature_no_leakage():
    """Verify feature has no lookahead bias."""
    from TRAINING.ranking.predictability.leakage_detection import analyze_feature_leakage

    result = analyze_feature_leakage(
        feature_name="my_new_feature",
        target_horizon_minutes=10,
        feature_registry=load_registry(),
    )

    assert not result['leakage_detected'], f"Leakage: {result['reason']}"
```

## Feature Selection Process

### Multi-Model Consensus

Feature selection uses multiple models to avoid overfitting:

```
Feature Importance Sources:
├── LightGBM feature_importances_
├── XGBoost feature_importances_
├── Ridge coefficients
└── Random Forest feature_importances_

Consensus: Features selected if important across multiple models
```

### Selection Pipeline

```
1. Load candidate features
2. Apply leakage filter (registry + runtime checks)
3. Train importance models
4. Extract importances from each model
5. Apply consensus rule (e.g., top K from majority)
6. Output selected feature set
```

### Key Files

| File | Purpose |
|------|---------|
| `TRAINING/ranking/feature_selector.py` | Main selection logic |
| `TRAINING/ranking/multi_model_feature_selection.py` | Multi-model consensus |
| `TRAINING/ranking/multi_model_feature_selection/importance_extractors.py` | Per-model extraction |

## Testing Features for Leakage

### Automated Leakage Detection

```python
from TRAINING.ranking.predictability.leakage_detection import run_leakage_detection

results = run_leakage_detection(
    features=["feat_a", "feat_b", "feat_c"],
    target="fwd_ret_10m",
    feature_registry=registry,
)

for feature, status in results.items():
    if status['leakage_detected']:
        print(f"LEAKY: {feature} - {status['reason']}")
```

### Manual Leakage Check

```python
def check_no_future_correlation(feature_series, target_series, max_lag=10):
    """
    Feature at time T should not correlate with target at T+1, T+2, etc.
    High future correlation indicates leakage.
    """
    for lag in range(1, max_lag + 1):
        future_target = target_series.shift(-lag)
        corr = feature_series.corr(future_target)
        if abs(corr) > 0.3:  # Threshold
            print(f"WARNING: Correlation {corr:.2f} with target at T+{lag}")
```

## SST Compliance

### Feature Registry Access

```python
from CONFIG.config_loader import get_config_path
import yaml

# Load registry via SST path
registry_path = get_config_path("feature_registry")
with open(registry_path) as f:
    registry = yaml.safe_load(f)
```

### Feature Metadata Access

```python
from TRAINING.common.feature_registry import get_feature_metadata

# Get lag_bars and allowed_horizons for a feature
metadata = get_feature_metadata("my_feature")
print(f"lag_bars: {metadata['lag_bars']}")
print(f"allowed_horizons: {metadata['allowed_horizons']}")
```

## Common Mistakes

| Mistake | Why It's Leaky | Fix |
|---------|----------------|-----|
| Using `close` without shift | Contains future info | Add `lag_bars: 1` or shift in calculation |
| Normalizing with full-sample stats | Future stats leak | Use rolling normalization |
| Forward-fill then feature | Future values propagate | Feature before forward-fill |
| Joining data on timestamp | May align future data | Verify temporal alignment |
| Using `pct_change()` without lag | Includes current bar | Use `pct_change().shift(1)` |

## Determinism in Feature Engineering

### Ordered Operations

```python
from TRAINING.common.utils.determinism_ordering import sorted_items

# WRONG: Feature order varies
for feature, meta in registry.items():  # ❌
    process(feature)

# CORRECT: Deterministic order
for feature, meta in sorted_items(registry):  # ✅
    process(feature)
```

### Reproducible Feature Computation

```python
# Use deterministic seed for any randomness
from TRAINING.common.determinism import BASE_SEED

rng = np.random.default_rng(BASE_SEED or 42)
noise = rng.normal(0, 0.01, len(data))  # Reproducible noise
```

## Memory-Efficient Patterns (SST)

When working with large feature matrices, use SST helpers:

```python
from TRAINING.common.utils.memory_utils import streaming_concat

# WRONG: Memory spike with pd.concat on large list
big_df = pd.concat(list_of_dfs)  # ❌ 2x memory

# CORRECT: Streaming concatenation (constant memory)
big_df = streaming_concat(list_of_dfs, chunk_size=10)  # ✅
```

For Polars lazy frames:
```python
import polars as pl

# Prefer lazy operations, collect at the end
lf = pl.scan_parquet("data/*.parquet")
result = (
    lf.filter(pl.col("symbol") == symbol)
    .select(feature_cols)
    .collect()  # Only materialize at the end
)
```

**See `sst-and-coding-standards.md`** for full memory efficiency patterns.

## Skill Updates

This skill should be updated when:
- New leakage patterns are discovered
- Feature registry schema changes
- New feature families are added
- Selection consensus rules change
- New anti-patterns are identified

## Related Documentation

- `CONFIG/data/feature_registry.yaml` - Feature family patterns
- `CONFIG/feature_registry.yaml` - Individual overrides
- `TRAINING/ranking/predictability/leakage_detection.py` - Detection logic
- `TRAINING/ranking/feature_selector.py` - Selection pipeline
- `DATA_PROCESSING/features/` - Feature implementations
