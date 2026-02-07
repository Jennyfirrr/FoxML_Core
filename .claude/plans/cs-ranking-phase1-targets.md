# Phase 1: Cross-Sectional Target Construction

**Parent**: `cross-sectional-ranking-objective.md`
**Status**: Complete ✅
**Estimated Effort**: 1-2 hours
**Actual Effort**: ~1.5 hours

## Objective

Replace raw forward returns with cross-sectionally normalized targets that are:
1. Comparable across symbols (vol-adjusted)
2. Centered per timestamp (relative performance)
3. Robust to outliers (winsorized)

## Current State

```python
# Current target (raw return)
y = df['fwd_ret_5m']  # Ranges from -0.10 to +0.15 depending on symbol volatility

# Problem: TSLA return of 0.02 ≠ T return of 0.02 in "skill" terms
```

## Proposed Implementation

### Target Types

#### Option A: Cross-Sectional Percentile Rank (Recommended)

```python
def compute_cs_percentile_target(
    df: pd.DataFrame,
    return_col: str = 'fwd_ret_5m',
    time_col: str = 'ts',
    residualize: bool = True,
    winsorize_pct: tuple = (0.01, 0.99),
) -> pd.Series:
    """
    Compute cross-sectional percentile rank target.

    For each timestamp t:
    1. Get forward returns for all symbols
    2. Optionally residualize (subtract mean)
    3. Convert to percentile rank [0, 1]
    4. Winsorize extremes

    Args:
        df: DataFrame with columns [time_col, 'symbol', return_col]
        return_col: Forward return column name
        time_col: Timestamp column name
        residualize: If True, subtract cross-sectional mean first
        winsorize_pct: (lower, upper) percentile for clipping

    Returns:
        Series of percentile targets, same index as df
    """
    def _cs_percentile(group):
        r = group[return_col].values

        # Step 1: Residualize (optional)
        if residualize:
            r = r - np.nanmean(r)

        # Step 2: Rank to percentile
        # rankdata handles NaN by assigning NaN rank
        ranks = scipy.stats.rankdata(r, nan_policy='omit')
        n_valid = np.sum(~np.isnan(r))
        percentiles = ranks / (n_valid + 1)  # (0, 1) range

        # Step 3: Winsorize
        percentiles = np.clip(percentiles, winsorize_pct[0], winsorize_pct[1])

        return pd.Series(percentiles, index=group.index)

    return df.groupby(time_col, group_keys=False).apply(_cs_percentile)
```

#### Option B: Cross-Sectional Robust Z-Score

```python
def compute_cs_zscore_target(
    df: pd.DataFrame,
    return_col: str = 'fwd_ret_5m',
    time_col: str = 'ts',
    residualize: bool = True,
    winsorize_std: float = 3.0,
) -> pd.Series:
    """
    Compute cross-sectional robust z-score target.

    Uses median and MAD for robustness to outliers.

    z = (r - median(r)) / (MAD(r) + eps)
    """
    def _cs_zscore(group):
        r = group[return_col].values

        if residualize:
            r = r - np.nanmean(r)

        median = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - median))

        z = (r - median) / (mad + 1e-8)
        z = np.clip(z, -winsorize_std, winsorize_std)

        return pd.Series(z, index=group.index)

    return df.groupby(time_col, group_keys=False).apply(_cs_zscore)
```

#### Option C: Vol-Scaled + CS-Demeaned (User's Original Suggestion)

```python
def compute_vol_scaled_cs_target(
    df: pd.DataFrame,
    return_col: str = 'fwd_ret_5m',
    time_col: str = 'ts',
    vol_col: str = 'rolling_vol_20',  # Must be pre-computed
    winsorize_pct: tuple = (0.01, 0.99),
) -> pd.Series:
    """
    Compute vol-scaled, cross-section demeaned target.

    y = (r / vol) - mean_cs(r / vol)

    This is "how much did you beat the universe, per unit risk"
    """
    def _vol_scaled_cs(group):
        r = group[return_col].values
        vol = group[vol_col].values

        # Vol-scale
        r_scaled = r / (vol + 1e-8)

        # CS demean
        r_demeaned = r_scaled - np.nanmean(r_scaled)

        # Winsorize
        lower = np.nanpercentile(r_demeaned, winsorize_pct[0] * 100)
        upper = np.nanpercentile(r_demeaned, winsorize_pct[1] * 100)
        r_clipped = np.clip(r_demeaned, lower, upper)

        return pd.Series(r_clipped, index=group.index)

    return df.groupby(time_col, group_keys=False).apply(_vol_scaled_cs)
```

## Comparison

| Target Type | Pros | Cons | Use When |
|-------------|------|------|----------|
| **CS Percentile** | Bounded, robust, directly ordinal | Loses magnitude info | Default choice |
| **CS Z-Score** | Keeps magnitude, interpretable | Unbounded, needs clipping | Want signal strength |
| **Vol-Scaled CS** | Risk-adjusted, finance-standard | Needs vol estimate, more complex | Have good vol data |

## Implementation Location

```
TRAINING/
├── common/
│   └── targets/
│       ├── __init__.py
│       ├── cross_sectional.py    # NEW: CS target functions
│       └── validators.py         # NEW: No-leakage validation
```

## Data Leakage Prevention

**Critical**: All normalization must use only past/current data, never future.

```python
def validate_no_future_leakage(df, target_col, time_col):
    """
    Validate that target at time t doesn't use data from t+1.

    For CS targets computed per-timestamp, this is safe because
    we only use data from the SAME timestamp (cross-sectionally),
    not future timestamps.
    """
    # CS percentile/zscore within same t: SAFE
    # Rolling vol using past data only: SAFE
    # Any forward-looking computation: UNSAFE
    pass
```

## Config Integration

```yaml
# In experiment config
pipeline:
  cross_sectional_ranking:
    target:
      type: "cs_percentile"      # "cs_percentile", "cs_zscore", "vol_scaled"
      residualize: true
      winsorize: [0.01, 0.99]

      # For vol_scaled type only
      vol_lookback_bars: 20
      vol_method: "realized"     # "realized", "parkinson", "garman_klass"
```

## Testing Strategy

### Unit Tests

```python
def test_cs_percentile_sums_to_expected():
    """Percentiles should average to ~0.5 per timestamp."""
    ...

def test_cs_percentile_bounded():
    """All values in (0, 1) after winsorization."""
    ...

def test_residualization_zeros_mean():
    """After residualization, CS mean should be ~0."""
    ...

def test_no_future_leakage():
    """Target at t should not change if we modify data at t+1."""
    ...
```

### Integration Tests

```python
def test_cs_target_with_sequence_builder():
    """CS target integrates with build_sequences_from_ohlcv."""
    ...

def test_cs_target_determinism():
    """Same input → same targets."""
    ...
```

## Deliverables

1. [x] `TRAINING/common/targets/cross_sectional.py` with:
   - `compute_cs_percentile_target()`
   - `compute_cs_zscore_target()`
   - `compute_vol_scaled_cs_target()`
   - `compute_cs_target()` (unified interface)

2. [x] `TRAINING/common/targets/validators.py` with:
   - `validate_no_future_leakage()`
   - `validate_cs_target_quality()`
   - `validate_vol_column_no_leakage()`

3. [x] Unit tests in `tests/test_cs_targets.py` (38 tests passing)

4. [x] Config schema in `CONFIG/pipeline/ranking.yaml`

## Definition of Done

- [x] All three target types implemented
- [x] Unit tests passing (38/38)
- [x] No future data leakage (validated)
- [x] Deterministic (same input → same output)
- [x] Integrated with config system

## Session Log

### 2026-01-21: Implementation Complete
- Created `TRAINING/common/targets/` directory structure
- Implemented all three target functions with proper edge case handling
- Added unified `compute_cs_target()` interface for config-driven selection
- Added comprehensive validators for leakage and quality checking
- Written 38 unit tests covering:
  - Basic computation for all three target types
  - Bounded outputs and statistical properties
  - Edge cases (single timestamp, same returns, extreme values, empty data)
  - Determinism verification
  - Validation functions
- Created config schema in `CONFIG/pipeline/ranking.yaml`

**Next**: Phase 2 (Timestamp-Grouped Dataset) - see `cs-ranking-phase2-batching.md`
