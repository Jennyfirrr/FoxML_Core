# Interval-Agnostic Metric Scaling

**Phase 23 Documentation** - How metrics scale across different data intervals.

When training models at different data intervals (1m, 5m, 15m, 60m), certain metrics
require interval-aware scaling to ensure comparability. This document provides the
scaling rules and best practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Volatility Scaling](#volatility-scaling)
3. [Sharpe Ratio Annualization](#sharpe-ratio-annualization)
4. [Return Metrics](#return-metrics)
5. [Hit Rate and IC](#hit-rate-and-ic)
6. [Purge and Embargo Scaling](#purge-and-embargo-scaling)
7. [Best Practices](#best-practices)

---

## Overview

### The Problem

When comparing model performance across different data intervals, raw metrics
can be misleading:

| Metric | At 1m | At 5m | At 60m | Issue |
|--------|-------|-------|--------|-------|
| Raw volatility | 0.02% | 0.05% | 0.15% | Not comparable |
| Raw Sharpe | 0.8 | 1.2 | 2.5 | Time-scaling artifact |
| Hit rate | 51% | 52% | 54% | Comparable (no scaling needed) |
| IC | 0.02 | 0.03 | 0.05 | Comparable (no scaling needed) |

### Core Principle

**Time-normalized metrics** allow cross-interval comparison:

```
Annualized metrics use: sqrt(periods_per_year)
Daily metrics use: sqrt(periods_per_day)
```

---

## Volatility Scaling

### Statistical Foundation

Volatility scales with the square root of time (under i.i.d. returns assumption):

```
σ_T = σ_1 × sqrt(T)
```

Where:
- `σ_1` = per-period volatility
- `T` = number of periods
- `σ_T` = volatility over T periods

### Interval-Specific Annualization Factors

| Interval | Periods/Day | Periods/Year | Annualization Factor |
|----------|-------------|--------------|----------------------|
| 1m       | 390         | 97,500       | sqrt(97,500) ≈ 312.2 |
| 5m       | 78          | 19,500       | sqrt(19,500) ≈ 139.6 |
| 15m      | 26          | 6,500        | sqrt(6,500) ≈ 80.6 |
| 60m      | 6.5         | 1,625        | sqrt(1,625) ≈ 40.3 |
| Daily    | 1           | 250          | sqrt(250) ≈ 15.8 |

**Note**: Assumes 390 trading minutes per day and 250 trading days per year.

### Code Example

```python
import math

def annualize_volatility(
    per_bar_vol: float,
    interval_minutes: int,
    trading_minutes_per_day: int = 390,
    trading_days_per_year: int = 250
) -> float:
    """
    Annualize per-bar volatility.

    Args:
        per_bar_vol: Volatility per bar (standard deviation of returns)
        interval_minutes: Bar interval in minutes
        trading_minutes_per_day: Trading minutes per day (default: 390)
        trading_days_per_year: Trading days per year (default: 250)

    Returns:
        Annualized volatility
    """
    bars_per_day = trading_minutes_per_day / interval_minutes
    bars_per_year = bars_per_day * trading_days_per_year
    return per_bar_vol * math.sqrt(bars_per_year)


# Example: 5m volatility of 0.05% → annualized
annualized = annualize_volatility(0.0005, interval_minutes=5)
# Result: ~7% annualized volatility
```

### Volatility Comparison Table

To compare volatility across intervals, normalize to the same time scale:

```python
def normalize_volatility(
    vol: float,
    from_interval: int,
    to_interval: int
) -> float:
    """Convert volatility from one interval to another."""
    return vol * math.sqrt(to_interval / from_interval)


# 1m vol of 0.02% converted to 5m equivalent:
vol_5m = normalize_volatility(0.0002, from_interval=1, to_interval=5)
# Result: 0.045% (approximately 0.02% × sqrt(5))
```

---

## Sharpe Ratio Annualization

### The Problem

Raw Sharpe ratios are **not comparable** across intervals because:

```
Sharpe = E[R] / σ(R)
```

Both expected return and volatility scale with time, but at different rates.

### Annualization Formula

```
Sharpe_annual = Sharpe_per_bar × sqrt(bars_per_year)
```

### Interval-Specific Scaling

| Interval | Bars/Year | Sharpe Multiplier |
|----------|-----------|-------------------|
| 1m       | 97,500    | 312.2 |
| 5m       | 19,500    | 139.6 |
| 15m      | 6,500     | 80.6 |
| 60m      | 1,625     | 40.3 |
| Daily    | 250       | 15.8 |

### Code Example

```python
def annualize_sharpe(
    sharpe_per_bar: float,
    interval_minutes: int,
    trading_minutes_per_day: int = 390,
    trading_days_per_year: int = 250
) -> float:
    """
    Annualize Sharpe ratio.

    Args:
        sharpe_per_bar: Sharpe ratio calculated per bar
        interval_minutes: Bar interval in minutes

    Returns:
        Annualized Sharpe ratio
    """
    bars_per_day = trading_minutes_per_day / interval_minutes
    bars_per_year = bars_per_day * trading_days_per_year
    return sharpe_per_bar * math.sqrt(bars_per_year)


# Example: 5m Sharpe of 0.02 → annualized
annualized_sharpe = annualize_sharpe(0.02, interval_minutes=5)
# Result: ~2.79 annualized Sharpe
```

### Warning: Overstated Sharpe at Small Intervals

Small intervals can produce **misleadingly high** annualized Sharpe ratios due to:

1. **Autocorrelation**: Returns at small intervals may be autocorrelated
2. **Transaction costs**: Not factored into raw Sharpe
3. **Market microstructure**: Bid-ask bounce, quote staleness

**Recommendation**: Always report Sharpe with the interval used and be skeptical
of annualized Sharpe > 3.0 from high-frequency intervals.

---

## Return Metrics

### Expected Return Scaling

Expected returns compound over time:

```
E[R_T] ≈ E[R_1] × T  (for small returns)
E[R_T] = (1 + E[R_1])^T - 1  (exact)
```

### Annualized Return

```python
def annualize_return(
    per_bar_return: float,
    interval_minutes: int,
    compound: bool = True,
    trading_minutes_per_day: int = 390,
    trading_days_per_year: int = 250
) -> float:
    """
    Annualize per-bar return.

    Args:
        per_bar_return: Return per bar (e.g., 0.0001 = 1 basis point)
        interval_minutes: Bar interval in minutes
        compound: Whether to use compound return (recommended)

    Returns:
        Annualized return
    """
    bars_per_day = trading_minutes_per_day / interval_minutes
    bars_per_year = bars_per_day * trading_days_per_year

    if compound:
        return (1 + per_bar_return) ** bars_per_year - 1
    else:
        return per_bar_return * bars_per_year


# Example: 5m return of 0.5 bps → annualized
annualized_ret = annualize_return(0.00005, interval_minutes=5)
# Result: ~170% annualized return (but unrealistic due to costs)
```

---

## Hit Rate and IC

### Hit Rate (No Scaling Needed)

Hit rate (directional accuracy) is **interval-agnostic**:

```
Hit Rate = P(sign(prediction) = sign(actual_return))
```

A 52% hit rate at 1m is directly comparable to 52% at 5m.

**However**, the *economic value* of a hit differs by interval:
- 1m: More opportunities but smaller moves
- 60m: Fewer opportunities but larger moves

### Information Coefficient (IC)

IC (correlation between prediction and outcome) is also **interval-agnostic**:

```
IC = corr(prediction, actual_return)
```

A 0.03 IC at 1m is directly comparable to 0.03 IC at 5m.

**Important**: IC decay over time matters:
- Models often show higher IC at short horizons
- IC may decay as prediction horizon extends

---

## Purge and Embargo Scaling

### Time-Based Purge (Interval-Agnostic)

Purge windows should be specified in **minutes**, not bars:

```python
from TRAINING.ranking.utils.purge import make_purge_spec

# Define purge in minutes (works at any interval)
spec = make_purge_spec(
    target_horizon_minutes=60,  # 60-minute prediction horizon
    buffer_minutes=5            # 5-minute safety buffer
)

# Convert to bars at runtime
bars_at_5m = spec.purge_bars(interval_minutes=5)   # 13 bars
bars_at_1m = spec.purge_bars(interval_minutes=1)   # 65 bars
bars_at_15m = spec.purge_bars(interval_minutes=15) # 5 bars
```

### Why Time-Based Purge Matters

At different intervals, the same bar count represents different time:

| Purge Bars | At 1m | At 5m | At 60m |
|------------|-------|-------|--------|
| 17 bars    | 17 min | 85 min | 1020 min |

**Problem**: Using `purge_overlap=17` (the 5m default) at 1m gives only 17 minutes
of purge, which is insufficient for a 60-minute target horizon → **data leakage**.

**Solution**: Always specify purge in minutes:

```python
# WRONG: Bar-based (leaks at different intervals)
cv = PurgedTimeSeriesSplit(purge_overlap=17)

# RIGHT: Time-based (safe at all intervals)
cv = PurgedTimeSeriesSplit(purge_overlap_minutes=85.0)
```

---

## Best Practices

### 1. Always Specify Units

```python
# Bad: Ambiguous
lookback = 14

# Good: Explicit units
lookback_minutes = 70
lookback_bars = minutes_to_bars(70, interval_minutes)
```

### 2. Report Metrics with Interval Context

When reporting metrics, always include the data interval:

```markdown
## Model Performance (5-minute bars)

| Metric | Value |
|--------|-------|
| Hit Rate | 52.3% |
| IC | 0.032 |
| Annualized Sharpe | 2.1 |
| Per-bar Sharpe | 0.015 |
```

### 3. Use Time-Based Configuration

```yaml
# CONFIG/experiments/multi_interval.yaml

# Define everything in time units
leakage:
  purge_minutes: 85
  embargo_minutes: 10

features:
  max_lookback_minutes: 1440  # 1 day

# Interval is detected from data, not hardcoded
interval_detection:
  source: auto
  fallback_minutes: 5
```

### 4. Compare Apples to Apples

When comparing models across intervals:

1. **Annualize** volatility and Sharpe for comparison
2. **Don't annualize** hit rate and IC (already comparable)
3. **Factor in costs** - smaller intervals have more trades
4. **Consider capacity** - smaller intervals have less capacity

### 5. Be Skeptical of High-Frequency Sharpe

If your 1-minute model shows Sharpe > 5:

1. Check for lookahead bias (timestamps off by even 1 bar matter at 1m)
2. Verify purge window is sufficient
3. Add realistic transaction costs
4. Consider market microstructure effects

---

## Summary Table

| Metric | Scales with Interval? | Scaling Rule |
|--------|----------------------|--------------|
| Volatility | Yes | σ × sqrt(T) |
| Sharpe Ratio | Yes | SR × sqrt(bars_per_year) |
| Expected Return | Yes | Compound or linear |
| Hit Rate | No | Directly comparable |
| IC (correlation) | No | Directly comparable |
| Purge Window | Must use time | Specify in minutes |
| Lookback | Must use time | Specify in minutes |

---

## Related Documentation

- [Interval-Agnostic Pipeline Plan](../../../.claude/plans/interval_agnostic_pipeline.md)
- [Deterministic Runs](DETERMINISTIC_RUNS.md)
- [Leakage and Safety Configs](SAFETY_LEAKAGE_CONFIGS.md)
- [Feature Registry](../../03_technical/data/FEATURE_REGISTRY_SCHEMA.md)
