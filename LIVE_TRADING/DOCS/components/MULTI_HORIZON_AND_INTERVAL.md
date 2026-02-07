# Multi-Horizon and Multi-Interval Handling

This document describes how the LIVE_TRADING module handles multi-horizon predictions and interval mapping.

---

## Overview

The LIVE_TRADING module supports:

1. **Multi-Horizon Predictions**: Generating predictions for multiple time horizons (5m, 10m, 15m, 30m, 60m, 1d)
2. **Cross-Horizon Blending**: Combining predictions across horizons using ridge risk-parity
3. **Horizon Arbitration**: Selecting the optimal horizon based on cost-adjusted returns
4. **Interval Mapping**: Mapping between data intervals and prediction horizons

This design is based on the plans:
- `.claude/plans/multi-horizon-training-master.md`
- `.claude/plans/phase8-multi-horizon-training.md`
- `.claude/plans/phase9-cross-horizon-ensemble.md`
- `.claude/plans/phase10-multi-interval-experiments.md`

---

## Multi-Horizon Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HORIZON PREDICTION FLOW                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Market Data                                                             │
│      │                                                                   │
│      ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  PER-HORIZON PREDICTIONS                         │    │
│  │                                                                  │    │
│  │   5m:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_5m            │    │
│  │  10m:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_10m           │    │
│  │  15m:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_15m           │    │
│  │  30m:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_30m           │    │
│  │  60m:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_60m           │    │
│  │   1d:  [LightGBM] [XGBoost] [Ridge] [MLP] ... → α_1d            │    │
│  │                                                                  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RIDGE RISK-PARITY BLENDING                    │    │
│  │                                                                  │    │
│  │   Per horizon: w ∝ (Σ + λI)^{-1} μ                              │    │
│  │   Temperature: w^(T) for short horizons                          │    │
│  │                                                                  │    │
│  │   Result: blended_alpha[horizon] for each horizon                │    │
│  │                                                                  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    HORIZON ARBITRATION                           │    │
│  │                                                                  │    │
│  │   Cost Model: cost = k₁×spread + k₂×σ×√(h/5) + k₃×impact        │    │
│  │   Net Score: net_h = α_h - cost_h                                │    │
│  │   Selection: best = argmax(net_h / √(h/5))                       │    │
│  │                                                                  │    │
│  │   Result: selected_horizon, net_score, decision                  │    │
│  │                                                                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Horizons

### Standard Horizons

The LIVE_TRADING module supports these prediction horizons:

| Horizon | Minutes | Typical Use | Temperature |
|---------|---------|-------------|-------------|
| `5m` | 5 | Intraday scalping | T = 0.75 |
| `10m` | 10 | Intraday momentum | T = 0.85 |
| `15m` | 15 | Intraday swing | T = 0.90 |
| `30m` | 30 | Intraday position | T = 1.0 |
| `60m` | 60 | Intraday/overnight | T = 1.0 |
| `1d` | 1440 | Swing trading | T = 1.0 |

### Horizon Configuration

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  horizons:
    - "5m"
    - "10m"
    - "15m"
    - "30m"
    - "60m"
    - "1d"

  # Temperature compression for short horizons
  temperature:
    "5m": 0.75
    "10m": 0.85
    "15m": 0.90
    "30m": 1.0
    "60m": 1.0
    "1d": 1.0
```

---

## MultiHorizonPredictor

**Location**: `LIVE_TRADING/prediction/predictor.py`

Orchestrates predictions across all horizons and model families.

### Usage

```python
from LIVE_TRADING.prediction import MultiHorizonPredictor

# Initialize with path to TRAINING run artifacts
predictor = MultiHorizonPredictor(
    run_root="/path/to/RESULTS/runs/my_run",
    horizons=["5m", "10m", "15m", "30m", "60m"],  # Optional, uses config defaults
    families=["lightgbm", "xgboost", "ridge"],    # Optional, uses all available
    device="cpu",
)

# Generate predictions for all horizons
result = predictor.predict_all_horizons(
    target="ret_5m",              # Target name from training
    prices=ohlcv_df,              # OHLCV DataFrame
    symbol="AAPL",                # Trading symbol
    data_timestamp=datetime.now(timezone.utc),  # Optional
    adv=1_000_000,                # Average daily volume
    planned_dollars=10_000,       # Planned trade size
)

# Access predictions
print(result.get_horizon("15m").mean_calibrated)  # Blended calibrated score
print(result.get_best_horizon())                   # Horizon with strongest signal

# Access individual model predictions
hp = result.get_horizon("15m")
print(hp.predictions["lightgbm"].calibrated)  # Calibrated prediction
print(hp.predictions["lightgbm"].confidence)  # Confidence components
```

### Output Structure

```python
@dataclass
class ModelPrediction:
    """Single model prediction with metadata."""
    family: str
    horizon: str
    raw: float          # Raw model output
    standardized: float # Z-score standardized
    confidence: ConfidenceComponents  # IC, freshness, capacity, stability
    calibrated: float   # standardized × confidence.overall

@dataclass
class HorizonPredictions:
    """Predictions for a single horizon."""
    horizon: str
    timestamp: datetime
    predictions: Dict[str, ModelPrediction]  # family -> prediction

    @property
    def mean_calibrated(self) -> float: ...
    @property
    def mean_standardized(self) -> float: ...
    def get_calibrated_dict(self) -> Dict[str, float]: ...

@dataclass
class AllPredictions:
    """Predictions across all horizons."""
    symbol: str
    timestamp: datetime
    horizons: Dict[str, HorizonPredictions]  # horizon -> predictions

    def get_horizon(self, horizon: str) -> Optional[HorizonPredictions]: ...
    def get_best_horizon(self) -> Optional[str]: ...
```

---

## Ridge Risk-Parity Blending

### HorizonBlender

**Location**: `LIVE_TRADING/blending/horizon_blender.py`

Combines predictions from multiple model families within each horizon.

```python
from LIVE_TRADING.blending import HorizonBlender

# Initialize with optional ridge lambda and history size
blender = HorizonBlender(
    ridge_lambda=0.15,            # Optional, uses config default
    prediction_history_size=50,   # History for correlation estimation
)

# Blend predictions for a single horizon
# Takes HorizonPredictions from MultiHorizonPredictor
blend_result = blender.blend(
    horizon_predictions,  # HorizonPredictions object
    ic_values={"lightgbm": 0.05, "xgboost": 0.04},  # Optional
    cost_shares={"lightgbm": 0.01},                 # Optional
)

# Access blended result
print(blend_result.alpha)        # Blended alpha (weighted prediction)
print(blend_result.weights)      # Dict of weights used
print(blend_result.confidence)   # Aggregate confidence
print(blend_result.temperature)  # Temperature applied

# Blend all horizons at once
all_blended = blender.blend_all_horizons(all_predictions.horizons)
```

### BlendedAlpha Output

```python
@dataclass
class BlendedAlpha:
    horizon: str
    alpha: float              # Blended prediction
    weights: Dict[str, float] # Model weights used
    temperature: float        # Temperature applied
    confidence: float         # Aggregate confidence
```

### RidgeWeightCalculator

**Location**: `LIVE_TRADING/blending/ridge_weights.py`

Calculates optimal weights using ridge regression.

```python
from LIVE_TRADING.blending import RidgeWeightCalculator

calculator = RidgeWeightCalculator(ridge_lambda=0.15)

# Calculate weights
weights = calculator.calculate(
    predictions={"lightgbm": 0.5, "xgboost": 0.3, "ridge": 0.2},
    target_ics={"lightgbm": 0.05, "xgboost": 0.04, "ridge": 0.03},
    correlation_matrix=corr_matrix,
)
# Result: {"lightgbm": 0.45, "xgboost": 0.35, "ridge": 0.20}
```

### Ridge Formula

```
w ∝ (Σ + λI)^{-1} μ

Where:
- Σ = correlation matrix of predictions across model families
- λ = ridge regularization parameter (0.15 default)
- μ = vector of net ICs (information coefficients after costs)
- I = identity matrix

Post-processing:
1. Clip to non-negative: w ← max(w, 0)
2. Normalize: w ← w / sum(w)
```

### TemperatureCompressor

**Location**: `LIVE_TRADING/blending/temperature.py`

Applies temperature compression to weights for short horizons.

```python
from LIVE_TRADING.blending import TemperatureCompressor

compressor = TemperatureCompressor(temperatures={
    "5m": 0.75,
    "10m": 0.85,
    "15m": 0.90,
    "30m": 1.0,
})

# Compress weights
compressed = compressor.compress(weights, horizon="5m")
```

### Temperature Formula

```
w^{(T)} ∝ w^{1/T}

Effect:
- T < 1: Compresses weights toward uniform (more conservative)
- T = 1: No compression (original weights)
- T > 1: Sharpens weights (more aggressive)

Example (T = 0.75):
- Original: [0.5, 0.3, 0.2]
- Compressed: [0.44, 0.32, 0.24]  (more uniform)
```

---

## Horizon Arbitration

### HorizonArbiter

**Location**: `LIVE_TRADING/arbitration/horizon_arbiter.py`

Selects the optimal horizon to trade based on cost-adjusted returns.

```python
from LIVE_TRADING.arbitration import HorizonArbiter

arbiter = HorizonArbiter(
    cost_model=cost_model,
    horizon_penalty_enabled=True,
)

# Select best horizon
result = arbiter.select(
    blended_alphas={"5m": 0.8, "10m": 0.6, "15m": 0.5, "30m": 0.3},
    market_data=market_data,
)

print(result.selected_horizon)  # "5m"
print(result.net_score)          # 0.65
print(result.decision)           # "TRADE" or "HOLD"
```

### CostModel

**Location**: `LIVE_TRADING/arbitration/cost_model.py`

Calculates trading costs for each horizon.

```python
from LIVE_TRADING.arbitration import CostModel

cost_model = CostModel(k1=1.0, k2=0.15, k3=1.0)

# Calculate cost
cost = cost_model.calculate(
    horizon_minutes=15,
    spread_bps=5,
    volatility=0.02,
    order_size=1000,
    adv=100000,
)
# Returns TradingCosts(spread_cost=0.05, vol_cost=0.026, impact_cost=0.10)
```

### Cost Formula

```
cost = k₁ × spread_bps + k₂ × σ × √(h/5) + k₃ × √(q/ADV)

Where:
- k₁ = 1.0 (spread coefficient)
- k₂ = 0.15 (volatility timing coefficient)
- k₃ = 1.0 (market impact coefficient)
- spread_bps = bid-ask spread in basis points
- σ = volatility
- h = horizon in minutes
- q = order size
- ADV = average daily volume
```

### Horizon Selection

```
net_h = α_h - cost_h
score_h = net_h / √(h/5)

selected = argmax(score_h)

Trade if: score_{selected} ≥ θ_enter
```

---

## Cross-Horizon Ensemble (Phase 9)

The LIVE_TRADING module implements cross-horizon ensembling as designed in `.claude/plans/phase9-cross-horizon-ensemble.md`.

### CrossHorizonStacker

Blends predictions from different horizons for improved final predictions.

```python
# Architecture
pred_5m  ──┐
pred_15m ──┼──► Ridge Meta-Learner ──► final_pred
pred_60m ──┘
```

### Decay Functions

Weight nearer horizons higher when blending:

```python
# Exponential decay
weight = exp(-ln(2) × distance / half_life)

# Linear decay
weight = max(0, 1 - distance / max_distance)

# Inverse decay
weight = 1 / (distance + epsilon)
```

---

## Multi-Interval Mapping (Phase 10)

Based on `.claude/plans/phase10-multi-interval-experiments.md`.

### Interval Conversion

```python
from TRAINING.common.interval_helpers import minutes_to_bars, horizon_minutes_to_bars

# Feature lookback (use ceil for conservative estimate)
bars = minutes_to_bars(60, interval="5m", rounding="ceil")  # 12 bars

# Horizon validation (use round for exact match)
bars = horizon_minutes_to_bars(15, interval="5m")  # 3 bars (exact)
bars = horizon_minutes_to_bars(17, interval="5m")  # None (not exact)
```

### Interval Specification

```python
from TRAINING.common.interval_helpers import get_interval_spec

# Single source of truth for interval
spec = get_interval_spec(
    data=df,                    # Priority 1: infer from data
    config_path="experiment",   # Priority 2: from config
    default=5                   # Priority 3: default
)
```

---

## Configuration Reference

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  # Horizons to trade
  horizons:
    - "5m"
    - "10m"
    - "15m"
    - "30m"
    - "60m"
    - "1d"

  # Blending parameters
  blending:
    ridge_lambda: 0.15
    temperature:
      "5m": 0.75
      "10m": 0.85
      "15m": 0.90
      "30m": 1.0
      "60m": 1.0
      "1d": 1.0

  # Cost model parameters
  cost_model:
    k1: 1.0   # Spread penalty
    k2: 0.15  # Volatility timing
    k3: 1.0   # Market impact

  # Arbitration
  arbitration:
    horizon_penalty: true      # Apply √(h/5) penalty
    min_net_score: 0.001       # Minimum net score to trade
    reserve_bps: 2.0           # Additional buffer

  # Cross-horizon ensemble (optional)
  cross_horizon:
    enabled: false
    decay_function: "exponential"
    decay_half_life: 30  # minutes
    ridge_alpha: 1.0
```

---

## Performance Metrics

### Per-Horizon Metrics

```python
# Track per-horizon performance
metrics = {
    "5m": {"ic": 0.05, "sharpe": 1.2, "trades": 150},
    "10m": {"ic": 0.04, "sharpe": 1.0, "trades": 100},
    "15m": {"ic": 0.035, "sharpe": 0.9, "trades": 75},
    ...
}
```

### Horizon Selection Distribution

Track which horizons are being selected:

```python
selection_counts = {
    "5m": 450,   # 45%
    "10m": 300,  # 30%
    "15m": 150,  # 15%
    "30m": 75,   # 7.5%
    "60m": 25,   # 2.5%
}
```

---

## Testing

```bash
# Run multi-horizon tests
pytest LIVE_TRADING/tests/test_prediction.py -v    # 34 tests
pytest LIVE_TRADING/tests/test_blending.py -v      # 29 tests
pytest LIVE_TRADING/tests/test_arbitration.py -v   # 25 tests

# Run all pipeline tests
pytest LIVE_TRADING/tests/ -k "horizon" -v
```

---

## Related Documentation

- [MODEL_INFERENCE.md](MODEL_INFERENCE.md) - Model loading and inference
- [ONLINE_LEARNING.md](ONLINE_LEARNING.md) - Adaptive weight learning
- [../architecture/PIPELINE_STAGES.md](../architecture/PIPELINE_STAGES.md) - Pipeline stage details
- [../reference/PLAN_REFERENCES.md](../reference/PLAN_REFERENCES.md) - Plan documents
