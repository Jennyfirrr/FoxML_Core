# Pipeline Stages

This document provides detailed documentation of each stage in the 6-stage trading pipeline.

---

## Pipeline Overview

```
Market Data
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: PREDICTION (MultiHorizonPredictor)                             │
│                                                                          │
│   For each symbol:                                                       │
│   ├── Load models from TRAINING artifacts                                │
│   ├── Generate predictions for each horizon (5m, 10m, 15m, 30m, 60m, 1d)│
│   ├── Apply Z-score standardization per horizon                          │
│   └── Calculate confidence scores (IC × freshness × capacity × stability)│
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2: BLENDING (HorizonBlender)                                      │
│                                                                          │
│   For each horizon:                                                      │
│   ├── Calculate ridge risk-parity weights: w ∝ (Σ + λI)^{-1} μ         │
│   ├── Apply temperature compression: w^T (T<1 for short horizons)       │
│   └── Aggregate model predictions → single alpha per horizon            │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3: ARBITRATION (HorizonArbiter)                                   │
│                                                                          │
│   Cost model: cost = k₁×spread + k₂×σ×√(h/5) + k₃×impact                │
│   Net score: net_h = α_h - costs                                         │
│   Selection: Choose horizon with highest net_h / √(h/5)                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 4: GATING (BarrierGate + SpreadGate)                              │
│                                                                          │
│   Barrier Gate:                                                          │
│   ├── Block if P(peak) > 0.6 (prevent buying at tops)                   │
│   ├── Prefer if P(valley) > 0.55 (favor buying at bottoms)              │
│   └── Reduce size by (1 - P(peak))                                      │
│                                                                          │
│   Spread Gate:                                                           │
│   ├── Block if spread > 12 bps                                          │
│   └── Block if quote age > 200ms                                        │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 5: SIZING (PositionSizer)                                         │
│                                                                          │
│   ├── Volatility scaling: z = clip(α/σ, -z_max, z_max)                  │
│   ├── Apply no-trade band (80 bps) to prevent churning                  │
│   ├── Normalize to gross exposure target (50%)                          │
│   └── Calculate target shares from weights                               │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 6: RISK (RiskGuardrails)                                          │
│                                                                          │
│   Kill Switch Checks:                                                    │
│   ├── Daily loss limit: 2% max                                          │
│   ├── Maximum drawdown: 10% max                                         │
│   └── Position concentration: 20% max per symbol                        │
│                                                                          │
│   If any kill switch triggered → HOLD all positions                     │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Orders → Broker (PaperBroker / IBKR / Alpaca)
```

---

## Stage 1: Prediction

### Purpose
Generate predictions for each symbol across all prediction horizons using trained models.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `MultiHorizonPredictor` | `prediction/predictor.py` | Orchestrates multi-horizon predictions |
| `ZScoreStandardizer` | `prediction/standardization.py` | Rolling buffer Z-score normalization |
| `ConfidenceScorer` | `prediction/confidence.py` | Calculate prediction confidence |
| `ModelLoader` | `models/loader.py` | Load models from TRAINING artifacts |
| `InferenceEngine` | `models/inference.py` | Family-specific model inference |
| `FeatureBuilder` | `models/feature_builder.py` | Technical indicator calculation |

### Data Flow

```python
# Input to MultiHorizonPredictor.predict_all_horizons()
target: str          # e.g., "ret_5m", "fwd_ret_15m"
prices: pd.DataFrame # OHLCV data with columns: Open, High, Low, Close, Volume
symbol: str          # e.g., "AAPL"
data_timestamp: datetime  # For freshness calculation
adv: float           # Average daily volume
planned_dollars: float    # Planned trade size (for capacity)

# Processing
for horizon in self.horizons:
    for family in available_families:
        # Get feature list from model metadata
        builder = self._get_feature_builder(target, family)
        features = builder.build_features(prices, symbol)

        # Run inference (family-specific: tree, keras, sequential)
        raw_pred = self.engine.predict(target, family, features, symbol)

        # Standardize: s = clip((raw - μ) / σ, -3, 3)
        std_pred = self.standardizer.standardize(raw_pred, family, horizon)

        # Calculate confidence: IC × freshness × capacity × stability
        confidence = self.confidence_scorer.calculate_confidence(
            model=family, horizon=horizon, data_timestamp=data_timestamp,
            adv=adv, planned_dollars=planned_dollars,
        )

        # Calibrated: s̃ = s × confidence
        calibrated = self.confidence_scorer.apply_confidence(std_pred, confidence.overall)

# Output
AllPredictions:
    symbol: str
    timestamp: datetime
    horizons: Dict[horizon, HorizonPredictions]
        └── HorizonPredictions:
            horizon: str
            predictions: Dict[family, ModelPrediction]
                └── ModelPrediction:
                    family, horizon, raw, standardized, confidence, calibrated
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rolling_window_days` | 10 | Days of history for Z-score calculation |
| `z_clip` | 3.0 | Maximum absolute Z-score |
| `min_samples` | 100 | Minimum samples before standardization |

### Z-Score Standardization

```
s_{m,h} = clip((r̂_{m,h} - μ_{m,h}) / σ_{m,h}, -3, 3)

Where:
- s_{m,h} = standardized score for model m, horizon h
- r̂_{m,h} = raw prediction
- μ_{m,h} = rolling mean (10 trading days)
- σ_{m,h} = rolling standard deviation
```

### Confidence Scoring

```
confidence = IC × freshness × capacity × stability

Where:
- IC = Spearman correlation with actual returns
- freshness = e^{-Δt/τ_h} (decay for stale predictions)
- capacity = min(1, κ × ADV / planned_dollars)
- stability = 1 / rolling_RMSE
```

---

## Stage 2: Blending

### Purpose
Combine predictions from multiple model families into a single alpha per horizon using ridge risk-parity weighting.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `HorizonBlender` | `blending/horizon_blender.py` | Per-horizon blending orchestrator |
| `RidgeWeightCalculator` | `blending/ridge_weights.py` | Ridge regression weight calculation |
| `TemperatureCompressor` | `blending/temperature.py` | Weight compression for short horizons |

### Data Flow

```python
# Input
horizon_predictions: HorizonPredictions

# Processing
for horizon in HORIZONS:
    family_preds = horizon_predictions.predictions[horizon]

    # Calculate correlation matrix from history
    correlation_matrix = calculate_correlation(family_preds_history)

    # Ridge weights
    weights = ridge_weight_calculator.calculate(
        correlation_matrix=correlation_matrix,
        target_vector=family_ics,
        ridge_lambda=0.15
    )

    # Temperature compression for short horizons
    if horizon in ["5m", "10m"]:
        weights = temperature_compressor.compress(weights, T=0.75)

    # Blend
    alpha = sum(w * pred for w, pred in zip(weights, family_preds.values()))

# Output
BlendResult:
    alphas: Dict[horizon, float]
    weights: Dict[horizon, Dict[family, float]]
```

### Ridge Risk-Parity Formula

```
w_h ∝ (Σ_h + λI)^{-1} μ_h

Where:
- Σ_h = correlation matrix of standardized scores
- λ = ridge regularization (0.15 default)
- μ_h = target vector of net IC after costs
- I = identity matrix

Post-processing:
- w_h ← clip(w_h, 0, ∞)  # Non-negative
- ∑w_h = 1               # Normalize
```

### Temperature Compression

```
w_h^{(T)} ∝ w_h^{1/T}

Temperature by horizon:
- 5m:  T = 0.75 (most conservative)
- 10m: T = 0.85
- 15m: T = 0.90
- 30m+: T = 1.0 (no compression)
```

---

## Stage 3: Arbitration

### Purpose
Select the optimal horizon to trade after accounting for all trading costs.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `HorizonArbiter` | `arbitration/horizon_arbiter.py` | Cost-aware horizon selection |
| `CostModel` | `arbitration/cost_model.py` | Trading cost estimation |

### Data Flow

```python
# Input
blend_result: BlendResult
market_data: MarketSnapshot

# Processing
for horizon in HORIZONS:
    alpha = blend_result.alphas[horizon]

    # Calculate costs
    spread_cost = k1 * spread_bps
    vol_cost = k2 * volatility * sqrt(horizon_minutes / 5)
    impact_cost = k3 * sqrt(order_size / adv)

    total_cost = spread_cost + vol_cost + impact_cost

    # Net score
    net_score = alpha - total_cost

    # Horizon penalty (favor shorter horizons)
    score = net_score / sqrt(horizon_minutes / 5)

# Select best horizon
best_horizon = argmax(scores)

# Trade gate
if scores[best_horizon] < threshold:
    decision = HOLD

# Output
ArbitrationResult:
    selected_horizon: str
    net_score: float
    decision: "TRADE" | "HOLD"
    costs: TradingCosts
```

### Cost Model Formula

```
cost = k₁ × spread_bps + k₂ × σ × √(h/5) + k₃ × √(q/ADV)

Where:
- k₁ = 1.0 (spread penalty)
- k₂ = 0.15 (volatility timing)
- k₃ = 1.0 (market impact)
- spread_bps = bid-ask spread in basis points
- σ = volatility estimate
- h = horizon in minutes
- q = order size
- ADV = average daily volume
```

### Horizon Selection

```
score_h = net_h / √(h/5)

Trade if: score_{h*} ≥ θ_enter
θ_enter = cost_bps + reserve_bps
```

---

## Stage 4: Gating

### Purpose
Apply entry barriers based on market conditions and barrier model predictions.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `BarrierGate` | `gating/barrier_gate.py` | Peak/valley probability gating |
| `SpreadGate` | `gating/spread_gate.py` | Spread and quote freshness gates |

### Barrier Gate Logic

```python
# Input
arbitration_result: ArbitrationResult
barrier_predictions: Dict[str, float]  # P(peak), P(valley)

# Processing
p_peak = barrier_predictions.get("will_peak_5m", 0.0)
p_valley = barrier_predictions.get("will_valley_5m", 0.0)

# Gate formula
gate_factor = max(
    g_min,  # 0.2 minimum
    (1 - p_peak) ** gamma *    # Penalize peaks (γ=1.0)
    (0.5 + 0.5 * p_valley) ** delta  # Reward valleys (δ=0.5)
)

# Decision
if p_peak > 0.6:
    result = BLOCKED, reason="peak_probability_exceeded"
elif gate_factor < 0.3:
    result = BLOCKED, reason="low_gate_factor"
else:
    result = ALLOWED, size_reduction=1-gate_factor

# Output
GateResult:
    allowed: bool
    gate_factor: float
    reason: str
```

### Spread Gate Logic

```python
# Input
quote: Quote
    - bid, ask: float
    - timestamp: datetime

# Processing
spread_bps = (ask - bid) / mid * 10000
quote_age_ms = (now - quote.timestamp).total_seconds() * 1000

# Gates
if spread_bps > max_spread_bps:  # 12 bps default
    result = BLOCKED, reason="spread_exceeded"
elif quote_age_ms > max_quote_age_ms:  # 200ms default
    result = BLOCKED, reason="stale_quote"
else:
    result = ALLOWED

# Output
SpreadGateResult:
    allowed: bool
    spread_bps: float
    quote_age_ms: float
    reason: str
```

### Gate Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_min` | 0.2 | Minimum gate factor |
| `gamma` | 1.0 | Peak penalty exponent |
| `delta` | 0.5 | Valley reward exponent |
| `max_spread_bps` | 12 | Maximum spread (bps) |
| `max_quote_age_ms` | 200 | Maximum quote age (ms) |

---

## Stage 5: Sizing

### Purpose
Calculate target position sizes based on alpha strength and volatility.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `PositionSizer` | `sizing/position_sizer.py` | Full sizing pipeline |
| `VolatilityScaler` | `sizing/vol_scaling.py` | Volatility-scaled sizing |
| `TurnoverManager` | `sizing/turnover.py` | No-trade band management |

### Data Flow

```python
# Input
arbitration_result: ArbitrationResult
gate_result: GateResult
market_data: MarketSnapshot
current_position: int

# Processing
alpha = arbitration_result.net_score
volatility = market_data.volatility

# Volatility scaling
z = clip(alpha / volatility, -z_max, z_max)  # z_max = 3.0
raw_weight = z * (max_weight / z_max)  # max_weight = 0.05

# Apply gate reduction
weight = raw_weight * gate_result.gate_factor

# No-trade band
current_weight = current_position * price / portfolio_value
if abs(weight - current_weight) < no_trade_band:  # 0.008 = 80 bps
    weight = current_weight  # Don't trade

# Normalize to gross target
gross_exposure = sum(abs(weights))
if gross_exposure > gross_target:  # 0.5 = 50%
    weights = weights * gross_target / gross_exposure

# Calculate shares
target_shares = int(weight * portfolio_value / price)

# Output
SizingResult:
    target_weight: float
    target_shares: int
    trade_shares: int  # target_shares - current_position
```

### Sizing Formulas

```
Volatility Scaling:
z = clip(α/σ, -z_max, z_max)
weight = z × (max_weight / z_max)

No-Trade Band:
|w_new - w_current| < band → w_new = w_current

Gross Normalization:
if Σ|w_i| > gross_target:
    w_i ← w_i × gross_target / Σ|w_i|
```

### Sizing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_max` | 3.0 | Maximum Z-score |
| `max_weight` | 0.05 | Maximum weight per position (5%) |
| `gross_target` | 0.50 | Target gross exposure (50%) |
| `no_trade_band` | 0.008 | No-trade band (80 bps) |

---

## Stage 6: Risk

### Purpose
Validate all trades against risk limits and kill switches before execution.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `RiskGuardrails` | `risk/guardrails.py` | Kill switch orchestrator |
| `DrawdownMonitor` | `risk/drawdown.py` | Drawdown tracking |
| `ExposureTracker` | `risk/exposure.py` | Position exposure tracking |

### Kill Switch Logic

```python
# Input
sizing_results: Dict[symbol, SizingResult]
engine_state: EngineState

# Processing

# 1. Daily Loss Check
daily_pnl = calculate_daily_pnl(engine_state)
if daily_pnl < -max_daily_loss_pct * portfolio_value:
    kill_switch = "daily_loss"
    action = FLATTEN_ALL

# 2. Drawdown Check
current_drawdown = drawdown_monitor.get_drawdown()
if current_drawdown > max_drawdown_pct:
    kill_switch = "max_drawdown"
    action = FLATTEN_ALL

# 3. Position Concentration Check
for symbol, result in sizing_results.items():
    position_pct = abs(result.target_weight)
    if position_pct > max_position_pct:
        result.target_weight = sign(result.target_weight) * max_position_pct

# Output
RiskStatus:
    kill_switch_triggered: bool
    kill_switch_reason: str
    daily_pnl: float
    current_drawdown: float
    max_position_violated: bool
```

### Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_loss_pct` | 0.02 | Maximum daily loss (2%) |
| `max_drawdown_pct` | 0.10 | Maximum drawdown (10%) |
| `max_position_pct` | 0.20 | Maximum position concentration (20%) |

---

## Cycle Result

After all stages complete, the engine produces a `CycleResult`:

```python
@dataclass
class CycleResult:
    cycle_number: int
    timestamp: datetime
    portfolio_value: float

    # Per-symbol results
    predictions: Dict[str, HorizonPredictions]
    blend_results: Dict[str, BlendResult]
    arbitration_results: Dict[str, ArbitrationResult]
    gate_results: Dict[str, GateResult]
    sizing_results: Dict[str, SizingResult]

    # Aggregates
    num_trades: int
    num_holds: int
    num_blocked: int

    # Risk status
    risk_status: RiskStatus

    # Trace for debugging
    pipeline_trace: PipelineTrace
```

---

## Performance Benchmarks

| Stage | Target Latency | Typical |
|-------|----------------|---------|
| Prediction | < 100ms | 50-80ms |
| Blending | < 10ms | 2-5ms |
| Arbitration | < 10ms | 1-3ms |
| Gating | < 5ms | 1-2ms |
| Sizing | < 5ms | 1-2ms |
| Risk | < 5ms | 1-2ms |
| **Total Cycle** | **< 500ms** | **100-200ms** |

---

## Related Documentation

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Overall system architecture
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) - Mathematical foundations
- [../components/MODEL_INFERENCE.md](../components/MODEL_INFERENCE.md) - Model inference details
