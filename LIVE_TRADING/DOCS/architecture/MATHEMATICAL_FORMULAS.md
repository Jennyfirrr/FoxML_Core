# Mathematical Formulas

This document contains all mathematical equations and formulas used in the LIVE_TRADING module.

---

## 1. Standardization & Calibration

### Z-Score Standardization

```
s_{m,h} = clip((r̂_{m,h} - μ_{m,h}) / σ_{m,h}, -3, 3)
```

**Where:**
- `s_{m,h}` = standardized score for model m, horizon h
- `r̂_{m,h}` = raw prediction from model
- `μ_{m,h}` = rolling mean (N ≈ 10 trading days)
- `σ_{m,h}` = rolling standard deviation
- `clip(-3, 3)` = prevents extreme outliers

**Used in:** `ZScoreStandardizer.standardize()`

**Why:** Different model families produce predictions on different scales. Standardization ensures fair comparison across models.

---

### Confidence Calculation

```
c_{m,h} = IC_{m,h} × freshness × capacity × stability
```

**Where:**
- `IC_{m,h} = Spearman(r̂_{m,h}^{t-1}, r_h^{t})` = Information Coefficient
- `freshness = e^{-Δt/τ_h}` with τ_{5m}=150s, τ_{10m}=300s
- `capacity = min(1, κ × ADV / planned_dollars)`
- `stability = 1 / rolling_RMSE_of_calibration`

**Used in:** `ConfidenceScorer.score()`

**Why:** Short horizons need freshness checks. Stale data or low-capacity situations should reduce model confidence.

---

### Calibrated Score

```
s̃_{m,h} = s_{m,h} × c_{m,h}
```

**Used in:** Final model score after confidence adjustment

**Why:** Combines standardized prediction with confidence to get realistic signal strength.

---

## 2. Within-Horizon Ensemble Blending

### Net IC After Costs

```
μ_{m,h} = IC_{m,h} - λ_c × cost_share_{m,h}
```

**Where:**
- `IC_{m,h}` = Information Coefficient for model m, horizon h
- `λ_c` = cost penalty parameter (typically 0.5)
- `cost_share_{m,h}` = expected_cost / expected_|alpha|

**Used in:** Target vector for ridge weights

**Why:** Penalizes models with high cost-to-alpha ratios.

---

### Ridge Risk-Parity Weights

```
w_h ∝ (Σ_h + λI)^{-1} μ_h
```

**Post-processing:**
```
w_h ← clip(w_h, 0, ∞)
Σw_h = 1
```

**Where:**
- `Σ_h` = correlation matrix of standardized scores
- `λ` = ridge regularization (0.15 default)
- `μ_h` = target vector of net IC after costs
- `I` = identity matrix

**Used in:** `RidgeWeightCalculator.calculate()`

**Why:** Ridge regression prevents overfitting to correlated models. Risk parity ensures no single model dominates.

---

### Temperature Compression

```
w_h^{(T)} ∝ w_h^{1/T}
```

**Temperature by horizon:**
| Horizon | Temperature |
|---------|-------------|
| 5m | 0.75 |
| 10m | 0.85 |
| 15m | 0.90 |
| 30m+ | 1.0 |

**Used in:** `TemperatureCompressor.compress()`

**Why:** Short horizons (5m, 10m) need more conservative weighting. T < 1 compresses weights toward uniform.

---

### Horizon Alpha

```
α_h = Σ_m w_{m,h}^{(T)} × s̃_{m,h}
```

**Used in:** `HorizonBlender.blend()`

**Why:** Combines all model predictions within a horizon using optimized weights.

---

## 3. Across-Horizon Arbitration

### Cost Model

```
cost_h = k₁ × spread_bps + k₂ × σ × √(h/5) + k₃ × √(q/ADV)
```

**Where:**
- `k₁` = spread penalty (1.0)
- `k₂` = volatility timing (0.15)
- `k₃` = market impact (1.0)
- `spread_bps` = bid-ask spread in basis points
- `σ` = volatility estimate
- `h` = horizon in minutes
- `q` = order size
- `ADV` = average daily volume

**Used in:** `CostModel.calculate()`

**Why:** Accounts for all trading costs when evaluating horizons.

---

### Net Score Calculation

```
net_h = α_h - cost_h
```

**Used in:** `HorizonArbiter.calculate_net_score()`

**Why:** Alpha must exceed all costs to be profitable.

---

### Horizon Score with Penalty

```
score_h = net_h / √(h/5)
```

**Used in:** `HorizonArbiter.select()`

**Why:** Penalizes longer horizons when deciding entries in a short-term (5m) trading loop. Favors actionable short horizons.

---

### Trade/No-Trade Gate

```
Trade if: score_{h*} ≥ θ_enter
θ_enter = cost_bps + reserve_bps
```

**Where:**
- `reserve_bps` = 0.5-1.0 × spread for 5m, 0.3-0.7 × spread for 10m

**Used in:** `HorizonArbiter.decide()`

**Why:** Only trade when expected alpha exceeds all costs plus safety margin.

---

## 4. Barrier Target Gating

### Barrier Gate Formula

```
g = max(g_min, (1 - P(peak))^γ × (0.5 + 0.5 × P(valley))^δ)
```

**Where:**
- `g_min` = 0.2 (minimum gate factor)
- `γ` = 1.0 (peak penalty exponent)
- `δ` = 0.5 (valley reward exponent)

**Used in:** `BarrierGate.calculate_gate_factor()`

---

### Long Entry Blocking

```
Block if: P(will_peak_5m) > 0.6
```

**Used in:** `BarrierGate.check()` for long entries

**Why:** Prevents buying into local tops when barrier models predict peak probability > 60%.

---

### Long Entry Preference

```
Prefer if: P(will_valley_5m) > 0.55 AND ΔP > 0
```

**Used in:** `BarrierGate.check()` for valley bounce detection

**Why:** Favors entries after valleys when price is turning up.

---

### Position Size Reduction

```
size_adjusted = size × g
```

**Where:** `g` is the gate factor from barrier calculation

**Used in:** `PositionSizer.apply_gate_reduction()`

**Why:** Reduces position size when peak probability is elevated but not blocking.

---

### Long Exit Signal

```
Exit if: P(will_peak_5m) > 0.65 OR α_{5m} < 0
```

**Used in:** `BarrierGate.check_exit()`

**Why:** Exit when peak probability is very high or alpha turns negative.

---

## 5. Position Sizing

### Volatility Scaling

```
z = clip(α/σ, -z_max, z_max)
weight = z × (max_weight / z_max)
```

**Where:**
- `z_max` = 3.0 (maximum z-score)
- `max_weight` = 0.05 (5% maximum position)
- `α` = net alpha score
- `σ` = volatility estimate

**Used in:** `VolatilityScaler.scale()`

**Why:** Scales position size inversely to volatility while capping extreme signals.

---

### No-Trade Band

```
if |w_new - w_current| < band:
    w_new = w_current  # No trade
```

**Where:** `band` = 0.008 (80 bps)

**Used in:** `TurnoverManager.apply_band()`

**Why:** Prevents excessive trading from small signal changes.

---

### Gross Exposure Normalization

```
if Σ|w_i| > gross_target:
    w_i ← w_i × gross_target / Σ|w_i|
```

**Where:** `gross_target` = 0.50 (50%)

**Used in:** `PositionSizer.normalize_gross()`

**Why:** Ensures total exposure doesn't exceed target.

---

### Share Calculation

```
shares = floor(weight × portfolio_value / price)
```

**Used in:** `PositionSizer.calculate_shares()`

---

## 6. Risk Management

### Drawdown Calculation

```
drawdown = (peak - current) / peak
```

**Used in:** `DrawdownMonitor.get_drawdown()`

---

### Maximum Drawdown

```
MDD = max_{t} [(peak_t - trough_t) / peak_t]
```

**Used in:** Risk monitoring

---

### Daily P&L

```
daily_pnl = portfolio_value_now - portfolio_value_sod
daily_pnl_pct = daily_pnl / portfolio_value_sod
```

**Used in:** Daily loss kill switch

---

### Gross Exposure

```
gross_exposure = Σ|w_i|
```

**Used in:** `ExposureTracker.calculate_gross()`

---

### Net Exposure

```
net_exposure = Σw_i
```

**Used in:** `ExposureTracker.calculate_net()`

---

## 7. Online Learning (Exp3-IX)

### Weight Update

```
u_i ← u_i × exp(η × r̂_i)
r̂_i = r_i / p_i
```

**Where:**
- `u_i` = unnormalized weight for arm i
- `η` = learning rate
- `r_i` = observed reward (net P&L bps)
- `p_i` = probability of selecting arm i

**Used in:** `Exp3IXBandit.update()`

---

### Probability Selection

```
p_i = (1 - γ) × (u_i / Σu) + γ / K
```

**Where:**
- `γ` = exploration rate (0.05)
- `K` = number of arms

**Used in:** `Exp3IXBandit.get_probabilities()`

---

### Adaptive Learning Rate

```
η = min(η_max, √(ln K / (K × T)))
```

**Where:**
- `η_max` = 0.07
- `K` = number of arms
- `T` = total time steps

**Used in:** `Exp3IXBandit._compute_adaptive_eta()`

---

### Reward Calculation

```
reward_bps = (exit_price - entry_price) × quantity - fees - slippage
            ───────────────────────────────────────────────────────
                           entry_price × quantity

reward_bps = reward_bps × 10000
```

**Used in:** `RewardTracker.calculate_reward()`

---

## 8. Performance Metrics

### Information Coefficient

```
IC = Spearman(predictions, actual_returns)
```

**Used in:** Model performance evaluation

---

### Sharpe Ratio

```
Sharpe = (mean_returns - r_f) / std_returns × √252
```

**Where:** `r_f` = risk-free rate (often assumed 0 for intraday)

**Used in:** Performance evaluation

---

### Maximum Drawdown

```
MDD = max[(peak_t - valley_t) / peak_t]
```

**Used in:** Risk metrics

---

## 9. Cross-Horizon Ensemble (Phase 9)

### Decay Functions

**Exponential:**
```
w = exp(-ln(2) × distance / half_life)
```

**Linear:**
```
w = max(0, 1 - distance / max_distance)
```

**Inverse:**
```
w = 1 / (distance + ε)
```

**Used in:** Cross-horizon stacking weights

---

### Cross-Horizon Stacking

```
final_pred = Σ_h (decay_weight_h × blend_weight_h × pred_h)
```

**Used in:** `CrossHorizonStacker.predict()`

---

## 10. Configuration Parameters

### Temperature by Horizon

| Horizon | Temperature |
|---------|-------------|
| 5m | 0.75 |
| 10m | 0.85 |
| 15m | 0.90 |
| 30m | 1.0 |
| 60m | 1.0 |
| 1d | 1.0 |

---

### Cost Coefficients

| Coefficient | Value | Description |
|-------------|-------|-------------|
| k₁ | 1.0 | Spread penalty |
| k₂ | 0.15 | Volatility timing |
| k₃ | 1.0 | Market impact |

---

### Risk Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_daily_loss` | 2% | Daily loss limit |
| `max_drawdown` | 10% | Maximum drawdown |
| `max_position` | 20% | Position concentration |
| `max_gross` | 50% | Gross exposure |

---

### Sizing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_max` | 3.0 | Maximum Z-score |
| `max_weight` | 5% | Maximum position weight |
| `gross_target` | 50% | Target gross exposure |
| `no_trade_band` | 80 bps | No-trade band |

---

### Bandit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.05 | Exploration rate |
| `eta_max` | 0.07 | Maximum learning rate |

---

## Summary Table

| Component | Formula | Parameters |
|-----------|---------|------------|
| Z-Score | `clip((r̂ - μ) / σ, -3, 3)` | N=10 days |
| Ridge Weights | `(Σ + λI)^{-1} μ` | λ=0.15 |
| Temperature | `w^{1/T}` | T=0.75-1.0 |
| Cost Model | `k₁×spread + k₂×σ×√h + k₃×√q` | k₁=1, k₂=0.15, k₃=1 |
| Barrier Gate | `max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)` | g_min=0.2, γ=1, δ=0.5 |
| Vol Scaling | `z = clip(α/σ, -z_max, z_max)` | z_max=3 |
| Exp3-IX | `u × exp(η × r/p)` | γ=0.05, η_max=0.07 |

---

## Related Documentation

- [PIPELINE_STAGES.md](PIPELINE_STAGES.md) - How formulas are used in each stage
- [../components/MULTI_HORIZON_AND_INTERVAL.md](../components/MULTI_HORIZON_AND_INTERVAL.md) - Multi-horizon details
- [../reference/CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md) - Parameter configuration
