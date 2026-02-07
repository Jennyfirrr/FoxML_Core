# Signal Generation

Guidelines for generating trading signals from model outputs.

## Pipeline Overview

```
Model Predictions → Standardization → Horizon Blending →
Barrier Gating → Cost Arbitration → Position Sizing → Signals
```

## Per-Horizon Blending

Ridge risk-parity ensemble:

```python
def blend_horizon(predictions: dict[str, np.ndarray],
                  corr_matrix: np.ndarray,
                  ic_vector: np.ndarray,
                  ridge_lambda: float = 0.15) -> np.ndarray:
    """
    w ∝ (Σ + λI)^{-1} μ
    """
    n = len(predictions)
    regularized = corr_matrix + ridge_lambda * np.eye(n)
    weights = np.linalg.solve(regularized, ic_vector)
    weights = np.maximum(weights, 0)  # No negative weights
    weights /= weights.sum()

    # Apply temperature compression for short horizons
    if horizon in ['5m', '10m']:
        T = 0.75 if horizon == '5m' else 0.85
        weights = weights ** (1/T)
        weights /= weights.sum()

    return sum(w * predictions[k] for k, w in zip(predictions, weights))
```

## Barrier Gating

```python
def apply_barrier_gate(alpha: float,
                       p_peak: float,
                       p_valley: float,
                       g_min: float = 0.3,
                       gamma: float = 2.0,
                       delta: float = 1.0) -> tuple[float, str]:
    """
    g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)
    """
    gate = max(g_min,
               (1 - p_peak)**gamma * (0.5 + 0.5*p_valley)**delta)

    # Blocking logic
    if p_peak > 0.6:
        return 0.0, "BLOCKED_PEAK"
    if p_peak > 0.65 and alpha > 0:
        return -alpha, "EXIT_SIGNAL"

    return alpha * gate, "GATED"
```

## Cost Arbitration

```python
def select_horizon(horizon_scores: dict[str, float],
                   spreads: dict[str, float],
                   volatility: float,
                   k1: float = 1.0,
                   k2: float = 0.15,
                   k3: float = 1.0) -> tuple[str, float]:
    """
    net_h = α_h - k₁×spread - k₂×σ×√(h/5) - k₃×impact
    """
    horizon_minutes = {'5m': 5, '10m': 10, '15m': 15,
                       '30m': 30, '60m': 60, '1d': 390}

    best_horizon, best_score = None, float('-inf')
    for h, alpha in horizon_scores.items():
        minutes = horizon_minutes[h]
        cost = (k1 * spreads.get(h, 2.0) +
                k2 * volatility * np.sqrt(minutes/5))
        net = alpha - cost
        if net > best_score:
            best_horizon, best_score = h, net

    return best_horizon, best_score
```

## Position Sizing

```python
def size_position(signal: float,
                  volatility: float,
                  adv: float,
                  capital: float,
                  z_max: float = 3.0,
                  max_weight: float = 0.1,
                  beta: float = 0.08) -> float:
    """
    Z-score clipping + volatility scaling
    """
    z = np.clip(signal, -z_max, z_max)

    # Volatility-scaled size
    vol_adjusted = z / (volatility * np.sqrt(252))

    # ADV constraint
    adv_limit = beta * adv

    # Capital constraint
    capital_limit = capital * max_weight

    return min(abs(vol_adjusted), adv_limit, capital_limit) * np.sign(z)
```

## Output Format

```python
signal = {
    'symbol': 'AAPL',
    'action': 'BUY',  # BUY, SELL, HOLD
    'target_weight': 0.05,
    'target_shares': 100,
    'horizon': '10m',
    'net_score': 0.72,
    'barrier_gate': 0.85,
    'confidence': 0.68,
}
```

## Related Skills

- `model-inference.md` - Loading models for prediction
- `risk-management.md` - Risk gates and limits
- `execution-engine.md` - Signal to order execution
- `broker-integration.md` - Order submission

## Related Documentation

- `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
