# Execution Engine Development

Guidelines for developing the execution engine in `LIVE_TRADING/`.

## Core Components

### Trading Engine (`core/engine/`)
- Main loop: data acquisition → prediction → sizing → execution
- Position tracking: `positions: dict[str, float]` (symbol → quantity)
- P&L tracking and daily returns
- Kill switch integration

### Prediction Layer (`core/prediction/`)
- Load models from TRAINING registry
- Handle tabular models: shape (N, F)
- Handle sequential models: shape (N, T, F) with SeqBufferManager
- Graceful missing model handling

### Horizon Blending
- Ridge risk-parity: `w ∝ (Σ + λI)^{-1} μ`
- Temperature compression for short horizons: `w^(T) ∝ w^(1/T)`
- λ = 0.15, T_5m = 0.75, T_10m = 0.85

### Cost Arbitration (`core/arbitration/`)
- Net score: `net_h = α_h - k₁×spread - k₂×σ×√(h/5) - k₃×impact`
- Trade gate: `score ≥ cost_bps + reserve_bps`
- Winner-takes-most or softmax selection

### Barrier Gating (`core/gating/`)
- Block long entry if P(peak) > 0.6
- Prefer entry if P(valley) > 0.55
- Position size reduction: `size × (1 - P(peak))`
- Exit signal if P(peak) > 0.65

### Position Sizing (`core/sizing/`)
- Z-score clipping: clip to ±3
- Volatility scaling: `size = min(β×ADV, risk_cap)`
- No-trade band for small signals
- Gross exposure targeting

## SST Compliance

Always use:
```python
from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.file_utils import write_atomic_json
from TRAINING.common.utils.determinism_ordering import sorted_items
```

## Testing

Reference patterns in:
- `DOCS/03_technical/trading/testing/TESTING_PLAN.md`
- Unit tests in `LIVE_TRADING/tests/`

## Related Skills

- `broker-integration.md` - Broker protocol and implementations
- `model-inference.md` - Model loading and prediction
- `signal-generation.md` - Signal blending and gating
- `risk-management.md` - Kill switches and risk limits

## Related Documentation

- `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
- `DOCS/03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md`
