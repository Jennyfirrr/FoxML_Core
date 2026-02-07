# Risk Management

Guidelines for implementing risk guardrails.

## Kill Switches

Automatic trading halt when limits breached:

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_daily_loss_pct | 2.0% | Daily loss limit |
| max_daily_loss_dollars | $2000 | Absolute daily limit |
| max_drawdown_pct | 10.0% | Max drawdown from peak |
| max_position_size_pct | 20.0% | Single position limit |
| max_sector_exposure_pct | 30.0% | Sector concentration |

## Pre-Trade Validation

Before every order:
1. Check kill switches not triggered
2. Validate position size within limits
3. Verify sufficient capital (buy) or shares (sell)
4. Check spread gate (< 8-12 bps)
5. Verify quote freshness (< 200ms)

## Risk Limits

```python
risk_limits = {
    "max_position_size": 0.1,     # 10% of capital
    "stop_loss_pct": 0.02,        # 2% stop loss
    "take_profit_pct": 0.04,      # 4% take profit
    "max_leverage": 1.0,          # No leverage
    "max_correlation": 0.7,       # Portfolio correlation
}
```

## Risk Metrics

- **VaR**: Value at Risk at 95%/99% confidence
- **Expected Shortfall**: Conditional VaR (tail risk)
- **Leverage**: Portfolio value / capital
- **Concentration**: Max single position weight

## Implementation

Location: `LIVE_TRADING/risk/`

Key class: `RiskGuardrails`
- `check_kill_switches(daily_returns, capital) -> bool`
- `validate_position_size(symbol, shares, price, capital) -> bool`
- `validate_trade(trade, capital, positions) -> bool`
- `get_risk_report(...) -> dict`

## Related Skills

- `execution-engine.md` - Engine integration with risk checks
- `signal-generation.md` - Signal gating before execution
- `broker-integration.md` - Order submission with validation

## Related Documentation

- `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
