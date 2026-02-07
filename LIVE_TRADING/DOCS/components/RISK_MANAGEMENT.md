# Risk Management

This document describes the risk management system in the LIVE_TRADING module, including kill switches, drawdown monitoring, and exposure tracking.

---

## Overview

The risk management system provides multiple layers of protection:

1. **Kill Switches**: Hard limits that halt all trading
2. **Drawdown Monitoring**: Track peak-to-trough losses
3. **Exposure Tracking**: Monitor position concentrations
4. **Pre-Trade Gates**: Validate conditions before trading
5. **Position Limits**: Enforce maximum position sizes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RISK MANAGEMENT SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      KILL SWITCHES                               │    │
│  │                                                                  │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │    │
│  │   │  Daily Loss  │  │  Drawdown    │  │  Position    │         │    │
│  │   │    2% max    │  │   10% max    │  │   20% max    │         │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘         │    │
│  │                                                                  │    │
│  │   Any trigger → FLATTEN ALL POSITIONS                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      PRE-TRADE GATES                             │    │
│  │                                                                  │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │    │
│  │   │ Spread Gate  │  │ Quote Age    │  │ Barrier Gate │         │    │
│  │   │  12 bps max  │  │  200ms max   │  │  Peak/Valley │         │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘         │    │
│  │                                                                  │    │
│  │   Any gate fails → BLOCK TRADE                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      MONITORING                                  │    │
│  │                                                                  │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │    │
│  │   │  Drawdown    │  │  Exposure    │  │  P&L         │         │    │
│  │   │  Monitor     │  │  Tracker     │  │  Calculator  │         │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘         │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kill Switches

### RiskGuardrails

**Location**: `LIVE_TRADING/risk/guardrails.py`

Main orchestrator for risk checks.

```python
from LIVE_TRADING.risk import RiskGuardrails

guardrails = RiskGuardrails(
    max_daily_loss_pct=0.02,     # 2%
    max_drawdown_pct=0.10,       # 10%
    max_gross_exposure=0.50,     # 50%
    max_position_pct=0.20,       # 20%
    initial_capital=100_000.0,   # Starting capital
)

# Check risk status before trading
status = guardrails.check_trading_allowed(
    portfolio_value=95_000,
    weights={"AAPL": 0.15, "MSFT": 0.10},
    current_time=datetime.now(timezone.utc),  # Optional
)

if not status.is_trading_allowed:
    print(f"Kill switch: {status.kill_switch_reason}")
    # Flatten all positions
    engine.flatten_all()
```

### Kill Switch Types

| Kill Switch | Threshold | Trigger | Action |
|-------------|-----------|---------|--------|
| Daily Loss | 2% | P&L today < -2% of NAV | Flatten all |
| Max Drawdown | 10% | Peak-to-trough > 10% | Flatten all |
| Position Concentration | 20% | Single position > 20% | Reduce position |

### RiskStatus Output

```python
@dataclass
class RiskStatus:
    """Current risk status summary."""
    is_trading_allowed: bool      # False if kill switch active
    daily_pnl_pct: float          # Today's P&L as percentage
    drawdown_pct: float           # Current drawdown percentage
    gross_exposure: float         # Total absolute position weights
    net_exposure: float           # Sum of signed position weights
    kill_switch_reason: Optional[str]  # Why trading is blocked
    warnings: List[str]           # Non-blocking warnings
```

### Trade Validation

```python
# Validate a proposed trade before execution
result = guardrails.validate_trade(
    symbol="AAPL",
    target_weight=0.15,
    current_weights={"MSFT": 0.10},
)

if not result.passed:
    print(f"Trade blocked: {result.message}")

# Apply risk adjustments to a list of decisions
adjusted = guardrails.apply_risk_adjustments(
    decisions=trade_decisions,
    current_weights=current_weights,
)
```

---

## Drawdown Monitoring

### DrawdownMonitor

**Location**: `LIVE_TRADING/risk/drawdown.py`

Tracks portfolio peaks and calculates drawdown.

```python
from LIVE_TRADING.risk import DrawdownMonitor

monitor = DrawdownMonitor()

# Update with current portfolio value
monitor.update(portfolio_value=105000)

# Get current drawdown
drawdown = monitor.get_drawdown()  # 0.0 if at peak

# After loss
monitor.update(portfolio_value=98000)
drawdown = monitor.get_drawdown()  # 0.0667 (6.67%)

# Get peak
peak = monitor.peak  # 105000
```

### Drawdown Calculation

```
drawdown = (peak - current) / peak

Where:
- peak = highest portfolio value seen
- current = current portfolio value
```

### Daily vs All-Time Drawdown

```python
# All-time drawdown
all_time_dd = monitor.get_drawdown()

# Daily drawdown (resets at market open)
daily_dd = monitor.get_daily_drawdown()

# Weekly drawdown
weekly_dd = monitor.get_weekly_drawdown()
```

### Persistence

```python
# Save state
state = monitor.to_dict()
# {
#     "peak": 105000,
#     "daily_peak": 103000,
#     "weekly_peak": 100000,
#     "last_reset": "2024-01-15T09:30:00Z"
# }

# Restore
monitor = DrawdownMonitor.from_dict(state)
```

---

## Exposure Tracking

### ExposureTracker

**Location**: `LIVE_TRADING/risk/exposure.py`

Tracks position-level and portfolio-level exposure.

```python
from LIVE_TRADING.risk import ExposureTracker

tracker = ExposureTracker(
    max_position_pct=0.20,
    max_gross_exposure=0.50,
    max_net_exposure=0.30,
)

# Update positions
tracker.update_positions({
    "AAPL": {"value": 15000, "weight": 0.15},
    "MSFT": {"value": 10000, "weight": 0.10},
    "GOOGL": {"value": -5000, "weight": -0.05},
}, portfolio_value=100000)

# Get exposure metrics
metrics = tracker.get_metrics()
# {
#     "gross_exposure": 0.30,  # |0.15| + |0.10| + |-0.05|
#     "net_exposure": 0.20,    # 0.15 + 0.10 - 0.05
#     "max_position": 0.15,
#     "num_positions": 3,
#     "long_exposure": 0.25,
#     "short_exposure": 0.05,
# }

# Check for violations
violations = tracker.check_limits()
# []  # No violations
```

### Exposure Limits

| Metric | Default | Description |
|--------|---------|-------------|
| `max_position_pct` | 20% | Maximum single position |
| `max_gross_exposure` | 50% | Maximum sum of absolute weights |
| `max_net_exposure` | 30% | Maximum sum of signed weights |
| `max_sector_exposure` | 30% | Maximum exposure to single sector |

### Position Sizing Integration

```python
def apply_position_limits(
    target_weights: Dict[str, float],
    tracker: ExposureTracker,
) -> Dict[str, float]:
    """Apply exposure limits to target weights."""
    result = {}

    for symbol, weight in target_weights.items():
        # Cap individual position
        capped = np.sign(weight) * min(
            abs(weight),
            tracker.max_position_pct
        )
        result[symbol] = capped

    # Normalize if gross exceeds target
    gross = sum(abs(w) for w in result.values())
    if gross > tracker.max_gross_exposure:
        scale = tracker.max_gross_exposure / gross
        result = {s: w * scale for s, w in result.items()}

    return result
```

---

## Pre-Trade Gates

### Spread Gate

Block trades when spreads are too wide.

```python
from LIVE_TRADING.gating import SpreadGate

gate = SpreadGate(
    max_spread_bps=12,
    max_quote_age_ms=200,
)

result = gate.check(quote)

if not result.allowed:
    print(f"Blocked: {result.reason}")
    # "spread_exceeded" or "stale_quote"
```

### Quote Age Gate

Reject stale market data.

```python
quote_age_ms = (now - quote.timestamp).total_seconds() * 1000

if quote_age_ms > 200:
    # Reject - data too stale for intraday trading
    return GateResult(allowed=False, reason="stale_quote")
```

### Barrier Gate

Block entries based on peak/valley probabilities.

```python
from LIVE_TRADING.gating import BarrierGate

gate = BarrierGate(
    g_min=0.2,
    gamma=1.0,
    delta=0.5,
)

result = gate.check(
    p_peak=0.65,
    p_valley=0.30,
    side="long",
)

if not result.allowed:
    print(f"Blocked: {result.reason}")  # "peak_probability_exceeded"
```

### Gate Summary

| Gate | Condition | Action |
|------|-----------|--------|
| Spread | spread > 12 bps | Block trade |
| Quote Age | age > 200ms | Block trade |
| Barrier (Long) | P(peak) > 0.6 | Block long entry |
| Barrier (Short) | P(valley) > 0.6 | Block short entry |
| Cost Sanity | slippage > 0.6 × spread | Block trade |

---

## Daily Loss Limit

### Implementation

```python
def check_daily_loss(
    engine_state: EngineState,
    max_daily_loss_pct: float = 0.02,
) -> bool:
    """Check if daily loss limit exceeded."""
    # Get start-of-day value
    sod_value = engine_state.start_of_day_value

    # Current P&L
    current_value = engine_state.portfolio_value
    daily_pnl = current_value - sod_value
    daily_pnl_pct = daily_pnl / sod_value

    if daily_pnl_pct < -max_daily_loss_pct:
        return True  # Kill switch triggered

    return False
```

### Reset Logic

```python
def _check_day_reset(self) -> None:
    """Reset daily metrics at market open."""
    now = datetime.now(timezone.utc)
    market_open = self._get_market_open(now)

    if self._last_reset < market_open <= now:
        self._daily_peak = self._current_value
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_reset = now
```

---

## Configuration

```yaml
# CONFIG/live_trading/live_trading.yaml

live_trading:
  risk:
    # Kill switches
    max_daily_loss_pct: 0.02      # 2% max daily loss
    max_drawdown_pct: 0.10        # 10% max drawdown
    max_position_pct: 0.20        # 20% max single position

    # Exposure limits
    max_gross_exposure: 0.50      # 50% gross
    max_net_exposure: 0.30        # 30% net
    max_sector_exposure: 0.30     # 30% per sector

    # Pre-trade gates
    max_spread_bps: 12            # 12 bps max spread
    max_quote_age_ms: 200         # 200ms max quote age
    max_slippage_ratio: 0.6       # slippage < 60% of spread

    # Barrier gates
    barrier_enabled: true
    peak_threshold: 0.6           # Block if P(peak) > 0.6
    valley_threshold: 0.6         # Block short if P(valley) > 0.6

    # Recovery
    cool_off_minutes: 30          # Time before resuming after kill switch
    gradual_recovery: true        # Gradually increase limits after kill switch
```

---

## Kill Switch Recovery

### Automatic Recovery

```python
class KillSwitchManager:
    def __init__(self, cool_off_minutes: int = 30):
        self._cool_off = timedelta(minutes=cool_off_minutes)
        self._triggered_at: Optional[datetime] = None

    def trigger(self, reason: str) -> None:
        """Trigger kill switch."""
        self._triggered_at = datetime.now(timezone.utc)
        self._reason = reason
        logger.critical(f"KILL SWITCH: {reason}")

    def is_active(self) -> bool:
        """Check if kill switch is still active."""
        if self._triggered_at is None:
            return False

        elapsed = datetime.now(timezone.utc) - self._triggered_at
        return elapsed < self._cool_off

    def can_resume(self) -> bool:
        """Check if trading can resume."""
        return not self.is_active()
```

### Gradual Recovery

```python
def get_recovery_scale(
    time_since_trigger: timedelta,
    cool_off: timedelta,
) -> float:
    """
    Gradually increase limits after kill switch.

    Returns scale factor 0.0 to 1.0.
    """
    if time_since_trigger < cool_off:
        return 0.0

    recovery_time = cool_off  # Same duration for full recovery
    elapsed_recovery = time_since_trigger - cool_off

    scale = min(1.0, elapsed_recovery / recovery_time)
    return scale
```

---

## Alerting

### Alert Integration

```python
from LIVE_TRADING.alerting import AlertManager

alert_manager = AlertManager(channels=["slack", "email"])

# On kill switch
if status.kill_switch_triggered:
    alert_manager.send_alert(
        severity="critical",
        title="Kill Switch Triggered",
        message=f"Reason: {status.kill_switch_reason}",
        details={
            "daily_pnl": status.daily_pnl,
            "drawdown": status.current_drawdown,
            "max_position": status.max_position_pct,
        },
    )
```

### Alert Types

| Severity | Condition | Channels |
|----------|-----------|----------|
| Critical | Kill switch triggered | All |
| Warning | Approaching limit (80%) | Slack |
| Info | Daily summary | Email |

---

## Testing

```bash
# Run risk management tests
pytest LIVE_TRADING/tests/test_phase0.py -v -k "risk"

# Specific components
pytest LIVE_TRADING/tests/test_phase0.py::test_drawdown_monitor -v
pytest LIVE_TRADING/tests/test_phase0.py::test_exposure_tracker -v
pytest LIVE_TRADING/tests/test_phase0.py::test_risk_guardrails -v
```

---

## Best Practices

### 1. Always Check Risk Before Trading

```python
# In trading loop
risk_status = guardrails.check(engine_state)

if risk_status.kill_switch_triggered:
    engine.flatten_all()
    return

if risk_status.action == "HOLD":
    logger.warning(f"Risk hold: {risk_status.reason}")
    return

# Proceed with trading...
```

### 2. Use Multiple Layers

Don't rely on a single check:

```python
# Layer 1: Kill switches
if daily_loss > max_daily:
    flatten_all()

# Layer 2: Pre-trade gates
if spread > max_spread:
    block_trade()

# Layer 3: Position limits
weight = min(target_weight, max_position)

# Layer 4: Gross exposure normalization
if gross > max_gross:
    scale_down_all()
```

### 3. Log All Risk Decisions

```python
logger.info(f"Risk check: daily_pnl={daily_pnl:.2%}, dd={drawdown:.2%}, "
            f"gross={gross:.2%}, max_pos={max_pos:.2%}")

if blocked:
    logger.warning(f"Trade blocked: {reason}")
```

---

## Related Documentation

- [../architecture/PIPELINE_STAGES.md](../architecture/PIPELINE_STAGES.md) - Risk stage in pipeline
- [../architecture/MATHEMATICAL_FORMULAS.md](../architecture/MATHEMATICAL_FORMULAS.md) - Risk formulas
- [../reference/CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md) - Risk configuration
