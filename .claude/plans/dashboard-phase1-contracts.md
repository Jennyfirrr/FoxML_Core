# Dashboard Phase 1: Contracts

**Status**: Ready for Implementation
**Parent**: [dashboard-integration-master.md](./dashboard-integration-master.md)
**Estimated Effort**: 1 hour

---

## Objective

Define formal contracts between LIVE_TRADING and DASHBOARD modules, following the same pattern as TRAINING → LIVE_TRADING contracts.

---

## New Contracts

### Contract 1: Engine State (Enhanced)

**Purpose**: Real-time engine state including pipeline stage.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/trading_engine.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/api/state` |

#### Schema

```python
{
    # === REQUIRED FIELDS ===
    "status": str,              # "running", "paused", "stopped", "error"
    "current_stage": str,       # "idle", "prediction", "blending", "arbitration", "gating", "sizing", "risk", "execution"
    "last_cycle": str,          # ISO timestamp of last cycle completion
    "uptime_seconds": float,    # Seconds since engine start

    # === OPTIONAL FIELDS ===
    "cycle_count": int,         # Total cycles completed
    "symbols_active": int,      # Symbols being processed
    "last_error": str | None,   # Last error message if any
}
```

#### Contract Rules

1. `current_stage` MUST be updated when entering each pipeline stage
2. `status` MUST reflect actual engine state (not file-based control state)
3. `last_cycle` MUST be UTC ISO format

---

### Contract 2: Position Details

**Purpose**: Per-position information for position table display.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/state.py` → `EngineState.positions` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/api/positions` |

#### Schema

```python
{
    "positions": [
        {
            # === REQUIRED FIELDS ===
            "symbol": str,              # Stock symbol
            "shares": int,              # Number of shares (negative = short)
            "entry_price": float,       # Average entry price
            "current_price": float,     # Latest price
            "market_value": float,      # shares * current_price
            "unrealized_pnl": float,    # Current P&L
            "unrealized_pnl_pct": float, # P&L as percentage
            "weight": float,            # Portfolio weight [0, 1]

            # === OPTIONAL FIELDS ===
            "entry_time": str | None,   # ISO timestamp when position opened
            "side": str,                # "long" or "short"
            "target_weight": float,     # Target allocation
        }
    ],
    "total_positions": int,
    "long_count": int,
    "short_count": int,
    "total_market_value": float,
}
```

#### Contract Rules

1. `positions` array MUST be sorted by symbol for determinism
2. `unrealized_pnl` = `(current_price - entry_price) * shares`
3. `weight` = `market_value / total_portfolio_value`

---

### Contract 3: Trade History

**Purpose**: Audit trail of executed trades.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/state.py` → `EngineState.trade_history` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/api/trades/history` |

#### Schema

```python
{
    "trades": [
        {
            # === REQUIRED FIELDS ===
            "id": str,                  # Trade ID
            "timestamp": str,           # ISO timestamp
            "symbol": str,              # Stock symbol
            "side": str,                # "buy" or "sell"
            "shares": int,              # Shares traded
            "price": float,             # Execution price
            "value": float,             # Total trade value

            # === OPTIONAL FIELDS ===
            "order_id": str | None,     # Broker order ID
            "commission": float,        # Trading commission
            "slippage": float,          # Estimated slippage
            "decision_id": str | None,  # Link to decision that triggered trade
        }
    ],
    "total_trades": int,
    "total_value": float,
}
```

#### Query Parameters

- `limit`: Max trades to return (default 100)
- `offset`: Pagination offset
- `symbol`: Filter by symbol
- `since`: Filter by timestamp (ISO format)

#### Contract Rules

1. `trades` array MUST be sorted by timestamp descending (newest first)
2. `id` MUST be unique within a trading session
3. `timestamp` MUST be UTC ISO format

---

### Contract 4: CILS Statistics

**Purpose**: Online learning optimizer metrics.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/trading_engine.py` → `get_cils_stats()` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/api/cils/stats` |

#### Schema

```python
{
    # === REQUIRED FIELDS ===
    "enabled": bool,            # Whether CILS is enabled
    "algorithm": str,           # "thompson_sampling", "ucb", etc.

    # === OPTIONAL (when enabled) ===
    "arms": [
        {
            "horizon": str,     # Horizon name (e.g., "5m", "15m", "60m")
            "weight": float,    # Current selection weight [0, 1]
            "pulls": int,       # Number of times selected
            "rewards": float,   # Cumulative reward
            "mean_reward": float, # Average reward
        }
    ],
    "total_pulls": int,
    "exploration_rate": float,  # Current exploration vs exploitation
    "last_update": str,         # ISO timestamp
}
```

#### Contract Rules

1. `arms` array MUST include all configured horizons
2. `weight` values MUST sum to approximately 1.0
3. Return `{"enabled": false}` if CILS not configured

---

### Contract 5: Risk Status

**Purpose**: Current risk metrics and warnings.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/gating/risk.py` → `RiskGuardrails.get_status()` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/api/risk/status` |

#### Schema

```python
{
    # === REQUIRED FIELDS ===
    "trading_allowed": bool,    # Can execute trades?
    "kill_switch_active": bool, # Manual kill switch on?
    "kill_switch_reason": str | None,

    # === RISK METRICS ===
    "daily_pnl_pct": float,     # Daily P&L as % of portfolio
    "daily_loss_limit_pct": float, # Configured limit
    "daily_loss_remaining_pct": float, # Headroom before limit

    "drawdown_pct": float,      # Current drawdown from peak
    "max_drawdown_limit_pct": float, # Configured limit
    "drawdown_remaining_pct": float, # Headroom before limit

    "gross_exposure": float,    # Total absolute exposure
    "net_exposure": float,      # Long - short exposure
    "max_gross_exposure": float, # Configured limit

    # === WARNINGS ===
    "warnings": [
        {
            "type": str,        # "daily_loss", "drawdown", "exposure"
            "message": str,     # Human-readable warning
            "severity": str,    # "warning", "critical"
            "threshold_pct": float,
            "current_pct": float,
        }
    ],

    "last_check": str,          # ISO timestamp
}
```

#### Contract Rules

1. `trading_allowed` MUST be `false` if any limit is breached
2. `warnings` array MUST include all active warnings
3. Percentages MUST be in range [-100, 100] (not decimals)

---

### Contract 6: Training Progress (New!)

**Purpose**: Real-time training pipeline progress.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/orchestration/intelligent_trainer.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → `/ws/training` |

#### Schema (WebSocket Message)

```python
{
    # === REQUIRED FIELDS ===
    "event_type": str,          # "progress", "stage_change", "target_complete", "run_complete", "error"
    "timestamp": str,           # ISO timestamp
    "run_id": str,              # Current run ID

    # === PROGRESS EVENT ===
    "stage": str,               # "ranking", "feature_selection", "training"
    "progress_pct": float,      # Overall progress [0, 100]
    "current_target": str | None, # Target being processed
    "targets_complete": int,    # Targets finished
    "targets_total": int,       # Total targets

    # === STAGE CHANGE EVENT ===
    "previous_stage": str | None,
    "new_stage": str,

    # === TARGET COMPLETE EVENT ===
    "target": str,
    "status": str,              # "success", "failed", "skipped"
    "models_trained": int,
    "best_auc": float | None,

    # === ERROR EVENT ===
    "error_message": str,
    "error_type": str,
    "recoverable": bool,
}
```

#### Contract Rules

1. Events MUST be emitted at stage boundaries
2. Progress MUST be updated at least every 30 seconds during long operations
3. `run_id` MUST match the manifest run_id

---

## Implementation Checklist

### Step 1: Update INTEGRATION_CONTRACTS.md

- [ ] Add "LIVE_TRADING → DASHBOARD Contracts" section
- [ ] Document all 6 contracts with schemas
- [ ] Add to data flow diagram
- [ ] Update version to 1.2

### Step 2: Update Skills

- [ ] Update `dashboard-ipc-bridge.md` with new endpoints
- [ ] Add response schemas
- [ ] Document query parameters

### Step 3: Validate Against Existing Code

- [ ] Verify EngineState has required fields
- [ ] Verify RiskGuardrails exposes needed data
- [ ] Verify CILS optimizer has get_stats() method
- [ ] Identify any gaps requiring engine changes

---

## Files to Modify

| File | Changes |
|------|---------|
| `INTEGRATION_CONTRACTS.md` | Add LIVE_TRADING → DASHBOARD section |
| `.claude/skills/dashboard-ipc-bridge.md` | Update with new endpoints |
| `.claude/skills/dashboard-overview.md` | Update feature status |

---

## Validation

After implementation, verify:
1. All schemas are documented in INTEGRATION_CONTRACTS.md
2. Skills accurately reflect new capabilities
3. No breaking changes to existing contracts
