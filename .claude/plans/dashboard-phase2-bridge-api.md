# Dashboard Phase 2: Bridge API Enhancements

**Status**: COMPLETE
**Parent**: [dashboard-integration-master.md](./dashboard-integration-master.md)
**Estimated Effort**: 3 hours
**Dependencies**: Phase 1 (Contracts) complete
**Completed**: 2026-01-21

---

## Objective

Add missing endpoints to the IPC bridge and enable stage tracking in the trading engine.

---

## Current Bridge Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/api/metrics` | ✅ Works | Basic metrics |
| `/api/state` | ⚠️ Partial | Reads file, hardcoded stage |
| `/api/events/recent` | ✅ Works | From EventBus |
| `/api/control/*` | ✅ Works | File-based control |
| `/ws/events` | ✅ Works | Event streaming |
| `/ws/alpaca` | ✅ Works | Alpaca events |

## New Endpoints

### Endpoint 1: `/api/positions`

**Purpose**: Return detailed position information.

**Implementation**:

```python
@app.get("/api/positions")
async def get_positions() -> Dict[str, Any]:
    """
    Get detailed position information.

    Returns list of positions with entry price, current price, P&L, weight.
    """
    if not OBSERVABILITY_AVAILABLE:
        return {"positions": [], "total_positions": 0}

    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        positions = []
        for symbol in sorted(state.positions.keys()):
            pos = state.positions[symbol]
            positions.append({
                "symbol": symbol,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.shares * pos.current_price,
                "unrealized_pnl": (pos.current_price - pos.entry_price) * pos.shares,
                "unrealized_pnl_pct": ((pos.current_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0,
                "weight": pos.weight,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "side": "long" if pos.shares > 0 else "short",
            })

        total_value = sum(p["market_value"] for p in positions)
        return {
            "positions": positions,
            "total_positions": len(positions),
            "long_count": sum(1 for p in positions if p["shares"] > 0),
            "short_count": sum(1 for p in positions if p["shares"] < 0),
            "total_market_value": total_value,
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"positions": [], "total_positions": 0, "error": str(e)}
```

---

### Endpoint 2: `/api/trades/history`

**Purpose**: Return trade audit trail.

**Implementation**:

```python
@app.get("/api/trades/history")
async def get_trade_history(
    limit: int = 100,
    offset: int = 0,
    symbol: Optional[str] = None,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get trade history with optional filtering.

    Args:
        limit: Max trades to return (default 100)
        offset: Pagination offset
        symbol: Filter by symbol
        since: Filter by timestamp (ISO format)
    """
    if not OBSERVABILITY_AVAILABLE:
        return {"trades": [], "total_trades": 0}

    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        trades = list(state.trade_history)

        # Apply filters
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            trades = [t for t in trades if datetime.fromisoformat(t["timestamp"]) >= since_dt]

        # Sort by timestamp descending
        trades.sort(key=lambda t: t["timestamp"], reverse=True)

        # Paginate
        total = len(trades)
        trades = trades[offset:offset + limit]

        return {
            "trades": trades,
            "total_trades": total,
            "total_value": sum(t.get("value", 0) for t in trades),
        }
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"trades": [], "total_trades": 0, "error": str(e)}
```

---

### Endpoint 3: `/api/cils/stats`

**Purpose**: Return CILS optimizer statistics.

**Implementation**:

```python
@app.get("/api/cils/stats")
async def get_cils_stats() -> Dict[str, Any]:
    """
    Get CILS (online learning) optimizer statistics.

    Returns bandit arm weights, pull counts, and rewards.
    """
    if not OBSERVABILITY_AVAILABLE:
        return {"enabled": False}

    try:
        from LIVE_TRADING.engine import get_engine
        engine = get_engine()

        if engine is None:
            return {"enabled": False}

        stats = engine.get_cils_stats()
        if stats is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "algorithm": stats.get("algorithm", "unknown"),
            "arms": sorted(stats.get("arms", []), key=lambda a: a.get("horizon", "")),
            "total_pulls": stats.get("total_pulls", 0),
            "exploration_rate": stats.get("exploration_rate", 0),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting CILS stats: {e}")
        return {"enabled": False, "error": str(e)}
```

---

### Endpoint 4: `/api/risk/status`

**Purpose**: Return current risk metrics and warnings.

**Implementation**:

```python
@app.get("/api/risk/status")
async def get_risk_status() -> Dict[str, Any]:
    """
    Get current risk status including drawdown, exposure, and warnings.
    """
    if not OBSERVABILITY_AVAILABLE:
        return {
            "trading_allowed": False,
            "kill_switch_active": False,
            "warnings": [],
        }

    try:
        from LIVE_TRADING.gating.risk import get_risk_status
        status = get_risk_status()

        return {
            "trading_allowed": status.trading_allowed,
            "kill_switch_active": status.kill_switch_active,
            "kill_switch_reason": status.kill_switch_reason,

            "daily_pnl_pct": status.daily_pnl_pct,
            "daily_loss_limit_pct": status.daily_loss_limit_pct,
            "daily_loss_remaining_pct": status.daily_loss_limit_pct - abs(min(0, status.daily_pnl_pct)),

            "drawdown_pct": status.drawdown_pct,
            "max_drawdown_limit_pct": status.max_drawdown_limit_pct,
            "drawdown_remaining_pct": status.max_drawdown_limit_pct - status.drawdown_pct,

            "gross_exposure": status.gross_exposure,
            "net_exposure": status.net_exposure,
            "max_gross_exposure": status.max_gross_exposure,

            "warnings": [
                {
                    "type": w.type,
                    "message": w.message,
                    "severity": w.severity,
                    "threshold_pct": w.threshold_pct,
                    "current_pct": w.current_pct,
                }
                for w in status.warnings
            ],

            "last_check": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        return {
            "trading_allowed": False,
            "error": str(e),
            "warnings": [],
        }
```

---

### Endpoint 5: Enhanced `/api/state`

**Purpose**: Return live engine state with actual current stage.

**Changes needed in trading engine**:

```python
# In LIVE_TRADING/engine/trading_engine.py

class TradingEngine:
    def __init__(self, ...):
        ...
        self._current_stage = "idle"
        self._stage_lock = threading.Lock()

    def _set_stage(self, stage: str) -> None:
        """Update current pipeline stage."""
        with self._stage_lock:
            self._current_stage = stage
        # Emit event for dashboard
        from LIVE_TRADING.observability import events
        events.emit(events.EventType.STAGE_CHANGE, {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_current_stage(self) -> str:
        """Get current pipeline stage."""
        with self._stage_lock:
            return self._current_stage

    def _process_symbol(self, symbol: str, ...) -> Optional[TradeDecision]:
        """Process a symbol through the pipeline."""
        self._set_stage("prediction")
        predictions = self._generate_predictions(symbol, ...)

        self._set_stage("blending")
        blended = self._blend_predictions(predictions, ...)

        self._set_stage("arbitration")
        selected = self._arbitrate(blended, ...)

        self._set_stage("gating")
        if not self._gate_check(selected, ...):
            return None

        self._set_stage("sizing")
        sized = self._size_position(selected, ...)

        self._set_stage("risk")
        if not self._risk_check(sized, ...):
            return None

        self._set_stage("execution")
        return sized

    def _run_cycle(self) -> None:
        """Run one trading cycle."""
        self._set_stage("idle")  # Reset at cycle start
        ...
        for symbol in symbols:
            decision = self._process_symbol(symbol, ...)
        ...
        self._set_stage("idle")  # Reset at cycle end
```

**Bridge enhancement**:

```python
@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current engine state including pipeline stage.
    """
    if not OBSERVABILITY_AVAILABLE:
        return {
            "status": "unknown",
            "current_stage": "idle",
            "last_cycle": None,
            "uptime_seconds": 0,
        }

    try:
        from LIVE_TRADING.engine import get_engine
        engine = get_engine()

        if engine is None:
            return {
                "status": "stopped",
                "current_stage": "idle",
                "last_cycle": None,
                "uptime_seconds": 0,
            }

        state = engine.get_state_summary()
        return {
            "status": state.get("status", "unknown"),
            "current_stage": engine.get_current_stage(),  # Live stage!
            "last_cycle": state.get("last_cycle"),
            "uptime_seconds": state.get("uptime_seconds", 0),
            "cycle_count": state.get("cycle_count", 0),
            "symbols_active": state.get("symbols_active", 0),
        }
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return {
            "status": "error",
            "current_stage": "idle",
            "error": str(e),
        }
```

---

### WebSocket: `/ws/training`

**Purpose**: Stream training progress events.

**Implementation**:

```python
# Training event queue
training_event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

def on_training_event(event: Dict[str, Any]) -> None:
    """Callback for training events."""
    try:
        training_event_queue.put_nowait(event)
    except asyncio.QueueFull:
        # Drop oldest event
        try:
            training_event_queue.get_nowait()
            training_event_queue.put_nowait(event)
        except:
            pass

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for training progress streaming.
    """
    await websocket.accept()

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    training_event_queue.get(),
                    timeout=1.0
                )
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
```

---

## Engine Modifications Required

### 1. Add STAGE_CHANGE Event Type

In `LIVE_TRADING/observability/events.py`:

```python
class EventType(Enum):
    ...
    # Pipeline stage events
    STAGE_CHANGE = auto()
```

### 2. Add Stage Tracking to Engine

See enhanced `/api/state` section above.

### 3. Add Risk Status Export

In `LIVE_TRADING/gating/risk.py`, add:

```python
def get_risk_status() -> RiskStatus:
    """Get current risk status for dashboard."""
    guardrails = get_guardrails()
    if guardrails is None:
        return RiskStatus(trading_allowed=False, ...)
    return guardrails.get_status()
```

### 4. Add Engine State Export

In `LIVE_TRADING/engine/__init__.py`:

```python
_engine_instance: Optional[TradingEngine] = None

def get_engine() -> Optional[TradingEngine]:
    """Get the global engine instance."""
    return _engine_instance

def get_engine_state() -> Optional[EngineState]:
    """Get the engine's state object."""
    if _engine_instance:
        return _engine_instance.state
    return None
```

---

## Implementation Checklist

### Step 1: Engine Changes (1h)
- [x] Add `STAGE_CHANGE` event type
- [x] Add `_current_stage` tracking to TradingEngine
- [x] Add `_set_stage()` method with event emission
- [x] Call `_set_stage()` at each pipeline stage
- [x] Add `get_engine()` and `get_engine_state()` exports

### Step 2: Risk Status Export (30m)
- [x] Add `get_risk_status()` function
- [x] Ensure RiskStatus has all required fields (added `DashboardRiskStatus`, `RiskWarning`)
- [ ] Test with mock guardrails (deferred to testing phase)

### Step 3: Bridge Endpoints (1h)
- [x] Add `/api/positions` endpoint
- [x] Add `/api/trades/history` endpoint
- [x] Add `/api/cils/stats` endpoint
- [x] Add `/api/risk/status` endpoint
- [x] Enhance `/api/state` with live stage
- [x] Add `/ws/training` WebSocket
- [x] Add `/api/training/event` POST endpoint (for pushing events from training pipeline)

### Step 4: Testing (30m)
- [ ] Test each endpoint with curl
- [ ] Test with mock mode (no engine)
- [ ] Test with actual engine running

---

## Files Modified

| File | Changes |
|------|---------|
| `LIVE_TRADING/observability/events.py` | Added `STAGE_CHANGE` event type, `emit_stage_change()` helper |
| `LIVE_TRADING/engine/trading_engine.py` | Added `_current_stage`, `_stage_lock`, `_set_stage()`, `get_current_stage()`, stage tracking at all pipeline stages |
| `LIVE_TRADING/engine/__init__.py` | Added `get_engine()`, `get_engine_state()`, `set_engine()` |
| `LIVE_TRADING/risk/guardrails.py` | Added `RiskWarning`, `DashboardRiskStatus`, `get_risk_status()`, `get_guardrails()`, `set_guardrails()` |
| `LIVE_TRADING/risk/__init__.py` | Exported new types and functions |
| `DASHBOARD/bridge/server.py` | Added 5 new endpoints, 1 WebSocket, enhanced `/api/state` |

---

## Testing

```bash
# Test positions endpoint
curl http://127.0.0.1:8765/api/positions | jq

# Test trade history
curl "http://127.0.0.1:8765/api/trades/history?limit=10" | jq

# Test CILS stats
curl http://127.0.0.1:8765/api/cils/stats | jq

# Test risk status
curl http://127.0.0.1:8765/api/risk/status | jq

# Test enhanced state
curl http://127.0.0.1:8765/api/state | jq

# Test training WebSocket
websocat ws://127.0.0.1:8765/ws/training
```

---

## Rollback Plan

If issues arise:
1. New endpoints don't break existing ones
2. Stage tracking is additive (engine still works without dashboard)
3. Mock mode ensures bridge runs even without engine
