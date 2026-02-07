# Dashboard IPC Bridge

## Overview

The IPC Bridge is a Python FastAPI server that connects the Rust TUI dashboard to the Python trading engine. It exposes EventBus events and metrics via HTTP/WebSocket.

**Location**: `DASHBOARD/bridge/server.py`

**Key characteristics**:
- Lightweight bridge - no changes needed to trading engine code
- Graceful degradation - runs in mock mode if trading engine not available
- Logs to file (not stdout) to avoid interfering with TUI
- Auto-started by `bin/foxml` if not running

## API Endpoints

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns status and connection count |
| `/api/metrics` | GET | Current trading metrics (portfolio, P&L, etc.) |
| `/api/state` | GET | Engine state (status, current stage, uptime) |
| `/api/events/recent?count=N` | GET | Recent events (default 100) |
| `/api/control/status` | GET | Control state (paused, kill switch) |
| `/api/control/pause` | POST | Pause trading engine |
| `/api/control/resume` | POST | Resume trading engine |
| `/api/control/kill_switch` | POST | Toggle kill switch |
| `/api/positions` | GET | Position details (symbol, shares, P&L, weight) |
| `/api/trades/history` | GET | Trade audit trail (with filtering) |
| `/api/cils/stats` | GET | CILS optimizer statistics (bandit weights) |
| `/api/risk/status` | GET | Risk metrics (drawdown, exposure, warnings) |
| `/api/training/status` | GET | Training progress summary |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/events` | Real-time trading event streaming |
| `/ws/alpaca` | Real-time Alpaca trade/account updates |
| `/ws/training` | Real-time training progress streaming |

## Response Schemas

### Metrics Response (`/api/metrics`)

```json
{
  "portfolio_value": 100000.0,
  "daily_pnl": 1234.56,
  "cash_balance": 50000.0,
  "positions_count": 5,
  "sharpe_ratio": 1.5,
  "trades_total": 42,
  "cycles_total": 100,
  "errors_total": 2
}
```

### State Response (`/api/state`)

```json
{
  "status": "running",
  "current_stage": "prediction",
  "last_cycle": "2025-01-21T10:30:00Z",
  "uptime_seconds": 3600
}
```

### Control Status Response (`/api/control/status`)

```json
{
  "paused": false,
  "kill_switch_active": false,
  "kill_switch_reason": null,
  "last_updated": "2025-01-21T10:30:00Z"
}
```

### Kill Switch Request (`POST /api/control/kill_switch`)

```json
{
  "action": "enable",
  "reason": "Manual stop for maintenance"
}
```

### Positions Response (`/api/positions`)

```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "shares": 100,
      "entry_price": 150.00,
      "current_price": 155.00,
      "market_value": 15500.00,
      "unrealized_pnl": 500.00,
      "unrealized_pnl_pct": 3.33,
      "weight": 0.15,
      "entry_time": "2025-01-21T09:30:00Z",
      "side": "long"
    }
  ],
  "total_positions": 1,
  "long_count": 1,
  "short_count": 0,
  "total_market_value": 15500.00
}
```

### Risk Status Response (`/api/risk/status`)

```json
{
  "trading_allowed": true,
  "kill_switch_active": false,
  "kill_switch_reason": null,
  "daily_pnl_pct": 0.5,
  "daily_loss_limit_pct": 2.0,
  "drawdown_pct": 1.2,
  "max_drawdown_limit_pct": 5.0,
  "gross_exposure": 0.95,
  "net_exposure": 0.85,
  "max_gross_exposure": 2.0,
  "warnings": [],
  "last_check": "2025-01-21T10:30:00Z"
}
```

### Training Status Response (`/api/training/status`)

```json
{
  "running": true,
  "run_id": "prod_20260121_143000",
  "stage": "training",
  "progress_pct": 45.0,
  "current_target": "ret_5m",
  "events": []
}
```

### Decisions Response (`/api/decisions/recent`)

Returns recent trading decisions with full pipeline traces showing why trades were triggered or blocked.

```json
{
  "decisions": [
    {
      "symbol": "AAPL",
      "decision": "TRADE",
      "horizon": "5m",
      "alpha": 0.0025,
      "target_weight": 0.05,
      "current_weight": 0.0,
      "shares": 100,
      "reason": "alpha_exceeds_threshold",
      "timestamp": "2025-01-21T10:30:00Z",
      "trace": {
        "market_snapshot": {
          "symbol": "AAPL",
          "close": 150.00,
          "spread_bps": 5.2,
          "volatility": 0.25
        },
        "blended_alphas": {
          "5m": 0.0025,
          "15m": 0.0018,
          "60m": 0.0012
        },
        "horizon_scores": {
          "5m": 1.85,
          "15m": 1.20,
          "60m": 0.90
        },
        "selected_horizon": "5m",
        "barrier_gate_result": {
          "allowed": true,
          "p_peak": 0.65,
          "p_valley": 0.35
        },
        "spread_gate_result": {
          "allowed": true,
          "spread_bps": 5.2
        },
        "raw_weight": 0.06,
        "gate_adjusted_weight": 0.052,
        "final_weight": 0.05,
        "risk_checks": {
          "passed": true,
          "reason": "Trade validated"
        }
      }
    }
  ],
  "total_decisions": 150,
  "trade_count": 25,
  "hold_count": 120,
  "blocked_count": 5
}
```

Query parameters:
- `limit` (int): Max decisions to return (default 50)
- `symbol` (str): Filter by symbol
- `decision_type` (str): Filter by TRADE, HOLD, or BLOCKED
- `include_trace` (bool): Include full trace (default true, set false for smaller payload)

### Latest Decision Response (`/api/decisions/{symbol}/latest`)

Get the most recent decision for a specific symbol with human-readable explanation.

```json
{
  "decision": { /* same as above */ },
  "explanation": "Decision: TRADE for AAPL\nReason: alpha_exceeds_threshold\n\nMarket Context:\n  Price: 150.00\n  Spread: 5.2 bps\n  Volatility: 25.00%\n\nHorizon Analysis:\n  5m: alpha=25.0bps, score=1.85 <-- SELECTED\n  15m: alpha=18.0bps, score=1.20\n  60m: alpha=12.0bps, score=0.90\n..."
}
```

## Running the Bridge

### Automatic (via launcher)

```bash
bin/foxml  # Auto-starts bridge if not running
```

### Manual

```bash
cd DASHBOARD/bridge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Bridge runs on `http://127.0.0.1:8765` by default.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Trading Engine (LIVE_TRADING/)                             │
│  ├── observability/events.py  (EventBus)                   │
│  └── observability/metrics.py (MetricsRegistry)            │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Python imports)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  IPC Bridge (DASHBOARD/bridge/server.py)                    │
│  ├── FastAPI app                                            │
│  ├── EventBus subscriber (on_event callback)               │
│  ├── Event queue (asyncio.Queue)                           │
│  ├── Control state file (control_state.json)               │
│  └── WebSocket connections list                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ (HTTP/WebSocket)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Rust TUI (DASHBOARD/dashboard/)                            │
│  └── src/api/client.rs (reqwest HTTP + tokio-tungstenite)  │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Adding a New Endpoint

```python
@app.get("/api/my_endpoint")
async def my_endpoint() -> Dict[str, Any]:
    """
    Endpoint description.

    Returns:
        Response data
    """
    # Implementation
    return {"key": "value"}
```

### Adding a New Metric

1. Add to metrics registry in `LIVE_TRADING/observability/metrics.py`:
```python
my_metric = Gauge("my_metric", "Description")
```

2. Expose in bridge (`/api/metrics`):
```python
return {
    # ... existing metrics
    "my_metric": metrics.my_metric.get(),
}
```

3. Add to Rust client (`src/api/metrics.rs`):
```rust
pub struct Metrics {
    // ... existing fields
    pub my_metric: f64,
}
```

### Adding a New Event Type

Events flow automatically if they're published to EventBus:

1. Publish in trading engine:
```python
from LIVE_TRADING.observability import events
events.emit(EventType.MY_EVENT, {"data": "value"})
```

2. Events appear automatically in:
   - WebSocket stream
   - `/api/events/recent`

3. Handle in Rust (`src/api/events.rs`):
```rust
match event.event_type.as_str() {
    "MY_EVENT" => { /* handle */ }
    // ...
}
```

### Adding Control Functionality

1. Add state field:
```python
def _read_control_state() -> Dict[str, Any]:
    return {
        # ... existing fields
        "my_control": False,
    }
```

2. Add endpoint:
```python
@app.post("/api/control/my_action")
async def my_action() -> Dict[str, Any]:
    state = _read_control_state()
    state["my_control"] = True
    _write_control_state(state)
    return {"status": "ok"}
```

3. Trading engine reads state file to respond to controls.

## Logging

Bridge logs to file to avoid interfering with TUI:

- Log file: `state/logs/bridge.log` (or `$FOXML_STATE_DIR/logs/bridge.log`)
- uvicorn access logs disabled
- Application logs go to file handler

## Mock Mode

When trading engine observability is not available:
- `OBSERVABILITY_AVAILABLE = False`
- `/api/metrics` returns zero values
- `/api/events/recent` returns empty list
- WebSocket connects but sends only pings
- Control endpoints still work (file-based)

## Dependencies

`requirements.txt`:
```
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
```

## Files

| File | Purpose |
|------|---------|
| `DASHBOARD/bridge/server.py` | Main FastAPI server |
| `DASHBOARD/bridge/requirements.txt` | Python dependencies |
| `state/control_state.json` | Control state persistence |
| `state/logs/bridge.log` | Bridge log file |

## Client Integration (Rust)

The Rust TUI uses `reqwest` for HTTP and `tokio-tungstenite` for WebSocket:

```rust
// HTTP client
let response = reqwest::get("http://127.0.0.1:8765/api/metrics")
    .await?
    .json::<Metrics>()
    .await?;

// WebSocket
let (ws_stream, _) = connect_async("ws://127.0.0.1:8765/ws/events").await?;
```

See `DASHBOARD/dashboard/src/api/` for client implementation.
