# Phase 2: Bridge Protocol Bugs

**Master plan**: `dashboard-fixes-round2-master.md`
**Status**: Complete
**Scope**: 1 file modified (bridge/server.py)
**Depends on**: Nothing

---

## Context

The Python IPC bridge has four protocol/data issues: inconsistent timestamps, a stubbed Sharpe ratio, an incomplete error schema, and no authentication on destructive endpoints.

---

## 2a: Fix Timezone Inconsistency in Control Endpoints

**File**: `DASHBOARD/bridge/server.py` (lines ~990, ~1010, ~1044)

### Current Problem
Control endpoints (pause, resume, kill_switch) use `datetime.now().isoformat()` (naive/local timezone), while all other endpoints use `datetime.now(timezone.utc).isoformat()` (UTC-aware).

This creates mixed timestamp formats:
- UTC: `"2026-02-09T15:30:00+00:00"`
- Naive: `"2026-02-09T10:30:00"` (no offset)

### Fix
Replace all 3 instances of `datetime.now().isoformat()` with `datetime.now(timezone.utc).isoformat()`.

### Affected Lines
Search for `datetime.now().isoformat()` (without `timezone.utc`) in:
- `pause_engine()`
- `resume_engine()`
- `toggle_kill_switch()`

### Changes
```python
# Before
"paused_at": datetime.now().isoformat()

# After
"paused_at": datetime.now(timezone.utc).isoformat()
```

---

## 2b: Implement Sharpe Ratio Calculation

**File**: `DASHBOARD/bridge/server.py` (line ~268, in `get_metrics()`)

### Current State
```python
"sharpe_ratio": None,  # TODO: Calculate from metrics
```

### Fix
Calculate Sharpe ratio from the P&L history already tracked by the bridge. The bridge stores trade history and position data; we need to compute daily returns.

### Implementation
1. Maintain a rolling window of daily P&L snapshots (already partially available from metrics)
2. Calculate: `sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)`
3. Return `None` if insufficient data (< 2 data points)

### Changes
Add a helper method and P&L tracking:

```python
class DashboardServer:
    def __init__(self):
        # ... existing init ...
        self._pnl_history: list[float] = []
        self._last_pnl_snapshot: float | None = None

    def _update_pnl_history(self, current_pnl: float):
        """Track daily P&L for Sharpe calculation."""
        if self._last_pnl_snapshot is not None:
            daily_return = current_pnl - self._last_pnl_snapshot
            self._pnl_history.append(daily_return)
            # Keep rolling window of 252 trading days
            if len(self._pnl_history) > 252:
                self._pnl_history.pop(0)
        self._last_pnl_snapshot = current_pnl

    def _calculate_sharpe(self) -> float | None:
        """Calculate annualized Sharpe ratio from P&L history."""
        if len(self._pnl_history) < 2:
            return None
        import statistics
        mean_ret = statistics.mean(self._pnl_history)
        std_ret = statistics.stdev(self._pnl_history)
        if std_ret == 0:
            return None
        return (mean_ret / std_ret) * (252 ** 0.5)
```

### Notes
- The bridge already fetches P&L each time `/api/metrics` is called
- Call `_update_pnl_history()` during each metrics fetch
- For initial implementation, track per-poll-interval returns (not true daily)
- The Sharpe will be approximate but better than `None`

---

## 2c: Fix `get_metrics()` Error Path Schema

**File**: `DASHBOARD/bridge/server.py` (lines ~273-281)

### Current Problem
When an exception occurs in `get_metrics()`, the error fallback returns only ~4 fields:
```python
except Exception:
    return {
        "portfolio_value": 0.0,
        "daily_pnl": 0.0,
        "cash_balance": 0.0,
        "positions_count": 0,
    }
```

But the success path returns many more fields. The Rust client may interpret zero values as real data.

### Fix
Return the complete schema with all expected fields set to null/zero, plus an `"error"` field:
```python
except Exception as e:
    logger.error("Failed to get metrics: %s", e)
    return {
        "portfolio_value": 0.0,
        "daily_pnl": 0.0,
        "cash_balance": 0.0,
        "positions_count": 0,
        "sharpe_ratio": None,
        "total_trades": 0,
        "win_rate": None,
        "max_drawdown": None,
        "exposure": 0.0,
        "error": str(e),
    }
```

### Notes
- Check the full success-path schema to ensure all fields are covered
- Add `"error"` field so client can distinguish real zeros from error fallback
- Optionally return HTTP 503 instead of 200 for error case

---

## 2d: Add Bearer Token Auth to Control Endpoints

**File**: `DASHBOARD/bridge/server.py`

### Current Problem
Control endpoints (`/api/control/kill_switch`, `/api/control/pause`, `/api/control/resume`) have no authentication. Any local process can hit them.

### Implementation
1. Generate a random token on bridge startup
2. Write token to a file readable by the dashboard (`/tmp/foxml_bridge_token`)
3. Require `Authorization: Bearer <token>` header on all POST control endpoints
4. GET endpoints (metrics, state, positions) remain open (read-only)

### Changes

**Bridge side** (`server.py`):
```python
import secrets

class DashboardServer:
    def __init__(self):
        self._auth_token = secrets.token_urlsafe(32)
        # Write token for dashboard to read
        token_path = Path("/tmp/foxml_bridge_token")
        token_path.write_text(self._auth_token)
        token_path.chmod(0o600)

    def _verify_auth(self, request) -> bool:
        auth = request.headers.get("Authorization", "")
        return auth == f"Bearer {self._auth_token}"
```

**Rust client side** (`api/client.rs`):
```rust
impl DashboardClient {
    fn load_auth_token() -> Option<String> {
        std::fs::read_to_string("/tmp/foxml_bridge_token").ok()
    }

    // Add Authorization header to POST requests
}
```

### Notes
- Token file should be `chmod 600` (owner-only)
- Only protect POST endpoints — GET endpoints are read-only and safe
- Dashboard reads the token file on startup
- If token file doesn't exist, fall back to unauthenticated (for development)

---

## Verification

- [ ] All timestamps from bridge are UTC-aware (include `+00:00` suffix)
- [ ] `/api/metrics` returns numeric `sharpe_ratio` after sufficient data points
- [ ] `/api/metrics` error path returns complete schema with all expected fields
- [ ] POST to `/api/control/kill_switch` without auth token returns 401
- [ ] POST to `/api/control/kill_switch` with valid auth token succeeds
- [ ] Dashboard reads token and passes it in requests
- [ ] GET endpoints still work without auth token
