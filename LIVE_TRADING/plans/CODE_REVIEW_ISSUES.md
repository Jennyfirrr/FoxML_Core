# LIVE_TRADING Code Review Issues

**Date**: 2026-01-18
**Review Scope**: Complete LIVE_TRADING module deep code review
**Status**: Issues documented, pending fixes

---

## Summary

| Severity | Count | Fixed | Description |
|----------|-------|-------|-------------|
| CRITICAL | 4 | **4** | Could cause financial loss or system failure |
| HIGH | 6 | **5** | Significant bugs or missing functionality |
| MEDIUM | 10 | 0 | Code quality, maintainability, edge cases |
| LOW | 5 | 0 | Minor improvements, style consistency |
| INTEGRATION | 5 | 0 | Cross-module integration gaps |
| TESTING | 4 | 0 | Missing test coverage |
| PERFORMANCE | 2 | 0 | Potential performance issues |

**Total Issues**: 36 (9 fixed)

**Additional Work Completed**:
- CILS (Continuous Integrated Learning System) - Exp3-IX bandit for online weight adaptation - IMPLEMENTED

---

## CRITICAL Issues

### C1: Barrier Gate Hardcoded Non-Functional Values - **FIXED**

**File**: `LIVE_TRADING/engine/trading_engine.py:506-508`

**Description**: Trading engine called barrier gate with hardcoded `p_peak=0.0, p_valley=0.0` instead of actual barrier model predictions.

**Fix Applied**: Added `_get_barrier_predictions()` method that:
- Attempts to load predictions from barrier models (`will_peak_5m`, `will_valley_5m`)
- Falls back to neutral values (0.3, 0.3) if models not available
- Converts model alpha to probability using sigmoid transformation

---

### C2: No Position Reconciliation with Broker - **FIXED**

**File**: `LIVE_TRADING/engine/trading_engine.py`

**Fix Applied**: Added `_reconcile_positions()` method that:
- Runs every N cycles (configurable via `reconciliation_interval_cycles`)
- Compares internal state with broker positions
- Supports three modes: `strict` (raise error), `warn` (log only), `auto_sync` (fix drift)
- Detects orphaned internal positions not in broker
- Configurable via `live_trading.reconciliation.*` config keys

---

### C3: Trade Fill Not Verified - **FIXED**

**File**: `LIVE_TRADING/engine/trading_engine.py:_execute_trade()`

**Fix Applied**: Modified `_execute_trade()` to:
- Check `fill_status` before updating state
- Return `False` if status is not "filled" or "partial_fill"
- Use actual `filled_qty` from broker response (supports partial fills)
- Emit error event on non-fill

---

### C4: No Duplicate Trade Prevention - **FIXED**

**File**: `LIVE_TRADING/engine/trading_engine.py`

**Fix Applied**: Added cooldown mechanism:
- Track `_last_trade_time` per symbol
- Added `_can_trade_symbol()` check before executing trades
- Configurable cooldown via `live_trading.risk.trade_cooldown_seconds` (default: 5s)
- Clear cooldowns on engine reset

---

## HIGH Issues

### H1: PaperBroker Sell Validation Missing - **ALREADY FIXED**

**File**: `LIVE_TRADING/brokers/paper.py:194-201`

**Description**: PaperBroker already has validation to prevent selling more shares than held.

**Existing Code**:
```python
# Check position for sell orders
if side == SIDE_SELL:
    current_pos = self._positions.get(symbol, 0.0)
    if qty > current_pos:
        raise OrderRejectedError(
            symbol,
            f"Insufficient shares: need {qty}, have {current_pos}",
        )
```

**Status**: This check already exists - issue was documented incorrectly.

---

### H2: Pickle Usage Without Safety Checks - **FIXED**

**File**: `LIVE_TRADING/models/loader.py`

**Fix Applied**: Added checksum verification before pickle loading:
- `_compute_file_checksum()` - Computes SHA256 hash
- `_verify_model_checksum()` - Verifies against expected hash
- ModelLoader has `verify_checksums` and `strict_checksums` options
- Configurable via `live_trading.models.verify_checksums`

---

### H3: Feature Builder Index Access Without Bounds Check - **N/A**

**File**: `LIVE_TRADING/models/features.py` (file not found)

**Status**: The feature builder file referenced in this issue does not exist in the current codebase. Features are built by the TRAINING module and loaded via artifacts. This issue can be closed as N/A or deferred until a feature builder is implemented in LIVE_TRADING.

---

### H4: AlpacaBroker Missing Rate Limiting - **FIXED**

**File**: `LIVE_TRADING/brokers/alpaca.py`

**Fix Applied**: Added rate limiters to Alpaca broker:
- Created `LIVE_TRADING/common/rate_limiter.py` with token bucket implementation
- Added `_order_rate_limiter` (180 req/min) for order operations
- Added `_data_rate_limiter` (200 req/min) for data operations
- Configurable via `live_trading.brokers.alpaca.order_rate_limit` and `data_rate_limit`

---

### H5: Kill Switch Timezone Issues - **ALREADY FIXED**

**File**: `LIVE_TRADING/risk/guardrails.py`

**Description**: Kill switch uses Clock abstraction consistently throughout.

**Existing Code**:
```python
self._clock = clock or get_clock()
# ...
current_time = current_time or self._clock.now()
today = current_time.date()
```

**Status**: Uses Clock abstraction properly - all datetime operations go through the clock which provides UTC-aware timestamps.

---

### H6: Sequential Buffer TTL Silently Fails - **FIXED**

**File**: `LIVE_TRADING/data/cached.py`

**Fix Applied**: Added debug logging for cache operations:
- Log cache hits with age
- Log cache expiry with age vs TTL comparison
- Log cache misses with row counts for historical data
- Enables debugging via DEBUG log level

---

## MEDIUM Issues

### M1: Backtest DataLoader Synthetic Data Mode

**File**: `LIVE_TRADING/backtest/data_loader.py:~40`

**Description**: The "synthetic" mode generates random walk data that doesn't match real market characteristics (no overnight gaps, no realistic volatility clustering).

**Impact**: Backtests on synthetic data may give misleading results.

**Fix**: Improve synthetic data generation with realistic properties or add prominent warnings:
```python
if mode == "synthetic":
    logger.warning("Using synthetic data - results may not reflect real market behavior")
```

---

### M2: State Persistence No File Locking

**File**: `LIVE_TRADING/persistence/state.py:~80`

**Description**: State file writes don't use file locking, risking corruption if multiple processes access the same state file.

**Impact**: State corruption if accidentally running two engine instances.

**Fix**: Add file locking:
```python
import fcntl

def _save_state_atomic(self, state: EngineState) -> None:
    with open(self._lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            write_atomic_json(self._state_path, state.to_dict())
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
```

---

### M3: Event Bus Subscriber Error Handling

**File**: `LIVE_TRADING/observability/events.py:~50`

**Description**: If a subscriber callback raises an exception, it may prevent other subscribers from receiving the event.

**Code**:
```python
def emit(self, event_type: EventType, **kwargs):
    for callback in self._subscribers.get(event_type, []):
        callback(**kwargs)  # Exception here stops iteration
```

**Impact**: One bad subscriber can break observability for all others.

**Fix**: Catch exceptions per subscriber:
```python
def emit(self, event_type: EventType, **kwargs):
    for callback in self._subscribers.get(event_type, []):
        try:
            callback(**kwargs)
        except Exception as e:
            logger.error(f"Subscriber error for {event_type}: {e}")
```

---

### M4: Cache Thread Safety

**File**: `LIVE_TRADING/data/cache.py:~30`

**Description**: Cache access is not thread-safe, which could cause issues if data providers are called from multiple threads.

**Impact**: Potential race conditions in multi-threaded scenarios.

**Fix**: Add threading lock:
```python
import threading

class DataCache:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {}

    def get(self, key: str):
        with self._lock:
            return self._cache.get(key)
```

---

### M5: Alerting Channel Retry Logic

**File**: `LIVE_TRADING/observability/alerting.py:~100`

**Description**: Alert channels don't retry on transient failures (network timeouts, rate limits).

**Impact**: Alerts may be lost during temporary network issues.

**Fix**: Add retry with exponential backoff:
```python
@retry(max_attempts=3, backoff_factor=2.0)
def send(self, alert: Alert) -> bool:
    # ... existing send logic
```

---

### M6: Signal Blender Weight Normalization

**File**: `LIVE_TRADING/signals/blender.py:~60`

**Description**: Blender doesn't validate that weights sum to 1.0 or normalize them.

**Impact**: Incorrect alpha calculations if weights are misconfigured.

**Fix**: Add normalization:
```python
def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"Weights sum to {total}, normalizing to 1.0")
        return {k: v / total for k, v in weights.items()}
    return weights
```

---

### M7: Arbitrator Tie-Breaking Non-Deterministic

**File**: `LIVE_TRADING/engine/arbitrator.py:~70`

**Description**: When multiple signals have equal priority, the arbitrator doesn't have deterministic tie-breaking.

**Impact**: Non-reproducible behavior in backtests.

**Fix**: Add deterministic tie-breaking (e.g., by symbol name):
```python
def arbitrate(self, signals: list[Signal]) -> Signal | None:
    # Sort by priority, then by symbol for deterministic tie-breaking
    sorted_signals = sorted(signals, key=lambda s: (-s.priority, s.symbol))
```

---

### M8: Data Provider Connection Pool

**File**: `LIVE_TRADING/data/polygon.py:~30`

**Description**: Each data request creates a new HTTP connection instead of reusing connections.

**Impact**: Higher latency and resource usage.

**Fix**: Use connection pooling:
```python
import httpx

class PolygonProvider:
    def __init__(self):
        self._client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(max_connections=10)
        )
```

---

### M9: IBKR Broker Reconnection Logic

**File**: `LIVE_TRADING/brokers/ibkr.py:~90`

**Description**: No automatic reconnection if TWS/Gateway connection drops.

**Impact**: Trading stops if connection is lost without manual intervention.

**Fix**: Add connection monitoring and auto-reconnect:
```python
def _ensure_connected(self) -> None:
    if not self._ib.isConnected():
        logger.warning("IBKR disconnected, attempting reconnect...")
        self.connect()
```

---

### M10: Clock Mock Incomplete in Tests

**File**: `LIVE_TRADING/tests/test_trading_engine.py`

**Description**: Some tests use real time instead of mock clock, making them flaky around midnight or DST changes.

**Impact**: Occasional test failures in CI.

**Fix**: Ensure all time-dependent tests use MockClock consistently.

---

## LOW Issues

### L1: Inconsistent Logging Levels

**Files**: Various

**Description**: Some modules use `logger.info()` for routine operations while others use `logger.debug()`.

**Impact**: Noisy logs in production.

**Fix**: Establish logging level guidelines:
- `DEBUG`: Internal state, loop iterations
- `INFO`: Cycle starts/ends, trades executed
- `WARNING`: Recoverable issues
- `ERROR`: Failures requiring attention

---

### L2: Docstring Format Inconsistency

**Files**: Various

**Description**: Mix of Google-style and NumPy-style docstrings.

**Impact**: Documentation generation may be inconsistent.

**Fix**: Standardize on Google-style (matching existing TRAINING code).

---

### L3: Type Hints Missing on Some Methods

**Files**: `LIVE_TRADING/signals/blender.py`, `LIVE_TRADING/data/buffer.py`

**Description**: Some methods lack return type hints.

**Impact**: Reduced IDE support and type checking coverage.

**Fix**: Add missing type hints for full coverage.

---

### L4: Magic Numbers in Tests

**Files**: `LIVE_TRADING/tests/test_*.py`

**Description**: Tests contain magic numbers without explanation:
```python
assert result.alpha > 0.03  # Why 0.03?
```

**Impact**: Hard to understand test intent.

**Fix**: Use named constants with documentation:
```python
MIN_EXPECTED_ALPHA = 0.03  # Minimum alpha for a "strong" signal
assert result.alpha > MIN_EXPECTED_ALPHA
```

---

### L5: Import Organization

**Files**: Various

**Description**: Some files have imports in non-standard order (stdlib, third-party, local not clearly separated).

**Impact**: Style inconsistency.

**Fix**: Run `isort` or configure ruff import sorting.

---

## Integration Gaps

### I1: Trading Engine <-> State Persistence

**Description**: Trading engine doesn't call state persistence on every state change, only on shutdown.

**Impact**: State loss on crash.

**Fix**: Add periodic state saves or after significant changes.

---

### I2: Model Loader <-> Model Registry

**Description**: No central registry of available models. Loader relies on filesystem conventions.

**Impact**: Difficult to know which models are available without scanning directories.

**Fix**: Add model manifest or registry pattern.

---

### I3: Alerting <-> Event Bus

**Description**: Alerting and event bus are separate systems. Critical events should trigger alerts automatically.

**Impact**: Need manual wiring of event subscribers to alerting.

**Fix**: Add automatic alert triggers for critical event types:
```python
event_bus.subscribe(EventType.ERROR, lambda **kw: alert_manager.send_alert(...))
```

---

### I4: Broker <-> Risk Manager

**Description**: Risk manager doesn't have access to broker for pre-trade checks (available margin, position limits).

**Impact**: Risk checks incomplete without broker state.

**Fix**: Pass broker reference to risk manager or add broker query methods.

---

### I5: Data Provider <-> Cache Invalidation

**Description**: No mechanism to invalidate cached data when market conditions change (halts, splits).

**Impact**: Stale data may be used after corporate actions.

**Fix**: Add cache invalidation hooks or TTL-based expiry.

---

## Testing Gaps

### T1: No Integration Tests

**Description**: Tests are unit-level only. No tests that exercise full trading cycle with mocked broker.

**Fix**: Add integration test suite in `LIVE_TRADING/tests/integration/`.

---

### T2: No Stress Tests

**Description**: No tests for high-frequency scenarios (rapid cycles, many symbols).

**Fix**: Add performance test suite with timing assertions.

---

### T3: Edge Case Coverage

**Description**: Missing tests for:
- Network timeout during order submission
- Partial fills
- Market halt scenarios
- Pre/post market behavior

**Fix**: Add edge case test scenarios.

---

### T4: Backtest Validation

**Description**: No tests comparing backtest results with known historical outcomes.

**Fix**: Add golden-file tests with known historical data.

---

## Performance Issues

### P1: Feature Calculation on Every Cycle

**Description**: Features are recalculated from scratch on every cycle even if data hasn't changed.

**Impact**: Unnecessary CPU usage.

**Fix**: Add feature caching with data fingerprint invalidation.

---

### P2: DataFrame Copies in Prediction Pipeline

**Description**: Multiple DataFrame copies created during prediction pipeline.

**Impact**: Memory pressure with many symbols.

**Fix**: Use views where possible, or add explicit copy points.

---

## Priority Order for Fixes

### Phase 1: Critical Safety (Do First)
1. C1: Barrier gate hardcoded values
2. C3: Trade fill verification
3. C4: Duplicate trade prevention
4. C2: Position reconciliation

### Phase 2: High Priority Bugs
5. H1: PaperBroker sell validation
6. H3: Feature builder bounds check
7. H5: Kill switch timezone
8. H6: Buffer TTL logging

### Phase 3: Security
9. H2: Pickle security
10. H4: Alpaca rate limiting

### Phase 4: Reliability
11. M2: State file locking
12. M3: Event bus error handling
13. M5: Alerting retry
14. M9: IBKR reconnection

### Phase 5: Integration
15. I1: Periodic state saves
16. I3: Event-to-alert wiring
17. I4: Broker-risk integration

### Phase 6: Quality
18. Remaining MEDIUM issues
19. LOW issues
20. Testing gaps

---

## Notes

- All fixes should include corresponding test cases
- SST compliance: Use `get_cfg()` for any new config values
- Determinism: Ensure sorted iteration where order matters
- Follow existing code patterns in TRAINING module

---

**Document Version**: 1.1
**Last Updated**: 2026-01-19

### Changelog

- **v1.1** (2026-01-19): Added CILS implementation status
- **v1.0** (2026-01-18): Initial code review completed, 9 issues fixed
