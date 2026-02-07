# Plan 01: Common Infrastructure

## Overview

Foundation layer providing exceptions, constants, and base classes for the entire LIVE_TRADING module.

## Files to Create

### 1. `LIVE_TRADING/__init__.py`
**Purpose:** Package initialization with repro_bootstrap import

```python
# MUST import repro_bootstrap FIRST for determinism
import TRAINING.common.repro_bootstrap  # noqa: F401

from LIVE_TRADING.common.constants import HORIZONS, FAMILIES
from LIVE_TRADING.common.exceptions import LiveTradingError

__version__ = "0.1.0"
__all__ = ["HORIZONS", "FAMILIES", "LiveTradingError"]
```

### 2. `LIVE_TRADING/common/__init__.py`
**Purpose:** Common subpackage init

```python
from .exceptions import (
    LiveTradingError,
    BrokerError,
    ModelLoadError,
    InferenceError,
    GatingError,
    SizingError,
    RiskError,
)
from .constants import HORIZONS, FAMILIES, DEFAULT_CONFIG

__all__ = [
    "LiveTradingError",
    "BrokerError",
    "ModelLoadError",
    "InferenceError",
    "GatingError",
    "SizingError",
    "RiskError",
    "HORIZONS",
    "FAMILIES",
    "DEFAULT_CONFIG",
]
```

### 3. `LIVE_TRADING/common/exceptions.py`
**Purpose:** Exception hierarchy extending FoxMLError

```python
"""
Live Trading Exceptions
=======================

Exception hierarchy for LIVE_TRADING module.
All exceptions extend FoxMLError for consistency.
"""

from TRAINING.common.exceptions import FoxMLError


class LiveTradingError(FoxMLError):
    """Base exception for all live trading errors."""
    pass


class BrokerError(LiveTradingError):
    """Error communicating with broker."""
    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by broker."""
    def __init__(self, symbol: str, reason: str):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Order rejected for {symbol}: {reason}")


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order."""
    pass


class ModelLoadError(LiveTradingError):
    """Error loading model from artifacts."""
    def __init__(self, family: str, target: str, reason: str):
        self.family = family
        self.target = target
        self.reason = reason
        super().__init__(f"Failed to load {family} model for {target}: {reason}")


class InferenceError(LiveTradingError):
    """Error during model inference."""
    def __init__(self, family: str, symbol: str, reason: str):
        self.family = family
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Inference failed for {family}/{symbol}: {reason}")


class GatingError(LiveTradingError):
    """Error in gating logic."""
    pass


class SizingError(LiveTradingError):
    """Error in position sizing."""
    pass


class RiskError(LiveTradingError):
    """Risk limit violation."""
    pass


class KillSwitchTriggered(RiskError):
    """Kill switch has been triggered."""
    def __init__(self, reason: str, current_value: float, limit: float):
        self.reason = reason
        self.current_value = current_value
        self.limit = limit
        super().__init__(f"Kill switch: {reason} ({current_value:.2f} vs limit {limit:.2f})")


class MaxDrawdownExceeded(KillSwitchTriggered):
    """Maximum drawdown exceeded."""
    def __init__(self, drawdown_pct: float, limit_pct: float):
        super().__init__("max_drawdown", drawdown_pct, limit_pct)


class DailyLossExceeded(KillSwitchTriggered):
    """Daily loss limit exceeded."""
    def __init__(self, loss_pct: float, limit_pct: float):
        super().__init__("daily_loss", loss_pct, limit_pct)


class StaleDataError(LiveTradingError):
    """Data is too old for safe trading."""
    def __init__(self, symbol: str, age_ms: float, max_age_ms: float):
        self.symbol = symbol
        self.age_ms = age_ms
        self.max_age_ms = max_age_ms
        super().__init__(f"Stale data for {symbol}: {age_ms:.0f}ms > {max_age_ms:.0f}ms")


class SpreadTooWideError(LiveTradingError):
    """Spread exceeds maximum threshold."""
    def __init__(self, symbol: str, spread_bps: float, max_bps: float):
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.max_bps = max_bps
        super().__init__(f"Spread too wide for {symbol}: {spread_bps:.1f}bps > {max_bps:.1f}bps")
```

### 4. `LIVE_TRADING/common/constants.py`
**Purpose:** Constants and default configuration values

```python
"""
Live Trading Constants
======================

Central location for all constants used in LIVE_TRADING module.
"""

from typing import Dict, List, Any

# Supported horizons (in order from shortest to longest)
HORIZONS: List[str] = ["5m", "10m", "15m", "30m", "60m", "1d"]

# Horizon to minutes mapping
HORIZON_MINUTES: Dict[str, int] = {
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 390,  # Trading day minutes
}

# Model families from TRAINING (subset commonly used for live trading)
FAMILIES: List[str] = [
    "LightGBM",
    "XGBoost",
    "MLP",
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
    "RewardBased",
    "QuantileLightGBM",
    "NGBoost",
    "Ensemble",
]

# Sequential families that need SeqBufferManager
SEQUENTIAL_FAMILIES: List[str] = [
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
]

# TensorFlow/Keras families
TF_FAMILIES: List[str] = [
    "MLP",
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
]

# Tree-based families (direct predict)
TREE_FAMILIES: List[str] = [
    "LightGBM",
    "XGBoost",
    "QuantileLightGBM",
    "NGBoost",
]

# Barrier target prefixes for gating
BARRIER_TARGETS: Dict[str, str] = {
    "will_peak": "will_peak_5m",
    "will_valley": "will_valley_5m",
    "y_will_peak": "y_will_peak_5m",
    "y_will_valley": "y_will_valley_5m",
}

# Default configuration values (fallbacks if config missing)
DEFAULT_CONFIG: Dict[str, Any] = {
    # Blending
    "ridge_lambda": 0.15,
    "temperature": {
        "5m": 0.75,
        "10m": 0.85,
        "15m": 0.90,
        "30m": 1.0,
        "60m": 1.0,
        "1d": 1.0,
    },

    # Cost model
    "k1_spread": 1.0,
    "k2_volatility": 0.15,
    "k3_impact": 1.0,

    # Barrier gate
    "g_min": 0.2,
    "gamma": 1.0,
    "delta": 0.5,
    "peak_threshold": 0.6,
    "valley_threshold": 0.55,

    # Sizing
    "z_max": 3.0,
    "max_weight": 0.05,
    "gross_target": 0.5,
    "no_trade_band": 0.008,

    # Risk
    "max_daily_loss_pct": 2.0,
    "max_drawdown_pct": 10.0,
    "max_position_pct": 20.0,
    "spread_max_bps": 12.0,
    "quote_age_max_ms": 200.0,

    # Paper broker defaults
    "slippage_bps": 5.0,
    "fee_bps": 1.0,
    "initial_cash": 100_000.0,
}

# Z-score standardization bounds
ZSCORE_CLIP_MIN: float = -3.0
ZSCORE_CLIP_MAX: float = 3.0

# Rolling window for standardization (trading days)
STANDARDIZATION_WINDOW: int = 10

# Minimum IC for model inclusion
MIN_IC_THRESHOLD: float = 0.01

# Freshness decay constants (seconds)
FRESHNESS_TAU: Dict[str, float] = {
    "5m": 150.0,
    "10m": 300.0,
    "15m": 450.0,
    "30m": 900.0,
    "60m": 1800.0,
    "1d": 7200.0,
}

# Capacity participation rate
CAPACITY_KAPPA: float = 0.1

# Order types
ORDER_TYPE_MARKET: str = "market"
ORDER_TYPE_LIMIT: str = "limit"

# Side constants
SIDE_BUY: str = "BUY"
SIDE_SELL: str = "SELL"

# Trade decision results
DECISION_TRADE: str = "TRADE"
DECISION_HOLD: str = "HOLD"
DECISION_BLOCKED: str = "BLOCKED"
```

## Tests

### `LIVE_TRADING/tests/test_common.py`

```python
"""Tests for common infrastructure."""

import pytest
from LIVE_TRADING.common.exceptions import (
    LiveTradingError,
    BrokerError,
    KillSwitchTriggered,
    StaleDataError,
)
from LIVE_TRADING.common.constants import (
    HORIZONS,
    FAMILIES,
    HORIZON_MINUTES,
    DEFAULT_CONFIG,
)


class TestExceptions:
    def test_live_trading_error_is_foxmlerror(self):
        from TRAINING.common.exceptions import FoxMLError
        assert issubclass(LiveTradingError, FoxMLError)

    def test_broker_error_inheritance(self):
        assert issubclass(BrokerError, LiveTradingError)

    def test_kill_switch_triggered(self):
        exc = KillSwitchTriggered("test_reason", 5.0, 2.0)
        assert exc.reason == "test_reason"
        assert exc.current_value == 5.0
        assert exc.limit == 2.0
        assert "test_reason" in str(exc)

    def test_stale_data_error(self):
        exc = StaleDataError("AAPL", 500.0, 200.0)
        assert exc.symbol == "AAPL"
        assert exc.age_ms == 500.0


class TestConstants:
    def test_horizons_ordered(self):
        # Horizons should be shortest to longest
        minutes = [HORIZON_MINUTES[h] for h in HORIZONS]
        assert minutes == sorted(minutes)

    def test_all_horizons_have_minutes(self):
        for h in HORIZONS:
            assert h in HORIZON_MINUTES

    def test_families_not_empty(self):
        assert len(FAMILIES) > 0

    def test_default_config_has_required_keys(self):
        required = ["ridge_lambda", "z_max", "max_daily_loss_pct"]
        for key in required:
            assert key in DEFAULT_CONFIG
```

## SST Compliance Checklist

- [ ] `__init__.py` imports `repro_bootstrap` first
- [ ] Exceptions extend `FoxMLError`
- [ ] No hardcoded config values (use `DEFAULT_CONFIG` as fallback)
- [ ] Constants are typed with `List[str]`, `Dict[str, Any]`, etc.
- [ ] All imports are absolute

## Dependencies

- `TRAINING.common.exceptions.FoxMLError` - Base exception class
- `TRAINING.common.repro_bootstrap` - Determinism setup

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 15 |
| `common/__init__.py` | 25 |
| `common/exceptions.py` | 120 |
| `common/constants.py` | 160 |
| `tests/test_common.py` | 60 |
| **Total** | ~380 |
