# Plan 09: Risk Management

## Overview

Kill switches and risk guardrails that protect against excessive losses. Includes daily loss limits, drawdown monitoring, and position limits.

## Risk Gates

### Kill Switches
- **Daily Loss:** Stop trading if daily loss > 2%
- **Max Drawdown:** Stop trading if drawdown > 10%
- **Position Size:** Cap any position at 20% of portfolio

### Data Quality Gates
- **Spread Gate:** Reject if spread > 12 bps
- **Quote Age:** Reject if quote age > 200ms
- **Latency:** Warn if inference-to-order > 2s

## Files to Create

### 1. `LIVE_TRADING/risk/__init__.py`

```python
from .guardrails import RiskGuardrails, RiskStatus
from .drawdown import DrawdownMonitor
from .exposure import ExposureTracker

__all__ = ["RiskGuardrails", "RiskStatus", "DrawdownMonitor", "ExposureTracker"]
```

### 2. `LIVE_TRADING/risk/drawdown.py`

```python
"""
Drawdown Monitoring
===================

Tracks portfolio drawdown and triggers alerts.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import MaxDrawdownExceeded

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Current drawdown state."""
    peak_value: float
    current_value: float
    drawdown_pct: float
    peak_date: datetime
    is_exceeded: bool


class DrawdownMonitor:
    """
    Monitors portfolio drawdown.
    """

    def __init__(
        self,
        max_drawdown_pct: float | None = None,
        history_size: int = 252,
    ):
        self.max_drawdown_pct = max_drawdown_pct or get_cfg(
            "live_trading.risk.max_drawdown_pct",
            default=DEFAULT_CONFIG["max_drawdown_pct"],
        )
        self.history_size = history_size

        self._peak_value: float = 0.0
        self._peak_date: Optional[datetime] = None
        self._value_history: deque = deque(maxlen=history_size)

        logger.info(f"DrawdownMonitor: max_drawdown={self.max_drawdown_pct}%")

    def update(self, portfolio_value: float, timestamp: datetime | None = None) -> DrawdownState:
        """
        Update with new portfolio value.

        Args:
            portfolio_value: Current portfolio value
            timestamp: Current timestamp

        Returns:
            Current drawdown state
        """
        timestamp = timestamp or datetime.now()
        self._value_history.append((timestamp, portfolio_value))

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._peak_date = timestamp

        # Calculate drawdown
        drawdown = 0.0
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value * 100

        is_exceeded = drawdown > self.max_drawdown_pct

        if is_exceeded:
            logger.warning(f"Max drawdown exceeded: {drawdown:.2f}% > {self.max_drawdown_pct}%")

        return DrawdownState(
            peak_value=self._peak_value,
            current_value=portfolio_value,
            drawdown_pct=drawdown,
            peak_date=self._peak_date,
            is_exceeded=is_exceeded,
        )

    def check_or_raise(self, portfolio_value: float) -> None:
        """Check drawdown and raise if exceeded."""
        state = self.update(portfolio_value)
        if state.is_exceeded:
            raise MaxDrawdownExceeded(state.drawdown_pct, self.max_drawdown_pct)

    def reset(self, initial_value: float) -> None:
        """Reset monitor with new initial value."""
        self._peak_value = initial_value
        self._peak_date = datetime.now()
        self._value_history.clear()
```

### 3. `LIVE_TRADING/risk/exposure.py`

```python
"""
Exposure Tracking
=================

Tracks gross and net portfolio exposure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items
from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ExposureState:
    """Current exposure state."""
    gross_exposure: float  # Sum of |weights|
    net_exposure: float    # Sum of weights
    long_exposure: float   # Sum of positive weights
    short_exposure: float  # Sum of negative weights (absolute)
    position_count: int


class ExposureTracker:
    """
    Tracks portfolio exposure.
    """

    def __init__(
        self,
        max_gross_exposure: float | None = None,
        max_position_pct: float | None = None,
    ):
        self.max_gross = max_gross_exposure or get_cfg(
            "live_trading.risk.max_gross_exposure",
            default=1.0,  # 100% gross
        )
        self.max_position_pct = max_position_pct or get_cfg(
            "live_trading.risk.max_position_pct",
            default=DEFAULT_CONFIG["max_position_pct"],
        ) / 100

        logger.info(f"ExposureTracker: max_gross={self.max_gross}, max_pos={self.max_position_pct}")

    def calculate_exposure(self, weights: Dict[str, float]) -> ExposureState:
        """
        Calculate current exposure.

        Args:
            weights: Position weights

        Returns:
            ExposureState
        """
        gross = 0.0
        net = 0.0
        long_exp = 0.0
        short_exp = 0.0

        for symbol, weight in sorted_items(weights):
            gross += abs(weight)
            net += weight
            if weight > 0:
                long_exp += weight
            else:
                short_exp += abs(weight)

        return ExposureState(
            gross_exposure=gross,
            net_exposure=net,
            long_exposure=long_exp,
            short_exposure=short_exp,
            position_count=len(weights),
        )

    def validate_position(self, symbol: str, weight: float) -> bool:
        """Check if position is within limits."""
        if abs(weight) > self.max_position_pct:
            logger.warning(f"Position {symbol} weight {weight:.2%} exceeds max {self.max_position_pct:.2%}")
            return False
        return True

    def validate_portfolio(self, weights: Dict[str, float]) -> bool:
        """Validate all positions and gross exposure."""
        exposure = self.calculate_exposure(weights)

        if exposure.gross_exposure > self.max_gross:
            logger.warning(f"Gross exposure {exposure.gross_exposure:.2%} exceeds max {self.max_gross:.2%}")
            return False

        for symbol, weight in sorted_items(weights):
            if not self.validate_position(symbol, weight):
                return False

        return True
```

### 4. `LIVE_TRADING/risk/guardrails.py`

```python
"""
Risk Guardrails
===============

Main risk management coordinator with kill switches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import (
    RiskError,
    KillSwitchTriggered,
    DailyLossExceeded,
    MaxDrawdownExceeded,
)
from .drawdown import DrawdownMonitor
from .exposure import ExposureTracker

logger = logging.getLogger(__name__)


@dataclass
class RiskStatus:
    """Current risk status."""
    is_trading_allowed: bool
    daily_pnl_pct: float
    drawdown_pct: float
    gross_exposure: float
    kill_switch_reason: Optional[str] = None


class RiskGuardrails:
    """
    Main risk management with kill switches.
    """

    def __init__(
        self,
        max_daily_loss_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        initial_capital: float = 100_000.0,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct or get_cfg(
            "live_trading.risk.max_daily_loss_pct",
            default=DEFAULT_CONFIG["max_daily_loss_pct"],
        )

        self.drawdown_monitor = DrawdownMonitor(max_drawdown_pct=max_drawdown_pct)
        self.exposure_tracker = ExposureTracker()

        # Daily P&L tracking
        self._initial_capital = initial_capital
        self._day_start_value = initial_capital
        self._current_date: Optional[date] = None
        self._kill_switch_active = False
        self._kill_reason: Optional[str] = None

        logger.info(f"RiskGuardrails: max_daily_loss={self.max_daily_loss_pct}%")

    def check_trading_allowed(
        self,
        portfolio_value: float,
        weights: Dict[str, float],
        current_time: datetime | None = None,
    ) -> RiskStatus:
        """
        Check if trading is allowed.

        Args:
            portfolio_value: Current portfolio value
            weights: Current position weights
            current_time: Current timestamp

        Returns:
            RiskStatus
        """
        current_time = current_time or datetime.now()
        today = current_time.date()

        # Reset daily tracking on new day
        if self._current_date != today:
            self._day_start_value = portfolio_value
            self._current_date = today
            self._kill_switch_active = False
            self._kill_reason = None

        # Calculate daily P&L
        daily_pnl_pct = 0.0
        if self._day_start_value > 0:
            daily_pnl_pct = (portfolio_value - self._day_start_value) / self._day_start_value * 100

        # Check drawdown
        dd_state = self.drawdown_monitor.update(portfolio_value, current_time)

        # Check exposure
        exp_state = self.exposure_tracker.calculate_exposure(weights)

        # Check kill switches
        if not self._kill_switch_active:
            # Daily loss check
            if daily_pnl_pct < -self.max_daily_loss_pct:
                self._kill_switch_active = True
                self._kill_reason = f"daily_loss: {daily_pnl_pct:.2f}%"
                logger.critical(f"KILL SWITCH: Daily loss {daily_pnl_pct:.2f}%")

            # Drawdown check
            if dd_state.is_exceeded:
                self._kill_switch_active = True
                self._kill_reason = f"drawdown: {dd_state.drawdown_pct:.2f}%"
                logger.critical(f"KILL SWITCH: Drawdown {dd_state.drawdown_pct:.2f}%")

        return RiskStatus(
            is_trading_allowed=not self._kill_switch_active,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=dd_state.drawdown_pct,
            gross_exposure=exp_state.gross_exposure,
            kill_switch_reason=self._kill_reason,
        )

    def validate_trade(
        self,
        symbol: str,
        target_weight: float,
        current_weights: Dict[str, float],
    ) -> bool:
        """
        Validate a proposed trade.

        Args:
            symbol: Symbol to trade
            target_weight: Target weight
            current_weights: Current weights

        Returns:
            True if trade is allowed
        """
        if self._kill_switch_active:
            logger.warning("Trade rejected: kill switch active")
            return False

        if not self.exposure_tracker.validate_position(symbol, target_weight):
            return False

        # Check new portfolio exposure
        new_weights = dict(current_weights)
        new_weights[symbol] = target_weight
        if not self.exposure_tracker.validate_portfolio(new_weights):
            return False

        return True

    def reset(self, initial_value: float) -> None:
        """Reset all risk tracking."""
        self._initial_capital = initial_value
        self._day_start_value = initial_value
        self._current_date = None
        self._kill_switch_active = False
        self._kill_reason = None
        self.drawdown_monitor.reset(initial_value)
```

## Tests

### `LIVE_TRADING/tests/test_risk.py`

```python
"""Tests for risk management."""

import pytest
from datetime import datetime

from LIVE_TRADING.risk.drawdown import DrawdownMonitor, DrawdownState
from LIVE_TRADING.risk.exposure import ExposureTracker
from LIVE_TRADING.risk.guardrails import RiskGuardrails, RiskStatus
from LIVE_TRADING.common.exceptions import MaxDrawdownExceeded


class TestDrawdownMonitor:
    def test_no_drawdown_initially(self):
        monitor = DrawdownMonitor(max_drawdown_pct=10.0)
        state = monitor.update(100_000)
        assert state.drawdown_pct == 0.0

    def test_drawdown_calculation(self):
        monitor = DrawdownMonitor(max_drawdown_pct=10.0)
        monitor.update(100_000)  # Peak
        state = monitor.update(95_000)  # 5% loss
        assert state.drawdown_pct == pytest.approx(5.0)

    def test_exceeded_flag(self):
        monitor = DrawdownMonitor(max_drawdown_pct=10.0)
        monitor.update(100_000)
        state = monitor.update(85_000)  # 15% loss
        assert state.is_exceeded


class TestExposureTracker:
    def test_calculate_exposure(self):
        tracker = ExposureTracker()
        weights = {"AAPL": 0.3, "MSFT": -0.1}
        state = tracker.calculate_exposure(weights)

        assert state.gross_exposure == pytest.approx(0.4)
        assert state.net_exposure == pytest.approx(0.2)
        assert state.long_exposure == pytest.approx(0.3)
        assert state.short_exposure == pytest.approx(0.1)


class TestRiskGuardrails:
    def test_trading_allowed_normal(self):
        guard = RiskGuardrails(max_daily_loss_pct=2.0, initial_capital=100_000)
        status = guard.check_trading_allowed(100_000, {})
        assert status.is_trading_allowed

    def test_kill_switch_on_daily_loss(self):
        guard = RiskGuardrails(max_daily_loss_pct=2.0, initial_capital=100_000)
        guard.check_trading_allowed(100_000, {})  # Start of day
        status = guard.check_trading_allowed(97_000, {})  # 3% loss
        assert not status.is_trading_allowed
        assert "daily_loss" in status.kill_switch_reason
```

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `drawdown.py` | 120 |
| `exposure.py` | 110 |
| `guardrails.py` | 200 |
| `tests/test_risk.py` | 80 |
| **Total** | ~520 |
