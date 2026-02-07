# Plan 07: Gating

## Overview

Barrier probability gates that use peak/valley predictions to filter entries and reduce position sizes. Prevents entering positions right before local tops.

## Mathematical Foundation

### Barrier Gate Formula
```
g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)
```
- `g_min` = minimum gate value (0.2)
- `γ` = peak sensitivity (1.0)
- `δ` = valley sensitivity (0.5)
- `p_peak` = probability of upcoming peak
- `p_valley` = probability of upcoming valley

### Long Entry Rules
- **Block if:** `P(will_peak_5m) > 0.6` OR `P(y_will_peak_5m) > 0.6`
- **Prefer if:** `P(will_valley_5m) > 0.55` AND `ΔP > 0`

### Position Reduction
```
size_reduction = (1 - p_peak)
```

## Files to Create

### 1. `LIVE_TRADING/gating/__init__.py`

```python
from .barrier_gate import BarrierGate, GateResult
from .spread_gate import SpreadGate

__all__ = ["BarrierGate", "GateResult", "SpreadGate"]
```

### 2. `LIVE_TRADING/gating/barrier_gate.py`

```python
"""
Barrier Probability Gate
========================

Uses peak/valley predictions to gate entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG, DECISION_TRADE, DECISION_BLOCKED

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of gate evaluation."""
    allowed: bool
    gate_value: float  # 0-1, 1 = fully allowed
    reason: str
    p_peak: Optional[float] = None
    p_valley: Optional[float] = None


class BarrierGate:
    """
    Gates entries using barrier probability predictions.

    g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)
    """

    def __init__(
        self,
        g_min: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        peak_threshold: float | None = None,
        valley_threshold: float | None = None,
    ):
        """
        Initialize barrier gate.

        Args:
            g_min: Minimum gate value
            gamma: Peak sensitivity exponent
            delta: Valley sensitivity exponent
            peak_threshold: Block threshold for peak probability
            valley_threshold: Prefer threshold for valley probability
        """
        self.g_min = g_min or get_cfg(
            "live_trading.barrier_gate.g_min",
            default=DEFAULT_CONFIG["g_min"],
        )
        self.gamma = gamma or get_cfg(
            "live_trading.barrier_gate.gamma",
            default=DEFAULT_CONFIG["gamma"],
        )
        self.delta = delta or get_cfg(
            "live_trading.barrier_gate.delta",
            default=DEFAULT_CONFIG["delta"],
        )
        self.peak_threshold = peak_threshold or get_cfg(
            "live_trading.barrier_gate.peak_threshold",
            default=DEFAULT_CONFIG["peak_threshold"],
        )
        self.valley_threshold = valley_threshold or get_cfg(
            "live_trading.barrier_gate.valley_threshold",
            default=DEFAULT_CONFIG["valley_threshold"],
        )

        logger.info(f"BarrierGate: g_min={self.g_min}, γ={self.gamma}, δ={self.delta}")

    def evaluate_long_entry(
        self,
        p_peak: float,
        p_valley: float = 0.0,
        y_p_peak: float | None = None,
    ) -> GateResult:
        """
        Evaluate gate for long entry.

        Args:
            p_peak: P(will_peak_5m)
            p_valley: P(will_valley_5m)
            y_p_peak: P(y_will_peak_5m) - optional auxiliary

        Returns:
            GateResult
        """
        # Check hard block
        if p_peak > self.peak_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"peak_prob {p_peak:.2f} > {self.peak_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        if y_p_peak is not None and y_p_peak > self.peak_threshold:
            return GateResult(
                allowed=False,
                gate_value=0.0,
                reason=f"y_peak_prob {y_p_peak:.2f} > {self.peak_threshold}",
                p_peak=p_peak,
                p_valley=p_valley,
            )

        # Calculate gate value
        gate_value = self._calculate_gate(p_peak, p_valley)

        # Prefer entry if valley detected
        prefer = p_valley > self.valley_threshold
        reason = "valley_bounce" if prefer else "normal_entry"

        return GateResult(
            allowed=True,
            gate_value=gate_value,
            reason=reason,
            p_peak=p_peak,
            p_valley=p_valley,
        )

    def _calculate_gate(self, p_peak: float, p_valley: float) -> float:
        """
        Calculate gate value.

        g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)
        """
        peak_factor = (1 - p_peak) ** self.gamma
        valley_factor = (0.5 + 0.5 * p_valley) ** self.delta
        g = peak_factor * valley_factor
        return max(self.g_min, min(1.0, g))

    def evaluate_long_exit(
        self,
        p_peak: float,
        alpha: float,
    ) -> GateResult:
        """
        Evaluate exit signal for long position.

        Exit if: P(will_peak_5m) > 0.65 OR alpha < 0
        """
        exit_threshold = self.peak_threshold + 0.05  # Slightly higher for exit

        if p_peak > exit_threshold:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"peak_exit: {p_peak:.2f} > {exit_threshold}",
                p_peak=p_peak,
            )

        if alpha < 0:
            return GateResult(
                allowed=True,
                gate_value=1.0,
                reason=f"alpha_exit: {alpha:.4f} < 0",
                p_peak=p_peak,
            )

        return GateResult(
            allowed=False,
            gate_value=0.0,
            reason="hold_position",
            p_peak=p_peak,
        )

    def get_size_reduction(self, p_peak: float) -> float:
        """
        Calculate position size reduction based on peak probability.

        size_reduction = (1 - p_peak)
        """
        return max(0.0, 1.0 - p_peak)
```

### 3. `LIVE_TRADING/gating/spread_gate.py`

```python
"""
Spread Gate
===========

Gates based on bid-ask spread thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import SpreadTooWideError, StaleDataError

logger = logging.getLogger(__name__)


@dataclass
class SpreadGateResult:
    """Result of spread gate evaluation."""
    allowed: bool
    spread_bps: float
    reason: str


class SpreadGate:
    """
    Gates trades based on spread and data freshness.
    """

    def __init__(
        self,
        max_spread_bps: float | None = None,
        max_quote_age_ms: float | None = None,
    ):
        """
        Initialize spread gate.

        Args:
            max_spread_bps: Maximum allowed spread
            max_quote_age_ms: Maximum quote age in milliseconds
        """
        self.max_spread_bps = max_spread_bps or get_cfg(
            "live_trading.risk.spread_max_bps",
            default=DEFAULT_CONFIG["spread_max_bps"],
        )
        self.max_quote_age_ms = max_quote_age_ms or get_cfg(
            "live_trading.risk.quote_age_max_ms",
            default=DEFAULT_CONFIG["quote_age_max_ms"],
        )

        logger.info(f"SpreadGate: max_spread={self.max_spread_bps}bps")

    def evaluate(
        self,
        spread_bps: float,
        quote_timestamp: datetime | None = None,
    ) -> SpreadGateResult:
        """
        Evaluate spread gate.

        Args:
            spread_bps: Current spread in basis points
            quote_timestamp: Quote timestamp for freshness check

        Returns:
            SpreadGateResult
        """
        # Check spread
        if spread_bps > self.max_spread_bps:
            return SpreadGateResult(
                allowed=False,
                spread_bps=spread_bps,
                reason=f"spread {spread_bps:.1f}bps > max {self.max_spread_bps}bps",
            )

        # Check quote freshness
        if quote_timestamp is not None:
            age_ms = (datetime.now() - quote_timestamp).total_seconds() * 1000
            if age_ms > self.max_quote_age_ms:
                return SpreadGateResult(
                    allowed=False,
                    spread_bps=spread_bps,
                    reason=f"quote age {age_ms:.0f}ms > max {self.max_quote_age_ms}ms",
                )

        return SpreadGateResult(
            allowed=True,
            spread_bps=spread_bps,
            reason="spread_ok",
        )

    def validate_or_raise(
        self,
        symbol: str,
        spread_bps: float,
        quote_timestamp: datetime | None = None,
    ) -> None:
        """
        Validate and raise exception if gate fails.

        Args:
            symbol: Trading symbol
            spread_bps: Current spread
            quote_timestamp: Quote timestamp
        """
        if spread_bps > self.max_spread_bps:
            raise SpreadTooWideError(symbol, spread_bps, self.max_spread_bps)

        if quote_timestamp is not None:
            age_ms = (datetime.now() - quote_timestamp).total_seconds() * 1000
            if age_ms > self.max_quote_age_ms:
                raise StaleDataError(symbol, age_ms, self.max_quote_age_ms)
```

## Tests

### `LIVE_TRADING/tests/test_gating.py`

```python
"""Tests for gating components."""

import pytest
from datetime import datetime, timedelta

from LIVE_TRADING.gating.barrier_gate import BarrierGate, GateResult
from LIVE_TRADING.gating.spread_gate import SpreadGate, SpreadGateResult
from LIVE_TRADING.common.exceptions import SpreadTooWideError


class TestBarrierGate:
    def test_block_high_peak_prob(self):
        gate = BarrierGate(peak_threshold=0.6)
        result = gate.evaluate_long_entry(p_peak=0.7)
        assert not result.allowed
        assert "peak_prob" in result.reason

    def test_allow_low_peak_prob(self):
        gate = BarrierGate(peak_threshold=0.6)
        result = gate.evaluate_long_entry(p_peak=0.3)
        assert result.allowed

    def test_gate_value_calculation(self):
        gate = BarrierGate(g_min=0.2, gamma=1.0, delta=0.5)

        # Low peak, no valley: gate near 1
        result = gate.evaluate_long_entry(p_peak=0.1, p_valley=0.0)
        assert result.gate_value > 0.8

        # High peak: gate lower but above g_min
        result = gate.evaluate_long_entry(p_peak=0.5, p_valley=0.0)
        assert result.gate_value < 0.8
        assert result.gate_value >= 0.2

    def test_size_reduction(self):
        gate = BarrierGate()
        assert gate.get_size_reduction(0.0) == 1.0
        assert gate.get_size_reduction(0.5) == 0.5
        assert gate.get_size_reduction(1.0) == 0.0


class TestSpreadGate:
    def test_allow_narrow_spread(self):
        gate = SpreadGate(max_spread_bps=10.0)
        result = gate.evaluate(spread_bps=5.0)
        assert result.allowed

    def test_block_wide_spread(self):
        gate = SpreadGate(max_spread_bps=10.0)
        result = gate.evaluate(spread_bps=15.0)
        assert not result.allowed
        assert "spread" in result.reason

    def test_stale_quote_blocked(self):
        gate = SpreadGate(max_quote_age_ms=200)
        old_time = datetime.now() - timedelta(seconds=1)
        result = gate.evaluate(spread_bps=5.0, quote_timestamp=old_time)
        assert not result.allowed
        assert "age" in result.reason

    def test_validate_or_raise(self):
        gate = SpreadGate(max_spread_bps=10.0)
        with pytest.raises(SpreadTooWideError):
            gate.validate_or_raise("AAPL", spread_bps=15.0)
```

## SST Compliance Checklist

- [ ] Uses `get_cfg()` for configuration
- [ ] Proper exception classes for gate violations
- [ ] All thresholds configurable

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `barrier_gate.py` | 180 |
| `spread_gate.py` | 120 |
| `tests/test_gating.py` | 80 |
| **Total** | ~390 |
