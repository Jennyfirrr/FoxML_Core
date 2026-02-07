# Plan 10: Trading Engine

## Overview

Main orchestrator that coordinates all components into a cohesive trading pipeline. Handles state management and the live trading loop.

## Pipeline Flow

```
Market Data
    ↓
┌─────────────────────────────────────────────┐
│ 1. PREDICTION: Multi-horizon predictions     │
├─────────────────────────────────────────────┤
│ 2. BLENDING: Ridge risk-parity per horizon  │
├─────────────────────────────────────────────┤
│ 3. ARBITRATION: Select best horizon         │
├─────────────────────────────────────────────┤
│ 4. GATING: Barrier + spread gates           │
├─────────────────────────────────────────────┤
│ 5. SIZING: Volatility-scaled weights        │
├─────────────────────────────────────────────┤
│ 6. RISK: Final validation                   │
└─────────────────────────────────────────────┘
    ↓
Orders → Broker
```

## Files to Create

### 1. `LIVE_TRADING/engine/__init__.py`

```python
from .trading_engine import TradingEngine, TradeDecision
from .state import EngineState

__all__ = ["TradingEngine", "TradeDecision", "EngineState"]
```

### 2. `LIVE_TRADING/engine/state.py`

```python
"""
Engine State Management
=======================

Manages trading engine state between cycles.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from TRAINING.common.utils.file_utils import write_atomic_json

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """State of a single position."""
    symbol: str
    weight: float
    shares: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0


@dataclass
class EngineState:
    """Complete engine state."""
    portfolio_value: float
    cash: float
    positions: Dict[str, PositionState] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    cycle_count: int = 0
    daily_pnl: float = 0.0

    # History
    trade_history: List[Dict] = field(default_factory=list)
    decision_history: List[Dict] = field(default_factory=list)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current position weights."""
        return {s: p.weight for s, p in self.positions.items()}

    def update_position(
        self,
        symbol: str,
        weight: float,
        shares: float,
        price: float,
    ) -> None:
        """Update or create position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.weight = weight
            pos.shares = shares
        else:
            self.positions[symbol] = PositionState(
                symbol=symbol,
                weight=weight,
                shares=shares,
                entry_price=price,
                entry_time=datetime.now(),
            )

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        if symbol in self.positions:
            del self.positions[symbol]

    def save(self, path: Path) -> None:
        """Save state to file."""
        data = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "cycle_count": self.cycle_count,
            "daily_pnl": self.daily_pnl,
        }
        write_atomic_json(path, data)
        logger.info(f"State saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "EngineState":
        """Load state from file."""
        with open(path) as f:
            data = json.load(f)

        state = cls(
            portfolio_value=data["portfolio_value"],
            cash=data["cash"],
            cycle_count=data.get("cycle_count", 0),
            daily_pnl=data.get("daily_pnl", 0.0),
        )

        if data.get("last_update"):
            state.last_update = datetime.fromisoformat(data["last_update"])

        for symbol, pos_data in data.get("positions", {}).items():
            state.positions[symbol] = PositionState(
                symbol=pos_data["symbol"],
                weight=pos_data["weight"],
                shares=pos_data["shares"],
                entry_price=pos_data["entry_price"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]),
            )

        logger.info(f"State loaded from {path}")
        return state
```

### 3. `LIVE_TRADING/engine/trading_engine.py`

```python
"""
Trading Engine
==============

Main orchestrator for the live trading pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import (
    DECISION_TRADE,
    DECISION_HOLD,
    DECISION_BLOCKED,
    HORIZONS,
)
from LIVE_TRADING.common.exceptions import RiskError
from LIVE_TRADING.brokers.interface import Broker
from LIVE_TRADING.brokers.data_provider import DataProvider
from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor
from LIVE_TRADING.blending.horizon_blender import HorizonBlender
from LIVE_TRADING.arbitration.horizon_arbiter import HorizonArbiter
from LIVE_TRADING.gating.barrier_gate import BarrierGate
from LIVE_TRADING.gating.spread_gate import SpreadGate
from LIVE_TRADING.sizing.position_sizer import PositionSizer
from LIVE_TRADING.risk.guardrails import RiskGuardrails
from .state import EngineState

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Result of a trading cycle."""
    symbol: str
    decision: str  # TRADE, HOLD, BLOCKED
    horizon: Optional[str]
    target_weight: float
    current_weight: float
    alpha: float
    shares: int
    reason: str


class TradingEngine:
    """
    Main trading engine orchestrator.

    Pipeline: predict → blend → arbitrate → gate → size → risk → execute
    """

    def __init__(
        self,
        broker: Broker,
        data_provider: DataProvider,
        run_root: str,
        targets: List[str] | None = None,
        state_path: Path | None = None,
    ):
        """
        Initialize trading engine.

        Args:
            broker: Broker instance
            data_provider: Data provider instance
            run_root: Path to TRAINING run artifacts
            targets: Target names to trade (default: discover from run)
            state_path: Path to save/load state
        """
        self.broker = broker
        self.data_provider = data_provider

        # Initialize components
        self.loader = ModelLoader(run_root)
        self.predictor = MultiHorizonPredictor(run_root)
        self.blender = HorizonBlender()
        self.arbiter = HorizonArbiter()
        self.barrier_gate = BarrierGate()
        self.spread_gate = SpreadGate()
        self.sizer = PositionSizer()
        self.risk_guardrails = RiskGuardrails(
            initial_capital=broker.get_cash()
        )

        # Targets to trade
        self.targets = targets or self.loader.list_available_targets()

        # State management
        self.state_path = state_path or Path("state/engine_state.json")
        self._state: Optional[EngineState] = None

        logger.info(f"TradingEngine initialized: {len(self.targets)} targets")

    @property
    def state(self) -> EngineState:
        """Get or initialize state."""
        if self._state is None:
            if self.state_path.exists():
                self._state = EngineState.load(self.state_path)
            else:
                self._state = EngineState(
                    portfolio_value=self.broker.get_cash(),
                    cash=self.broker.get_cash(),
                )
        return self._state

    def run_cycle(self, symbols: List[str]) -> List[TradeDecision]:
        """
        Run one trading cycle.

        Args:
            symbols: Symbols to process

        Returns:
            List of trade decisions
        """
        self.state.cycle_count += 1
        self.state.last_update = datetime.now()

        decisions = []

        # Check if trading is allowed
        risk_status = self.risk_guardrails.check_trading_allowed(
            self.broker.get_cash() + self._get_position_value(),
            self.state.get_current_weights(),
        )

        if not risk_status.is_trading_allowed:
            logger.warning(f"Trading blocked: {risk_status.kill_switch_reason}")
            for symbol in symbols:
                decisions.append(TradeDecision(
                    symbol=symbol,
                    decision=DECISION_BLOCKED,
                    horizon=None,
                    target_weight=0.0,
                    current_weight=self.state.get_current_weights().get(symbol, 0.0),
                    alpha=0.0,
                    shares=0,
                    reason=risk_status.kill_switch_reason or "kill_switch",
                ))
            return decisions

        for symbol in symbols:
            try:
                decision = self._process_symbol(symbol)
                decisions.append(decision)

                # Execute if TRADE
                if decision.decision == DECISION_TRADE and decision.shares != 0:
                    self._execute_trade(decision)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                decisions.append(TradeDecision(
                    symbol=symbol,
                    decision=DECISION_HOLD,
                    horizon=None,
                    target_weight=0.0,
                    current_weight=self.state.get_current_weights().get(symbol, 0.0),
                    alpha=0.0,
                    shares=0,
                    reason=f"error: {e}",
                ))

        # Save state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.save(self.state_path)

        return decisions

    def _process_symbol(self, symbol: str) -> TradeDecision:
        """Process a single symbol through the pipeline."""
        current_weight = self.state.get_current_weights().get(symbol, 0.0)

        # 1. Get market data
        quote = self.data_provider.get_quote(symbol)
        prices = self.data_provider.get_historical(symbol, period="1mo")

        spread_bps = quote.get("spread_bps", 5.0)
        volatility = prices["Close"].pct_change().std() * (252 ** 0.5)

        # 2. Spread gate
        spread_result = self.spread_gate.evaluate(
            spread_bps=spread_bps,
            quote_timestamp=quote.get("timestamp"),
        )
        if not spread_result.allowed:
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_HOLD,
                horizon=None,
                target_weight=current_weight,
                current_weight=current_weight,
                alpha=0.0,
                shares=0,
                reason=spread_result.reason,
            )

        # 3. Multi-horizon prediction (for first target)
        target = self.targets[0] if self.targets else "ret_5m"
        all_preds = self.predictor.predict_all_horizons(
            target=target,
            prices=prices,
            symbol=symbol,
        )

        # 4. Blend per horizon
        blended = self.blender.blend_all_horizons(
            {h: all_preds.get_horizon(h) for h in HORIZONS if all_preds.get_horizon(h)}
        )

        # 5. Arbitrate
        arb_result = self.arbiter.arbitrate(
            blended_alphas=blended,
            spread_bps=spread_bps,
            volatility=volatility,
        )

        if arb_result.decision != DECISION_TRADE:
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_HOLD,
                horizon=arb_result.selected_horizon,
                target_weight=current_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason=arb_result.reason,
            )

        # 6. Barrier gate (if available)
        # TODO: Load barrier model predictions here
        gate_result = self.barrier_gate.evaluate_long_entry(p_peak=0.0, p_valley=0.0)

        if not gate_result.allowed:
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_BLOCKED,
                horizon=arb_result.selected_horizon,
                target_weight=current_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason=gate_result.reason,
            )

        # 7. Position sizing
        target_weight = self.sizer.calculate_target_weight(
            alpha=arb_result.alpha,
            volatility=volatility,
            gate_result=gate_result,
        )

        # 8. Risk validation
        if not self.risk_guardrails.validate_trade(
            symbol, target_weight, self.state.get_current_weights()
        ):
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_BLOCKED,
                horizon=arb_result.selected_horizon,
                target_weight=target_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason="risk_validation_failed",
            )

        # Calculate shares
        portfolio_value = self.broker.get_cash() + self._get_position_value()
        price = (quote["bid"] + quote["ask"]) / 2
        trade_weight = target_weight - current_weight
        shares = int(abs(trade_weight) * portfolio_value / price)

        if trade_weight < 0:
            shares = -shares

        return TradeDecision(
            symbol=symbol,
            decision=DECISION_TRADE,
            horizon=arb_result.selected_horizon,
            target_weight=target_weight,
            current_weight=current_weight,
            alpha=arb_result.alpha,
            shares=shares,
            reason=arb_result.reason,
        )

    def _execute_trade(self, decision: TradeDecision) -> None:
        """Execute a trade decision."""
        side = "BUY" if decision.shares > 0 else "SELL"
        qty = abs(decision.shares)

        try:
            result = self.broker.submit_order(
                symbol=decision.symbol,
                side=side,
                qty=qty,
            )
            logger.info(f"Trade executed: {side} {qty} {decision.symbol}")

            # Update state
            self.state.update_position(
                symbol=decision.symbol,
                weight=decision.target_weight,
                shares=self.state.positions.get(decision.symbol, PositionState(decision.symbol, 0, 0, 0, datetime.now())).shares + decision.shares,
                price=result.get("fill_price", 0),
            )

            # Record in history
            self.state.trade_history.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": decision.symbol,
                "side": side,
                "qty": qty,
                "fill_price": result.get("fill_price"),
                "horizon": decision.horizon,
            })

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    def _get_position_value(self) -> float:
        """Get total position value."""
        value = 0.0
        for symbol, pos in self.state.positions.items():
            try:
                quote = self.data_provider.get_quote(symbol)
                mid = (quote["bid"] + quote["ask"]) / 2
                value += pos.shares * mid
            except Exception:
                pass
        return value
```

## Tests

### `LIVE_TRADING/tests/test_engine_integration.py`

```python
"""Integration tests for trading engine."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from LIVE_TRADING.engine.trading_engine import TradingEngine, TradeDecision
from LIVE_TRADING.engine.state import EngineState
from LIVE_TRADING.common.constants import DECISION_TRADE, DECISION_HOLD


@pytest.fixture
def mock_broker():
    broker = Mock()
    broker.get_cash.return_value = 100_000
    broker.get_positions.return_value = {}
    broker.submit_order.return_value = {"order_id": "123", "fill_price": 150.0}
    return broker


@pytest.fixture
def mock_data_provider():
    provider = Mock()
    provider.get_quote.return_value = {
        "bid": 149.9,
        "ask": 150.1,
        "spread_bps": 5.0,
        "timestamp": datetime.now(),
    }

    import pandas as pd
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=30)
    provider.get_historical.return_value = pd.DataFrame({
        "Close": np.random.randn(30).cumsum() + 150,
        "Volume": np.random.randint(1000000, 10000000, 30),
    }, index=dates)

    return provider


class TestEngineState:
    def test_get_current_weights(self):
        state = EngineState(portfolio_value=100000, cash=90000)
        assert state.get_current_weights() == {}

    def test_update_position(self):
        state = EngineState(portfolio_value=100000, cash=90000)
        state.update_position("AAPL", 0.1, 100, 150.0)
        assert "AAPL" in state.positions
        assert state.positions["AAPL"].weight == 0.1


class TestTradingEngine:
    # Note: Full integration tests require more setup
    pass
```

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `state.py` | 150 |
| `trading_engine.py` | 350 |
| `tests/test_engine_integration.py` | 80 |
| **Total** | ~590 |
