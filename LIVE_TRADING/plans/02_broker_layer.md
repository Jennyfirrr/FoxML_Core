# Plan 02: Broker Layer

## Overview

The broker layer provides a protocol-based interface for order execution, supporting both paper trading and live broker integrations. This design allows switching between paper and live modes without changing pipeline code.

## Files to Create

### 1. `LIVE_TRADING/brokers/__init__.py`

```python
from .interface import Broker, get_broker, normalize_order
from .paper import PaperBroker

__all__ = ["Broker", "get_broker", "PaperBroker", "normalize_order"]
```

### 2. `LIVE_TRADING/brokers/interface.py`
**Purpose:** Broker Protocol definition and factory function

**Reference:** `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/brokers/interface.py`

```python
"""
Broker Interface Protocol
=========================

Protocol definition for live trading broker adapters.
All broker implementations must satisfy this protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class Broker(Protocol):
    """Protocol for live trading broker adapters."""

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Submit an order to the broker.

        Args:
            symbol: Trading symbol (e.g., "SPY", "AAPL")
            side: "BUY" or "SELL"
            qty: Quantity to trade (shares)
            order_type: "market" or "limit"
            limit_price: Limit price (required if order_type="limit")

        Returns:
            Dict with order_id, status, timestamp, fill_price, etc.

        Raises:
            BrokerError: If order submission fails
            OrderRejectedError: If order is rejected
        """
        ...

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict with status, timestamp
        """
        ...

    def get_positions(self) -> dict[str, float]:
        """
        Get current positions by symbol.

        Returns:
            Dict mapping symbol to position size
            Positive = long, negative = short
        """
        ...

    def get_cash(self) -> float:
        """
        Get available cash balance.

        Returns:
            Available cash amount
        """
        ...

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """
        Get recent fills.

        Args:
            since: Get fills since this timestamp (UTC)

        Returns:
            List of fill dicts with symbol, side, qty, price, timestamp, etc.
        """
        ...

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with bid, ask, bid_size, ask_size, timestamp
        """
        ...

    def now(self) -> datetime:
        """
        Get current broker time (UTC).

        Returns:
            Current timestamp from broker
        """
        ...


def get_broker(venue: str = "paper", **kwargs) -> Broker:
    """
    Factory function to get broker instance.

    Args:
        venue: Broker venue ("paper", "ibkr", etc.)
        **kwargs: Broker-specific configuration

    Returns:
        Broker instance implementing the Broker protocol

    Raises:
        ValueError: If venue is unknown
    """
    if venue == "paper":
        from .paper import PaperBroker
        return PaperBroker(**kwargs)
    elif venue == "ibkr":
        raise NotImplementedError("IBKR broker not yet implemented")
    else:
        raise ValueError(f"Unknown broker venue: {venue}")


def normalize_order(
    symbol: str,
    side: str,
    qty: float,
    px: float | None = None,
) -> dict[str, Any]:
    """
    Normalize order parameters to standard format.

    Args:
        symbol: Trading symbol
        side: "BUY" or "SELL"
        qty: Quantity
        px: Price (optional)

    Returns:
        Normalized order dict
    """
    return {
        "symbol": str(symbol).upper(),
        "side": str(side).upper(),
        "qty": float(qty),
        "px": float(px) if px is not None else None,
    }
```

### 3. `LIVE_TRADING/brokers/paper.py`
**Purpose:** Paper trading broker with slippage simulation

**Reference:** `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/brokers/paper.py`

```python
"""
Paper Trading Broker
====================

Simulated broker for paper trading with configurable slippage and fees.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.file_utils import write_atomic_json

from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    SIDE_BUY,
    SIDE_SELL,
)
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    InsufficientFundsError,
    OrderRejectedError,
)


class PaperBroker:
    """
    Paper trading broker with slippage simulation.

    Implements the Broker protocol for testing and simulation.
    """

    def __init__(
        self,
        *,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
        initial_cash: float | None = None,
        log_dir: str | Path = "logs/paper_trades",
    ):
        """
        Initialize paper broker.

        Args:
            slippage_bps: Slippage in basis points (default from config)
            fee_bps: Fee in basis points (default from config)
            initial_cash: Initial cash balance (default from config)
            log_dir: Directory for trade logs
        """
        # Load from config with defaults
        self._slippage_bps = slippage_bps or get_cfg(
            "live_trading.paper.slippage_bps",
            default=DEFAULT_CONFIG["slippage_bps"]
        )
        self._fee_bps = fee_bps or get_cfg(
            "live_trading.paper.fee_bps",
            default=DEFAULT_CONFIG["fee_bps"]
        )
        initial = initial_cash or get_cfg(
            "live_trading.paper.initial_cash",
            default=DEFAULT_CONFIG["initial_cash"]
        )

        # Convert bps to decimal
        self._slip = float(self._slippage_bps) * 1e-4
        self._fee = float(self._fee_bps) * 1e-4

        # State
        self._positions: dict[str, float] = {}
        self._cash: float = float(initial)
        self._fills: list[dict[str, Any]] = []
        self._orders: dict[str, dict[str, Any]] = {}
        self._quotes: dict[str, dict[str, Any]] = {}

        # Logging
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Submit order with slippage simulation."""
        symbol = symbol.upper()
        side = side.upper()
        qty = float(qty)

        if side not in (SIDE_BUY, SIDE_SELL):
            raise OrderRejectedError(symbol, f"Invalid side: {side}")

        if qty <= 0:
            raise OrderRejectedError(symbol, f"Invalid quantity: {qty}")

        # Get quote for fill price
        quote = self._quotes.get(symbol)
        if quote is None:
            raise BrokerError(f"No quote available for {symbol}")

        # Calculate fill price with slippage
        if side == SIDE_BUY:
            base_price = quote["ask"]
            fill_price = base_price * (1 + self._slip)
        else:
            base_price = quote["bid"]
            fill_price = base_price * (1 - self._slip)

        # Calculate trade value and fee
        notional = qty * fill_price
        fee = notional * self._fee

        # Check cash for buy orders
        if side == SIDE_BUY:
            required_cash = notional + fee
            if required_cash > self._cash:
                raise InsufficientFundsError(
                    f"Insufficient cash: need ${required_cash:.2f}, have ${self._cash:.2f}"
                )

        # Check position for sell orders
        if side == SIDE_SELL:
            current_pos = self._positions.get(symbol, 0.0)
            if qty > current_pos:
                raise OrderRejectedError(
                    symbol, f"Insufficient shares: need {qty}, have {current_pos}"
                )

        # Execute the order
        order_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(UTC)

        # Update positions
        if side == SIDE_BUY:
            self._positions[symbol] = self._positions.get(symbol, 0.0) + qty
            self._cash -= (notional + fee)
        else:
            self._positions[symbol] = self._positions.get(symbol, 0.0) - qty
            self._cash += (notional - fee)
            # Clean up zero positions
            if abs(self._positions[symbol]) < 1e-9:
                del self._positions[symbol]

        # Create fill record
        fill = {
            "order_id": order_id,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "fill_price": fill_price,
            "notional": notional,
            "fee": fee,
            "slippage_bps": self._slippage_bps,
            "fee_bps": self._fee_bps,
        }

        self._fills.append(fill)
        self._orders[order_id] = fill
        self._append_log(fill)

        return {
            "order_id": order_id,
            "status": "filled",
            "fill_price": fill_price,
            "qty": qty,
            "timestamp": timestamp,
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel order (no-op for paper broker since orders fill instantly)."""
        return {
            "order_id": order_id,
            "status": "cancelled" if order_id not in self._orders else "filled",
            "timestamp": datetime.now(UTC),
        }

    def get_positions(self) -> dict[str, float]:
        """Get current positions."""
        return dict(self._positions)

    def get_cash(self) -> float:
        """Get available cash."""
        return float(self._cash)

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Get fills, optionally filtered by time."""
        if since is None:
            return list(self._fills)

        return [
            f for f in self._fills
            if datetime.fromisoformat(f["timestamp"]) >= since
        ]

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for symbol."""
        symbol = symbol.upper()
        if symbol not in self._quotes:
            raise BrokerError(f"No quote for {symbol}")
        return dict(self._quotes[symbol])

    def set_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: float = 100,
        ask_size: float = 100,
    ) -> None:
        """
        Set quote for a symbol (for testing/simulation).

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
        """
        self._quotes[symbol.upper()] = {
            "symbol": symbol.upper(),
            "bid": float(bid),
            "ask": float(ask),
            "bid_size": float(bid_size),
            "ask_size": float(ask_size),
            "timestamp": datetime.now(UTC),
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 10000,
        }

    def now(self) -> datetime:
        """Get current time."""
        return datetime.now(UTC)

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        value = self._cash
        for symbol, qty in self._positions.items():
            if symbol in self._quotes:
                mid = (self._quotes[symbol]["bid"] + self._quotes[symbol]["ask"]) / 2
                value += qty * mid
        return value

    def reset(self, initial_cash: float | None = None) -> None:
        """Reset broker state."""
        self._positions.clear()
        self._fills.clear()
        self._orders.clear()
        self._cash = initial_cash or get_cfg(
            "live_trading.paper.initial_cash",
            default=DEFAULT_CONFIG["initial_cash"]
        )

    def _append_log(self, record: dict[str, Any]) -> None:
        """Append record to daily log file."""
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        log_file = self._log_dir / f"{date_str}.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
```

### 4. `LIVE_TRADING/brokers/data_provider.py`
**Purpose:** Market data abstraction for getting quotes and historical data

```python
"""
Market Data Provider
====================

Abstraction for market data access (quotes, historical data).
Supports multiple backends with fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get current quote for a symbol."""
        ...

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        ...

    @abstractmethod
    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols."""
        ...


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider (free, for paper trading)."""

    def __init__(self, cache_ttl_seconds: float = 30.0):
        self._cache: dict[str, tuple[datetime, dict]] = {}
        self._cache_ttl = cache_ttl_seconds

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get quote from yfinance."""
        import yfinance as yf

        # Check cache
        if symbol in self._cache:
            cached_time, cached_quote = self._cache[symbol]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self._cache_ttl:
                return cached_quote

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get bid/ask or fall back to price
            bid = info.get("bid", info.get("regularMarketPrice", 0))
            ask = info.get("ask", info.get("regularMarketPrice", 0))

            # If bid/ask are 0, use last price with synthetic spread
            if bid == 0 or ask == 0:
                price = info.get("regularMarketPrice", 0)
                spread_pct = 0.001  # 10 bps synthetic spread
                bid = price * (1 - spread_pct / 2)
                ask = price * (1 + spread_pct / 2)

            quote = {
                "symbol": symbol,
                "bid": float(bid),
                "ask": float(ask),
                "bid_size": float(info.get("bidSize", 100)),
                "ask_size": float(info.get("askSize", 100)),
                "timestamp": datetime.now(),
                "spread_bps": (ask - bid) / ((ask + bid) / 2) * 10000 if (ask + bid) > 0 else 0,
            }

            self._cache[symbol] = (datetime.now(), quote)
            return quote

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical data from yfinance."""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get quotes for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        return results


def get_data_provider(backend: str = "yfinance") -> DataProvider:
    """
    Factory function for data providers.

    Args:
        backend: Data provider backend ("yfinance", "ibkr")

    Returns:
        DataProvider instance
    """
    if backend == "yfinance":
        return YFinanceProvider()
    elif backend == "ibkr":
        raise NotImplementedError("IBKR data provider not yet implemented")
    else:
        raise ValueError(f"Unknown data provider backend: {backend}")
```

## Tests

### `LIVE_TRADING/tests/test_broker_interface.py`

```python
"""Tests for broker interface."""

import pytest
from datetime import datetime
from typing import Protocol, runtime_checkable

from LIVE_TRADING.brokers.interface import Broker, normalize_order, get_broker


class TestBrokerProtocol:
    def test_broker_is_protocol(self):
        assert hasattr(Broker, "__protocol_attrs__") or isinstance(Broker, type)

    def test_normalize_order(self):
        order = normalize_order("aapl", "buy", 100, 150.50)
        assert order["symbol"] == "AAPL"
        assert order["side"] == "BUY"
        assert order["qty"] == 100.0
        assert order["px"] == 150.50

    def test_normalize_order_no_price(self):
        order = normalize_order("SPY", "SELL", 50)
        assert order["px"] is None


class TestGetBroker:
    def test_get_paper_broker(self):
        broker = get_broker("paper")
        assert hasattr(broker, "submit_order")
        assert hasattr(broker, "get_positions")

    def test_unknown_venue_raises(self):
        with pytest.raises(ValueError, match="Unknown broker"):
            get_broker("unknown_venue")
```

### `LIVE_TRADING/tests/test_paper_broker.py`

```python
"""Tests for paper broker."""

import pytest
from datetime import datetime

from LIVE_TRADING.brokers.paper import PaperBroker
from LIVE_TRADING.common.exceptions import (
    InsufficientFundsError,
    OrderRejectedError,
    BrokerError,
)


@pytest.fixture
def broker():
    """Create a paper broker for testing."""
    b = PaperBroker(initial_cash=100_000, slippage_bps=5, fee_bps=1)
    # Set up a quote
    b.set_quote("AAPL", bid=150.0, ask=150.10)
    return b


class TestPaperBroker:
    def test_initial_state(self, broker):
        assert broker.get_cash() == 100_000
        assert broker.get_positions() == {}
        assert broker.get_fills() == []

    def test_buy_order(self, broker):
        result = broker.submit_order("AAPL", "BUY", 100)

        assert result["status"] == "filled"
        assert result["qty"] == 100
        assert "order_id" in result

        positions = broker.get_positions()
        assert positions["AAPL"] == 100
        assert broker.get_cash() < 100_000  # Cash reduced

    def test_sell_order(self, broker):
        # First buy
        broker.submit_order("AAPL", "BUY", 100)
        initial_cash = broker.get_cash()

        # Then sell
        result = broker.submit_order("AAPL", "SELL", 50)

        assert result["status"] == "filled"
        assert broker.get_positions()["AAPL"] == 50
        assert broker.get_cash() > initial_cash

    def test_insufficient_funds(self, broker):
        with pytest.raises(InsufficientFundsError):
            # Try to buy too much
            broker.submit_order("AAPL", "BUY", 10000)  # Would cost ~$1.5M

    def test_insufficient_shares(self, broker):
        with pytest.raises(OrderRejectedError):
            # Try to sell without position
            broker.submit_order("AAPL", "SELL", 100)

    def test_no_quote_error(self, broker):
        with pytest.raises(BrokerError, match="No quote"):
            broker.submit_order("UNKNOWN", "BUY", 100)

    def test_invalid_side(self, broker):
        with pytest.raises(OrderRejectedError, match="Invalid side"):
            broker.submit_order("AAPL", "INVALID", 100)

    def test_slippage_applied(self, broker):
        # Buy should pay more than ask
        result = broker.submit_order("AAPL", "BUY", 100)
        assert result["fill_price"] > 150.10  # Ask was 150.10

    def test_fills_recorded(self, broker):
        broker.submit_order("AAPL", "BUY", 100)
        broker.submit_order("AAPL", "SELL", 50)

        fills = broker.get_fills()
        assert len(fills) == 2
        assert fills[0]["side"] == "BUY"
        assert fills[1]["side"] == "SELL"

    def test_portfolio_value(self, broker):
        initial_value = broker.get_portfolio_value()
        assert initial_value == 100_000

        broker.submit_order("AAPL", "BUY", 100)
        # Value should be approximately same (minus fees/slippage)
        assert broker.get_portfolio_value() < 100_000

    def test_reset(self, broker):
        broker.submit_order("AAPL", "BUY", 100)
        broker.reset()

        assert broker.get_cash() == 100_000
        assert broker.get_positions() == {}
```

## SST Compliance Checklist

- [ ] Uses `get_cfg()` for all configuration with defaults
- [ ] Uses `write_atomic_json` for file writes (in logging)
- [ ] No hardcoded values - uses `DEFAULT_CONFIG` fallbacks
- [ ] Protocol-based design for testability
- [ ] Proper exception hierarchy (BrokerError, etc.)

## Dependencies

- `CONFIG.config_loader.get_cfg`
- `TRAINING.common.utils.file_utils.write_atomic_json`
- `LIVE_TRADING.common.exceptions`
- `LIVE_TRADING.common.constants`
- External: `yfinance` (optional, for data provider)

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `interface.py` | 120 |
| `paper.py` | 280 |
| `data_provider.py` | 180 |
| `tests/test_broker_interface.py` | 50 |
| `tests/test_paper_broker.py` | 120 |
| **Total** | ~760 |
