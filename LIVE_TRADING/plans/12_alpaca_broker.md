# Plan 12: Alpaca Broker

## Overview

Alpaca broker integration for live/paper trading via REST API.

## Files to Create

### 1. `LIVE_TRADING/brokers/alpaca.py`

```python
"""
Alpaca Broker Implementation
============================

Live trading via Alpaca Markets API.

Features:
- Paper and live trading modes
- Order submission (market, limit)
- Position tracking
- Account info
- Fill retrieval

SST Compliance:
- Credentials from environment variables
- No hardcoded API keys
- Uses get_cfg() for configuration
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.brokers.interface import Broker
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
    ConnectionError as BrokerConnectionError,
)

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """
    Alpaca Markets broker implementation.

    Implements the Broker Protocol for live trading.

    Example:
        >>> broker = AlpacaBroker()  # Uses env vars
        >>> broker.submit_order("AAPL", "BUY", 10)
        {'order_id': 'xxx', 'status': 'filled', ...}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca broker.

        Args:
            api_key: API key (default: from ALPACA_API_KEY env)
            api_secret: API secret (default: from ALPACA_API_SECRET env)
            paper: Use paper trading (default: True)
        """
        # Load credentials
        self._api_key = api_key or os.environ.get(
            get_cfg("live_trading.brokers.alpaca.api_key_env", default="ALPACA_API_KEY")
        )
        self._api_secret = api_secret or os.environ.get(
            get_cfg("live_trading.brokers.alpaca.api_secret_env", default="ALPACA_API_SECRET")
        )

        if not self._api_key or not self._api_secret:
            raise BrokerConnectionError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_API_SECRET"
            )

        self._paper = paper

        # Initialize client
        try:
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
                paper=paper,
            )
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to Alpaca: {e}")

        logger.info(f"AlpacaBroker initialized (paper={paper})")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Submit an order to Alpaca."""
        try:
            alpaca_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

            if order_type.lower() == "market":
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise BrokerError("Limit price required for limit orders")
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                raise BrokerError(f"Unsupported order type: {order_type}")

            order = self._client.submit_order(request)

            return {
                "order_id": str(order.id),
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": float(order.qty),
                "order_type": order.type.value,
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "fill_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            error_msg = str(e)
            if "insufficient" in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds for {symbol}: {e}")
            elif "rejected" in error_msg.lower():
                raise OrderRejectedError(symbol, error_msg)
            else:
                raise BrokerError(f"Order submission failed: {e}")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        try:
            self._client.cancel_order_by_id(order_id)
            return {"order_id": order_id, "status": "cancelled"}
        except Exception as e:
            raise BrokerError(f"Cancel failed for {order_id}: {e}")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        try:
            order = self._client.get_order_by_id(order_id)
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": float(order.qty),
                "status": order.status.value,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "fill_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            raise BrokerError(f"Failed to get order {order_id}: {e}")

    def get_positions(self) -> Dict[str, float]:
        """Get current positions (symbol -> quantity)."""
        try:
            positions = self._client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            raise BrokerError(f"Failed to get positions: {e}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        try:
            pos = self._client.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "side": pos.side.value,
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
            }
        except Exception as e:
            if "not found" in str(e).lower():
                return None
            raise BrokerError(f"Failed to get position {symbol}: {e}")

    def get_cash(self) -> float:
        """Get available cash."""
        try:
            account = self._client.get_account()
            return float(account.cash)
        except Exception as e:
            raise BrokerError(f"Failed to get account: {e}")

    def get_buying_power(self) -> float:
        """Get buying power."""
        try:
            account = self._client.get_account()
            return float(account.buying_power)
        except Exception as e:
            raise BrokerError(f"Failed to get buying power: {e}")

    def get_account(self) -> Dict[str, Any]:
        """Get full account info."""
        try:
            account = self._client.get_account()
            return {
                "account_id": account.id,
                "status": account.status.value,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            raise BrokerError(f"Failed to get account: {e}")

    def get_fills(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get recent fills."""
        try:
            orders = self._client.get_orders(
                status="filled",
                limit=100,
                after=since,
            )
            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.filled_qty),
                    "price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                }
                for o in orders
            ]
        except Exception as e:
            raise BrokerError(f"Failed to get fills: {e}")

    def now(self) -> datetime:
        """Get current broker time."""
        try:
            clock = self._client.get_clock()
            return clock.timestamp.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def is_market_open(self) -> bool:
        """Check if market is open."""
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self._paper
```

### 2. `LIVE_TRADING/brokers/__init__.py` (update)

Add AlpacaBroker to exports and factory function:

```python
# Add to existing file
from .alpaca import AlpacaBroker

__all__ = [
    "Broker",
    "PaperBroker",
    "AlpacaBroker",
    "get_broker",
]

def get_broker(broker_type: str, **kwargs) -> Broker:
    """Factory function for brokers."""
    if broker_type == "paper":
        return PaperBroker(**kwargs)
    elif broker_type == "alpaca":
        return AlpacaBroker(**kwargs)
    elif broker_type == "ibkr":
        raise NotImplementedError("IBKR broker not yet implemented")
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
```

## Tests

### `LIVE_TRADING/tests/test_alpaca_broker.py`

```python
"""
Alpaca Broker Tests
===================

Unit tests with mocked API, plus manual integration tests.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from LIVE_TRADING.brokers.alpaca import AlpacaBroker
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
)


class TestAlpacaBrokerInit:
    """Tests for broker initialization."""

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_API_SECRET": "test_secret"})
    @patch("LIVE_TRADING.brokers.alpaca.TradingClient")
    def test_init_from_env(self, mock_client):
        """Test initialization from environment variables."""
        broker = AlpacaBroker()
        assert broker._api_key == "test_key"
        assert broker._paper is True
        mock_client.assert_called_once()

    @patch("LIVE_TRADING.brokers.alpaca.TradingClient")
    def test_init_explicit_credentials(self, mock_client):
        """Test initialization with explicit credentials."""
        broker = AlpacaBroker(api_key="key", api_secret="secret", paper=False)
        assert broker._api_key == "key"
        assert broker._paper is False


class TestAlpacaBrokerOrders:
    """Tests for order operations."""

    @pytest.fixture
    def broker(self):
        """Create broker with mocked client."""
        with patch("LIVE_TRADING.brokers.alpaca.TradingClient") as mock_client:
            broker = AlpacaBroker(api_key="key", api_secret="secret")
            broker._client = Mock()
            return broker

    def test_submit_market_order(self, broker):
        """Test market order submission."""
        mock_order = Mock()
        mock_order.id = "order_123"
        mock_order.client_order_id = "client_123"
        mock_order.symbol = "AAPL"
        mock_order.side.value = "buy"
        mock_order.qty = 10
        mock_order.type.value = "market"
        mock_order.status.value = "filled"
        mock_order.submitted_at = datetime.now(timezone.utc)
        mock_order.filled_qty = 10
        mock_order.filled_avg_price = 150.0

        broker._client.submit_order.return_value = mock_order

        result = broker.submit_order("AAPL", "BUY", 10)

        assert result["order_id"] == "order_123"
        assert result["symbol"] == "AAPL"
        assert result["status"] == "filled"

    def test_get_positions(self, broker):
        """Test position retrieval."""
        mock_pos = Mock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = 100

        broker._client.get_all_positions.return_value = [mock_pos]

        positions = broker.get_positions()

        assert positions == {"AAPL": 100.0}

    def test_get_cash(self, broker):
        """Test cash retrieval."""
        mock_account = Mock()
        mock_account.cash = 50000.0

        broker._client.get_account.return_value = mock_account

        assert broker.get_cash() == 50000.0


# Integration tests (require real credentials, skip in CI)
@pytest.mark.skipif(
    not all([
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_API_SECRET"),
    ]),
    reason="Alpaca credentials not available"
)
class TestAlpacaBrokerIntegration:
    """Integration tests with real Alpaca API."""

    def test_connect_paper(self):
        """Test paper trading connection."""
        broker = AlpacaBroker(paper=True)
        account = broker.get_account()
        assert "account_id" in account

    def test_get_positions_real(self):
        """Test getting real positions."""
        broker = AlpacaBroker(paper=True)
        positions = broker.get_positions()
        assert isinstance(positions, dict)
```

## SST Compliance

- [ ] Credentials from environment variables only
- [ ] Uses get_cfg() for configuration
- [ ] All methods return dicts with sorted keys where applicable
- [ ] Comprehensive error handling with typed exceptions
- [ ] Timezone-aware timestamps

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `brokers/alpaca.py` | 300 |
| `tests/test_alpaca_broker.py` | 150 |
| **Total** | ~450 |
