# Plan 13: IBKR Broker

## Overview

Interactive Brokers integration via ib_insync library for TWS/Gateway connection.

## Prerequisites

- TWS or IB Gateway running locally
- API connections enabled in TWS settings
- Paper trading account (port 7497) or live (port 7496)

## Files to Create

### 1. `LIVE_TRADING/brokers/ibkr.py`

```python
"""
Interactive Brokers Implementation
==================================

Live trading via TWS/IB Gateway using ib_insync.

Features:
- Async connection management
- Order submission (market, limit)
- Position tracking
- Account info
- Real-time fills

SST Compliance:
- Configuration via get_cfg()
- No hardcoded connection params
- Comprehensive error handling
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, Trade, Contract

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.brokers.interface import Broker
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
    ConnectionError as BrokerConnectionError,
)

logger = logging.getLogger(__name__)


class IBKRBroker(Broker):
    """
    Interactive Brokers broker implementation.

    Connects to TWS or IB Gateway for order execution.

    Example:
        >>> broker = IBKRBroker()  # Connects to local TWS
        >>> broker.connect()
        >>> broker.submit_order("AAPL", "BUY", 10)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize IBKR broker.

        Args:
            host: TWS/Gateway host (default: from config)
            port: TWS/Gateway port (default: from config)
            client_id: Client ID for connection (default: from config)
            timeout: Connection timeout in seconds
        """
        self._host = host or get_cfg("live_trading.brokers.ibkr.host", default="127.0.0.1")
        self._port = port or get_cfg("live_trading.brokers.ibkr.port", default=7497)
        self._client_id = client_id or get_cfg("live_trading.brokers.ibkr.client_id", default=1)
        self._timeout = timeout

        self._ib: Optional[IB] = None
        self._connected = False

        logger.info(f"IBKRBroker initialized: {self._host}:{self._port}")

    def connect(self) -> None:
        """Connect to TWS/Gateway."""
        if self._connected:
            return

        try:
            self._ib = IB()
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                timeout=self._timeout,
            )
            self._connected = True
            logger.info(f"Connected to IBKR at {self._host}:{self._port}")
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to IBKR: {e}")

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _ensure_connected(self) -> None:
        """Ensure we're connected."""
        if not self._connected or not self._ib:
            self.connect()

    def _create_contract(self, symbol: str, sec_type: str = "STK") -> Contract:
        """Create IB contract for symbol."""
        if sec_type == "STK":
            return Stock(symbol, "SMART", "USD")
        else:
            raise BrokerError(f"Unsupported security type: {sec_type}")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Submit an order to IBKR."""
        self._ensure_connected()

        try:
            contract = self._create_contract(symbol)

            # Qualify contract
            self._ib.qualifyContracts(contract)

            action = "BUY" if side.upper() == "BUY" else "SELL"

            if order_type.lower() == "market":
                order = MarketOrder(action, qty)
            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise BrokerError("Limit price required for limit orders")
                order = LimitOrder(action, qty, limit_price)
            else:
                raise BrokerError(f"Unsupported order type: {order_type}")

            trade = self._ib.placeOrder(contract, order)

            # Wait briefly for initial fill
            self._ib.sleep(0.5)

            return self._trade_to_dict(trade)

        except Exception as e:
            error_msg = str(e)
            if "insufficient" in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds for {symbol}: {e}")
            elif "rejected" in error_msg.lower():
                raise OrderRejectedError(symbol, error_msg)
            else:
                raise BrokerError(f"Order submission failed: {e}")

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert IB Trade to dict."""
        order = trade.order
        fills = trade.fills

        fill_price = None
        filled_qty = 0
        if fills:
            filled_qty = sum(f.execution.shares for f in fills)
            if filled_qty > 0:
                fill_price = sum(
                    f.execution.shares * f.execution.avgPrice for f in fills
                ) / filled_qty

        return {
            "order_id": str(order.orderId),
            "perm_id": str(order.permId) if order.permId else None,
            "symbol": trade.contract.symbol,
            "side": order.action,
            "qty": float(order.totalQuantity),
            "order_type": order.orderType,
            "status": trade.orderStatus.status,
            "filled_qty": float(filled_qty),
            "fill_price": float(fill_price) if fill_price else None,
            "timestamp": datetime.now(timezone.utc),
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        self._ensure_connected()

        try:
            # Find the trade
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return {"order_id": order_id, "status": "cancelled"}

            raise BrokerError(f"Order not found: {order_id}")
        except Exception as e:
            raise BrokerError(f"Cancel failed for {order_id}: {e}")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        self._ensure_connected()

        try:
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    return self._trade_to_dict(trade)

            raise BrokerError(f"Order not found: {order_id}")
        except Exception as e:
            raise BrokerError(f"Failed to get order {order_id}: {e}")

    def get_positions(self) -> Dict[str, float]:
        """Get current positions (symbol -> quantity)."""
        self._ensure_connected()

        try:
            positions = self._ib.positions()
            return {
                p.contract.symbol: float(p.position)
                for p in positions
                if p.position != 0
            }
        except Exception as e:
            raise BrokerError(f"Failed to get positions: {e}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        self._ensure_connected()

        try:
            positions = self._ib.positions()
            for p in positions:
                if p.contract.symbol == symbol and p.position != 0:
                    return {
                        "symbol": p.contract.symbol,
                        "qty": float(p.position),
                        "avg_cost": float(p.avgCost),
                        "market_value": float(p.position * p.avgCost),
                    }
            return None
        except Exception as e:
            raise BrokerError(f"Failed to get position {symbol}: {e}")

    def get_cash(self) -> float:
        """Get available cash."""
        self._ensure_connected()

        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.tag == "CashBalance" and av.currency == "USD":
                    return float(av.value)
            return 0.0
        except Exception as e:
            raise BrokerError(f"Failed to get cash: {e}")

    def get_buying_power(self) -> float:
        """Get buying power."""
        self._ensure_connected()

        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.tag == "BuyingPower":
                    return float(av.value)
            return 0.0
        except Exception as e:
            raise BrokerError(f"Failed to get buying power: {e}")

    def get_account(self) -> Dict[str, Any]:
        """Get full account info."""
        self._ensure_connected()

        try:
            summary = self._ib.accountSummary()
            result = {}
            for item in summary:
                result[item.tag] = item.value
            return result
        except Exception as e:
            raise BrokerError(f"Failed to get account: {e}")

    def get_fills(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get recent fills."""
        self._ensure_connected()

        try:
            fills = self._ib.fills()
            result = []
            for fill in fills:
                fill_time = fill.execution.time
                if since and fill_time < since:
                    continue
                result.append({
                    "order_id": str(fill.execution.orderId),
                    "symbol": fill.contract.symbol,
                    "side": fill.execution.side,
                    "qty": float(fill.execution.shares),
                    "price": float(fill.execution.avgPrice),
                    "filled_at": fill_time.isoformat() if fill_time else None,
                })
            return result
        except Exception as e:
            raise BrokerError(f"Failed to get fills: {e}")

    def now(self) -> datetime:
        """Get current broker time."""
        self._ensure_connected()
        return datetime.now(timezone.utc)

    def is_market_open(self) -> bool:
        """Check if market is open (basic check)."""
        now = datetime.now(timezone.utc)
        # US market hours: 9:30 AM - 4:00 PM ET
        # This is a simplified check
        hour = now.hour
        return 13 <= hour < 21  # Rough UTC equivalent

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading (port 7497)."""
        return self._port == 7497

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
```

### 2. Update `LIVE_TRADING/brokers/__init__.py`

```python
# Add to existing exports
from .ibkr import IBKRBroker

__all__ = [
    "Broker",
    "PaperBroker",
    "AlpacaBroker",
    "IBKRBroker",
    "get_broker",
]

def get_broker(broker_type: str, **kwargs) -> Broker:
    """Factory function for brokers."""
    if broker_type == "paper":
        return PaperBroker(**kwargs)
    elif broker_type == "alpaca":
        return AlpacaBroker(**kwargs)
    elif broker_type == "ibkr":
        return IBKRBroker(**kwargs)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
```

## Tests

### `LIVE_TRADING/tests/test_ibkr_broker.py`

```python
"""
IBKR Broker Tests
=================

Unit tests with mocked IB connection.
Integration tests require running TWS/Gateway.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from LIVE_TRADING.brokers.ibkr import IBKRBroker
from LIVE_TRADING.common.exceptions import BrokerError


class TestIBKRBrokerInit:
    """Tests for broker initialization."""

    @patch("LIVE_TRADING.brokers.ibkr.IB")
    def test_init_defaults(self, mock_ib):
        """Test initialization with defaults."""
        broker = IBKRBroker()
        assert broker._host == "127.0.0.1"
        assert broker._port == 7497
        assert broker._client_id == 1

    @patch("LIVE_TRADING.brokers.ibkr.IB")
    def test_init_custom(self, mock_ib):
        """Test initialization with custom values."""
        broker = IBKRBroker(host="192.168.1.1", port=7496, client_id=5)
        assert broker._host == "192.168.1.1"
        assert broker._port == 7496
        assert broker._client_id == 5


class TestIBKRBrokerOperations:
    """Tests for broker operations."""

    @pytest.fixture
    def broker(self):
        """Create broker with mocked IB."""
        with patch("LIVE_TRADING.brokers.ibkr.IB") as mock_ib:
            broker = IBKRBroker()
            broker._ib = Mock()
            broker._connected = True
            return broker

    def test_get_positions(self, broker):
        """Test position retrieval."""
        mock_pos = Mock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.position = 100

        broker._ib.positions.return_value = [mock_pos]

        positions = broker.get_positions()
        assert positions == {"AAPL": 100.0}

    def test_get_cash(self, broker):
        """Test cash retrieval."""
        mock_av = Mock()
        mock_av.tag = "CashBalance"
        mock_av.currency = "USD"
        mock_av.value = "50000.00"

        broker._ib.accountValues.return_value = [mock_av]

        assert broker.get_cash() == 50000.0

    def test_is_paper_port_7497(self, broker):
        """Test paper detection for port 7497."""
        broker._port = 7497
        assert broker.is_paper is True

    def test_is_paper_port_7496(self, broker):
        """Test paper detection for port 7496."""
        broker._port = 7496
        assert broker.is_paper is False


# Integration tests (require TWS/Gateway)
@pytest.mark.skipif(True, reason="Requires TWS/Gateway running")
class TestIBKRBrokerIntegration:
    """Integration tests with real TWS/Gateway."""

    def test_connect(self):
        """Test connection to TWS."""
        with IBKRBroker() as broker:
            assert broker._connected

    def test_get_account(self):
        """Test account retrieval."""
        with IBKRBroker() as broker:
            account = broker.get_account()
            assert isinstance(account, dict)
```

## Configuration

Add to `CONFIG/live_trading/live_trading.yaml`:

```yaml
live_trading:
  brokers:
    ibkr:
      host: "127.0.0.1"
      port: 7497        # Paper: 7497, Live: 7496
      client_id: 1
      timeout: 30.0
```

## SST Compliance

- [ ] Configuration via get_cfg()
- [ ] Comprehensive error handling
- [ ] Context manager support for clean disconnect
- [ ] All methods return dicts with consistent structure

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `brokers/ibkr.py` | 350 |
| `tests/test_ibkr_broker.py` | 150 |
| **Total** | ~500 |
