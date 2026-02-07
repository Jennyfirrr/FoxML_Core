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
- Context manager support

SST Compliance:
- Configuration via get_cfg()
- No hardcoded connection params
- Comprehensive error handling
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    ConnectionError as BrokerConnectionError,
    InsufficientFundsError,
    OrderRejectedError,
)

logger = logging.getLogger(__name__)


def _get_ib():
    """Lazy import of IB class to handle missing dependency."""
    try:
        from ib_insync import IB

        return IB
    except ImportError:
        raise ImportError(
            "ib_insync package required for IBKR broker. "
            "Install with: pip install ib_insync"
        )


def _get_stock():
    """Lazy import of Stock contract class."""
    try:
        from ib_insync import Stock

        return Stock
    except ImportError:
        raise ImportError(
            "ib_insync package required for IBKR broker. "
            "Install with: pip install ib_insync"
        )


def _get_order_classes():
    """Lazy import of order classes."""
    try:
        from ib_insync import LimitOrder, MarketOrder

        return MarketOrder, LimitOrder
    except ImportError:
        raise ImportError(
            "ib_insync package required for IBKR broker. "
            "Install with: pip install ib_insync"
        )


class IBKRBroker:
    """
    Interactive Brokers broker implementation.

    Connects to TWS or IB Gateway for order execution.

    Example:
        >>> broker = IBKRBroker()  # Connects to local TWS
        >>> broker.connect()
        >>> broker.submit_order("AAPL", "BUY", 10)

        # Or use as context manager:
        >>> with IBKRBroker() as broker:
        ...     broker.submit_order("AAPL", "BUY", 10)
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
            host: TWS/Gateway host (default: from config or 127.0.0.1)
            port: TWS/Gateway port (default: from config or 7497 for paper)
            client_id: Client ID for connection (default: from config or 1)
            timeout: Connection timeout in seconds
        """
        self._host = host or get_cfg(
            "live_trading.brokers.ibkr.host", default="127.0.0.1"
        )
        self._port = port or get_cfg("live_trading.brokers.ibkr.port", default=7497)
        self._client_id = client_id or get_cfg(
            "live_trading.brokers.ibkr.client_id", default=1
        )
        self._timeout = timeout

        self._ib = None
        self._connected = False

        logger.info(f"IBKRBroker initialized: {self._host}:{self._port}")

    def connect(self) -> None:
        """
        Connect to TWS/Gateway.

        Raises:
            BrokerConnectionError: If connection fails
        """
        if self._connected:
            return

        IB = _get_ib()

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
            try:
                self._ib.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _ensure_connected(self) -> None:
        """Ensure we're connected, auto-connecting if needed."""
        if not self._connected or not self._ib:
            self.connect()

    def _create_contract(self, symbol: str, sec_type: str = "STK"):
        """
        Create IB contract for symbol.

        Args:
            symbol: Trading symbol
            sec_type: Security type (default: STK for stocks)

        Returns:
            ib_insync Contract object

        Raises:
            BrokerError: If security type not supported
        """
        Stock = _get_stock()

        if sec_type == "STK":
            return Stock(symbol.upper(), "SMART", "USD")
        else:
            raise BrokerError(f"Unsupported security type: {sec_type}")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit an order to IBKR.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            **kwargs: Additional order parameters

        Returns:
            Order result dict

        Raises:
            BrokerError: If submission fails
            OrderRejectedError: If order rejected
            InsufficientFundsError: If insufficient funds
        """
        self._ensure_connected()
        MarketOrder, LimitOrder = _get_order_classes()

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

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "insufficient" in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds for {symbol}: {e}")
            elif "rejected" in error_msg.lower():
                raise OrderRejectedError(symbol, error_msg)
            else:
                raise BrokerError(f"Order submission failed: {e}")

    def _trade_to_dict(self, trade) -> Dict[str, Any]:
        """Convert IB Trade to dict."""
        order = trade.order
        fills = trade.fills

        fill_price = None
        filled_qty = 0
        if fills:
            filled_qty = sum(f.execution.shares for f in fills)
            if filled_qty > 0:
                fill_price = (
                    sum(f.execution.shares * f.execution.avgPrice for f in fills)
                    / filled_qty
                )

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
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation result dict

        Raises:
            BrokerError: If cancellation fails
        """
        self._ensure_connected()

        try:
            # Find the trade
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return {
                        "order_id": order_id,
                        "status": "cancelled",
                        "timestamp": datetime.now(timezone.utc),
                    }

            raise BrokerError(f"Order not found: {order_id}")
        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Cancel failed for {order_id}: {e}")

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.

        Args:
            order_id: Order ID to query

        Returns:
            Order status dict

        Raises:
            BrokerError: If query fails
        """
        self._ensure_connected()

        try:
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    return self._trade_to_dict(trade)

            raise BrokerError(f"Order not found: {order_id}")
        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get order {order_id}: {e}")

    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions (symbol -> quantity).

        Returns:
            Dict mapping symbol to position size

        Raises:
            BrokerError: If retrieval fails
        """
        self._ensure_connected()

        try:
            positions = self._ib.positions()
            return {
                p.contract.symbol: float(p.position) for p in positions if p.position != 0
            }
        except Exception as e:
            raise BrokerError(f"Failed to get positions: {e}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position dict or None if no position

        Raises:
            BrokerError: If retrieval fails
        """
        self._ensure_connected()

        try:
            positions = self._ib.positions()
            for p in positions:
                if p.contract.symbol == symbol.upper() and p.position != 0:
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
        """
        Get available cash.

        Returns:
            Available cash balance (USD)

        Raises:
            BrokerError: If retrieval fails
        """
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
        """
        Get buying power.

        Returns:
            Available buying power

        Raises:
            BrokerError: If retrieval fails
        """
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
        """
        Get full account info.

        Returns:
            Account details dict

        Raises:
            BrokerError: If retrieval fails
        """
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
        """
        Get recent fills.

        Args:
            since: Get fills since this time (UTC)

        Returns:
            List of fill dicts

        Raises:
            BrokerError: If retrieval fails
        """
        self._ensure_connected()

        try:
            fills = self._ib.fills()
            result = []
            for fill in fills:
                fill_time = fill.execution.time
                if since and fill_time and fill_time < since:
                    continue
                result.append(
                    {
                        "order_id": str(fill.execution.orderId),
                        "symbol": fill.contract.symbol,
                        "side": fill.execution.side,
                        "qty": float(fill.execution.shares),
                        "fill_price": float(fill.execution.avgPrice),
                        "filled_at": fill_time.isoformat() if fill_time else None,
                        "timestamp": fill_time or datetime.now(timezone.utc),
                        "fee": 0.0,  # Fee info not directly available
                    }
                )
            return result
        except Exception as e:
            raise BrokerError(f"Failed to get fills: {e}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote dict with bid/ask info

        Raises:
            BrokerError: If quote unavailable
        """
        self._ensure_connected()

        try:
            contract = self._create_contract(symbol)
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(1)  # Wait for data

            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 0

            self._ib.cancelMktData(contract)

            return {
                "symbol": symbol.upper(),
                "bid": bid,
                "ask": ask,
                "bid_size": int(ticker.bidSize) if ticker.bidSize else 0,
                "ask_size": int(ticker.askSize) if ticker.askSize else 0,
                "timestamp": datetime.now(timezone.utc),
                "spread_bps": spread_bps,
            }
        except Exception as e:
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    def now(self) -> datetime:
        """
        Get current broker time.

        Returns:
            Current UTC timestamp
        """
        return datetime.now(timezone.utc)

    def is_market_open(self) -> bool:
        """
        Check if market is open (basic check).

        Returns:
            True if US market hours (approximate)
        """
        now = datetime.now(timezone.utc)
        # US market hours: 9:30 AM - 4:00 PM ET
        # This is a simplified check (13:30 - 21:00 UTC)
        hour = now.hour
        minute = now.minute

        if hour < 13 or hour >= 21:
            return False
        if hour == 13 and minute < 30:
            return False
        return True

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading (port 7497)."""
        return self._port == 7497

    @property
    def is_connected(self) -> bool:
        """Check if connected to TWS/Gateway."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
