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
- Quote retrieval
- H4 FIX: Rate limiting to prevent API throttling

SST Compliance:
- Credentials from environment variables
- No hardcoded API keys
- Uses get_cfg() for configuration
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    ConnectionError as BrokerConnectionError,
    InsufficientFundsError,
    OrderRejectedError,
)
from LIVE_TRADING.common.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Alpaca rate limits:
# - 200 requests/minute for orders
# - Higher limits for data endpoints
DEFAULT_ORDER_RATE_LIMIT = 180  # Stay under 200 with margin
DEFAULT_DATA_RATE_LIMIT = 200


def _get_trading_client():
    """Lazy import of TradingClient to handle missing dependency."""
    try:
        from alpaca.trading.client import TradingClient

        return TradingClient
    except ImportError:
        raise ImportError(
            "alpaca-py package required for Alpaca broker. "
            "Install with: pip install alpaca-py"
        )


def _get_data_client():
    """Lazy import of StockHistoricalDataClient for quotes."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient

        return StockHistoricalDataClient
    except ImportError:
        raise ImportError(
            "alpaca-py package required for Alpaca broker. "
            "Install with: pip install alpaca-py"
        )


def _get_order_classes():
    """Lazy import of order request classes."""
    try:
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        return MarketOrderRequest, LimitOrderRequest
    except ImportError:
        raise ImportError(
            "alpaca-py package required for Alpaca broker. "
            "Install with: pip install alpaca-py"
        )


def _get_trading_enums():
    """Lazy import of trading enums."""
    try:
        from alpaca.trading.enums import OrderSide, TimeInForce

        return OrderSide, TimeInForce
    except ImportError:
        raise ImportError(
            "alpaca-py package required for Alpaca broker. "
            "Install with: pip install alpaca-py"
        )


def _get_quote_request():
    """Lazy import of quote request class."""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest

        return StockLatestQuoteRequest
    except ImportError:
        raise ImportError(
            "alpaca-py package required for Alpaca broker. "
            "Install with: pip install alpaca-py"
        )


class AlpacaBroker:
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
        api_key_env = get_cfg(
            "live_trading.brokers.alpaca.api_key_env", default="ALPACA_API_KEY"
        )
        api_secret_env = get_cfg(
            "live_trading.brokers.alpaca.api_secret_env", default="ALPACA_API_SECRET"
        )

        self._api_key = api_key or os.environ.get(api_key_env)
        self._api_secret = api_secret or os.environ.get(api_secret_env)

        if not self._api_key or not self._api_secret:
            raise BrokerConnectionError(
                f"Alpaca credentials not found. Set {api_key_env} and {api_secret_env}"
            )

        self._paper = paper

        # Initialize trading client
        TradingClient = _get_trading_client()
        try:
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
                paper=paper,
            )
        except Exception as e:
            raise BrokerConnectionError(f"Failed to connect to Alpaca: {e}")

        # Initialize data client for quotes
        StockHistoricalDataClient = _get_data_client()
        try:
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        except Exception as e:
            logger.warning(f"Data client init failed (quotes may not work): {e}")
            self._data_client = None

        # H4 FIX: Rate limiters for API calls
        order_rate_limit = get_cfg(
            "live_trading.brokers.alpaca.order_rate_limit",
            default=DEFAULT_ORDER_RATE_LIMIT,
        )
        data_rate_limit = get_cfg(
            "live_trading.brokers.alpaca.data_rate_limit",
            default=DEFAULT_DATA_RATE_LIMIT,
        )
        self._order_rate_limiter = RateLimiter(requests_per_minute=order_rate_limit)
        self._data_rate_limiter = RateLimiter(requests_per_minute=data_rate_limit)

        logger.info(
            f"AlpacaBroker initialized (paper={paper}, "
            f"order_rate={order_rate_limit}/min, data_rate={data_rate_limit}/min)"
        )

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
        Submit an order to Alpaca.

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
        MarketOrderRequest, LimitOrderRequest = _get_order_classes()
        OrderSide, TimeInForce = _get_trading_enums()

        # H4 FIX: Acquire rate limit before API call
        self._order_rate_limiter.acquire()

        try:
            alpaca_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

            if order_type.lower() == "market":
                request = MarketOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise BrokerError("Limit price required for limit orders")
                request = LimitOrderRequest(
                    symbol=symbol.upper(),
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
                "qty": float(order.qty) if order.qty else qty,
                "order_type": order.type.value if order.type else order_type,
                "status": order.status.value if order.status else "pending",
                "submitted_at": (
                    order.submitted_at.isoformat() if order.submitted_at else None
                ),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "fill_price": (
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
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
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation result dict

        Raises:
            BrokerError: If cancellation fails
        """
        try:
            self._client.cancel_order_by_id(order_id)
            return {
                "order_id": order_id,
                "status": "cancelled",
                "timestamp": datetime.now(timezone.utc),
            }
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
        try:
            order = self._client.get_order_by_id(order_id)
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": float(order.qty) if order.qty else 0,
                "status": order.status.value if order.status else "unknown",
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "fill_price": (
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
            }
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
        try:
            positions = self._client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
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
        try:
            pos = self._client.get_open_position(symbol.upper())
            return {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "side": pos.side.value if pos.side else "long",
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
        """
        Get available cash.

        Returns:
            Available cash balance

        Raises:
            BrokerError: If retrieval fails
        """
        try:
            account = self._client.get_account()
            return float(account.cash)
        except Exception as e:
            raise BrokerError(f"Failed to get account: {e}")

    def get_buying_power(self) -> float:
        """
        Get buying power.

        Returns:
            Available buying power

        Raises:
            BrokerError: If retrieval fails
        """
        try:
            account = self._client.get_account()
            return float(account.buying_power)
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
        try:
            account = self._client.get_account()
            return {
                "account_id": str(account.id),
                "status": account.status.value if account.status else "unknown",
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
        """
        Get recent fills.

        Args:
            since: Get fills since this time (UTC)

        Returns:
            List of fill dicts

        Raises:
            BrokerError: If retrieval fails
        """
        try:
            kwargs = {"status": "filled", "limit": 100}
            if since:
                kwargs["after"] = since

            orders = self._client.get_orders(**kwargs)
            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.filled_qty) if o.filled_qty else 0,
                    "fill_price": (
                        float(o.filled_avg_price) if o.filled_avg_price else None
                    ),
                    "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                    "timestamp": o.filled_at or datetime.now(timezone.utc),
                    "fee": 0.0,  # Alpaca has no commission
                }
                for o in orders
            ]
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
        if self._data_client is None:
            raise BrokerError("Data client not available for quotes")

        # H4 FIX: Acquire rate limit before API call
        self._data_rate_limiter.acquire()

        try:
            StockLatestQuoteRequest = _get_quote_request()
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
            quotes = self._data_client.get_stock_latest_quote(request)

            if symbol.upper() not in quotes:
                raise BrokerError(f"No quote available for {symbol}")

            quote = quotes[symbol.upper()]

            bid = float(quote.bid_price) if quote.bid_price else 0
            ask = float(quote.ask_price) if quote.ask_price else 0
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 0

            return {
                "symbol": symbol.upper(),
                "bid": bid,
                "ask": ask,
                "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                "timestamp": quote.timestamp if quote.timestamp else datetime.now(timezone.utc),
                "spread_bps": spread_bps,
            }
        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    def now(self) -> datetime:
        """
        Get current broker time.

        Returns:
            Current UTC timestamp from broker
        """
        try:
            clock = self._client.get_clock()
            return clock.timestamp.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def is_market_open(self) -> bool:
        """
        Check if market is open.

        Returns:
            True if market is currently open
        """
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception:
            return False

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self._paper
