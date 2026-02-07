"""
Broker Interface Protocol
=========================

Protocol definition for live trading broker adapters.
All broker implementations must satisfy this protocol.

This uses Python's Protocol for structural subtyping, allowing
any class that implements the required methods to be used as a Broker
without explicit inheritance.

SST Compliance:
- Protocol-based for testability and swappability
- No hardcoded values
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Broker(Protocol):
    """
    Protocol for live trading broker adapters.

    All broker implementations must provide these methods.
    The protocol is runtime_checkable for isinstance() support.

    Example:
        >>> broker = get_broker("paper")
        >>> isinstance(broker, Broker)
        True
    """

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
            Dict with:
                - order_id: Unique order identifier
                - status: "filled", "pending", "rejected"
                - fill_price: Execution price (if filled)
                - qty: Executed quantity
                - timestamp: Execution time

        Raises:
            BrokerError: If order submission fails
            OrderRejectedError: If order is rejected
            InsufficientFundsError: If insufficient cash
        """
        ...

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict with:
                - order_id: The order ID
                - status: "cancelled" or "filled" (if already filled)
                - timestamp: Cancellation time

        Raises:
            BrokerError: If cancellation fails
        """
        ...

    def get_positions(self) -> dict[str, float]:
        """
        Get current positions by symbol.

        Returns:
            Dict mapping symbol to position size.
            Positive = long, negative = short.

        Example:
            >>> broker.get_positions()
            {"AAPL": 100.0, "SPY": -50.0}
        """
        ...

    def get_cash(self) -> float:
        """
        Get available cash balance.

        Returns:
            Available cash amount in the account.
        """
        ...

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """
        Get recent fills.

        Args:
            since: Get fills since this timestamp (UTC).
                   If None, return all fills for the session.

        Returns:
            List of fill dicts, each containing:
                - order_id: Order identifier
                - symbol: Trading symbol
                - side: "BUY" or "SELL"
                - qty: Filled quantity
                - fill_price: Execution price
                - timestamp: Execution time
                - fee: Commission/fee amount
        """
        ...

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with:
                - symbol: The symbol
                - bid: Best bid price
                - ask: Best ask price
                - bid_size: Bid quantity
                - ask_size: Ask quantity
                - timestamp: Quote time
                - spread_bps: Spread in basis points

        Raises:
            BrokerError: If quote unavailable
        """
        ...

    def now(self) -> datetime:
        """
        Get current broker time (UTC).

        Returns:
            Current timestamp from broker's perspective.
            For paper trading, this is system time.
            For live trading, this should be exchange time.
        """
        ...


def get_broker(venue: str = "paper", **kwargs: Any) -> Broker:
    """
    Factory function to get broker instance.

    Args:
        venue: Broker venue identifier:
            - "paper": Paper trading broker (simulation)
            - "ibkr": Interactive Brokers (not yet implemented)
            - "alpaca": Alpaca Markets (not yet implemented)
        **kwargs: Broker-specific configuration:
            - initial_cash: Starting cash (paper only)
            - slippage_bps: Simulated slippage (paper only)
            - fee_bps: Simulated fees (paper only)

    Returns:
        Broker instance implementing the Broker protocol.

    Raises:
        ValueError: If venue is unknown
        NotImplementedError: If venue is not yet implemented

    Example:
        >>> broker = get_broker("paper", initial_cash=100_000)
        >>> broker.get_cash()
        100000.0
    """
    if venue == "paper":
        from .paper import PaperBroker

        return PaperBroker(**kwargs)
    elif venue == "ibkr":
        from .ibkr import IBKRBroker

        return IBKRBroker(**kwargs)
    elif venue == "alpaca":
        from .alpaca import AlpacaBroker

        return AlpacaBroker(**kwargs)
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

    Ensures consistent casing and types for order fields.

    Args:
        symbol: Trading symbol (will be uppercased)
        side: "BUY" or "SELL" (will be uppercased)
        qty: Quantity (will be converted to float)
        px: Price (optional, will be converted to float if provided)

    Returns:
        Normalized order dict with:
            - symbol: Uppercased symbol
            - side: Uppercased side
            - qty: Float quantity
            - px: Float price or None

    Example:
        >>> normalize_order("aapl", "buy", 100, 150.50)
        {"symbol": "AAPL", "side": "BUY", "qty": 100.0, "px": 150.50}
    """
    return {
        "symbol": str(symbol).upper(),
        "side": str(side).upper(),
        "qty": float(qty),
        "px": float(px) if px is not None else None,
    }


def validate_side(side: str) -> str:
    """
    Validate and normalize order side.

    Args:
        side: Order side string

    Returns:
        Normalized side ("BUY" or "SELL")

    Raises:
        ValueError: If side is invalid
    """
    side = str(side).upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order side: {side}. Must be 'BUY' or 'SELL'.")
    return side


def validate_order_type(order_type: str) -> str:
    """
    Validate and normalize order type.

    Args:
        order_type: Order type string

    Returns:
        Normalized order type ("market" or "limit")

    Raises:
        ValueError: If order type is invalid
    """
    order_type = str(order_type).lower()
    if order_type not in ("market", "limit"):
        raise ValueError(
            f"Invalid order type: {order_type}. Must be 'market' or 'limit'."
        )
    return order_type
