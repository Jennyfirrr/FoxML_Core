"""
Backtest Broker
===============

Simulated broker for backtesting with realistic fills.

Features:
- Slippage models (fixed, volume-based)
- Transaction costs
- Position tracking
- Immediate fill simulation

SST Compliance:
- Implements Broker protocol
- Configurable via get_cfg()
- All times UTC
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.clock import Clock, SimulatedClock
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class BacktestBroker:
    """
    Simulated broker for backtesting.

    Models realistic execution with slippage and fees.
    Implements the Broker protocol for compatibility with TradingEngine.

    Example:
        >>> broker = BacktestBroker(initial_cash=100_000)
        >>> broker.set_quote("AAPL", Quote(symbol="AAPL", bid=149.9, ask=150.1, ...))
        >>> result = broker.submit_order("AAPL", "BUY", 100)
        >>> print(result["fill_price"])
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: Optional[float] = None,
        fee_bps: Optional[float] = None,
        slippage_model: Optional[str] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize backtest broker.

        Args:
            initial_cash: Starting capital
            slippage_bps: Slippage in basis points (default from config)
            fee_bps: Transaction fee in basis points (default from config)
            slippage_model: Slippage model (fixed, volume_based)
            clock: Clock for time (default: SimulatedClock)
        """
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._slippage_bps = slippage_bps or get_cfg(
            "live_trading.backtest.slippage_bps", default=5.0
        )
        self._fee_bps = fee_bps or get_cfg("live_trading.paper.fee_bps", default=1.0)
        self._slippage_model = slippage_model or get_cfg(
            "live_trading.backtest.slippage_model", default="fixed"
        )

        # Initialize clock
        if clock is None:
            self._clock = SimulatedClock(datetime.now(timezone.utc))
        else:
            self._clock = clock

        # Position tracking
        self._positions: Dict[str, float] = {}  # symbol -> shares
        self._position_costs: Dict[str, float] = {}  # symbol -> avg cost basis
        self._fills: List[Dict[str, Any]] = []
        self._order_id = 0

        # Current quotes (set by backtest engine)
        self._quotes: Dict[str, Quote] = {}

    # =========================================================================
    # Quote/Time Management (Backtest-specific)
    # =========================================================================

    def set_quote(self, symbol: str, quote: Quote) -> None:
        """
        Set current quote for symbol.

        Called by BacktestEngine before each simulation step.

        Args:
            symbol: Trading symbol
            quote: Current quote data
        """
        self._quotes[symbol] = quote

    def set_time(self, timestamp: datetime) -> None:
        """
        Set current simulation time.

        Called by BacktestEngine to advance time.

        Args:
            timestamp: New simulation time (UTC)
        """
        if isinstance(self._clock, SimulatedClock):
            self._clock.set_time(timestamp)

    # =========================================================================
    # Broker Protocol Implementation
    # =========================================================================

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit and immediately fill an order.

        In backtesting, orders are filled immediately at the simulated price.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade
            order_type: "market" or "limit"
            limit_price: Limit price (for limit orders)

        Returns:
            Dict with order result
        """
        self._order_id += 1
        order_id = f"bt_{self._order_id}"
        timestamp = self._clock.now()

        # Get quote for this symbol
        quote = self._quotes.get(symbol)
        if not quote:
            logger.warning(f"No quote available for {symbol}")
            return {
                "order_id": order_id,
                "status": "rejected",
                "reason": "no_quote",
                "symbol": symbol,
                "timestamp": timestamp,
            }

        # Calculate fill price with slippage
        side_upper = side.upper()
        if side_upper == "BUY":
            base_price = quote.ask
            slippage = self._calculate_slippage(base_price, qty, quote)
            fill_price = base_price + slippage

            # Check limit price for limit orders
            if order_type == "limit" and limit_price is not None:
                if fill_price > limit_price:
                    return {
                        "order_id": order_id,
                        "status": "rejected",
                        "reason": "limit_price_not_met",
                        "symbol": symbol,
                        "limit_price": limit_price,
                        "market_price": fill_price,
                        "timestamp": timestamp,
                    }
        else:
            base_price = quote.bid
            slippage = self._calculate_slippage(base_price, qty, quote)
            fill_price = base_price - slippage

            # Check limit price for limit orders
            if order_type == "limit" and limit_price is not None:
                if fill_price < limit_price:
                    return {
                        "order_id": order_id,
                        "status": "rejected",
                        "reason": "limit_price_not_met",
                        "symbol": symbol,
                        "limit_price": limit_price,
                        "market_price": fill_price,
                        "timestamp": timestamp,
                    }

        # Calculate costs
        notional = qty * fill_price
        fee = notional * self._fee_bps / 10000

        # Check funds for buy
        if side_upper == "BUY":
            total_cost = notional + fee
            if total_cost > self._cash:
                return {
                    "order_id": order_id,
                    "status": "rejected",
                    "reason": "insufficient_funds",
                    "symbol": symbol,
                    "required": total_cost,
                    "available": self._cash,
                    "timestamp": timestamp,
                }
            self._cash -= total_cost
        else:
            # Check we have shares to sell
            current_qty = self._positions.get(symbol, 0)
            if qty > current_qty:
                return {
                    "order_id": order_id,
                    "status": "rejected",
                    "reason": "insufficient_shares",
                    "symbol": symbol,
                    "required": qty,
                    "available": current_qty,
                    "timestamp": timestamp,
                }
            self._cash += notional - fee

        # Update position
        current_qty = self._positions.get(symbol, 0)
        if side_upper == "BUY":
            new_qty = current_qty + qty
            # Update average cost
            current_cost = self._position_costs.get(symbol, 0)
            if new_qty > 0:
                total_cost_basis = current_cost * current_qty + notional
                self._position_costs[symbol] = total_cost_basis / new_qty
            else:
                self._position_costs[symbol] = 0
        else:
            new_qty = current_qty - qty

        # Update or clear position
        if abs(new_qty) < 1e-8:
            self._positions.pop(symbol, None)
            self._position_costs.pop(symbol, None)
        else:
            self._positions[symbol] = new_qty

        # Record fill
        fill = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side_upper,
            "qty": qty,
            "fill_price": fill_price,
            "fee": fee,
            "timestamp": timestamp,
            "slippage_bps": (slippage / base_price) * 10000 if base_price > 0 else 0,
        }
        self._fills.append(fill)

        logger.debug(f"Filled {side_upper} {qty} {symbol} @ {fill_price:.2f}")

        return {
            "order_id": order_id,
            "status": "filled",
            "symbol": symbol,
            "side": side_upper,
            "qty": qty,
            "fill_price": fill_price,
            "fee": fee,
            "timestamp": timestamp,
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order (not supported in backtest - immediate fills).

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict indicating order not found
        """
        return {
            "order_id": order_id,
            "status": "not_found",
            "message": "Backtest broker fills immediately",
            "timestamp": self._clock.now(),
        }

    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions.

        Returns:
            Dict mapping symbol to shares held
        """
        return dict(self._positions)

    def get_cash(self) -> float:
        """
        Get available cash.

        Returns:
            Available cash balance
        """
        return self._cash

    def get_fills(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get fills since timestamp.

        Args:
            since: Get fills after this time (None = all)

        Returns:
            List of fill dicts
        """
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f["timestamp"] >= since]

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote as dict
        """
        quote = self._quotes.get(symbol)
        if not quote:
            return {
                "symbol": symbol,
                "error": "no_quote",
            }

        return {
            "symbol": quote.symbol,
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "timestamp": quote.timestamp,
            "spread_bps": quote.spread_bps,
        }

    def now(self) -> datetime:
        """
        Get current simulation time.

        Returns:
            Current UTC timestamp
        """
        return self._clock.now()

    # =========================================================================
    # Backtest-Specific Methods
    # =========================================================================

    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            Cash + market value of positions
        """
        value = self._cash
        for symbol, qty in self._positions.items():
            quote = self._quotes.get(symbol)
            if quote:
                value += qty * quote.mid
        return value

    def get_unrealized_pnl(self) -> Dict[str, float]:
        """
        Get unrealized P&L per position.

        Returns:
            Dict mapping symbol to unrealized P&L
        """
        pnl = {}
        for symbol, qty in self._positions.items():
            quote = self._quotes.get(symbol)
            cost = self._position_costs.get(symbol, 0)
            if quote and cost > 0:
                current_value = qty * quote.mid
                cost_basis = qty * cost
                pnl[symbol] = current_value - cost_basis
        return pnl

    def get_realized_pnl(self) -> float:
        """
        Calculate total realized P&L from fills.

        Returns:
            Total realized P&L
        """
        # Sum up (sell_value - buy_value) for closed trades
        # Simplified: total cash change minus initial
        return self._cash - self._initial_cash + sum(
            qty * quote.mid
            for symbol, qty in self._positions.items()
            if (quote := self._quotes.get(symbol)) is not None
        ) - self._initial_cash

    def get_total_fees(self) -> float:
        """
        Get total fees paid.

        Returns:
            Sum of all fees
        """
        return sum(f.get("fee", 0) for f in self._fills)

    def get_position_cost(self, symbol: str) -> float:
        """
        Get average cost basis for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Average cost per share
        """
        return self._position_costs.get(symbol, 0.0)

    def reset(self) -> None:
        """Reset broker state to initial."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._position_costs.clear()
        self._fills.clear()
        self._quotes.clear()
        self._order_id = 0

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _calculate_slippage(
        self,
        price: float,
        qty: float,
        quote: Quote,
    ) -> float:
        """
        Calculate slippage based on model.

        Args:
            price: Base price
            qty: Order quantity
            quote: Current quote

        Returns:
            Slippage amount (to add to buy price, subtract from sell price)
        """
        if self._slippage_model == "fixed":
            return price * self._slippage_bps / 10000

        elif self._slippage_model == "volume_based":
            # Larger orders have more slippage
            # Assume ADV of 5M shares as baseline
            assumed_adv = 5_000_000
            participation = qty / assumed_adv
            # Square root model for market impact
            impact_factor = participation**0.5
            return price * self._slippage_bps / 10000 * (1 + impact_factor * 10)

        else:
            return price * self._slippage_bps / 10000
