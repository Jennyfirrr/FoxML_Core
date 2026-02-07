"""
Simulated Data Provider
=======================

Synthetic data provider for testing and paper trading.

Generates realistic price movements and quote data without
requiring external API connections.

SST Compliance:
- Uses Clock abstraction for timestamps
- Configuration via get_cfg()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class SimulatedDataProvider:
    """
    Simulated data provider for testing.

    Generates synthetic price data and quotes.
    Useful for unit tests and paper trading simulations.

    Example:
        >>> provider = SimulatedDataProvider(symbols=["AAPL", "MSFT"])
        >>> quote = provider.get_quote("AAPL")
        >>> hist = provider.get_historical("AAPL", period="1mo")
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.02,
        spread_bps: float = 5.0,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize simulated data provider.

        Args:
            symbols: List of symbols to simulate
            base_prices: Starting prices per symbol
            volatility: Daily volatility (std)
            spread_bps: Bid-ask spread in basis points
            clock: Clock instance for time (default: system clock)
        """
        self._clock = clock or get_clock()
        self.symbols = symbols or ["SPY", "AAPL", "MSFT", "GOOGL"]
        self.base_prices = base_prices or {
            "SPY": 450.0,
            "AAPL": 180.0,
            "MSFT": 380.0,
            "GOOGL": 140.0,
        }
        self.volatility = volatility
        self.spread_bps = spread_bps

        # Current prices (start at base)
        self._current_prices: Dict[str, float] = dict(self.base_prices)

        # Pre-generate historical data cache
        self._historical_cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"SimulatedDataProvider: {len(self.symbols)} symbols, "
            f"vol={volatility:.1%}, spread={spread_bps:.1f}bps"
        )

    def get_quote(self, symbol: str) -> Quote:
        """
        Get simulated quote.

        Args:
            symbol: Trading symbol

        Returns:
            Quote with bid/ask based on current price
        """
        price = self._get_current_price(symbol)
        spread = price * self.spread_bps / 10000 / 2

        return Quote(
            symbol=symbol,
            bid=price - spread,
            ask=price + spread,
            bid_size=1000.0,
            ask_size=1000.0,
            timestamp=self._clock.now(),
            spread_bps=self.spread_bps,
        )

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get simulated historical data.

        Args:
            symbol: Trading symbol
            period: Lookback period
            interval: Bar interval

        Returns:
            DataFrame with OHLCV columns
        """
        cache_key = f"{symbol}_{period}_{interval}"

        if cache_key not in self._historical_cache:
            self._historical_cache[cache_key] = self._generate_historical(
                symbol, period, interval
            )

        return self._historical_cache[cache_key].copy()

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Get simulated average daily volume in dollars.

        Args:
            symbol: Trading symbol
            lookback_days: Days for average

        Returns:
            ADV in dollars
        """
        hist = self.get_historical(symbol, period="3mo", interval="1d")
        if len(hist) < lookback_days:
            lookback_days = len(hist)

        recent = hist.tail(lookback_days)
        if "Volume" in recent.columns and "Close" in recent.columns:
            return float((recent["Volume"] * recent["Close"]).mean())

        # Fallback
        return self._get_current_price(symbol) * 10_000_000

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a symbol.

        Args:
            symbol: Trading symbol
            price: New price
        """
        self._current_prices[symbol] = price

    def simulate_price_change(self, symbol: str) -> float:
        """
        Simulate a random price change.

        Args:
            symbol: Trading symbol

        Returns:
            New price
        """
        current = self._get_current_price(symbol)
        # Random walk
        change = np.random.normal(0, self.volatility)
        new_price = current * (1 + change)
        self._current_prices[symbol] = new_price
        return new_price

    def set_prices(self, prices: Dict[str, float]) -> None:
        """
        Set multiple prices at once.

        Args:
            prices: Dict mapping symbol to price
        """
        for symbol, price in prices.items():
            self._current_prices[symbol] = price

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear historical data cache.

        Args:
            symbol: Symbol to clear (None = all)
        """
        if symbol is None:
            self._historical_cache.clear()
        else:
            keys_to_remove = [k for k in self._historical_cache if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._historical_cache[key]

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        if symbol in self._current_prices:
            return self._current_prices[symbol]

        # Default price for unknown symbols
        return 100.0

    def _generate_historical(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        """Generate synthetic historical data."""
        # Parse period
        n_days = self._parse_period(period)
        n_bars = n_days * self._bars_per_day(interval)

        # Generate dates
        end = self._clock.now()
        if interval == "1d":
            dates = pd.bdate_range(end=end, periods=n_bars)
        else:
            dates = pd.date_range(
                end=end, periods=n_bars, freq=interval.replace("m", "min")
            )

        # Generate random walk prices
        base_price = self._get_current_price(symbol)
        returns = np.random.normal(0, self.volatility / np.sqrt(252), n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = df["Close"].shift(1).fillna(base_price)

        # Random intraday range
        daily_range = np.abs(np.random.normal(0, self.volatility, n_bars)) * prices
        df["High"] = np.maximum(df["Open"], df["Close"]) + daily_range * 0.5
        df["Low"] = np.minimum(df["Open"], df["Close"]) - daily_range * 0.5

        # Volume (with some randomness)
        base_volume = 10_000_000
        df["Volume"] = np.random.lognormal(np.log(base_volume), 0.5, n_bars).astype(
            int
        )

        return df

    def _parse_period(self, period: str) -> int:
        """Parse period string to days."""
        if period.endswith("d"):
            return int(period[:-1])
        elif period.endswith("mo"):
            return int(period[:-2]) * 21  # ~21 trading days per month
        elif period.endswith("y"):
            return int(period[:-1]) * 252
        else:
            return 21  # Default: 1 month

    def _bars_per_day(self, interval: str) -> int:
        """Calculate bars per day for interval."""
        if interval == "1d":
            return 1
        elif interval.endswith("m"):
            minutes = int(interval[:-1])
            return 390 // minutes  # 6.5 hours = 390 minutes
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            return 7 // hours
        else:
            return 1
