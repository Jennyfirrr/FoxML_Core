# Plan 14: Live Data Providers

## Overview

Real-time market data providers for live trading:
- Alpaca Data API (included with trading account)
- Polygon.io (professional data feed)

## Files to Create

### Directory Restructure

Move existing data providers to new `data/` directory:

```
LIVE_TRADING/data/
├── __init__.py
├── interface.py          # DataProvider Protocol (moved)
├── simulated.py          # SimulatedDataProvider (moved)
├── cached.py             # CachedDataProvider (moved)
├── alpaca.py             # NEW: Alpaca data
└── polygon.py            # NEW: Polygon data
```

### 1. `LIVE_TRADING/data/__init__.py`

```python
"""
Data Providers Module
=====================

Market data providers for live trading.

Providers:
- SimulatedDataProvider: Synthetic data for testing
- AlpacaDataProvider: Alpaca Markets data
- PolygonDataProvider: Polygon.io data
- CachedDataProvider: Caching wrapper
"""

from .interface import DataProvider
from .simulated import SimulatedDataProvider
from .cached import CachedDataProvider
from .alpaca import AlpacaDataProvider
from .polygon import PolygonDataProvider

__all__ = [
    "DataProvider",
    "SimulatedDataProvider",
    "CachedDataProvider",
    "AlpacaDataProvider",
    "PolygonDataProvider",
    "get_data_provider",
]


def get_data_provider(provider_type: str, **kwargs) -> DataProvider:
    """
    Factory function for data providers.

    Args:
        provider_type: Provider type (simulated, alpaca, polygon)
        **kwargs: Provider-specific configuration

    Returns:
        DataProvider instance
    """
    if provider_type == "simulated":
        return SimulatedDataProvider(**kwargs)
    elif provider_type == "alpaca":
        return AlpacaDataProvider(**kwargs)
    elif provider_type == "polygon":
        return PolygonDataProvider(**kwargs)
    else:
        raise ValueError(f"Unknown data provider: {provider_type}")
```

### 2. `LIVE_TRADING/data/alpaca.py`

```python
"""
Alpaca Data Provider
====================

Real-time and historical data from Alpaca Markets.

Features:
- Real-time quotes via REST
- Historical bars (minute, hour, day)
- ADV calculation
- Quote caching with TTL

SST Compliance:
- Credentials from environment
- Configuration via get_cfg()
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.types import Quote
from LIVE_TRADING.common.exceptions import LiveTradingError

logger = logging.getLogger(__name__)


class DataError(LiveTradingError):
    """Error fetching market data."""
    pass


class AlpacaDataProvider:
    """
    Alpaca Markets data provider.

    Provides real-time quotes and historical data.

    Example:
        >>> provider = AlpacaDataProvider()
        >>> quote = provider.get_quote("AAPL")
        >>> hist = provider.get_historical("AAPL", period="1mo")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: str = "iex",
    ):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: API key (default: from env)
            api_secret: API secret (default: from env)
            feed: Data feed (iex or sip)
        """
        self._api_key = api_key or os.environ.get(
            get_cfg("live_trading.brokers.alpaca.api_key_env", default="ALPACA_API_KEY")
        )
        self._api_secret = api_secret or os.environ.get(
            get_cfg("live_trading.brokers.alpaca.api_secret_env", default="ALPACA_API_SECRET")
        )
        self._feed = feed

        if not self._api_key or not self._api_secret:
            raise DataError("Alpaca credentials not found")

        self._client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._api_secret,
        )

        # Quote cache
        self._quote_cache: Dict[str, tuple[Quote, datetime]] = {}
        self._cache_ttl_seconds = get_cfg(
            "live_trading.data.cache_ttl_seconds", default=1.0
        )

        logger.info(f"AlpacaDataProvider initialized (feed={feed})")

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote with bid/ask data
        """
        # Check cache
        now = datetime.now(timezone.utc)
        if symbol in self._quote_cache:
            cached_quote, cache_time = self._quote_cache[symbol]
            if (now - cache_time).total_seconds() < self._cache_ttl_seconds:
                return cached_quote

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
            quotes = self._client.get_stock_latest_quote(request)

            if symbol not in quotes:
                raise DataError(f"No quote available for {symbol}")

            q = quotes[symbol]

            quote = Quote(
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                bid_size=float(q.bid_size),
                ask_size=float(q.ask_size),
                timestamp=q.timestamp.replace(tzinfo=timezone.utc),
            )

            # Cache
            self._quote_cache[symbol] = (quote, now)

            return quote

        except Exception as e:
            raise DataError(f"Failed to get quote for {symbol}: {e}")

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol
            period: Lookback period (1d, 5d, 1mo, 3mo, 1y)
            interval: Bar interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with OHLCV columns
        """
        try:
            # Parse period to start date
            end = datetime.now(timezone.utc)
            start = self._parse_period(period, end)

            # Parse interval to TimeFrame
            timeframe = self._parse_interval(interval)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=self._feed,
            )

            bars = self._client.get_stock_bars(request)

            if symbol not in bars.data:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    "Open": float(bar.open),
                    "High": float(bar.high),
                    "Low": float(bar.low),
                    "Close": float(bar.close),
                    "Volume": int(bar.volume),
                })

            df = pd.DataFrame(data)
            if not df.empty:
                # Set index from timestamps
                df.index = pd.DatetimeIndex([
                    bar.timestamp for bar in bars.data[symbol]
                ])

            return df

        except Exception as e:
            raise DataError(f"Failed to get historical data for {symbol}: {e}")

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Get average daily volume in dollars.

        Args:
            symbol: Trading symbol
            lookback_days: Days for average

        Returns:
            ADV in dollars
        """
        try:
            hist = self.get_historical(symbol, period=f"{lookback_days + 5}d", interval="1d")

            if hist.empty or len(hist) < 5:
                # Fallback estimate
                quote = self.get_quote(symbol)
                return quote.mid * 5_000_000  # Assume 5M shares

            recent = hist.tail(lookback_days)
            return float((recent["Volume"] * recent["Close"]).mean())

        except Exception as e:
            logger.warning(f"Failed to calculate ADV for {symbol}: {e}")
            return 0.0

    def _parse_period(self, period: str, end: datetime) -> datetime:
        """Parse period string to start datetime."""
        if period.endswith("d"):
            days = int(period[:-1])
            return end - timedelta(days=days)
        elif period.endswith("mo"):
            months = int(period[:-2])
            return end - timedelta(days=months * 30)
        elif period.endswith("y"):
            years = int(period[:-1])
            return end - timedelta(days=years * 365)
        else:
            return end - timedelta(days=30)  # Default 1 month

    def _parse_interval(self, interval: str) -> TimeFrame:
        """Parse interval string to TimeFrame."""
        if interval == "1m":
            return TimeFrame.Minute
        elif interval == "5m":
            return TimeFrame(5, "Min")
        elif interval == "15m":
            return TimeFrame(15, "Min")
        elif interval == "1h":
            return TimeFrame.Hour
        elif interval == "1d":
            return TimeFrame.Day
        else:
            return TimeFrame.Day

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear quote cache."""
        if symbol:
            self._quote_cache.pop(symbol, None)
        else:
            self._quote_cache.clear()
```

### 3. `LIVE_TRADING/data/polygon.py`

```python
"""
Polygon.io Data Provider
========================

Professional market data from Polygon.io.

Features:
- Real-time quotes
- Historical bars
- Extended hours data

SST Compliance:
- API key from environment
- Configuration via get_cfg()
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import requests

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.types import Quote
from LIVE_TRADING.common.exceptions import LiveTradingError

logger = logging.getLogger(__name__)


class DataError(LiveTradingError):
    """Error fetching market data."""
    pass


class PolygonDataProvider:
    """
    Polygon.io data provider.

    Provides real-time quotes and historical data.

    Example:
        >>> provider = PolygonDataProvider()
        >>> quote = provider.get_quote("AAPL")
        >>> hist = provider.get_historical("AAPL", period="1mo")
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon data provider.

        Args:
            api_key: API key (default: from POLYGON_API_KEY env)
        """
        self._api_key = api_key or os.environ.get(
            get_cfg("live_trading.data.polygon.api_key_env", default="POLYGON_API_KEY")
        )

        if not self._api_key:
            raise DataError("Polygon API key not found. Set POLYGON_API_KEY env var.")

        # Quote cache
        self._quote_cache: Dict[str, tuple[Quote, datetime]] = {}
        self._cache_ttl_seconds = get_cfg(
            "live_trading.data.cache_ttl_seconds", default=1.0
        )

        logger.info("PolygonDataProvider initialized")

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request."""
        params = params or {}
        params["apiKey"] = self._api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise DataError(f"Polygon API request failed: {e}")

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote with bid/ask data
        """
        # Check cache
        now = datetime.now(timezone.utc)
        if symbol in self._quote_cache:
            cached_quote, cache_time = self._quote_cache[symbol]
            if (now - cache_time).total_seconds() < self._cache_ttl_seconds:
                return cached_quote

        try:
            data = self._request(f"/v2/last/nbbo/{symbol}")

            if data.get("status") != "OK":
                raise DataError(f"Failed to get quote for {symbol}")

            result = data.get("results", {})

            quote = Quote(
                symbol=symbol,
                bid=float(result.get("P", 0)),  # Bid price
                ask=float(result.get("p", 0)),  # Ask price (lowercase)
                bid_size=float(result.get("S", 0)),  # Bid size
                ask_size=float(result.get("s", 0)),  # Ask size
                timestamp=datetime.fromtimestamp(
                    result.get("t", 0) / 1e9, tz=timezone.utc
                ),
            )

            # Cache
            self._quote_cache[symbol] = (quote, now)

            return quote

        except Exception as e:
            raise DataError(f"Failed to get quote for {symbol}: {e}")

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol
            period: Lookback period
            interval: Bar interval

        Returns:
            DataFrame with OHLCV columns
        """
        try:
            end = datetime.now(timezone.utc)
            start = self._parse_period(period, end)

            # Map interval to Polygon timespan
            timespan, multiplier = self._parse_interval(interval)

            endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

            data = self._request(endpoint, {"limit": 5000})

            if data.get("status") != "OK":
                return pd.DataFrame()

            results = data.get("results", [])
            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(results)
            df = df.rename(columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "t": "timestamp",
            })

            # Convert timestamp to datetime index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            raise DataError(f"Failed to get historical data for {symbol}: {e}")

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """Get average daily volume in dollars."""
        try:
            hist = self.get_historical(symbol, period=f"{lookback_days + 5}d", interval="1d")

            if hist.empty or len(hist) < 5:
                quote = self.get_quote(symbol)
                return quote.mid * 5_000_000

            recent = hist.tail(lookback_days)
            return float((recent["Volume"] * recent["Close"]).mean())

        except Exception as e:
            logger.warning(f"Failed to calculate ADV for {symbol}: {e}")
            return 0.0

    def _parse_period(self, period: str, end: datetime) -> datetime:
        """Parse period string to start datetime."""
        if period.endswith("d"):
            days = int(period[:-1])
            return end - timedelta(days=days)
        elif period.endswith("mo"):
            months = int(period[:-2])
            return end - timedelta(days=months * 30)
        elif period.endswith("y"):
            years = int(period[:-1])
            return end - timedelta(days=years * 365)
        else:
            return end - timedelta(days=30)

    def _parse_interval(self, interval: str) -> tuple[str, int]:
        """Parse interval to Polygon timespan and multiplier."""
        if interval == "1m":
            return "minute", 1
        elif interval == "5m":
            return "minute", 5
        elif interval == "15m":
            return "minute", 15
        elif interval == "1h":
            return "hour", 1
        elif interval == "1d":
            return "day", 1
        else:
            return "day", 1

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear quote cache."""
        if symbol:
            self._quote_cache.pop(symbol, None)
        else:
            self._quote_cache.clear()
```

## Tests

### `LIVE_TRADING/tests/test_data_providers.py`

```python
"""
Data Provider Tests
===================

Tests for Alpaca and Polygon data providers.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from LIVE_TRADING.data.alpaca import AlpacaDataProvider
from LIVE_TRADING.data.polygon import PolygonDataProvider


class TestAlpacaDataProvider:
    """Tests for Alpaca data provider."""

    @patch("LIVE_TRADING.data.alpaca.StockHistoricalDataClient")
    def test_init(self, mock_client):
        """Test initialization."""
        with patch.dict("os.environ", {"ALPACA_API_KEY": "key", "ALPACA_API_SECRET": "secret"}):
            provider = AlpacaDataProvider()
            assert provider._api_key == "key"

    @patch("LIVE_TRADING.data.alpaca.StockHistoricalDataClient")
    def test_get_quote(self, mock_client):
        """Test quote retrieval."""
        with patch.dict("os.environ", {"ALPACA_API_KEY": "key", "ALPACA_API_SECRET": "secret"}):
            provider = AlpacaDataProvider()

            mock_quote = Mock()
            mock_quote.bid_price = 149.90
            mock_quote.ask_price = 150.10
            mock_quote.bid_size = 100
            mock_quote.ask_size = 100
            mock_quote.timestamp = datetime.now(timezone.utc)

            provider._client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

            quote = provider.get_quote("AAPL")
            assert quote.symbol == "AAPL"
            assert quote.bid == 149.90
            assert quote.ask == 150.10


class TestPolygonDataProvider:
    """Tests for Polygon data provider."""

    @patch("LIVE_TRADING.data.polygon.requests.get")
    def test_get_quote(self, mock_get):
        """Test quote retrieval."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "key"}):
            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "OK",
                "results": {
                    "P": 149.90,
                    "p": 150.10,
                    "S": 100,
                    "s": 100,
                    "t": 1700000000000000000,
                }
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            provider = PolygonDataProvider()
            quote = provider.get_quote("AAPL")

            assert quote.symbol == "AAPL"
            assert quote.bid == 149.90
```

## Configuration

Add to `CONFIG/live_trading/live_trading.yaml`:

```yaml
live_trading:
  data:
    provider: "alpaca"  # alpaca, polygon, simulated
    cache_ttl_seconds: 1.0

    alpaca:
      feed: "iex"  # iex (free) or sip (paid)

    polygon:
      api_key_env: "POLYGON_API_KEY"
```

## Migration Notes

1. Move `engine/data_provider.py` contents to `data/` directory
2. Update imports throughout codebase
3. Keep backward compatibility in `engine/__init__.py`

## SST Compliance

- [ ] All API keys from environment variables
- [ ] Configuration via get_cfg()
- [ ] Consistent Quote type across providers
- [ ] Cache management for rate limiting

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `data/__init__.py` | 50 |
| `data/alpaca.py` | 200 |
| `data/polygon.py` | 200 |
| `tests/test_data_providers.py` | 100 |
| **Total** | ~550 |
