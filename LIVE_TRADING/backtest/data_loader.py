"""
Historical Data Loader
======================

Load and manage historical data for backtesting.

Features:
- Load from CSV/Parquet files
- Download from data providers
- Data validation and cleaning
- Quote generation from OHLC bars
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.exceptions import LiveTradingError
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class DataLoadError(LiveTradingError):
    """Error loading historical data."""

    pass


class HistoricalDataLoader:
    """
    Load historical data for backtesting.

    Example:
        >>> loader = HistoricalDataLoader(data_dir="data/historical")
        >>> data = loader.load("AAPL", start="2023-01-01", end="2023-12-31")
    """

    def __init__(
        self,
        data_dir: Optional[Union[Path, str]] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory for cached data
            provider: Data provider for downloads (alpaca, polygon)
        """
        self._data_dir = Path(
            data_dir
            or get_cfg("live_trading.backtest.data_dir", default="data/historical")
        )
        self._provider = provider
        self._cache: Dict[str, pd.DataFrame] = {}

        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self._data_dir

    def load(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol.

        Args:
            symbol: Trading symbol
            start: Start date (string or datetime)
            end: End date (string or datetime)
            interval: Bar interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data indexed by timestamp

        Raises:
            DataLoadError: If no data available
        """
        # Parse dates
        start_dt = self._parse_date(start)
        end_dt = self._parse_date(end)

        # Check cache
        cache_key = f"{symbol}_{start_dt.date()}_{end_dt.date()}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Try loading from file
        file_path = self._get_file_path(symbol, interval)
        if file_path.exists():
            df = self._load_from_file(file_path, start_dt, end_dt)
            if not df.empty:
                self._cache[cache_key] = df
                return df.copy()

        # Download if provider available
        if self._provider:
            df = self._download(symbol, start_dt, end_dt, interval)
            if not df.empty:
                self._save_to_file(df, file_path)
                self._cache[cache_key] = df
                return df.copy()

        raise DataLoadError(f"No data available for {symbol}")

    def load_multiple(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start: Start date
            end: End date
            interval: Bar interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, start, end, interval)
            except DataLoadError as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        return result

    def get_quote_at(
        self,
        symbol: str,
        timestamp: datetime,
        data: pd.DataFrame,
    ) -> Quote:
        """
        Generate quote at a specific timestamp from historical data.

        Simulates bid/ask from OHLC data using a spread estimate.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp for quote
            data: Historical DataFrame

        Returns:
            Quote with simulated bid/ask

        Raises:
            DataLoadError: If no data at timestamp
        """
        # Find the bar containing this timestamp
        if timestamp in data.index:
            bar = data.loc[timestamp]
        else:
            # Find nearest previous bar
            mask = data.index <= timestamp
            if not mask.any():
                raise DataLoadError(f"No data at {timestamp} for {symbol}")
            bar = data.loc[mask].iloc[-1]

        # Simulate spread from volatility
        close = float(bar["Close"])
        high = float(bar["High"])
        low = float(bar["Low"])

        # Estimate spread as fraction of range
        range_pct = (high - low) / close if close > 0 else 0.001
        spread_pct = min(range_pct * 0.1, 0.001)  # Cap at 10 bps

        half_spread = close * spread_pct / 2

        return Quote(
            symbol=symbol,
            bid=close - half_spread,
            ask=close + half_spread,
            bid_size=10000.0,
            ask_size=10000.0,
            timestamp=timestamp,
        )

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear data cache.

        Args:
            symbol: Clear only this symbol (None = all)
        """
        if symbol:
            # Clear entries containing this symbol
            keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

    def _parse_date(self, date: Union[str, datetime]) -> datetime:
        """Parse date string or datetime to UTC datetime."""
        if isinstance(date, str):
            dt = datetime.fromisoformat(date)
        else:
            dt = date

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    def _get_file_path(self, symbol: str, interval: str) -> Path:
        """Get file path for symbol data."""
        return self._data_dir / f"{symbol}_{interval}.parquet"

    def _load_from_file(
        self,
        path: Path,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Load data from file."""
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                return pd.DataFrame()

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            elif df.index.tz != timezone.utc:
                df.index = df.index.tz_convert("UTC")

            # Filter by date range
            mask = (df.index >= start) & (df.index <= end)

            return df.loc[mask]

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return pd.DataFrame()

    def _save_to_file(self, df: pd.DataFrame, path: Path) -> None:
        """Save data to file."""
        try:
            df.to_parquet(path)
            logger.info(f"Saved historical data to {path}")
        except Exception as e:
            logger.warning(f"Failed to save {path}: {e}")

    def _download(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """Download data from provider."""
        try:
            if self._provider == "alpaca":
                from LIVE_TRADING.data.alpaca import AlpacaDataProvider

                provider = AlpacaDataProvider()
            elif self._provider == "polygon":
                from LIVE_TRADING.data.polygon import PolygonDataProvider

                provider = PolygonDataProvider()
            else:
                logger.warning(f"Unknown provider: {self._provider}")
                return pd.DataFrame()

            # Calculate period
            days = (end - start).days
            period = f"{days}d"

            logger.info(f"Downloading {symbol} data from {self._provider}")
            return provider.get_historical(symbol, period=period, interval=interval)

        except Exception as e:
            logger.warning(f"Failed to download {symbol}: {e}")
            return pd.DataFrame()


def generate_synthetic_data(
    symbols: List[str],
    start: datetime,
    end: datetime,
    interval: str = "1d",
    base_prices: Optional[Dict[str, float]] = None,
    volatility: float = 0.02,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic historical data for testing.

    Args:
        symbols: List of symbols
        start: Start date
        end: End date
        interval: Bar interval
        base_prices: Starting prices per symbol
        volatility: Daily volatility

    Returns:
        Dict mapping symbol to DataFrame
    """
    import numpy as np

    base_prices = base_prices or {}

    result = {}
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)

        # Generate date range
        if interval == "1d":
            freq = "D"
        elif interval == "1h":
            freq = "H"
        else:
            freq = "D"

        dates = pd.date_range(start=start, end=end, freq=freq, tz=timezone.utc)

        # Generate random returns
        np.random.seed(hash(symbol) % 2**32)  # Reproducible per symbol
        returns = np.random.normal(0, volatility, len(dates))

        # Generate prices
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLCV
        df = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
            },
            index=dates,
        )

        result[symbol] = df

    return result
