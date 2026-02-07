# Plan 15: Backtesting Engine

## Overview

Historical simulation mode for strategy validation and performance analysis.

## Design Goals

1. **Consistency**: Same pipeline as live trading
2. **Realism**: Configurable slippage and transaction costs
3. **Performance**: Vectorized where possible, event-driven where necessary
4. **Analysis**: Comprehensive performance metrics and visualization

## Files to Create

### 1. `LIVE_TRADING/backtest/__init__.py`

```python
"""
Backtesting Module
==================

Historical simulation for strategy validation.

Components:
- BacktestEngine: Main orchestrator
- HistoricalDataLoader: Load historical data
- BacktestBroker: Simulated execution with realistic fills
- PerformanceReport: Metrics and analysis
"""

from .engine import BacktestEngine, BacktestConfig
from .data_loader import HistoricalDataLoader
from .broker import BacktestBroker
from .report import PerformanceReport

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "HistoricalDataLoader",
    "BacktestBroker",
    "PerformanceReport",
]
```

### 2. `LIVE_TRADING/backtest/data_loader.py`

```python
"""
Historical Data Loader
======================

Load and manage historical data for backtesting.

Features:
- Load from CSV/Parquet files
- Download from data providers
- Data validation and cleaning
- Bar generation at multiple frequencies
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.types import Quote
from LIVE_TRADING.common.exceptions import LiveTradingError

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
        data_dir: Optional[Path] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory for cached data
            provider: Data provider for downloads (alpaca, polygon)
        """
        self._data_dir = Path(data_dir or get_cfg(
            "live_trading.backtest.data_dir", default="data/historical"
        ))
        self._provider = provider
        self._cache: Dict[str, pd.DataFrame] = {}

        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol.

        Args:
            symbol: Trading symbol
            start: Start date
            end: End date
            interval: Bar interval

        Returns:
            DataFrame with OHLCV data
        """
        # Parse dates
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        # Check cache
        cache_key = f"{symbol}_{start.date()}_{end.date()}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Try loading from file
        file_path = self._get_file_path(symbol, interval)
        if file_path.exists():
            df = self._load_from_file(file_path, start, end)
            if not df.empty:
                self._cache[cache_key] = df
                return df.copy()

        # Download if provider available
        if self._provider:
            df = self._download(symbol, start, end, interval)
            if not df.empty:
                self._save_to_file(df, file_path)
                self._cache[cache_key] = df
                return df.copy()

        raise DataLoadError(f"No data available for {symbol}")

    def load_multiple(
        self,
        symbols: List[str],
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        return {symbol: self.load(symbol, start, end, interval) for symbol in symbols}

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

            # Filter by date range
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            mask = (df.index >= start.replace(tzinfo=timezone.utc)) & \
                   (df.index <= end.replace(tzinfo=timezone.utc))

            return df.loc[mask]

        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return pd.DataFrame()

    def _save_to_file(self, df: pd.DataFrame, path: Path) -> None:
        """Save data to file."""
        try:
            df.to_parquet(path)
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
        if self._provider == "alpaca":
            from LIVE_TRADING.data.alpaca import AlpacaDataProvider
            provider = AlpacaDataProvider()
        elif self._provider == "polygon":
            from LIVE_TRADING.data.polygon import PolygonDataProvider
            provider = PolygonDataProvider()
        else:
            return pd.DataFrame()

        # Calculate period
        days = (end - start).days
        period = f"{days}d"

        return provider.get_historical(symbol, period=period, interval=interval)

    def get_quote_at(
        self,
        symbol: str,
        timestamp: datetime,
        data: pd.DataFrame,
    ) -> Quote:
        """
        Generate quote at a specific timestamp from historical data.

        Simulates bid/ask from OHLC data.
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
```

### 3. `LIVE_TRADING/backtest/broker.py`

```python
"""
Backtest Broker
===============

Simulated broker for backtesting with realistic fills.

Features:
- Slippage models (fixed, volume-based)
- Transaction costs
- Position tracking
- Fill simulation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.brokers.interface import Broker
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class BacktestBroker(Broker):
    """
    Simulated broker for backtesting.

    Models realistic execution with slippage and fees.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 5.0,
        fee_bps: float = 1.0,
        slippage_model: str = "fixed",
    ):
        """
        Initialize backtest broker.

        Args:
            initial_cash: Starting capital
            slippage_bps: Slippage in basis points
            fee_bps: Transaction fee in basis points
            slippage_model: Slippage model (fixed, volume_based)
        """
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._slippage_bps = slippage_bps
        self._fee_bps = fee_bps
        self._slippage_model = slippage_model

        self._positions: Dict[str, float] = {}  # symbol -> shares
        self._position_costs: Dict[str, float] = {}  # symbol -> avg cost
        self._fills: List[Dict[str, Any]] = []
        self._order_id = 0

        # Current quotes (set by backtest engine)
        self._quotes: Dict[str, Quote] = {}
        self._current_time = datetime.now(timezone.utc)

    def set_quote(self, symbol: str, quote: Quote) -> None:
        """Set current quote for symbol."""
        self._quotes[symbol] = quote

    def set_time(self, timestamp: datetime) -> None:
        """Set current simulation time."""
        self._current_time = timestamp

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        **kwargs,
    ) -> Dict[str, Any]:
        """Submit and immediately fill an order."""
        self._order_id += 1
        order_id = f"bt_{self._order_id}"

        quote = self._quotes.get(symbol)
        if not quote:
            return {
                "order_id": order_id,
                "status": "rejected",
                "reason": "no_quote",
            }

        # Calculate fill price with slippage
        if side.upper() == "BUY":
            base_price = quote.ask
            slippage = base_price * self._slippage_bps / 10000
            fill_price = base_price + slippage
        else:
            base_price = quote.bid
            slippage = base_price * self._slippage_bps / 10000
            fill_price = base_price - slippage

        # Calculate costs
        notional = qty * fill_price
        fee = notional * self._fee_bps / 10000

        # Check funds for buy
        if side.upper() == "BUY":
            total_cost = notional + fee
            if total_cost > self._cash:
                return {
                    "order_id": order_id,
                    "status": "rejected",
                    "reason": "insufficient_funds",
                }
            self._cash -= total_cost
        else:
            self._cash += notional - fee

        # Update position
        current_qty = self._positions.get(symbol, 0)
        if side.upper() == "BUY":
            new_qty = current_qty + qty
            # Update average cost
            current_cost = self._position_costs.get(symbol, 0)
            self._position_costs[symbol] = (
                current_cost * current_qty + notional
            ) / new_qty if new_qty > 0 else 0
        else:
            new_qty = current_qty - qty

        if abs(new_qty) < 1e-8:
            self._positions.pop(symbol, None)
            self._position_costs.pop(symbol, None)
        else:
            self._positions[symbol] = new_qty

        # Record fill
        fill = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "price": fill_price,
            "fee": fee,
            "timestamp": self._current_time,
        }
        self._fills.append(fill)

        return {
            "order_id": order_id,
            "status": "filled",
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "fill_price": fill_price,
            "fee": fee,
            "timestamp": self._current_time,
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel not supported in backtest (immediate fills)."""
        return {"order_id": order_id, "status": "not_found"}

    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return dict(self._positions)

    def get_cash(self) -> float:
        """Get available cash."""
        return self._cash

    def get_fills(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get fills since timestamp."""
        if since is None:
            return list(self._fills)
        return [f for f in self._fills if f["timestamp"] >= since]

    def now(self) -> datetime:
        """Get current simulation time."""
        return self._current_time

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        value = self._cash
        for symbol, qty in self._positions.items():
            quote = self._quotes.get(symbol)
            if quote:
                value += qty * quote.mid
        return value

    def reset(self) -> None:
        """Reset broker state."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._position_costs.clear()
        self._fills.clear()
        self._order_id = 0
```

### 4. `LIVE_TRADING/backtest/engine.py`

```python
"""
Backtest Engine
===============

Main orchestrator for historical simulation.

Runs the trading engine against historical data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.engine import TradingEngine, EngineConfig
from LIVE_TRADING.backtest.broker import BacktestBroker
from LIVE_TRADING.backtest.data_loader import HistoricalDataLoader
from LIVE_TRADING.backtest.report import PerformanceReport

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    start_date: str
    end_date: str
    symbols: List[str]
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0
    fee_bps: float = 1.0
    interval: str = "1d"  # Trading frequency
    data_provider: Optional[str] = None

    @classmethod
    def from_config(cls) -> "BacktestConfig":
        """Load from config files."""
        return cls(
            start_date=get_cfg("live_trading.backtest.default_start", default="2023-01-01"),
            end_date=get_cfg("live_trading.backtest.default_end", default="2024-01-01"),
            symbols=get_cfg("live_trading.symbols.default", default=["SPY"]),
            initial_cash=get_cfg("live_trading.paper.initial_cash", default=100_000.0),
            slippage_bps=get_cfg("live_trading.backtest.slippage_bps", default=5.0),
            fee_bps=get_cfg("live_trading.paper.fee_bps", default=1.0),
        )


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    trades: List[Dict[str, Any]]
    daily_returns: pd.Series
    equity_curve: pd.Series
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "final_value": self.final_value,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
        }


class BacktestEngine:
    """
    Backtest engine for historical simulation.

    Example:
        >>> config = BacktestConfig(
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     symbols=["SPY", "AAPL"],
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run()
        >>> print(f"Total return: {result.total_return:.2%}")
    """

    def __init__(
        self,
        config: BacktestConfig,
        run_root: Optional[str] = None,
        engine_config: Optional[EngineConfig] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            run_root: Path to TRAINING run artifacts
            engine_config: Trading engine configuration
        """
        self.config = config

        # Initialize broker
        self.broker = BacktestBroker(
            initial_cash=config.initial_cash,
            slippage_bps=config.slippage_bps,
            fee_bps=config.fee_bps,
        )

        # Initialize data loader
        self.data_loader = HistoricalDataLoader(
            provider=config.data_provider,
        )

        # Initialize trading engine
        self.trading_engine = TradingEngine(
            broker=self.broker,
            data_provider=self._create_data_provider(),
            run_root=run_root,
            config=engine_config or EngineConfig(save_state=False, save_history=False),
        )

        self._equity_curve: List[tuple[datetime, float]] = []
        self._trades: List[Dict[str, Any]] = []

    def _create_data_provider(self):
        """Create a data provider wrapper for the backtest."""
        # Use a simple adapter that returns data from the loader
        from LIVE_TRADING.data.simulated import SimulatedDataProvider
        return SimulatedDataProvider()

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")

        # Load historical data
        start = datetime.fromisoformat(self.config.start_date).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(self.config.end_date).replace(tzinfo=timezone.utc)

        data = self.data_loader.load_multiple(
            self.config.symbols,
            start,
            end,
            self.config.interval,
        )

        # Get trading dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)

        logger.info(f"Simulating {len(trading_dates)} trading days")

        # Run simulation
        for timestamp in trading_dates:
            # Update broker time and quotes
            self.broker.set_time(timestamp)

            for symbol in self.config.symbols:
                if symbol in data and timestamp in data[symbol].index:
                    quote = self.data_loader.get_quote_at(symbol, timestamp, data[symbol])
                    self.broker.set_quote(symbol, quote)

            # Run trading cycle
            try:
                result = self.trading_engine.run_cycle(
                    self.config.symbols,
                    current_time=timestamp,
                )

                # Record trades
                for decision in result.decisions:
                    if decision.decision == "TRADE" and decision.shares != 0:
                        self._trades.append(decision.to_dict())

            except Exception as e:
                logger.warning(f"Error at {timestamp}: {e}")

            # Record equity
            portfolio_value = self.broker.get_portfolio_value()
            self._equity_curve.append((timestamp, portfolio_value))

        # Calculate metrics
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self._equity_curve:
            return BacktestResult(
                config=self.config,
                trades=self._trades,
                daily_returns=pd.Series(),
                equity_curve=pd.Series(),
                final_value=self.config.initial_cash,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                num_trades=0,
            )

        # Build equity curve
        equity = pd.Series(
            {t: v for t, v in self._equity_curve},
            name="equity",
        )

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Calculate metrics
        final_value = equity.iloc[-1]
        total_return = (final_value / self.config.initial_cash) - 1

        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * (252 ** 0.5)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = abs(drawdown.min())

        # Win rate
        wins = sum(1 for t in self._trades if t.get("alpha", 0) > 0)
        win_rate = wins / len(self._trades) if self._trades else 0.0

        return BacktestResult(
            config=self.config,
            trades=self._trades,
            daily_returns=returns,
            equity_curve=equity,
            final_value=final_value,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(self._trades),
        )
```

### 5. `LIVE_TRADING/backtest/report.py`

```python
"""
Performance Report
==================

Generate performance reports from backtest results.
"""

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class PerformanceReport:
    """Performance report generator."""

    @staticmethod
    def summary(result) -> str:
        """Generate text summary."""
        return f"""
Backtest Results
================
Period: {result.config.start_date} to {result.config.end_date}
Symbols: {', '.join(result.config.symbols)}

Performance
-----------
Final Value:    ${result.final_value:,.2f}
Total Return:   {result.total_return:.2%}
Sharpe Ratio:   {result.sharpe_ratio:.2f}
Max Drawdown:   {result.max_drawdown:.2%}

Trading
-------
Total Trades:   {result.num_trades}
Win Rate:       {result.win_rate:.1%}
"""

    @staticmethod
    def to_html(result) -> str:
        """Generate HTML report."""
        # Implementation for HTML report
        pass
```

## Tests

### `LIVE_TRADING/tests/test_backtest.py`

Tests for backtest components (data loader, broker, engine).

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `backtest/__init__.py` | 30 |
| `backtest/data_loader.py` | 200 |
| `backtest/broker.py` | 180 |
| `backtest/engine.py` | 220 |
| `backtest/report.py` | 80 |
| `tests/test_backtest.py` | 150 |
| **Total** | ~860 |
