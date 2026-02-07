"""
Backtest Engine
===============

Main orchestrator for historical simulation.

Runs the trading engine against historical data to validate strategies
before live deployment.

Features:
- Uses same pipeline as live trading
- Configurable slippage and fees
- Performance metrics calculation
- Equity curve tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.clock import SimulatedClock
from LIVE_TRADING.backtest.broker import BacktestBroker
from LIVE_TRADING.backtest.data_loader import (
    HistoricalDataLoader,
    generate_synthetic_data,
    DataLoadError,
)

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
    data_provider: Optional[str] = None  # alpaca, polygon, or None for local files
    data_dir: Optional[str] = None

    @classmethod
    def from_config(cls) -> "BacktestConfig":
        """Load from config files."""
        return cls(
            start_date=get_cfg(
                "live_trading.backtest.default_start", default="2023-01-01"
            ),
            end_date=get_cfg("live_trading.backtest.default_end", default="2024-01-01"),
            symbols=get_cfg("live_trading.symbols.default", default=["SPY"]),
            initial_cash=get_cfg("live_trading.paper.initial_cash", default=100_000.0),
            slippage_bps=get_cfg("live_trading.backtest.slippage_bps", default=5.0),
            fee_bps=get_cfg("live_trading.paper.fee_bps", default=1.0),
            data_dir=get_cfg("live_trading.backtest.data_dir", default=None),
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
    total_fees: float
    avg_slippage_bps: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "symbols": self.config.symbols,
            "initial_cash": self.config.initial_cash,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "total_fees": self.total_fees,
            "avg_slippage_bps": self.avg_slippage_bps,
        }


class BacktestEngine:
    """
    Backtest engine for historical simulation.

    Runs a strategy against historical data to measure performance
    before deploying with real money.

    Example:
        >>> config = BacktestConfig(
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     symbols=["SPY", "AAPL"],
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run()
        >>> print(f"Total return: {result.total_return:.2%}")

    For integration with TradingEngine:
        >>> from LIVE_TRADING.engine import TradingEngine, EngineConfig
        >>> bt_engine = BacktestEngine(config)
        >>> trading_engine = TradingEngine(
        ...     broker=bt_engine.broker,
        ...     data_provider=bt_engine.data_provider,
        ...     config=EngineConfig(save_state=False),
        ... )
        >>> result = bt_engine.run_with_engine(trading_engine)
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_loader: Optional[HistoricalDataLoader] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            data_loader: Optional custom data loader
        """
        self.config = config

        # Parse dates
        self._start = self._parse_date(config.start_date)
        self._end = self._parse_date(config.end_date)

        # Initialize clock at start time
        self._clock = SimulatedClock(self._start)

        # Initialize broker
        self.broker = BacktestBroker(
            initial_cash=config.initial_cash,
            slippage_bps=config.slippage_bps,
            fee_bps=config.fee_bps,
            clock=self._clock,
        )

        # Initialize data loader
        self.data_loader = data_loader or HistoricalDataLoader(
            data_dir=config.data_dir,
            provider=config.data_provider,
        )

        # Results tracking
        self._equity_curve: List[tuple[datetime, float]] = []
        self._trades: List[Dict[str, Any]] = []
        self._historical_data: Dict[str, pd.DataFrame] = {}

    @property
    def clock(self) -> SimulatedClock:
        """Get the simulation clock."""
        return self._clock

    def run(
        self,
        strategy: Optional[callable] = None,
    ) -> BacktestResult:
        """
        Run the backtest with a simple strategy function.

        Args:
            strategy: Optional strategy function(symbol, quote, positions) -> (side, qty)
                     If None, uses buy-and-hold strategy

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")

        # Load historical data
        try:
            self._historical_data = self.data_loader.load_multiple(
                self.config.symbols,
                self._start,
                self._end,
                self.config.interval,
            )
        except DataLoadError:
            self._historical_data = {}

        # Fall back to synthetic data if no real data available
        if not self._historical_data:
            logger.info("No historical data found, using synthetic data")
            self._historical_data = generate_synthetic_data(
                self.config.symbols,
                self._start,
                self._end,
                self.config.interval,
            )

        if not self._historical_data:
            logger.error("No data available for backtest")
            return self._empty_result()

        # Get trading dates (union of all symbols)
        all_dates = set()
        for df in self._historical_data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)

        logger.info(f"Simulating {len(trading_dates)} trading periods")

        # Default strategy: buy and hold equal weight
        if strategy is None:
            strategy = self._default_strategy
            # Execute initial buys
            self._initialize_positions()

        # Run simulation
        for timestamp in trading_dates:
            # Update simulation time
            self._clock.set_time(timestamp)
            self.broker.set_time(timestamp)

            # Update quotes
            self._update_quotes(timestamp)

            # Execute strategy
            positions = self.broker.get_positions()
            for symbol in self.config.symbols:
                if symbol not in self._historical_data:
                    continue

                quote = self.broker._quotes.get(symbol)
                if quote:
                    try:
                        action = strategy(symbol, quote, positions)
                        if action:
                            side, qty = action
                            if qty > 0:
                                result = self.broker.submit_order(symbol, side, qty)
                                if result.get("status") == "filled":
                                    self._trades.append(result)
                    except Exception as e:
                        logger.warning(f"Strategy error for {symbol}: {e}")

            # Record equity
            portfolio_value = self.broker.get_portfolio_value()
            self._equity_curve.append((timestamp, portfolio_value))

        # Calculate metrics
        return self._calculate_results()

    def run_with_engine(
        self,
        trading_engine: Any,
    ) -> BacktestResult:
        """
        Run backtest using a TradingEngine instance.

        This method allows backtesting the full trading pipeline
        including prediction, blending, arbitration, etc.

        Args:
            trading_engine: TradingEngine instance configured with
                           this engine's broker and data provider

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest with TradingEngine: {self.config.start_date} to {self.config.end_date}")

        # Load historical data
        try:
            self._historical_data = self.data_loader.load_multiple(
                self.config.symbols,
                self._start,
                self._end,
                self.config.interval,
            )
        except DataLoadError:
            self._historical_data = {}

        # Fall back to synthetic data if no real data available
        if not self._historical_data:
            logger.info("No historical data found, using synthetic data")
            self._historical_data = generate_synthetic_data(
                self.config.symbols,
                self._start,
                self._end,
                self.config.interval,
            )

        if not self._historical_data:
            logger.error("No data available for backtest")
            return self._empty_result()

        # Get trading dates
        all_dates = set()
        for df in self._historical_data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)

        logger.info(f"Simulating {len(trading_dates)} trading periods")

        # Run simulation
        for timestamp in trading_dates:
            # Update simulation time
            self._clock.set_time(timestamp)
            self.broker.set_time(timestamp)

            # Update quotes
            self._update_quotes(timestamp)

            # Run trading cycle
            try:
                result = trading_engine.run_cycle(
                    self.config.symbols,
                    current_time=timestamp,
                )

                # Record trades
                for decision in result.decisions:
                    if hasattr(decision, "to_dict"):
                        if decision.decision == "TRADE" and decision.shares != 0:
                            self._trades.append(decision.to_dict())

            except Exception as e:
                logger.warning(f"Trading cycle error at {timestamp}: {e}")

            # Record equity
            portfolio_value = self.broker.get_portfolio_value()
            self._equity_curve.append((timestamp, portfolio_value))

        # Calculate metrics
        return self._calculate_results()

    def reset(self) -> None:
        """Reset engine state for a new backtest run."""
        self._clock = SimulatedClock(self._start)
        self.broker.reset()
        self._equity_curve.clear()
        self._trades.clear()
        self._historical_data.clear()

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _parse_date(self, date: Union[str, datetime]) -> datetime:
        """Parse date string to datetime."""
        if isinstance(date, str):
            dt = datetime.fromisoformat(date)
        else:
            dt = date

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    def _update_quotes(self, timestamp: datetime) -> None:
        """Update broker quotes from historical data."""
        for symbol, df in self._historical_data.items():
            try:
                quote = self.data_loader.get_quote_at(symbol, timestamp, df)
                self.broker.set_quote(symbol, quote)
            except DataLoadError:
                pass  # Skip symbols without data at this timestamp

    def _default_strategy(
        self,
        symbol: str,
        quote: Any,
        positions: Dict[str, float],
    ) -> Optional[tuple[str, float]]:
        """Default buy-and-hold strategy (no action after initial buy)."""
        return None

    def _initialize_positions(self) -> None:
        """Initialize equal-weight positions for buy-and-hold."""
        # Update quotes first
        first_date = self._start
        for symbol, df in self._historical_data.items():
            if not df.empty:
                first_date = min(first_date, df.index[0])

        self._clock.set_time(first_date)
        self._update_quotes(first_date)

        # Calculate equal weight per symbol
        n_symbols = len(self.config.symbols)
        if n_symbols == 0:
            return

        cash_per_symbol = self.broker.get_cash() / n_symbols

        for symbol in self.config.symbols:
            quote = self.broker._quotes.get(symbol)
            if quote and quote.ask > 0:
                shares = int(cash_per_symbol / quote.ask)
                if shares > 0:
                    result = self.broker.submit_order(symbol, "BUY", shares)
                    if result.get("status") == "filled":
                        self._trades.append(result)

    def _calculate_results(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self._equity_curve:
            return self._empty_result()

        # Build equity curve series
        equity = pd.Series(
            {t: v for t, v in self._equity_curve},
            name="equity",
        )

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Final value and total return
        final_value = equity.iloc[-1]
        total_return = (final_value / self.config.initial_cash) - 1

        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            # Annualize based on interval
            if self.config.interval == "1d":
                periods_per_year = 252
            elif self.config.interval == "1h":
                periods_per_year = 252 * 6.5  # Trading hours per year
            else:
                periods_per_year = 252

            sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Win rate (based on fill prices vs average entry)
        wins = 0
        total_trades = len(self._trades)
        if total_trades > 0:
            for trade in self._trades:
                if trade.get("side") == "SELL":
                    # Check if sold for profit
                    symbol = trade.get("symbol")
                    fill_price = trade.get("fill_price", 0)
                    cost = self.broker.get_position_cost(symbol)
                    if fill_price > cost:
                        wins += 1

            # For buy-and-hold, use final unrealized P&L
            if wins == 0 and total_return > 0:
                wins = sum(1 for pnl in self.broker.get_unrealized_pnl().values() if pnl > 0)
                total_trades = len(self.broker.get_positions()) or 1

        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Total fees
        total_fees = self.broker.get_total_fees()

        # Average slippage
        slippage_values = [t.get("slippage_bps", 0) for t in self._trades]
        avg_slippage = np.mean(slippage_values) if slippage_values else 0.0

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
            total_fees=total_fees,
            avg_slippage_bps=avg_slippage,
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result for failed backtest."""
        return BacktestResult(
            config=self.config,
            trades=[],
            daily_returns=pd.Series(dtype=float),
            equity_curve=pd.Series(dtype=float),
            final_value=self.config.initial_cash,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            total_fees=0.0,
            avg_slippage_bps=0.0,
        )


def run_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 5.0,
    fee_bps: float = 1.0,
    strategy: Optional[callable] = None,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Starting capital
        slippage_bps: Slippage in basis points
        fee_bps: Fee in basis points
        strategy: Optional strategy function

    Returns:
        BacktestResult with performance metrics
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_cash=initial_cash,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
    )

    engine = BacktestEngine(config)
    return engine.run(strategy=strategy)
