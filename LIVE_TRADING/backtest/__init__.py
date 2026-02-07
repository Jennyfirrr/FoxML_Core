"""
Backtesting Module
==================

Historical simulation for strategy validation.

Components:
- BacktestEngine: Main orchestrator for running backtests
- BacktestConfig: Configuration for backtest runs
- BacktestResult: Results container with performance metrics
- BacktestBroker: Simulated broker with slippage/fees
- HistoricalDataLoader: Load historical data from files or providers
- PerformanceReport: Generate reports from results

Usage:
    >>> from LIVE_TRADING.backtest import BacktestEngine, BacktestConfig
    >>> config = BacktestConfig(
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     symbols=["SPY", "AAPL"],
    ... )
    >>> engine = BacktestEngine(config)
    >>> result = engine.run()
    >>> print(f"Total return: {result.total_return:.2%}")

Quick backtest:
    >>> from LIVE_TRADING.backtest import run_backtest
    >>> result = run_backtest(["SPY"], "2023-01-01", "2023-12-31")
"""

from LIVE_TRADING.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from LIVE_TRADING.backtest.broker import BacktestBroker
from LIVE_TRADING.backtest.data_loader import (
    HistoricalDataLoader,
    DataLoadError,
    generate_synthetic_data,
)
from LIVE_TRADING.backtest.report import (
    PerformanceReport,
    PerformanceMetrics,
    compare_results,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    # Broker
    "BacktestBroker",
    # Data
    "HistoricalDataLoader",
    "DataLoadError",
    "generate_synthetic_data",
    # Reporting
    "PerformanceReport",
    "PerformanceMetrics",
    "compare_results",
]
