"""
Performance Report
==================

Generate performance reports from backtest results.

Features:
- Text summary report
- Key metrics calculation
- Trade analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from LIVE_TRADING.backtest.engine import BacktestResult


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    num_trades: int
    total_fees: float
    avg_holding_period_days: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "num_trades": self.num_trades,
            "total_fees": self.total_fees,
            "avg_holding_period_days": self.avg_holding_period_days,
        }


class PerformanceReport:
    """
    Performance report generator.

    Example:
        >>> result = engine.run()
        >>> report = PerformanceReport(result)
        >>> print(report.summary())
    """

    def __init__(self, result: "BacktestResult"):
        """
        Initialize report generator.

        Args:
            result: Backtest result to analyze
        """
        self.result = result
        self._metrics: Optional[PerformanceMetrics] = None

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get detailed performance metrics."""
        if self._metrics is None:
            self._metrics = self._calculate_metrics()
        return self._metrics

    def summary(self) -> str:
        """
        Generate text summary.

        Returns:
            Formatted summary string
        """
        r = self.result
        m = self.metrics

        return f"""
================================================================================
                            BACKTEST RESULTS
================================================================================

Period: {r.config.start_date} to {r.config.end_date}
Symbols: {', '.join(r.config.symbols)}
Initial Capital: ${r.config.initial_cash:,.2f}

PERFORMANCE
--------------------------------------------------------------------------------
Final Value:        ${r.final_value:,.2f}
Total Return:       {r.total_return:+.2%}
Annualized Return:  {m.annualized_return:+.2%}
Sharpe Ratio:       {m.sharpe_ratio:.2f}
Sortino Ratio:      {m.sortino_ratio:.2f}
Max Drawdown:       {r.max_drawdown:.2%}
Calmar Ratio:       {m.calmar_ratio:.2f}

TRADING
--------------------------------------------------------------------------------
Total Trades:       {r.num_trades}
Win Rate:           {m.win_rate:.1%}
Profit Factor:      {m.profit_factor:.2f}
Avg Trade Return:   {m.avg_trade_return:+.2%}
Total Fees:         ${r.total_fees:,.2f}
Avg Slippage:       {r.avg_slippage_bps:.1f} bps

SIMULATION
--------------------------------------------------------------------------------
Slippage Model:     {r.config.slippage_bps:.1f} bps
Fee Model:          {r.config.fee_bps:.1f} bps
================================================================================
"""

    def trade_analysis(self) -> pd.DataFrame:
        """
        Analyze individual trades.

        Returns:
            DataFrame with trade details
        """
        if not self.result.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.result.trades)

        # Add calculated columns if data available
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        if "fill_price" in df.columns and "qty" in df.columns:
            df["notional"] = df["fill_price"] * df["qty"]

        return df

    def monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns.

        Returns:
            Series of monthly returns
        """
        if self.result.equity_curve.empty:
            return pd.Series(dtype=float)

        equity = self.result.equity_curve
        monthly = equity.resample("ME").last()
        return monthly.pct_change().dropna()

    def drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series.

        Returns:
            Series of drawdowns
        """
        if self.result.equity_curve.empty:
            return pd.Series(dtype=float)

        equity = self.result.equity_curve
        cummax = equity.cummax()
        return (equity - cummax) / cummax

    def rolling_sharpe(self, window: int = 30) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window size

        Returns:
            Series of rolling Sharpe ratios
        """
        if self.result.daily_returns.empty:
            return pd.Series(dtype=float)

        returns = self.result.daily_returns
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # Annualize
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return sharpe.dropna()

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate detailed performance metrics."""
        r = self.result

        # Duration in years
        start = datetime.fromisoformat(r.config.start_date)
        end = datetime.fromisoformat(r.config.end_date)
        years = (end - start).days / 365.25

        # Annualized return
        if years > 0 and r.total_return > -1:
            ann_return = (1 + r.total_return) ** (1 / years) - 1
        else:
            ann_return = r.total_return

        # Sortino ratio (uses downside deviation)
        if not r.daily_returns.empty:
            downside = r.daily_returns[r.daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = r.daily_returns.mean() / downside.std() * np.sqrt(252)
            else:
                sortino = r.sharpe_ratio
        else:
            sortino = 0.0

        # Calmar ratio (annualized return / max drawdown)
        if r.max_drawdown > 0:
            calmar = ann_return / r.max_drawdown
        else:
            calmar = 0.0

        # Profit factor (gross profits / gross losses)
        profits = 0.0
        losses = 0.0
        for trade in r.trades:
            pnl = trade.get("pnl", 0)
            if pnl > 0:
                profits += pnl
            else:
                losses += abs(pnl)

        if losses > 0:
            profit_factor = profits / losses
        else:
            profit_factor = float("inf") if profits > 0 else 0.0

        # Average trade return
        if r.trades:
            trade_returns = [t.get("pnl", 0) / t.get("notional", 1) for t in r.trades if t.get("notional", 0) > 0]
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0
        else:
            avg_trade_return = 0.0

        # Average holding period (simplified)
        avg_holding_days = years * 365.25 / max(r.num_trades, 1)

        return PerformanceMetrics(
            total_return=r.total_return,
            annualized_return=ann_return,
            sharpe_ratio=r.sharpe_ratio,
            sortino_ratio=sortino,
            max_drawdown=r.max_drawdown,
            calmar_ratio=calmar,
            win_rate=r.win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            num_trades=r.num_trades,
            total_fees=r.total_fees,
            avg_holding_period_days=avg_holding_days,
        )


def compare_results(
    results: List["BacktestResult"],
    names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: List of backtest results
        names: Optional names for each result

    Returns:
        DataFrame comparing key metrics
    """
    if names is None:
        names = [f"Strategy {i+1}" for i in range(len(results))]

    data = []
    for name, result in zip(names, results):
        report = PerformanceReport(result)
        metrics = report.metrics

        data.append({
            "Strategy": name,
            "Total Return": f"{result.total_return:.2%}",
            "Ann. Return": f"{metrics.annualized_return:.2%}",
            "Sharpe": f"{result.sharpe_ratio:.2f}",
            "Sortino": f"{metrics.sortino_ratio:.2f}",
            "Max DD": f"{result.max_drawdown:.2%}",
            "Trades": result.num_trades,
            "Win Rate": f"{metrics.win_rate:.1%}",
        })

    return pd.DataFrame(data)
