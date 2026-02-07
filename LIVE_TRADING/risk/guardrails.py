"""
Risk Guardrails
===============

Main risk management coordinator with kill switches.
Integrates drawdown monitoring, exposure tracking, and daily P&L limits.

SST Compliance:
- Uses get_cfg() for configuration
- Uses sorted_items() for deterministic dict iteration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import (
    DailyLossExceeded,
    KillSwitchTriggered,
    MaxDrawdownExceeded,
    RiskError,
)
from LIVE_TRADING.common.types import TradeDecision

from .drawdown import DrawdownMonitor, DrawdownState
from .exposure import ExposureState, ExposureTracker

logger = logging.getLogger(__name__)


@dataclass
class RiskWarning:
    """Individual risk warning with threshold info."""

    warning_type: str
    message: str
    severity: str  # "low", "medium", "high"
    threshold_pct: float
    current_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "type": self.warning_type,
            "message": self.message,
            "severity": self.severity,
            "threshold_pct": self.threshold_pct,
            "current_pct": self.current_pct,
        }


@dataclass
class RiskStatus:
    """
    Current risk status summary.

    Captures all risk metrics and kill switch state.
    """

    is_trading_allowed: bool
    daily_pnl_pct: float
    drawdown_pct: float
    gross_exposure: float
    net_exposure: float
    kill_switch_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "daily_pnl_pct": self.daily_pnl_pct,
            "drawdown_pct": self.drawdown_pct,
            "gross_exposure": self.gross_exposure,
            "is_trading_allowed": self.is_trading_allowed,
            "kill_switch_reason": self.kill_switch_reason,
            "net_exposure": self.net_exposure,
            "warnings": self.warnings,
        }


@dataclass
class DashboardRiskStatus:
    """
    Extended risk status for dashboard display.

    Includes all fields from RiskStatus plus limits and remaining thresholds.
    """

    # Trading permission
    trading_allowed: bool
    kill_switch_active: bool
    kill_switch_reason: Optional[str]

    # Daily P&L
    daily_pnl_pct: float
    daily_loss_limit_pct: float

    # Drawdown
    drawdown_pct: float
    max_drawdown_limit_pct: float

    # Exposure
    gross_exposure: float
    net_exposure: float
    max_gross_exposure: float

    # Warnings
    warnings: List[RiskWarning] = field(default_factory=list)

    @property
    def daily_loss_remaining_pct(self) -> float:
        """Calculate remaining daily loss budget."""
        return self.daily_loss_limit_pct - abs(min(0, self.daily_pnl_pct))

    @property
    def drawdown_remaining_pct(self) -> float:
        """Calculate remaining drawdown budget."""
        return self.max_drawdown_limit_pct - self.drawdown_pct

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "trading_allowed": self.trading_allowed,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "daily_pnl_pct": self.daily_pnl_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "daily_loss_remaining_pct": self.daily_loss_remaining_pct,
            "drawdown_pct": self.drawdown_pct,
            "max_drawdown_limit_pct": self.max_drawdown_limit_pct,
            "drawdown_remaining_pct": self.drawdown_remaining_pct,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "max_gross_exposure": self.max_gross_exposure,
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass
class RiskCheckResult:
    """Result from a risk check."""

    check_name: str
    passed: bool
    message: str
    value: float = 0.0
    limit: float = 0.0


class RiskGuardrails:
    """
    Main risk management coordinator with kill switches.

    Integrates:
    - Drawdown monitoring (kill switch on max drawdown)
    - Daily P&L tracking (kill switch on daily loss limit)
    - Exposure tracking (position and gross limits)

    Kill switches are "sticky" - once triggered, trading stops for the day.

    Example:
        >>> guard = RiskGuardrails(initial_capital=100_000)
        >>> status = guard.check_trading_allowed(95_000, {"AAPL": 0.2})
        >>> if not status.is_trading_allowed:
        ...     print(f"Trading blocked: {status.kill_switch_reason}")
    """

    def __init__(
        self,
        max_daily_loss_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        max_gross_exposure: float | None = None,
        max_position_pct: float | None = None,
        initial_capital: float = 100_000.0,
        clock: Clock | None = None,
    ) -> None:
        """
        Initialize risk guardrails.

        Args:
            max_daily_loss_pct: Kill switch trigger for daily loss.
                               Default from config or 2%.
            max_drawdown_pct: Kill switch trigger for drawdown.
                             Default from config or 10%.
            max_gross_exposure: Maximum gross exposure.
                               Default from config or 1.0 (100%).
            max_position_pct: Maximum single position size.
                             Default from config or 20%.
            initial_capital: Starting capital for P&L tracking.
            clock: Clock instance for time (default: system clock)
        """
        self._clock = clock or get_clock()

        # Load from config with defaults
        self.max_daily_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else get_cfg(
            "live_trading.risk.max_daily_loss_pct",
            default=DEFAULT_CONFIG["max_daily_loss_pct"],
        )

        # Initialize sub-monitors
        self.drawdown_monitor = DrawdownMonitor(max_drawdown_pct=max_drawdown_pct, clock=self._clock)
        self.exposure_tracker = ExposureTracker(
            max_gross_exposure=max_gross_exposure,
            max_position_pct=max_position_pct,
        )

        # Daily P&L tracking
        self._initial_capital = float(initial_capital)
        self._day_start_value = float(initial_capital)
        self._current_date: Optional[date] = None

        # Kill switch state
        self._kill_switch_active = False
        self._kill_reason: Optional[str] = None
        self._kill_time: Optional[datetime] = None

        # Audit trail
        self._risk_events: List[Dict[str, Any]] = []

        logger.info(
            f"RiskGuardrails: max_daily_loss={self.max_daily_loss_pct}%, "
            f"initial_capital=${initial_capital:,.2f}"
        )

    def check_trading_allowed(
        self,
        portfolio_value: float,
        weights: Dict[str, float],
        current_time: datetime | None = None,
    ) -> RiskStatus:
        """
        Check if trading is allowed.

        This is the main method to call before any trade.
        Checks all risk limits and updates kill switch state.

        Args:
            portfolio_value: Current portfolio value
            weights: Current position weights (symbol -> weight)
            current_time: Current timestamp (default: now)

        Returns:
            RiskStatus with trading permission and metrics
        """
        current_time = current_time or self._clock.now()
        today = current_time.date()
        warnings: List[str] = []

        # Reset daily tracking on new day
        if self._current_date != today:
            self._day_start_value = portfolio_value
            self._current_date = today
            self._kill_switch_active = False
            self._kill_reason = None
            self._kill_time = None
            logger.info(f"New trading day: {today}, start value=${portfolio_value:,.2f}")

        # Calculate daily P&L
        daily_pnl_pct = 0.0
        if self._day_start_value > 0:
            daily_pnl_pct = (
                (portfolio_value - self._day_start_value) / self._day_start_value * 100
            )

        # Update drawdown monitor
        dd_state = self.drawdown_monitor.update(portfolio_value, current_time)

        # Calculate exposure
        exp_state = self.exposure_tracker.calculate_exposure(weights)

        # Check kill switches (only if not already triggered)
        if not self._kill_switch_active:
            # Daily loss check
            if daily_pnl_pct < -self.max_daily_loss_pct:
                self._trigger_kill_switch(
                    f"daily_loss: {daily_pnl_pct:.2f}% (limit: {-self.max_daily_loss_pct}%)",
                    current_time,
                )

            # Drawdown check
            elif dd_state.is_exceeded:
                self._trigger_kill_switch(
                    f"drawdown: {dd_state.drawdown_pct:.2f}% (limit: {self.drawdown_monitor.max_drawdown_pct}%)",
                    current_time,
                )

        # Check exposure warnings (not kill switches, but warnings)
        if exp_state.gross_exposure > self.exposure_tracker.max_gross * 0.9:
            warnings.append(
                f"Gross exposure at {exp_state.gross_exposure:.0%} "
                f"(limit: {self.exposure_tracker.max_gross:.0%})"
            )

        return RiskStatus(
            is_trading_allowed=not self._kill_switch_active,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=dd_state.drawdown_pct,
            gross_exposure=exp_state.gross_exposure,
            net_exposure=exp_state.net_exposure,
            kill_switch_reason=self._kill_reason,
            warnings=warnings,
        )

    def validate_trade(
        self,
        symbol: str,
        target_weight: float,
        current_weights: Dict[str, float],
    ) -> RiskCheckResult:
        """
        Validate a proposed trade.

        Checks position limits and exposure impact.

        Args:
            symbol: Symbol to trade
            target_weight: Target weight after trade
            current_weights: Current position weights

        Returns:
            RiskCheckResult with validation outcome
        """
        # Kill switch check
        if self._kill_switch_active:
            return RiskCheckResult(
                check_name="kill_switch",
                passed=False,
                message=f"Kill switch active: {self._kill_reason}",
            )

        # Position limit check
        if not self.exposure_tracker.validate_position(symbol, target_weight):
            return RiskCheckResult(
                check_name="position_limit",
                passed=False,
                message=f"Position {symbol} weight {target_weight:.2%} exceeds limit",
                value=abs(target_weight),
                limit=self.exposure_tracker.max_position,
            )

        # Gross exposure check with new position
        new_weights = dict(current_weights)
        new_weights[symbol] = target_weight
        is_valid, violations = self.exposure_tracker.validate_portfolio(new_weights)

        if not is_valid:
            return RiskCheckResult(
                check_name="gross_exposure",
                passed=False,
                message="; ".join(violations),
            )

        return RiskCheckResult(
            check_name="all_checks",
            passed=True,
            message="Trade validated",
        )

    def apply_risk_adjustments(
        self,
        decisions: List[TradeDecision],
        current_weights: Dict[str, float],
    ) -> List[TradeDecision]:
        """
        Apply risk adjustments to a list of trade decisions.

        Scales down weights if needed to fit exposure limits.

        Args:
            decisions: List of proposed trade decisions
            current_weights: Current position weights

        Returns:
            Adjusted trade decisions
        """
        if self._kill_switch_active:
            logger.warning("Kill switch active - blocking all trades")
            return []

        # Build proposed new weights
        proposed_weights = dict(current_weights)
        for decision in decisions:
            if decision.decision == "TRADE":
                proposed_weights[decision.symbol] = decision.target_weight

        # Check if scaling is needed
        exp_state = self.exposure_tracker.calculate_exposure(proposed_weights)

        if exp_state.gross_exposure <= self.exposure_tracker.max_gross:
            return decisions  # No adjustment needed

        # Scale down to fit
        scaled_weights = self.exposure_tracker.scale_weights_to_limit(proposed_weights)

        # Update decisions with scaled weights
        adjusted = []
        for decision in decisions:
            if decision.symbol in scaled_weights:
                new_weight = scaled_weights[decision.symbol]
                if abs(new_weight - decision.target_weight) > 1e-6:
                    # Weight was adjusted
                    decision.target_weight = new_weight
                    decision.reason = f"{decision.reason} (risk-scaled)"
            adjusted.append(decision)

        return adjusted

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics summary.

        Returns:
            Dict of metric name to value
        """
        dd_state = self.drawdown_monitor.get_current_state()

        return {
            "daily_pnl_pct": self._calculate_daily_pnl_pct(),
            "drawdown_pct": dd_state.drawdown_pct if dd_state else 0.0,
            "peak_value": self.drawdown_monitor.peak_value,
            "max_drawdown_in_window": self.drawdown_monitor.get_max_drawdown_in_window(),
            "kill_switch_active": float(self._kill_switch_active),
        }

    def _calculate_daily_pnl_pct(self) -> float:
        """Calculate current daily P&L percentage."""
        dd_state = self.drawdown_monitor.get_current_state()
        if dd_state is None or self._day_start_value <= 0:
            return 0.0
        return (dd_state.current_value - self._day_start_value) / self._day_start_value * 100

    def _trigger_kill_switch(self, reason: str, timestamp: datetime) -> None:
        """Trigger kill switch."""
        self._kill_switch_active = True
        self._kill_reason = reason
        self._kill_time = timestamp

        self._risk_events.append({
            "event": "kill_switch_triggered",
            "reason": reason,
            "timestamp": timestamp.isoformat(),
        })

        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

    def reset(self, initial_value: float | None = None) -> None:
        """
        Reset all risk tracking.

        Args:
            initial_value: New initial value (optional)
        """
        value = initial_value or self._initial_capital
        self._initial_capital = value
        self._day_start_value = value
        self._current_date = None
        self._kill_switch_active = False
        self._kill_reason = None
        self._kill_time = None
        self._risk_events.clear()
        self.drawdown_monitor.reset(value)
        logger.info(f"RiskGuardrails reset: initial_value=${value:,.2f}")

    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    @property
    def kill_switch_reason(self) -> Optional[str]:
        """Get kill switch reason if active."""
        return self._kill_reason

    def get_risk_events(self) -> List[Dict[str, Any]]:
        """Get audit trail of risk events."""
        return list(self._risk_events)

    def get_dashboard_status(self) -> DashboardRiskStatus:
        """
        Get extended risk status for dashboard display.

        Returns:
            DashboardRiskStatus with all metrics and limits
        """
        # Get current drawdown state
        dd_state = self.drawdown_monitor.get_current_state()
        drawdown_pct = dd_state.drawdown_pct if dd_state else 0.0

        # Get current daily P&L
        daily_pnl_pct = self._calculate_daily_pnl_pct()

        # Build warnings list
        warnings: List[RiskWarning] = []

        # Daily loss warning (at 70% of limit)
        daily_loss_warn_threshold = self.max_daily_loss_pct * 0.7
        if daily_pnl_pct < -daily_loss_warn_threshold:
            warnings.append(RiskWarning(
                warning_type="daily_loss",
                message=f"Daily P&L at {daily_pnl_pct:.1f}% (limit: -{self.max_daily_loss_pct}%)",
                severity="high" if daily_pnl_pct < -self.max_daily_loss_pct * 0.9 else "medium",
                threshold_pct=self.max_daily_loss_pct,
                current_pct=abs(daily_pnl_pct),
            ))

        # Drawdown warning (at 70% of limit)
        max_dd_pct = self.drawdown_monitor.max_drawdown_pct
        dd_warn_threshold = max_dd_pct * 0.7
        if drawdown_pct > dd_warn_threshold:
            warnings.append(RiskWarning(
                warning_type="drawdown",
                message=f"Drawdown at {drawdown_pct:.1f}% (limit: {max_dd_pct}%)",
                severity="high" if drawdown_pct > max_dd_pct * 0.9 else "medium",
                threshold_pct=max_dd_pct,
                current_pct=drawdown_pct,
            ))

        # Gross exposure warning (at 90% of limit)
        # Note: We don't have a current exposure value here, so we'd need it passed in
        # For now, return without exposure warning

        return DashboardRiskStatus(
            trading_allowed=not self._kill_switch_active,
            kill_switch_active=self._kill_switch_active,
            kill_switch_reason=self._kill_reason,
            daily_pnl_pct=daily_pnl_pct,
            daily_loss_limit_pct=self.max_daily_loss_pct,
            drawdown_pct=drawdown_pct,
            max_drawdown_limit_pct=max_dd_pct,
            gross_exposure=0.0,  # Need to be updated with actual exposure
            net_exposure=0.0,    # Need to be updated with actual exposure
            max_gross_exposure=self.exposure_tracker.max_gross,
            warnings=warnings,
        )


# =============================================================================
# Module-level accessor for dashboard
# =============================================================================

_guardrails_instance: Optional[RiskGuardrails] = None


def set_guardrails(guardrails: Optional[RiskGuardrails]) -> None:
    """
    Set the global guardrails instance for dashboard access.

    Args:
        guardrails: RiskGuardrails instance or None to clear
    """
    global _guardrails_instance
    _guardrails_instance = guardrails


def get_guardrails() -> Optional[RiskGuardrails]:
    """
    Get the global guardrails instance.

    Returns:
        RiskGuardrails instance if set, None otherwise
    """
    return _guardrails_instance


def get_risk_status() -> Optional[DashboardRiskStatus]:
    """
    Get current risk status for dashboard.

    Returns:
        DashboardRiskStatus if guardrails are available, None otherwise
    """
    if _guardrails_instance is None:
        return None
    return _guardrails_instance.get_dashboard_status()
