"""
Live Trading Exceptions
=======================

Exception hierarchy for LIVE_TRADING module.
All exceptions extend FoxMLError for consistency with TRAINING.
"""

from __future__ import annotations

from TRAINING.common.exceptions import FoxMLError


class LiveTradingError(FoxMLError):
    """Base exception for all live trading errors."""

    pass


# =============================================================================
# Broker Exceptions
# =============================================================================


class BrokerError(LiveTradingError):
    """Error communicating with broker."""

    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by broker."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Order rejected for {symbol}: {reason}")


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order."""

    pass


class ConnectionError(BrokerError):
    """Failed to connect to broker."""

    pass


# =============================================================================
# Model Exceptions
# =============================================================================


class ModelLoadError(LiveTradingError):
    """Error loading model from artifacts."""

    def __init__(self, family: str, target: str, reason: str) -> None:
        self.family = family
        self.target = target
        self.reason = reason
        super().__init__(f"Failed to load {family} model for {target}: {reason}")


class InferenceError(LiveTradingError):
    """Error during model inference."""

    def __init__(self, family: str, symbol: str, reason: str) -> None:
        self.family = family
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Inference failed for {family}/{symbol}: {reason}")


class FeatureBuildError(LiveTradingError):
    """Error building features for inference."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Feature build failed for {symbol}: {reason}")


# =============================================================================
# Pipeline Exceptions
# =============================================================================


class GatingError(LiveTradingError):
    """Error in gating logic."""

    pass


class SizingError(LiveTradingError):
    """Error in position sizing."""

    pass


class ArbitrationError(LiveTradingError):
    """Error in horizon arbitration."""

    pass


class BlendingError(LiveTradingError):
    """Error in model blending."""

    pass


# =============================================================================
# Risk Exceptions
# =============================================================================


class RiskError(LiveTradingError):
    """Risk limit violation."""

    pass


class KillSwitchTriggered(RiskError):
    """Kill switch has been triggered - trading must stop."""

    def __init__(self, reason: str, current_value: float, limit: float) -> None:
        self.reason = reason
        self.current_value = current_value
        self.limit = limit
        super().__init__(
            f"Kill switch: {reason} ({current_value:.2f} vs limit {limit:.2f})"
        )


class MaxDrawdownExceeded(KillSwitchTriggered):
    """Maximum drawdown exceeded."""

    def __init__(self, drawdown_pct: float, limit_pct: float) -> None:
        super().__init__("max_drawdown", drawdown_pct, limit_pct)


class DailyLossExceeded(KillSwitchTriggered):
    """Daily loss limit exceeded."""

    def __init__(self, loss_pct: float, limit_pct: float) -> None:
        super().__init__("daily_loss", loss_pct, limit_pct)


class PositionLimitExceeded(RiskError):
    """Position size limit exceeded."""

    def __init__(self, symbol: str, weight: float, limit: float) -> None:
        self.symbol = symbol
        self.weight = weight
        self.limit = limit
        super().__init__(
            f"Position limit exceeded for {symbol}: {weight:.2%} > {limit:.2%}"
        )


# =============================================================================
# Data Quality Exceptions
# =============================================================================


class StaleDataError(LiveTradingError):
    """Data is too old for safe trading."""

    def __init__(self, symbol: str, age_ms: float, max_age_ms: float) -> None:
        self.symbol = symbol
        self.age_ms = age_ms
        self.max_age_ms = max_age_ms
        super().__init__(
            f"Stale data for {symbol}: {age_ms:.0f}ms > {max_age_ms:.0f}ms"
        )


class SpreadTooWideError(LiveTradingError):
    """Spread exceeds maximum threshold."""

    def __init__(self, symbol: str, spread_bps: float, max_bps: float) -> None:
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.max_bps = max_bps
        super().__init__(
            f"Spread too wide for {symbol}: {spread_bps:.1f}bps > {max_bps:.1f}bps"
        )


class InsufficientLiquidityError(LiveTradingError):
    """Insufficient liquidity for the requested trade size."""

    def __init__(self, symbol: str, requested_qty: float, available_qty: float) -> None:
        self.symbol = symbol
        self.requested_qty = requested_qty
        self.available_qty = available_qty
        super().__init__(
            f"Insufficient liquidity for {symbol}: "
            f"requested {requested_qty}, available {available_qty}"
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigValidationError(LiveTradingError):
    """Configuration validation error."""

    pass


class SymbolConfigError(ConfigValidationError):
    """Error loading or parsing symbol configuration."""

    pass
