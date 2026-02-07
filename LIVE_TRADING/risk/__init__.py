"""
Risk management for LIVE_TRADING.

Provides:
- DrawdownMonitor for tracking portfolio drawdown
- ExposureTracker for gross/net exposure limits
- RiskGuardrails for kill switches and risk checks
- get_risk_status() for dashboard access
"""

from .drawdown import DrawdownMonitor, DrawdownState
from .exposure import ExposureState, ExposureTracker
from .guardrails import (
    RiskGuardrails,
    RiskStatus,
    RiskWarning,
    DashboardRiskStatus,
    get_guardrails,
    set_guardrails,
    get_risk_status,
)

__all__ = [
    "DrawdownMonitor",
    "DrawdownState",
    "ExposureTracker",
    "ExposureState",
    "RiskGuardrails",
    "RiskStatus",
    "RiskWarning",
    "DashboardRiskStatus",
    "get_guardrails",
    "set_guardrails",
    "get_risk_status",
]
