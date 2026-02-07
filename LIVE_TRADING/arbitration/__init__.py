"""
Arbitration Module
==================

Cost-aware horizon selection with trading cost estimation.
"""

from .cost_model import CostModel, TradingCosts
from .horizon_arbiter import HorizonArbiter, ArbitrationResult

__all__ = [
    "CostModel",
    "TradingCosts",
    "HorizonArbiter",
    "ArbitrationResult",
]
