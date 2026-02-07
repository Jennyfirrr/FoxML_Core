"""
Gating Module
=============

Barrier probability gates and spread gates for trade filtering.
"""

from .barrier_gate import BarrierGate, GateResult
from .spread_gate import SpreadGate, SpreadGateResult

__all__ = [
    "BarrierGate",
    "GateResult",
    "SpreadGate",
    "SpreadGateResult",
]
