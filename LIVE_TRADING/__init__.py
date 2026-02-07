"""
LIVE_TRADING - Execution Engine for FoxML
==========================================

This module implements the live trading execution engine for deploying
models trained by the TRAINING pipeline. It provides:

1. Multi-horizon prediction (5m, 10m, 15m, 30m, 60m, 1d)
2. Ridge risk-parity blending across model families
3. Cost-aware horizon arbitration
4. Barrier gating using peak/valley probabilities
5. Volatility-scaled position sizing
6. Kill switches and risk guardrails

SST Compliance:
- Uses get_cfg() for all configuration
- Uses sorted_items() for deterministic dict iteration
- Uses write_atomic_json() for file writes
"""

# MUST import repro_bootstrap FIRST for determinism
import TRAINING.common.repro_bootstrap  # noqa: F401

from LIVE_TRADING.common.constants import FAMILIES, HORIZONS
from LIVE_TRADING.common.exceptions import LiveTradingError

__version__ = "0.1.0"
__all__ = ["HORIZONS", "FAMILIES", "LiveTradingError"]
