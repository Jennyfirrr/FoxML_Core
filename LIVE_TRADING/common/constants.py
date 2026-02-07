"""
Live Trading Constants
======================

Central location for all constants used in LIVE_TRADING module.
These serve as fallback defaults when config values are not available.

For runtime configuration, always use:
    from CONFIG.config_loader import get_cfg
    value = get_cfg("live_trading.xyz", default=DEFAULT_CONFIG["xyz"])
"""

from __future__ import annotations

from typing import Any, Dict, List

# =============================================================================
# Horizons
# =============================================================================

# Supported horizons (in order from shortest to longest)
HORIZONS: List[str] = ["5m", "10m", "15m", "30m", "60m", "1d"]

# Horizon to minutes mapping
HORIZON_MINUTES: Dict[str, int] = {
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 390,  # Full trading day in minutes
}

# Horizon to seconds mapping
HORIZON_SECONDS: Dict[str, int] = {h: m * 60 for h, m in HORIZON_MINUTES.items()}

# =============================================================================
# Model Families
# =============================================================================

# All model families from TRAINING
FAMILIES: List[str] = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "MLP",
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
    "RewardBased",
    "QuantileLightGBM",
    "NGBoost",
    "GMMRegime",
    "ChangePoint",
    "FTRLProximal",
    "VAE",
    "GAN",
    "Ensemble",
    "MetaLearning",
    "MultiTask",
]

# Sequential families that require SeqBufferManager for inference
SEQUENTIAL_FAMILIES: List[str] = [
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
]

# TensorFlow/Keras families (load .h5 files)
TF_FAMILIES: List[str] = [
    "MLP",
    "CNN1D",
    "LSTM",
    "Transformer",
    "TabCNN",
    "TabLSTM",
    "TabTransformer",
    "VAE",
    "GAN",
]

# Tree-based families (direct pickle load + predict)
TREE_FAMILIES: List[str] = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "QuantileLightGBM",
    "NGBoost",
]

# Experimental families (require experimental flag)
EXPERIMENTAL_FAMILIES: List[str] = [
    "GMMRegime",
    "ChangePoint",
    "FTRLProximal",
    "VAE",
    "GAN",
    "MetaLearning",
    "MultiTask",
]

# =============================================================================
# Barrier Targets
# =============================================================================

# Barrier target prefixes for gating
BARRIER_TARGETS: Dict[str, str] = {
    "will_peak": "will_peak_5m",
    "will_valley": "will_valley_5m",
    "y_will_peak": "y_will_peak_5m",
    "y_will_valley": "y_will_valley_5m",
}

# =============================================================================
# Order Constants
# =============================================================================

# Order types
ORDER_TYPE_MARKET: str = "market"
ORDER_TYPE_LIMIT: str = "limit"

# Side constants
SIDE_BUY: str = "BUY"
SIDE_SELL: str = "SELL"

# Trade decision results
DECISION_TRADE: str = "TRADE"
DECISION_HOLD: str = "HOLD"
DECISION_BLOCKED: str = "BLOCKED"

# =============================================================================
# Standardization Constants
# =============================================================================

# Z-score standardization bounds
ZSCORE_CLIP_MIN: float = -3.0
ZSCORE_CLIP_MAX: float = 3.0

# Rolling window for standardization (trading days)
STANDARDIZATION_WINDOW: int = 10

# Minimum IC for model inclusion
MIN_IC_THRESHOLD: float = 0.01

# =============================================================================
# Confidence/Freshness Constants
# =============================================================================

# Freshness decay time constants (seconds)
FRESHNESS_TAU: Dict[str, float] = {
    "5m": 150.0,
    "10m": 300.0,
    "15m": 450.0,
    "30m": 900.0,
    "60m": 1800.0,
    "1d": 7200.0,
}

# Capacity participation rate
CAPACITY_KAPPA: float = 0.1

# =============================================================================
# Default Configuration Values
# =============================================================================

# These are fallback defaults when config values are not available.
# At runtime, always use get_cfg() to load from config files.
DEFAULT_CONFIG: Dict[str, Any] = {
    # Blending parameters
    "ridge_lambda": 0.15,
    "temperature": {
        "5m": 0.75,
        "10m": 0.85,
        "15m": 0.90,
        "30m": 1.0,
        "60m": 1.0,
        "1d": 1.0,
    },
    # Cost model parameters
    "k1_spread": 1.0,  # Spread penalty coefficient
    "k2_volatility": 0.15,  # Volatility timing coefficient
    "k3_impact": 1.0,  # Market impact coefficient
    # Barrier gate parameters
    "g_min": 0.2,  # Minimum gate value
    "gamma": 1.0,  # Peak penalty exponent
    "delta": 0.5,  # Valley bonus exponent
    "peak_threshold": 0.6,  # Block long if p_peak > threshold
    "valley_threshold": 0.55,  # Prefer entry if p_valley > threshold
    # Sizing parameters
    "z_max": 3.0,  # Maximum z-score for sizing
    "max_weight": 0.05,  # Maximum position weight (5%)
    "gross_target": 0.5,  # Target gross exposure (50%)
    "no_trade_band": 0.008,  # No-trade band (80 bps)
    # Risk parameters
    "max_daily_loss_pct": 2.0,  # Kill switch: daily loss
    "max_drawdown_pct": 10.0,  # Kill switch: drawdown
    "max_position_pct": 20.0,  # Max single position
    "max_gross_exposure": 1.0,  # Max gross exposure (100%)
    "spread_max_bps": 12.0,  # Max spread for trading
    "quote_age_max_ms": 200.0,  # Max quote age
    "latency_warn_ms": 2000.0,  # Latency warning threshold
    # Paper broker defaults
    "slippage_bps": 5.0,  # Simulated slippage
    "fee_bps": 1.0,  # Simulated fee
    "initial_cash": 100_000.0,  # Starting capital
}

# =============================================================================
# Validation Helpers
# =============================================================================


def is_valid_horizon(h: str) -> bool:
    """Check if horizon is valid."""
    return h in HORIZONS


def is_valid_family(f: str) -> bool:
    """Check if family is valid."""
    return f in FAMILIES


def is_sequential_family(f: str) -> bool:
    """Check if family requires sequential buffer."""
    return f in SEQUENTIAL_FAMILIES


def is_tree_family(f: str) -> bool:
    """Check if family is tree-based."""
    return f in TREE_FAMILIES


def horizon_to_minutes(h: str) -> int:
    """Convert horizon string to minutes."""
    if h not in HORIZON_MINUTES:
        raise ValueError(f"Unknown horizon: {h}")
    return HORIZON_MINUTES[h]


# =============================================================================
# Phase 21: Configurable Horizons (Interval-Agnostic Pipeline)
# =============================================================================

def get_configured_horizons() -> List[str]:
    """
    Phase 21: Get horizons from config with fallback to constants.

    This function loads horizons from config, allowing experiments to
    use non-standard horizons (1m, 2m, 3m) without code changes.

    Returns:
        List of horizon strings (e.g., ["5m", "15m", "60m"])

    Config path: live_trading.horizons
    """
    try:
        from CONFIG.config_loader import get_cfg
        configured = get_cfg("live_trading.horizons", default=None)
        if configured is not None and isinstance(configured, list):
            return configured
    except Exception:
        pass
    return HORIZONS


def get_configured_horizon_minutes() -> Dict[str, int]:
    """
    Phase 21: Get horizon to minutes mapping from config.

    Returns:
        Dict mapping horizon strings to minutes
    """
    horizons = get_configured_horizons()
    result = {}

    for h in horizons:
        # First check if in default mapping
        if h in HORIZON_MINUTES:
            result[h] = HORIZON_MINUTES[h]
        else:
            # Try to parse horizon string (e.g., "1m" -> 1, "2h" -> 120)
            try:
                result[h] = _parse_horizon_to_minutes(h)
            except ValueError:
                pass

    return result


def _parse_horizon_to_minutes(h: str) -> int:
    """
    Parse horizon string to minutes.

    Args:
        h: Horizon string (e.g., "5m", "1h", "1d")

    Returns:
        Minutes as integer

    Raises:
        ValueError: If format is invalid
    """
    h = h.strip().lower()

    if h.endswith('m'):
        return int(h[:-1])
    elif h.endswith('h'):
        return int(h[:-1]) * 60
    elif h.endswith('d'):
        return int(h[:-1]) * 390  # Trading day
    else:
        raise ValueError(f"Cannot parse horizon: {h}")


def discover_horizons_from_models(model_dir: str) -> List[str]:
    """
    Phase 21: Discover available horizons from model metadata.

    This function scans model directories to find what horizons
    have trained models available.

    Args:
        model_dir: Path to model artifacts directory

    Returns:
        List of horizon strings found in model metadata
    """
    import os
    import json
    from pathlib import Path

    found_horizons = set()

    try:
        model_path = Path(model_dir)
        if not model_path.exists():
            return list(found_horizons)

        # Look for model_meta.json files
        for meta_file in model_path.rglob("model_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)

                # Extract horizon from target name (e.g., "fwd_ret_60m" -> "60m")
                target = meta.get('target', '')
                if '_' in target:
                    parts = target.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].endswith('m'):
                        found_horizons.add(parts[1])
            except Exception:
                continue

    except Exception:
        pass

    # Sort by minutes for consistent ordering
    def _horizon_sort_key(h: str) -> int:
        try:
            return _parse_horizon_to_minutes(h)
        except ValueError:
            return 9999

    return sorted(found_horizons, key=_horizon_sort_key)
