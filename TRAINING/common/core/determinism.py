# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
DEPRECATED: This module is deprecated. Use TRAINING.common.determinism instead.

All functions are re-exported from the canonical module for backward compatibility.
"""

import warnings

# Re-export from canonical module
from TRAINING.common.determinism import (
    set_global_determinism as _set_global_determinism,
    get_deterministic_params as _get_deterministic_params,
    seed_for as _seed_for,
    init_determinism_from_config,
    stable_seed_from,
    is_strict_mode,
    BASE_SEED,
)

__all__ = [
    "set_global_determinism",
    "ensure_deterministic_environment",
    "get_deterministic_params",
    "seed_for",
    # Re-export canonical functions for convenience
    "init_determinism_from_config",
    "stable_seed_from",
    "is_strict_mode",
    "BASE_SEED",
]


def set_global_determinism(seed: int = 42) -> None:
    """Set global determinism for all random operations.

    DEPRECATED: Use TRAINING.common.determinism.set_global_determinism instead.
    """
    warnings.warn(
        "TRAINING.common.core.determinism.set_global_determinism is deprecated. "
        "Use TRAINING.common.determinism.set_global_determinism instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _set_global_determinism(seed)


def ensure_deterministic_environment() -> None:
    """Ensure deterministic environment for training.

    DEPRECATED: Use TRAINING.common.determinism.init_determinism_from_config instead.
    """
    warnings.warn(
        "TRAINING.common.core.determinism.ensure_deterministic_environment is deprecated. "
        "Use TRAINING.common.determinism.init_determinism_from_config instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # The canonical module handles environment setup via init_determinism_from_config
    init_determinism_from_config()


def get_deterministic_params(seed: int = None) -> dict:
    """Get deterministic parameters for model training.

    DEPRECATED: Use TRAINING.common.determinism.get_deterministic_params instead.
    """
    warnings.warn(
        "TRAINING.common.core.determinism.get_deterministic_params is deprecated. "
        "Use TRAINING.common.determinism.get_deterministic_params instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_deterministic_params("lightgbm", seed or 42)


def seed_for(operation: str, base_seed: int = 42) -> int:
    """Get a deterministic seed for a specific operation.

    DEPRECATED: Use TRAINING.common.determinism.stable_seed_from instead.
    """
    warnings.warn(
        "TRAINING.common.core.determinism.seed_for is deprecated. "
        "Use TRAINING.common.determinism.stable_seed_from instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return stable_seed_from([operation], modulo=2**31-1)
