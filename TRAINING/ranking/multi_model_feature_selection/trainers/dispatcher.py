# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Model trainer dispatcher for multi-model feature selection.

Routes model family training requests to the appropriate trainer function,
with fallback to the original monolithic implementation for complex families
that haven't been fully extracted yet.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    TrainerResult,
    get_trainer,
    TRAINER_REGISTRY,
)

logger = logging.getLogger(__name__)


# Marker error that signals dispatcher to use original implementation
_USE_ORIGINAL_MARKER = "_USE_ORIGINAL_IMPLEMENTATION"


def dispatch_trainer(
    model_family: str,
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> Tuple[Optional[TrainerResult], bool]:
    """Dispatch training to the appropriate trainer.

    Args:
        model_family: Name of the model family
        model_config: Model hyperparameters
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to trainer

    Returns:
        Tuple of (TrainerResult or None, use_fallback)
        - If use_fallback is True, caller should use original implementation
        - If use_fallback is False, TrainerResult contains the result
    """
    trainer = get_trainer(model_family)

    if trainer is None:
        logger.debug(f"No trainer registered for {model_family}, using fallback")
        return None, True

    try:
        result = trainer(
            model_config=model_config,
            X=X,
            y=y,
            feature_names=feature_names,
            model_seed=model_seed,
            **kwargs
        )

        # Check for marker that indicates trainer wants fallback
        if result.error == _USE_ORIGINAL_MARKER:
            logger.debug(f"Trainer {model_family} requested fallback to original implementation")
            return None, True

        return result, False

    except Exception as e:
        logger.warning(f"Trainer {model_family} failed with {e}, using fallback")
        return None, True


def get_available_trainers() -> List[str]:
    """Get list of registered trainer names."""
    return sorted(TRAINER_REGISTRY.keys())


def is_trainer_fully_extracted(model_family: str) -> bool:
    """Check if a trainer is fully extracted (not using fallback).

    Args:
        model_family: Name of the model family

    Returns:
        True if trainer is fully extracted and doesn't require fallback
    """
    trainer = get_trainer(model_family)
    if trainer is None:
        return False

    # Check module for extraction flag
    module = trainer.__module__
    try:
        import importlib
        mod = importlib.import_module(module)
        # Look for _FULLY_EXTRACTED flags
        family_upper = model_family.upper().replace('-', '_')
        flag_name = f"_{family_upper}_FULLY_EXTRACTED"
        return getattr(mod, flag_name, True)  # Default True if no flag
    except Exception:
        return True  # Assume extracted if we can't check
