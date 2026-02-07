# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Specialized model trainers for multi-model feature selection.

Includes: FTRL Proximal, NGBoost
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import (
    TrainerResult,
    register_trainer,
)

logger = logging.getLogger(__name__)


# These trainers have specialized implementations that require careful extraction
_FTRL_FULLY_EXTRACTED = False
_NGBOOST_FULLY_EXTRACTED = False


@register_trainer('ftrl_proximal')
def train_ftrl_proximal(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train an FTRL Proximal model.

    FTRL (Follow The Regularized Leader) is an online learning algorithm
    that's efficient for large-scale sparse features.
    """
    if not _FTRL_FULLY_EXTRACTED:
        return TrainerResult(
            model=None,
            train_score=0.0,
            error="_USE_ORIGINAL_IMPLEMENTATION"
        )

    raise NotImplementedError("FTRL trainer not yet fully extracted")


@register_trainer('ngboost')
def train_ngboost(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train an NGBoost model.

    NGBoost is a natural gradient boosting method for probabilistic prediction.
    """
    if not _NGBOOST_FULLY_EXTRACTED:
        return TrainerResult(
            model=None,
            train_score=0.0,
            error="_USE_ORIGINAL_IMPLEMENTATION"
        )

    raise NotImplementedError("NGBoost trainer not yet fully extracted")
