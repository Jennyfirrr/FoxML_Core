# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Model Family Trainers

Modular trainers for each model family in the multi-model feature selection pipeline.
Each trainer handles model instantiation, training, and returns model + train_score.
"""

from .base import (
    TrainerResult,
    TaskType,
    detect_task_type,
    validate_training_target,
    get_trainer,
    register_trainer,
    TRAINER_REGISTRY,
)

# Import all trainers to register them
from . import lightgbm_trainer
from . import xgboost_trainer
from . import random_forest_trainer
from . import catboost_trainer
from . import neural_trainer
from . import linear_trainers
from . import specialized_trainers
from . import selection_trainers

# Import dispatcher
from .dispatcher import (
    dispatch_trainer,
    get_available_trainers,
    is_trainer_fully_extracted,
)

__all__ = [
    # Types
    'TrainerResult',
    'TaskType',
    # Functions
    'detect_task_type',
    'validate_training_target',
    'get_trainer',
    'register_trainer',
    'dispatch_trainer',
    'get_available_trainers',
    'is_trainer_fully_extracted',
    # Registry
    'TRAINER_REGISTRY',
]
