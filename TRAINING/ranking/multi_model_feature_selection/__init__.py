# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Multi-Model Feature Selection Module

Modular components for the multi-model feature selection pipeline.
Combines feature importance from multiple model families to identify
robust features that have predictive power across diverse architectures.

Submodules:
    - types: Data classes (ModelFamilyConfig, ImportanceResult)
    - config_loader: Configuration loading
    - importance_extractors: Native/SHAP/permutation importance extraction
    - trainers: Model family trainers (modular per-family implementations)
    - symbol_processing: Per-symbol processing
    - aggregation: Cross-model importance aggregation
    - persistence: Results saving/loading
"""

# Core types
from .types import ModelFamilyConfig, ImportanceResult

# Configuration
from .config_loader import load_multi_model_config, get_default_config

# Importance extractors
from .importance_extractors import (
    safe_load_dataframe,
    extract_native_importance,
    extract_shap_importance,
    extract_permutation_importance,
)

# Trainers (modular implementations)
from .trainers import (
    TrainerResult,
    TaskType,
    detect_task_type,
    validate_training_target,
    dispatch_trainer,
    get_available_trainers,
    TRAINER_REGISTRY,
)

# Symbol processing (delegates to parent file for now)
from .symbol_processing import process_single_symbol

# Aggregation (delegates to parent file for now)
from .aggregation import (
    aggregate_multi_model_importance,
    compute_target_confidence,
)

# Persistence (delegates to parent file for now)
from .persistence import (
    save_multi_model_results,
    load_previous_model_results,
    save_model_metadata,
)

__all__ = [
    # Types
    'ModelFamilyConfig',
    'ImportanceResult',
    'TrainerResult',
    'TaskType',
    # Configuration
    'load_multi_model_config',
    'get_default_config',
    # Importance extractors
    'safe_load_dataframe',
    'extract_native_importance',
    'extract_shap_importance',
    'extract_permutation_importance',
    # Trainers
    'detect_task_type',
    'validate_training_target',
    'dispatch_trainer',
    'get_available_trainers',
    'TRAINER_REGISTRY',
    # Symbol processing
    'process_single_symbol',
    # Aggregation
    'aggregate_multi_model_importance',
    'compute_target_confidence',
    # Persistence
    'save_multi_model_results',
    'load_previous_model_results',
    'save_model_metadata',
]
