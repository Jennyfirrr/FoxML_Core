# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Model Evaluation Module

Modular components for model evaluation and target predictability ranking.

This package contains:
- safety: Final gatekeeper for safety validation before model training
- training: Model training and evaluation logic
- ranking: Core target predictability ranking
- autofix: Automatic leakage detection and fixing
- config_helpers: Configuration loading helpers
- leakage_helpers: Leakage detection utilities
- reporting: Result reporting and logging

Public API:
    - train_and_evaluate_models: Train multiple model families
    - evaluate_target_predictability: Evaluate target predictability
    - evaluate_target_with_autofix: Evaluate with automatic leakage fixing
    - enforce_final_safety_gate: Final safety validation before training
"""

# Config helpers
from .config_helpers import get_importance_top_fraction

# Leakage helpers
from .leakage_helpers import (
    compute_suspicion_score,
    detect_and_fix_leakage,
    LeakageArtifacts,
    detect_leakage,
)

# Reporting
from .reporting import (
    log_canonical_summary,
    save_feature_importances,
    log_suspicious_features,
)

# Safety gate
from .safety import enforce_final_safety_gate, _enforce_final_safety_gate

# Training (re-exported from parent module via stub)
from .training import train_and_evaluate_models

# Ranking (re-exported from parent module via stub)
from .ranking import evaluate_target_predictability

# Autofix
from .autofix import evaluate_target_with_autofix

# Also import detect_leakage from leakage_detection.py for backward compatibility
# (leakage_helpers.detect_leakage has a different signature)
from TRAINING.ranking.predictability.leakage_detection import detect_leakage as detect_leakage_from_scan

# Backward compatibility: validate_target is now in target_validation module
try:
    from TRAINING.ranking.utils.target_validation import validate_target
except ImportError:
    validate_target = None


__all__ = [
    # Config helpers
    'get_importance_top_fraction',
    # Leakage helpers
    'compute_suspicion_score',
    'detect_and_fix_leakage',
    'LeakageArtifacts',
    'detect_leakage',
    'detect_leakage_from_scan',
    # Reporting
    'log_canonical_summary',
    'save_feature_importances',
    'log_suspicious_features',
    # Safety
    'enforce_final_safety_gate',
    '_enforce_final_safety_gate',
    # Training
    'train_and_evaluate_models',
    # Ranking
    'evaluate_target_predictability',
    # Autofix
    'evaluate_target_with_autofix',
    # Backward compat
    'validate_target',
]
