# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Predictability Ranking - Thin Wrapper Module

This module re-exports functions from the model_evaluation/ subpackage.
For implementation details, see:
- training.py: train_and_evaluate_models()
- ranking.py: evaluate_target_predictability()
- autofix.py: evaluate_target_with_autofix()
- safety.py: enforce_final_safety_gate()

Uses multiple model families to evaluate which of your targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Usage:
  # Import from this module (backward compatible)
  from TRAINING.ranking.predictability.model_evaluation import (
      train_and_evaluate_models,
      evaluate_target_predictability,
      evaluate_target_with_autofix,
  )

  # Or import directly from submodules
  from TRAINING.ranking.predictability.model_evaluation.training import train_and_evaluate_models
  from TRAINING.ranking.predictability.model_evaluation.ranking import evaluate_target_predictability
"""

# Re-export from submodules
from TRAINING.ranking.predictability.model_evaluation.training import train_and_evaluate_models
from TRAINING.ranking.predictability.model_evaluation.ranking import evaluate_target_predictability
from TRAINING.ranking.predictability.model_evaluation.autofix import evaluate_target_with_autofix
from TRAINING.ranking.predictability.model_evaluation.safety import enforce_final_safety_gate

# Re-export from existing helpers
from TRAINING.ranking.predictability.model_evaluation.config_helpers import (
    get_importance_top_fraction,
)
from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import (
    detect_and_fix_leakage,
    LeakageArtifacts,
    compute_suspicion_score,
)
from TRAINING.ranking.predictability.model_evaluation.reporting import (
    log_canonical_summary,
    save_feature_importances,
    log_suspicious_features,
)

__all__ = [
    # Core functions
    "train_and_evaluate_models",
    "evaluate_target_predictability",
    "evaluate_target_with_autofix",
    # Safety
    "enforce_final_safety_gate",
    # Config helpers
    "get_importance_top_fraction",
    # Leakage helpers
    "detect_and_fix_leakage",
    "LeakageArtifacts",
    "compute_suspicion_score",
    # Reporting
    "log_canonical_summary",
    "save_feature_importances",
    "log_suspicious_features",
]
