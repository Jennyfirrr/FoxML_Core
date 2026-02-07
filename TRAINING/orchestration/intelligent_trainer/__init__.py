# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Trainer Module

Modular components for intelligent training orchestrator.

This package contains:
- utils: JSON serialization helpers, sample size binning
- cli: Command-line argument parsing
- config: Experiment config loading helpers
- caching: Cache management for rankings and features
"""

# ============================================================================
# CRITICAL: Import repro_bootstrap FIRST before ANY numeric libraries
# This sets thread env vars BEFORE numpy/torch/sklearn are imported.
# DO NOT move this import or add imports above it!
# ============================================================================
import TRAINING.common.repro_bootstrap  # noqa: F401 - side effects only

from .utils import json_default, get_sample_size_bin
from .cli import create_argument_parser, parse_args
from .config import get_experiment_config_path, load_experiment_config_safe
from .caching import (
    get_cache_key,
    load_cached_rankings,
    save_cached_rankings,
    get_feature_cache_path,
    load_cached_features,
    save_cached_features,
)
from .pipeline_stages import PipelineStageMixin

# Re-export IntelligentTrainer and main from the sibling module file
# This is needed because Python imports the package dir over the .py file
import importlib.util
import sys
from pathlib import Path

# Load the intelligent_trainer.py module file (sibling to this package)
_module_path = Path(__file__).parent.parent / "intelligent_trainer.py"
if _module_path.exists():
    _spec = importlib.util.spec_from_file_location("intelligent_trainer_module", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["TRAINING.orchestration.intelligent_trainer_module"] = _module
    _spec.loader.exec_module(_module)
    
    # Re-export the main class and function
    IntelligentTrainer = _module.IntelligentTrainer
    main = _module.main
    # Re-export CS ranking helpers (Phase 5)
    is_cs_ranking_enabled = _module.is_cs_ranking_enabled
    get_cs_ranking_config = _module.get_cs_ranking_config
else:
    # Fallback: class not available
    IntelligentTrainer = None
    main = None
    is_cs_ranking_enabled = None
    get_cs_ranking_config = None

__all__ = [
    # Utils
    'json_default',
    'get_sample_size_bin',
    # CLI
    'create_argument_parser',
    'parse_args',
    # Config helpers
    'get_experiment_config_path',
    'load_experiment_config_safe',
    # Caching
    'get_cache_key',
    'load_cached_rankings',
    'save_cached_rankings',
    'get_feature_cache_path',
    'load_cached_features',
    'save_cached_features',
    # Pipeline stage mixin
    'PipelineStageMixin',
    # Main class and function (from sibling .py file)
    'IntelligentTrainer',
    'main',
    # CS ranking helpers (Phase 5)
    'is_cs_ranking_enabled',
    'get_cs_ranking_config',
]

