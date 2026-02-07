# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Trainer Config Helpers

Configuration loading utilities for intelligent training orchestrator.
"""

from pathlib import Path
from typing import Any, Dict

# Import config loader if available
try:
    from CONFIG.config_loader import (
        get_experiment_config_path as _get_experiment_config_path_loader,
        load_experiment_config as _load_experiment_config_loader,
    )
    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False


def get_experiment_config_path(exp_name: str) -> Path:
    """
    Get experiment config path using config loader if available, otherwise fallback.

    Args:
        exp_name: Experiment name (without .yaml extension)

    Returns:
        Path to the experiment config file
    """
    if _CONFIG_LOADER_AVAILABLE:
        return _get_experiment_config_path_loader(exp_name)
    else:
        return Path("CONFIG/experiments") / f"{exp_name}.yaml"


def load_experiment_config_safe(exp_name: str) -> Dict[str, Any]:
    """
    Load experiment config using config loader if available, otherwise fallback.

    This is a safe version that returns an empty dict on errors instead of raising.

    Args:
        exp_name: Experiment name (without .yaml extension)

    Returns:
        Experiment config dict, or empty dict if not found/error
    """
    if _CONFIG_LOADER_AVAILABLE:
        try:
            return _load_experiment_config_loader(exp_name)
        except FileNotFoundError:
            return {}
    else:
        import yaml
        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
        if exp_file.exists():
            with open(exp_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
