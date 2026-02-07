# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Configuration Loading for Multi-Model Feature Selection

Standalone implementation to avoid circular imports.
The parent multi_model_feature_selection.py has the authoritative implementation,
but this module provides a lightweight version for use within this package.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Add project root for _REPO_ROOT
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration.

    Uses centralized config loader if available, otherwise falls back to manual path resolution.
    Checks new location first (CONFIG/ranking/features/multi_model.yaml),
    then old location (CONFIG/feature_selection/multi_model.yaml).
    """
    if config_path is None:
        # Try using centralized config loader first
        try:
            from CONFIG.config_loader import get_config_path
            config_path = get_config_path("feature_selection_multi_model")
            if config_path.exists():
                logger.debug(f"Using centralized config loader: {config_path}")
            else:
                config_path = None
        except (ImportError, AttributeError):
            config_path = None

        if config_path is None:
            # Manual path resolution (fallback)
            newest_path = _REPO_ROOT / "CONFIG" / "ranking" / "features" / "multi_model.yaml"
            old_path = _REPO_ROOT / "CONFIG" / "feature_selection" / "multi_model.yaml"

            if newest_path.exists():
                config_path = newest_path
                logger.debug(f"Using new config location: {config_path}")
            elif old_path.exists():
                config_path = old_path
                logger.debug(f"Using old config location: {config_path}")
            else:
                logger.warning(f"Config not found, using defaults")
                return get_default_config()

    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()

    try:
        # Use SST config loader if available
        try:
            from CONFIG.config_builder import load_yaml
            config = load_yaml(config_path)
        except ImportError:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        logger.info(f"Loaded multi-model config from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default multi-model feature selection configuration."""
    return {
        'model_families': {
            'lightgbm': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            },
            'xgboost': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            },
            'random_forest': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {}
            }
        },
        'aggregation': {
            'method': 'mean',
            'top_fraction': 0.10,
            'min_consensus': 2,
            'fallback': {
                'enabled': True,
                'method': 'uniform',
                'threshold': 0.0
            }
        },
        'parallel': {
            'enabled': True,
            'max_workers': None
        }
    }
