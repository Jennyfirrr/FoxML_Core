# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Config Helpers - SST-compliant model config loading

Provides clean helpers for loading model and training configs without
boilerplate sys.path manipulation.

Usage in trainers:
    from TRAINING.common.utils.config_helpers import load_model_config_safe

    class MyTrainer(BaseModelTrainer):
        def __init__(self, config: Dict[str, Any] = None):
            if config is None:
                config = load_model_config_safe("my_model")
            super().__init__(config or {})
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_model_config_safe(model_family: str, variant: str = None) -> Dict[str, Any]:
    """
    Load model config from CONFIG/models/{model_family}.yaml.

    SST-compliant: Uses CONFIG.config_loader instead of sys.path manipulation.

    Args:
        model_family: Model family name (e.g., "lightgbm", "xgboost")
        variant: Optional variant name (e.g., "lightweight", "heavy")

    Returns:
        Model config dictionary (empty dict if not found)
    """
    try:
        from CONFIG.config_loader import load_model_config
        config = load_model_config(model_family, variant)
        logger.debug(f"Loaded {model_family} config from CONFIG/models/{model_family}.yaml")
        return config
    except ImportError:
        logger.warning("CONFIG.config_loader not available, returning empty config")
        return {}
    except FileNotFoundError:
        logger.warning(f"Model config not found: CONFIG/models/{model_family}.yaml")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load {model_family} config: {e}, returning empty config")
        return {}


def load_training_config_safe(config_name: str) -> Dict[str, Any]:
    """
    Load training config from CONFIG/pipeline/training/{config_name}.yaml.

    SST-compliant: Uses CONFIG.config_loader instead of sys.path manipulation.

    Args:
        config_name: Config name (e.g., "routing", "reproducibility")

    Returns:
        Training config dictionary (empty dict if not found)
    """
    try:
        from CONFIG.config_loader import load_training_config
        config = load_training_config(config_name)
        logger.debug(f"Loaded training config from CONFIG/pipeline/training/{config_name}.yaml")
        return config
    except ImportError:
        logger.warning("CONFIG.config_loader not available, returning empty config")
        return {}
    except FileNotFoundError:
        logger.warning(f"Training config not found: CONFIG/pipeline/training/{config_name}.yaml")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load {config_name} training config: {e}, returning empty config")
        return {}


def get_model_param(
    model_family: str,
    param_path: str,
    default: Any = None,
    variant: str = None
) -> Any:
    """
    Get a specific parameter from a model config.

    SST-compliant: Uses CONFIG.config_loader.get_cfg under the hood.

    Args:
        model_family: Model family name (e.g., "lightgbm")
        param_path: Dot-notation path to parameter (e.g., "max_depth" or "callbacks.early_stopping.patience")
        default: Default value if not found
        variant: Optional variant name

    Returns:
        Parameter value or default

    Example:
        max_depth = get_model_param("lightgbm", "max_depth", default=8)
        patience = get_model_param("lightgbm", "callbacks.early_stopping.patience", default=50)
    """
    try:
        from CONFIG.config_loader import get_cfg
        return get_cfg(param_path, default=default, config_name=model_family)
    except Exception:
        return default
