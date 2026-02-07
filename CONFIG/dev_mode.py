# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Dev Mode Helper

Centralized helper for dev_mode configuration to avoid repeated config reads
and circular imports. Uses lru_cache for file reads, but accepts dict for
non-cached path.
"""

import logging
from functools import lru_cache
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_dev_mode_eligibility_overrides_from_file() -> Dict[str, Any]:
    """
    Get dev_mode eligibility overrides from config file (cached).
    
    Returns:
        Dict with dev_mode eligibility overrides (e.g., min_registry_coverage)
    """
    try:
        from CONFIG.config_loader import get_cfg
        
        # Try to get dev_mode eligibility config
        eligibility_config = get_cfg(
            "ranking.metrics_schema.scoring.eligibility.dev_mode",
            default={},
            config_name="ranking_config"
        )
        
        return eligibility_config if isinstance(eligibility_config, dict) else {}
    except Exception as e:
        logger.debug(f"Failed to load dev_mode eligibility overrides from file: {e}")
        return {}


def get_dev_mode_eligibility_overrides_from_dict(eligibility_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get dev_mode eligibility overrides from provided dict (no cache).
    
    Args:
        eligibility_config: Dict containing eligibility config (may have dev_mode section)
    
    Returns:
        Dict with dev_mode eligibility overrides
    """
    if not isinstance(eligibility_config, dict):
        return {}
    
    return eligibility_config.get("dev_mode", {})


def get_dev_mode_eligibility_overrides(eligibility_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get dev_mode eligibility overrides.
    
    If eligibility_config dict is provided, uses non-cached path.
    Otherwise, uses cached file-read path.
    
    Args:
        eligibility_config: Optional dict containing eligibility config
    
    Returns:
        Dict with dev_mode eligibility overrides
    """
    if eligibility_config is not None:
        return get_dev_mode_eligibility_overrides_from_dict(eligibility_config)
    else:
        return _get_dev_mode_eligibility_overrides_from_file()


def get_dev_mode() -> bool:
    """
    Get dev_mode flag from config (cached).
    
    Returns:
        True if dev_mode is enabled, False otherwise
    """
    try:
        from CONFIG.config_loader import get_cfg
        
        # Try multiple config paths (SST: check in order of precedence)
        # 1. routing.dev_mode from ranking/targets/configs.yaml
        dev_mode = get_cfg(
            "routing.dev_mode",
            default=None,
            config_name="target_ranking_config"
        )
        if dev_mode is not None:
            return bool(dev_mode)
        
        # 2. routing.dev_mode from pipeline/training/routing.yaml
        dev_mode = get_cfg(
            "routing.dev_mode",
            default=False,
            config_name="routing_config"
        )
        return bool(dev_mode)
    except Exception as e:
        logger.debug(f"Failed to load dev_mode from config: {e}, defaulting to False")
        return False


def get_dev_mode_source() -> str:
    """
    Get the config source for dev_mode (for provenance/tracing).
    
    Returns:
        String indicating which config path was used, or "default"/"unknown" if not found.
        Format: "config_name:key_path" or "default" or "unknown"
    """
    try:
        from CONFIG.config_loader import get_cfg
        
        # Check in order of precedence (same as get_dev_mode)
        # Use default=None to distinguish missing vs present
        dev_mode = get_cfg(
            "routing.dev_mode",
            default=None,
            config_name="target_ranking_config"
        )
        if dev_mode is not None:
            return "target_ranking_config:routing.dev_mode"
        
        # Check second path with default=None to detect if actually present
        dev_mode = get_cfg(
            "routing.dev_mode",
            default=None,
            config_name="routing_config"
        )
        if dev_mode is not None:
            return "routing_config:routing.dev_mode"
        
        # Neither path had a value - return default
        return "default"
    except Exception as e:
        # Don't silently swallow - log at appropriate level
        logger.warning(f"Failed to determine dev_mode source: {e}, returning 'unknown'")
        return "unknown"


def clear_cache():
    """
    Clear the cached file-read function.
    
    Only clears the cached function, not the dict-accepting function.
    """
    _get_dev_mode_eligibility_overrides_from_file.cache_clear()
