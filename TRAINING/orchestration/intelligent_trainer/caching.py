# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Trainer Caching

Cache management utilities for intelligent training orchestrator.
Handles target ranking cache and feature selection cache.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_cache_key(symbols: List[str], config_hash: str) -> str:
    """
    Generate cache key from symbols and config.

    Uses centralized config hashing for consistency.

    Args:
        symbols: List of symbol names
        config_hash: Hash of configuration

    Returns:
        32-character cache key
    """
    from TRAINING.common.utils.config_hashing import compute_config_hash_from_values
    # Use centralized config hashing for consistency
    return compute_config_hash_from_values(
        symbols=sorted(symbols),
        config_hash=config_hash
    )[:32]  # Truncate for backward compatibility


def load_cached_rankings(
    cache_path: Path,
    cache_key: str,
    use_cache: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Load cached target rankings.

    Args:
        cache_path: Path to the ranking cache file
        cache_key: Key to look up in cache
        use_cache: Whether to use cache (if False, returns None)

    Returns:
        List of ranking dicts if found, None otherwise
    """
    if not use_cache or not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get(cache_key)
    except Exception as e:
        logger.warning(f"Failed to load ranking cache: {e}")
        return None


def save_cached_rankings(
    cache_path: Path,
    cache_key: str,
    rankings: List[Dict[str, Any]]
) -> None:
    """
    Save target rankings to cache.

    Uses atomic write for crash consistency.

    Args:
        cache_path: Path to the ranking cache file
        cache_key: Key to store rankings under
        rankings: List of ranking dicts to cache
    """
    try:
        cache_data = {}
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

        cache_data[cache_key] = rankings
        # SST: Use write_atomic_json for atomic write with canonical serialization
        from TRAINING.common.utils.file_utils import write_atomic_json
        write_atomic_json(cache_path, cache_data)
    except Exception as e:
        logger.warning(f"Failed to save ranking cache: {e}")


def get_feature_cache_path(cache_dir: Path, target: str) -> Path:
    """
    Get cache path for feature selection results.

    Args:
        cache_dir: Base cache directory for feature selection
        target: Target name

    Returns:
        Path to the feature cache file for this target
    """
    return cache_dir / f"{target}.json"


def load_cached_features(cache_dir: Path, target: str) -> Optional[List[str]]:
    """
    Load cached feature selection for a target.

    Args:
        cache_dir: Base cache directory for feature selection
        target: Target name

    Returns:
        List of selected feature names if found, None otherwise
    """
    from TRAINING.common.utils.cache_manager import load_cache
    cache_path = get_feature_cache_path(cache_dir, target)
    cache_data = load_cache(cache_path, verify_hash=False)
    if cache_data:
        return cache_data.get('selected_features')
    return None


def save_cached_features(
    cache_dir: Path,
    target: str,
    features: List[str]
) -> None:
    """
    Save feature selection results to cache.

    Args:
        cache_dir: Base cache directory for feature selection
        target: Target name
        features: List of selected feature names
    """
    from TRAINING.common.utils.cache_manager import save_cache
    cache_path = get_feature_cache_path(cache_dir, target)
    cache_data = {
        'target': target,
        'selected_features': features
    }
    save_cache(cache_path, cache_data, include_timestamp=True)
