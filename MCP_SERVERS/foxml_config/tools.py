# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Tool implementations for FoxML Config MCP Server."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache
import time

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# Cache with TTL tracking
_cache_timestamps: Dict[str, float] = {}
_CACHE_TTL_SECONDS = 60.0


def _is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid."""
    if cache_key not in _cache_timestamps:
        return False
    age = time.time() - _cache_timestamps[cache_key]
    return age < _CACHE_TTL_SECONDS


def _update_cache_timestamp(cache_key: str) -> None:
    """Update cache timestamp for a key."""
    _cache_timestamps[cache_key] = time.time()


def clear_cache() -> None:
    """Clear all caches."""
    _cache_timestamps.clear()
    _load_config_yaml.cache_clear()


@lru_cache(maxsize=32)
def _load_config_yaml(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file with caching."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            content = yaml.safe_load(f)
            return content if content else {}
    except Exception:
        return {}


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Get nested value from dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _extract_keys_recursive(
    data: Dict[str, Any],
    prefix: str = "",
    max_depth: int = 3,
    current_depth: int = 0
) -> List[str]:
    """Extract all keys from nested dict with dot notation."""
    if current_depth >= max_depth:
        return []

    keys = []
    for key, value in sorted(data.items()):
        full_key = f"{prefix}.{key}" if prefix else key
        keys.append(full_key)
        if isinstance(value, dict) and current_depth < max_depth - 1:
            keys.extend(_extract_keys_recursive(value, full_key, max_depth, current_depth + 1))
    return keys


def get_config_value(
    path: str,
    config_name: str = "pipeline_config",
    show_precedence: bool = False
) -> Dict[str, Any]:
    """
    Get configuration value with optional precedence chain.

    Args:
        path: Dot-notation config path (e.g., 'pipeline.determinism.base_seed')
        config_name: Config file name (default: pipeline_config)
        show_precedence: Whether to show full precedence chain

    Returns:
        Dict with value, source, and optionally precedence_chain
    """
    from CONFIG.config_loader import get_cfg, CONFIG_DIR, get_config_path

    # Get the value using SST helper
    value = get_cfg(path, default=None, config_name=config_name)

    result = {
        "path": path,
        "value": value,
        "source": config_name,
        "found": value is not None
    }

    if show_precedence:
        # Build precedence chain
        precedence_chain = []

        # Check defaults.yaml
        defaults_path = CONFIG_DIR / "defaults.yaml"
        if defaults_path.exists():
            defaults = _load_config_yaml(str(defaults_path))
            defaults_value = _get_nested_value(defaults, path)
            if defaults_value is not None:
                precedence_chain.append({
                    "source": "defaults.yaml",
                    "value": defaults_value,
                    "priority": 1
                })

        # Check pipeline config
        config_path = get_config_path(config_name)
        if config_path.exists():
            config = _load_config_yaml(str(config_path))
            config_value = _get_nested_value(config, path)
            if config_value is not None:
                precedence_chain.append({
                    "source": str(config_path.relative_to(CONFIG_DIR)),
                    "value": config_value,
                    "priority": 2
                })

        # Check intelligent training config (higher priority)
        intelligent_path = CONFIG_DIR / "pipeline" / "training" / "intelligent.yaml"
        if intelligent_path.exists():
            intelligent = _load_config_yaml(str(intelligent_path))
            intelligent_value = _get_nested_value(intelligent, path)
            if intelligent_value is not None:
                precedence_chain.append({
                    "source": "pipeline/training/intelligent.yaml",
                    "value": intelligent_value,
                    "priority": 3
                })

        result["precedence_chain"] = sorted(precedence_chain, key=lambda x: x["priority"], reverse=True)
        if precedence_chain:
            result["effective_source"] = precedence_chain[-1]["source"] if precedence_chain else None

    return result


def list_config_keys(
    config_name: str = "pipeline_config",
    prefix: str = "",
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    List configuration keys with optional prefix filter.

    Args:
        config_name: Config file name
        prefix: Filter by key prefix (e.g., 'pipeline.determinism')
        max_depth: Maximum nesting depth to traverse

    Returns:
        Dict with keys list, structure preview, and count
    """
    from CONFIG.config_loader import get_config_path, CONFIG_DIR

    config_path = get_config_path(config_name)
    if not config_path.exists():
        return {
            "config_name": config_name,
            "keys": [],
            "count": 0,
            "error": f"Config file not found: {config_path.relative_to(CONFIG_DIR)}"
        }

    config = _load_config_yaml(str(config_path))

    # If prefix specified, navigate to that section first
    if prefix:
        section = _get_nested_value(config, prefix)
        if section is None:
            return {
                "config_name": config_name,
                "prefix": prefix,
                "keys": [],
                "count": 0,
                "error": f"Prefix '{prefix}' not found in config"
            }
        if isinstance(section, dict):
            all_keys = _extract_keys_recursive(section, prefix, max_depth)
        else:
            return {
                "config_name": config_name,
                "prefix": prefix,
                "value": section,
                "count": 1,
                "note": "Prefix points to a leaf value, not a section"
            }
    else:
        all_keys = _extract_keys_recursive(config, "", max_depth)

    return {
        "config_name": config_name,
        "prefix": prefix or "(root)",
        "keys": all_keys,
        "count": len(all_keys),
        "max_depth": max_depth
    }


def load_experiment_config(
    experiment_name: str,
    show_effective: bool = False
) -> Dict[str, Any]:
    """
    Load experiment configuration by name.

    Args:
        experiment_name: Experiment name (without .yaml extension)
        show_effective: Whether to show effective (merged) config

    Returns:
        Dict with experiment config and optionally effective config
    """
    from CONFIG.config_loader import (
        load_experiment_config as _load_experiment_config,
        load_training_config,
        CONFIG_DIR
    )

    try:
        config = _load_experiment_config(experiment_name)
    except FileNotFoundError:
        # List available experiments
        experiments_dir = CONFIG_DIR / "experiments"
        available = []
        if experiments_dir.exists():
            available = sorted([f.stem for f in experiments_dir.glob("*.yaml")])

        return {
            "experiment_name": experiment_name,
            "error": f"Experiment config not found: {experiment_name}",
            "available_experiments": available
        }

    result = {
        "experiment_name": experiment_name,
        "config": config,
        "path": f"experiments/{experiment_name}.yaml"
    }

    if show_effective:
        # Merge with intelligent training config
        base_config = load_training_config("intelligent_training_config")

        # Deep merge (experiment overrides base)
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        effective = deep_merge(base_config, config)
        result["effective_config"] = effective

    return result


def show_config_precedence(path: str) -> Dict[str, Any]:
    """
    Show full precedence chain for a config path.

    Args:
        path: Dot-notation config path

    Returns:
        Dict with path, effective value, source, and full precedence chain
    """
    # Delegate to get_config_value with show_precedence=True
    return get_config_value(path, show_precedence=True)


def validate_config_structure(
    config_name: str,
    expected_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate configuration structure and check for required keys.

    Args:
        config_name: Config file name
        expected_keys: Optional list of required keys to check

    Returns:
        Dict with validation status, missing keys, and warnings
    """
    from CONFIG.config_loader import get_config_path, CONFIG_DIR

    config_path = get_config_path(config_name)

    result = {
        "config_name": config_name,
        "valid": True,
        "missing_keys": [],
        "warnings": []
    }

    if not config_path.exists():
        result["valid"] = False
        result["error"] = f"Config file not found: {config_path.relative_to(CONFIG_DIR)}"
        return result

    config = _load_config_yaml(str(config_path))

    if not config:
        result["valid"] = False
        result["error"] = "Config file is empty or invalid YAML"
        return result

    # Check expected keys if provided
    if expected_keys:
        for key in expected_keys:
            value = _get_nested_value(config, key)
            if value is None:
                result["missing_keys"].append(key)

    # Common required keys for specific config types
    common_required = {
        "pipeline_config": [
            "pipeline.determinism.base_seed",
            "pipeline.determinism.strict_mode"
        ],
        "intelligent_training_config": [
            "targets",
            "model_families"
        ]
    }

    if config_name in common_required and not expected_keys:
        for key in common_required[config_name]:
            value = _get_nested_value(config, key)
            if value is None:
                result["warnings"].append(f"Commonly expected key '{key}' not found")

    if result["missing_keys"]:
        result["valid"] = False

    return result


def list_available_configs() -> Dict[str, Any]:
    """
    List all available configuration files.

    Returns:
        Dict with model_configs and training_configs lists
    """
    from CONFIG.config_loader import list_available_configs as _list_available_configs

    available = _list_available_configs()

    return {
        "model_configs": available.get("model_configs", []),
        "training_configs": available.get("training_configs", []),
        "total_count": len(available.get("model_configs", [])) + len(available.get("training_configs", []))
    }
