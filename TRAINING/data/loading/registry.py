# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Registry for data loaders.

Provides registration and retrieval of data loader implementations.
Supports both programmatic registration and config-based loading.

Example:
    ```python
    from TRAINING.data.loading.registry import get_loader, register_loader

    # Get default loader
    loader = get_loader()

    # Get specific loader
    csv_loader = get_loader("csv")

    # Register custom loader
    register_loader("my_loader", MyCustomLoader)
    ```
"""

import importlib
import logging
import threading
from typing import Dict, Optional, Type

from CONFIG.config_loader import get_cfg

from .interface import DataLoader

logger = logging.getLogger(__name__)

# Thread-safe registry using RLock (per CLAUDE.md pattern)
_registry_lock = threading.RLock()
_LOADERS: Dict[str, Type[DataLoader]] = {}
_INSTANCES: Dict[str, DataLoader] = {}


def register_loader(name: str, loader_cls: Type[DataLoader]) -> None:
    """Register a data loader class.

    Args:
        name: Unique name for the loader (e.g., "parquet", "csv")
        loader_cls: DataLoader subclass

    Raises:
        TypeError: If loader_cls is not a DataLoader subclass

    Example:
        ```python
        from TRAINING.data.loading.registry import register_loader
        from TRAINING.data.loading.interface import DataLoader

        class MyLoader(DataLoader):
            def load(self, source, symbols, interval="5m", **kwargs):
                # Custom loading logic
                pass

            def validate_schema(self, df, schema):
                return True

            def discover_symbols(self, source, interval=None):
                return []

        register_loader("my_loader", MyLoader)
        ```
    """
    if not issubclass(loader_cls, DataLoader):
        raise TypeError(f"{loader_cls} must be a subclass of DataLoader")

    with _registry_lock:
        _LOADERS[name] = loader_cls
        logger.debug(f"Registered data loader: {name}")


def get_loader(
    name: Optional[str] = None, options: Optional[Dict] = None
) -> DataLoader:
    """Get a data loader instance by name.

    Args:
        name: Loader name. If None, uses default from config.
        options: Optional override options for the loader.

    Returns:
        DataLoader instance

    Raises:
        ValueError: If loader not found

    Example:
        ```python
        # Get default loader
        loader = get_loader()

        # Get CSV loader with custom delimiter
        loader = get_loader("csv", options={"delimiter": ";"})
        ```
    """
    if name is None:
        name = get_cfg("data_loading.default_loader", default="parquet")

    # Create cache key from name and options
    options_key = tuple(sorted((options or {}).items()))
    cache_key = f"{name}:{hash(options_key)}"

    with _registry_lock:
        # Check cache first
        if cache_key in _INSTANCES:
            return _INSTANCES[cache_key]

        # Get loader class
        if name not in _LOADERS:
            # Try loading from config
            loader_config = get_cfg(f"data_loading.loaders.{name}", default=None)
            if loader_config and "class" in loader_config:
                _load_loader_from_config(name, loader_config)
            else:
                raise ValueError(
                    f"Unknown data loader: {name}. "
                    f"Available: {list(_LOADERS.keys())}"
                )

        loader_cls = _LOADERS[name]

        # Merge options: config defaults + runtime overrides
        config_options = get_cfg(f"data_loading.loaders.{name}.options", default={})
        merged_options = {**config_options, **(options or {})}

        # Create instance
        instance = loader_cls(**merged_options)
        _INSTANCES[cache_key] = instance

        return instance


def _load_loader_from_config(name: str, config: Dict) -> None:
    """Dynamically load a loader class from config.

    Args:
        name: Loader name
        config: Config dict with 'class' key

    Raises:
        ImportError: If loader class cannot be loaded
    """
    class_path = config["class"]
    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        loader_cls = getattr(module, class_name)
        register_loader(name, loader_cls)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load loader {name} from {class_path}: {e}")


def list_loaders() -> Dict[str, Type[DataLoader]]:
    """List all registered loaders.

    Returns:
        Dictionary mapping loader names to classes
    """
    with _registry_lock:
        return dict(_LOADERS)


def clear_cache() -> None:
    """Clear loader instance cache.

    Useful for testing or when config changes.
    """
    with _registry_lock:
        _INSTANCES.clear()
        logger.debug("Loader instance cache cleared")


def unregister_loader(name: str) -> bool:
    """Unregister a loader by name.

    Args:
        name: Loader name to unregister

    Returns:
        True if loader was registered and removed, False otherwise
    """
    with _registry_lock:
        if name in _LOADERS:
            del _LOADERS[name]
            # Also clear any cached instances for this loader
            keys_to_remove = [k for k in _INSTANCES if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del _INSTANCES[key]
            logger.debug(f"Unregistered data loader: {name}")
            return True
        return False
