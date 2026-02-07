# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Determinism Serialization Helpers

Canonical helpers for deterministic JSON/YAML serialization.
These functions ensure consistent output formatting across runs, preventing
non-determinism in financial outputs (target rankings, feature selection, routing decisions).

CRITICAL: Use these helpers for all Tier A JSON/YAML outputs.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Optional
from enum import Enum

# Import deterministic ordering helpers
from TRAINING.common.utils.determinism_ordering import sorted_items


def canonical_json(obj: Any, *, indent: Optional[int] = None) -> str:
    """
    Convert object to canonical JSON string (deterministic).
    
    Uses _sanitize_for_artifact() to normalize keys and handle special types.
    
    Args:
        obj: Object to serialize
        indent: Optional indentation (None for compact, int for pretty)
    
    Returns:
        Canonical JSON string (sorted keys, consistent formatting)
    """
    # Sanitize first (normalize keys to strings, handle numpy/pandas/etc)
    sanitized = _sanitize_for_artifact(obj)
    
    return json.dumps(
        sanitized,
        sort_keys=True,
        separators=(',', ':') if indent is None else (',', ': '),
        ensure_ascii=False,
        indent=indent
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """
    Convert object to canonical JSON bytes (deterministic).
    
    Args:
        obj: Object to serialize
    
    Returns:
        Canonical JSON bytes (sorted keys, consistent formatting)
    """
    return canonical_json(obj).encode('utf-8')


def write_canonical_json(path: Path, obj: Any, *, indent: Optional[int] = None) -> None:
    """
    Write object to JSON file with canonical formatting.
    
    Args:
        path: Output file path
        obj: Object to serialize
        indent: Optional indentation
    """
    path.write_text(canonical_json(obj, indent=indent), encoding='utf-8')


def _sanitize_for_artifact(obj: Any) -> Any:
    """
    Recursively sanitize object for canonical JSON/YAML serialization.
    
    CRITICAL: Normalizes dict keys to strings to prevent TypeError on mixed-type keys.
    This is the SST helper for canonical serialization (separate from file_utils.sanitize_for_serialization).
    
    Args:
        obj: Object to sanitize (can be dict, list, tuple, Timestamp, Enum, Path, numpy types, sets, etc.)
    
    Returns:
        Sanitized object with:
        - Dict keys normalized to strings
        - pandas Timestamp → ISO strings
        - Enum → string values
        - Path → string
        - numpy types → Python native types
        - sets → sorted lists
        - All dict iterations use sorted_items() for determinism
    """
    import pandas as pd
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    if obj is None:
        return None
    
    # Handle pandas Timestamp
    if has_numpy and isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # Handle Enum
    if isinstance(obj, Enum):
        return obj.value
    
    # Handle Path
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle numpy types
    if has_numpy:
        obj_module = getattr(obj.__class__, '__module__', '')
        if obj_module.startswith('numpy'):
            # numpy array
            if hasattr(obj, 'ndim') and obj.ndim > 0:
                if hasattr(obj, 'tolist'):
                    return _sanitize_for_artifact(obj.tolist())
            # numpy scalar
            if hasattr(obj, 'item'):
                return _sanitize_for_artifact(obj.item())
    
    # Handle dict - CRITICAL: normalize keys to strings + use sorted_items()
    if isinstance(obj, dict):
        out = {}
        # DETERMINISM: Use sorted_items() for deterministic iteration
        for k, v in sorted_items(obj):
            # CRITICAL: Normalize key to string (prevents TypeError on mixed-type keys)
            sanitized_key = str(k)
            sanitized_value = _sanitize_for_artifact(v)
            out[sanitized_key] = sanitized_value
        return out
    
    # Handle list/tuple
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_artifact(v) for v in obj]
    
    # Handle set - convert to sorted list for determinism
    if isinstance(obj, (set, frozenset)):
        return sorted([_sanitize_for_artifact(v) for v in obj])
    
    # Handle dataclasses
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return _sanitize_for_artifact(asdict(obj))
    
    # Handle Decimal
    if type(obj).__name__ == 'Decimal':
        return str(obj)
    
    # Primitive types (int, float, str, bool) pass through
    return obj


class _NoAliasSafeDumper(yaml.SafeDumper):
    """Disables anchors/aliases for deterministic YAML output."""
    def ignore_aliases(self, data: Any) -> bool:
        return True


def canonical_yaml(obj: Any) -> bytes:
    """
    Canonical YAML for human-facing artifacts.
    
    NOT for hashing/signatures (use canonical_json() instead).
    
    Operates on sanitized objects (calls _sanitize_for_artifact() internally).
    
    Args:
        obj: Object to serialize
    
    Returns:
        Canonical YAML bytes (deterministic, anchors disabled, explicit settings)
    """
    # Sanitize first (normalize keys to strings, handle numpy/pandas/etc)
    sanitized = _sanitize_for_artifact(obj)
    
    text = yaml.dump(
        sanitized,
        Dumper=_NoAliasSafeDumper,
        default_flow_style=False,
        sort_keys=True,          # explicit (never rely on defaults)
        allow_unicode=True,      # consistent encoding
        width=10_000,            # avoid line-wrap drift
        indent=2,                # stable indent (readability + avoids emitter variation)
        line_break="\n",         # avoid platform weirdness
    )
    # Force stable newline semantics
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def write_canonical_yaml(path: Path, obj: Any) -> None:
    """
    Write object to YAML file with canonical formatting.
    
    Args:
        path: Output file path
        obj: Object to serialize
    """
    path.write_bytes(canonical_yaml(obj))
