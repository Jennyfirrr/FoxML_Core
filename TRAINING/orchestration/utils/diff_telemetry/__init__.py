# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Diff Telemetry Module

Modular components for diff telemetry system.

This package contains:
- types: DataclassesKim and enums for telemetry
- run_hash: Run hash computation functions
"""

from .types import (
    ChangeSeverity,
    ComparabilityStatus,
    ResolvedRunContext,
    ComparisonGroup,
    NormalizedSnapshot,
    DiffResult,
    BaselineState,
    compute_config_signature,
)

# Import run hash functions from submodule
from .run_hash import (
    compute_full_run_hash,
    compute_run_hash_with_changes,
    save_run_hash,
    _can_runs_be_compared,
    _normalize_run_id_for_comparison,
    _extract_deterministic_fields,
    _load_manifest_comparability_flags,
)

# Import mixins
from .diff_engine import DiffEngineMixin
from .fingerprint_mixin import FingerprintMixin
from .comparison_group_mixin import ComparisonGroupMixin
from .normalization_mixin import NormalizationMixin
from .digest_mixin import DigestMixin
from .context_builder_mixin import ContextBuilderMixin

# Import from parent file (DiffTelemetry class still in main file)
import importlib.util
from pathlib import Path
_parent_file = Path(__file__).parent.parent / "diff_telemetry.py"
if _parent_file.exists():
    spec = importlib.util.spec_from_file_location("diff_telemetry_main", _parent_file)
    diff_telemetry_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff_telemetry_main)

    DiffTelemetry = diff_telemetry_main.DiffTelemetry
    FINGERPRINT_SCHEMA_VERSION = getattr(diff_telemetry_main, 'FINGERPRINT_SCHEMA_VERSION', "1.0")
    # Export utility function (still in main file)
    _sanitize_for_json = getattr(diff_telemetry_main, '_sanitize_for_json', None)
else:
    raise ImportError(f"Could not find diff_telemetry.py at {_parent_file}")

__all__ = [
    # Types (from types.py)
    'ChangeSeverity',
    'ComparabilityStatus',
    'ResolvedRunContext',
    'ComparisonGroup',
    'NormalizedSnapshot',
    'DiffResult',
    'BaselineState',
    'compute_config_signature',
    # Class still in main file
    'DiffTelemetry',
    'FINGERPRINT_SCHEMA_VERSION',
    # Mixins (from submodules)
    'DiffEngineMixin',
    'FingerprintMixin',
    'ComparisonGroupMixin',
    'NormalizationMixin',
    'DigestMixin',
    'ContextBuilderMixin',
    # Run hash functions (from run_hash.py)
    'compute_full_run_hash',
    'compute_run_hash_with_changes',
    'save_run_hash',
    '_can_runs_be_compared',
    '_normalize_run_id_for_comparison',
    '_extract_deterministic_fields',
    '_load_manifest_comparability_flags',
    # Utility function (still in main file)
    '_sanitize_for_json',
]

