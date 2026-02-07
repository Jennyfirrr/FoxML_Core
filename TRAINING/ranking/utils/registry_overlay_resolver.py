# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry Overlay Resolver for Feature Selection

Canonical SST function for resolving registry overlay directory with safety-first precedence.
Ensures Feature Selection consumes registry patches created during Target Ranking.
"""

from pathlib import Path
from typing import Optional, Literal, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegistryOverlayResolution:
    """Result of registry overlay resolution (SST-safe return type)."""
    overlay_dir: Optional[Path]  # Directory to pass to filter_features_for_target()
    overlay_kind: Literal["patch", "config", "none"]  # Which source was used
    patch_file: Optional[Path]  # Actual patch file path (if patch kind)
    overlay_signature: Optional[str]  # Hash of effective policy (for identity/metadata)


def resolve_registry_overlay_dir_for_feature_selection(
    *,
    run_output_root: Path,
    experiment_config: Optional[Any],
    target_column: str,
    current_bar_minutes: Optional[float] = None,
) -> RegistryOverlayResolution:
    """
    Returns the registry overlay directory and metadata for Feature Selection.
    
    SAFETY-FIRST PRECEDENCE (patches win over config):
      1) run_output_root / "registry_patches" (if patch exists for target) - PREFERRED
      2) experiment_config.registry_overlay_dir (if provided) - FALLBACK
      3) None (no patches)
    
    Args:
        run_output_root: Run root directory (use run_root() helper to resolve)
        experiment_config: Optional experiment config
        target_column: Target column name (for patch file lookup)
        current_bar_minutes: Optional current bar interval (for signature computation)
    
    Returns:
        RegistryOverlayResolution with overlay_dir, kind, patch_file, and signature
    
    DETERMINISM:
    - Uses find_patch_file() for deterministic patch lookup (not glob)
    - Computes signature using compute_registry_signature() (canonical)
    - Returns stable kind enum (not free-form string)
    """
    from TRAINING.common.registry_patch_naming import find_patch_file
    from TRAINING.common.utils.fingerprinting import compute_registry_signature
    
    # Precedence 1: Target ranking patches (SAFETY-FIRST - patches detected actual leakage)
    patch_dir = run_output_root / "registry_patches"
    patch_file = None
    if patch_dir.exists():
        patch_file = find_patch_file(patch_dir, target_column)
        if patch_file and patch_file.exists():
            # Compute signature for this patch
            overlay_signature = compute_registry_signature(
                registry_overlay_dir=patch_dir,
                persistent_override_dir=None,  # Not used in feature selection
                persistent_unblock_dir=None,  # Not used in feature selection
                target_column=target_column,
                current_bar_minutes=current_bar_minutes
            )
            logger.debug(
                f"Resolved registry overlay for {target_column}: "
                f"kind=patch, file={patch_file.name}, signature={overlay_signature[:16] if overlay_signature else None}..."
            )
            return RegistryOverlayResolution(
                overlay_dir=patch_dir,
                overlay_kind="patch",
                patch_file=patch_file,
                overlay_signature=overlay_signature
            )
    
    # Precedence 2: Explicit config override (fallback - advanced user override)
    if experiment_config and hasattr(experiment_config, 'registry_overlay_dir'):
        config_overlay = experiment_config.registry_overlay_dir
        if config_overlay:
            config_path = Path(config_overlay)
            if config_path.exists():
                # Check if it's a directory (expected) or file (error)
                if config_path.is_dir():
                    # Try to find patch file in config overlay
                    config_patch_file = find_patch_file(config_path, target_column)
                    overlay_signature = compute_registry_signature(
                        registry_overlay_dir=config_path,
                        persistent_override_dir=None,
                        persistent_unblock_dir=None,
                        target_column=target_column,
                        current_bar_minutes=current_bar_minutes
                    ) if config_patch_file else None
                    logger.debug(
                        f"Resolved registry overlay for {target_column}: "
                        f"kind=config, dir={config_path}, signature={overlay_signature[:16] if overlay_signature else None}..."
                    )
                    return RegistryOverlayResolution(
                        overlay_dir=config_path,
                        overlay_kind="config",
                        patch_file=config_patch_file,
                        overlay_signature=overlay_signature
                    )
                else:
                    logger.warning(f"experiment_config.registry_overlay_dir is a file, not directory: {config_path}")
    
    # Precedence 3: None (no patches)
    logger.debug(f"Resolved registry overlay for {target_column}: kind=none (no patches found)")
    return RegistryOverlayResolution(
        overlay_dir=None,
        overlay_kind="none",
        patch_file=None,
        overlay_signature=None
    )
