# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry Provenance for Worker Reconstruction

Single Source of Truth for serializing registry state across multiprocessing boundaries.
Ensures workers reconstruct the same registry instance as the parent process.
"""

import logging
import os
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RegistryProvenance:
    """Complete registry state for worker reconstruction (SST contract)."""
    registry_path_abs: Path  # Absolute, validated path
    overlay_paths: Dict[str, Optional[Path]]  # {"global": Path, "per_target": Path} (paths only, no ints)
    overlay_interval_minutes: Optional[float]  # Interval used for overlay selection (separate from paths)
    current_bar_minutes: Optional[float]  # Interval for horizon logic
    allow_overwrite: bool  # Merge policy toggle
    # Validation flags (computed in parent, but worker should revalidate)
    registry_path_exists: bool  # Parent observed (worker should still verify)
    registry_path_readable: bool  # Parent observed (worker should still verify)
    # Identity signature (for verification that parent and worker use same inputs)
    registry_identity_hash: Optional[str]  # Hash of (registry + overlays + allow_overwrite + interval)


def compute_registry_identity_hash(
    registry_path: Path,
    overlay_paths: Dict[str, Optional[Path]],
    overlay_interval_minutes: Optional[float],
    allow_overwrite: bool,
    current_bar_minutes: Optional[float]
) -> str:
    """
    Compute identity hash of registry configuration for parent/worker verification.
    
    Reuses canonical_json() + sha256_full() from config_hashing.py (SST helpers).
    
    Args:
        registry_path: Path to registry file
        overlay_paths: Dict of overlay file paths
        overlay_interval_minutes: Interval used for overlay selection
        allow_overwrite: Merge policy toggle
        current_bar_minutes: Current bar interval
    
    Returns:
        SHA256 hash of registry identity payload
    """
    from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
    
    # Build payload (deterministic order)
    payload = {
        "registry_path": str(registry_path),
        "overlay_global": str(overlay_paths.get("global")) if overlay_paths.get("global") else None,
        "overlay_per_target": str(overlay_paths.get("per_target")) if overlay_paths.get("per_target") else None,
        "overlay_interval_minutes": overlay_interval_minutes,
        "allow_overwrite": allow_overwrite,
        "current_bar_minutes": current_bar_minutes
    }
    
    # Hash file contents if files exist (for true identity verification)
    try:
        if registry_path.exists():
            with open(registry_path, 'rb') as f:
                registry_content = f.read()
            payload["registry_content_hash"] = sha256_full(registry_content.decode('utf-8', errors='ignore'))
    except Exception as e:
        logger.debug(f"Could not hash registry file content: {e}")
    
    for overlay_key, overlay_path in overlay_paths.items():
        if overlay_path and overlay_path.exists():
            try:
                with open(overlay_path, 'rb') as f:
                    overlay_content = f.read()
                payload[f"overlay_{overlay_key}_content_hash"] = sha256_full(overlay_content.decode('utf-8', errors='ignore'))
            except Exception as e:
                logger.debug(f"Could not hash overlay file content ({overlay_key}): {e}")
    
    return sha256_full(canonical_json(payload))


def resolve_registry_provenance(
    registry: Any,
    explicit_interval: Optional[Any],
    experiment_config: Optional[Any],
    strict: bool = False
) -> RegistryProvenance:
    """
    Resolve complete registry provenance from registry instance.
    
    Reuses resolve_registry_path_for_interval() from feature_registry.py (SST helper).
    
    Args:
        registry: FeatureRegistry instance
        explicit_interval: Explicit interval from config (e.g., "5m")
        experiment_config: Optional experiment config
        strict: If True, raise on validation failure
    
    Returns:
        RegistryProvenance with all state needed for worker reconstruction
    
    Raises:
        RegistryLoadError: If strict=True and validation fails
    """
    from TRAINING.common.feature_registry import resolve_registry_path_for_interval
    from TRAINING.common.exceptions import RegistryLoadError
    
    # Get registry path (use stable API)
    if hasattr(registry, 'config_path') and registry.config_path:
        registry_path = Path(registry.config_path)
    else:
        # Fallback: resolve using SST helper
        current_bar_minutes = None
        if explicit_interval:
            from TRAINING.ranking.utils.data_interval import normalize_interval
            current_bar_minutes = normalize_interval(explicit_interval)
        elif experiment_config:
            try:
                from CONFIG.config_loader import get_cfg
                raw_interval = get_cfg('data.bar_interval', default=None)
                if raw_interval is not None:
                    try:
                        current_bar_minutes = float(raw_interval)
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        
        registry_path = resolve_registry_path_for_interval(
            base_path=None,
            interval_minutes=current_bar_minutes
        )
    
    # Resolve to absolute path (required for worker processes)
    if not registry_path.is_absolute():
        registry_path = registry_path.resolve()
    
    # Get overlay paths (use stable API)
    overlay_paths = {}
    overlay_interval_minutes = None
    if hasattr(registry, 'get_selected_overlay_paths'):
        overlay_paths = registry.get_selected_overlay_paths()
    if hasattr(registry, 'get_overlay_interval_minutes'):
        overlay_interval_minutes = registry.get_overlay_interval_minutes()
    
    # Get current_bar_minutes
    current_bar_minutes = None
    if explicit_interval:
        from TRAINING.ranking.utils.data_interval import normalize_interval
        current_bar_minutes = normalize_interval(explicit_interval)
    elif experiment_config:
        try:
            from CONFIG.config_loader import get_cfg
            raw_interval = get_cfg('data.bar_interval', default=None)
            if raw_interval is not None:
                try:
                    current_bar_minutes = float(raw_interval)
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
    elif hasattr(registry, 'current_bar_minutes'):
        current_bar_minutes = registry.current_bar_minutes
    
    # Get allow_overwrite
    allow_overwrite = getattr(registry, 'allow_overwrite', False)
    
    # Resolve overlay paths to absolute
    overlay_paths_abs = {}
    for key, path in overlay_paths.items():
        if path:
            if not path.is_absolute():
                overlay_paths_abs[key] = path.resolve()
            else:
                overlay_paths_abs[key] = path
        else:
            overlay_paths_abs[key] = None
    
    # Validate in parent
    registry_path_exists = registry_path.exists()
    registry_path_readable = False
    if registry_path_exists:
        try:
            with open(registry_path, 'r'):
                registry_path_readable = True
        except Exception:
            pass
    
    if strict:
        if not registry_path_exists:
            raise RegistryLoadError(
                message=f"Registry file does not exist: {registry_path}",
                registry_path=str(registry_path),
                stage="TARGET_RANKING",
                error_code="REGISTRY_FILE_MISSING"
            )
        if not registry_path_readable:
            raise RegistryLoadError(
                message=f"Registry file not readable: {registry_path}",
                registry_path=str(registry_path),
                stage="TARGET_RANKING",
                error_code="REGISTRY_FILE_UNREADABLE"
            )
    
    # Compute identity hash
    registry_identity_hash = compute_registry_identity_hash(
        registry_path=registry_path,
        overlay_paths=overlay_paths_abs,
        overlay_interval_minutes=overlay_interval_minutes,
        allow_overwrite=allow_overwrite,
        current_bar_minutes=current_bar_minutes
    )
    
    return RegistryProvenance(
        registry_path_abs=registry_path,
        overlay_paths=overlay_paths_abs,
        overlay_interval_minutes=overlay_interval_minutes,
        current_bar_minutes=current_bar_minutes,
        allow_overwrite=allow_overwrite,
        registry_path_exists=registry_path_exists,
        registry_path_readable=registry_path_readable,
        registry_identity_hash=registry_identity_hash
    )


def load_registry_from_provenance(
    prov: RegistryProvenance,
    fail_closed: bool = True
) -> Any:
    """
    Load registry from provenance in worker process.
    
    Reuses get_registry() from feature_registry.py (with strict parameter).
    Revalidates file existence/readability (don't trust parent flags).
    
    Args:
        prov: RegistryProvenance from parent
        fail_closed: If True, raise on failure. If False, return degraded registry (never None)
    
    Returns:
        FeatureRegistry instance (or degraded wrapper in best-effort mode)
    
    Raises:
        RegistryLoadError: If fail_closed=True and load fails
    """
    from TRAINING.common.feature_registry import get_registry
    from TRAINING.common.exceptions import RegistryLoadError
    
    # Revalidate in worker (don't trust parent flags - spawn may change env/CWD)
    registry_path_exists = prov.registry_path_abs.exists()
    registry_path_readable = False
    if registry_path_exists:
        try:
            with open(prov.registry_path_abs, 'r'):
                registry_path_readable = True
        except Exception as e:
            if fail_closed:
                raise RegistryLoadError(
                    message=f"Registry file not readable in worker: {prov.registry_path_abs}",
                    registry_path=str(prov.registry_path_abs),
                    stage="TARGET_RANKING",
                    error_code="REGISTRY_FILE_UNREADABLE"
                ) from e
    
    if not registry_path_exists:
        if fail_closed:
            raise RegistryLoadError(
                message=f"Registry file does not exist in worker: {prov.registry_path_abs}",
                registry_path=str(prov.registry_path_abs),
                stage="TARGET_RANKING",
                error_code="REGISTRY_FILE_MISSING"
            )
        else:
            # Best-effort: return degraded registry (membership-only mode)
            logger.warning(
                f"Registry file does not exist in worker: {prov.registry_path_abs}. "
                f"Using degraded membership-only registry."
            )
            # TODO: Create degraded registry wrapper that forces coverage_mode="membership_only"
            # For now, still try to load (may return empty registry)
            pass
    
    # Note: Auto overlays (from CONFIG/data/overrides/) are loaded automatically by FeatureRegistry._load_auto_overlay()
    # The overlay_paths in provenance are for tracking/verification only, not for loading
    # registry_overlay_dir parameter is specifically for run patches (per-target patches from target ranking),
    # which are in a different location (run_root / "registry_patches")
    # For target ranking workers, we don't have run patches at registry loading time (registry is loaded once per worker)
    # So we don't pass registry_overlay_dir here - auto overlays will be loaded automatically
    
    # Load registry (with strict parameter)
    try:
        registry = get_registry(
            config_path=prov.registry_path_abs,
            registry_overlay_dir=None,  # Auto overlays load automatically, run patches loaded per-target later
            current_bar_minutes=prov.current_bar_minutes,
            strict=fail_closed
        )
        
        # Verify identity hash matches (log mismatch, but don't fail)
        if prov.registry_identity_hash:
            worker_identity = compute_registry_identity_hash(
                registry_path=prov.registry_path_abs,
                overlay_paths=prov.overlay_paths,
                overlay_interval_minutes=prov.overlay_interval_minutes,
                allow_overwrite=prov.allow_overwrite,
                current_bar_minutes=prov.current_bar_minutes
            )
            if worker_identity != prov.registry_identity_hash:
                logger.warning(
                    "Registry identity hash mismatch: parent=%s worker=%s. "
                    "Worker may be using different registry inputs. "
                    "cwd=%s pid=%s start_method=%s",
                    prov.registry_identity_hash[:16] if prov.registry_identity_hash else None,
                    worker_identity[:16] if worker_identity else None,
                    Path.cwd(),
                    os.getpid(),
                    multiprocessing.get_start_method()
                )
        
        # Defensive: validate registry is not None
        if registry is None:
            raise RegistryLoadError(
                message="get_registry() returned None (this should be impossible)",
                registry_path=str(prov.registry_path_abs),
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            )
        
        return registry
        
    except RegistryLoadError:
        # Re-raise RegistryLoadError as-is
        raise
    except Exception as e:
        # Wrap other exceptions
        if fail_closed:
            raise RegistryLoadError(
                message=f"Registry reconstruction failed: {e}",
                registry_path=str(prov.registry_path_abs),
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            ) from e
        else:
            # Best-effort: log and return degraded registry
            logger.warning(
                f"Registry reconstruction failed (best-effort mode): {e}. "
                f"Using degraded membership-only registry. "
                f"registry_path=%s exists=%s readable=%s overlay_paths=%s "
                f"cwd=%s pid=%s start_method=%s",
                prov.registry_path_abs,
                registry_path_exists,
                registry_path_readable,
                prov.overlay_paths,
                Path.cwd(),
                os.getpid(),
                multiprocessing.get_start_method()
            )
            # TODO: Create degraded registry wrapper
            # For now, re-raise (best-effort mode not fully implemented)
            raise RegistryLoadError(
                message=f"Registry reconstruction failed (best-effort mode should return degraded registry): {e}",
                registry_path=str(prov.registry_path_abs),
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            ) from e
