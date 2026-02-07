# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target-First Path Resolution Utilities

Helper functions for organizing run outputs using target-first structure.
Target is the stable join key - all per-target artifacts live together under targets/<target>/.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Set, Union
import logging

# SST: Import View and Stage enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

# SST: Import WriteScope for scope-aware path construction
try:
    from TRAINING.orchestration.utils.scope_resolution import WriteScope
    _WRITE_SCOPE_AVAILABLE = True
except ImportError:
    _WRITE_SCOPE_AVAILABLE = False
    WriteScope = None

logger = logging.getLogger(__name__)

# Module-level warn-once set for view/view mismatches
_view_mismatch_warned: Set[tuple] = set()


def _validate_view_for_path_construction(
    view: Union[str, View],
    symbol: Optional[str],
    universe_sig: Optional[str],
    base_output_dir: Optional[Path] = None
) -> Union[str, View]:
    """
    Validate view matches symbol count. Returns corrected view.
    
    If view is SYMBOL_SPECIFIC but we have multiple symbols, force CROSS_SECTIONAL.
    This is a defensive validation to prevent multi-symbol runs from routing to
    SYMBOL_SPECIFIC folders.
    
    Args:
        view: View enum or string
        symbol: Optional symbol name
        universe_sig: Optional universe signature
        base_output_dir: Optional base output directory for cache lookup
    
    Returns:
        Validated view (may be corrected from SYMBOL_SPECIFIC to CROSS_SECTIONAL)
    """
    view_enum = View.from_string(view) if isinstance(view, str) else view
    
    if view_enum != View.SYMBOL_SPECIFIC:
        return view  # No validation needed for CROSS_SECTIONAL
    
    # SYMBOL_SPECIFIC requires n_symbols=1
    # Try to determine symbol count
    
    # Method 1: If symbol is provided, assume single-symbol (valid)
    if symbol:
        return view  # Valid - single symbol provided
    
    # Method 2: Try to load from universe_sig cache
    n_symbols = None
    if universe_sig and base_output_dir:
        try:
            from TRAINING.orchestration.utils.run_context import get_view_for_universe
            entry = get_view_for_universe(base_output_dir, universe_sig)
            if entry:
                n_symbols = entry.get('n_symbols')
        except Exception as e:
            logger.debug(f"Could not load universe view cache for validation: {e}")
    
    # Method 3: If we can't determine, log warning but allow (defensive)
    if n_symbols is None:
        logger.warning(
            f"⚠️  Cannot validate SYMBOL_SPECIFIC view (universe_sig={universe_sig}). "
            f"Assuming valid. If this is multi-symbol, artifacts may route incorrectly."
        )
        return view
    
    # Validate: SYMBOL_SPECIFIC requires n_symbols=1
    if n_symbols > 1:
        logger.warning(
            f"⚠️  Invalid view=SYMBOL_SPECIFIC for multi-symbol run (n_symbols={n_symbols}). "
            f"Forcing view=CROSS_SECTIONAL for path construction."
        )
        return View.CROSS_SECTIONAL
    
    return view  # Valid - single symbol


def normalize_target_name(target: str) -> str:
    """
    Normalize target name for filesystem paths.
    
    This is the SST helper for target name normalization. Replaces '/' and '\\'
    with '_' to prevent path issues. All path construction functions should use
    this helper to ensure consistency.
    
    Args:
        target: Target name (may contain '/' or '\\')
    
    Returns:
        Normalized target name safe for filesystem paths
    
    Examples:
        "fwd_ret/5d" -> "fwd_ret_5d"
        "target\\name" -> "target_name"
        "normal_target" -> "normal_target"
    """
    if not target:
        return "unknown"
    return target.replace('/', '_').replace('\\', '_')


def normalize_run_root(path: Path) -> Path:
    """
    Normalize path to run root by stripping at first 'targets' segment.
    
    Handles all cases:
    - /run → /run
    - /run/targets → /run
    - /run/targets/<target> → /run
    - /run/targets/<target>/anything/deeper → /run
    - /run/targets/targets/<target> → /run (bug case)
    
    Args:
        path: Path that may contain 'targets/' at any depth
    
    Returns:
        Path guaranteed to be run root (stripped at first 'targets' segment)
    """
    from pathlib import Path
    p = Path(path).resolve()
    parts = p.parts
    if "targets" not in parts:
        return p
    idx = parts.index("targets")  # first occurrence
    # idx==0 would be bizarre; just return root/anchor in that case
    return Path(*parts[:idx]) if idx > 0 else Path(p.anchor)


def get_target_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get the target directory for a given target.
    
    Args:
        base_output_dir: Base run output directory (will be normalized)
        target: Target name (e.g., "fwd_ret_5d")
    
    Returns:
        Path to targets/<target>/ directory
    """
    # Defensive: normalize to ensure base_output_dir is run root
    base_output_dir = normalize_run_root(base_output_dir)
    # Assertion hardening: make bug impossible to silently reintroduce
    if base_output_dir.name == "targets":
        raise AssertionError(
            f"normalize_run_root() failed: base_output_dir still ends with 'targets/': {base_output_dir}"
        )
    return base_output_dir / "targets" / target


def get_globals_dir(base_output_dir: Path) -> Path:
    """
    Get the globals directory for run-level summaries.
    
    Args:
        base_output_dir: Base run output directory
    
    Returns:
        Path to globals/ directory
    """
    return base_output_dir / "globals"


def get_target_decision_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get decision directory for a target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/decision/
    """
    return get_target_dir(base_output_dir, target) / "decision"


def get_target_models_dir(base_output_dir: Path, target: str, family: Optional[str] = None) -> Path:
    """
    Get models directory for a target (optionally for a specific family).
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        family: Optional model family name (e.g., "lightgbm")
    
    Returns:
        Path to targets/<target>/models/ or targets/<target>/models/<family>/
    """
    models_dir = get_target_dir(base_output_dir, target) / "models"
    if family:
        return models_dir / family
    return models_dir


def get_target_metrics_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get metrics directory for a target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/metrics/
    """
    return get_target_dir(base_output_dir, target) / "metrics"


def get_target_trends_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get trends directory for a target (within-run trends).
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/trends/
    """
    return get_target_dir(base_output_dir, target) / "trends"


def get_target_reproducibility_dir(
    base_output_dir: Path,
    target: str,
    stage: Optional[Union[str, Stage]] = None,
) -> Path:
    """
    Get reproducibility directory for a target, optionally scoped by stage.
    
    If stage is provided, returns stage-scoped path.
    If stage is None, attempts SST lookup, then falls back to legacy path.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        stage: Optional stage name (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Returns:
        Path to targets/<target>/reproducibility/[stage={stage}/]
    
    Examples:
        With stage: targets/fwd_ret_10m/reproducibility/stage=TARGET_RANKING/
        Without stage (legacy): targets/fwd_ret_10m/reproducibility/
    
    Note:
        Target name is normalized internally (replaces '/' and '\\' with '_')
        to ensure filesystem-safe paths. Callers can pass raw target names.
    """
    # FIX: Normalize target name for consistent path construction (SST)
    target_normalized = normalize_target_name(target)
    base_repro = get_target_dir(base_output_dir, target_normalized) / "reproducibility"
    
    # Normalize stage to enum if provided, then convert to string for path
    stage_enum = Stage.from_string(stage) if isinstance(stage, str) and stage else (stage if isinstance(stage, Stage) else None)
    # Priority: explicit > SST > legacy
    resolved_stage = stage_enum
    if resolved_stage is None:
        try:
            from TRAINING.orchestration.utils.run_context import get_current_stage
            resolved_stage = get_current_stage(base_output_dir)
        except Exception as e:
            logger.debug(f"Could not resolve current stage from run context: {e}")
            # SST not available, use legacy
    
    if resolved_stage:
        # SST: Explicitly convert stage enum to string for path construction (defensive)
        stage_str = resolved_stage.value if isinstance(resolved_stage, Stage) else str(resolved_stage)
        return base_repro / f"stage={stage_str}"
    
    return base_repro  # Legacy fallback


def get_scoped_artifact_dir(
    base_output_dir: Path,
    target: str,
    artifact_type: str,
    # SST: Preferred - accept WriteScope directly
    scope: Optional["WriteScope"] = None,
    # DEPRECATED: Loose args (for backward compat)
    view: Optional[Union[str, View]] = None,
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None,
    stage: Optional[Union[str, Stage]] = None,
    attempt_id: Optional[int] = None,  # NEW: Attempt identifier for per-attempt artifacts (defaults to 0)
) -> Path:
    """
    Get view-scoped artifact directory for a target.
    
    Artifacts are scoped by stage, view, attempt, and optionally by symbol to support:
    - Different feature exclusions per stage/view/attempt/symbol
    - Different feature importance snapshots per stage/view/attempt/symbol
    - Different featureset artifacts per stage/view/attempt/symbol
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        artifact_type: Type of artifact ("feature_exclusions", "feature_importance_snapshots", "featureset_artifacts")
        scope: WriteScope object (preferred, SST-compliant)
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" (deprecated, use scope)
        symbol: Symbol name for SYMBOL_SPECIFIC view (deprecated, use scope)
        universe_sig: Optional universe signature for additional scoping (deprecated, use scope)
        stage: Optional stage name (deprecated, use scope)
        attempt_id: Optional attempt identifier (defaults to 0 for backward compatibility)
    
    Returns:
        Path to targets/<target>/reproducibility/[stage={stage}/]<VIEW>/[symbol=<symbol>/][batch_<sig>/]attempt_{id}/<artifact_type>/
    
    Examples:
        With stage and attempt: targets/fwd_ret_10m/reproducibility/stage=TARGET_RANKING/CROSS_SECTIONAL/batch_abc123/attempt_0/feature_exclusions/
        Without stage (legacy): targets/fwd_ret_10m/reproducibility/CROSS_SECTIONAL/batch_abc123/attempt_0/feature_exclusions/
    """
    # SST: Extract from WriteScope if provided
    if scope is not None:
        if not _WRITE_SCOPE_AVAILABLE:
            raise ValueError("WriteScope not available but scope was passed")
        view = scope.view
        symbol = scope.symbol
        universe_sig = scope.universe_sig
        stage = scope.stage
    elif view is None:
        # Default to CROSS_SECTIONAL if neither scope nor view provided
        view = View.CROSS_SECTIONAL
    
    # CRITICAL: Validate view matches symbol count before path construction
    view = _validate_view_for_path_construction(
        view=view,
        symbol=symbol,
        universe_sig=universe_sig,
        base_output_dir=base_output_dir
    )
    
    # Get reproducibility dir with stage scoping
    repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage)
    
    # Normalize view (handles both enum and string)
    view_str = str(view) if hasattr(view, 'value') else str(view)
    view_upper = view_str.upper() if view_str else View.CROSS_SECTIONAL.value
    if view_upper not in (View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value):
        view_upper = View.CROSS_SECTIONAL.value
    
    # Build path
    artifact_path = repro_dir / view_upper
    
    # Add symbol for SYMBOL_SPECIFIC view
    if view_upper == View.SYMBOL_SPECIFIC.value and symbol:
        artifact_path = artifact_path / f"symbol={symbol}"
    elif view_upper == View.SYMBOL_SPECIFIC.value and not symbol:
        # CRITICAL: Validate symbol is provided for SYMBOL_SPECIFIC view
        logger.warning(
            f"SYMBOL_SPECIFIC view but symbol is None for {artifact_type}. "
            f"Path will be created without symbol component, which may cause routing issues. "
            f"Expected path: {view_upper}/symbol=<symbol>/ but got: {view_upper}/"
        )
        # Continue without symbol (may need to fix caller)
    
    # DETERMINISTIC: Add universe signature if provided (use batch_ prefix for CROSS_SECTIONAL)
    # For SYMBOL_SPECIFIC: No universe= (symbol= is sufficient)
    if universe_sig:
        view_enum = View.from_string(view) if isinstance(view, View) else view
        if view_enum == View.CROSS_SECTIONAL:
            artifact_path = artifact_path / f"batch_{universe_sig[:12]}"  # Use batch_ prefix (deterministic slice)
        # SYMBOL_SPECIFIC: No universe= (symbol= is sufficient, already added above)
    
    # DETERMINISTIC: Add attempt level for per-attempt artifacts (preserves leakage fix history)
    # Default to attempt_0 for backward compatibility
    attempt_id_normalized = attempt_id if attempt_id is not None else 0
    artifact_path = artifact_path / f"attempt_{attempt_id_normalized}"
    
    # Add artifact type
    artifact_path = artifact_path / artifact_type
    
    return artifact_path


def ensure_scoped_artifact_dir(
    base_output_dir: Path,
    target: str,
    artifact_type: str,
    # SST: Preferred - accept WriteScope directly
    scope: Optional["WriteScope"] = None,
    # DEPRECATED: Loose args (for backward compat)
    view: Optional[Union[str, View]] = None,
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None,
    stage: Optional[Union[str, Stage]] = None,
    attempt_id: Optional[int] = None,  # NEW: Attempt identifier for per-attempt artifacts (defaults to 0)
) -> Path:
    """
    Get view-scoped artifact directory and ensure it exists.
    
    Same as get_scoped_artifact_dir but creates the directory if it doesn't exist.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        artifact_type: Type of artifact
        scope: WriteScope object (preferred, SST-compliant)
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" (deprecated, use scope)
        symbol: Symbol name for SYMBOL_SPECIFIC view (deprecated, use scope)
        universe_sig: Optional universe signature for additional scoping (deprecated, use scope)
        stage: Optional stage name (deprecated, use scope)
        attempt_id: Optional attempt identifier (defaults to 0 for backward compatibility)
    
    Returns:
        Path to artifact directory (created if needed)
    """
    artifact_dir = get_scoped_artifact_dir(
        base_output_dir, target, artifact_type, scope=scope, view=view, symbol=symbol, universe_sig=universe_sig, stage=stage, attempt_id=attempt_id
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def get_global_trends_dir(base_output_dir: Path) -> Path:
    """
    Get global trends directory (cross-target within-run analyses).
    
    Args:
        base_output_dir: Base run output directory
    
    Returns:
        Path to globals/trends/
    """
    return get_globals_dir(base_output_dir) / "trends"


def ensure_target_structure(base_output_dir: Path, target: str) -> None:
    """
    Ensure all target directories exist.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    """
    target_dir = get_target_dir(base_output_dir, target)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "decision").mkdir(exist_ok=True)
    (target_dir / "models").mkdir(exist_ok=True)
    (target_dir / "metrics").mkdir(exist_ok=True)
    (target_dir / "trends").mkdir(exist_ok=True)
    (target_dir / "reproducibility").mkdir(exist_ok=True)


def ensure_globals_structure(base_output_dir: Path) -> None:
    """
    Ensure globals directory structure exists.
    
    Args:
        base_output_dir: Base run output directory
    """
    globals_dir = get_globals_dir(base_output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    (globals_dir / "trends").mkdir(exist_ok=True)


def initialize_run_structure(base_output_dir: Path) -> None:
    """
    Initialize the target-first run structure.
    
    Creates:
    - targets/ directory
    - globals/ directory with trends/ subdirectory
    
    Args:
        base_output_dir: Base run output directory
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "targets").mkdir(exist_ok=True)
    ensure_globals_structure(base_output_dir)
    
    # Only create essential directories - no legacy REPRODUCIBILITY structure
    (base_output_dir / "cache").mkdir(exist_ok=True)
    (base_output_dir / "logs").mkdir(exist_ok=True)


def get_metrics_path_from_cohort_dir(cohort_dir: Path, base_output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Map a cohort directory to the corresponding metrics directory.
    
    Maps from:
    - targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/ 
      → targets/<target>/metrics/view=CROSS_SECTIONAL/
    - targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<id>/
      → targets/<target>/metrics/view=SYMBOL_SPECIFIC/symbol=<symbol>/
    
    Args:
        cohort_dir: Cohort directory path (from reproducibility/)
        base_output_dir: Optional base output directory (will be inferred if not provided)
    
    Returns:
        Path to metrics directory, or None if path cannot be resolved
    """
    cohort_dir = Path(cohort_dir)
    
    # Find base_output_dir if not provided
    if base_output_dir is None:
        temp_dir = cohort_dir
        for _ in range(10):
            if temp_dir.name == "targets" and (temp_dir.parent / "targets").exists():
                base_output_dir = temp_dir.parent
                break
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        if base_output_dir is None:
            logger.warning(f"Could not find base_output_dir from cohort_dir: {cohort_dir}")
            return None
    
    # Use parse_reproducibility_path which correctly handles stage= prefixes
    parsed = parse_reproducibility_path(cohort_dir)
    
    target = parsed.get("target")
    stage = parsed.get("stage")
    view = parsed.get("view")
    symbol = parsed.get("symbol")
    
    if not target:
        logger.warning(f"Could not extract target from cohort_dir: {cohort_dir}")
        return None
    
    if not view:
        logger.warning(f"Could not extract view from cohort_dir: {cohort_dir}")
        return None
    
    # Build metrics path with stage (if present) for proper separation
    metrics_dir = get_target_metrics_dir(base_output_dir, target)
    if stage:
        metrics_dir = metrics_dir / f"stage={stage}"
    metrics_dir = metrics_dir / f"view={view}"
    if symbol:
        metrics_dir = metrics_dir / f"symbol={symbol}"
    
    return metrics_dir


def find_cohort_dir_by_id(
    base_output_dir: Path,
    cohort_id: str,
    target: str,
    view: Union[str, View] = View.CROSS_SECTIONAL,
    stage: Optional[str] = None
) -> Optional[Path]:
    """
    Find cohort directory by cohort_id.
    
    Args:
        base_output_dir: Base run output directory
        cohort_id: Cohort ID to find
        target: Target name
        view: View type ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        stage: Optional stage name (e.g., "FEATURE_SELECTION")
    
    Returns:
        Path to cohort directory, or None if not found
    """
    try:
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage)
        # SST: Normalize view to string for path construction (defensive)
        view_str = view.value if isinstance(view, View) else str(view)
        view_dir = target_repro_dir / view_str
        
        # FIX: Add directory existence check before iterating
        if not view_dir.exists() or not view_dir.is_dir():
            return None
        
        # For SYMBOL_SPECIFIC, need to search in symbol= subdirectories
        view_str = str(view) if hasattr(view, 'value') else str(view)
        if view_str == View.SYMBOL_SPECIFIC.value:
            # Search all symbol= directories for the cohort
            # DETERMINISM: Use iterdir_sorted for deterministic iteration order
            from TRAINING.common.utils.determinism_ordering import iterdir_sorted
            symbol_dirs = [
                d for d in iterdir_sorted(view_dir, filter_fn=lambda p: p.is_dir() and p.name.startswith("symbol="))
            ]
            for symbol_dir in symbol_dirs:
                # Search in attempt subdirectories (new structure) and direct (legacy)
                # DETERMINISTIC: Sort attempt dirs by numeric attempt_id (not lexicographic)
                # Parse attempt_id from name for proper numeric ordering (attempt_2 before attempt_10)
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                attempt_dirs_raw = [
                    d for d in iterdir_sorted(symbol_dir, filter_fn=lambda p: p.is_dir() and p.name.startswith("attempt_"))
                ]
                attempt_dirs = sorted(
                    attempt_dirs_raw,
                    key=lambda x: (
                        int(x.name.split("_", 1)[1]) if x.name.split("_", 1)[1].isdigit() else 999,  # Numeric sort
                        x.name  # Fallback to name for non-numeric
                    )
                )
                # Try attempt subdirectories first (new structure)
                for attempt_dir in attempt_dirs:
                    cohort_dir = attempt_dir / f"cohort={cohort_id}"
                    if cohort_dir.exists():
                        return cohort_dir
                # Fallback: legacy structure (direct cohort= under symbol=)
                cohort_dir = symbol_dir / f"cohort={cohort_id}"
                if cohort_dir.exists():
                    return cohort_dir
        else:
            # CROSS_SECTIONAL: search in batch_*/attempt_*/cohort={id}/ or universe=*/attempt_*/cohort={id}/
            # DETERMINISTIC: Use glob_sorted for deterministic iteration with relative paths
            from TRAINING.common.utils.determinism_ordering import glob_sorted
            universe_dirs = glob_sorted(view_dir, 'universe=*', filter_fn=lambda p: p.is_dir())
            batch_dirs = glob_sorted(view_dir, 'batch_*', filter_fn=lambda p: p.is_dir())
            # DETERMINISTIC: Prefer universe= (legacy) for backward compat, then batch_ (new)
            # Concatenation order is deterministic: universe= first, then batch_
            all_dirs = universe_dirs + batch_dirs
            
            if all_dirs:
                # Search within each batch_/universe= directory
                for base_dir in all_dirs:
                    # Search in attempt_* subdirectories within batch_/universe=
                    # DETERMINISTIC: Sort attempt dirs by numeric attempt_id (not lexicographic)
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    attempt_dirs_raw = [
                        d for d in iterdir_sorted(base_dir, filter_fn=lambda p: p.is_dir() and p.name.startswith("attempt_"))
                    ]
                    attempt_dirs = sorted(
                        attempt_dirs_raw,
                        key=lambda x: (
                            int(x.name.split("_", 1)[1]) if x.name.split("_", 1)[1].isdigit() else 999,  # Numeric sort
                            x.name  # Fallback to name for non-numeric
                        )
                    )
                    # Try attempt subdirectories first (new structure)
                    for attempt_dir in attempt_dirs:
                        cohort_dir = attempt_dir / f"cohort={cohort_id}"
                        if cohort_dir.exists():
                            return cohort_dir
                    # Fallback: legacy structure (direct cohort= under batch_/universe=, no attempt_ level)
                    cohort_dir = base_dir / f"cohort={cohort_id}"
                    if cohort_dir.exists():
                        return cohort_dir
            else:
                # Fallback: legacy structure without batch_/universe= level (direct attempt_* or cohort= under view_dir)
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                attempt_dirs_raw = [
                    d for d in iterdir_sorted(view_dir, filter_fn=lambda p: p.is_dir() and p.name.startswith("attempt_"))
                ]
                attempt_dirs = sorted(
                    attempt_dirs_raw,
                    key=lambda x: (
                        int(x.name.split("_", 1)[1]) if x.name.split("_", 1)[1].isdigit() else 999,  # Numeric sort
                        x.name  # Fallback to name for non-numeric
                    )
                )
                # Try attempt subdirectories first (new structure without batch_ level)
                for attempt_dir in attempt_dirs:
                    cohort_dir = attempt_dir / f"cohort={cohort_id}"
                    if cohort_dir.exists():
                        return cohort_dir
                # Fallback: legacy structure (direct cohort= under view_dir)
                cohort_dir = view_dir / f"cohort={cohort_id}"
                if cohort_dir.exists():
                    return cohort_dir
        
        return None
    except Exception as e:
        logger.debug(f"Failed to find cohort directory by ID: {e}")
        return None


def get_cohort_dir_from_metrics_path(metrics_path: Path, base_output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Map a metrics path to the corresponding cohort directory (for finding diffs).
    
    Maps from:
    - targets/<target>/metrics/view=CROSS_SECTIONAL/metrics.json
      → targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/
    - targets/<target>/metrics/view=SYMBOL_SPECIFIC/symbol=<symbol>/metrics.json
      → targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<id>/
    
    Note: This function returns the view-level directory. The specific cohort directory
    must be determined by finding the matching cohort_id.
    
    Args:
        metrics_path: Metrics file or directory path (from metrics/)
        base_output_dir: Optional base output directory (will be inferred if not provided)
    
    Returns:
        Path to reproducibility view directory, or None if path cannot be resolved
    """
    metrics_path = Path(metrics_path)
    
    # If it's a file, get the parent directory
    if metrics_path.is_file():
        metrics_dir = metrics_path.parent
    else:
        metrics_dir = metrics_path
    
    # Find base_output_dir if not provided
    if base_output_dir is None:
        temp_dir = metrics_dir
        for _ in range(10):
            if temp_dir.name == "targets" and (temp_dir.parent / "targets").exists():
                base_output_dir = temp_dir.parent
                break
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        if base_output_dir is None:
            logger.warning(f"Could not find base_output_dir from metrics_path: {metrics_path}")
            return None
    
    # Extract target, view, and symbol from metrics path
    parts = metrics_dir.parts
    target = None
    view = None
    symbol = None
    
    # Find target (should be after "targets")
    for i, part in enumerate(parts):
        if part == "targets" and i + 1 < len(parts):
            target = parts[i + 1]
            break
    
    if not target:
        logger.warning(f"Could not extract target from metrics_path: {metrics_path}")
        return None
    
    # Find stage, view and symbol from metrics path
    # Handle both new (stage=*/view=*/) and legacy (view=*/) structures
    stage = None
    for i, part in enumerate(parts):
        if part == "metrics" and i + 1 < len(parts):
            next_part = parts[i + 1]
            # Check if next part is stage= or view=
            if next_part.startswith("stage="):
                stage = next_part.replace("stage=", "")
                if i + 2 < len(parts):
                    view_part = parts[i + 2]
                    if view_part.startswith("view="):
                        view = view_part.replace("view=", "")
                        if i + 3 < len(parts):
                            symbol_part = parts[i + 3]
                            if symbol_part.startswith("symbol="):
                                symbol = symbol_part.replace("symbol=", "")
            elif next_part.startswith("view="):
                view = next_part.replace("view=", "")
                if i + 2 < len(parts):
                    symbol_part = parts[i + 2]
                    if symbol_part.startswith("symbol="):
                        symbol = symbol_part.replace("symbol=", "")
            break
    
    if not view:
        logger.warning(f"Could not extract view from metrics_path: {metrics_path}")
        return None
    
    # Build reproducibility path with stage if present
    # SST: Normalize view to string for path construction (defensive)
    view_str = view.value if isinstance(view, View) else str(view)
    repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage) / view_str
    if symbol:
        repro_dir = repro_dir / f"symbol={symbol}"
    
    return repro_dir


def run_root(output_dir: Path) -> Path:
    """
    Get run root directory (has targets/, globals/, cache/).
    
    Walks up from output_dir to find the run directory that contains
    targets/, globals/, or cache/ directories.
    
    Args:
        output_dir: Any path within the run directory
    
    Returns:
        Path to run root directory
    """
    current = Path(output_dir).resolve()
    for _ in range(20):  # Limit search depth
        if (current / "targets").exists() or (current / "globals").exists() or (current / "cache").exists():
            return current
        if not current.parent.exists() or current.parent == current:
            break
        current = current.parent
    
    # Fallback: return original if we can't find run root
    logger.warning(f"Could not find run root from {output_dir}, using as-is")
    return Path(output_dir).resolve()


def training_results_root(run_root: Path) -> Path:
    """
    DEPRECATED: Get training_results/ directory.
    
    This function is deprecated. All models now go to targets/<target>/models/.
    Use ArtifactPaths.model_dir() instead.
    
    Args:
        run_root: Run root directory
    
    Returns:
        Path to training_results/ directory (deprecated - for backward compatibility only)
    """
    import warnings
    warnings.warn(
        "training_results_root() is deprecated. Use ArtifactPaths.model_dir() instead. "
        "All models now go to targets/<target>/models/.",
        DeprecationWarning,
        stacklevel=2
    )
    return Path(run_root) / "training_results"


def globals_dir(run_root: Path, kind: Optional[str] = None) -> Path:
    """
    Get globals directory with optional subfolder.
    
    Args:
        run_root: Run root directory
        kind: Optional subfolder name ("routing", "training", "summaries", "rankings", or None for root)
    
    Returns:
        Path to globals/ or globals/{kind}/ directory
    """
    base = get_globals_dir(Path(run_root))
    if kind:
        return base / kind
    return base


def target_repro_dir(
    run_root: Path, 
    target: str, 
    view: Union[str, View], 
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None,  # For cross-run reproducibility
    stage: Optional[Union[str, Stage]] = None,  # Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
) -> Path:
    """
    Get reproducibility directory for target, scoped by stage/view/universe/symbol.
    
    Args:
        run_root: Run root directory
        target: Target name
        view: REQUIRED view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
        universe_sig: Optional universe signature (hash of sorted symbols list).
                      When provided, adds universe={universe_sig}/ to path for
                      cross-run reproducibility and collision prevention.
        stage: Pipeline stage for path scoping (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Returns:
        With stage:
        - targets/{target}/reproducibility/stage={stage}/CROSS_SECTIONAL/[universe={universe_sig}/]
        - targets/{target}/reproducibility/stage={stage}/SYMBOL_SPECIFIC/[universe={universe_sig}/]symbol={symbol}/
        Without stage (legacy):
        - targets/{target}/reproducibility/CROSS_SECTIONAL/[universe={universe_sig}/]
        - targets/{target}/reproducibility/SYMBOL_SPECIFIC/[universe={universe_sig}/]symbol={symbol}/
    
    Raises:
        ValueError: If view is None or invalid, or if SYMBOL_SPECIFIC without symbol
    
    Note:
        This function uses the passed view parameter. Callers should pass view from run context (SST)
        to ensure consistency. If view differs from view, a warning is logged.
        
        universe_sig should be computed from compute_universe_signature(symbols) and passed in for
        cross-run reproducibility. Different universes get different directories.
    """
    if view is None:
        raise ValueError("view parameter is required for feature selection artifacts")
    # Normalize view to enum for validation
    view_enum = View.from_string(view) if isinstance(view, str) else view
    if view_enum not in (View.CROSS_SECTIONAL, View.SYMBOL_SPECIFIC):
        raise ValueError(f"Invalid view: {view}. Must be View.CROSS_SECTIONAL or View.SYMBOL_SPECIFIC")
    if view_enum == View.SYMBOL_SPECIFIC and symbol is None:
        raise ValueError("symbol parameter is required when view='SYMBOL_SPECIFIC'")
    
    # CRITICAL: Validate view matches symbol count before path construction
    view = _validate_view_for_path_construction(
        view=view,
        symbol=symbol,
        universe_sig=universe_sig,
        base_output_dir=Path(run_root) if run_root else None
    )
    # Re-normalize after validation (may have changed)
    view_enum = View.from_string(view) if isinstance(view, str) else view
    view_str = str(view) if hasattr(view, 'value') else str(view)
    
    # Derive path_mode from SST view (convert enum to string for path construction)
    path_mode = str(view_enum)  # View enum's __str__ returns .value
    try:
        from TRAINING.orchestration.utils.run_context import get_view
        view = get_view(run_root)
        if view and view != view:
            # Warn-once: don't spam logs for every file/target
            warn_key = (view, view)
            if warn_key not in _view_mismatch_warned:
                _view_mismatch_warned.add(warn_key)
                logger.warning(
                    f"View mismatch (once): passed view={view} but view={view}. "
                    f"Using view={view} for path construction."
                )
            path_mode = view  # Use SST for paths, but don't mutate view
    except Exception as e:
        logger.debug(f"Run context not available for view resolution: {e}")
    
    # Pass stage to get_target_reproducibility_dir for stage-scoped paths
    base_repro_dir = get_target_reproducibility_dir(Path(run_root), target, stage=stage)
    
    # Build path with optional universe_sig using path_mode (SST-derived, already string)
    # path_mode is already a string from view_enum conversion above
    # DETERMINISTIC: String comparison and path construction are deterministic
    if path_mode == View.CROSS_SECTIONAL.value:
        path = base_repro_dir / View.CROSS_SECTIONAL.value
        if universe_sig:
            path = path / f"batch_{universe_sig[:12]}"  # Use batch_ prefix (deterministic slice)
        return path
    else:  # SYMBOL_SPECIFIC
        path = base_repro_dir / View.SYMBOL_SPECIFIC.value
        # NO universe= for SYMBOL_SPECIFIC (symbol= is sufficient)
        return path / f"symbol={symbol}"


def target_repro_file_path(
    run_root: Path, 
    target: str, 
    filename: str, 
    view: Union[str, View],
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None,
    stage: Optional[Union[str, Stage]] = None,  # Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
) -> Path:
    """
    Get file path in stage/view-scoped reproducibility directory.
    
    This function constructs the path to a file in the reproducibility directory,
    scoped by stage/view/universe/symbol. For feature selection artifacts, view is REQUIRED.
    
    Args:
        run_root: Run root directory
        target: Target name
        filename: Filename (e.g., "selected_features.txt")
        view: REQUIRED view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
        universe_sig: Optional universe signature for cross-run reproducibility
        stage: Pipeline stage for path scoping (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Returns:
        Path to file in stage/view-scoped reproducibility directory
    
    Raises:
        ValueError: If view is None or invalid, or if SYMBOL_SPECIFIC without symbol
    """
    repro_dir = target_repro_dir(run_root, target, view, symbol, universe_sig, stage=stage)
    return repro_dir / filename


def model_output_dir(
    training_results_root: Path, 
    family: str,
    # SST: Preferred - accept WriteScope directly
    scope: Optional["WriteScope"] = None,
    # DEPRECATED: Loose args (for backward compat)
    view: Optional[Union[str, View]] = None,
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None
) -> Path:
    """
    Get model output directory.
    
    Args:
        training_results_root: Training results root directory
        family: Model family name (e.g., "lightgbm")
        scope: WriteScope object (preferred, SST-compliant)
        view: View name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC") (deprecated, use scope)
        symbol: Optional symbol name (deprecated, use scope)
        universe_sig: Optional universe signature for cross-run reproducibility (deprecated, use scope)
    
    Returns:
        Path to training_results/{family}/view={view}/[universe={universe_sig}/][symbol={symbol}/]
    """
    # SST: Extract from WriteScope if provided
    if scope is not None:
        if not _WRITE_SCOPE_AVAILABLE:
            raise ValueError("WriteScope not available but scope was passed")
        view = scope.view
        symbol = scope.symbol
        universe_sig = scope.universe_sig
    elif view is None:
        # Default to CROSS_SECTIONAL if neither scope nor view provided
        view = View.CROSS_SECTIONAL
    
    # CRITICAL: Validate view matches symbol count before path construction
    # Note: base_output_dir not available here, but we can still validate from universe_sig
    base_output_dir_for_validation = Path(training_results_root).parent if Path(training_results_root).name == 'training_results' else Path(training_results_root)
    view = _validate_view_for_path_construction(
        view=view,
        symbol=symbol,
        universe_sig=universe_sig,
        base_output_dir=base_output_dir_for_validation
    )
    
    family_dir = Path(training_results_root) / family
    # SST: Explicitly convert enum to string for path construction (defensive)
    view_str = view.value if isinstance(view, View) else (view if isinstance(view, str) else str(view))
    view_dir = family_dir / f"view={view_str}"
    
    # Add universe scoping if provided
    if universe_sig:
        view_dir = view_dir / f"universe={universe_sig}"
    
    # Use normalized view_str for comparisons
    if view_str == View.SYMBOL_SPECIFIC.value and symbol:
        return view_dir / f"symbol={symbol}"
    elif view_str == View.CROSS_SECTIONAL.value:
        return view_dir
    else:
        # Invalid combination - log warning and return view_dir
        if view_str == View.SYMBOL_SPECIFIC.value and not symbol:
            logger.warning(f"SYMBOL_SPECIFIC view requires symbol parameter, returning view directory without symbol")
        return view_dir


# =============================================================================
# SST-Aware Path Scanning Helpers
# =============================================================================

from typing import List, Iterator, Tuple


def iter_stage_dirs(
    base_output_dir: Path,
    target: str,
) -> Iterator[Tuple[Optional[str], Path]]:
    """
    Iterate over all stage directories for a target.
    
    Handles both new (stage=*/) and legacy (no stage prefix) structures.
    
    Yields:
        Tuples of (stage_name, path) where:
        - stage_name: Stage name (e.g., "TARGET_RANKING") or None for legacy
        - path: Path to the stage-scoped or legacy reproducibility directory
    
    Examples:
        ("TARGET_RANKING", .../stage=TARGET_RANKING/)
        ("FEATURE_SELECTION", .../stage=FEATURE_SELECTION/)
        (None, .../reproducibility/) for legacy structure
    """
    repro_base = get_target_dir(base_output_dir, target) / "reproducibility"
    
    if not repro_base.exists():
        return
    
    has_stage_dirs = False
    
    # New structure: stage=* directories
    # DETERMINISTIC: Use glob_sorted for deterministic iteration with relative paths
    from TRAINING.common.utils.determinism_ordering import glob_sorted
    for stage_dir in glob_sorted(repro_base, "stage=*", filter_fn=lambda p: p.is_dir()):
        if stage_dir.is_dir():
            stage_name = stage_dir.name.replace("stage=", "")
            has_stage_dirs = True
            yield (stage_name, stage_dir)
    
    # Legacy structure: direct view directories (no stage= prefix)
    # Only yield if no stage= directories found (pure legacy) or always for backward compat
    if not has_stage_dirs:
        # Check for view directories directly under reproducibility
        for view_name in (View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value):
            view_dir = repro_base / view_name
            if view_dir.exists():
                yield (None, repro_base)
                break  # Only yield once for legacy


def find_cohort_dirs(
    base_output_dir: Path,
    target: Optional[str] = None,
    stage: Optional[Union[str, Stage]] = None,
    view: Optional[str] = None,
) -> List[Path]:
    """
    SST-aware cohort directory scanner.
    
    Scans for cohort directories in both new (stage=*/) and legacy structures.
    
    Priority:
    1. If stage provided: targets/{target}/reproducibility/stage={stage}/{view}/*/cohort=*
    2. If stage=None, try SST: get_current_stage() and scan new structure
    3. Fallback: scan all structures (new + legacy)
    
    Args:
        base_output_dir: Base run output directory
        target: Optional target name (if None, scans all targets)
        stage: Optional stage filter (if None, scans all stages)
        view: Optional view filter ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
    
    Returns:
        List of cohort directory paths found
    """
    cohort_dirs = []
    
    # Get targets to scan
    targets_dir = base_output_dir / "targets"
    if not targets_dir.exists():
        return cohort_dirs
    
    if target:
        target_names = [target]
    else:
        # DETERMINISTIC: Sort target names for consistent iteration order
        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        from TRAINING.common.utils.determinism_ordering import iterdir_sorted
        target_names = sorted([d.name for d in iterdir_sorted(targets_dir, filter_fn=lambda p: p.is_dir())])
    
    # Resolve stage from SST if not provided
    resolved_stage = stage
    if resolved_stage is None:
        try:
            from TRAINING.orchestration.utils.run_context import get_current_stage
            resolved_stage = get_current_stage(base_output_dir)
        except Exception:
            pass
    
    for target_name in target_names:
        for stage_name, stage_path in iter_stage_dirs(base_output_dir, target_name):
            # Filter by stage if specified
            if resolved_stage and stage_name and stage_name != resolved_stage:
                continue
            
            # Scan view directories
            # Normalize view to string for path scanning
            view_str = str(view) if isinstance(view, View) or (view and hasattr(view, 'value')) else view
            views_to_scan = [view_str] if view else [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]
            for view_name in views_to_scan:
                view_path = stage_path / view_name
                if not view_path.exists():
                    continue
                
                # Scan for cohort directories (may be nested under universe=* or symbol=*)
                # Pattern: {view}/[universe=*]/[symbol=*]/cohort=*
                # DETERMINISTIC: Use rglob_sorted for deterministic iteration with relative paths
                from TRAINING.common.utils.determinism_ordering import rglob_sorted
                for cohort_path in rglob_sorted(view_path, "cohort=*", filter_fn=lambda p: p.is_dir()):
                    cohort_dirs.append(cohort_path)
    
    # DETERMINISTIC: Sort by semantic keys (attempt_id, cohort_id, stable path) instead of str(p)
    # This prevents machine-dependent ordering from absolute paths
    
    def _parse_cohort_id_from_dirname(name: str) -> str:
        """Parse cohort_id from directory name."""
        if not name.startswith("cohort="):
            return f"__INVALID__{name}"  # Sentinel to expose bugs early
        return name.split("cohort=", 1)[1]
    
    def _stable_path_id(p: Path, tail_parts: int = 6) -> str:
        """Get stable path identifier (tail parts) for deterministic sorting."""
        try:
            # Try to find repo root by walking up for CONFIG/.git
            current = p.resolve()
            repo_root = None
            for _ in range(10):
                if (current / "CONFIG").is_dir() or (current / ".git").exists():
                    repo_root = current
                    break
                if current.parent == current:
                    break
                current = current.parent
            
            if repo_root:
                return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except Exception:
            pass
        
        # Fallback: use tail parts (deterministic across machines if layout matches)
        return "/".join(p.resolve().parts[-tail_parts:])
    
    cohort_dirs.sort(
        key=lambda p: (
            parse_attempt_id_from_cohort_dir(p),  # Numeric attempt_id (defined above in this module)
            _parse_cohort_id_from_dirname(p.name),  # Cohort ID
            _stable_path_id(p)  # Stable path identifier
        )
    )
    
    return cohort_dirs


def build_target_cohort_dir(
    base_output_dir: Path,
    target: str,
    stage: Union[str, Stage],
    view: Union[str, View],
    cohort_id: str,
    symbol: Optional[str] = None,
    attempt_id: int = 0,
    universe_sig: Optional[str] = None  # NEW: Required for CROSS_SECTIONAL batch_ level
) -> Path:
    """
    Build canonical cohort directory path.
    
    Always creates attempt_{attempt_id}/ subdirectory (including attempt_0).
    This ensures consistent structure and avoids mixed legacy/new paths.
    
    Structure:
    - CROSS_SECTIONAL: .../CROSS_SECTIONAL/batch_{universe_sig[:12]}/attempt_{attempt_id}/cohort={cohort_id}/
    - SYMBOL_SPECIFIC: .../SYMBOL_SPECIFIC/symbol={symbol}/attempt_{attempt_id}/cohort={cohort_id}/
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
        view: View (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        cohort_id: Cohort identifier
        symbol: Optional symbol name (required for SYMBOL_SPECIFIC)
        attempt_id: Attempt identifier (default 0, always included in path)
        universe_sig: Optional universe signature (required for CROSS_SECTIONAL batch_ level)
    
    Returns:
        Path to cohort directory with attempt subdirectory
    """
    target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage)
    view_str = view.value if isinstance(view, View) else str(view)
    
    if view_str == View.SYMBOL_SPECIFIC.value and symbol:
        return target_repro_dir / view_str / f"symbol={symbol}" / f"attempt_{attempt_id}" / f"cohort={cohort_id}"
    else:  # CROSS_SECTIONAL
        path = target_repro_dir / view_str
        # CRITICAL: CROSS_SECTIONAL always requires batch_ level for consistency
        # If universe_sig is missing, try to extract from cohort_id or use fallback
        if not universe_sig:
            # Try to extract universe_sig from cohort_id (format: cs_<date>_<universe_sig>_<config>_<version>_<hash>)
            import re
            cohort_match = re.match(r'^cs_[^_]+_([a-f0-9]{12})[^_]*_', cohort_id)
            if cohort_match:
                universe_sig = cohort_match.group(1) + '000000000000'  # Pad to full length for slice
            else:
                # Fallback: use "unknown" batch for legacy cohorts without universe_sig
                # This ensures consistent structure even for old data
                universe_sig = '0' * 64  # Full-length hash for consistent slicing
        path = path / f"batch_{universe_sig[:12]}"  # Always add batch_ level (deterministic slice)
        return path / f"attempt_{attempt_id}" / f"cohort={cohort_id}"


def resolve_snapshot_path(
    base_output_dir: Path,
    target: str,
    stage: Union[str, Stage],
    view: Union[str, View],
    cohort_id: str,
    symbol: Optional[str] = None,
    attempt_id: int = 0,
    universe_sig: Optional[str] = None
) -> Path:
    """
    Resolve canonical path to snapshot.json (SST).
    
    Returns the exact path where snapshot.json should be written/read.
    Both save_snapshot() and validation should use this.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        stage: Pipeline stage
        view: View (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        cohort_id: Cohort identifier
        symbol: Optional symbol name (required for SYMBOL_SPECIFIC)
        attempt_id: Attempt identifier (default 0)
        universe_sig: Optional universe signature (required for CROSS_SECTIONAL batch_ level)
    
    Returns:
        Path to snapshot.json
    """
    cohort_dir = build_target_cohort_dir(
        base_output_dir=base_output_dir,
        target=target,
        stage=stage,
        view=view,
        cohort_id=cohort_id,
        symbol=symbol,
        attempt_id=attempt_id,
        universe_sig=universe_sig
    )
    return cohort_dir / "snapshot.json"


def resolve_diff_prev_path(
    base_output_dir: Path,
    target: str,
    stage: Union[str, Stage],
    view: Union[str, View],
    cohort_id: str,
    symbol: Optional[str] = None,
    attempt_id: int = 0,
    universe_sig: Optional[str] = None
) -> Path:
    """
    Resolve canonical path to diff_prev.json (SST).
    
    Returns the exact path where diff_prev.json should be written/read.
    Both save_diff() and validation should use this.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        stage: Pipeline stage
        view: View (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        cohort_id: Cohort identifier
        symbol: Optional symbol name (required for SYMBOL_SPECIFIC)
        attempt_id: Attempt identifier (default 0)
        universe_sig: Optional universe signature (required for CROSS_SECTIONAL batch_ level)
    
    Returns:
        Path to diff_prev.json
    """
    cohort_dir = build_target_cohort_dir(
        base_output_dir=base_output_dir,
        target=target,
        stage=stage,
        view=view,
        cohort_id=cohort_id,
        symbol=symbol,
        attempt_id=attempt_id,
        universe_sig=universe_sig
    )
    return cohort_dir / "diff_prev.json"


def parse_attempt_id_from_cohort_dir(cohort_dir: Path) -> int:
    """
    Parse attempt_id from cohort directory path using regex-based ancestor scan.
    
    Rules:
    - Scan ancestors for attempt_<n> pattern → attempt_id = n
    - Else → attempt_id = 0 (legacy, no attempt dir)
    - Malformed attempt_foo → ignored → 0
    - attempt_01 → 1 (leading zeros stripped)
    
    This provides single source of truth for attempt_id discovery.
    
    Args:
        cohort_dir: Path to cohort directory (e.g., .../attempt_1/cohort=abc123/)
    
    Returns:
        attempt_id (0 if legacy structure, parsed number if attempt dir exists)
    """
    import re
    _ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")
    
    # Scan ancestors for attempt_* pattern
    for parent in cohort_dir.parents:
        m = _ATTEMPT_RE.match(parent.name)
        if m:
            return int(m.group(1))
    return 0  # Legacy structure or no match


def parse_reproducibility_path(path: Path) -> Dict[str, Optional[str]]:
    """
    Parse a reproducibility path and extract components.
    
    Handles both:
    - New: targets/{target}/reproducibility/stage={stage}/{view}/universe={u}/cohort={id}
    - Legacy: targets/{target}/reproducibility/{view}/universe={u}/cohort={id}
    
    Args:
        path: Path to parse (can be cohort dir or any path within reproducibility structure)
    
    Returns:
        Dict with extracted components:
        {
            "target": str or None,
            "stage": str or None (None for legacy paths),
            "view": str or None,
            "universe_sig": str or None,
            "cohort_id": str or None,
            "symbol": str or None,
        }
    """
    result: Dict[str, Optional[str]] = {
        "target": None,
        "stage": None,
        "view": None,
        "universe_sig": None,
        "cohort_id": None,
        "symbol": None,
    }
    
    parts = path.parts
    
    for i, part in enumerate(parts):
        # Extract target (after "targets")
        if part == "targets" and i + 1 < len(parts):
            result["target"] = parts[i + 1]
        
        # Extract stage (stage=*)
        if part.startswith("stage="):
            result["stage"] = part.replace("stage=", "")
        
        # Extract view
        if part in (View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value):
            result["view"] = part
        
        # DETERMINISTIC: Extract universe_sig (universe=* or batch_*)
        # Check universe= first (legacy), then batch_ (new) - order matters for consistency
        if part.startswith("universe="):
            result["universe_sig"] = part.replace("universe=", "")
        elif part.startswith("batch_"):
            # Extract short hash from batch_ prefix (first 12 chars of full hash)
            # Note: This returns short hash, not full hash. For comparison/identity,
            # full hash is still used from ComparisonGroup.universe_sig or RunIdentity
            result["universe_sig"] = part.replace("batch_", "")  # Short hash (12 chars)
        
        # Extract symbol (symbol=*)
        if part.startswith("symbol="):
            result["symbol"] = part.replace("symbol=", "")
        
        # Extract cohort_id (cohort=*)
        if part.startswith("cohort="):
            result["cohort_id"] = part.replace("cohort=", "")
    
    return result

