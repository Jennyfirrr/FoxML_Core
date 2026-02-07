# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Registry

Manages feature metadata and enforces temporal rules to prevent data leakage.
Makes leakage structurally impossible without lying to the configuration.
"""

import re
import yaml
import json
import logging
import numbers
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# Import patch file utilities from neutral module (avoids import cycles)
from TRAINING.common.registry_patch_naming import safe_target_filename, find_patch_file

logger = logging.getLogger(__name__)

# Try to import config loader for path configuration
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass

# Global registry instance (lazy-loaded)
_REGISTRY: Optional['FeatureRegistry'] = None

# TS-001: Thread-safe lock for registry singleton (double-check locking pattern)
# CRITICAL: Must use RLock (re-entrant) because _load_config() also acquires this lock
# and is called from within FeatureRegistry.__init__ which is called while holding the lock
_REGISTRY_LOCK = threading.RLock()

# Track which registry paths we've already logged (to avoid duplicate "Using feature registry from CWD" messages)
_LOGGED_REGISTRY_PATHS: set = set()

# CRITICAL FIX #1: Module-level tracking for auto-enabled features (shared across all instances)
# Since get_registry() creates new instances per-target, we need shared tracking
# DETERMINISM: Dict insertion order preserved in Python 3.7+, and we always sort before reading
# Feature processing order is deterministic (sorted features from filter_features_for_target)
# Target processing order is deterministic (sorted targets in orchestrator)
_AUTO_ENABLED_FEATURES_GLOBAL: Dict[str, Dict[str, Any]] = {}  # Key: feature_name, Value: tracking dict

# TS-003: Thread-safe lock for auto-enabled features global dict
_AUTO_ENABLED_FEATURES_LOCK = threading.Lock()

# Module-level cache for feature inference config (loaded once, reused)
_FEATURE_INFERENCE_CONFIG: Optional[Dict[str, Any]] = None
_FEATURE_INFERENCE_CONFIG_PATH: Optional[Path] = None


def _load_feature_inference_config() -> Dict[str, Any]:
    """
    Load feature inference config from CONFIG/data/feature_inference.yaml (cached).
    
    Returns:
        Feature inference config dict, or empty dict if not found (falls back to hardcoded patterns)
    """
    global _FEATURE_INFERENCE_CONFIG, _FEATURE_INFERENCE_CONFIG_PATH
    
    # Return cached config if already loaded
    if _FEATURE_INFERENCE_CONFIG is not None:
        return _FEATURE_INFERENCE_CONFIG
    
    # Try to find config file
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "CONFIG" / "data" / "feature_inference.yaml"
    
    if not config_path.exists():
        # Try alternative location
        alt_path = repo_root / "CONFIG" / "feature_inference.yaml"
        if alt_path.exists():
            config_path = alt_path
        else:
            logger.debug(f"Feature inference config not found at {config_path} or {alt_path}, using hardcoded patterns")
            _FEATURE_INFERENCE_CONFIG = {}
            _FEATURE_INFERENCE_CONFIG_PATH = None
            return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        inference_config = config.get('feature_inference', {})
        _FEATURE_INFERENCE_CONFIG = inference_config
        _FEATURE_INFERENCE_CONFIG_PATH = config_path
        logger.debug(f"Loaded feature inference config from {config_path}")
        return inference_config
    except Exception as e:
        logger.warning(f"Failed to load feature inference config from {config_path}: {e}, using hardcoded patterns")
        _FEATURE_INFERENCE_CONFIG = {}
        _FEATURE_INFERENCE_CONFIG_PATH = None
        return {}


def resolve_registry_path_for_interval(
    base_path: Optional[Path] = None,
    interval_minutes: Optional[float] = None,
    selection_mode: str = "manual",
    selection_strict: bool = False,
    interval_path_map: Optional[Dict[int, str]] = None
) -> Path:
    """
    Resolve registry path with optional interval-based selection.
    
    CRITICAL: This is the single source of truth for registry path resolution.
    All registry loading must go through this function.
    
    Selection logic (deterministic, no globbing):
    1. If selection_mode="manual" or interval_minutes is None:
       → Return base_path (or canonical default)
    
    2. If selection_mode="auto" and interval_minutes is provided:
       a. If interval_path_map provided and interval_minutes in map:
          → Use explicit mapping (most robust)
       b. Else if interval_minutes is integer:
          → Try: feature_registry.{interval}m.yaml
          → Fallback: feature_registry.yaml (if selection_strict=False)
          → Error: if selection_strict=True and interval-specific missing
       c. Else (non-integer interval):
          → Return base_path (manual only for non-integer intervals)
    
    Args:
        base_path: Base registry path (if None, uses canonical from config)
        interval_minutes: Current bar interval (for auto-selection)
        selection_mode: "manual" or "auto" (from config)
        selection_strict: If True, error if interval-specific missing (from config)
        interval_path_map: Optional explicit mapping {interval: path} (from config)
    
    Returns:
        Resolved registry path
    
    Raises:
        FileNotFoundError: If selection_strict=True and interval-specific file missing
        ValueError: If selection_mode invalid or interval_path_map has invalid keys
    """
    # Resolve base path (SST)
    if base_path is None:
        base_path = _resolve_registry_path_base()
    
    # Manual mode or no interval → return base
    if selection_mode != "auto" or interval_minutes is None:
        return base_path
    
    # Auto mode: try interval-specific selection
    base_dir = base_path.parent
    base_stem = base_path.stem  # "feature_registry"
    base_suffix = base_path.suffix  # ".yaml"
    
    # Check explicit mapping first (most robust)
    if interval_path_map:
        interval_int = int(interval_minutes) if interval_minutes == int(interval_minutes) else None
        if interval_int is not None and interval_int in interval_path_map:
            mapped_path_str = interval_path_map[interval_int]
            mapped_path = Path(mapped_path_str)
            if not mapped_path.is_absolute():
                # Resolve relative to base_dir
                mapped_path = base_dir / mapped_path
            if mapped_path.exists():
                return mapped_path
            elif selection_strict:
                raise FileNotFoundError(
                    f"Registry selection_strict=True: interval_path_map[{interval_int}] = "
                    f"'{mapped_path_str}' does not exist at {mapped_path}"
                )
            else:
                logger.warning(
                    f"Registry interval_path_map[{interval_int}] = '{mapped_path_str}' not found. "
                    f"Falling back to base registry: {base_path}"
                )
                return base_path
    
    # Try naming convention (only for integer intervals)
    if interval_minutes == int(interval_minutes):
        interval_int = int(interval_minutes)
        candidate_path = base_dir / f"{base_stem}.{interval_int}m{base_suffix}"
        if candidate_path.exists():
            return candidate_path
        elif selection_strict:
            raise FileNotFoundError(
                f"Registry selection_strict=True: interval-specific registry not found: {candidate_path}. "
                f"Expected for interval_minutes={interval_int}. "
                f"Create {candidate_path} or set selection_strict=False to fallback to base registry."
            )
        else:
            logger.debug(
                f"Registry auto-selection: interval-specific {candidate_path} not found. "
                f"Falling back to base registry: {base_path}"
            )
            return base_path
    else:
        # Non-integer interval → manual only
        logger.debug(
            f"Registry auto-selection: interval_minutes={interval_minutes} is not integer. "
            f"Using base registry (manual selection required for non-integer intervals): {base_path}"
        )
        return base_path


def _resolve_registry_path_base() -> Path:
    """
    Resolve base canonical feature registry path (SST helper).
    
    This is the base path resolution without interval selection.
    Used internally by resolve_registry_path_for_interval().
    
    Returns:
        Path to feature_registry.yaml (canonical path)
    """
    if _CONFIG_AVAILABLE:
        try:
            system_cfg = get_system_config()
            config_paths = system_cfg.get('system', {}).get('paths', {})
            registry_path_str = config_paths.get('feature_registry')
            if registry_path_str:
                config_path = Path(registry_path_str)
                if not config_path.is_absolute():
                    repo_root = Path(__file__).resolve().parents[2]
                    config_path = repo_root / registry_path_str
                return config_path
            else:
                # Use config_dir from config
                config_dir = config_paths.get('config_dir', 'CONFIG')
                repo_root = Path(__file__).resolve().parents[2]
                return repo_root / config_dir / "data" / "feature_registry.yaml"
        except Exception:
            # Fallback to canonical registry (SST)
            repo_root = Path(__file__).resolve().parents[2]
            return repo_root / "CONFIG" / "data" / "feature_registry.yaml"
    else:
        # Fallback to canonical registry (SST)
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "CONFIG" / "data" / "feature_registry.yaml"


def _resolve_registry_path() -> Path:
    """
    Resolve canonical feature registry path (SST helper).
    
    Now delegates to resolve_registry_path_for_interval() with manual mode by default.
    Used by FeatureRegistry.__init__(), get_registry_path(), and LeakageAutoFixer.
    
    Returns:
        Path to feature_registry.yaml (canonical path)
    """
    # Read selection config (SST)
    selection_mode = "manual"  # Safe default
    selection_strict = False
    interval_path_map = None
    try:
        from CONFIG.config_loader import get_cfg
        selection_mode = get_cfg('registry.selection_mode', default="manual")
        selection_strict = get_cfg('registry.selection_strict', default=False)
        interval_path_map = get_cfg('registry.interval_path_map', default=None)
    except Exception:
        pass  # Use safe defaults
    
    # Get interval from config if available (for auto-selection)
    interval_minutes = None
    try:
        from CONFIG.config_loader import get_cfg
        interval_minutes = get_cfg('data.bar_interval', default=None)
    except Exception:
        pass
    
    # Delegate to interval-aware resolver
    return resolve_registry_path_for_interval(
        base_path=None,  # Will resolve canonical default
        interval_minutes=interval_minutes,
        selection_mode=selection_mode,
        selection_strict=selection_strict,
        interval_path_map=interval_path_map
    )


def _build_pattern_list_with_precedence(inference_config: Dict[str, Any]) -> List[Tuple[int, str, Dict[str, Any]]]:
    """
    Build sorted list of patterns with precedence for deterministic matching.
    
    Returns:
        List of (priority, pattern_string, pattern_config) tuples, sorted by priority (desc), then pattern (asc)
    """
    patterns_list = []
    
    pattern_groups = inference_config.get('patterns', {})
    
    # DETERMINISM: Sort pattern groups by priority (descending), then by group name (ascending) for tie-breaking
    from TRAINING.common.utils.determinism_ordering import sorted_keys
    
    # Build list of (priority, group_name, group_config) for sorting
    group_items = []
    for group_name, group_config in pattern_groups.items():
        priority = group_config.get('priority', 0)
        group_items.append((priority, group_name, group_config))
    
    # Sort by priority (descending), then by group_name (ascending) for deterministic tie-breaking
    group_items.sort(key=lambda x: (-x[0], x[1]))
    
    # Process groups in sorted order
    for priority, group_name, group_config in group_items:
        # Handle rejection_patterns (list of pattern dicts)
        if 'patterns' in group_config and isinstance(group_config['patterns'], list):
            for pattern_item in group_config['patterns']:
                if isinstance(pattern_item, dict) and 'pattern' in pattern_item:
                    pattern_str = pattern_item['pattern']
                    patterns_list.append((priority, pattern_str, {**group_config, **pattern_item}))
        
        # Handle single pattern (e.g., lagged_returns)
        elif 'pattern' in group_config:
            pattern_str = group_config['pattern']
            patterns_list.append((priority, pattern_str, group_config))
        
        # Handle exact_matches (e.g., metadata_columns)
        elif 'exact_matches' in group_config:
            for exact_match in group_config.get('exact_matches', []):
                # Convert exact match to pattern (escape special chars)
                pattern_str = f"^{re.escape(exact_match)}$"
                patterns_list.append((priority, pattern_str, {**group_config, 'exact_match': exact_match}))
    
    # Sort by priority (descending), then by pattern string (ascending) for deterministic matching
    patterns_list.sort(key=lambda x: (-x[0], x[1]))
    
    return patterns_list


class FeatureRegistry:
    """
    Manages feature metadata and enforces temporal rules.
    
    Features must have:
    - lag_bars: How many bars back the feature is allowed to peek (>= 0)
    - allowed_horizons: List of target horizons this feature can predict
    - source: Where the feature comes from (price, volume, derived, etc.)
    
    Hard rules:
    - lag_bars >= 0 (cannot look into future)
    - lag_bars >= horizon_bars for price/derived features
    - allowed_horizons must be non-empty for usable features
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        target_column: Optional[str] = None,
        registry_overlay_dir: Optional[Path] = None,  # NEW: explicit patch directory
        current_bar_minutes: Optional[float] = None,  # NEW: for compatibility check
        allow_overwrite: bool = False,  # NEW: allow overlay to overwrite explicit values
        _suppress_log: bool = False,  # Internal: suppress logging for non-cached instances
        _strict: bool = False  # Internal: strict mode for _load_config (set by get_registry)
    ):
        """
        Initialize feature registry from YAML config.
        
        Args:
            config_path: Path to feature_registry.yaml (default: from config or CONFIG/feature_registry.yaml)
            target_column: Optional target column name (for per-target patch loading)
            registry_overlay_dir: Optional directory containing registry patches (explicit discovery only)
            current_bar_minutes: Optional current bar interval in minutes (for patch compatibility check)
            allow_overwrite: Allow overlay to overwrite explicit non-null values (default: False, safe)
        """
        if config_path is None:
            config_path = _resolve_registry_path()
        
        self.config_path = Path(config_path)
        self.config = self._load_config(strict=_strict)
        self.features = self.config.get('features', {})
        # DETERMINISM: feature_families is a YAML mapping (dict)
        # Python 3.7+ preserves insertion order for dicts, so YAML order is preserved
        # This is sufficient for deterministic family matching (author-defined precedence)
        self.families = self.config.get('feature_families', {})
        self.validation_rules = self.config.get('validation', {})

        # Phase 19: Feature aliases for interval-agnostic naming
        # Aliases map interval-agnostic names to canonical (interval-specific) names
        # Example: ret_short -> ret_5m (so code can use interval-agnostic names)
        self._aliases = self._build_alias_lookup()
        
        # CRITICAL: Enforce registry interval compatibility
        # Registry stores allowed_horizons in bars, but bars are only meaningful with a specific interval_minutes
        registry_bar_minutes = self.config.get('metadata', {}).get('registry_bar_minutes')
        
        # CRITICAL: Missing metadata is unsafe in strict mode (don't default to 5)
        if registry_bar_minutes is None:
            if current_bar_minutes is not None:
                # Registry missing interval metadata - unsafe to use
                # In strict mode: error. In best_effort: warning + proceed with stamp
                error_msg = (
                    f"Registry {self.config_path} missing 'metadata.registry_bar_minutes'. "
                    f"Cannot verify interval compatibility. Registry allowed_horizons may be incorrect."
                )
                
                # Check error policy (from config or default to best_effort for backward compat)
                error_policy = "best_effort"  # Safe default
                try:
                    from CONFIG.config_loader import get_cfg
                    error_policy = get_cfg('registry.error_policy', default='best_effort')
                except Exception:
                    pass
                
                if error_policy == "strict":
                    raise ValueError(
                        f"{error_msg} "
                        f"Add 'metadata.registry_bar_minutes' to registry or set error_policy='best_effort'."
                    )
                else:
                    logger.warning(
                        f"{error_msg} "
                        f"Proceeding with assumption that registry matches current interval ({current_bar_minutes}m). "
                        f"Outputs will be stamped as 'interval_unknown'."
                    )
                    # Stamp registry as interval_unknown for downstream
                    self._interval_unknown = True
        else:
            # Registry has interval metadata - enforce compatibility
            if current_bar_minutes is not None:
                # Normalize to int for comparison (handle float vs int)
                registry_bar_int = int(registry_bar_minutes)
                current_bar_int = int(current_bar_minutes)
                if registry_bar_int != current_bar_int:
                    error_msg = (
                        f"Registry interval mismatch: registry was created for {registry_bar_int}-minute bars, "
                        f"but current run uses {current_bar_int}-minute bars. "
                        f"Registry allowed_horizons are not compatible."
                    )
                    
                    # Check error policy
                    error_policy = "best_effort"  # Safe default
                    try:
                        from CONFIG.config_loader import get_cfg
                        error_policy = get_cfg('registry.error_policy', default='best_effort')
                    except Exception:
                        pass
                    
                    if error_policy == "strict":
                        raise ValueError(
                            f"{error_msg} "
                            f"Use a registry created for {current_bar_int}-minute bars or adjust run interval."
                        )
                    else:
                        logger.warning(
                            f"{error_msg} "
                            f"Registry allowed_horizons may be incorrect for this interval."
                        )
        
        # Load per-target overrides in precedence order
        self.per_target_patches = {}  # Run patches (highest priority)
        self.per_target_overrides = {}  # Persistent overrides (medium priority)
        self.per_target_unblocks = {}  # Unblock patches (allow-cancel, highest priority allow)
        
        # Load auto overlay (workspace overlay, lowest priority but applied last)
        self.auto_overlay = {}  # Auto-generated overlay patches
        
        # Store allow_overwrite flag (set at construction, not mutated later)
        self.allow_overwrite = allow_overwrite
        
        # Store target_column for per-target overlay loading
        self.target_column = target_column
        
        # Store current_bar_minutes for overlay compatibility checks
        self.current_bar_minutes = current_bar_minutes
        
        if target_column:
            # 1. Load run patch (highest priority deny) - EXPLICIT discovery only
            if registry_overlay_dir:
                patch_file = find_patch_file(registry_overlay_dir, target_column)
                if patch_file and patch_file.exists():
                    try:
                        with open(patch_file, 'r') as f:
                            patch_data = yaml.safe_load(f) or {}
                        
                        # COMPATIBILITY CHECK: bar_minutes must match (treat as int)
                        patch_bar_minutes = patch_data.get('bar_minutes')
                        if current_bar_minutes is not None and patch_bar_minutes is not None:
                            # Normalize to int for comparison (handle float vs int)
                            patch_bar_int = int(patch_bar_minutes)
                            current_bar_int = int(current_bar_minutes)
                            if patch_bar_int != current_bar_int:
                                logger.warning(
                                    f"Patch {patch_file} has bar_minutes={patch_bar_minutes} (normalized: {patch_bar_int}), "
                                    f"but current bar_minutes={current_bar_minutes} (normalized: {current_bar_int}). "
                                    f"Ignoring patch (incompatible)."
                                )
                                patch_data = {}  # Ignore incompatible patch
                        
                        self.per_target_patches = patch_data
                        logger.debug(f"Loaded run patch for {target_column}: {patch_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load patch {patch_file}: {e}")
            
            # 2. Load persistent override (medium priority deny)
            # Use CONFIG/data/feature_registry_per_target/ (consistent with promotion/unblock)
            repo_root = Path(__file__).resolve().parents[2]
            per_target_dir = repo_root / "CONFIG" / "data" / "feature_registry_per_target"
            override_file = find_patch_file(per_target_dir, target_column)
            if override_file and override_file.exists():
                try:
                    with open(override_file, 'r') as f:
                        override_data = yaml.safe_load(f) or {}
                    
                    # COMPATIBILITY CHECK: bar_minutes must match (treat as int)
                    override_bar_minutes = override_data.get('bar_minutes')
                    if current_bar_minutes is not None and override_bar_minutes is not None:
                        # Normalize to int for comparison (handle float vs int)
                        override_bar_int = int(override_bar_minutes)
                        current_bar_int = int(current_bar_minutes)
                        if override_bar_int != current_bar_int:
                            logger.warning(
                                f"Override {override_file} has bar_minutes={override_bar_minutes} (normalized: {override_bar_int}), "
                                f"but current bar_minutes={current_bar_minutes} (normalized: {current_bar_int}). "
                                f"Ignoring override (incompatible)."
                            )
                            override_data = {}  # Ignore incompatible override
                    
                    self.per_target_overrides = override_data
                    logger.debug(f"Loaded persistent override for {target_column}: {override_file}")
                except Exception as e:
                    logger.warning(f"Failed to load override {override_file}: {e}")
            
            # 3. Load unblock patches (allow-cancel, highest priority allow)
            unblock_file = find_patch_file(per_target_dir, target_column, suffix=".unblock.yaml")
            if unblock_file and unblock_file.exists():
                try:
                    with open(unblock_file, 'r') as f:
                        unblock_data = yaml.safe_load(f) or {}
                    
                    # COMPATIBILITY CHECK: bar_minutes must match (treat as int)
                    unblock_bar_minutes = unblock_data.get('bar_minutes')
                    if current_bar_minutes is not None and unblock_bar_minutes is not None:
                        # Normalize to int for comparison (handle float vs int)
                        unblock_bar_int = int(unblock_bar_minutes)
                        current_bar_int = int(current_bar_minutes)
                        if unblock_bar_int != current_bar_int:
                            logger.warning(
                                f"Unblock {unblock_file} has bar_minutes={unblock_bar_minutes} (normalized: {unblock_bar_int}), "
                                f"but current bar_minutes={current_bar_minutes} (normalized: {current_bar_int}). "
                                f"Ignoring unblock (incompatible)."
                            )
                            unblock_data = {}  # Ignore incompatible unblock
                    
                    self.per_target_unblocks = unblock_data
                    logger.debug(f"Loaded unblock patch for {target_column}: {unblock_file}")
                except Exception as e:
                    logger.warning(f"Failed to load unblock {unblock_file}: {e}")
        
        # Load auto overlay (workspace overlay, applied last if enabled)
        self._load_auto_overlay()
        
        # Validate registry on load
        self._validate_registry()
        
        # Load auto-enable config once (cache in instance, not per-call)
        self._auto_enable_family_features = False
        self._auto_enable_threshold_count = 50
        self._auto_enable_threshold_percent = 5.0
        self._auto_enable_config_load_error = None
        
        try:
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            leakage_cfg = safety_cfg.get('safety', {}).get('leakage_detection', {})
            self._auto_enable_family_features = leakage_cfg.get(
                'auto_enable_family_features_with_empty_allowed_horizons', False
            )
            self._auto_enable_threshold_count = leakage_cfg.get(
                'auto_enable_family_features_threshold_count', 50
            )
            self._auto_enable_threshold_percent = leakage_cfg.get(
                'auto_enable_family_features_threshold_percent', 5.0
            )
        except Exception as e:
            self._auto_enable_config_load_error = str(e)
            # Log once (not per-call) if load fails
            logger.warning(f"Failed to load auto-enable config: {e}. Using safe defaults (disabled).")
        
        # Initialize tracking dict (de-duped by feature name)
        # CRITICAL FIX #1: Use module-level dict shared across all instances
        # This ensures tracking works even when get_registry() creates new instances per-target
        global _AUTO_ENABLED_FEATURES_GLOBAL
        # Reference to module-level dict (all instances share same dict)
        self._auto_enabled_features = _AUTO_ENABLED_FEATURES_GLOBAL
        
        # Only log for cached global instance (first load), not for every per-target instance
        # This reduces duplicate logging when get_registry() creates new instances
        if not _suppress_log:
            logger.info(f"Loaded feature registry: {len(self.features)} features, {len(self.families)} families")
    
    def _load_auto_overlay(self) -> None:
        """
        Load auto-generated overlay(s) from CONFIG/data/overrides/ if enabled.
        
        Loads both global and per-target overlays (layering model):
        - Global overlay applied first (baseline)
        - Per-target overlay applied second (overrides global, more specific wins)
        
        Auto overlay is applied last (after per-target overrides) but should not
        override explicit values unless allow_overwrite=True.
        
        CRITICAL: 
        - Overlay is only loaded when `autopatch.enabled=True` (use enabled for loading, apply is for promotion)
        - If `apply=True` and overlay parse fails, hard-fails the run (Tier A safety)
        - When `enabled=False`, this method returns early (write-only/observational mode)
        """
        self._overlay_loaded = False  # Track actual application status
        
        try:
            from TRAINING.common.utils.registry_autopatch import get_autopatch
            autopatch = get_autopatch()
            
            # Use 'enabled' flag for loading (not 'apply' - apply is for promotion)
            if not autopatch.enabled:
                return  # Auto overlay not enabled
            
            repo_root = Path(__file__).resolve().parents[2]
            overlay_dir = repo_root / "CONFIG" / "data" / "overrides"
            
            # Start with empty overlay (will be merged)
            self.auto_overlay = {}
            
            # Initialize overlay variables (for fingerprint computation)
            global_overrides = {}
            per_target_overrides = {}
            
            # Initialize overlay paths tracking early (deterministic, set once at end)
            from TRAINING.common.utils.horizon_conversion import is_effectively_integer
            
            self._overlay_paths_selected = {
                "global": None,
                "per_target": None,
                "interval_int": None
            }
            
            # CRITICAL: Normalize current_bar_minutes to float|int|None before is_effectively_integer()
            current_bar_minutes = getattr(self, 'current_bar_minutes', None)
            if current_bar_minutes is not None:
                try:
                    # Normalize to numeric (fail-closed for non-numeric)
                    current_bar_minutes = float(current_bar_minutes)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid current_bar_minutes: {current_bar_minutes} (type: {type(current_bar_minutes)}). Treating as None.")
                    current_bar_minutes = None
            
            interval_int = is_effectively_integer(current_bar_minutes)
            
            # Step 1: Load global overlay (baseline) - try interval-specific first
            global_overlay_file = None
            if interval_int is not None:
                global_overlay_file = overlay_dir / f"feature_registry_overrides.auto.{interval_int}m.yaml"
                if not global_overlay_file.exists():
                    global_overlay_file = None  # Fallback to interval-agnostic
            
            # Fallback to interval-agnostic
            if global_overlay_file is None:
                global_overlay_file = overlay_dir / "feature_registry_overrides.auto.yaml"
            if global_overlay_file.exists():
                try:
                    with open(global_overlay_file, 'r') as f:
                        global_data = yaml.safe_load(f) or {}

                    # Validate overlay structure
                    if not isinstance(global_data, dict):
                        raise ValueError(f"Global overlay file must contain a dict, got {type(global_data)}")

                    # CRITICAL: Check overlay compatibility (fail-closed for missing metadata)
                    overlay_metadata = global_data.get('_metadata', {})
                    overlay_interval = overlay_metadata.get('interval_minutes')
                    overlay_tool_version = overlay_metadata.get('tool_version')
                    overlay_registry_bar_minutes = overlay_metadata.get('registry_bar_minutes')
                    current_bar_minutes = getattr(self, 'current_bar_minutes', None)
                    
                    # CRITICAL: Missing overlay interval/tool_version is unsafe (potentially polluted)
                    if overlay_interval is None or overlay_tool_version is None:
                        # Check if we can prove it's safe (post-fix tool version + registry interval matches)
                        # CRITICAL: Use exact equality for tool version (not string comparison)
                        # TOOL_VERSION is a string constant - compare exactly, don't use < or >
                        from TRAINING.common.utils.registry_autopatch import TOOL_VERSION
                        if overlay_tool_version is None or overlay_tool_version != TOOL_VERSION:
                            # Old overlay without interval metadata - potentially polluted
                            if current_bar_minutes is not None:
                                # Only load if registry interval matches AND we can verify it's post-fix
                                if overlay_registry_bar_minutes is not None:
                                    if int(overlay_registry_bar_minutes) == int(current_bar_minutes):
                                        logger.warning(
                                            f"Overlay {global_overlay_file} missing interval_minutes but registry_bar_minutes "
                                            f"({overlay_registry_bar_minutes}m) matches current interval ({current_bar_minutes}m). "
                                            f"Loading with caution (overlay may have been created with hardcoded /5 assumption)."
                                        )
                                        # Proceed but mark as potentially unsafe
                                    else:
                                        logger.warning(
                                            f"Overlay {global_overlay_file} missing interval_minutes and registry_bar_minutes "
                                            f"({overlay_registry_bar_minutes}m) doesn't match current interval ({current_bar_minutes}m). "
                                            f"Ignoring overlay (potentially polluted)."
                                        )
                                        global_data = {}  # Ignore incompatible overlay
                                        global_overrides = {}
                                else:
                                    # No interval metadata at all - refuse to load in strict mode
                                    error_policy = "best_effort"  # Safe default
                                    try:
                                        from CONFIG.config_loader import get_cfg
                                        error_policy = get_cfg('registry.error_policy', default='best_effort')
                                    except Exception:
                                        pass
                                    
                                    if error_policy == "strict":
                                        logger.error(
                                            f"Overlay {global_overlay_file} missing interval_minutes and tool_version. "
                                            f"Cannot verify compatibility. Refusing to load (strict mode)."
                                        )
                                        global_data = {}  # Ignore incompatible overlay
                                        global_overrides = {}
                                    else:
                                        logger.warning(
                                            f"Overlay {global_overlay_file} missing interval_minutes. "
                                            f"Potentially polluted (created with hardcoded /5 assumption). Ignoring overlay."
                                        )
                                        global_data = {}  # Ignore incompatible overlay
                                        global_overrides = {}
                            else:
                                # No current_bar_minutes available - can't verify, refuse in strict mode
                                error_policy = "best_effort"  # Safe default
                                try:
                                    from CONFIG.config_loader import get_cfg
                                    error_policy = get_cfg('registry.error_policy', default='best_effort')
                                except Exception:
                                    pass
                                
                                if error_policy == "strict":
                                    logger.error(
                                        f"Overlay {global_overlay_file} missing interval_minutes and current_bar_minutes unknown. "
                                        f"Cannot verify compatibility. Refusing to load (strict mode)."
                                    )
                                    global_data = {}  # Ignore incompatible overlay
                                    global_overrides = {}
                        else:
                            # Post-fix tool version but missing interval - should not happen, but handle gracefully
                            logger.warning(
                                f"Overlay {global_overlay_file} has tool_version={overlay_tool_version} but missing interval_minutes. "
                                f"This should not happen. Ignoring overlay."
                            )
                            global_data = {}  # Ignore incompatible overlay
                            global_overrides = {}
                    
                    # Overlay has interval metadata - enforce compatibility
                    if overlay_interval is not None and current_bar_minutes is not None:
                        if int(overlay_interval) != int(current_bar_minutes):
                            logger.warning(
                                f"Overlay {global_overlay_file} was created for {overlay_interval}-minute bars, "
                                f"but current run uses {current_bar_minutes}-minute bars. "
                                f"Overlay allowed_horizons may be incorrect. Ignoring overlay."
                            )
                            global_data = {}  # Ignore incompatible overlay
                            global_overrides = {}

                    global_overrides = global_data.get('feature_overrides', {})
                    if global_overrides:
                        # Merge global overlay (baseline)
                        self.auto_overlay.update(global_overrides)
                        logger.debug(f"Loaded global auto overlay: {len(global_overrides)} features from {global_overlay_file}")
                except Exception as e:
                    # HARD FAIL: If apply=True and parse fails, fail the run (Tier A safety)
                    if autopatch.apply:
                        error_msg = f"Failed to load global auto overlay {global_overlay_file}: {e}"
                        logger.error(f"❌ CRITICAL: {error_msg}")
                        raise RuntimeError(
                            f"Registry overlay parse failed with apply=True. "
                            f"This is a Tier A safety requirement. Error: {error_msg}"
                        ) from e
                    else:
                        logger.warning(f"Failed to load global auto overlay {global_overlay_file}: {e}")
            
            # Step 2: Load per-target overlay (override layer, if target_column is set)
            per_target_file = None
            if hasattr(self, 'target_column') and self.target_column:
                from TRAINING.common.registry_patch_naming import safe_target_filename_stem
                target_stem = safe_target_filename_stem(self.target_column)
                
                # Try interval-specific path first
                if interval_int is not None:
                    per_target_file = overlay_dir / f"feature_registry_overrides.auto.{target_stem}.{interval_int}m.yaml"
                    if not per_target_file.exists():
                        per_target_file = None  # Fallback to interval-agnostic
                
                # Fallback to interval-agnostic
                if per_target_file is None:
                    per_target_file = overlay_dir / f"feature_registry_overrides.auto.{target_stem}.yaml"

                if per_target_file.exists():
                    try:
                        with open(per_target_file, 'r') as f:
                            per_target_data = yaml.safe_load(f) or {}

                        # Validate overlay structure
                        if not isinstance(per_target_data, dict):
                            raise ValueError(f"Per-target overlay file must contain a dict, got {type(per_target_data)}")

                        # CRITICAL: Check per-target overlay compatibility (same logic as global)
                        per_target_metadata = per_target_data.get('_metadata', {})
                        per_target_interval = per_target_metadata.get('interval_minutes')
                        per_target_tool_version = per_target_metadata.get('tool_version')
                        per_target_registry_bar_minutes = per_target_metadata.get('registry_bar_minutes')
                        current_bar_minutes = getattr(self, 'current_bar_minutes', None)
                        
                        # CRITICAL: Missing overlay interval/tool_version is unsafe (potentially polluted)
                        if per_target_interval is None or per_target_tool_version is None:
                            from TRAINING.common.utils.registry_autopatch import TOOL_VERSION
                            if per_target_tool_version is None or per_target_tool_version != TOOL_VERSION:
                                # Old overlay without interval metadata - potentially polluted
                                if current_bar_minutes is not None:
                                    if per_target_registry_bar_minutes is not None:
                                        if int(per_target_registry_bar_minutes) == int(current_bar_minutes):
                                            logger.warning(
                                                f"Overlay {per_target_file} missing interval_minutes but registry_bar_minutes "
                                                f"({per_target_registry_bar_minutes}m) matches current interval ({current_bar_minutes}m). "
                                                f"Loading with caution (overlay may have been created with hardcoded /5 assumption)."
                                            )
                                        else:
                                            logger.warning(
                                                f"Overlay {per_target_file} missing interval_minutes and registry_bar_minutes "
                                                f"({per_target_registry_bar_minutes}m) doesn't match current interval ({current_bar_minutes}m). "
                                                f"Ignoring overlay (potentially polluted)."
                                            )
                                            per_target_data = {}  # Ignore incompatible overlay
                                    else:
                                        error_policy = "best_effort"
                                        try:
                                            from CONFIG.config_loader import get_cfg
                                            error_policy = get_cfg('registry.error_policy', default='best_effort')
                                        except Exception:
                                            pass
                                        
                                        if error_policy == "strict":
                                            logger.error(
                                                f"Overlay {per_target_file} missing interval_minutes and tool_version. "
                                                f"Cannot verify compatibility. Refusing to load (strict mode)."
                                            )
                                            per_target_data = {}  # Ignore incompatible overlay
                                        else:
                                            logger.warning(
                                                f"Overlay {per_target_file} missing interval_minutes. "
                                                f"Potentially polluted (created with hardcoded /5 assumption). Ignoring overlay."
                                            )
                                            per_target_data = {}  # Ignore incompatible overlay
                                else:
                                    error_policy = "best_effort"
                                    try:
                                        from CONFIG.config_loader import get_cfg
                                        error_policy = get_cfg('registry.error_policy', default='best_effort')
                                    except Exception:
                                        pass
                                    
                                    if error_policy == "strict":
                                        logger.error(
                                            f"Overlay {per_target_file} missing interval_minutes and current_bar_minutes unknown. "
                                            f"Cannot verify compatibility. Refusing to load (strict mode)."
                                        )
                                        per_target_data = {}  # Ignore incompatible overlay
                            else:
                                logger.warning(
                                    f"Overlay {per_target_file} has tool_version={per_target_tool_version} but missing interval_minutes. "
                                    f"This should not happen. Ignoring overlay."
                                )
                                per_target_data = {}  # Ignore incompatible overlay
                        
                        # Per-target overlay has interval metadata - enforce compatibility
                        if per_target_interval is not None and current_bar_minutes is not None:
                            if int(per_target_interval) != int(current_bar_minutes):
                                logger.warning(
                                    f"Overlay {per_target_file} was created for {per_target_interval}-minute bars, "
                                    f"but current run uses {current_bar_minutes}-minute bars. "
                                    f"Overlay allowed_horizons may be incorrect. Ignoring overlay."
                                )
                                per_target_data = {}  # Ignore incompatible overlay

                        per_target_overrides = per_target_data.get('feature_overrides', {})
                        if per_target_overrides:
                            # Merge per-target overlay (overrides global, more specific wins)
                            # Per-target always wins for same feature+field
                            # DETERMINISM: Use sorted iteration for deterministic merge order
                            from TRAINING.common.utils.determinism_ordering import sorted_items
                            for feature_name, feature_patch in sorted_items(per_target_overrides):
                                if feature_name not in self.auto_overlay:
                                    self.auto_overlay[feature_name] = {}
                                # Per-target fields override global fields
                                # DETERMINISM: Use sorted iteration for deterministic update order
                                for field, value in sorted_items(feature_patch):
                                    self.auto_overlay[feature_name][field] = value
                            logger.debug(f"Loaded per-target auto overlay: {len(per_target_overrides)} features from {per_target_file}")
                    except Exception as e:
                        # HARD FAIL: If apply=True and parse fails, fail the run (Tier A safety)
                        if autopatch.apply:
                            error_msg = f"Failed to load per-target auto overlay {per_target_file}: {e}"
                            logger.error(f"❌ CRITICAL: {error_msg}")
                            raise RuntimeError(
                                f"Registry overlay parse failed with apply=True. "
                                f"This is a Tier A safety requirement. Error: {error_msg}"
                            ) from e
                        else:
                            logger.warning(f"Failed to load per-target auto overlay {per_target_file}: {e}")
            
            # Store selected overlay paths for provenance (set once at end, deterministic)
            # This is the final outcome after all fallback logic completes
            self._overlay_paths_selected = {
                "global": str(global_overlay_file) if global_overlay_file and global_overlay_file.exists() else None,
                "per_target": str(per_target_file) if per_target_file and per_target_file.exists() else None,
                "interval_int": interval_int  # Canonicalized interval (None if not effectively integer)
            }
            
            # Mark as loaded if any overlay was loaded
            if self.auto_overlay:
                self._overlay_loaded = True
                logger.info(f"Loaded auto overlay(s): {len(self.auto_overlay)} total features (global + per-target)")
                
                # Validate overlay structure (if loaded)
                for feature_name, feature_patch in self.auto_overlay.items():
                    if not isinstance(feature_patch, dict):
                        raise ValueError(f"Overlay feature {feature_name} must be a dict, got {type(feature_patch)}")
                
                # Compute overlay fingerprint (three-layer hashing for layer attribution)
                from TRAINING.common.utils.registry_overlay_fingerprint import hash_overrides
                
                global_hash = None
                per_target_hash = None
                effective_hash = None
                sources = []
                
                # Hash global overlay (if loaded)
                if global_overlay_file.exists() and global_overrides:
                    global_hash = hash_overrides(global_overrides)
                    sources.append({
                        "path": str(global_overlay_file.relative_to(overlay_dir)),
                        "hash": global_hash,
                        "exists": True,
                        "n_features": len(global_overrides)
                    })
                
                # Hash per-target overlay (if loaded)
                if hasattr(self, 'target_column') and self.target_column:
                    from TRAINING.common.registry_patch_naming import safe_target_filename_stem
                    target_stem = safe_target_filename_stem(self.target_column)
                    per_target_file = overlay_dir / f"feature_registry_overrides.auto.{target_stem}.yaml"
                    if per_target_file.exists() and per_target_overrides:
                        per_target_hash = hash_overrides(per_target_overrides)
                        sources.append({
                            "path": str(per_target_file.relative_to(overlay_dir)),
                            "hash": per_target_hash,
                            "exists": True,
                            "n_features": len(per_target_overrides)
                        })
                
                # Hash effective overlay (merged result - what's actually applied)
                if self.auto_overlay:
                    effective_hash = hash_overrides(self.auto_overlay)
                
                # Store fingerprint for run identity
                self._overlay_fingerprint = {
                    "global_hash": global_hash,
                    "per_target_hash": per_target_hash,
                    "effective_hash": effective_hash,
                    "sources": sources
                }
                
                logger.debug(
                    f"Computed overlay fingerprint: global={global_hash[:16] if global_hash else 'None'}..., "
                    f"per_target={per_target_hash[:16] if per_target_hash else 'None'}..., "
                    f"effective={effective_hash[:16] if effective_hash else 'None'}..."
                )
            else:
                # No overlays loaded → overlay_loaded=False (not an error)
                self._overlay_loaded = False
                self._overlay_fingerprint = None
        except RuntimeError:
            # Re-raise hard-fail errors
            raise
        except Exception as e:
            # If autopatch not available, log and continue (not an error)
            logger.debug(f"Auto overlay not available: {e}")
            self._overlay_loaded = False
    
    def get_overlay_loaded_status(self) -> bool:
        """
        Get overlay loaded status (whether overlay was successfully applied).
        
        Returns:
            True if overlay was successfully loaded and applied, False otherwise
        """
        return getattr(self, '_overlay_loaded', False)
    
    def get_overlay_fingerprint(self) -> Optional[Dict[str, Any]]:
        """
        Get overlay fingerprint (for run identity and audit).
        
        Returns:
            Dict with keys: global_hash, per_target_hash, effective_hash, sources
            or None if no overlay loaded
        """
        return getattr(self, '_overlay_fingerprint', None)
    
    def get_overlay_content_hash(self) -> Optional[str]:
        """
        Get hash of effective overlay content (for run identity).
        
        Returns:
            SHA256 hash of merged overlay content, or None if no overlay loaded
        """
        fingerprint = self.get_overlay_fingerprint()
        return fingerprint.get('effective_hash') if fingerprint else None
    
    def _load_config(self, strict: bool = False) -> Dict[str, Any]:
        """Load feature registry from YAML file."""
        global _LOGGED_REGISTRY_PATHS

        if not self.config_path.exists():
            # Try fallback: CWD/CONFIG/data/feature_registry.yaml (canonical)
            cwd_config = Path.cwd() / "CONFIG" / "data" / "feature_registry.yaml"
            if cwd_config.exists():
                # TS-002: Protect _LOGGED_REGISTRY_PATHS with lock
                # Only log once per unique path to avoid duplicate messages
                cwd_config_str = str(cwd_config)
                with _REGISTRY_LOCK:
                    if cwd_config_str not in _LOGGED_REGISTRY_PATHS:
                        logger.info(f"Using feature registry from CWD: {cwd_config}")
                        _LOGGED_REGISTRY_PATHS.add(cwd_config_str)
                self.config_path = cwd_config
            else:
                logger.warning(
                    f"Feature registry not found at {self.config_path} or {cwd_config}. "
                    f"Using empty registry (all features will be auto-inferred)."
                )
                return {
                    'features': {},
                    'feature_families': {},
                    'validation': {
                        'hard_rules': [],
                        'warnings': []
                    }
                }
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required keys exist
            config.setdefault('features', {})
            config.setdefault('feature_families', {})
            config.setdefault('validation', {
                'hard_rules': [],
                'warnings': []
            })
            
            return config
        except Exception as e:
            if strict:
                from TRAINING.common.exceptions import RegistryLoadError
                raise RegistryLoadError(
                    message=f"Failed to load feature registry from {self.config_path}: {e}",
                    registry_path=str(self.config_path),
                    stage="TARGET_RANKING",
                    error_code="REGISTRY_LOAD_FAILED"
                ) from e
            logger.error(f"Failed to load feature registry from {self.config_path}: {e}")
            return {
                'features': {},
                'feature_families': {},
                'validation': {
                    'hard_rules': [],
                    'warnings': []
                }
            }

    def _build_alias_lookup(self) -> Dict[str, str]:
        """
        Phase 19: Build alias-to-canonical lookup from config.

        Aliases allow interval-agnostic feature naming. Example config:

            feature_aliases:
              ret_short:
                canonical: ret_5m
                description: Short-term return (5m at reference interval)
              vol_short:
                canonical: vol_5m

        Features with `aliases` field in their metadata also contribute:

            features:
              ret_5m:
                lag_bars: 1
                aliases: [ret_short, return_5min]

        Returns:
            Dict mapping alias name to canonical feature name
        """
        aliases = {}

        # Load from top-level feature_aliases section
        alias_config = self.config.get('feature_aliases', {})
        for alias_name, alias_def in alias_config.items():
            if isinstance(alias_def, dict):
                canonical = alias_def.get('canonical')
                if canonical:
                    aliases[alias_name] = canonical
            elif isinstance(alias_def, str):
                # Simple format: alias: canonical
                aliases[alias_name] = alias_def

        # Load from individual feature metadata (aliases field)
        for feature_name, metadata in self.features.items():
            if isinstance(metadata, dict):
                feature_aliases = metadata.get('aliases', [])
                if isinstance(feature_aliases, list):
                    for alias in feature_aliases:
                        if isinstance(alias, str):
                            aliases[alias] = feature_name

        if aliases:
            logger.debug(f"Loaded {len(aliases)} feature aliases")

        return aliases

    def resolve_alias(self, name: str) -> str:
        """
        Phase 19: Resolve feature alias to canonical name.

        If name is an alias, returns the canonical name.
        If name is not an alias, returns the name unchanged.

        Args:
            name: Feature name or alias

        Returns:
            Canonical feature name
        """
        return self._aliases.get(name, name)

    def get_all_aliases(self) -> Dict[str, str]:
        """
        Phase 19: Get all registered aliases.

        Returns:
            Dict mapping alias names to canonical feature names
        """
        return dict(self._aliases)

    def is_alias(self, name: str) -> bool:
        """
        Phase 19: Check if name is a registered alias.

        Args:
            name: Feature name to check

        Returns:
            True if name is an alias, False otherwise
        """
        return name in self._aliases

    def _validate_registry(self):
        """Validate all features against hard rules."""
        errors = []
        warnings = []
        
        # DETERMINISM: Sort feature names for deterministic validation order
        from TRAINING.common.utils.determinism_ordering import sorted_keys
        for name in sorted_keys(self.features):
            metadata = self.features[name]
            try:
                self._validate_feature(name, metadata)
            except ValueError as e:
                errors.append(str(e))
            except Warning as w:
                warnings.append(str(w))
        
        if errors:
            raise ValueError(f"Feature registry validation failed:\n" + "\n".join(errors))
        
        if warnings:
            for w in warnings:
                logger.warning(w)
    
    def _validate_feature(self, name: str, metadata: Dict[str, Any]):
        """
        Validate a single feature against hard rules.
        
        Raises:
            ValueError: If hard rule violated
            Warning: If soft rule violated
        """
        lag_bars = metadata.get('lag_bars', 0)
        allowed_horizons = metadata.get('allowed_horizons', [])
        source = metadata.get('source', 'unknown')
        rejected = metadata.get('rejected', False)
        
        # Skip validation for explicitly rejected features (they're documented as leaky)
        if rejected:
            return
        
        # Hard rule: Cannot look into future (for non-rejected features)
        if lag_bars < 0:
            raise ValueError(
                f"Feature '{name}': lag_bars={lag_bars} < 0 "
                f"(looks into future - structurally impossible). "
                f"Mark as rejected: true if this is intentional."
            )
        
        # Hard rule: For price/derived features, lag must be >= 0 (can't look into future)
        # Note: A feature with lag_bars=1 CAN predict a 3-bar horizon target.
        # The feature just needs to be from the past, not necessarily lag by the full horizon.
        # The actual temporal safety is enforced by PurgedTimeSeriesSplit (purge gap = horizon).
        if source in ['price', 'derived']:
            # Only check that lag is non-negative
            if lag_bars < 0:
                raise ValueError(
                    f"Feature '{name}': lag_bars={lag_bars} < 0 "
                    f"(cannot look into future - structurally impossible)"
                )
        
        # Hard rule: Usable features must have allowed horizons
        # Use explicit check: None (unknown/inherit) vs [] (explicit none)
        # Note: Empty allowed_horizons is intentional for disabled features - use DEBUG to reduce spam
        if allowed_horizons is None:
            logger.debug(
                f"Feature '{name}': allowed_horizons is None (unknown/inherit). "
                f"Will be rejected unless family defaults apply. "
                f"Mark as rejected: true if this is intentional."
            )
        elif allowed_horizons == []:
            # Empty list is intentional for disabled features - this is expected behavior, not a warning
            logger.debug(
                f"Feature '{name}': allowed_horizons is [] (explicit none, intentionally disabled). "
                f"Will be rejected by default - safe."
            )
    
    def is_allowed(
        self,
        feature_name: str,
        target_horizon: int,
        target_column: Optional[str] = None  # NEW: for per-target checks
    ) -> bool:
        """
        Check if feature is allowed for a target horizon.
        
        **CRITICAL**: This method does NOT change feature order.
        It's called per-feature by get_allowed_features() which preserves input order.
        
        Two-phase check:
        Phase A (base eligibility - hard gate):
        - Global rejected: true → False
        - Base allowed_horizons check → False if not in list
        - Unknown feature policy → False if rejected
        
        Phase B (overlays - soft policy):
        - Overlay/override excludes → False
        - Unblock cancels overlay deny → True (only if Phase A passed)
        
        Unblocks CANNOT override base eligibility (Phase A).
        
        Args:
            feature_name: Name of the feature
            target_horizon: Target horizon in bars (e.g., 12 for 60-minute target with 5m bars)
            target_column: Optional target column name (for per-target patch checks)
        
        Returns:
            True if feature is allowed, False otherwise
        """
        # ========================================================================
        # PHASE A: Base Eligibility (Hard Gate)
        # ========================================================================
        
        # 1. Global rejected (structural leaks)
        if feature_name in self.features:
            if self.features[feature_name].get('rejected', False):
                return False
        
        # 2. Base allowed_horizons check
        # CRITICAL: Use effective metadata to get family defaults merged (schema-consistent)
        # This ensures None (inherit) vs [] (explicit none) are handled correctly
        phase_a_base_eligible = False
        if feature_name in self.features:
            # Use effective metadata to get family defaults for None/missing
            effective_metadata = self.get_feature_metadata_effective(feature_name, resolve_defaults=True)
            allowed_horizons = effective_metadata.get('allowed_horizons')
            
            # Explicit checks: None (unknown/inherit not resolved) vs [] (explicit none)
            if allowed_horizons is None:
                # Unknown/inherit not resolved - treat as not allowed (conservative)
                # Note: This could inherit from family, but if family matching failed, be conservative
                return False
            elif allowed_horizons == []:
                # Explicit empty list = disallowed
                return False
            elif allowed_horizons:
                # Check if horizon is in allowed list
                if target_horizon not in allowed_horizons:
                    return False
                # Feature is eligible in base registry (Phase A passed)
                phase_a_base_eligible = True
            else:
                # Should not happen, but defensive
                return False
        else:
            # Check feature families (pattern matching)
            # DETERMINISM_CRITICAL: Preserve YAML order (same as _match_feature_family)
            for family_name, family_config in self.families.items():
                pattern = family_config.get('pattern')
                if pattern and re.match(pattern, feature_name):
                    # Rejected families
                    if family_name.startswith('rejected_') or family_config.get('rejected', False):
                        return False
                    
                    # Allowed families - check default horizons
                    default_horizons = family_config.get('default_allowed_horizons', [])
                    if target_horizon in default_horizons:
                        phase_a_base_eligible = True
                        break
            
            # Unknown feature: auto-infer or reject (safe default)
            if not phase_a_base_eligible:
                inferred = self.auto_infer_metadata(feature_name)
                if inferred.get('rejected', False):
                    return False
                allowed_horizons = inferred.get('allowed_horizons', [])
                phase_a_base_eligible = target_horizon in allowed_horizons
        
        # If base eligibility fails, unblocks cannot save it
        if not phase_a_base_eligible:
            return False
        
        # ========================================================================
        # PHASE B: Overlays (Soft Policy)
        # ========================================================================
        
        # 3. Check if unblocked (cancels overlay denies, but only if Phase A passed)
        phase_b_overlay_allowed = True  # Default: allowed if no overlay denies
        if target_column and self.per_target_unblocks:
            unblock_features = self.per_target_unblocks.get('features', {})
            if feature_name in unblock_features:
                unblocked = unblock_features[feature_name].get('unblocked_horizons_bars', [])
                if target_horizon in unblocked:
                    # Unblock cancels overlay denies (but base eligibility already checked)
                    # Skip overlay deny checks if unblocked
                    return True
        
        # 4. Run patch excludes (highest priority deny)
        if target_column and self.per_target_patches:
            patch_features = self.per_target_patches.get('features', {})
            if feature_name in patch_features:
                excluded = patch_features[feature_name].get('excluded_horizons_bars', [])
                if target_horizon in excluded:
                    phase_b_overlay_allowed = False
        
        # 5. Persistent override excludes (medium priority deny)
        if target_column and self.per_target_overrides:
            override_features = self.per_target_overrides.get('features', {})
            if feature_name in override_features:
                excluded = override_features[feature_name].get('excluded_horizons_bars', [])
                if target_horizon in excluded:
                    phase_b_overlay_allowed = False
        
        # Phase A passed, Phase B check → allowed if overlay_allowed
        return phase_b_overlay_allowed
    
    def get_allowed_features(
        self, 
        all_features: List[str], 
        target_horizon: int,
        verbose: bool = False,
        target_column: Optional[str] = None  # NEW: for per-target checks
    ) -> List[str]:
        """
        Get list of allowed features for a target horizon.
        
        **CRITICAL**: Preserves input order from all_features.
        Final sorting happens in filter_features_for_target() (line 952).
        Do NOT sort here - it would break X array column alignment.
        
        Args:
            all_features: List of all feature names
            target_horizon: Target horizon in bars
            verbose: If True, log excluded features
            target_column: Optional target column name (for per-target patch checks)
        
        Returns:
            List of allowed feature names (NOT SORTED - caller handles sorting)
        """
        allowed = []
        excluded = []
        
        # PRESERVE INPUT ORDER: Iterate through all_features in order
        for feature in all_features:
            if self.is_allowed(feature, target_horizon, target_column=target_column):
                allowed.append(feature)  # Preserves order
            else:
                excluded.append(feature)
        
        if verbose and excluded:
            logger.info(
                f"Feature registry: Allowed {len(allowed)} features, "
                f"excluded {len(excluded)} features for horizon={target_horizon}"
            )
            if len(excluded) <= 10:
                logger.debug(f"  Excluded: {', '.join(excluded)}")
            else:
                logger.debug(f"  Excluded: {', '.join(excluded[:10])}... ({len(excluded)} total)")
        
        return allowed
    
    def auto_infer_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Auto-infer metadata for unknown features (backward compatibility).
        
        Uses pattern matching from CONFIG/data/feature_inference.yaml with explicit precedence.
        Falls back to hardcoded patterns if config not available (defensive boundary).
        
        DETERMINISM: Patterns matched in priority order (highest priority wins, ties break lexicographically).
        
        Returns:
            Metadata dictionary with inferred values
        """
        # Try to load feature inference config
        inference_config = _load_feature_inference_config()
        
        if inference_config:
            # Use config-based matching with precedence
            patterns_list = _build_pattern_list_with_precedence(inference_config)
            
            # Match in precedence order (first match wins)
            for priority, pattern_str, pattern_config in patterns_list:
                match = re.match(pattern_str, feature_name, re.I)
                if match:
                    # Build metadata from pattern config
                    metadata = self._build_metadata_from_pattern(feature_name, match, pattern_config)
                    if metadata:
                        return metadata
        
        # Fallback to hardcoded patterns (defensive boundary - config unavailable or no match)
        return self._auto_infer_metadata_fallback(feature_name)
    
    def _build_metadata_from_pattern(self, feature_name: str, match: re.Match, pattern_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build metadata dict from pattern match and config.
        
        Args:
            feature_name: Feature name being matched
            match: Regex match object
            pattern_config: Pattern configuration dict
        
        Returns:
            Metadata dict, or None if pattern doesn't provide enough info
        """
        metadata = {
            'source': pattern_config.get('source', 'derived'),
            'description': pattern_config.get('description', f"Auto-inferred from pattern")
        }
        
        # Handle rejected patterns
        if pattern_config.get('rejected', False):
            metadata['rejected'] = True
            metadata['lag_bars'] = pattern_config.get('lag_bars', 0)
            metadata['allowed_horizons'] = pattern_config.get('allowed_horizons', [])
            return metadata
        
        # Handle lagged returns (extract lag from pattern group)
        if 'allowed_horizons_multipliers' in pattern_config:
            # Pattern like "^ret_(\\d+)$" - extract lag from group 1
            try:
                lag = int(match.group(1))
                multipliers = pattern_config['allowed_horizons_multipliers']
                metadata['lag_bars'] = lag
                metadata['allowed_horizons'] = [lag * m for m in multipliers] if lag > 0 else []
                return metadata
            except (IndexError, ValueError):
                pass
        
        # Handle patterns with group_idx (extract lookback from specific group)
        group_idx = pattern_config.get('group_idx')
        if group_idx:
            try:
                lookback = int(match.group(group_idx))
                metadata['lag_bars'] = lookback
                default_horizons = pattern_config.get('default_allowed_horizons', [1, 3, 5, 12, 24, 60])
                metadata['allowed_horizons'] = default_horizons if lookback > 0 else []
                return metadata
            except (IndexError, ValueError):
                pass
        
        # Handle exact matches (metadata columns)
        if 'exact_match' in pattern_config:
            metadata['lag_bars'] = pattern_config.get('lag_bars', 0)
            metadata['allowed_horizons'] = pattern_config.get('allowed_horizons', [])
            metadata['rejected'] = pattern_config.get('rejected', True)
            return metadata
        
        # Pattern matched but config incomplete - return None to fall through to fallback
        return None
    
    def _auto_infer_metadata_fallback(self, feature_name: str) -> Dict[str, Any]:
        """
        Hardcoded fallback patterns (used when config unavailable or pattern not in config).
        
        This is the defensive boundary - ensures inference works even if config missing.
        """
        # Lagged returns: ret_N where N is the lag
        ret_match = re.match(r"^ret_(\d+)$", feature_name)
        if ret_match:
            lag = int(ret_match.group(1))
            return {
                'source': 'price',
                'lag_bars': lag,
                'allowed_horizons': [lag, lag*3, lag*5, lag*12] if lag > 0 else [],
                'description': f"Auto-inferred: {lag}-bar lagged return"
            }
        
        # Forward returns (leaky): ret_future_N or fwd_ret_N
        if re.match(r"^(ret_future_|fwd_ret_)", feature_name):
            return {
                'source': 'price',
                'lag_bars': -1,  # Negative = looks into future
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - forward return (leaky)'
            }
        
        # Technical indicators with lookback: rsi_N, sma_N, ema_N, cci_N, stoch_k_N, etc.
        simple_patterns = [
            (r'^(stoch_d|stoch_k|williams_r)_(\d+)$', 2),
            (r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var)_(\d+)$', 2),
            (r'^(ret|sma|ema|vol)_(\d+)$', 2),
        ]
        for pattern, group_idx in simple_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                indicator_type = match.group(1)
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: {indicator_type.upper()} with {lookback}-bar lookback"
                }
        
        # Compound indicator patterns
        compound_patterns = [
            (r'^bb_(upper|lower|width|percent_b|middle)_(\d+)$', 2),
            (r'^macd_(signal|hist|diff)_(\d+)$', 2),
            (r'^(stoch_k|stoch_d|rsi|cci|mfi|atr|adx|mom|std|var)_(fast|slow|wilder|smooth|upper|lower|width)_(\d+)$', 3),
        ]
        for pattern, group_idx in compound_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                indicator_type = match.group(1)
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: {indicator_type.upper()} (compound) with {lookback}-bar lookback"
                }
        
        # Volume/volatility features
        vol_patterns = [
            (r'^volume_(ema|sma)_(\d+)$', 2),
            (r'^realized_vol_(\d+)$', 1),
            (r'^vol_(ema|sma|std)_(\d+)$', 2),
        ]
        for pattern, group_idx in vol_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: volume/volatility feature with {lookback}-bar lookback"
                }
        
        # Rejection patterns (leaky features)
        rejection_patterns = [
            (r"^tth_", 'derived', 'time-to-hit requires future path'),
            (r"^(mfe|mdd)_", 'derived', 'MFE/MDD requires future path'),
            (r"^barrier_", 'derived', 'barrier features encode barrier logic'),
            (r"^(y_|target_)", 'target', 'target column (leaky)'),
            (r"^p_", 'derived', 'prediction/probability feature (leaky)'),
        ]
        for pattern, source, desc in rejection_patterns:
            if re.match(pattern, feature_name):
                return {
                    'source': source,
                    'lag_bars': 0,
                    'allowed_horizons': [],
                    'rejected': True,
                    'description': f'Auto-inferred: REJECTED - {desc}'
                }
        
        # Timestamp/metadata columns
        if feature_name in ['ts', 'timestamp', 'symbol', 'time']:
            return {
                'source': 'metadata',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - metadata column'
            }
        
        # Unknown feature: reject by default (safe)
        logger.debug(
            f"Unknown feature '{feature_name}': rejecting by default (safe). "
            f"Add to feature_registry.yaml to allow."
        )
        return {
            'source': 'unknown',
            'lag_bars': 0,
            'allowed_horizons': [],
            'rejected': True,
            'description': 'Auto-inferred: REJECTED - unknown feature (safe default)'
        }
    
    def get_feature_metadata_raw(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata exactly as stored in registry (no defaults merged, but with overlays applied).

        Note: Auto overlay is only applied when `autopatch.apply=True`. When `apply=False`,
        overlay is not loaded and this method returns base registry metadata only.

        Phase 19: Supports aliases - if feature_name is an alias, resolves to canonical name.

        Use for diagnostics and provenance tracing.

        Args:
            feature_name: Name of the feature (or alias)

        Returns:
            Metadata dict exactly as stored in registry or auto-inferred, with overlays applied
        """
        # Phase 19: Resolve alias to canonical name first
        canonical_name = self.resolve_alias(feature_name)
        lookup_name = canonical_name

        # Start with base registry or auto-inferred
        if lookup_name in self.features:
            metadata = self.features[lookup_name].copy()
        else:
            metadata = self.auto_infer_metadata(lookup_name)

        # Track if we resolved via alias (for provenance)
        if canonical_name != feature_name:
            metadata['_resolved_from_alias'] = feature_name

        # Apply auto overlay (last, but should not override explicit values unless allow_overwrite)
        # Check both canonical name and original name for overlay
        overlay_patch = None
        if self.auto_overlay:
            if lookup_name in self.auto_overlay:
                overlay_patch = self.auto_overlay[lookup_name]
            elif feature_name in self.auto_overlay:
                overlay_patch = self.auto_overlay[feature_name]

        if overlay_patch:
            
            # Protected fields that can NEVER be overridden by overlay (hard-stop safety)
            PROTECTED_FIELDS = {"rejected"}  # Add more if needed (e.g., "source" for provenance)
            
            # Optional: Allowlist of fields overlays can touch (even in overwrite mode)
            # Prevents future overlay schema creep from accidentally overriding sensitive fields
            # Minimal allowlist for current use case: just allowed_horizons
            # Set to None to allow all (except protected), or set to {"allowed_horizons", "lag_bars", ...}
            ALLOWED_OVERLAY_FIELDS = None
            
            # DETERMINISM_CRITICAL: Sort fields for deterministic merge order
            from TRAINING.common.utils.determinism_ordering import sorted_keys
            for field in sorted_keys(overlay_patch):
                value = overlay_patch[field]
                
                # Skip all private keys (provenance/metadata fields)
                if field.startswith('_'):
                    continue
                
                # Skip protected fields (hard-stop safety, regardless of allow_overwrite)
                if field in PROTECTED_FIELDS:
                    continue
                
                # Optional: Enforce allowlist if set (prevents schema creep)
                if ALLOWED_OVERLAY_FIELDS is not None and field not in ALLOWED_OVERLAY_FIELDS:
                    continue
                
                # Skip None values (don't replace real metadata with None)
                if value is None:
                    continue
                
                # Toggleable behavior based on allow_overwrite
                if self.allow_overwrite:
                    # Overwrite mode: apply overlay value (except protected fields)
                    metadata[field] = value
                else:
                    # Safe mode: only fill missing/None fields
                    if field not in metadata or metadata[field] is None:
                        metadata[field] = value
        
        return metadata
    
    def _match_feature_family(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Match feature to family pattern (deterministic).
        
        Returns:
            Family metadata dict or None if no match
        
        Determinism:
        - Families evaluated in stable order (YAML list order preserved)
        - If multiple patterns match, select by priority (if defined) or first match (YAML order)
        - Log which family matched for provenance
        
        Note: If priority-based precedence is desired, sort by (-priority, family_name).
        If YAML order precedence is desired, iterate in stored order (no sorting).
        """
        # DETERMINISM_CRITICAL: Preserve YAML list order (author-defined precedence)
        # Option A: YAML order precedence (preserve stored order)
        matched_families = []
        for family_name, family_meta in self.families.items():
            pattern = family_meta.get('pattern')
            if pattern and re.match(pattern, feature_name, re.I):
                priority = family_meta.get('priority', 0)  # Higher = more specific
                matched_families.append((priority, family_name, family_meta))
        
        if not matched_families:
            return None
        
        # Option B: Priority-based precedence (if priority field is used)
        # Sort by priority (descending), then by name for ties
        # matched_families.sort(key=lambda x: (-x[0], x[1]))
        
        # For YAML order: return first match (preserves author-defined precedence)
        # For priority order: return highest priority match
        _, family_name, family_meta = matched_families[0]
        logger.debug(f"Feature '{feature_name}' matched family '{family_name}' (priority={matched_families[0][0]})")
        
        return family_meta
    
    def get_feature_metadata_effective(
        self, 
        feature_name: str, 
        resolve_defaults: bool = True
    ) -> Dict[str, Any]:
        """
        Get metadata with family defaults merged (for coverage/gating).
        
        Args:
            feature_name: Feature name
            resolve_defaults: If True, merge family defaults for null/missing fields
        
        Returns:
            Metadata dict with defaults merged (if resolve_defaults=True)
        """
        # Shallow copy + discipline (no deepcopy needed if we never mutate nested structures)
        metadata = dict(self.get_feature_metadata_raw(feature_name))
        
        if not resolve_defaults:
            return metadata
        
        # Config loaded in __init__(), cached as self._auto_enable_family_features
        
        # Simplify None check (get() already returns None when missing)
        if metadata.get('allowed_horizons') is None:
            family_metadata = self._match_feature_family(feature_name)
            if family_metadata and family_metadata.get('default_allowed_horizons'):
                # Clone list to avoid aliasing (never mutate nested structures)
                metadata['allowed_horizons'] = list(family_metadata['default_allowed_horizons'])
                metadata['_allowed_horizons_source'] = 'family_default'
        elif metadata.get('allowed_horizons') == []:
            # Auto-enable: Treat empty list as None for eligible features (when flag enabled)
            if self._auto_enable_family_features and not metadata.get('rejected', False):
                family_metadata = self._match_feature_family(feature_name)
                lag_bars = metadata.get('lag_bars')
                
                # Integer-ish validation (not float)
                lag_bars_valid = (
                    lag_bars is not None and 
                    isinstance(lag_bars, (numbers.Integral, int)) and 
                    lag_bars >= 0
                )
                
                # Eligibility checks
                if (family_metadata and 
                    family_metadata.get('default_allowed_horizons') and
                    lag_bars_valid):
                    # Get family_name from families dict (stable key, not regex)
                    # CRITICAL FIX #4: More efficient lookup - iterate once, verify match
                    # NOTE: Could optimize further by modifying _match_feature_family to return (name, meta) tuple
                    family_name = None
                    for fam_name, fam_meta in self.families.items():
                        if fam_meta is family_metadata:  # Same object reference
                            family_name = fam_name
                            break
                    
                    # Defensive check: if not found, log warning (shouldn't happen but safer)
                    if family_name is None:
                        logger.warning(
                            f"Could not find family_name for feature '{feature_name}' "
                            f"(matched family_metadata but not found in self.families). "
                            f"Using 'unknown' as family_id."
                        )
                    
                    inherited_horizons = list(family_metadata['default_allowed_horizons'])  # Clone list
                    
                    # Apply inheritance immediately (no "fall through")
                    metadata['allowed_horizons'] = inherited_horizons
                    metadata['_allowed_horizons_source'] = 'family_default_auto_enabled'
                    
                    # De-dupe tracking (use dict keyed by feature name)
                    # CRITICAL FIX #3: Remove redundant hasattr check (initialized in __init__)
                    # DETERMINISM: First write wins (idempotent). Feature metadata should be identical
                    # across targets, so which target processes it first doesn't matter for correctness.
                    # Target processing order is deterministic (sorted), so first write is deterministic.
                    # TS-003: Thread-safe check-then-add pattern
                    with _AUTO_ENABLED_FEATURES_LOCK:
                        if feature_name not in self._auto_enabled_features:
                            self._auto_enabled_features[feature_name] = {
                                'feature_name': feature_name,
                                'matched_family_id': family_name or 'unknown',  # Stable string key (YAML dict key)
                                'inherited_horizons': inherited_horizons,
                                'lag_bars': lag_bars
                            }
                    # Note: If same feature processed with different metadata (shouldn't happen),
                    # first write wins. This is deterministic if target order is deterministic.
            # If not eligible or flag disabled, keep disabled (explicit empty list)
        
        return metadata
    
    def get_auto_enable_audit(self) -> Dict[str, Any]:
        """
        Get deterministic audit payload for auto-enabled features.

        Returns:
            Dict with sorted features, counts, and metadata (ready for JSON serialization)
        """
        # TS-003: Thread-safe read of auto-enabled features
        # Take a snapshot under lock to avoid iteration over changing dict
        with _AUTO_ENABLED_FEATURES_LOCK:
            features_snapshot = dict(self._auto_enabled_features)

        # CRITICAL FIX #1: Use module-level dict (always exists, shared across instances)
        if not features_snapshot:
            return {
                'config_flag': self._auto_enable_family_features,
                'n_features_enabled': 0,
                'n_total_registry_features': len(self.features),
                'enabled_percent': 0.0,
                'threshold_warning_triggered': False,
                'enabled_features': [],
                'enabled_features_hash': '',
                'registry_path': str(self.config_path)
            }

        # Sort features alphabetically for determinism
        sorted_features = sorted(features_snapshot.items())
        enabled_list = [entry for _, entry in sorted_features]
        
        n_enabled = len(enabled_list)
        n_total = len(self.features)
        enabled_percent = (n_enabled / n_total * 100.0) if n_total > 0 else 0.0
        
        # Compute threshold warning (min of count and percent-based limit)
        threshold_count = self._auto_enable_threshold_count
        threshold_percent = self._auto_enable_threshold_percent
        warn_limit = min(threshold_count, int((n_total * threshold_percent / 100.0) + 0.5))  # ceil via +0.5
        threshold_warning_triggered = n_enabled > warn_limit
        
        # Hash includes horizons + family_id (not just names) for drift detection
        # CRITICAL FIX #2: json and hashlib already imported at top
        
        # Create stable hash payload: (feature_name, matched_family_id, sorted(horizons), lag_bars)
        hash_payload = []
        for entry in enabled_list:
            hash_payload.append((
                entry['feature_name'],
                entry['matched_family_id'],
                tuple(sorted(entry['inherited_horizons'])),
                entry['lag_bars']
            ))
        
        # Sort hash payload for determinism
        hash_payload_sorted = sorted(hash_payload)
        hash_str = json.dumps(hash_payload_sorted, sort_keys=True)
        hash_digest = hashlib.sha256(hash_str.encode()).hexdigest()
        
        return {
            'config_flag': self._auto_enable_family_features,
            'n_features_enabled': n_enabled,
            'n_total_registry_features': n_total,
            'enabled_percent': round(enabled_percent, 2),
            'threshold_warning_triggered': threshold_warning_triggered,
            'threshold_count': threshold_count,
            'threshold_percent': threshold_percent,
            'enabled_features': enabled_list,
            'enabled_features_hash': hash_digest,
            'registry_path': str(self.config_path)
        }
    
    def get_auto_enable_hash(self) -> str:
        """
        Get stable hash of auto-enabled features (for change detection).
        
        Returns:
            SHA256 hash string (64 chars)
        """
        audit = self.get_auto_enable_audit()
        return audit.get('enabled_features_hash', '')
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a feature (backward compatible - returns raw).
        
        For coverage/gating, use get_feature_metadata_effective() instead.
        """
        return self.get_feature_metadata_raw(feature_name)
    
    def trace_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Trace feature metadata resolution with full provenance.
        
        Returns:
            {
                'resolved_metadata': {...},
                'provenance': {
                    'lookup_key': str,
                    'found_in_registry': bool,
                    'source_file': str,
                    'overlay_applied': Optional[str],
                    'family_match': Optional[str],
                    'inferred': bool,
                    'final_lag_bars': int,
                    'final_allowed_horizons': List[int],
                    'allowed_horizons_source': str  # 'explicit', 'family_default', 'inferred'
                }
            }
        """
        provenance = {
            'lookup_key': feature_name,
            'found_in_registry': feature_name in self.features,
            'source_file': str(self.config_path),
            'overlay_applied': None,
            'family_match': None,
            'inferred': False,
            'final_lag_bars': None,
            'final_allowed_horizons': None,
            'allowed_horizons_source': 'unknown'
        }
        
        # Check for patches/overrides
        if self.per_target_patches.get('features', {}).get(feature_name):
            provenance['overlay_applied'] = 'per_target_patch'
        elif self.per_target_overrides.get('features', {}).get(feature_name):
            provenance['overlay_applied'] = 'per_target_override'
        
        # Get raw metadata
        if feature_name in self.features:
            raw_metadata = self.features[feature_name].copy()
            provenance['allowed_horizons_source'] = 'explicit'
        else:
            raw_metadata = self.auto_infer_metadata(feature_name)
            provenance['inferred'] = True
            provenance['allowed_horizons_source'] = 'inferred'
        
        # Check family match
        family_meta = self._match_feature_family(feature_name)
        if family_meta:
            provenance['family_match'] = family_meta.get('description', 'unknown')
            # If allowed_horizons is None/missing, would inherit from family
            if raw_metadata.get('allowed_horizons') is None:
                if family_meta.get('default_allowed_horizons'):
                    raw_metadata['allowed_horizons'] = family_meta['default_allowed_horizons']
                    provenance['allowed_horizons_source'] = 'family_default'
        
        provenance['final_lag_bars'] = raw_metadata.get('lag_bars')
        provenance['final_allowed_horizons'] = raw_metadata.get('allowed_horizons', [])
        
        return {
            'resolved_metadata': raw_metadata,
            'provenance': provenance
        }

    # =========================================================================
    # Interval-Agnostic Lookback Methods (v2 schema support)
    # =========================================================================

    def get_schema_version(self) -> int:
        """
        Get the feature registry schema version.

        Returns:
            1 for legacy (lag_bars only)
            2 for interval-agnostic (lookback_minutes + lag_bars)
        """
        return self.config.get('metadata', {}).get('schema_version', 1)

    def get_lookback_minutes(
        self,
        feature_name: str,
        interval_minutes: Optional[int] = None,
        default_interval_minutes: int = 5
    ) -> Optional[float]:
        """
        Get feature lookback in minutes (v2 schema, with v1 fallback).

        This is the SST method for getting time-based feature lookback.
        It supports both v2 (lookback_minutes) and v1 (lag_bars) schemas.

        Args:
            feature_name: Name of the feature
            interval_minutes: Current data interval (for v1 fallback computation)
            default_interval_minutes: Default interval if none provided (5m historical default)

        Returns:
            Lookback time in minutes, or None if feature not found

        Example:
            >>> registry.get_lookback_minutes("adx_14")
            70.0  # 14 bars * 5m (v1 fallback) or direct lookback_minutes (v2)
        """
        metadata = self.get_feature_metadata_effective(feature_name)
        if not metadata:
            return None

        # V2 schema: use lookback_minutes directly if present
        if 'lookback_minutes' in metadata and metadata['lookback_minutes'] is not None:
            return float(metadata['lookback_minutes'])

        # V1 fallback: compute from lag_bars * interval
        lag_bars = metadata.get('lag_bars')
        if lag_bars is None:
            return None

        # Use provided interval or default
        effective_interval = interval_minutes if interval_minutes is not None else default_interval_minutes
        return float(lag_bars * effective_interval)

    def get_allowed_horizon_minutes(
        self,
        feature_name: str,
        interval_minutes: Optional[int] = None,
        default_interval_minutes: int = 5
    ) -> Optional[List[float]]:
        """
        Get allowed horizons in minutes (v2 schema, with v1 fallback).

        Phase 7 (interval-agnostic pipeline): This is the SST method for getting
        time-based allowed horizons. It supports both v2 (allowed_horizon_minutes)
        and v1 (allowed_horizons in bars) schemas.

        Args:
            feature_name: Name of the feature
            interval_minutes: Current data interval (for v1 fallback computation)
            default_interval_minutes: Default interval if none provided (5m historical default)

        Returns:
            List of allowed horizon times in minutes, or None if feature not found

        Example:
            >>> registry.get_allowed_horizon_minutes("ret_5")
            [5.0, 10.0, 15.0, 25.0, 60.0, 120.0, 300.0]  # [1,2,3,5,12,24,60] bars * 5m
        """
        metadata = self.get_feature_metadata_effective(feature_name)
        if not metadata:
            return None

        # V2 schema: use allowed_horizon_minutes directly if present
        if 'allowed_horizon_minutes' in metadata and metadata['allowed_horizon_minutes'] is not None:
            return [float(h) for h in metadata['allowed_horizon_minutes']]

        # V1 fallback: compute from allowed_horizons (bars) * interval
        allowed_horizons = metadata.get('allowed_horizons')
        if allowed_horizons is None or allowed_horizons == []:
            return allowed_horizons  # Return None or []

        # Convert bars to minutes
        effective_interval = interval_minutes if interval_minutes is not None else default_interval_minutes
        return [float(h * effective_interval) for h in allowed_horizons]

    def is_horizon_allowed(
        self,
        feature_name: str,
        horizon_minutes: float,
        interval_minutes: Optional[int] = None,
        default_interval_minutes: int = 5
    ) -> bool:
        """
        Check if a feature is allowed for a specific horizon (in minutes).

        Phase 7 (interval-agnostic pipeline): This method checks horizon eligibility
        using time-based values, supporting both v2 and v1 schemas.

        Args:
            feature_name: Name of the feature
            horizon_minutes: Target horizon in minutes
            interval_minutes: Current data interval
            default_interval_minutes: Default interval if none provided

        Returns:
            True if horizon is allowed, False otherwise

        Example:
            >>> registry.is_horizon_allowed("ret_5", 60.0)  # 60 minutes
            True
        """
        allowed = self.get_allowed_horizon_minutes(
            feature_name, interval_minutes, default_interval_minutes
        )

        if allowed is None:
            return False  # Feature not found or has no allowed horizons
        if allowed == []:
            return False  # Explicitly disabled

        return horizon_minutes in allowed

    def get_feature_lookback_info(
        self,
        feature_name: str,
        interval_minutes: Optional[int] = None,
        default_interval_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Get comprehensive lookback info for a feature (for debugging/audit).

        Returns:
            {
                'feature_name': str,
                'schema_version': int,
                'lookback_minutes': Optional[float],  # Resolved value
                'lookback_source': str,  # 'v2_explicit', 'v1_computed', 'unknown'
                'lag_bars': Optional[int],  # Original v1 value if present
                'interval_minutes_used': int,  # Interval used for v1 computation
            }
        """
        metadata = self.get_feature_metadata_effective(feature_name)
        if not metadata:
            return {
                'feature_name': feature_name,
                'schema_version': self.get_schema_version(),
                'lookback_minutes': None,
                'lookback_source': 'unknown',
                'lag_bars': None,
                'interval_minutes_used': None,
            }

        effective_interval = interval_minutes if interval_minutes is not None else default_interval_minutes

        # Check for v2 explicit lookback_minutes
        if 'lookback_minutes' in metadata and metadata['lookback_minutes'] is not None:
            return {
                'feature_name': feature_name,
                'schema_version': self.get_schema_version(),
                'lookback_minutes': float(metadata['lookback_minutes']),
                'lookback_source': 'v2_explicit',
                'lag_bars': metadata.get('lag_bars'),
                'interval_minutes_used': effective_interval,
            }

        # V1 fallback
        lag_bars = metadata.get('lag_bars')
        if lag_bars is not None:
            return {
                'feature_name': feature_name,
                'schema_version': self.get_schema_version(),
                'lookback_minutes': float(lag_bars * effective_interval),
                'lookback_source': 'v1_computed',
                'lag_bars': lag_bars,
                'interval_minutes_used': effective_interval,
            }

        return {
            'feature_name': feature_name,
            'schema_version': self.get_schema_version(),
            'lookback_minutes': None,
            'lookback_source': 'unknown',
            'lag_bars': None,
            'interval_minutes_used': effective_interval,
        }

    def register_feature(self, name: str, metadata: Dict[str, Any]):
        """
        Register a new feature with metadata.
        
        Validates the feature before adding to registry.
        
        Args:
            name: Feature name
            metadata: Feature metadata dict
        """
        # Validate before adding
        self._validate_feature(name, metadata)
        
        # Add to registry
        self.features[name] = metadata
        logger.info(f"Registered feature '{name}' with metadata: {metadata}")


def get_registry(
    config_path: Optional[Path] = None,
    target_column: Optional[str] = None,
    registry_overlay_dir: Optional[Path] = None,  # NEW: explicit patch directory
    current_bar_minutes: Optional[float] = None,  # NEW: for compatibility check + selection
    strict: bool = False  # NEW: if True, raise RegistryLoadError on failure instead of returning empty registry
) -> FeatureRegistry:
    """
    Get global feature registry instance (singleton pattern).
    
    **SST Contract**: 
    - If `target_column` or `registry_overlay_dir` provided, create new instance (don't cache)
    - If neither provided, use cached global instance (backward compatible)
    - `current_bar_minutes` used for compatibility checking and interval-based selection
    - `allow_overwrite` read from config (SST - direct config read, no circular import)
    
    Args:
        config_path: Optional path to feature_registry.yaml
        target_column: Optional target column name (for per-target patch loading)
        registry_overlay_dir: Optional directory containing registry patches (explicit discovery only)
        current_bar_minutes: Optional current bar interval in minutes (for patch compatibility check + selection)
    
    Returns:
        FeatureRegistry instance
    """
    global _REGISTRY

    # Read allow_overwrite from config (SST - direct config read, no circular import)
    allow_overwrite = False  # Safe default
    try:
        from CONFIG.config_loader import get_cfg
        allow_overwrite = get_cfg('registry_autopatch.allow_overwrite', default=False)
    except Exception:
        # Config unavailable - use safe default
        pass

    # If config_path not provided, resolve with interval-aware selection
    if config_path is None:
        # Read selection config
        selection_mode = "manual"
        selection_strict = False
        interval_path_map = None
        try:
            from CONFIG.config_loader import get_cfg
            selection_mode = get_cfg('registry.selection_mode', default="manual")
            selection_strict = get_cfg('registry.selection_strict', default=False)
            interval_path_map = get_cfg('registry.interval_path_map', default=None)
        except Exception:
            pass

        # Use current_bar_minutes for selection (if available)
        config_path = resolve_registry_path_for_interval(
            base_path=None,
            interval_minutes=current_bar_minutes,
            selection_mode=selection_mode,
            selection_strict=selection_strict,
            interval_path_map=interval_path_map
        )
    
    # For per-target loading, always create new instance (don't cache)
    # This ensures patches are loaded per-target, not globally cached
    if target_column or registry_overlay_dir:
        registry = FeatureRegistry(
            config_path=config_path,
            target_column=target_column,
            registry_overlay_dir=registry_overlay_dir,
            current_bar_minutes=current_bar_minutes,
            allow_overwrite=allow_overwrite,
            _suppress_log=True,  # Suppress logging for per-target instances
            _strict=strict  # Pass strict mode to constructor
        )
        # In strict mode, validate registry loaded successfully
        if strict and (not registry.features or not registry.config):
            from TRAINING.common.exceptions import RegistryLoadError
            raise RegistryLoadError(
                message=f"Registry loaded but is empty (no features found): {config_path}",
                registry_path=str(config_path) if config_path else None,
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            )
        return registry
    
    # Global registry: only cache if exact default case (no params, default allow_overwrite)
    # Otherwise create new instance to avoid stale policy
    # TS-001: Use double-check locking pattern for thread safety
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            # Double-check inside lock to prevent race condition
            if _REGISTRY is None:
                _REGISTRY = FeatureRegistry(
                    config_path=config_path,
                    current_bar_minutes=current_bar_minutes,
                    allow_overwrite=allow_overwrite,
                    _strict=strict  # Pass strict mode to constructor
                )
                # In strict mode, validate registry loaded successfully
                if strict and (not _REGISTRY.features or not _REGISTRY.config):
                    from TRAINING.common.exceptions import RegistryLoadError
                    raise RegistryLoadError(
                        message=f"Registry loaded but is empty (no features found): {config_path}",
                        registry_path=str(config_path) if config_path else None,
                        stage="TARGET_RANKING",
                        error_code="REGISTRY_LOAD_FAILED"
                    )
    elif config_path is not None or allow_overwrite != False:
        # Config changed or non-default allow_overwrite - create new instance
        registry = FeatureRegistry(
            config_path=config_path,
            current_bar_minutes=current_bar_minutes,
            allow_overwrite=allow_overwrite,
            _suppress_log=True,
            _strict=strict  # Pass strict mode to constructor
        )
        # In strict mode, validate registry loaded successfully
        if strict and (not registry.features or not registry.config):
            from TRAINING.common.exceptions import RegistryLoadError
            raise RegistryLoadError(
                message=f"Registry loaded but is empty (no features found): {config_path}",
                registry_path=str(config_path) if config_path else None,
                stage="TARGET_RANKING",
                error_code="REGISTRY_LOAD_FAILED"
            )
        return registry
    
    # In strict mode, validate cached registry
    if strict and (not _REGISTRY.features or not _REGISTRY.config):
        from TRAINING.common.exceptions import RegistryLoadError
        raise RegistryLoadError(
            message=f"Cached registry is empty (no features found): {_REGISTRY.config_path}",
            registry_path=str(_REGISTRY.config_path),
            stage="TARGET_RANKING",
            error_code="REGISTRY_LOAD_FAILED"
        )
    
    return _REGISTRY


def get_registry_path() -> Optional[Path]:
    """
    Get the canonical feature registry path (SST).
    
    Returns:
        Path to feature_registry.yaml, or None if not found
    """
    try:
        path = _resolve_registry_path()
        return path if path.exists() else None
    except Exception:
        return None


def reset_registry():
    """Reset global registry (useful for testing)."""
    global _REGISTRY
    _REGISTRY = None

