# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cohort Manager Mixin for ReproducibilityTracker.

Contains methods for cohort ID computation and directory management.
Extracted from reproducibility_tracker.py for maintainability.

SST COMPLIANCE:
- Uses compute_cohort_id() from cohort_id module (unified helper)
- Uses target_first_paths for path construction
- All path operations use sorted iteration where applicable
- Fingerprinting logic preserved exactly

DETERMINISM:
- No dict iteration without sorted()
- Enum comparisons used instead of string comparisons where possible
- Stable cohort ID computation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# SST: Import Stage and View enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

# SST: Import extraction helpers
from TRAINING.orchestration.utils.reproducibility.utils import (
    extract_n_effective,
    extract_universe_sig,
)

# Import WriteScope for scope-safe writes (optional)
try:
    from TRAINING.orchestration.utils.scope_resolution import (
        WriteScope,
        ScopePurpose,
        View as ScopeView,
        Stage as ScopeStage
    )
    _WRITE_SCOPE_AVAILABLE = True
except ImportError:
    _WRITE_SCOPE_AVAILABLE = False
    WriteScope = None
    ScopePurpose = None
    ScopeView = None
    ScopeStage = None

if TYPE_CHECKING:
    from TRAINING.orchestration.utils.scope_resolution import WriteScope

logger = logging.getLogger(__name__)


class CohortManagerMixin:
    """
    Mixin class providing cohort management methods for ReproducibilityTracker.

    This mixin contains methods related to:
    - Cohort metadata extraction
    - Cohort ID computation (fingerprinting)
    - Cohort directory path construction
    - Path validation for scope compliance

    Methods in this mixin expect the following attributes on self:
    - cohort_aware: bool - Whether cohort-aware mode is enabled
    - cohort_config_keys: List[str] - Config keys to include in cohort metadata
    - _repro_base_dir: Path - Base directory for reproducibility artifacts
    - _routing_eval_root: Path - Root directory for routing evaluation
    """

    # Type hints for expected attributes (set by the main class)
    cohort_aware: bool
    cohort_config_keys: List[str]
    _repro_base_dir: Path
    _routing_eval_root: Path

    def _extract_cohort_metadata(
        self,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract cohort metadata from metrics and additional_data.

        SST: Uses extract_n_effective and extract_universe_sig helpers.

        Returns:
            Dict with cohort metadata or None if insufficient data
        """
        if not self.cohort_aware:
            return None

        cohort = {}

        # Extract n_effective_cs (sample size) - use SST accessor
        n_effective = extract_n_effective(metrics, additional_data)

        if n_effective is None:
            return None  # Can't form cohort without sample size

        cohort['n_effective_cs'] = int(n_effective)

        # Extract n_symbols
        n_symbols = metrics.get('n_symbols')
        if n_symbols is None and additional_data:
            n_symbols = additional_data.get('n_symbols')
        cohort['n_symbols'] = int(n_symbols) if n_symbols is not None else 0

        # Extract date_range
        date_range = {}
        if additional_data:
            if 'date_range' in additional_data:
                date_range = additional_data['date_range']
            elif 'start_ts' in additional_data or 'end_ts' in additional_data:
                date_range = {
                    'start_ts': additional_data.get('start_ts'),
                    'end_ts': additional_data.get('end_ts')
                }
        cohort['date_range'] = date_range

        # Extract config hash components
        cs_config = {}
        if additional_data:
            config_data = additional_data.get('cs_config', {})
            for key in self.cohort_config_keys:
                if key in config_data:
                    cs_config[key] = config_data[key]
                elif key in additional_data:
                    cs_config[key] = additional_data[key]
        cohort['cs_config'] = cs_config

        # Extract universe_sig at top level (canonical key for routing)
        # Check top-level first, then nested cs_config as fallback
        universe_sig = None
        if additional_data:
            # Use SST accessor for universe_sig
            universe_sig = extract_universe_sig(additional_data)

        if universe_sig:
            cohort['universe_sig'] = universe_sig
            # Mirror into cs_config for backward compat (defensive: ensure cs_config is dict)
            if not isinstance(cohort.get('cs_config'), dict):
                cohort['cs_config'] = {}
            cohort['cs_config']['universe_sig'] = universe_sig

        return cohort

    def _compute_cohort_id(
        self,
        cohort: Dict[str, Any],
        view: str,  # REQUIRED: CROSS_SECTIONAL | SYMBOL_SPECIFIC
        mode: Optional[str] = None  # DEPRECATED: use view instead
    ) -> str:
        """
        Compute readable cohort ID from metadata.

        SST: Delegates to unified compute_cohort_id() helper.
        DETERMINISM: Stable ID computation from metadata.

        Args:
            cohort: Cohort metadata dict
            view: REQUIRED view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
            mode: DEPRECATED - use view instead. Kept for backward compatibility.

        Returns:
            Cohort ID string with prefix matching view

        Raises:
            ValueError: If view is invalid or mode doesn't match view
        """
        # SST: Use unified helper
        from TRAINING.orchestration.utils.cohort_id import compute_cohort_id

        # If legacy mode provided, validate it matches view (explicit startswith check)
        if mode:
            view_enum = View.from_string(view) if isinstance(view, str) else view
            mode_check = mode.lower()
            if view_enum == View.CROSS_SECTIONAL and not mode_check.startswith("cs") and mode_check not in ("cross_sectional",):
                logger.warning(
                    f"Mode/view mismatch: mode={mode} does not match view={view}. "
                    f"Using view-derived prefix"
                )
            elif view_enum == View.SYMBOL_SPECIFIC and not mode_check.startswith("sy") and mode_check not in ("symbol_specific", "individual"):
                logger.warning(
                    f"Mode/view mismatch: mode={mode} does not match view={view}. "
                    f"Using view-derived prefix"
                )

        return compute_cohort_id(cohort, view)

    def _calculate_cohort_relative_path(self, cohort_dir: Path) -> str:
        """
        Calculate relative path from cohort_dir to run root.

        SST: Uses target_first_paths.run_root helper.

        Args:
            cohort_dir: Cohort directory path

        Returns:
            Relative path string
        """
        # Calculate relative path from cohort_dir to run root using SST helper
        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
        run_root = get_run_root(Path(cohort_dir))

        # Calculate relative path
        try:
            path = str(Path(cohort_dir).relative_to(run_root))
        except ValueError:
            # If not relative, use absolute path as fallback
            path = str(cohort_dir)

        return path

    def _get_cohort_dir(
        self,
        stage: str,
        target: str,
        cohort_id: str,
        view: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None
    ) -> Path:
        """
        Get directory for a specific cohort following the structured layout.

        Structure:
        REPRODUCIBILITY/
          {STAGE}/
            {MODE}/  (for FEATURE_SELECTION, TRAINING)
              {target}/
                {symbol}/  (for INDIVIDUAL mode)
                  {model_family}/  (for TRAINING)
                    cohort={cohort_id}/

        Args:
            stage: Pipeline stage (e.g., "target_ranking", "feature_selection", "model_training")
            target: Item name (e.g., target name)
            cohort_id: Cohort identifier
            view: Optional route type ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
            symbol: Optional symbol name (for SYMBOL_SPECIFIC mode)
            model_family: Optional model family (for TRAINING stage)

        Returns:
            Path to cohort directory
        """
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"

        # Normalize stage to enum, then to string for path construction
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        stage_upper = str(stage_enum)  # Stage enum's __str__ returns .value

        # Build path components
        path_parts = [stage_upper]

        # For TARGET_RANKING, add view subdirectory (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
        if stage_enum == Stage.TARGET_RANKING:
            # Check if view is provided in additional_data or view
            view_local = None
            # Normalize view to enum if provided
            if view:
                try:
                    view_enum = View.from_string(view) if isinstance(view, str) else view
                    view_local = view_enum  # Store as enum
                except ValueError:
                    # Handle LOSO (not a View enum value)
                    if isinstance(view, str) and view.upper() == "LOSO":
                        view_local = View.SYMBOL_SPECIFIC  # LOSO maps to SYMBOL_SPECIFIC
                    else:
                        view_enum = View.from_string(view) if isinstance(view, str) else view
                        view_local = view_enum
            # If view not provided, check if we can infer from symbol presence
            if view_local is None and symbol:
                view_local = View.SYMBOL_SPECIFIC  # Default for symbol-specific
            if view_local is None:
                view_local = View.CROSS_SECTIONAL  # Default
            path_parts.append(str(view_local))  # Convert enum to string for path

        # Add mode subdirectory for FEATURE_SELECTION and TRAINING
        elif stage_enum in (Stage.FEATURE_SELECTION, Stage.TRAINING):
            if view:
                mode = view.upper()
                # FIX: Accept SYMBOL_SPECIFIC as valid mode for FEATURE_SELECTION
                # Normalize mode to View enum for validation
                try:
                    mode_enum = View.from_string(mode) if isinstance(mode, str) else mode
                    mode = str(mode_enum)  # Convert back to string for path
                except ValueError:
                    # Invalid mode, default to SYMBOL_SPECIFIC
                    mode = View.SYMBOL_SPECIFIC.value
            else:
                mode = View.CROSS_SECTIONAL.value  # Default
            path_parts.append(mode)

        # Add target/target
        path_parts.append(target)

        # Add symbol for SYMBOL_SPECIFIC/LOSO views (TARGET_RANKING) or SYMBOL_SPECIFIC/INDIVIDUAL mode (FEATURE_SELECTION/TRAINING)
        if stage_enum == Stage.TARGET_RANKING:
            # Normalize view to enum for comparison (handle LOSO as string)
            if isinstance(view, str) and view == "LOSO":
                if symbol:
                    path_parts.append(f"symbol={symbol}")
            else:
                view_enum = View.from_string(view) if isinstance(view, str) else view
                if view_enum == View.SYMBOL_SPECIFIC and symbol:
                    path_parts.append(f"symbol={symbol}")
        elif stage_enum in (Stage.FEATURE_SELECTION, Stage.TRAINING):
            # For FEATURE_SELECTION/TRAINING, add symbol if mode is SYMBOL_SPECIFIC
            # Normalize view to enum for comparison
            view_enum = View.from_string(view) if isinstance(view, str) else view
            if symbol and view_enum == View.SYMBOL_SPECIFIC:
                path_parts.append(f"symbol={symbol}")

        # Add model_family for TRAINING
        if stage_enum == Stage.TRAINING and model_family:
            path_parts.append(f"model_family={model_family}")

        # Add cohort directory
        path_parts.append(f"cohort={cohort_id}")

        return repro_dir / Path(*path_parts)

    def _get_cohort_dir_v2(
        self,
        scope: "WriteScope",
        cohort_id: str,
        target: str,
        model_family: Optional[str] = None
    ) -> Path:
        """
        Get directory for a specific cohort using WriteScope (v2 API).

        This method replaces _get_cohort_dir and provides:
        - Purpose-based routing (FINAL vs ROUTING_EVAL)
        - Enum-based view handling (no string drift)
        - Path-relative invariant validation

        Structure (FINAL):
        targets/{target}/reproducibility/
          {VIEW}/  (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
            universe={universe_sig}/
              [symbol={symbol}/]  (only for SYMBOL_SPECIFIC)
                [model_family={family}/]  (for TRAINING stage)
                  cohort={cohort_id}/

        Structure (ROUTING_EVAL):
        routing_evaluation/
          {VIEW}/
            universe={universe_sig}/
              [symbol={symbol}/]
                cohort={cohort_id}/

        Args:
            scope: WriteScope with view, universe_sig, symbol, purpose, stage
            cohort_id: Cohort identifier
            target: Target name
            model_family: Optional model family (for TRAINING stage)

        Returns:
            Path to cohort directory

        Raises:
            ValueError: If scope invariants violated or cohort_id prefix mismatch
        """
        if not _WRITE_SCOPE_AVAILABLE or scope is None:
            raise ValueError("WriteScope not available or scope is None")

        # Validate cohort prefix matches scope view
        scope.validate_cohort_id(cohort_id)

        # Determine root based on purpose
        if scope.purpose is ScopePurpose.ROUTING_EVAL:
            repro_root = self._routing_eval_root
        else:
            # FINAL: use target-first structure with stage scoping
            from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
            repro_root = get_target_reproducibility_dir(
                self._repro_base_dir, target, stage=scope.stage.value
            )

        # Build path components
        path_parts = [scope.view.value]  # Use enum value for path

        # DETERMINISTIC: Add universe scoping (CROSS_SECTIONAL only)
        # Enum comparison is deterministic (no dict iteration)
        if scope.view is ScopeView.CROSS_SECTIONAL:
            path_parts.append(f"batch_{scope.universe_sig[:12]}")  # Use batch_ prefix (deterministic slice)
        # Add symbol for SYMBOL_SPECIFIC (no universe=)
        if scope.view is ScopeView.SYMBOL_SPECIFIC and scope.symbol:
            path_parts.append(f"symbol={scope.symbol}")

        # Add model_family for TRAINING stage
        if scope.stage is ScopeStage.TRAINING and model_family:
            path_parts.append(f"model_family={model_family}")

        # CRITICAL: Always include attempt_ level for consistency
        # Extract attempt_id from scope if available, otherwise default to 0
        attempt_id = getattr(scope, 'attempt_id', 0) if hasattr(scope, 'attempt_id') else 0
        path_parts.append(f"attempt_{attempt_id}")

        # Add cohort directory
        path_parts.append(f"cohort={cohort_id}")

        cohort_dir = repro_root / Path(*path_parts)

        # Validate purpose/path invariant
        self._validate_purpose_path(scope, cohort_dir)

        return cohort_dir

    def _validate_purpose_path(self, scope: "WriteScope", cohort_dir: Path) -> None:
        """
        Validate that purpose matches path root using is_relative_to().

        This ensures:
        - ROUTING_EVAL purpose only writes under routing_evaluation/
        - FINAL purpose never writes under routing_evaluation/

        Args:
            scope: WriteScope with purpose
            cohort_dir: Target directory for write

        Raises:
            ValueError: If purpose/path mismatch detected
        """
        if not _WRITE_SCOPE_AVAILABLE or scope is None:
            return  # Skip validation if WriteScope not available

        # Check if cohort_dir is under routing_eval_root
        def is_relative_to(path: Path, other: Path) -> bool:
            try:
                path.relative_to(other)
                return True
            except ValueError:
                return False

        is_under_eval_root = is_relative_to(cohort_dir, self._routing_eval_root)

        if scope.purpose is ScopePurpose.ROUTING_EVAL:
            if not is_under_eval_root:
                raise ValueError(
                    f"SCOPE VIOLATION: ROUTING_EVAL purpose but path not under routing_evaluation root. "
                    f"path={cohort_dir}, eval_root={self._routing_eval_root}, scope={scope}"
                )
        else:  # FINAL
            if is_under_eval_root:
                raise ValueError(
                    f"SCOPE VIOLATION: FINAL purpose but path under routing_evaluation root. "
                    f"path={cohort_dir}, scope={scope}"
                )
