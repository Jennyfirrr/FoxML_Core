# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
FoxML Exception Taxonomy

Structured exceptions for fail-closed error handling in financial ML pipeline.
All exceptions carry structured payload for auditability and debugging.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class FoxMLError(Exception):
    """
    Base exception for all FoxML errors.
    
    All exceptions must carry structured payload for auditability:
    - run_id: Run identifier (if available)
    - stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    - error_code: Machine-readable error code
    - context: Additional context dict
    """
    message: str
    run_id: Optional[str] = None
    stage: Optional[str] = None
    error_code: str = "FOXML_ERROR"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize base exception with message."""
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dict for logging/audit."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "run_id": self.run_id,
            "stage": self.stage,
            "error_code": self.error_code,
            "context": self.context
        }


@dataclass
class ConfigError(FoxMLError):
    """
    Configuration error: missing, invalid, or conflicting config.
    
    Error codes:
    - CONFIG_MISSING: Required config key not found
    - CONFIG_INVALID: Config value fails validation
    - CONFIG_CONFLICT: Config values conflict
    """
    config_path: Optional[str] = None
    config_name: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "CONFIG_ERROR"
        if self.config_path:
            self.context["config_path"] = self.config_path
        if self.config_name:
            self.context["config_name"] = self.config_name


@dataclass
class DataIntegrityError(FoxMLError):
    """
    Data integrity error: schema mismatch, unexpected NaNs/infs, invalid values.
    
    Error codes:
    - DATA_SCHEMA_MISMATCH: Schema doesn't match expected
    - DATA_NAN_INF: Unexpected NaN or Inf values
    - DATA_INVALID_VALUE: Value fails validation
    - DATA_MISSING_REQUIRED: Required column/data missing
    """
    data_path: Optional[str] = None
    column: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "DATA_INTEGRITY_ERROR"
        if self.data_path:
            self.context["data_path"] = self.data_path
        if self.column:
            self.context["column"] = self.column


@dataclass
class LeakageError(FoxMLError):
    """
    Leakage detection error: lookahead bias, data leakage, or temporal violations.
    
    Error codes:
    - LEAKAGE_LOOKAHEAD: Lookahead bias detected
    - LEAKAGE_TEMPORAL: Temporal ordering violation
    - LEAKAGE_FEATURE: Feature contains future information
    """
    feature_name: Optional[str] = None
    target_id: Optional[str] = None
    horizon_minutes: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "LEAKAGE_ERROR"
        if self.feature_name:
            self.context["feature_name"] = self.feature_name
        if self.target_id:
            self.context["target_id"] = self.target_id
        if self.horizon_minutes:
            self.context["horizon_minutes"] = self.horizon_minutes


@dataclass
class ArtifactError(FoxMLError):
    """
    Artifact error: serialization/deserialization failure, invalid format.
    
    Error codes:
    - ARTIFACT_SERIALIZE_FAIL: Serialization failed
    - ARTIFACT_DESERIALIZE_FAIL: Deserialization failed
    - ARTIFACT_INVALID_FORMAT: Artifact format invalid
    - ARTIFACT_MISSING_PROVENANCE: Artifact missing required provenance
    """
    artifact_path: Optional[str] = None
    artifact_type: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "ARTIFACT_ERROR"
        if self.artifact_path:
            self.context["artifact_path"] = self.artifact_path
        if self.artifact_type:
            self.context["artifact_type"] = self.artifact_type


@dataclass
class StageBoundaryError(FoxMLError):
    """
    Stage boundary error: stage input/output validation failure.
    
    Error codes:
    - STAGE_INPUT_INVALID: Stage input validation failed
    - STAGE_OUTPUT_INVALID: Stage output validation failed
    - STAGE_MISSING_INPUT: Required stage input missing
    - STAGE_MISSING_OUTPUT: Required stage output missing
    """
    boundary_type: Optional[str] = None  # "input" or "output"
    feature_set_hash: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "STAGE_BOUNDARY_ERROR"
        if self.boundary_type:
            self.context["boundary_type"] = self.boundary_type
        if self.feature_set_hash:
            self.context["feature_set_hash"] = self.feature_set_hash


@dataclass
class RegistryLoadError(FoxMLError):
    """
    Registry load error: feature registry file missing, unreadable, or invalid.
    
    Error codes:
    - REGISTRY_FILE_MISSING: Registry file does not exist
    - REGISTRY_FILE_UNREADABLE: Registry file exists but cannot be read
    - REGISTRY_INVALID_FORMAT: Registry file format is invalid
    - REGISTRY_LOAD_FAILED: Registry loading failed (generic)
    """
    registry_path: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.error_code == "FOXML_ERROR":
            self.error_code = "REGISTRY_LOAD_ERROR"
        if self.registry_path:
            self.context["registry_path"] = self.registry_path


# ============================================================================
# CENTRALIZED ERROR HANDLING POLICY
# ============================================================================

def should_fail_closed(
    stage: str,
    error_type: str,
    affects_artifact: bool = True,
    affects_routing: bool = False,
    affects_selection: bool = False,
    affects_training_plan: bool = False,
    affects_manifest: bool = False
) -> bool:
    """
    Centralized policy: determine if error should fail closed (raise) or fail open (warn + continue).
    
    Policy:
    - Strict/deterministic mode: **Raise** (fail closed) for anything affecting:
      - Artifact content (routing, selection, training plan generation, manifest creation)
      - Core pipeline stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    - Best-effort mode: Warn + fallback allowed for non-core convenience paths only
    
    Args:
        stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING, etc.)
        error_type: Error type/category (e.g., "feature_pruning", "shared_harness", "training_family")
        affects_artifact: If True, error affects artifact content (default: True)
        affects_routing: If True, error affects routing decisions
        affects_selection: If True, error affects feature selection
        affects_training_plan: If True, error affects training plan generation
        affects_manifest: If True, error affects manifest creation
    
    Returns:
        True if should fail closed (raise), False if should fail open (warn + continue)
    """
    # Check if strict/deterministic mode
    try:
        from TRAINING.common.determinism import is_strict_mode
        strict = is_strict_mode()
    except Exception:
        strict = False
    
    # Core pipeline stages that always fail closed in strict mode
    core_stages = {"TARGET_RANKING", "FEATURE_SELECTION", "TRAINING"}
    is_core_stage = stage in core_stages if stage else False
    
    # Artifact-shaping operations always fail closed in strict mode
    affects_critical = (
        affects_artifact or
        affects_routing or
        affects_selection or
        affects_training_plan or
        affects_manifest
    )
    
    if strict:
        # Strict mode: fail closed for core stages and artifact-shaping operations
        if is_core_stage or affects_critical:
            return True
        # Even in strict mode, some convenience paths can fail open (e.g., optional metadata)
        return False
    else:
        # Best-effort mode: fail open for most errors, except critical invariants
        # Critical invariants that should always fail closed:
        critical_errors = {
            "leakage_detection",
            "data_integrity",
            "stage_boundary",
            "artifact_provenance"
        }
        if error_type in critical_errors:
            return True
        return False


def handle_error_with_policy(
    error: Exception,
    stage: str,
    error_type: str,
    affects_artifact: bool = True,
    affects_routing: bool = False,
    affects_selection: bool = False,
    affects_training_plan: bool = False,
    affects_manifest: bool = False,
    fallback_value: Any = None,
    logger_instance: Optional[Any] = None
) -> Any:
    """
    Handle error according to centralized policy.
    
    If should_fail_closed() returns True, raises the error.
    Otherwise, logs warning and returns fallback_value.
    
    Args:
        error: Exception to handle
        stage: Pipeline stage
        error_type: Error type/category
        affects_artifact: If error affects artifact content
        affects_routing: If error affects routing
        affects_selection: If error affects selection
        affects_training_plan: If error affects training plan
        affects_manifest: If error affects manifest
        fallback_value: Value to return if fail open
        logger_instance: Optional logger (uses module logger if None)
    
    Returns:
        fallback_value if fail open, otherwise raises
    
    Raises:
        The original error if should_fail_closed() returns True
    """
    import logging
    if logger_instance is None:
        logger_instance = logging.getLogger(__name__)
    
    if should_fail_closed(
        stage=stage,
        error_type=error_type,
        affects_artifact=affects_artifact,
        affects_routing=affects_routing,
        affects_selection=affects_selection,
        affects_training_plan=affects_training_plan,
        affects_manifest=affects_manifest
    ):
        # Fail closed: raise the error
        raise error
    else:
        # Fail open: log warning and return fallback
        logger_instance.warning(
            f"[{stage}] {error_type} failed (fail-open mode): {error}. "
            f"Continuing with fallback value."
        )
        if hasattr(error, 'to_dict'):
            logger_instance.debug(f"Error details: {error.to_dict()}")
        return fallback_value
