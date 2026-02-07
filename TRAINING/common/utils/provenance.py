# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Provenance Generation and Validation

Generate and validate provenance blocks for artifacts to ensure auditability.
Every artifact must include provenance for defensibility.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TRAINING.common.utils.fingerprinting import RunIdentity
from TRAINING.common.utils.config_hashing import compute_config_hash


def generate_provenance(
    run_identity: Optional[RunIdentity] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_identifier: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    code_version: Optional[str] = None,
    include_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Generate provenance block for artifact.
    
    Args:
        run_identity: RunIdentity object (provides run_id, replicate_key, strict_key)
        config: Configuration dict (for config fingerprint)
        dataset_identifier: Dataset identifier (e.g., data directory path)
        dataset_hash: Dataset hash (SHA256)
        code_version: Code version identifier (git commit or build id)
        include_timestamp: Whether to include creation timestamp (excluded from deterministic hashes)
    
    Returns:
        Provenance dict with required fields
    """
    provenance: Dict[str, Any] = {}
    
    # Run identity (from RunIdentity if available)
    if run_identity:
        if run_identity.is_final:
            provenance["run_id"] = run_identity.strict_key
            provenance["replicate_key"] = run_identity.replicate_key
            provenance["strict_key"] = run_identity.strict_key
        else:
            # Partial identity - use what's available
            provenance["run_id"] = getattr(run_identity, "debug_key", None) or "partial"
    
    # Config fingerprint
    if config:
        try:
            provenance["config_fingerprint"] = compute_config_hash(config)
        except Exception:
            provenance["config_fingerprint"] = None
    
    # Dataset identifiers
    if dataset_identifier:
        provenance["dataset_identifier"] = dataset_identifier
    if dataset_hash:
        provenance["dataset_hash"] = dataset_hash
    
    # Code version
    if code_version:
        provenance["code_version"] = code_version
    else:
        # Try to get git commit
        try:
            from TRAINING.common.utils.git_utils import get_git_commit
            commit = get_git_commit()
            if commit:
                provenance["code_version"] = commit
        except Exception:
            provenance["code_version"] = None
    
    # Timestamp (excluded from deterministic hashes, stored separately)
    if include_timestamp:
        provenance["created_at"] = datetime.now(timezone.utc).isoformat()
    
    return provenance


def validate_provenance(provenance: Dict[str, Any], required_fields: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
    """
    Validate provenance block.
    
    Args:
        provenance: Provenance dict to validate
        required_fields: List of required field names (default: ["run_id", "code_version"])
    
    Returns:
        (is_valid, error_message) where is_valid is True if valid, False otherwise
    """
    if not isinstance(provenance, dict):
        return False, "Provenance must be a dict"
    
    if required_fields is None:
        required_fields = ["run_id", "code_version"]
    
    missing_fields = []
    for field in required_fields:
        if field not in provenance or provenance[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required provenance fields: {', '.join(missing_fields)}"
    
    # Validate run_id format (should be hash-like if from RunIdentity)
    if "run_id" in provenance:
        run_id = provenance["run_id"]
        if isinstance(run_id, str) and len(run_id) > 0:
            # Valid
            pass
        else:
            return False, f"Invalid run_id format: {run_id}"
    
    # Validate code_version format (should be commit hash or build id)
    if "code_version" in provenance:
        code_version = provenance["code_version"]
        if code_version and not isinstance(code_version, str):
            return False, f"Invalid code_version format: {code_version}"
    
    return True, None


def add_provenance_to_artifact(
    artifact: Dict[str, Any],
    run_identity: Optional[RunIdentity] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_identifier: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    code_version: Optional[str] = None,
    include_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Add provenance block to artifact dict.
    
    Args:
        artifact: Artifact dict to add provenance to
        run_identity: RunIdentity object
        config: Configuration dict
        dataset_identifier: Dataset identifier
        dataset_hash: Dataset hash
        code_version: Code version
        include_timestamp: Whether to include timestamp
    
    Returns:
        Artifact dict with provenance added under "_provenance" key
    """
    provenance = generate_provenance(
        run_identity=run_identity,
        config=config,
        dataset_identifier=dataset_identifier,
        dataset_hash=dataset_hash,
        code_version=code_version,
        include_timestamp=include_timestamp
    )
    
    # Add provenance to artifact
    artifact["_provenance"] = provenance
    
    return artifact


def extract_provenance(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract provenance block from artifact.
    
    Args:
        artifact: Artifact dict
    
    Returns:
        Provenance dict if present, None otherwise
    """
    return artifact.get("_provenance")


def validate_artifact_provenance(
    artifact: Dict[str, Any],
    required_fields: Optional[List[str]] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate artifact has valid provenance.
    
    Args:
        artifact: Artifact dict to validate
        required_fields: Required provenance fields
    
    Returns:
        (is_valid, error_message)
    """
    provenance = extract_provenance(artifact)
    
    if provenance is None:
        return False, "Artifact missing provenance block"
    
    return validate_provenance(provenance, required_fields)
