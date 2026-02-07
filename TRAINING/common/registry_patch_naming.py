# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry Patch Naming Utilities

Collision-proof filename generation and patch file discovery.
Neutral module to avoid import cycles.
"""

import re
import hashlib
from pathlib import Path
from typing import Optional


def safe_target_filename(target: str, suffix: str = ".yaml") -> str:
    """
    Generate collision-proof filename from target name.
    
    Uses hash suffix to prevent collisions for similar target names
    (e.g., fwd_ret_5m@view=CROSS_SECTIONAL vs fwd_ret_5m@view=SYMBOL_SPECIFIC).
    
    Args:
        target: Target column name
        suffix: File suffix (default: ".yaml", can be ".unblock.yaml")
    
    Returns:
        Safe filename with hash suffix: {safe_base}__{hash}{suffix}
    """
    safe_base = re.sub(r'[^\w\-_]', '_', target)
    target_hash = hashlib.sha1(target.encode()).hexdigest()[:12]  # 12 hex chars
    return f"{safe_base}__{target_hash}{suffix}"


def find_patch_file(
    directory: Path,
    target: str,
    suffix: str = ".yaml"  # Can be ".yaml" or ".unblock.yaml"
) -> Optional[Path]:
    """
    Find patch file for target using collision-proof naming.
    
    Args:
        directory: Directory to search in
        target: Target column name
        suffix: File suffix (default: ".yaml", can be ".unblock.yaml")
    
    Returns:
        Path to patch file if found, None otherwise
    
    Note: Requires exact filename match (no hash suffix fallback).
    Hash changes if target changes, so fallback would risk matching wrong file.
    """
    expected_name = safe_target_filename(target, suffix=suffix)
    patch_file = directory / expected_name
    if patch_file.exists():
        return patch_file
    
    # NO FALLBACK: Require exact filename match (hash changes if target changes)
    return None
