# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Backward compatibility wrapper for TRAINING.core

This module has been moved to TRAINING.common.core
All imports are re-exported here to maintain backward compatibility.
"""

# Re-export everything from the new location
from TRAINING.common.core import *

__all__ = [
    'determinism',
    'environment',
]
