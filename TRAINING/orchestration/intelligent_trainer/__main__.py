# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Trainer Entry Point

Allows running the intelligent trainer as a module:
    python -m TRAINING.orchestration.intelligent_trainer

This module imports main() from the sibling intelligent_trainer.py file
and provides the entry point for package execution.
"""

# ============================================================================
# CRITICAL: Import repro_bootstrap FIRST before ANY numeric libraries
# This sets thread env vars BEFORE numpy/torch/sklearn are imported.
# DO NOT move this import or add imports above it!
# ============================================================================
import TRAINING.common.repro_bootstrap  # noqa: F401 - side effects only

import sys

# Import main from the sibling module file
# The __init__.py handles the complex import of the .py file that's a sibling to this package
from TRAINING.orchestration.intelligent_trainer import main


if __name__ == "__main__":
    sys.exit(main())
