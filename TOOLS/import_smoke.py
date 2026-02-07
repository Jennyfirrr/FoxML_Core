#!/usr/bin/env python3
"""
Import Smoke Test

Tests that critical modules can be imported without circular import errors.
This catches import-time cycles that would break E2E runs.

Minimum spec:
- Imports Level 2 + Level 3 modules (cycle-prone zone)
- Imports orchestration entrypoints
- Runs under same env vars/config as E2E
- Exits nonzero on ImportError

Usage:
    python TOOLS/import_smoke.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test critical imports for circular dependency issues."""
    errors = []
    
    # Level 2: Path Helpers (cycle-prone)
    try:
        import TRAINING.orchestration.utils.target_first_paths as tfp
        print("✓ target_first_paths imported")
    except ImportError as e:
        errors.append(f"target_first_paths: {e}")
    except Exception as e:
        errors.append(f"target_first_paths (unexpected): {e}")
    
    # Level 3: Orchestration (cycle-prone)
    try:
        import TRAINING.orchestration.utils.run_context as rc
        print("✓ run_context imported")
    except ImportError as e:
        errors.append(f"run_context: {e}")
    except Exception as e:
        errors.append(f"run_context (unexpected): {e}")
    
    try:
        import TRAINING.orchestration.utils.manifest as mf
        print("✓ manifest imported")
    except ImportError as e:
        errors.append(f"manifest: {e}")
    except Exception as e:
        errors.append(f"manifest (unexpected): {e}")
    
    # Orchestration entrypoints
    try:
        import TRAINING.orchestration.intelligent_trainer as it
        print("✓ intelligent_trainer imported")
    except ImportError as e:
        errors.append(f"intelligent_trainer: {e}")
    except Exception as e:
        errors.append(f"intelligent_trainer (unexpected): {e}")
    
    try:
        import TRAINING.orchestration.training_router as tr
        print("✓ training_router imported")
    except ImportError as e:
        errors.append(f"training_router: {e}")
    except Exception as e:
        errors.append(f"training_router (unexpected): {e}")
    
    try:
        import TRAINING.orchestration.training_plan_generator as tpg
        print("✓ training_plan_generator imported")
    except ImportError as e:
        errors.append(f"training_plan_generator: {e}")
    except Exception as e:
        errors.append(f"training_plan_generator (unexpected): {e}")
    
    # Report results
    if errors:
        print("\n❌ Import errors detected:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("\n✓ All imports successful")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())
