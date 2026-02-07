#!/usr/bin/env python3
"""
Pre-import bootstrap for reproducibility.

IMPORT THIS MODULE FIRST before any numeric libraries (numpy, torch, sklearn, etc.)
to ensure thread environment variables are set correctly.

Usage:
    # At the TOP of your entrypoint (before any other imports):
    import TRAINING.common.repro_bootstrap

This module is stdlib-only to ensure it can be imported before numpy.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC
"""

from __future__ import annotations

import os
import sys
import warnings

# ============================================================================
# MODE DETECTION
# ============================================================================

# Check if strict mode - DEFAULT TO BEST_EFFORT to avoid breaking normal runs
# Strict is OPT-IN via launcher or explicit env var
_STRICT_MODE = os.environ.get("REPRO_MODE", "best_effort").lower() == "strict"

# Required env vars for strict determinism (must be set BEFORE Python starts)
_REQUIRED_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
]

# Forbidden modules - if these are imported before bootstrap, strict mode is fake
_FORBIDDEN_MODULES = [
    "numpy", "torch", "sklearn", "lightgbm", "xgboost", "catboost",
    "scipy", "pandas"  # pandas imports numpy
]


# ============================================================================
# TOO-LATE DETECTION
# ============================================================================

def _check_too_late():
    """
    STRICT ONLY: Detect if numeric libs were imported before bootstrap.
    
    If numpy/torch/sklearn/lightgbm/xgboost are in sys.modules, bootstrap was
    imported too late and strict mode is fake.
    """
    if not _STRICT_MODE:
        return
    
    already_imported = [m for m in _FORBIDDEN_MODULES if m in sys.modules]
    
    if already_imported:
        print(f"🚨 STRICT MODE HARD-FAIL: Bootstrap imported too late!", file=sys.stderr)
        print(f"   Already imported: {already_imported}", file=sys.stderr)
        print(f"   repro_bootstrap must be imported BEFORE any numeric libraries.", file=sys.stderr)
        print(f"   Fix: Move 'import TRAINING.common.repro_bootstrap' to the FIRST line.", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# THREAD ENVIRONMENT SETUP
# ============================================================================

def _set_thread_env_vars():
    """
    Set thread-related environment variables.
    
    These MUST be set before importing numpy/MKL/OpenMP.
    In strict mode: force single-threaded.
    In best_effort mode: use reasonable defaults.
    """
    threads = "1" if _STRICT_MODE else str(max(1, (os.cpu_count() or 4) - 1))
    
    env_vars = {
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        # MKL settings for reproducibility
        "MKL_THREADING_LAYER": "GNU",
        "MKL_CBWR": "COMPATIBLE",
    }
    
    # CUDA determinism (only meaningful in strict mode)
    if _STRICT_MODE:
        env_vars["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch CUDNN determinism
        env_vars["CUBLAS_WORKSPACE_SIZE"] = "4194304"  # 4MB workspace

    for key, value in env_vars.items():
        os.environ.setdefault(key, value)


# ============================================================================
# PYTORCH DETERMINISM (Post-Import Hook)
# ============================================================================

_TORCH_DETERMINISM_INITIALIZED = False


def init_torch_determinism(seed: int = 42) -> None:
    """
    Initialize PyTorch determinism after torch is imported.

    MUST be called after importing torch but before any model operations.
    This is separate from env var setup because torch must be imported first.

    Args:
        seed: Random seed for torch (default: 42)

    Usage:
        import torch
        from TRAINING.common.repro_bootstrap import init_torch_determinism
        init_torch_determinism(seed=42)

    In strict mode, this:
        - Sets torch.use_deterministic_algorithms(True)
        - Disables CUDNN benchmark mode
        - Enables CUDNN deterministic mode
        - Sets random seeds

    In best_effort mode, this:
        - Sets random seeds only
        - Leaves CUDNN in default mode (faster but non-deterministic)

    Contract (raw-ohlcv-sequence-mode.md Phase 3):
        This ensures sequence models (Transformer, LSTM, CNN1D) produce
        bitwise identical results across runs in strict mode.
    """
    global _TORCH_DETERMINISM_INITIALIZED

    if _TORCH_DETERMINISM_INITIALIZED:
        return

    try:
        import torch
    except ImportError:
        # PyTorch not available - nothing to do
        return

    # Set seeds regardless of mode
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if _STRICT_MODE:
        # STRICT: Full determinism (may be slower)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            # Some operations don't have deterministic implementations
            # In strict mode, we want to know about these
            print(
                f"⚠️ STRICT MODE: torch.use_deterministic_algorithms failed: {e}",
                file=sys.stderr
            )
            print(
                "   Some operations may not have deterministic implementations.",
                file=sys.stderr
            )
            # Try enabling with warn_only for debugging
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (TypeError, RuntimeError):
                pass  # Older PyTorch versions don't support warn_only

        # CUDNN determinism
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"🔒 PyTorch determinism: STRICT mode, seed={seed}")
    else:
        # BEST_EFFORT: Seeds only, allow non-deterministic ops for speed
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True  # Auto-tune for speed

        print(f"🔒 PyTorch determinism: BEST_EFFORT mode, seed={seed}")

    _TORCH_DETERMINISM_INITIALIZED = True


def is_torch_determinism_initialized() -> bool:
    """Check if PyTorch determinism has been initialized."""
    return _TORCH_DETERMINISM_INITIALIZED


# ============================================================================
# PYTHONHASHSEED VALIDATION
# ============================================================================

def _validate_pythonhashseed():
    """
    Validate PYTHONHASHSEED is set. In strict mode, HARD-FAIL if missing.
    
    PYTHONHASHSEED MUST be set BEFORE Python starts - we can only check here.
    This is NON-NEGOTIABLE in strict mode.
    """
    hashseed = os.environ.get("PYTHONHASHSEED")
    
    if hashseed is None:
        msg = (
            "PYTHONHASHSEED not set. For deterministic runs, set it BEFORE Python starts:\n"
            "  export PYTHONHASHSEED=42\n"
            "  python your_script.py\n"
            "Or use: bin/run_deterministic.sh your_script.py"
        )
        if _STRICT_MODE:
            # HARD-FAIL: This is non-negotiable in strict mode
            print(f"🚨 STRICT MODE HARD-FAIL: {msg}", file=sys.stderr)
            print(f"   Strict mode requires PYTHONHASHSEED to be set BEFORE Python starts.", file=sys.stderr)
            print(f"   Use: bin/run_deterministic.sh <your_script.py>", file=sys.stderr)
            sys.exit(1)
        else:
            warnings.warn(f"⚠️ BEST_EFFORT MODE: {msg}")


# ============================================================================
# ENV VAR VALIDATION
# ============================================================================

def _validate_env_vars():
    """Validate required env vars are set (strict mode only)."""
    if not _STRICT_MODE:
        return
    
    missing = []
    wrong_value = []
    
    for var in _REQUIRED_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            missing.append(var)
        elif value != "1":
            wrong_value.append(f"{var}={value} (expected 1)")
    
    if missing:
        print(f"🚨 STRICT MODE: Missing required env vars: {missing}", file=sys.stderr)
        print(f"   These should have been set by _set_thread_env_vars().", file=sys.stderr)
        # Don't fail here - we just set them above
    
    if wrong_value:
        # In strict mode, thread count should be 1
        print(f"⚠️ STRICT MODE: Non-deterministic thread settings: {wrong_value}", file=sys.stderr)


# ============================================================================
# BOOTSTRAP STATE LOGGING
# ============================================================================

def _log_bootstrap_state():
    """Log bootstrap state for diagnostics."""
    mode = "STRICT" if _STRICT_MODE else "BEST_EFFORT"
    hashseed = os.environ.get("PYTHONHASHSEED", "NOT SET")
    omp = os.environ.get("OMP_NUM_THREADS", "NOT SET")
    mkl = os.environ.get("MKL_NUM_THREADS", "NOT SET")
    
    # Use print to avoid logging import issues
    print(f"🔒 repro_bootstrap: mode={mode}, PYTHONHASHSEED={hashseed}, OMP={omp}, MKL={mkl}")


# ============================================================================
# BOOTSTRAP EXECUTION (runs on import)
# ============================================================================

# 1. STRICT ONLY: Check if imported too late (numpy already in sys.modules)
_check_too_late()

# 2. Set thread env vars FIRST (before any numeric library imports)
_set_thread_env_vars()

# 3. Validate PYTHONHASHSEED (hard-fail in strict mode if missing)
_validate_pythonhashseed()

# 4. Validate all required env vars
_validate_env_vars()

# 5. Log bootstrap state
_log_bootstrap_state()

# 6. Mark bootstrap as complete (can be checked by determinism.py)
os.environ["_REPRO_BOOTSTRAP_COMPLETE"] = "1"
