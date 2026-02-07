"""
Implementation Progress Tracking
================================

Tools to track which files from LIVE_TRADING plans have been implemented.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base path for the project
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Expected files from each plan
PLAN_FILES: Dict[str, List[str]] = {
    "01": [
        "LIVE_TRADING/__init__.py",
        "LIVE_TRADING/common/__init__.py",
        "LIVE_TRADING/common/exceptions.py",
        "LIVE_TRADING/common/constants.py",
        "LIVE_TRADING/common/types.py",
        "LIVE_TRADING/common/hooks.py",
        "LIVE_TRADING/common/audit.py",
        "LIVE_TRADING/tests/test_common.py",
    ],
    "02": [
        "LIVE_TRADING/brokers/__init__.py",
        "LIVE_TRADING/brokers/interface.py",
        "LIVE_TRADING/brokers/paper.py",
        "LIVE_TRADING/brokers/data_provider.py",
        "LIVE_TRADING/tests/test_broker_interface.py",
        "LIVE_TRADING/tests/test_paper_broker.py",
    ],
    "03": [
        "LIVE_TRADING/models/__init__.py",
        "LIVE_TRADING/models/loader.py",
        "LIVE_TRADING/models/inference.py",
        "LIVE_TRADING/models/feature_builder.py",
        "LIVE_TRADING/tests/test_model_loader.py",
        "LIVE_TRADING/tests/test_inference.py",
    ],
    "04": [
        "LIVE_TRADING/prediction/__init__.py",
        "LIVE_TRADING/prediction/standardization.py",
        "LIVE_TRADING/prediction/confidence.py",
        "LIVE_TRADING/prediction/predictor.py",
        "LIVE_TRADING/tests/test_standardization.py",
        "LIVE_TRADING/tests/test_predictor.py",
    ],
    "05": [
        "LIVE_TRADING/blending/__init__.py",
        "LIVE_TRADING/blending/ridge_weights.py",
        "LIVE_TRADING/blending/temperature.py",
        "LIVE_TRADING/blending/horizon_blender.py",
        "LIVE_TRADING/tests/test_blending.py",
    ],
    "06": [
        "LIVE_TRADING/arbitration/__init__.py",
        "LIVE_TRADING/arbitration/cost_model.py",
        "LIVE_TRADING/arbitration/horizon_arbiter.py",
        "LIVE_TRADING/tests/test_arbitration.py",
    ],
    "07": [
        "LIVE_TRADING/gating/__init__.py",
        "LIVE_TRADING/gating/barrier_gate.py",
        "LIVE_TRADING/gating/spread_gate.py",
        "LIVE_TRADING/tests/test_gating.py",
    ],
    "08": [
        "LIVE_TRADING/sizing/__init__.py",
        "LIVE_TRADING/sizing/vol_scaling.py",
        "LIVE_TRADING/sizing/turnover.py",
        "LIVE_TRADING/sizing/position_sizer.py",
        "LIVE_TRADING/tests/test_sizing.py",
    ],
    "09": [
        "LIVE_TRADING/risk/__init__.py",
        "LIVE_TRADING/risk/drawdown.py",
        "LIVE_TRADING/risk/exposure.py",
        "LIVE_TRADING/risk/guardrails.py",
        "LIVE_TRADING/tests/test_risk.py",
    ],
    "10": [
        "LIVE_TRADING/engine/__init__.py",
        "LIVE_TRADING/engine/state.py",
        "LIVE_TRADING/engine/trading_engine.py",
        "LIVE_TRADING/tests/test_engine_integration.py",
    ],
    "11": [
        "CONFIG/live_trading/live_trading.yaml",
        "CONFIG/live_trading/symbols.yaml",
        "bin/run_live_trading.py",
        "LIVE_TRADING/README.md",
        "LIVE_TRADING/tests/conftest.py",
    ],
}

# Plan names for display
PLAN_NAMES: Dict[str, str] = {
    "01": "Common Infrastructure",
    "02": "Broker Layer",
    "03": "Model Integration",
    "04": "Prediction Pipeline",
    "05": "Blending",
    "06": "Arbitration",
    "07": "Gating",
    "08": "Sizing",
    "09": "Risk Management",
    "10": "Trading Engine",
    "11": "Config & CLI",
}

# Persistent state file
STATE_FILE = PROJECT_ROOT / "LIVE_TRADING" / ".dev_progress.json"


def _load_state() -> Dict[str, Any]:
    """Load persistent state."""
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {"completed_files": [], "test_status": {}}


def _save_state(state: Dict[str, Any]) -> None:
    """Save persistent state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


def check_implementation_progress(plan: str = "all") -> Dict[str, Any]:
    """
    Check which files from the plans have been implemented.

    Args:
        plan: Plan number (e.g., '01', '02') or 'all'

    Returns:
        Dict with progress summary
    """
    progress = {
        "total_files": 0,
        "implemented": 0,
        "missing": [],
        "by_plan": {},
        "by_phase": {"P0": {}, "P1": {}, "P2": {}},
    }

    # Phase mapping
    phases = {
        "P0": ["01", "02", "03", "09"],
        "P1": ["04", "05", "06", "07", "08"],
        "P2": ["10", "11"],
    }

    plans_to_check = list(PLAN_FILES.keys()) if plan == "all" else [plan]

    for p in plans_to_check:
        if p not in PLAN_FILES:
            continue

        files = PLAN_FILES[p]
        plan_progress = {
            "name": PLAN_NAMES.get(p, f"Plan {p}"),
            "total": len(files),
            "done": 0,
            "missing": [],
            "files": {},
        }

        for file_path in files:
            full_path = PROJECT_ROOT / file_path
            progress["total_files"] += 1
            exists = full_path.exists()

            if exists:
                progress["implemented"] += 1
                plan_progress["done"] += 1
                plan_progress["files"][file_path] = "implemented"
            else:
                progress["missing"].append(file_path)
                plan_progress["missing"].append(file_path)
                plan_progress["files"][file_path] = "missing"

        plan_progress["completion_pct"] = (
            plan_progress["done"] / plan_progress["total"] * 100
            if plan_progress["total"] > 0
            else 0
        )
        progress["by_plan"][p] = plan_progress

    # Aggregate by phase
    for phase, plan_nums in phases.items():
        phase_total = 0
        phase_done = 0
        for p in plan_nums:
            if p in progress["by_plan"]:
                phase_total += progress["by_plan"][p]["total"]
                phase_done += progress["by_plan"][p]["done"]
        progress["by_phase"][phase] = {
            "total": phase_total,
            "done": phase_done,
            "completion_pct": phase_done / phase_total * 100 if phase_total > 0 else 0,
        }

    progress["completion_pct"] = (
        progress["implemented"] / progress["total_files"] * 100
        if progress["total_files"] > 0
        else 0
    )

    return progress


def get_plan_status(plan: str) -> Dict[str, Any]:
    """
    Get detailed status for a single plan.

    Args:
        plan: Plan number

    Returns:
        Detailed plan status
    """
    if plan not in PLAN_FILES:
        return {"error": f"Unknown plan: {plan}"}

    files = PLAN_FILES[plan]
    state = _load_state()

    status = {
        "plan": plan,
        "name": PLAN_NAMES.get(plan, f"Plan {plan}"),
        "files": [],
    }

    for file_path in files:
        full_path = PROJECT_ROOT / file_path
        file_status = {
            "path": file_path,
            "exists": full_path.exists(),
            "tests_passing": state.get("test_status", {}).get(file_path, None),
        }

        # Check file size if exists
        if file_status["exists"]:
            file_status["lines"] = len(full_path.read_text().splitlines())

        status["files"].append(file_status)

    status["total"] = len(files)
    status["implemented"] = sum(1 for f in status["files"] if f["exists"])
    status["completion_pct"] = status["implemented"] / status["total"] * 100

    return status


def mark_file_complete(
    file_path: str,
    tests_passing: bool = False,
) -> Dict[str, Any]:
    """
    Mark a file as implemented with test status.

    Args:
        file_path: Relative path to file
        tests_passing: Whether tests pass for this file

    Returns:
        Updated status
    """
    state = _load_state()

    if file_path not in state["completed_files"]:
        state["completed_files"].append(file_path)

    state["test_status"][file_path] = tests_passing

    _save_state(state)

    return {
        "file": file_path,
        "marked_complete": True,
        "tests_passing": tests_passing,
    }


def get_next_files_to_implement() -> List[Dict[str, str]]:
    """
    Get the next files that should be implemented based on dependencies.

    Returns:
        List of files to implement next
    """
    progress = check_implementation_progress()

    # Priority order: P0 -> P1 -> P2
    phases = ["P0", "P1", "P2"]
    plan_order = {
        "P0": ["01", "02", "03", "09"],
        "P1": ["04", "05", "06", "07", "08"],
        "P2": ["10", "11"],
    }

    next_files = []

    for phase in phases:
        for plan in plan_order[phase]:
            plan_status = progress["by_plan"].get(plan, {})
            if plan_status.get("missing"):
                for f in plan_status["missing"]:
                    next_files.append({
                        "file": f,
                        "plan": plan,
                        "phase": phase,
                        "plan_name": PLAN_NAMES.get(plan, ""),
                    })
                # Only return files from current incomplete plan in phase
                if next_files:
                    return next_files

    return next_files
