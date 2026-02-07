"""
SST Compliance Checking
=======================

Tools to verify SST compliance in LIVE_TRADING code.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def check_sst_compliance(file_path: str) -> Dict[str, Any]:
    """
    Check a file for SST compliance issues.

    Checks for:
    - repro_bootstrap import in entry points
    - get_cfg() usage for config
    - sorted_items() for dict iteration
    - write_atomic_json() for JSON writes
    - iterdir_sorted() for filesystem iteration

    Args:
        file_path: Path to check (relative or absolute)

    Returns:
        Dict with compliance issues
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / file_path

    if not path.exists():
        return {"error": f"File not found: {file_path}", "compliant": False}

    content = path.read_text()
    lines = content.splitlines()
    issues: List[Dict[str, Any]] = []

    # Check 1: repro_bootstrap in entry points
    is_entry_point = (
        "if __name__" in content
        or (path.name == "__init__.py" and "LIVE_TRADING/__init__.py" in str(path))
    )
    if is_entry_point:
        if "import TRAINING.common.repro_bootstrap" not in content:
            issues.append({
                "type": "missing_repro_bootstrap",
                "message": "Entry point should import TRAINING.common.repro_bootstrap first",
                "severity": "error",
                "line": None,
            })
        else:
            # Check it's imported first (before numpy, pandas, etc.)
            bootstrap_line = None
            ml_lib_line = None
            for i, line in enumerate(lines, 1):
                if "repro_bootstrap" in line:
                    bootstrap_line = i
                if bootstrap_line is None and any(
                    lib in line for lib in ["import numpy", "import pandas", "import tensorflow"]
                ):
                    ml_lib_line = i
            if ml_lib_line and bootstrap_line and ml_lib_line < bootstrap_line:
                issues.append({
                    "type": "repro_bootstrap_order",
                    "message": "repro_bootstrap must be imported before ML libraries",
                    "severity": "error",
                    "line": bootstrap_line,
                })

    # Check 2: Config access without get_cfg
    config_patterns = [
        (r'DEFAULT_CONFIG\[', "Direct DEFAULT_CONFIG access - use get_cfg() with default"),
        (r'config\s*=\s*\{', "Hardcoded config dict - use get_cfg()"),
    ]
    for pattern, message in config_patterns:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line) and "get_cfg" not in line:
                issues.append({
                    "type": "hardcoded_config",
                    "message": message,
                    "severity": "warning",
                    "line": i,
                })

    # Check 3: Dict iteration without sorted_items
    for i, line in enumerate(lines, 1):
        if ".items()" in line and "for " in line:
            if "sorted_items" not in line and "sorted(" not in line:
                issues.append({
                    "type": "unsorted_dict_iteration",
                    "message": "Dict iteration should use sorted_items() for determinism",
                    "severity": "warning",
                    "line": i,
                })

    # Check 4: Direct json.dump without atomic write
    for i, line in enumerate(lines, 1):
        if "json.dump(" in line and "write_atomic" not in content:
            issues.append({
                "type": "non_atomic_write",
                "message": "JSON writes should use write_atomic_json()",
                "severity": "warning",
                "line": i,
            })

    # Check 5: Direct iterdir without sorted
    for i, line in enumerate(lines, 1):
        if ".iterdir()" in line and "iterdir_sorted" not in line:
            issues.append({
                "type": "unsorted_iterdir",
                "message": "Use iterdir_sorted() for deterministic filesystem iteration",
                "severity": "warning",
                "line": i,
            })

    # Check 6: glob without sorted
    for i, line in enumerate(lines, 1):
        if ".glob(" in line and "glob_sorted" not in line and "sorted(" not in line:
            issues.append({
                "type": "unsorted_glob",
                "message": "Use glob_sorted() for deterministic glob results",
                "severity": "warning",
                "line": i,
            })

    error_count = sum(1 for i in issues if i["severity"] == "error")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")

    return {
        "file": str(file_path),
        "issues": issues,
        "error_count": error_count,
        "warning_count": warning_count,
        "compliant": error_count == 0,
    }


def find_hardcoded_config(directory: str = "LIVE_TRADING") -> Dict[str, Any]:
    """
    Find hardcoded config values that should use get_cfg().

    Args:
        directory: Directory to scan

    Returns:
        Dict with findings
    """
    path = PROJECT_ROOT / directory
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}

    findings: List[Dict[str, Any]] = []

    # Patterns that suggest hardcoded config
    patterns = [
        (r'=\s*0\.\d+\s*[#\n]', "Possible hardcoded decimal"),
        (r'=\s*\d{2,}\s*[#\n]', "Possible hardcoded integer"),
        (r'["\'](5m|10m|15m|30m|60m|1d)["\']', "Hardcoded horizon string"),
        (r'lambda_?\s*=', "Possible hardcoded lambda"),
        (r'threshold\s*=', "Possible hardcoded threshold"),
    ]

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()
        lines = content.splitlines()

        for pattern, desc in patterns:
            for i, line in enumerate(lines, 1):
                # Skip if get_cfg is used
                if "get_cfg" in line or "default=" in line:
                    continue
                # Skip imports and comments
                if line.strip().startswith("#") or line.strip().startswith("import"):
                    continue
                if re.search(pattern, line):
                    findings.append({
                        "file": str(py_file.relative_to(PROJECT_ROOT)),
                        "line": i,
                        "pattern": desc,
                        "content": line.strip()[:80],
                    })

    return {
        "directory": directory,
        "total_findings": len(findings),
        "findings": findings[:50],  # Limit output
    }


def check_sorted_items_usage(file_path: str) -> Dict[str, Any]:
    """
    Detailed check for sorted_items() usage in dict iterations.

    Args:
        file_path: File to check

    Returns:
        Dict with analysis
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / file_path

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    content = path.read_text()

    # Parse AST to find for loops
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    iterations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Check if iterating over .items()
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Attribute):
                    if node.iter.func.attr == "items":
                        # Get line content
                        line_content = content.splitlines()[node.lineno - 1]

                        is_sorted = (
                            "sorted_items" in line_content
                            or "sorted(" in line_content
                        )

                        iterations.append({
                            "line": node.lineno,
                            "content": line_content.strip(),
                            "uses_sorted_items": is_sorted,
                        })

    unsorted = [i for i in iterations if not i["uses_sorted_items"]]

    return {
        "file": str(file_path),
        "total_dict_iterations": len(iterations),
        "unsorted_iterations": len(unsorted),
        "details": unsorted,
        "compliant": len(unsorted) == 0,
    }


def check_all_live_trading_compliance() -> Dict[str, Any]:
    """
    Run compliance checks on all LIVE_TRADING files.

    Returns:
        Summary of compliance issues
    """
    path = PROJECT_ROOT / "LIVE_TRADING"

    if not path.exists():
        return {"error": "LIVE_TRADING directory not found"}

    results = {
        "total_files": 0,
        "compliant_files": 0,
        "files_with_errors": [],
        "files_with_warnings": [],
        "all_issues": [],
    }

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        relative_path = str(py_file.relative_to(PROJECT_ROOT))
        result = check_sst_compliance(relative_path)

        results["total_files"] += 1

        if result.get("error"):
            continue

        if result["compliant"]:
            results["compliant_files"] += 1
        else:
            results["files_with_errors"].append(relative_path)

        if result["warning_count"] > 0:
            results["files_with_warnings"].append(relative_path)

        for issue in result["issues"]:
            results["all_issues"].append({
                "file": relative_path,
                **issue,
            })

    results["compliance_pct"] = (
        results["compliant_files"] / results["total_files"] * 100
        if results["total_files"] > 0
        else 0
    )

    return results
