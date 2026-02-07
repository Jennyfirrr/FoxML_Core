"""
Determinism Verification
========================

Tools to verify determinism patterns in LIVE_TRADING code.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# Patterns that indicate potential determinism issues
NONDETERMINISTIC_PATTERNS = [
    # Dict iteration without sorting
    (r'for\s+\w+\s+in\s+\w+\.keys\(\)', "Iterating over dict keys without sorting"),
    (r'for\s+\w+\s+in\s+\w+\.values\(\)', "Iterating over dict values without sorting"),
    (r'for\s+\w+,\s*\w+\s+in\s+\w+\.items\(\)(?!\s*#.*sorted)', "Dict items() without sorted_items"),

    # Set iteration
    (r'for\s+\w+\s+in\s+set\(', "Iterating over set (non-deterministic order)"),
    (r'for\s+\w+\s+in\s+\{[^}]+\}', "Iterating over set literal"),

    # Filesystem without sorting
    (r'\.iterdir\(\)(?!\s*#.*sorted)', "iterdir without sorting"),
    (r'\.glob\([^)]+\)(?!\s*#.*sorted)', "glob without sorting"),
    (r'os\.listdir\(', "os.listdir without sorting"),

    # Random without seed
    (r'random\.(choice|shuffle|sample)\(', "random function without documented seed"),
    (r'np\.random\.(rand|randn|randint)\(', "numpy random without seed context"),

    # Hash-based operations
    (r'hash\(', "hash() can vary between runs"),

    # Timestamps in artifacts
    (r'datetime\.now\(\)', "datetime.now() in artifact context"),
    (r'time\.time\(\)', "time.time() in artifact context"),

    # Dict/set comprehensions
    (r'\{[^:]+:\s*[^}]+for\s+\w+\s+in\s+\w+(?!\.items)', "Dict comprehension may have ordering issues"),
]


def check_determinism_violations(directory: str = "LIVE_TRADING") -> Dict[str, Any]:
    """
    Find potential determinism violations in code.

    Args:
        directory: Directory to scan

    Returns:
        Dict with findings
    """
    path = PROJECT_ROOT / directory
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}

    findings: List[Dict[str, Any]] = []

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()
        lines = content.splitlines()
        relative_path = str(py_file.relative_to(PROJECT_ROOT))

        for pattern, description in NONDETERMINISTIC_PATTERNS:
            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue
                # Skip if explicitly marked safe
                if "# deterministic" in line.lower() or "# sorted" in line.lower():
                    continue
                # Skip if using sorted_ helpers
                if "sorted_items" in line or "iterdir_sorted" in line or "glob_sorted" in line:
                    continue
                # Skip test files
                if "test_" in relative_path:
                    continue

                if re.search(pattern, line):
                    findings.append({
                        "file": relative_path,
                        "line": i,
                        "issue": description,
                        "content": line.strip()[:80],
                        "pattern": pattern,
                    })

    # Categorize by severity
    high_severity = []
    medium_severity = []
    low_severity = []

    high_patterns = ["dict keys", "dict values", "items()", "set(", "iterdir", "glob"]
    medium_patterns = ["random", "hash("]

    for f in findings:
        if any(p in f["issue"].lower() for p in high_patterns):
            high_severity.append(f)
        elif any(p in f["issue"].lower() for p in medium_patterns):
            medium_severity.append(f)
        else:
            low_severity.append(f)

    return {
        "directory": directory,
        "total_findings": len(findings),
        "high_severity": len(high_severity),
        "medium_severity": len(medium_severity),
        "low_severity": len(low_severity),
        "high_severity_details": high_severity[:20],
        "medium_severity_details": medium_severity[:10],
        "all_findings": findings,
    }


def verify_repro_bootstrap(directory: str = "LIVE_TRADING") -> Dict[str, Any]:
    """
    Verify repro_bootstrap is imported correctly in entry points.

    Args:
        directory: Directory to check

    Returns:
        Dict with verification results
    """
    path = PROJECT_ROOT / directory
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}

    entry_points = []
    issues = []

    # Find entry points
    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()
        relative_path = str(py_file.relative_to(PROJECT_ROOT))

        is_entry = False
        is_main_init = relative_path.endswith("LIVE_TRADING/__init__.py")

        if "if __name__" in content:
            is_entry = True
        if is_main_init:
            is_entry = True

        if not is_entry:
            continue

        entry_points.append(relative_path)

        # Check for repro_bootstrap
        has_bootstrap = "import TRAINING.common.repro_bootstrap" in content

        if not has_bootstrap:
            issues.append({
                "file": relative_path,
                "issue": "Missing repro_bootstrap import",
                "severity": "error",
            })
            continue

        # Check import order
        lines = content.splitlines()
        bootstrap_line = None
        ml_lib_lines = []

        for i, line in enumerate(lines, 1):
            if "repro_bootstrap" in line:
                bootstrap_line = i
            if any(lib in line for lib in ["import numpy", "import pandas", "import tensorflow", "import torch"]):
                ml_lib_lines.append((i, line.strip()))

        for lib_line, lib_content in ml_lib_lines:
            if bootstrap_line and lib_line < bootstrap_line:
                issues.append({
                    "file": relative_path,
                    "issue": f"ML library imported before repro_bootstrap (line {lib_line})",
                    "severity": "error",
                    "detail": lib_content,
                })

    return {
        "directory": directory,
        "entry_points_found": len(entry_points),
        "entry_points": entry_points,
        "issues": issues,
        "compliant": len([i for i in issues if i["severity"] == "error"]) == 0,
    }


def check_random_seed_usage(directory: str = "LIVE_TRADING") -> Dict[str, Any]:
    """
    Check for proper random seed management.

    Args:
        directory: Directory to check

    Returns:
        Dict with findings
    """
    path = PROJECT_ROOT / directory
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}

    findings = []

    random_patterns = [
        (r'random\.seed\(', "random.seed call found"),
        (r'np\.random\.seed\(', "np.random.seed call found"),
        (r'torch\.manual_seed\(', "torch.manual_seed call found"),
        (r'tf\.random\.set_seed\(', "tf.random.set_seed call found"),
    ]

    unsafe_random = [
        (r'random\.(choice|shuffle|sample|randint|random)\(', "random function"),
        (r'np\.random\.(rand|randn|randint|choice|shuffle)\(', "np.random function"),
    ]

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        content = py_file.read_text()
        relative_path = str(py_file.relative_to(PROJECT_ROOT))

        # Check for seed setting
        has_seed = any(re.search(p, content) for p, _ in random_patterns)

        # Check for random usage
        for pattern, desc in unsafe_random:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    "file": relative_path,
                    "line": line_num,
                    "type": desc,
                    "has_seed_in_file": has_seed,
                })

    return {
        "directory": directory,
        "random_usages_found": len(findings),
        "findings": findings[:30],
    }
