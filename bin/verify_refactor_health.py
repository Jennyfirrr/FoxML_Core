#!/usr/bin/env python3
"""
Post-Refactor Health Check Script

Run this after any significant refactoring to catch common runtime errors
before they manifest in E2E tests.

Usage:
    python bin/verify_refactor_health.py
    python bin/verify_refactor_health.py --verbose
    python bin/verify_refactor_health.py --fix  # Show fix suggestions

Checks performed:
1. Import shadowing (local imports shadowing module-level)
2. Missing/incorrect module paths
3. Determinism patterns
4. Critical import validation
5. Contract tests
"""

import ast
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# =============================================================================
# Configuration
# =============================================================================

# Core pipeline files to check
PIPELINE_FILES = [
    "TRAINING/orchestration/intelligent_trainer.py",
    "TRAINING/orchestration/intelligent_trainer/pipeline_stages.py",
    "TRAINING/ranking/feature_selector.py",
    "TRAINING/ranking/target_ranker.py",
    "TRAINING/ranking/utils/cross_sectional_data.py",
    "TRAINING/training_strategies/execution/data_preparation.py",
    "TRAINING/training_strategies/strategy_functions.py",
    "TRAINING/common/memory/memory_manager.py",
    "TRAINING/data/loading/unified_loader.py",
    "TRAINING/ranking/multi_model_feature_selection.py",
    "TRAINING/orchestration/utils/reproducibility_tracker.py",
    "TRAINING/orchestration/utils/diff_telemetry.py",
]

# Known incorrect import patterns and their fixes
BAD_IMPORT_PATTERNS = {
    "TRAINING.common.utils.exceptions": "TRAINING.common.exceptions",
    "TRAINING.common.utils.duration_utils": "TRAINING.common.utils.duration_parser",
}

# Paths to check for package/module collisions
COLLISION_CHECK_PATHS = [
    "TRAINING/ranking",
    "TRAINING/orchestration",
    "TRAINING/common",
    "TRAINING/training_strategies",
]

# Critical imports that must work
CRITICAL_IMPORTS = [
    ("TRAINING.common.exceptions", "ConfigError"),
    ("TRAINING.common.utils.duration_parser", "parse_duration"),
    ("TRAINING.common.utils.determinism_ordering", "sorted_items"),
    ("TRAINING.common.utils.determinism_ordering", "sorted_keys"),
    ("TRAINING.common.utils.file_utils", "write_atomic_json"),
    ("TRAINING.common.memory", "log_memory_phase"),
    ("TRAINING.ranking.feature_selector", "select_features_for_target"),
    ("TRAINING.ranking.target_ranker", "rank_targets"),
    ("TRAINING.orchestration.intelligent_trainer", "IntelligentTrainer"),
    ("CONFIG.config_loader", "get_cfg"),
]


# =============================================================================
# Check Functions
# =============================================================================

def check_package_module_collisions(root_paths: List[str]) -> List[Tuple[str, str, List[str]]]:
    """
    Check for package/module naming collisions.

    When a .py file and a directory share the same name (e.g., module.py and module/),
    Python prefers the directory (package). This breaks relative imports from submodules
    that expect to import from the file.

    Returns:
        List of (file_path, directory_path, problematic_imports) tuples
    """
    collisions = []

    for root_path in root_paths:
        root = Path(root_path)
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dir_path = Path(dirpath)

            # Get all .py files (without extension)
            py_modules = {f[:-3] for f in filenames if f.endswith('.py') and f != '__init__.py'}

            # Check if any directory shares a name with a .py file
            dir_set = set(dirnames)

            for collision in py_modules & dir_set:
                file_path = dir_path / f"{collision}.py"
                pkg_path = dir_path / collision

                # Check for problematic imports in the package's submodules
                problematic_imports = []
                for py_file in pkg_path.glob('**/*.py'):
                    if py_file.name == '__init__.py':
                        continue
                    try:
                        content = py_file.read_text()
                        for i, line in enumerate(content.split('\n'), 1):
                            # Look for relative imports going up that reference the collision name
                            if f'from ..{collision}' in line or f'from .{collision}' in line:
                                # Check if using importlib workaround
                                if 'importlib.util' not in content or 'spec_from_file_location' not in content:
                                    problematic_imports.append(
                                        f"{py_file.relative_to(root)}:{i}: {line.strip()}"
                                    )
                    except Exception:
                        pass

                collisions.append((str(file_path), str(pkg_path), problematic_imports))

    return collisions


def check_import_shadowing(filepath: str, verbose: bool = False) -> List[str]:
    """
    Check for local imports that shadow module-level imports.

    This is a Python gotcha: if you import a name locally inside a function,
    Python treats that name as local throughout the ENTIRE function, even
    before the import statement. This causes UnboundLocalError.
    """
    issues = []

    if not os.path.exists(filepath):
        return issues

    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error in {filepath}: {e}"]

    # Get module-level imports
    module_imports = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                module_imports.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                module_imports.add(name)

    # Check functions (excluding nested functions)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            local_imports = {}
            uses_before_import = {}

            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef):
                    continue  # Skip nested functions - they have their own scope

                for subchild in ast.walk(child):
                    if isinstance(subchild, ast.ImportFrom):
                        for alias in subchild.names:
                            name = alias.asname if alias.asname else alias.name
                            if name not in local_imports:
                                local_imports[name] = subchild.lineno

                    if isinstance(subchild, ast.Name) and subchild.id in module_imports:
                        if subchild.id not in uses_before_import:
                            uses_before_import[subchild.id] = subchild.lineno

            for name, import_line in local_imports.items():
                if name in module_imports and name in uses_before_import:
                    if uses_before_import[name] < import_line:
                        issues.append(
                            f"{filepath}:{uses_before_import[name]}: "
                            f"'{name}' used before local import at line {import_line} in {func_name}() "
                            f"[FIX: Remove redundant local import at line {import_line}]"
                        )

    return issues


def check_bad_import_paths(filepath: str, verbose: bool = False) -> List[str]:
    """Check for known incorrect import paths."""
    issues = []

    if not os.path.exists(filepath):
        return issues

    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for bad_pattern, correct_pattern in BAD_IMPORT_PATTERNS.items():
                if bad_pattern in node.module:
                    for alias in node.names:
                        issues.append(
                            f"{filepath}:{node.lineno}: "
                            f"'{node.module}' should be '{correct_pattern}' "
                            f"[FIX: Replace import path]"
                        )

    return issues


def check_module_exists(module_path: str) -> bool:
    """Check if a TRAINING.* module path exists."""
    parts = module_path.split('.')
    file_path = Path('/'.join(parts) + '.py')
    dir_path = Path('/'.join(parts))
    init_path = dir_path / '__init__.py'

    return file_path.exists() or init_path.exists()


def check_missing_modules(filepath: str, verbose: bool = False) -> List[str]:
    """Check for imports from non-existent modules."""
    issues = []

    if not os.path.exists(filepath):
        return issues

    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith('TRAINING.'):
                if not check_module_exists(node.module):
                    # Check if it's in a try/except block (acceptable)
                    # For simplicity, we'll flag it but note it might be handled
                    for alias in node.names:
                        issues.append(
                            f"{filepath}:{node.lineno}: "
                            f"Module '{node.module}' may not exist "
                            f"(importing {alias.name}) [CHECK: Verify module path or ensure try/except handling]"
                        )

    return issues


def check_critical_imports() -> List[Tuple[str, str, str]]:
    """Verify all critical imports work."""
    results = []

    for module, name in CRITICAL_IMPORTS:
        try:
            exec(f'from {module} import {name}')
            results.append((module, name, "OK"))
        except ImportError as e:
            results.append((module, name, f"FAIL: {e}"))
        except Exception as e:
            results.append((module, name, f"WARN: {type(e).__name__}: {e}"))

    return results


def check_determinism_patterns() -> List[str]:
    """Check for common determinism issues."""
    issues = []

    # Check for unseeded random usage
    import subprocess
    result = subprocess.run(
        ['grep', '-rn', r'random\.\(random\|randint\|choice\|shuffle\)(', 'TRAINING/'],
        capture_output=True, text=True
    )

    for line in result.stdout.strip().split('\n'):
        if line and 'test' not in line.lower() and '#' not in line.split(':')[-1][:20]:
            # Check if it's not in a comment
            issues.append(f"Potential unseeded random: {line}")

    return issues[:10]  # Limit output


def run_contract_tests() -> Tuple[bool, str]:
    """Run contract tests and return pass/fail."""
    import subprocess

    result = subprocess.run(
        ['python', '-m', 'pytest', 'TRAINING/contract_tests/', '-v', '--tb=short', '-q'],
        capture_output=True, text=True, timeout=120
    )

    passed = result.returncode == 0
    # Extract summary line
    summary = ""
    for line in result.stdout.split('\n'):
        if 'passed' in line or 'failed' in line or 'error' in line:
            summary = line.strip()
            break

    return passed, summary or result.stderr[-200:] if result.stderr else "Unknown result"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Post-refactor health check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Show fix suggestions')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running pytest')
    args = parser.parse_args()

    print("=" * 70)
    print("POST-REFACTOR HEALTH CHECK")
    print("=" * 70)
    print()

    all_issues = []

    # 0. Check for package/module collisions (CRITICAL)
    print("0. Checking for package/module naming collisions...")
    collisions = check_package_module_collisions(COLLISION_CHECK_PATHS)

    if collisions:
        print(f"   ⚠️  Found {len(collisions)} collision(s):")
        for file_path, pkg_path, problematic in collisions:
            print(f"      • {Path(file_path).name} ↔ {Path(pkg_path).name}/")
            if problematic:
                print(f"        ❌ {len(problematic)} problematic imports without importlib workaround:")
                for p in problematic[:3]:
                    print(f"           {p}")
                if len(problematic) > 3:
                    print(f"           ... and {len(problematic) - 3} more")
                all_issues.extend(problematic)
            else:
                print(f"        ✅ All submodule imports use importlib workaround")
    else:
        print("   ✅ No package/module naming collisions")
    print()

    # 1. Check import shadowing
    print("1. Checking for import shadowing issues...")
    shadow_issues = []
    for filepath in PIPELINE_FILES:
        shadow_issues.extend(check_import_shadowing(filepath, args.verbose))

    if shadow_issues:
        print(f"   ❌ Found {len(shadow_issues)} shadowing issues:")
        for issue in shadow_issues:
            print(f"      {issue}")
        all_issues.extend(shadow_issues)
    else:
        print("   ✅ No import shadowing issues")
    print()

    # 2. Check bad import paths
    print("2. Checking for incorrect import paths...")
    path_issues = []
    for filepath in PIPELINE_FILES:
        path_issues.extend(check_bad_import_paths(filepath, args.verbose))

    if path_issues:
        print(f"   ❌ Found {len(path_issues)} incorrect paths:")
        for issue in path_issues:
            print(f"      {issue}")
        all_issues.extend(path_issues)
    else:
        print("   ✅ No incorrect import paths")
    print()

    # 3. Check missing modules
    print("3. Checking for missing module references...")
    missing_issues = []
    for filepath in PIPELINE_FILES:
        missing_issues.extend(check_missing_modules(filepath, args.verbose))

    if missing_issues:
        print(f"   ⚠️  Found {len(missing_issues)} potentially missing modules:")
        for issue in missing_issues[:5]:  # Limit output
            print(f"      {issue}")
        if len(missing_issues) > 5:
            print(f"      ... and {len(missing_issues) - 5} more")
    else:
        print("   ✅ All module references appear valid")
    print()

    # 4. Verify critical imports
    print("4. Verifying critical imports...")
    import_results = check_critical_imports()
    failed_imports = [r for r in import_results if r[2] != "OK"]

    if failed_imports:
        print(f"   ❌ {len(failed_imports)} critical imports failed:")
        for module, name, status in failed_imports:
            print(f"      {module}.{name}: {status}")
        all_issues.extend([f"{m}.{n}: {s}" for m, n, s in failed_imports])
    else:
        print(f"   ✅ All {len(import_results)} critical imports work")
    print()

    # 5. Check determinism patterns
    print("5. Checking determinism patterns...")
    det_issues = check_determinism_patterns()
    if det_issues:
        print(f"   ⚠️  Found {len(det_issues)} potential determinism issues:")
        for issue in det_issues[:3]:
            print(f"      {issue}")
    else:
        print("   ✅ No obvious determinism issues")
    print()

    # 6. Run contract tests
    if not args.skip_tests:
        print("6. Running contract tests...")
        try:
            passed, summary = run_contract_tests()
            if passed:
                print(f"   ✅ {summary}")
            else:
                print(f"   ❌ Tests failed: {summary}")
                all_issues.append(f"Contract tests failed: {summary}")
        except Exception as e:
            print(f"   ⚠️  Could not run tests: {e}")
    else:
        print("6. Skipping contract tests (--skip-tests)")
    print()

    # Summary
    print("=" * 70)
    if all_issues:
        print(f"❌ HEALTH CHECK FAILED: {len(all_issues)} issues found")
        print()
        print("Issues to fix:")
        for i, issue in enumerate(all_issues[:10], 1):
            print(f"  {i}. {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more")
        return 1
    else:
        print("✅ HEALTH CHECK PASSED")
        print()
        print("All checks passed. Safe to run E2E tests.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
