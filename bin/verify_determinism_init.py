#!/usr/bin/env python3
"""
Verify all entry points initialize determinism before ML imports.

Scans Python files for:
- if __name__ == "__main__" blocks
- def main() functions
- Checks if file imports determinism module or calls init function
- Verifies initialization happens before numpy/pandas/sklearn imports

Valid patterns:
1. import TRAINING.common.repro_bootstrap (side-effect, no call needed)
2. from TRAINING.common.determinism import init_determinism_from_config; init_determinism_from_config()
3. from TRAINING.common.determinism import set_global_determinism; set_global_determinism()
4. Delegating entry point that imports from a module known to handle determinism
"""

import ast
import sys
from pathlib import Path
from typing import Tuple

ML_IMPORTS = {'numpy', 'pandas', 'sklearn', 'lightgbm', 'xgboost', 'torch', 'tensorflow', 'scipy'}
DETERMINISM_FUNCS = {'set_global_determinism', 'init_determinism_from_config'}

# Modules that handle determinism internally (delegating entry points can import these)
DETERMINISM_HANDLING_MODULES = {
    'TRAINING.orchestration.intelligent_trainer',
    'TRAINING.training_strategies.execution.main',
    'TRAINING.training_strategies.main',
}

# Skip these directories (non-critical tools, examples, archive)
SKIP_DIRS = {'archive', 'examples', 'tools', '__pycache__', '.git'}


def check_file(file_path: Path) -> Tuple[bool, str]:
    """
    Check if file has entry point and verifies determinism init.

    Returns:
        (is_valid, error_message)
    """
    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))

        # Check for entry point patterns
        has_main_guard = False
        has_main_func = False

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    # Check for __name__ == "__main__"
                    if (isinstance(node.test.left, ast.Name) and
                        node.test.left.id == "__name__" and
                        any(isinstance(op, ast.Eq) for op in node.test.ops)):
                        has_main_guard = True

            if isinstance(node, ast.FunctionDef) and node.name == 'main':
                has_main_func = True

        if not (has_main_guard or has_main_func):
            return (True, "")  # No entry point, skip

        # Track determinism patterns
        has_repro_bootstrap = False
        imports_determinism_func = False
        calls_determinism_func = False
        imports_determinism_module = False  # Delegating pattern
        repro_bootstrap_line = None
        determinism_call_line = None
        ml_import_line = None

        for node in ast.walk(tree):
            # Check for side-effect import of repro_bootstrap
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'repro_bootstrap' in alias.name:
                        has_repro_bootstrap = True
                        repro_bootstrap_line = node.lineno
                    if any(ml in alias.name for ml in ML_IMPORTS):
                        if ml_import_line is None:
                            ml_import_line = node.lineno

            if isinstance(node, ast.ImportFrom):
                # Check for determinism function imports
                if node.module and 'determinism' in node.module:
                    for alias in node.names:
                        if alias.name in DETERMINISM_FUNCS:
                            imports_determinism_func = True

                # Check for imports from modules that handle determinism
                if node.module in DETERMINISM_HANDLING_MODULES:
                    imports_determinism_module = True

                # Track ML imports
                if node.module:
                    for ml in ML_IMPORTS:
                        if ml in node.module:
                            if ml_import_line is None:
                                ml_import_line = node.lineno

            # Check for determinism function calls
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in DETERMINISM_FUNCS:
                    calls_determinism_func = True
                    determinism_call_line = node.lineno

        # Validation logic:
        # 1. repro_bootstrap import (side-effect) is sufficient
        # 2. determinism function import + call is sufficient
        # 3. Import from determinism-handling module is sufficient (delegating)

        if has_repro_bootstrap:
            # Verify bootstrap happens before ML imports
            if ml_import_line and repro_bootstrap_line:
                if repro_bootstrap_line > ml_import_line:
                    return (False, f"repro_bootstrap imported after ML import (line {repro_bootstrap_line} > {ml_import_line})")
            return (True, "")

        if imports_determinism_func and calls_determinism_func:
            # Verify call happens before ML imports
            if ml_import_line and determinism_call_line:
                if determinism_call_line > ml_import_line:
                    return (False, f"Determinism call after ML import (line {determinism_call_line} > {ml_import_line})")
            return (True, "")

        if imports_determinism_module:
            # Delegating entry point - the imported module handles determinism
            return (True, "")

        # No valid determinism initialization found
        if imports_determinism_func and not calls_determinism_func:
            return (False, "Imports determinism function but never calls it")

        return (False, "Missing determinism initialization (import repro_bootstrap or call init_determinism_from_config)")

    except Exception as e:
        return (False, f"Parse error: {e}")


def main():
    """Scan TRAINING directory for entry points."""
    violations = []

    training_dir = Path("TRAINING")
    if not training_dir.exists():
        print(f"ERROR: TRAINING directory not found at {training_dir.absolute()}")
        sys.exit(1)

    for py_file in training_dir.rglob("*.py"):
        # Skip test files, determinism modules, and non-critical directories
        if "test" in py_file.name.lower() or "determinism" in py_file.name:
            continue

        # Skip non-critical directories
        parts = set(py_file.parts)
        if parts & SKIP_DIRS:
            continue

        is_valid, error = check_file(py_file)
        if not is_valid:
            violations.append((py_file, error))

    if violations:
        print("ERROR: Entry points missing determinism initialization:")
        for file_path, error in violations:
            print(f"  {file_path}: {error}")
        sys.exit(1)
    else:
        print("✓ All entry points initialize determinism")
        sys.exit(0)


if __name__ == "__main__":
    main()
