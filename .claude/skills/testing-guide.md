# Testing Guide

Skill for running tests, writing new tests, and maintaining test quality.

## Test Commands

### Running Tests

```bash
# Run all tests (uses testpaths from pyproject.toml)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest TRAINING/contract_tests/test_determinism_strict.py

# Run specific test function
pytest TRAINING/contract_tests/test_determinism_strict.py::test_function_name

# Run tests matching pattern
pytest -k "determinism"  # Runs tests with "determinism" in name

# Run with coverage
pytest --cov=TRAINING --cov=CONFIG tests/

# Run only failed tests from last run
pytest --lf
```

### Linting and Type Checking

```bash
# Ruff lint check
ruff check .

# Ruff lint with auto-fix
ruff check --fix .

# Ruff format
ruff format .

# Type checking
mypy TRAINING/

# Type checking specific module
mypy TRAINING/orchestration/utils/
```

### Determinism Verification

```bash
# Check code for determinism violations
bash bin/check_determinism_patterns.sh

# Verify determinism bootstrap is correct
python bin/verify_determinism_init.py
```

## Test Categories

### Contract Tests (Primary)

**Location:** `TRAINING/contract_tests/`

Contract tests verify critical invariants:
- Determinism guarantees
- Stage boundary contracts
- SST compliance

```bash
# Run contract tests
pytest TRAINING/contract_tests/ -v
```

### Unit Tests

**Location:** `tests/`

Unit tests for individual components:
- Model trainers
- Utility functions
- Config loading

```bash
# Run unit tests
pytest tests/ -v
```

### Smoke Tests

**Location:** `tests/test_smoke_imports.py`

Quick sanity checks:
- Import chains work
- Basic functionality

```bash
# Run smoke tests
pytest tests/test_smoke_imports.py -v
```

## Writing New Tests

### Basic Test Structure

```python
# tests/test_my_module.py
import pytest
import numpy as np

from TRAINING.my_module import MyClass


class TestMyClass:
    """Tests for MyClass."""

    def test_basic_functionality(self):
        """Test basic operation works."""
        obj = MyClass()
        result = obj.process([1, 2, 3])
        assert result is not None
        assert len(result) == 3

    def test_handles_empty_input(self):
        """Test empty input handling."""
        obj = MyClass()
        result = obj.process([])
        assert result == []

    def test_raises_on_invalid_input(self):
        """Test invalid input raises appropriate error."""
        obj = MyClass()
        with pytest.raises(ValueError, match="must be positive"):
            obj.process([-1])
```

### Pytest Fixtures

```python
import pytest


@pytest.fixture
def sample_data():
    """Create sample data for tests."""
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for tests."""
    from TRAINING.model_fun.lightgbm_trainer import LightGBMTrainer

    X = sample_data
    y = np.random.randn(100).astype(np.float64)

    trainer = LightGBMTrainer()
    trainer.train(X, y)
    return trainer


def test_model_predicts(trained_model, sample_data):
    """Test trained model can predict."""
    preds = trained_model.predict(sample_data[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
```

### SST Compliance in Tests

```python
def test_config_access_uses_sst():
    """Test that config is accessed via SST helpers."""
    from CONFIG.config_loader import get_cfg

    # CORRECT: Use get_cfg()
    value = get_cfg("pipeline.determinism.base_seed", default=42)
    assert isinstance(value, int)

    # Test config exists (not just fallback)
    from CONFIG.config_loader import validate_config_exists
    assert validate_config_exists("pipeline.determinism.base_seed", "pipeline_config")
```

### Determinism Tests

```python
def test_deterministic_output():
    """Same inputs produce identical outputs."""
    from TRAINING.common.determinism import BASE_SEED
    import numpy as np

    # Create deterministic data
    rng = np.random.default_rng(BASE_SEED or 42)
    X = rng.random((100, 10)).astype(np.float32)
    y = rng.random(100).astype(np.float64)

    # Train twice
    from TRAINING.model_fun.lightgbm_trainer import LightGBMTrainer

    trainer1 = LightGBMTrainer()
    trainer1.train(X.copy(), y.copy())
    preds1 = trainer1.predict(X[:10])

    trainer2 = LightGBMTrainer()
    trainer2.train(X.copy(), y.copy())
    preds2 = trainer2.predict(X[:10])

    # Must be identical
    np.testing.assert_array_equal(preds1, preds2)


def test_ordering_determinism():
    """Dict iteration uses deterministic ordering."""
    from TRAINING.common.utils.determinism_ordering import sorted_items

    data = {"z": 1, "a": 2, "m": 3}

    # sorted_items always returns same order
    result1 = list(sorted_items(data))
    result2 = list(sorted_items(data))

    assert result1 == result2
    assert result1 == [("a", 2), ("m", 3), ("z", 1)]  # Lexicographic
```

### Testing Error Handling

```python
def test_raises_typed_exception():
    """Test that typed exceptions are raised."""
    from TRAINING.common.exceptions import LeakageError, ConfigError

    with pytest.raises(LeakageError) as exc_info:
        raise LeakageError(
            message="Feature has lookahead bias",
            feature_name="price_close",
            horizon_minutes=10,
        )

    error = exc_info.value
    assert error.feature_name == "price_close"
    assert error.error_code == "LEAKAGE_ERROR"

    # Test structured payload
    error_dict = error.to_dict()
    assert "feature_name" in error_dict["context"]
```

## Contract Test Patterns

### Stage Boundary Contract

```python
def test_stage_boundary_contract():
    """Features selected in stage 2 available in stage 3."""
    # Stage 2 output
    selected_features = ["feat_a", "feat_b", "feat_c"]

    # Stage 3 input validation
    available_features = get_available_features(data)

    missing = set(selected_features) - set(available_features)
    assert not missing, f"Missing features: {missing}"
```

### Atomic Write Contract

```python
def test_atomic_write_is_atomic(tmp_path):
    """Atomic writes don't leave partial files."""
    from TRAINING.common.utils.file_utils import write_atomic_json
    import os

    path = tmp_path / "test.json"
    data = {"key": "value"}

    write_atomic_json(path, data)

    # File exists and is complete
    assert path.exists()
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data

    # No temp files left behind
    temp_files = list(tmp_path.glob("*.tmp*"))
    assert not temp_files
```

## Pytest Configuration

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["TRAINING/contract_tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
```

## Coverage Requirements

```bash
# Run with coverage
pytest --cov=TRAINING --cov=CONFIG --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

Coverage configuration in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["TRAINING", "CONFIG"]
omit = ["*/tests/*", "*/__pycache__/*", "*/tools/*"]
```

## Debugging Tests

```bash
# Run with full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show local variables on failure
pytest -l
```

## Related Skills

- `determinism-and-reproducibility.md` - Determinism test patterns
- `debugging-pipelines.md` - Troubleshooting test failures

## Related Documentation

- `pyproject.toml` - Pytest/coverage configuration
- `TRAINING/contract_tests/` - Contract test examples
- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Determinism test patterns
