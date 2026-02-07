# Phase 4: Error Handling

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ Complete (All 10/10 items)
**Priority**: P1 (High - Silent Failures)
**Estimated Effort**: 1-2 days
**Depends On**: Phase 1

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 10/10 items (ALL items complete)
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 4 complete - proceed to Phase 5 (Config Hierarchy)
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| EH-001 | base_trainer.py bare except for seed | ✅ Complete | Strict mode + ImportError handling |
| EH-002 | leakage_auto_fixer.py bare except (2x) | ✅ Complete | Same pattern (2 locations) |
| EH-003 | leakage_sentinels.py bare except | ✅ Complete | Same pattern |
| EH-004 | target_routing.py write failure no strict | ✅ Complete | ArtifactError in strict mode |
| EH-005 | target_routing.py config fallback | ✅ Complete | ConfigError in strict mode (2 locations) |
| EH-006 | determinism.py library version logging | ✅ Complete | ImportError + Exception logging (5 libraries) |
| EH-007 | task_metrics.py silent NaN fallbacks | ✅ Complete | Error tracking + debug logging (5 locations) |
| EH-008 | feature_selector.py config fallback | ✅ Complete | ConfigError in strict mode (2 locations) |
| EH-009 | constants.py availability checks | ✅ Complete | ImportError + Exception logging |
| EH-010 | utils.py one-liner except | ✅ Complete | Debug logging added | |

---

## Problem Statement

30+ error handling violations including:
- Bare `except:` clauses that swallow all errors
- Hardcoded fallbacks without strict mode checks
- Artifact operations that warn but don't fail-closed
- Seed derivation failures silently using fallback values

---

## Issue Details

### EH-001: Bare except for BASE_SEED import (P0)

**File**: `TRAINING/model_fun/base_trainer.py`
**Line**: 60

```python
# CURRENT
try:
    from TRAINING.common.determinism import BASE_SEED
    ridge_seed = BASE_SEED if BASE_SEED is not None else 42
except:
    ridge_seed = 42
```

**Fix**:
```python
try:
    from TRAINING.common.determinism import BASE_SEED, is_strict_mode
    if BASE_SEED is None:
        if is_strict_mode():
            raise ConfigError("BASE_SEED not initialized in strict mode")
        ridge_seed = 42
        logger.warning("BASE_SEED is None, using fallback 42")
    else:
        ridge_seed = BASE_SEED
except ImportError as e:
    if is_strict_mode():
        raise ConfigError(f"Determinism module not available: {e}") from e
    logger.warning(f"Using fallback seed 42: {e}")
    ridge_seed = 42
```

---

### EH-002: leakage_auto_fixer.py bare except (P0)

**File**: `TRAINING/common/leakage_auto_fixer.py`
**Lines**: 423, 515

Same pattern as EH-001. Apply same fix template.

---

### EH-003: leakage_sentinels.py bare except (P0)

**File**: `TRAINING/common/leakage_sentinels.py`
**Line**: 317

Same pattern as EH-001. Apply same fix template.

---

### EH-004: Routing decision write failure no strict check (P0)

**File**: `TRAINING/ranking/target_routing.py`
**Lines**: 460-463

```python
# CURRENT
except Exception as e:
    logger.warning(f"[WARN] Failed to save routing decision for {target}: {e}")
```

**Fix**:
```python
except Exception as e:
    from TRAINING.common.determinism import is_strict_mode
    if is_strict_mode():
        raise ArtifactError(
            f"Failed to save routing decision for {target}",
            artifact_path=decision_path,
            stage="FEATURE_SELECTION"
        ) from e
    logger.warning(f"[WARN] Failed to save routing decision for {target}: {e}")
```

---

### EH-005: Config load fallback to hardcoded thresholds (P1)

**File**: `TRAINING/ranking/target_routing.py`
**Lines**: 71-77, 280-286

```python
# CURRENT
except Exception:
    T_cs = 0.65
    T_frac = 0.5
```

**Fix**:
```python
except Exception as e:
    from TRAINING.common.determinism import is_strict_mode
    if is_strict_mode():
        raise ConfigError(
            f"Failed to load routing thresholds: {e}",
            config_key="routing.thresholds",
            stage="ROUTING"
        ) from e
    # Documented fallbacks (see CONFIG/defaults.yaml)
    T_cs = 0.65
    T_frac = 0.5
    logger.warning(f"Using fallback thresholds T_cs={T_cs}, T_frac={T_frac}")
```

---

### EH-006: Library version logging bare except (P2)

**File**: `TRAINING/common/determinism.py`
**Lines**: 357-381

```python
# CURRENT (5 bare excepts)
try:
    import numpy
    versions['numpy'] = numpy.__version__
except:
    pass
```

**Fix**:
```python
try:
    import numpy
    versions['numpy'] = numpy.__version__
except ImportError:
    versions['numpy'] = 'not installed'
except Exception as e:
    versions['numpy'] = f'error: {e}'
    logger.debug(f"Failed to get numpy version: {e}")
```

---

### EH-007: Metric calculations silent NaN fallback (P1)

**File**: `TRAINING/common/utils/task_metrics.py`
**Lines**: 65, 115, 121, 169, 179

```python
# CURRENT
except:
    metrics["ic"] = 0.0

except:
    metrics["roc_auc"] = np.nan
```

**Fix**:
```python
except Exception as e:
    logger.warning(f"Failed to compute IC: {e}")
    metrics["ic"] = np.nan
    metrics["ic_error"] = str(e)  # Track why it failed
```

---

### EH-008: feature_selector.py config fallback (P1)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 252-262

```python
# CURRENT
except Exception:
    max_samples_per_symbol = 50000
```

**Fix**:
```python
except Exception as e:
    from TRAINING.common.determinism import is_strict_mode
    if is_strict_mode():
        raise ConfigError(
            f"Failed to load max_samples_per_symbol: {e}",
            config_key="pipeline.data_limits.default_max_rows_per_symbol_ranking",
            stage="FEATURE_SELECTION"
        ) from e
    max_samples_per_symbol = 50000
    logger.warning(f"Using fallback max_samples_per_symbol={max_samples_per_symbol}")
```

---

### EH-009: TensorFlow/NGBoost availability checks (P3)

**File**: `TRAINING/models/specialized/constants.py`
**Lines**: 24-25, 30-31

```python
# CURRENT
except:
    return False
```

**Fix**:
```python
except ImportError:
    return False
except Exception as e:
    logger.debug(f"Unexpected error checking TensorFlow availability: {e}")
    return False
```

---

### EH-010: One-liner except (P3)

**File**: `TRAINING/training_strategies/utils.py`
**Line**: 115

```python
# CURRENT
except Exception: return "n/a"
```

**Fix**:
```python
except Exception as e:
    logger.debug(f"Failed to get value: {e}")
    return "n/a"
```

---

## Pattern Template

For all seed/config fallbacks:

```python
def _get_seed_safe(context: str) -> int:
    """Get seed with strict mode enforcement."""
    try:
        from TRAINING.common.determinism import BASE_SEED, is_strict_mode, stable_seed_from

        if BASE_SEED is None:
            if is_strict_mode():
                raise ConfigError(
                    f"BASE_SEED not initialized for {context}",
                    config_key="pipeline.determinism.base_seed",
                    stage=context.upper()
                )
            return 42  # Documented fallback

        return stable_seed_from([context])

    except ImportError as e:
        if is_strict_mode():
            raise ConfigError(f"Determinism module not available: {e}") from e
        return 42
```

---

## Implementation Steps

### Step 1: Create error handling helper
```python
# TRAINING/common/utils/error_helpers.py

def get_config_or_fail(
    config_key: str,
    default: Any,
    stage: str,
    strict_mode: Optional[bool] = None
) -> Any:
    """Get config value, failing in strict mode if not available."""
    from TRAINING.common.determinism import is_strict_mode as _is_strict

    strict = strict_mode if strict_mode is not None else _is_strict()

    try:
        from CONFIG.config_loader import get_cfg
        return get_cfg(config_key, default=default)
    except Exception as e:
        if strict:
            raise ConfigError(
                f"Failed to load {config_key}: {e}",
                config_key=config_key,
                stage=stage
            ) from e
        logger.warning(f"Using fallback {config_key}={default}")
        return default
```

### Step 2: Replace bare except with specific exceptions
- `except:` → `except ImportError:` for imports
- `except:` → `except (TypeError, ValueError):` for conversions
- Add logging for unexpected exceptions

### Step 3: Add strict mode checks to P0 items
- EH-001, EH-002, EH-003, EH-004

### Step 4: Add strict mode checks to P1 items
- EH-005, EH-007, EH-008

### Step 5: Improve P2/P3 items
- Add specific exception types
- Add debug logging

---

## Contract Tests

```python
# tests/contract_tests/test_error_handling_contract.py

class TestStrictModeErrorHandling:
    def test_seed_fails_in_strict_mode_when_uninitialized(self):
        """Seed access should fail in strict mode if not initialized."""
        with strict_mode_enabled():
            with pytest.raises(ConfigError, match="BASE_SEED not initialized"):
                from TRAINING.model_fun.base_trainer import get_ridge_seed
                get_ridge_seed()

    def test_routing_write_fails_in_strict_mode(self):
        """Routing write failure should raise in strict mode."""
        with strict_mode_enabled():
            with pytest.raises(ArtifactError):
                save_routing_decision(target="test", decision={}, path="/invalid/path")

    def test_config_fallback_logs_warning(self):
        """Config fallback should log warning in best-effort mode."""
        with caplog.at_level(logging.WARNING):
            with strict_mode_disabled():
                value = get_config_or_fail("nonexistent.key", default=42, stage="TEST")

        assert value == 42
        assert "Using fallback" in caplog.text
```

---

## Verification

```bash
# Find bare except statements
grep -rn "except:" TRAINING/ | grep -v "except Exception" | grep -v test

# Find hardcoded fallbacks after except
grep -rn "except.*:" TRAINING/ -A2 | grep -E "= [0-9]+" | head -20

# Run error handling tests
pytest tests/contract_tests/test_error_handling_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 10 issues
- Created error handling helper template
- **Next**: Wait for Phase 1, then implement helper and fix P0 items

### Session 2: 2026-01-19
- Completed ALL 10 items:
  - **EH-001**: Fixed bare except in `base_trainer.py` - added strict mode + ImportError handling
  - **EH-002**: Fixed bare except in `leakage_auto_fixer.py` (2 locations) - same pattern
  - **EH-003**: Fixed bare except in `leakage_sentinels.py` - same pattern
  - **EH-004**: Added strict mode to routing write failure in `target_routing.py`
  - **EH-005**: Added strict mode to config fallback in `target_routing.py` (2 locations)
  - **EH-006**: Fixed library version logging in `determinism.py` - ImportError + debug logging
  - **EH-007**: Fixed silent NaN fallbacks in `task_metrics.py` - error tracking + debug logging
  - **EH-008**: Added strict mode to feature_selector fallback (2 locations)
  - **EH-009**: Fixed TF/NGBoost availability checks in `constants.py`
  - **EH-010**: Fixed one-liner except in `utils.py`
- **Phase 4 Complete** ✅ (All 10/10 items)

---

## Notes

- Track all fallback values in CONFIG/defaults.yaml for documentation
- Consider adding `--strict` CLI flag for testing
- Coordinate with Phase 5 (Config Hierarchy) for consistent pattern
- All error handling now uses specific exception types (ImportError, Exception)
- Strict mode checks use `ConfigError` and `ArtifactError` from typed exceptions
