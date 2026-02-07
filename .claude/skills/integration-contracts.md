# Integration Contracts

Guidelines for maintaining stable interfaces between FoxML modules.

## The Contract Document

**Always read `INTEGRATION_CONTRACTS.md` before:**
- Modifying artifact schemas (model_meta.json, manifest.json, etc.)
- Adding new fields to existing artifacts
- Changing how TRAINING writes outputs
- Changing how LIVE_TRADING reads inputs

## Core Principles

### 1. Producer-Consumer Pattern
- Each artifact has ONE producer, MANY consumers
- Producer owns the schema; consumers adapt
- Breaking changes require coordinated updates

### 2. Schema Stability
```
REQUIRED fields:  Removal = breaking change
OPTIONAL fields:  Consumers MUST handle absence
DEPRECATED:       Keep 2+ versions, log warnings
```

### 3. Field Naming Convention
```python
# WRONG: Inconsistent names across modules
TRAINING writes:    "features"
LIVE_TRADING reads: "feature_list"

# RIGHT: Same name everywhere
TRAINING writes:    "feature_list"
LIVE_TRADING reads: "feature_list"
```

### 4. Deterministic Serialization
```python
# WRONG: Non-deterministic order
metadata["features"] = feature_names.tolist()

# RIGHT: Sorted for determinism
metadata["feature_list"] = sorted(feature_names)
```

## Adding New Contract Fields

1. **Check INTEGRATION_CONTRACTS.md** for existing schema
2. **Add as OPTIONAL first** (non-breaking)
3. **Update consumers** to use new field
4. **Document in contract** with producer/consumer
5. **Promote to REQUIRED** only after all consumers handle it

## Modifying Existing Fields

```
SAFE:     Add optional field
SAFE:     Deprecate field (keep it, add warning)
UNSAFE:   Remove required field
UNSAFE:   Change field type
UNSAFE:   Change field semantics
```

## Cross-Module Testing

After any artifact change:
```bash
# Test producer
pytest TRAINING/contract_tests/ -v

# Test consumer
pytest LIVE_TRADING/tests/test_model_loader.py -v
pytest LIVE_TRADING/tests/test_inference.py -v

# Integration test
pytest LIVE_TRADING/tests/test_e2e.py -v
```

## Common Mistakes

### Mistake 1: Field Name Mismatch
```python
# Producer uses one name
metadata["features"] = [...]

# Consumer expects another
features = metadata.get("feature_list", [])  # Returns []!
```

**Fix**: Always match field names exactly, or add fallback:
```python
features = metadata.get("feature_list") or metadata.get("features", [])
```

### Mistake 2: Missing Required Fields
```python
# Some code paths skip required fields
if condition:
    metadata["interval_minutes"] = interval  # Written here
else:
    pass  # NOT written here!
```

**Fix**: Write required fields in ALL code paths, or make optional.

### Mistake 3: Non-Deterministic Order
```python
# Sets have no order
metadata["features"] = set(feature_names)

# Dicts before Python 3.7 have no order
metadata["importance"] = {f: v for f, v in zip(features, values)}
```

**Fix**: Always sort or use `sorted_items()`:
```python
metadata["feature_list"] = sorted(feature_names)
metadata["importance"] = dict(sorted_items(importance_dict))
```

## Related Documentation

- `INTEGRATION_CONTRACTS.md` - Full contract specification
- `sst-and-coding-standards.md` - SST patterns
- `determinism-and-reproducibility.md` - Determinism requirements
