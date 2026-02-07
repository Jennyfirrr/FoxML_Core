# Registry Patch System - Automatic Per-Target Feature Exclusion

**Date**: 2026-01-09  
**Type**: Feature Enhancement, Leakage Prevention  
**Impact**: High - Prevents data leakage with automatic per-target/per-horizon feature exclusion

## Overview

Implemented comprehensive registry patch system for automatic, per-target, per-horizon feature exclusion to prevent data leakage. Replaces previous over-aggressive global rejection with granular exclusions that only apply to specific (target, horizon) pairs.

**Core Principle**: A feature that leaks for `fwd_ret_5m` at horizon 12 bars is excluded **only** for that specific target/horizon, not globally rejected for all targets.

## Key Features

### 1. Automatic Patch Writing
- **Location**: `{output_dir}/registry_patches/{target}__{hash}.yaml`
- **Trigger**: When leakage is detected during target evaluation
- **Content**: Policy-only (exclusions), no evidence (deterministic)
- **Evidence**: Written separately to `RESULTS/audit/registry_patch_ops.jsonl` (gitignored)

### 2. Automatic Patch Loading
- **Integration**: `filter_features_for_target()` receives `registry_overlay_dir` parameter
- **Precedence**: Run patches → Persistent overrides → Unblocks → Base registry
- **Two-Phase Check**: Base eligibility (hard gate) → Overlays (soft policy)
- **Applies To**: Target ranking, feature selection, training (all stages)

### 3. Auto-Fix Rerun Wrapper
- **Function**: `evaluate_target_with_autofix()` in `TRAINING/ranking/predictability/model_evaluation.py`
- **Behavior**: Automatically re-evaluates targets after leakage fixes
- **Config-Driven**: Experiment config (SST) → Safety config → Defaults
- **Default**: Off by default in experiment config (must explicitly enable)

### 4. Patch Promotion (Explicit Ops Stage)
- **Class**: `RegistryPatchOps` in `TRAINING/common/leakage_auto_fixer.py`
- **Stage**: `REGISTRY_PATCH_OPS` (never runs during normal pipeline)
- **Config**: `CONFIG/data/registry_patches.yaml`
- **Action**: Promotes run patches to persistent storage (`CONFIG/data/feature_registry_per_target/`)

### 5. Unblocking System
- **Files**: `CONFIG/data/feature_registry_per_target/{target}__{hash}.unblock.yaml`
- **Purpose**: Cancel overlay denies (cannot override base eligibility)
- **Ops Stage**: Applied via `RegistryPatchOps` (explicit, config-driven)

### 6. Query/Explanation System
- **Module**: `TRAINING/common/registry_explainer.py`
- **Functions**: `explain_feature_exclusion()`, `query_excluded_features()`
- **Purpose**: Centralized explanation of why features are excluded

## Files Added

### Core Implementation
- `TRAINING/common/registry_patch_naming.py` - Collision-proof filename helpers (neutral module)
- `TRAINING/common/registry_explainer.py` - Query/explanation system
- `CONFIG/data/registry_patches.yaml` - Ops config for promotion/unblocking

### Documentation
- `INTERNAL/docs/REGISTRY_PATCH_SYSTEM.md` - Complete technical reference
- `INTERNAL/docs/REGISTRY_PATCH_SYSTEM_SIMPLE.md` - Simple explanation for quick understanding

## Files Modified

### Core Modules
- `TRAINING/common/feature_registry.py` - Two-phase eligibility check, unblock loading
- `TRAINING/common/leakage_auto_fixer.py` - Policy-only patches, promotion/unblock methods, `RegistryPatchOps` class
- `TRAINING/common/utils/fingerprinting.py` - `compute_registry_signature()` hashes effective policy only
- `TRAINING/ranking/predictability/model_evaluation.py` - `evaluate_target_with_autofix()` wrapper, explicit patch discovery
- `TRAINING/orchestration/utils/scope_resolution.py` - Added `REGISTRY_PATCH_OPS` stage enum

### Integration Points
- `TRAINING/ranking/target_ranker.py` - Calls `evaluate_target_with_autofix()` when enabled
- `TRAINING/ranking/utils/leakage_filtering.py` - Receives `registry_overlay_dir` parameter
- `TRAINING/ranking/feature_selector.py` - Passes `registry_overlay_dir` to filtering
- `TRAINING/ranking/multi_model_feature_selection.py` - Passes `registry_overlay_dir` to filtering
- `TRAINING/training_strategies/execution/data_preparation.py` - Passes `registry_overlay_dir` to filtering

### Configuration
- `CONFIG/experiments/determinism_test.yaml` - Added `target_ranking_overrides.auto_rerun` config
- `.gitignore` - Added `RESULTS/audit/` to prevent audit logs from being tracked

## Technical Details

### Merge Precedence (Two-Phase Check)

**Phase A: Base Eligibility (Hard Gate)**
1. Global `rejected: true` → **False** (structural leak)
2. Base `allowed_horizons` check → **False** if horizon not in list
3. Unknown feature policy → **False** if rejected

**Phase B: Overlays (Soft Policy)**
1. **Unblock** (highest priority allow) → **True** (cancels overlay denies)
2. **Run patch** excludes → **False** (highest priority deny)
3. **Persistent override** excludes → **False** (medium priority deny)
4. No overlay denies → **True**

**Critical**: Unblocks can **only cancel overlay denies**, not override base eligibility.

### Determinism Guarantees

1. **Policy-Only Patches**: No evidence in patches (deterministic)
2. **Signature Hashing**: Hashes effective policy only (exclusions - unblocks)
3. **Monotonic Updates**: Union-only semantics (features only added, never removed)
4. **Atomic Writes**: File locking with `fcntl.flock()` and `os.replace()`
5. **Collision-Proof Naming**: `{target}__{hash[:12]}.yaml` format

### Configuration Precedence (SST)

1. **Experiment Config** (if exists) - `target_ranking_overrides.auto_rerun`
2. **Safety Config** - `safety.leakage_detection.auto_rerun`
3. **Function Defaults** - `enabled: false`, `max_reruns: 3`

**Default Behavior**: Auto-rerun is **off by default** in experiment config (must explicitly enable). Safety config provides global default (`enabled: true`).

## Usage

### Enable Auto-Rerun (Experiment Config)

```yaml
# CONFIG/experiments/my_experiment.yaml
target_ranking_overrides:
  auto_rerun:
    enabled: true  # Enable automatic rerun
    max_reruns: 3  # Maximum reruns per target
    rerun_on_perfect_train_acc: true
    rerun_on_high_auc_only: false
```

### Promote Patches (Ops Stage)

```yaml
# CONFIG/data/registry_patches.yaml
promotion:
  enabled: true
  output_dirs: ["RESULTS/runs/run_123/"]
  targets: null  # or ["fwd_ret_5m"]
  review_mode: false
```

Then run `RegistryPatchOps().run_ops()` in `REGISTRY_PATCH_OPS` stage.

### Unblock Features (Ops Stage)

```yaml
# CONFIG/data/registry_patches.yaml
unblocking:
  enabled: true
  unblocks:
    - target: "fwd_ret_5m"
      bar_minutes: 5
      features:
        feature_name:
          - 12  # horizon_bars to unblock
      reason: "Manual unblock after investigation"
```

## Flow Example

```
Target Ranking (First Attempt)
  → Leakage Detected
  → Patches Written to output_dir/registry_patches/
  → Auto-Rerun (if enabled)
  → Target Ranked (with leaky features excluded)

Feature Selection (Later)
  → Reads Patches
  → Excludes Leaky Features
  → Only Safe Features Selected

Training (Later)
  → Reads Patches
  → Excludes Leaky Features
  → Only Safe Features Used
```

## Backward Compatibility

- All changes are backward compatible
- Existing feature registry behavior unchanged
- Patches are additive (union-only semantics)
- No breaking changes to existing APIs

## Related Documentation

- `INTERNAL/docs/REGISTRY_PATCH_SYSTEM.md` - Complete technical reference
- `INTERNAL/docs/REGISTRY_PATCH_SYSTEM_SIMPLE.md` - Simple explanation
- Plan: `/home/Jennifer/.cursor/plans/registry_patch_autonomy_enhancements_(tightened)_7064fb35.plan.md`
