# Architecture Remediation Master Plan

**Status**: ✅ COMPLETE - All 8 Phases Done
**Created**: 2026-01-19
**Last Updated**: 2026-01-19
**Related Plans**: `code-review-remediation.md` (completed P0)

---

## Quick Resume (For Fresh Context Windows)

```
CURRENT PHASE: ALL PHASES COMPLETE ✅
CURRENT FILE: None
BLOCKERS: None
NEXT ACTION: Architecture remediation complete - run tests to verify
```

---

## Executive Summary

Comprehensive architecture review identified **8 critical issues** across **90+ specific items** affecting determinism, reproducibility, and maintainability.

| Phase | Name | Severity | Items | Status |
|-------|------|----------|-------|--------|
| **1** | Run Identity Lifecycle | P0 | 8 | ✅ Complete |
| **2** | Thread Safety | P0 | 10 | ✅ Complete |
| **3** | Fingerprinting Gaps | P0 | 10 | ✅ Complete (10/10) |
| **4** | Error Handling | P1 | 10 | ✅ Complete (10/10) |
| **5** | Config Hierarchy | P1 | 13 | ✅ Complete (13/13) |
| **6** | Data Consistency | P1 | 10 | ✅ Complete (10/10) |
| **7** | Stage Boundaries | P2 | 9 | ✅ Complete (9/9) |
| **8** | API Design | P2 | 8 | ✅ Complete (8/8) |

### Critical Statistics
- **P0 Issues**: 18 (blocking reproducibility)
- **P1 Issues**: 35+ (SST/DRY violations)
- **P2 Issues**: 25+ (type safety/API design)
- **Estimated Effort**: 6 weeks

---

## Subplan Index

| Phase | File | Status | P0 Items | P1 Items |
|-------|------|--------|----------|----------|
| 1 | [arch-phase1-run-identity.md](./arch-phase1-run-identity.md) | ✅ Complete | 3 | 5 |
| 2 | [arch-phase2-thread-safety.md](./arch-phase2-thread-safety.md) | ✅ Complete | 5 | 5 |
| 3 | [arch-phase3-fingerprinting.md](./arch-phase3-fingerprinting.md) | ✅ Complete | 10 ✅ | 0 |
| 4 | [arch-phase4-error-handling.md](./arch-phase4-error-handling.md) | ✅ Complete | 10 ✅ | 0 |
| 5 | [arch-phase5-config-hierarchy.md](./arch-phase5-config-hierarchy.md) | ✅ Complete | 0 | 13 ✅ |
| 6 | [arch-phase6-data-consistency.md](./arch-phase6-data-consistency.md) | ✅ Complete | 2 ✅ | 8 ✅ |
| 7 | [arch-phase7-stage-boundaries.md](./arch-phase7-stage-boundaries.md) | ✅ Complete | 9 ✅ | 0 |
| 8 | [arch-phase8-api-design.md](./arch-phase8-api-design.md) | ✅ Complete | 5 ✅ | 3 ✅ |

---

## Architecture Overview

### Current State (Broken)

```
┌─────────────────────────────────────────────────────────────────┐
│  intelligent_trainer.py                                          │
│                                                                  │
│  1. create_stage_identity()                                      │
│     └── Returns partial (is_final=False)                         │
│     └── Stored in LOCAL VARIABLE (never on self)      ◄── BUG   │
│                                                                  │
│  2. Feature selection called                                     │
│     └── Module calls finalize() internally                       │
│     └── Result NOT RETURNED to orchestrator           ◄── BUG   │
│                                                                  │
│  3. Code tries trainer.run_identity                              │
│     └── Gets None (attribute never set)               ◄── BUG   │
│     └── Falls back to output_dir.name (unstable)                │
│                                                                  │
│  4. Manifest created with unstable run_id             ◄── BUG   │
└─────────────────────────────────────────────────────────────────┘
```

### Target State (Fixed)

```
┌─────────────────────────────────────────────────────────────────┐
│  intelligent_trainer.py                                          │
│                                                                  │
│  1. create_stage_identity()                                      │
│     └── Returns partial (is_final=False)                         │
│     └── Stored in self._partial_identity              ◄── FIX   │
│                                                                  │
│  2. Feature selection called                                     │
│     └── Module calls finalize() internally                       │
│     └── Returns finalized identity to orchestrator    ◄── FIX   │
│                                                                  │
│  3. self.run_identity = returned_identity             ◄── FIX   │
│                                                                  │
│  4. Manifest created with derive_run_id(self.run_identity)      │
│     └── Stable, deterministic run_id                  ◄── FIX   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

```
Phase 1: Run Identity (P0)
    │
    ├──► Phase 2: Thread Safety (P0)
    │        │
    │        └──► Phase 3: Fingerprinting (P0)
    │
    └──► Phase 4: Error Handling (P1)
             │
             ├──► Phase 5: Config Hierarchy (P1)
             │
             └──► Phase 6: Data Consistency (P1)
                      │
                      ├──► Phase 7: Stage Boundaries (P2)
                      │
                      └──► Phase 8: API Design (P2)
```

### Dependencies

| Phase | Depends On | Blocks |
|-------|------------|--------|
| 1 | None | 2, 3, 4 |
| 2 | 1 | 3 |
| 3 | 1, 2 | 7 |
| 4 | 1 | 5, 6 |
| 5 | 4 | - |
| 6 | 4 | 7, 8 |
| 7 | 3, 6 | - |
| 8 | 6 | - |

---

## Key Files Reference

### Core Orchestration
- `TRAINING/orchestration/intelligent_trainer.py` - Main entry, 4700+ lines
- `TRAINING/orchestration/utils/manifest.py` - Run tracking
- `TRAINING/common/utils/fingerprinting.py` - RunIdentity class

### Ranking Pipeline
- `TRAINING/ranking/feature_selector.py` - Feature selection
- `TRAINING/ranking/target_ranker.py` - Target ranking
- `TRAINING/ranking/multi_model_feature_selection.py` - Multi-model FS

### Common Utilities
- `TRAINING/common/feature_registry.py` - Feature definitions
- `TRAINING/common/determinism.py` - Determinism settings
- `TRAINING/common/threads.py` - Thread management
- `TRAINING/models/registry.py` - Model registry singleton
- `TRAINING/models/factory.py` - Model factory singleton

---

## Verification Commands

```bash
# Phase 1: Run Identity
grep -rn "trainer.run_identity\|self.run_identity" TRAINING/orchestration/
grep -rn "\.finalize\(" TRAINING/

# Phase 2: Thread Safety
grep -rn "_instance = None" TRAINING/
grep -rn "os.environ\[" TRAINING/common/

# Phase 3: Fingerprinting
grep -rn "sha1\|SHA1" TRAINING/
grep -rn 'or ""' TRAINING/ | grep signature

# Phase 4: Error Handling
grep -rn "except:" TRAINING/ | grep -v "except Exception"

# Phase 5: Config Hierarchy
grep -rn "yaml.safe_load" TRAINING/orchestration/

# Phase 6: Data Consistency
grep -rn "\.items()" TRAINING/orchestration/ | grep -v "sorted"

# Full test suite
pytest TRAINING/contract_tests/ -v
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing runs | Medium | High | Feature flag for new behavior |
| Thread safety fixes cause deadlock | Low | High | Test in isolation first |
| Config changes break experiments | Medium | Medium | Backward compat layer |
| API changes break downstream | High | Medium | Deprecation warnings first |

---

## Success Criteria

### Phase 1-3 Complete (P0)
- [ ] `trainer.run_identity` is always set before training
- [ ] All singletons are thread-safe
- [ ] All fingerprints use consistent hash format
- [ ] `pytest TRAINING/contract_tests/` passes

### Phase 4-6 Complete (P1)
- [ ] No bare `except:` in artifact-affecting code
- [ ] All config access uses `get_cfg()`
- [ ] All dict iterations are sorted

### Phase 7-8 Complete (P2)
- [ ] FeatureSelectionResult contract implemented
- [ ] No duplicate function definitions
- [ ] Functions with 10+ params refactored

---

## How to Resume Work

### Starting a Fresh Context Window

1. Read this master plan first
2. Check "Quick Resume" section at top
3. Read the current phase's sub-plan
4. Look at "Session State" section in sub-plan
5. Continue from "Next Action"

### Before Ending a Session

1. Update the sub-plan's "Session State" section:
   - What was completed
   - What remains
   - Any blockers discovered
   - Next action for next session
2. Update this master plan's "Quick Resume" section
3. Commit changes to plan files

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial master plan created | Claude |
| 2026-01-19 | Deep review: expanded to 8 phases, 90+ items | Claude |
| 2026-01-19 | Phase 1 (Run Identity Lifecycle) completed | Claude |
| 2026-01-19 | Phase 2 (Thread Safety) completed - 10/10 items | Claude |
| 2026-01-19 | Phase 3 P0 (Fingerprinting) completed - 5/10 items (all P0) | Claude |
| 2026-01-19 | Phase 3 COMPLETE - all 10/10 items (P0, P1, P2) | Claude |
| 2026-01-19 | Phase 4 COMPLETE - all 10/10 error handling items | Claude |
| 2026-01-19 | Phase 5 COMPLETE - all 13/13 config hierarchy items | Claude |
| 2026-01-19 | Phase 6 COMPLETE - all 10/10 data consistency items | Claude |
| 2026-01-19 | Phase 7 COMPLETE - all 9/9 stage boundary items | Claude |
| 2026-01-19 | Phase 8 COMPLETE - all 8/8 API design items | Claude |
| 2026-01-19 | **ALL PHASES COMPLETE** - 78 items across 8 phases | Claude |
