# Integration Audit Prompt

**Use this prompt to start a fresh context window for auditing TRAINING ↔ LIVE_TRADING integration.**

---

## Copy This Prompt

```
I need you to audit the integration between TRAINING and LIVE_TRADING modules in this codebase.

## Context Documents to Read First

Read these files in order:
1. `INTEGRATION_CONTRACTS.md` - The stable interface contract between modules
2. `LIVE_TRADING/IMPLEMENTATION_PROGRESS.md` - Current LIVE_TRADING status
3. `.claude/plans/modular-decomposition-master.md` - Planned TRAINING refactoring
4. `.claude/plans/interval-agnostic-pipeline.md` - Interval-agnostic implementation status

## Known Issues to Fix

The INTEGRATION_CONTRACTS.md identifies 4 critical issues:

### Issue 1: `feature_list` vs `features` Field Name Mismatch (CRITICAL)
- TRAINING writes: `"features"` to model_meta.json
- LIVE_TRADING reads: `"feature_list"`
- Fix both sides for compatibility

### Issue 2: `interval_minutes` Not Written in Symbol-Specific Path (MEDIUM)
- Cross-sectional training writes interval_minutes
- Symbol-specific training does NOT
- Phase 17 interval validation will fail

### Issue 3: Features Not Guaranteed Sorted (MEDIUM)
- Features written as `feature_names.tolist()` without sorting
- Non-deterministic feature ordering breaks reproducibility

### Issue 4: `model_checksum` Not Always Written (LOW)
- H2 security fix added checksum verification
- Not all training paths write the checksum

## Tasks

1. **Verify contract compliance**: Check that TRAINING actually writes what LIVE_TRADING expects
2. **Fix the 4 issues above**: Make minimal changes to fix integration
3. **Run tests**: Ensure both modules' tests pass after changes
4. **Update contracts**: If any schema changes, update INTEGRATION_CONTRACTS.md

## Key Files to Examine

### TRAINING (Producer)
- `TRAINING/training_strategies/execution/training.py` - Writes model_meta.json
- `TRAINING/models/specialized/core.py` - Symbol-specific training
- `TRAINING/orchestration/utils/manifest.py` - Writes manifest.json

### LIVE_TRADING (Consumer)
- `LIVE_TRADING/models/loader.py` - Reads model_meta.json
- `LIVE_TRADING/models/inference.py` - Validates interval_minutes
- `LIVE_TRADING/models/feature_builder.py` - Uses feature_list

## Success Criteria

- [ ] All 4 issues in INTEGRATION_CONTRACTS.md are resolved
- [ ] `pytest TRAINING/contract_tests/` passes
- [ ] `pytest LIVE_TRADING/tests/` passes
- [ ] INTEGRATION_CONTRACTS.md "Known Issues" section is cleared
```

---

## When to Use This Prompt

Use this prompt when:
- Starting work on TRAINING/LIVE_TRADING integration
- After major refactoring to either module
- When adding new artifacts or contract fields
- When investigating production issues with model loading

## Related Documents

- `INTEGRATION_CONTRACTS.md` - Contract specification
- `LIVE_TRADING/README.md` - Module overview
- `LIVE_TRADING/plans/03_model_integration.md` - Model integration plan
- `.claude/skills/model-inference.md` - Inference guidance
