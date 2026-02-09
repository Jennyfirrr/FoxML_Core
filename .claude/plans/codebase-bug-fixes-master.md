# Codebase Bug Fixes — Master Plan

**Status**: In progress
**Created**: 2026-02-08
**Branch**: `analysis/code-review-and-raw-ohlcv`

## Context

A comprehensive code review across the entire codebase identified ~48 bugs. The 7 most critical have been fixed. This plan tracks the remaining bugs organized by component and severity.

## Progress Summary

| Phase | Component | Status | CRITICAL | HIGH | MEDIUM | LOW |
|-------|-----------|--------|----------|------|--------|-----|
| 0 | Dashboard (round 1) | **Complete** | 3/3 | 2/2 | 5/5 | 1/1 |
| 1 | CRITICAL bugs (all components) | **Complete** | 7/7 | — | — | — |
| 2 | LIVE_TRADING remaining | Pending | 0 | 5 | 4 | 0 |
| 3 | TRAINING remaining | Pending | 0 | 1 | 5 | 4 |
| 4 | DATA_PROCESSING remaining | Pending | 0 | 0 | 0 | 0 |
| 5 | CONFIG remaining | Pending | 0 | 0 | 0 | 0 |
| 6 | DASHBOARD remaining | Pending | 0 | 6 | 5 | 3 |

## Completed Fixes

### Phase 0: Dashboard Bug Fixes (Round 1) — COMPLETE
**Plan file**: `dashboard-bug-fixes-master.md`
13 bugs fixed across Python bridge and Rust TUI (6 bridge + 2 panic + 5 defensive).

### Phase 1: CRITICAL Bug Fixes (All Components) — COMPLETE
All 7 CRITICAL bugs fixed and verified (Python syntax OK, Rust builds clean).

| # | Bug | File | Fix |
|---|-----|------|-----|
| C1 | Sell-side shares always positive | `trading_engine.py:1068` | Use signed shares from sizing_result.side |
| C2 | Barrier gate args swapped at call site | `trading_engine.py:983` | Pass actual values without swapping |
| C3 | Barrier target indentation (93% data loss) | `barrier.py:389` | Indent into inner for-loop |
| C4 | Missing interval_minutes in pipeline | `barrier_pipeline.py:287-317` | Parse from dir name, pass to all 4 calls |
| C5 | Config attribute error | `config_builder.py:566` | `experiment_cfg.data.bar_interval` |
| C6 | Nested tokio runtime panic | `trading.rs:417,426` | `block_in_place` + `Handle::current()` |
| C7 | Use-after-release in data prep | `data_preparation.py:390` | Pass `detected_interval` as parameter |

---

## Pending Fixes

### Phase 2: LIVE_TRADING Remaining Bugs

**Sub-plan file**: `bugfix-phase2-live-trading.md` (to be created)

| # | Bug | File | Severity | Description |
|---|-----|------|----------|-------------|
| L1 | CILS stats KeyError crash | `trading_engine.py:1513` | HIGH | `stats["bandit_stats"]["arm_stats"]` — key may not exist if bandit has 0 steps |
| L2 | fill_price defaults to 0.0 | `trading_engine.py:1148` | HIGH | `result.get("fill_price", 0.0)` — 0.0 fill price corrupts position tracking |
| L3 | Timezone mismatch in cooldown | `trading_engine.py:1089` | HIGH | `current_time - last_trade` may compare naive vs aware datetimes |
| L4 | Momentum period wraparound | `blending/horizon_blender.py` | HIGH | Momentum window can exceed available history |
| L5 | No short selling support | `brokers/paper_broker.py` | HIGH | Paper broker rejects negative qty without error |
| L6 | Dict mutation during iteration | `engine/state.py` | MEDIUM | `positions` dict modified during iteration |
| L7 | Paper broker field name mismatch | `brokers/paper_broker.py` | MEDIUM | Returns `avg_price` but engine expects `fill_price` |
| L8 | ZeroDivisionError in vol scaling | `sizing/vol_scaling.py` | MEDIUM | Division by zero when volatility is exactly 0.0 |
| L9 | TOCTOU in state file operations | `engine/state.py` | MEDIUM | Check-then-write race condition |

### Phase 3: TRAINING Remaining Bugs

**Sub-plan file**: `bugfix-phase3-training.md` (to be created)

| # | Bug | File | Severity | Description |
|---|-----|------|----------|-------------|
| T1 | Module-level np.random.seed(42) | `utils.py:217-219` | HIGH | Overwrites determinism framework seed on every import |
| T2 | Duplicate "seed" dict key (XGBoost) | `determinism.py:303` | MEDIUM | May be missing a distinct seed parameter |
| T3 | Bare `from common.xxx` imports | `training.py:259,888,...` | MEDIUM | Fragile relative imports in 6 locations |
| T4 | Hardcoded seed in raw sequence downsample | `data_preparation.py:1082` | MEDIUM | Uses `np.random.seed(42)` instead of `stable_seed_from()` |
| T5 | Dead code in `_extract_view()` | `reproducibility_tracker.py:746` | MEDIUM | Early return bypasses robust View enum normalization |
| T6 | Inconsistent dir naming (n_effective vs bin) | `pipeline_stages.py:377` | MEDIUM | Fallback path uses raw number instead of bin name |
| T7 | `_get_sample_size_bin()` return not unpacked | `pipeline_stages.py:314` | LOW | Logs entire dict instead of bin_name string |
| T8 | feature_names refers to last-iteration value | `training.py:1367` | LOW | Metadata filename uses last family from inner loop |
| T9 | `'X' in locals()` check is fragile | `training.py:1309,1425,1441` | LOW | Should initialize X=None and check `is not None` |
| T10 | route_info re-assigned redundantly | `training.py:1690` | LOW | Re-reads same value, potential inconsistency |

### Phase 4: DATA_PROCESSING Remaining Bugs
All CRITICAL bugs fixed in Phase 1. No remaining HIGH/MEDIUM bugs identified.

### Phase 5: CONFIG Remaining Bugs
All CRITICAL bugs fixed in Phase 1. No remaining HIGH/MEDIUM bugs identified.

### Phase 6: DASHBOARD Remaining Bugs

**Sub-plan file**: `bugfix-phase6-dashboard.md` (to be created)

| # | Bug | File | Severity | Description |
|---|-----|------|----------|-------------|
| D1 | Race condition on ws_connecting flag | `trading.rs` | HIGH | Flag checked without lock, can double-connect |
| D2 | Alpaca WebSocket reconnection infinite loop | `server.py` | HIGH | Reconnect loop has no backoff or max retries |
| D3 | Training file watcher blocks event loop | `server.py` | HIGH | Synchronous file I/O in async context |
| D4 | Theme color fallback crash | `themes.rs` | HIGH | Missing color key panics instead of defaulting |
| D5 | Event log unbounded memory growth | `widgets/event_log.rs` | HIGH | No eviction when events exceed max_entries |
| D6 | Config editor doesn't validate YAML | `views/config_editor.rs` | HIGH | Saves invalid YAML without warning |
| D7 | Sidebar selection wraps to usize::MAX | `ui/sidebar.rs` | MEDIUM | `selected - 1` underflows when selected=0 |
| D8 | Training view run list not sorted | `views/training.rs` | MEDIUM | Runs displayed in arbitrary HashMap order |
| D9 | Health endpoint no timeout | `bridge/server.py` | MEDIUM | Health check blocks indefinitely |
| D10 | Missing CORS headers | `bridge/server.py` | MEDIUM | Browser-based tools can't reach bridge |
| D11 | Run manager start_run fire-and-forget | `views/run_manager.rs` | MEDIUM | No feedback on success/failure |
| D12 | Service manager systemd assumption | `views/service_manager.rs` | LOW | Hard-codes systemd, fails on non-systemd |
| D13 | Model selector pagination off-by-one | `views/model_selector.rs` | LOW | Last page shows 1 fewer item |
| D14 | Log viewer doesn't follow tail | `views/log_viewer.rs` | LOW | New lines don't auto-scroll |

## Dependency Graph

```
Phase 0 (Dashboard round 1)    — COMPLETE
Phase 1 (CRITICAL all)          — COMPLETE
Phase 2 (LIVE_TRADING)          — standalone, priority 1
Phase 3 (TRAINING)              — standalone, priority 2
Phase 4 (DATA_PROCESSING)       — nothing remaining
Phase 5 (CONFIG)                — nothing remaining
Phase 6 (DASHBOARD remaining)   — standalone, priority 3
```

Phases 2, 3, and 6 are independent and can be done in any order.
Priority ordering: LIVE_TRADING (trading correctness) > TRAINING (determinism) > DASHBOARD (UX).

## Verification

After all phases:
- [ ] Python syntax check on all modified files
- [ ] `cargo build --release` for Rust TUI
- [ ] `pytest` for contract tests (if available)
- [ ] Manual review of sell-side trade flow
- [ ] Verify determinism framework not corrupted by stray seeds

## Session Notes

### 2026-02-08: Phase 0 complete
- 13 dashboard bugs fixed (bridge + Rust TUI)

### 2026-02-08: Phase 1 complete
- 7 CRITICAL bugs fixed across all components
- All Python syntax checks pass, Rust builds clean
- Remaining: ~30 bugs across HIGH/MEDIUM/LOW severity
