# UI Phase 4: Model Selector View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 2h

---

## Current State

Existing implementation in `src/launcher/model_selector.rs` (30 lines):
- Just a placeholder

**Needs full implementation.**

---

## Implementation

### Step 1: Scan RESULTS/ for trained models

Read `manifest.json` from each run to get:
- Run ID and timestamp
- Target count
- Status (completed, failed)
- Best AUC per target (from target index)

### Step 2: Display model list

Table view with:
- Run ID
- Date
- Targets trained
- Best overall AUC
- Status

### Step 3: Model detail view

On selection, show:
- All targets with AUC
- Model families used
- Feature counts
- Config fingerprint

### Step 4: Selection for live trading

Allow selecting a model to use for LIVE_TRADING:
- Update symlink or config file pointing to selected run
- Show currently active model

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/model_selector.rs` | NEW - Full implementation |
| `src/views/mod.rs` | Add ModelSelector variant |
| `src/app.rs` | Wire up ModelSelector view |
| `src/launcher/model_selector.rs` | Full rewrite |

---

## Checklist

- [ ] Scan RESULTS/ for models
- [ ] Display model list with metrics
- [ ] Add model detail view
- [ ] Add selection for live trading
- [ ] Wire up in app.rs
- [ ] Test with actual trained models
