# Phase 0: Critical Functional Fixes

**Master plan**: `dashboard-hardening-master.md`
**Status**: Pending
**Scope**: 3 files modified
**Depends on**: Nothing

---

## Context

Four features appear to work in the UI but are actually broken or stubbed. These are the highest priority because they mislead users.

---

## 0a: Wire Kill Switch to Bridge API

**File**: `DASHBOARD/dashboard/src/app.rs`
**Current**: Lines ~442-448 — Ctrl+K shows notification "Kill switch toggled" but doesn't call the bridge.
**Bridge endpoint**: POST `/api/control/kill_switch` (exists and works in server.py)

### Changes
1. Add async HTTP call to bridge endpoint when Ctrl+K pressed
2. Track kill switch state in App struct (`kill_switch_active: bool`)
3. Refresh state after toggle via GET `/api/control/status`
4. Update notification to show actual state ("Kill switch ACTIVATED" / "Kill switch deactivated")
5. Show kill switch state in status bar (red indicator when active)

### Notes
- Phase 1 will add a confirmation dialog before toggling — for now just wire it up
- The `DashboardClient` already has a `toggle_kill_switch()` method in `api/client.rs`

---

## 0b: Implement Log Search Filtering

**File**: `DASHBOARD/dashboard/src/views/log_viewer.rs`
**Current**: `search_query: String` field exists, search input UI exists, but filtered lines are never computed.

### Changes
1. Find where lines are rendered and add filter: skip lines that don't contain `search_query` (case-insensitive)
2. Highlight matching text in search results with accent color
3. Add match count display in footer ("N matches")
4. Add `n`/`N` keys for next/previous match navigation
5. Ensure search works with tail/follow mode (new lines also filtered)

### Notes
- Keep it simple: substring match, not regex
- Empty search query shows all lines (current behavior)

---

## 0c: Fix File Browser Date Formatting

**File**: `DASHBOARD/dashboard/src/views/file_browser.rs`
**Current**: Lines ~78-82 have hand-rolled leap year calculation that produces wrong dates.

### Changes
1. Replace manual date math with `chrono::NaiveDateTime::from_timestamp_opt()` or equivalent
2. Format as `YYYY-MM-DD HH:MM` using chrono's formatting
3. `chrono` is already in `Cargo.toml` dependencies

### Notes
- Small, isolated fix — just the date formatting function

---

## 0d: Wire File Browser Enter to Open Files

**File**: `DASHBOARD/dashboard/src/views/file_browser.rs`
**Current**: Enter key navigates into directories but does nothing for files.

### Changes
1. On Enter with a file selected: if it's a text file (using existing `is_text_file()` check), return `ViewAction::SpawnEditor(path)`
2. For non-text files, show a message "Cannot open binary file"
3. The `ViewAction::SpawnEditor` + `suspend_for_editor()` infrastructure already exists from the config editor work

### Notes
- This leverages the ViewAction pattern we just implemented
- app.rs already handles `ViewAction::SpawnEditor` in `process_view_action()`, but need to ensure file_browser's action is also handled (it goes through the generic view dispatch)

---

## Verification

- [ ] Ctrl+K calls bridge endpoint and shows actual toggle state
- [ ] Log search filters displayed lines, highlights matches
- [ ] File browser dates match `ls -la` output
- [ ] Enter on a text file in file browser opens $EDITOR
- [ ] `cargo build --release` passes
