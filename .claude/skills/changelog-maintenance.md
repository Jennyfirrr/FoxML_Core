# Changelog Maintenance

Guidelines for maintaining the project changelog after completing tasks.

## When to Update CHANGELOG.md

**ALWAYS update the changelog after:**
- Completing a significant feature or enhancement
- Fixing critical bugs
- Major refactoring or code reorganization
- Completing a plan phase
- Adding new integrations or contracts
- Security fixes

## Changelog Location

- **Main changelog**: `CHANGELOG.md` (project root)
- **Template**: `INTERNAL/docs/CHANGELOG_TEMPLATE.md`
- **Config changelog**: `CONFIG/CHANGELOG.md` (config-specific changes)

## Entry Format

```markdown
## YYYY-MM-DD

### Category Name

#### Feature/Change Name
- Bullet point describing what changed
- Another bullet point with details

#### Files Created
- `path/to/new/file.py`

#### Files Modified
- `path/to/modified/file.py` - Brief description of change
```

## Categories

Use these categories (from template):

| Category | Use For |
|----------|---------|
| **Added** | New features or capabilities |
| **Changed** | Changes to existing functionality |
| **Fixed** | Bug fixes |
| **Security** | Security improvements |
| **Documentation** | Doc updates (public-facing only) |
| **Deprecated** | Features to be removed |
| **Removed** | Removed features |

## What NOT to Include

Per `INTERNAL/docs/CHANGELOG_TEMPLATE.md`:

- ❌ Internal documentation (`INTERNAL/docs/`)
- ❌ Internal planning documents
- ❌ Research or experimental features not yet public
- ❌ Internal processes or workflow changes
- ✅ Only user-facing, public-appropriate changes

## Quick Update Pattern

After completing work:

```python
# 1. Identify what was done
changes = {
    "Added": ["New feature X"],
    "Fixed": ["Bug in Y"],
    "Files Created": ["path/to/new.py"],
    "Files Modified": ["path/to/existing.py - added feature"]
}

# 2. Add entry to CHANGELOG.md under today's date
# 3. Use the category format from template
```

## Example Entry

```markdown
## 2026-01-19

### Integration Contract Fixes

Fixed 4 critical integration issues between TRAINING and LIVE_TRADING modules.

#### Fixed
- `feature_list` vs `features` field name mismatch - TRAINING now writes both
- `interval_minutes` not written in symbol-specific path - added to all paths
- Features not guaranteed sorted - added `_get_sorted_feature_list()` helper
- `model_checksum` not always written - added SHA256 computation

#### Added
- `INTEGRATION_CONTRACTS.md` - Cross-module artifact schemas
- `.claude/skills/integration-contracts.md` - Integration guidelines
- `.claude/prompts/integration-audit.md` - Fresh context audit prompt

#### Files Modified
- `TRAINING/training_strategies/execution/training.py` - Added contract helpers
- `TRAINING/models/specialized/core.py` - Added feature_list field
- `LIVE_TRADING/models/loader.py` - Added backward-compatible fallback
- `LIVE_TRADING/models/inference.py` - Added fallback for feature_list
```

## Automation Reminder

After completing any task that involves:
1. Creating new files
2. Modifying existing code
3. Fixing bugs
4. Adding features

**→ Update CHANGELOG.md before considering the task complete**
