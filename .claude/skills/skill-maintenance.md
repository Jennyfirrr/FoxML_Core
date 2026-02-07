# Skill Maintenance

Guidelines for when and how to update skills files. Updates should be rare and only for significant issues.

## When to Update Skills

**ONLY update a skill file when a MAJOR problem is identified:**

| Trigger | Example | Action |
|---------|---------|--------|
| Incorrect import path | `from TRAINING.common.utils.file_utils import run_root` but function is in `target_first_paths` | Fix the import path |
| Broken code example | Example uses deleted function or wrong signature | Fix or remove the example |
| Factually wrong information | Skill says "use X" but X causes errors or is deprecated | Correct the information |
| Security issue | Skill recommends insecure pattern | Fix immediately |

## When NOT to Update Skills

**Do NOT update skills for:**

- Style preferences or "improvements"
- Adding more examples (unless existing ones are broken)
- Reorganizing or restructuring
- Minor clarifications that don't affect correctness
- New features (unless old documentation is now wrong)
- "Nice to have" additions

## Update Process

1. **Verify the problem** - Confirm the issue exists in actual code (grep, read files)
2. **Minimal fix** - Change only what's necessary to fix the problem
3. **No scope creep** - Don't "while I'm here" improve other things
4. **Note in commit** - Mention the specific fix (e.g., "fixed import path in model-inference.md")

## Examples

**CORRECT update trigger:**
```
During code review, discovered model-inference.md says:
  from TRAINING.common.utils.file_utils import run_root
But run_root is actually in:
  TRAINING.orchestration.utils.target_first_paths

This is a major problem - the example won't work. Fix it.
```

**INCORRECT update trigger:**
```
The testing-guide.md could benefit from more examples of
fixture usage patterns.

This is an enhancement, not a fix. Do not update.
```

## Scope

This policy applies to all files in `.claude/skills/` and `CLAUDE.md`.
