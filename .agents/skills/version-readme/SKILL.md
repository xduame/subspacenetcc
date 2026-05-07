---
name: version-readme
description: Update README.md, CHANGELOG.md, and optional per-version release notes when the user asks to document a new project version, release, milestone, experiment version, or reproducibility state.
---

# Version README Skill

Use this skill when the user asks to document a new version, release, milestone, experiment revision, or reproducibility checkpoint.

## Goal

Maintain accurate, version-aware project documentation without fabricating results.

The repository should normally contain:

- `README.md`: current project overview and current usage.
- `CHANGELOG.md`: chronological version history.
- `docs/releases/vX.Y.Z.md`: optional detailed release notes for important versions.

Do not create a separate full README for every version unless the user explicitly requests archival README snapshots.

## Required checks before editing

1. Run:

```powershell
git status
git log --oneline -n 10
Inspect changed files:
git diff --name-only
git diff --stat
If a version tag exists, inspect tags:
git tag --list
If Python files changed, inspect relevant entry points, tests, and examples before updating documentation.
Documentation rules
README.md

Update README.md only for the current project state.

README.md should include, when applicable:

Project purpose.
Core method or algorithmic idea.
Repository structure.
Environment setup.
Minimal reproducible run command.
Important scripts.
Input/output conventions.
Current limitations.
Citation or paper reference section, but only if verifiable from repository files or user-provided sources.

Do not claim performance improvements unless supported by committed results, tests, logs, or user-provided evidence.

CHANGELOG.md

If CHANGELOG.md does not exist, create it.

Use this format:
# Changelog

## [Unreleased]

## [vX.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Removed
- ...

### Notes
- ...
Every changelog item must be grounded in observed git diff, commit history, or explicit user instruction.

docs/releases/vX.Y.Z.md

Create this file only for meaningful versions or when explicitly requested.

Use this structure:
# Release vX.Y.Z

## Summary

## Main changes

## Reproducibility notes

## Validation performed

## Known limitations

## Files changed

## Next recommended work
DOA / SubspaceNet research documentation rules

When documenting this repository, preserve the distinction between:

snapshot-domain data;
covariance-domain data;
channel cross-correlation-domain features;
MUSIC / Root-MUSIC / ESPRIT / SubspaceNet pipelines.

Do not imply that a channel cross-correlation replacement is theoretically equivalent to SubspaceNet's original multi-lag empirical autocorrelation unless this has been derived or experimentally validated.

For any algorithmic claim, state whether it is:

implemented;
experimentally verified;
theoretically expected;
still an assumption.
Commit workflow

After updating documentation, use the git-commit skill if available.

Before committing, show:

files changed;
documentation sections updated;
proposed commit message.

Use Conventional Commits.

Preferred commit messages:
docs(readme): update project documentation for v0.1.0
docs(changelog): add notes for cross-correlation experiments
docs(release): add v0.1.0 reproducibility notes
Do not commit generated files, caches, datasets, model weights, or virtual environments.

保存后退出。

---

## 方案三：把它提交进仓库

执行：

```powershell id="wzpf6m"
git add .agents/skills/version-readme/SKILL.md
git commit -m "docs(codex): add version readme skill"