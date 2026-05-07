# AGENTS.md

## Project role

This repository is a research codebase for DOA estimation, SubspaceNet reproduction, and channel cross-correlation experiments. Correctness, reproducibility, and traceability are more important than speed.

## General rules

- Inspect the repository structure before modifying code.
- Prefer minimal and reviewable changes.
- Do not rewrite large parts of the codebase unless explicitly asked.
- Do not introduce new dependencies unless necessary.
- Preserve existing function names, file names, experiment interfaces, and dataset conventions unless refactoring is explicitly requested.
- Do not fabricate experimental results, benchmark numbers, citations, or paper claims.
- Clearly distinguish implemented behavior, expected theoretical behavior, unverified assumptions, and proposed improvements.

## Git commit workflow

When asked to commit changes, use the installed `git-commit` skill.

Before committing:

1. Run `git status`.
2. Inspect unstaged changes with `git diff`.
3. Inspect staged changes with `git diff --staged`.
4. Stage only files directly related to the current task.
5. Do not stage unrelated local edits.
6. If there are multiple unrelated changes, split them into separate commits.
7. Before committing, show:
   - files to be committed;
   - why each file is included;
   - the proposed commit message.

## Commit message style

Use Conventional Commits:

- `feat:` for new functionality.
- `fix:` for bug fixes.
- `refactor:` for behavior-preserving restructuring.
- `docs:` for documentation-only changes.
- `test:` for tests or validation scripts.
- `chore:` for tooling, configuration, or maintenance.
- `perf:` for performance-related changes.
- `style:` for formatting-only changes.

Examples:

```text
docs(codex): add repository agent instructions
fix(data): correct dataset import path
refactor(doa): simplify covariance preprocessing
test(cc): add smoke test for cross-correlation input

Files that must not be committed unless explicitly requested

Do not commit generated files, caches, temporary outputs, logs, large datasets, model checkpoints, virtual environments, or IDE metadata.

Common examples:
__pycache__/
*.pyc
.ipynb_checkpoints/
*.log
*.tmp
*.bak
*.mat
*.npy
*.npz
*.h5
*.pth
*.pt
*.ckpt
runs/
outputs/
results/
data/
dataset/
datasets/
.vscode/
.idea/
pyEnv/
venv/
env/
.venv/
Python research code rules
Prefer deterministic examples with explicit random seeds.
Preserve CLI arguments and existing default behavior unless the task requires otherwise.
If changing numerical algorithms, explain the mathematical reason and expected effect.
Run the smallest relevant validation first.

Useful checks:
python main.py
python test_cc_input.py
python test_cc_toeplitz.py
If full training or full simulation is too expensive, run a reduced deterministic smoke test and report what was not fully verified.

DOA / SubspaceNet rules

When working on DOA estimation code:

State the assumed array model before changing algorithm logic.
Preserve the distinction between snapshot-domain data, covariance-domain data, and cross-correlation-domain features.
Do not replace MUSIC, Root-MUSIC, ESPRIT, SubspaceNet, or cross-correlation preprocessing logic with another method unless explicitly requested.
If modifying SubspaceNet-related code, preserve the differentiable subspace-method pipeline unless the task is specifically about architectural changes.
Validate dimension consistency for:
number of sensors;
number of snapshots;
number of sources;
angle grid size;
covariance or correlation matrix shape.
Safety constraints
Never commit secrets, tokens, passwords, private keys, or account credentials.
Never remove .gitignore rules without explaining why.
Never force push unless explicitly instructed.
Never delete large parts of the repository without a prior explanation.
