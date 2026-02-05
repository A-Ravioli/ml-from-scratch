# Authoring Guide (ml-from-scratch)

This repo is a curriculum. Each topic directory should be self-contained and follow this layout:

- `lesson.md` — theory + reading + guidance
- `exercise.py` — intentionally incomplete implementation (learners fill TODOs)
- `test_implementation.py` — deterministic, fast tests for learner implementations
- `solutions/solution.py` — reference implementation matching `exercise.py`’s public API

## Golden rules

1. **Do not change public APIs casually.** Tests import from `exercise.py`, and `solutions/solution.py` must mirror it.
2. **Tests must be fast and deterministic.** Fix random seeds; avoid long training loops.
3. **No network/data downloads.** Use tiny synthetic data where needed.
4. **Solutions must contain no placeholders.** No `YOUR CODE HERE`, `TODO`, `NotImplementedError`, or placeholder `pass`.

## Repo-level tooling

- Audit completeness:
  - `python3 scripts/audit_curriculum.py`
- Verify solutions without touching repo-tracked files:
  - `python3 scripts/verify_solutions.py --all`
  - `python3 scripts/verify_solutions.py --chapter 05-architectures`
  - `python3 scripts/verify_solutions.py --topic 00-mathematical-foundations/02-analysis/01-real-analysis`
- Scaffold a new topic directory (starter template):
  - `python3 scripts/scaffold_topic.py --dir <topic-dir> --title "<Title>"`

