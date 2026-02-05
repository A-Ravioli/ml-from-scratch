#!/usr/bin/env python3
"""
Scaffold a new topic directory with lesson/exercise/test/solution.

This is intentionally minimal: it creates a consistent file layout and a tiny
"toy but faithful" default example. For large topics, you'll still customize
the generated files.

Usage:
  python3 scripts/scaffold_topic.py --dir 07-reinforcement-learning/02-tabular-methods/01-dynamic-programming --title "Dynamic Programming (Tabular RL)"
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


LESSON_TEMPLATE = """# {title}

## Objectives
- Understand the core ideas and when they apply.
- Implement the algorithm(s) from scratch.
- Validate correctness with deterministic unit tests.

## Theory (high level)
Write the key definitions and theorems here.

## Implementation Notes
- Keep implementations deterministic and CPU-friendly.
- Prefer small synthetic examples.

## Exercises
Open `exercise.py` and implement the TODO blocks. Then run:

```bash
python3 -m pytest test_implementation.py -q
```
"""


EXERCISE_TEMPLATE = '''"""
{title} — Exercises

Fill in the TODO sections, then run `pytest test_implementation.py -q`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set global seed for deterministic behavior."""
    np.random.seed(seed)


# TODO: Define your public API here.
# The tests import from this module, so keep names stable.


def toy_function(x: np.ndarray) -> np.ndarray:
    """
    A tiny placeholder to demonstrate structure.

    TODO: Implement a simple transformation (e.g., return 2*x + 1).
    """
    # YOUR CODE HERE
    raise NotImplementedError
'''


SOLUTION_TEMPLATE = '''"""
{title} — Solutions (Reference Implementation)

This file mirrors `exercise.py` but with all TODOs completed.
"""

from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def toy_function(x: np.ndarray) -> np.ndarray:
    return 2 * x + 1
'''


TEST_TEMPLATE = '''"""
{title} — Tests
"""

import numpy as np

from exercise import set_seed, toy_function


def test_toy_function_deterministic():
    set_seed(0)
    x = np.array([0.0, 1.0, -2.0])
    y = toy_function(x)
    np.testing.assert_allclose(y, 2 * x + 1)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Topic directory to scaffold (relative to repo root)")
    parser.add_argument("--title", required=True, help="Topic title for lesson and file headers")
    args = parser.parse_args()

    topic_dir = (ROOT / args.dir).resolve()
    topic_dir.mkdir(parents=True, exist_ok=True)
    (topic_dir / "solutions").mkdir(parents=True, exist_ok=True)

    lesson = topic_dir / "lesson.md"
    exercise = topic_dir / "exercise.py"
    test = topic_dir / "test_implementation.py"
    solution = topic_dir / "solutions" / "solution.py"

    if not lesson.exists():
        lesson.write_text(LESSON_TEMPLATE.format(title=args.title), encoding="utf-8")
    if not exercise.exists():
        exercise.write_text(EXERCISE_TEMPLATE.format(title=args.title), encoding="utf-8")
    if not test.exists():
        test.write_text(TEST_TEMPLATE.format(title=args.title), encoding="utf-8")
    if not solution.exists():
        solution.write_text(SOLUTION_TEMPLATE.format(title=args.title), encoding="utf-8")

    print(f"Scaffolded: {topic_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

