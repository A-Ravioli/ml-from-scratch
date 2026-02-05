#!/usr/bin/env python3
"""
Audit curriculum completeness.

Discovers "topic directories" (directories containing any of:
  - lesson.md
  - exercise.py
  - test_implementation.py
  - solutions/solution.py
)

For each topic directory, reports:
  - Missing required files (lesson/exercise/test/solution)
  - Placeholder markers in tests (TODO, placeholder `pass`, NotImplementedError)
  - Placeholder markers in solutions (YOUR CODE HERE, TODO, NotImplementedError, placeholder `pass`)

Outputs:
  - Human-readable summary to stdout
  - Optional JSON report (default: reports/curriculum_audit.json)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = ("lesson.md", "exercise.py", "test_implementation.py", "solutions/solution.py")


@dataclass(frozen=True)
class PlaceholderCounts:
    todo: int
    your_code_here: int
    not_implemented_error: int
    pass_lines: int


@dataclass(frozen=True)
class TopicAudit:
    topic_dir: str
    missing: list[str]
    test_placeholders: PlaceholderCounts | None
    solution_placeholders: PlaceholderCounts | None


_TODO_RE = re.compile(r"\bTODO\b")
_YOUR_CODE_RE = re.compile(r"YOUR CODE HERE")
_NIE_RE = re.compile(r"NotImplementedError")
_PASS_LINE_RE = re.compile(r"^\s*pass\s*(#.*)?$", re.MULTILINE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _count_placeholders(text: str) -> PlaceholderCounts:
    return PlaceholderCounts(
        todo=len(_TODO_RE.findall(text)),
        your_code_here=len(_YOUR_CODE_RE.findall(text)),
        not_implemented_error=len(_NIE_RE.findall(text)),
        pass_lines=len(_PASS_LINE_RE.findall(text)),
    )


def _is_topic_dir(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    # Ignore venv/git caches
    parts = set(dir_path.parts)
    if ".git" in parts or ".venv" in parts or "__pycache__" in parts:
        return False
    if (dir_path / "lesson.md").exists():
        return True
    if (dir_path / "exercise.py").exists():
        return True
    if (dir_path / "test_implementation.py").exists():
        return True
    if (dir_path / "solutions" / "solution.py").exists():
        return True
    return False


def discover_topics(root: Path) -> list[Path]:
    topics: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # prune noisy dirs
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", ".venv", "__pycache__", ".pytest_cache"} and not d.startswith(".")
        ]
        if _is_topic_dir(p):
            topics.append(p)
    # de-dup nested: keep deepest only (so a chapter root with CHAPTER_INTRO.md doesn't count as a topic)
    topics = sorted(set(topics))
    topic_set = set(topics)
    filtered: list[Path] = []
    for t in topics:
        # If any child directory is also a topic, treat parent as non-topic.
        has_child_topic = any((child != t and child.is_relative_to(t)) for child in topic_set)
        if not has_child_topic:
            filtered.append(t)
    return sorted(filtered)


def audit_topic(topic_dir: Path) -> TopicAudit:
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        if not (topic_dir / rel).exists():
            missing.append(rel)

    test_counts: PlaceholderCounts | None = None
    test_path = topic_dir / "test_implementation.py"
    if test_path.exists():
        test_counts = _count_placeholders(_read_text(test_path))

    sol_counts: PlaceholderCounts | None = None
    sol_path = topic_dir / "solutions" / "solution.py"
    if sol_path.exists():
        sol_counts = _count_placeholders(_read_text(sol_path))

    return TopicAudit(
        topic_dir=str(topic_dir.relative_to(ROOT)),
        missing=missing,
        test_placeholders=test_counts,
        solution_placeholders=sol_counts,
    )


def _is_chapter_dir(path: Path) -> bool:
    return path.is_dir() and re.match(r"^\d\d-", path.name) is not None


def _chapter_for_topic(rel_topic_dir: str) -> str:
    return rel_topic_dir.split(os.sep, 1)[0]


def _print_summary(audits: list[TopicAudit], *, show_ok: bool) -> None:
    by_chapter: dict[str, list[TopicAudit]] = {}
    for a in audits:
        by_chapter.setdefault(_chapter_for_topic(a.topic_dir), []).append(a)

    total_missing = sum(1 for a in audits if a.missing)
    total_test_placeholders = sum(
        1
        for a in audits
        if a.test_placeholders
        and (a.test_placeholders.todo or a.test_placeholders.not_implemented_error or a.test_placeholders.pass_lines)
    )
    total_solution_placeholders = sum(
        1
        for a in audits
        if a.solution_placeholders
        and (
            a.solution_placeholders.todo
            or a.solution_placeholders.your_code_here
            or a.solution_placeholders.not_implemented_error
            or a.solution_placeholders.pass_lines
        )
    )

    print("Curriculum audit")
    print(f"- topics: {len(audits)}")
    print(f"- topics missing required files: {total_missing}")
    print(f"- topics with test placeholders: {total_test_placeholders}")
    print(f"- topics with solution placeholders: {total_solution_placeholders}")
    print()

    for ch in sorted(by_chapter.keys()):
        print(ch)
        chapter_audits = by_chapter[ch]
        for a in chapter_audits:
            ok = not a.missing
            test_bad = bool(
                a.test_placeholders
                and (a.test_placeholders.todo or a.test_placeholders.not_implemented_error or a.test_placeholders.pass_lines)
            )
            sol_bad = bool(
                a.solution_placeholders
                and (
                    a.solution_placeholders.todo
                    or a.solution_placeholders.your_code_here
                    or a.solution_placeholders.not_implemented_error
                    or a.solution_placeholders.pass_lines
                )
            )

            if ok and not test_bad and not sol_bad and not show_ok:
                continue

            parts: list[str] = []
            if a.missing:
                parts.append("missing=" + ",".join(a.missing))
            if test_bad and a.test_placeholders:
                parts.append(
                    f"test(TODO={a.test_placeholders.todo},NIE={a.test_placeholders.not_implemented_error},pass={a.test_placeholders.pass_lines})"
                )
            if sol_bad and a.solution_placeholders:
                parts.append(
                    "solution("
                    + f"YCH={a.solution_placeholders.your_code_here},TODO={a.solution_placeholders.todo},"
                    + f"NIE={a.solution_placeholders.not_implemented_error},pass={a.solution_placeholders.pass_lines}"
                    + ")"
                )

            status = "OK" if not parts else "ISSUES"
            print(f"  - {a.topic_dir}: {status}" + ("" if not parts else " " + " ".join(parts)))
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="Repo root (default: repo root)")
    parser.add_argument("--json", default="reports/curriculum_audit.json", help="Write JSON report to this path")
    parser.add_argument("--no-json", action="store_true", help="Do not write JSON output")
    parser.add_argument("--show-ok", action="store_true", help="Show topics with no issues too")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    topics = discover_topics(root)
    audits = [audit_topic(t) for t in topics]

    _print_summary(audits, show_ok=args.show_ok)

    if not args.no_json:
        out_path = (root / args.json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(a) for a in audits], indent=2), encoding="utf-8")

    # exit non-zero if any required file missing or any placeholders in tests/solutions
    failed = False
    for a in audits:
        if a.missing:
            failed = True
            break
        if a.test_placeholders and (a.test_placeholders.todo or a.test_placeholders.not_implemented_error or a.test_placeholders.pass_lines):
            failed = True
            break
        if a.solution_placeholders and (
            a.solution_placeholders.todo
            or a.solution_placeholders.your_code_here
            or a.solution_placeholders.not_implemented_error
            or a.solution_placeholders.pass_lines
        ):
            failed = True
            break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
