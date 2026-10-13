"""Shared helpers for the curriculum scripts: parse CURRICULUM.md and find content on disk."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "templates"
CURRICULUM = REPO / "CURRICULUM.md"

NUMBERED = re.compile(r"^(\d{2})-")
TOPIC_FILES = (
    "lesson.md",
    "exercise.py",
    "test_exercise.py",
    "hints.md",
    "lab.py",
    "solutions/exercise.py",
    "solutions/lab_notes.md",
)
CHAPTER_FILES = (
    "README.md",
    "checkpoint.md",
    "capstone/brief.md",
    "capstone/rubric.md",
    "capstone/capstone.py",
    "capstone/test_capstone.py",
    "capstone/solutions/capstone.py",
)
LESSON_SECTIONS = (
    "why this matters",
    "warm-up",
    "prerequisites check",
    "core ideas",
    "worked example",
    "on paper",
    "experiment",
    "check your understanding",
    "common bugs",
    "going deeper",
    "connections",
)


def slugify(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


@dataclass
class PlannedTopic:
    num: str
    title: str
    learn: str = ""
    build: str = ""
    lab: str = ""

    @property
    def dirname(self) -> str:
        return f"{self.num}-{slugify(self.title)}"


@dataclass
class PlannedChapter:
    num: str
    title: str
    goal: str = ""
    meta: str = ""
    capstone: str = ""
    topics: list[PlannedTopic] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return f"{self.num}-{slugify(self.title)}"


_CHAPTER = re.compile(r"^## (\d{2}) · (.+?)\s*$")
_GOAL = re.compile(r"^\*goal: (.+?)\*\s*(?:·\s*(.*))?$")
_ROW = re.compile(r"^\|\s*(\d{2})\s*\|(.+)\|\s*$")


def parse_curriculum(path: Path = CURRICULUM) -> dict[str, PlannedChapter]:
    """Chapters keyed by two-digit number, with topics parsed from each chapter's table."""
    chapters: dict[str, PlannedChapter] = {}
    current: PlannedChapter | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):  # a new part ends the current chapter
            current = None
            continue
        if m := _CHAPTER.match(line):
            current = chapters[m.group(1)] = PlannedChapter(m.group(1), m.group(2))
            continue
        if current is None:
            continue
        if m := _GOAL.match(line):
            current.goal, current.meta = m.group(1), (m.group(2) or "").strip()
        elif line.startswith("**capstone:**"):
            current.capstone = line.removeprefix("**capstone:**").strip()
        elif m := _ROW.match(line):
            cells = [c.strip() for c in m.group(2).split("|")]
            title = cells[0]
            rest = cells[1:] + ["", "", ""]
            if len(cells) >= 4:  # | # | topic | core ideas | you build | lab |
                current.topics.append(PlannedTopic(m.group(1), title, rest[0], rest[1], rest[2]))
            else:  # | # | topic | you do |
                current.topics.append(PlannedTopic(m.group(1), title, build=rest[0]))
    return chapters


def numbered_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and NUMBERED.match(p.name))


def chapter_dirs(root: Path) -> list[Path]:
    return numbered_dirs(root)


def topic_dirs(chapter: Path) -> list[Path]:
    return numbered_dirs(chapter)


def find_numbered(parent: Path, num: str) -> Path | None:
    for p in numbered_dirs(parent):
        if p.name.startswith(f"{num}-"):
            return p
    return None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
