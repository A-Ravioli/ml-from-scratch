"""Show sections of a topic's `lesson.md` inside its notebooks.

The lesson text lives in one place (markdown, readable on GitHub). Notebooks pull in the
section that goes with each stage, so you can read a bit and build a bit in one window:

    lesson.section("the chain rule")
"""
from __future__ import annotations

import re
from pathlib import Path

from mlfs import notebooks

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def extract(text: str, heading: str) -> str | None:
    """The markdown under the first heading that starts with `heading` (case-insensitive),
    up to the next heading of the same or a higher level. Includes the heading itself."""
    want = heading.strip().lower()
    lines = text.splitlines()
    start = level = None
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.match(line)
        if not m:
            continue
        depth, title = len(m.group(1)), m.group(2).strip().lower()
        if start is None:
            if title.startswith(want):
                start, level = i, depth
        elif depth <= level:
            return "\n".join(lines[start:i]).strip()
    return None if start is None else "\n".join(lines[start:]).strip()


def section(heading: str, path: str = "lesson.md"):
    """Render one section of the topic's lesson as markdown in the notebook."""
    import marimo as mo

    here = mo.notebook_dir()
    if here is None:
        return None
    file = notebooks.topic_dir(here) / path
    body = extract(Path(file).read_text(encoding="utf-8"), heading) if file.exists() else None
    if body is None:
        return mo.callout(mo.md(f"section **{heading}** not found in `{path}`"), kind="warn")
    return mo.md(body)
