"""Create chapters and topics from the templates, pre-filled from CURRICULUM.md.

    python scripts/scaffold.py plan 14                  # chapter 14 + every topic in its plan
    python scripts/scaffold.py chapter 14               # just the chapter skeleton
    python scripts/scaffold.py topic 14 03 --provides tensor
                                                        # one topic; registers mlfs.tensor

Existing files are never overwritten unless you pass --force. Scaffolded files contain
TEMPLATE markers; `scripts/audit.py` reports them until they're replaced with real content.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CURRICULUM,
    TEMPLATES,
    PlannedChapter,
    PlannedTopic,
    find_numbered,
    parse_curriculum,
    rel,
)

from mlfs.registry import RESERVED  # noqa: E402


def render(text: str, tokens: dict[str, str]) -> str:
    for key, value in tokens.items():
        text = text.replace(f"__{key}__", value)
    return text


def copy_tree(src: Path, dst: Path, tokens: dict[str, str], root: Path, force: bool) -> None:
    for file in sorted(src.rglob("*")):
        if file.is_dir() or "__pycache__" in file.parts:
            continue
        target = dst / file.relative_to(src)
        if target.exists() and not force:
            print(f"  skip   {rel(target, root)} (exists)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(file.read_text(encoding="utf-8"), tokens), encoding="utf-8")
        print(f"  create {rel(target, root)}")


def planned_chapter(plan: dict[str, PlannedChapter], num: str, title: str | None) -> PlannedChapter:
    if num in plan:
        chapter = plan[num]
        if title:
            chapter.title = title
        return chapter
    if not title:
        sys.exit(f"chapter {num} isn't in CURRICULUM.md; pass --title")
    return PlannedChapter(num, title)


def chapter_dir(root: Path, chapter: PlannedChapter) -> Path:
    return find_numbered(root, chapter.num) or root / chapter.dirname


def topic_table(chapter: PlannedChapter) -> str:
    if not chapter.topics:
        return "| # | topic | you build |\n|---|---|---|"
    rows = [f"| {t.num} | [{t.title}]({t.dirname}/) | {t.build} |" for t in chapter.topics]
    return "| # | topic | you build |\n|---|---|---|\n" + "\n".join(rows)


def scaffold_chapter(root: Path, chapter: PlannedChapter, force: bool) -> Path:
    dst = chapter_dir(root, chapter)
    print(f"chapter {rel(dst, root)}")
    tokens = {
        "CHAPTER_NUM": chapter.num,
        "CHAPTER_TITLE": chapter.title,
        "GOAL": chapter.goal or "one-paragraph hook",
        "META": chapter.meta or "~N h · L?",
        "TOPIC_TABLE": topic_table(chapter),
        "CAPSTONE": chapter.capstone or "…",
    }
    copy_tree(TEMPLATES / "chapter", dst, tokens, root, force)
    return dst


def scaffold_topic(root: Path, chapter: PlannedChapter, topic: PlannedTopic, force: bool) -> Path:
    parent = chapter_dir(root, chapter)
    dst = find_numbered(parent, topic.num) or parent / topic.dirname
    print(f"topic {rel(dst, root)}")
    tokens = {
        "CHAPTER_NUM": chapter.num,
        "CHAPTER_TITLE": chapter.title,
        "NUM": topic.num,
        "TITLE": topic.title,
        "PLAN_LEARN": topic.learn or "…",
        "PLAN_BUILD": topic.build or "…",
        "PLAN_LAB": topic.lab or "…",
    }
    copy_tree(TEMPLATES / "topic", dst, tokens, root, force)
    return dst


def register(root: Path, name: str, topic: Path) -> None:
    """Add `topic` to mlfs.<name> in the registry (a module can merge several topics)."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name) or name in RESERVED:
        sys.exit(f"invalid mlfs module name {name!r}")
    registry = root / "mlfs" / "registry.toml"
    text = registry.read_text(encoding="utf-8") if registry.exists() else "[modules]\n"
    modules = tomllib.loads(text).get("modules", {})
    target = topic.relative_to(root).as_posix()
    topics = [modules[name]] if isinstance(modules.get(name), str) else list(modules.get(name, []))
    if target in topics:
        print(f"  mlfs.{name} already includes {target}")
        return
    modules[name] = topics + [target]
    header = text[: text.index("[modules]")] if "[modules]" in text else ""
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(header + format_registry(modules), encoding="utf-8")
    print(f"  register mlfs.{name} -> {target}")


def format_registry(modules: dict) -> str:
    lines = ["[modules]"]
    for key, value in modules.items():
        topics = [value] if isinstance(value, str) else value
        if len(topics) == 1:
            lines.append(f'{key} = "{topics[0]}"')
        else:
            lines.append(f"{key} = [")
            lines += [f'    "{t}",' for t in topics]
            lines.append("]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="curriculum root to write into (default: this repo)")
    ap.add_argument("--curriculum", type=Path, default=CURRICULUM)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="a chapter and every topic in its CURRICULUM.md table")
    p.add_argument("chapter")
    c = sub.add_parser("chapter", help="just the chapter skeleton")
    c.add_argument("chapter")
    c.add_argument("--title")
    t = sub.add_parser("topic", help="one topic")
    t.add_argument("chapter")
    t.add_argument("topic")
    t.add_argument("--title")
    t.add_argument("--provides", help="register this topic as mlfs.<name>")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    plan = parse_curriculum(args.curriculum)
    num = f"{int(args.chapter):02d}"
    chapter = planned_chapter(plan, num, getattr(args, "title", None) if args.cmd == "chapter" else None)

    if args.cmd in ("plan", "chapter"):
        scaffold_chapter(root, chapter, args.force)
    if args.cmd == "plan":
        for topic in chapter.topics:
            scaffold_topic(root, chapter, topic, args.force)
    if args.cmd == "topic":
        tnum = f"{int(args.topic):02d}"
        planned = next((x for x in chapter.topics if x.num == tnum), None)
        if planned is None:
            if not args.title:
                sys.exit(f"topic {num}·{tnum} isn't in CURRICULUM.md; pass --title")
            planned = PlannedTopic(tnum, args.title)
        elif args.title:
            planned.title = args.title
        dst = scaffold_topic(root, chapter, planned, args.force)
        if args.provides:
            register(root, args.provides, dst)


if __name__ == "__main__":
    main()
