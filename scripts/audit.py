"""Audit the curriculum against the canonical chapter model (AUTHORING.md) and the plan.

    python scripts/audit.py                 # summary + every issue
    python scripts/audit.py --chapter 14    # one chapter
    python scripts/audit.py --marimo        # also run `marimo check` on every notebook
    python scripts/audit.py --strict        # warnings (incl. unfinished templates) fail too
    python scripts/audit.py --json          # machine-readable report

Errors (always fail): missing files, placeholders left in solutions, stages without tests,
broken registry entries. Warnings: TEMPLATE markers, missing lesson sections, callouts,
hints, or live check cells.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))  # this checkout's mlfs, not the installed one
from _common import (  # noqa: E402
    CHAPTER_FILES,
    CURRICULUM,
    LESSON_SECTIONS,
    TOPIC_FILES,
    chapter_dirs,
    find_numbered,
    parse_curriculum,
    rel,
    topic_dirs,
)
from mlfs.registry import RESERVED  # noqa: E402

PLACEHOLDERS = re.compile(r"NotImplementedError\(\s*[\"']stage\d|\bTODO\b|YOUR CODE|TEMPLATE")
STUB_STAGE = re.compile(r"NotImplementedError\(\s*[\"']stage(\d+)")
TEST_STAGE = re.compile(r"^def (test_stage(\d+)_\w*)\(", re.M)
ANY_TEST = re.compile(r"^def (test_\w*)\(", re.M)
CHECK_CELL = re.compile(r"checks\.run\(\s*(\d+)")
BUILD_CALLOUT = re.compile(r"▶ build: stage (\d+)")
PREDICT_CALLOUT = re.compile(r"🔮 predict")
HINT_STAGE = re.compile(r"^## stage (\d+)", re.M)
H2 = re.compile(r"^## (.+?)\s*$", re.M)


@dataclass
class Unit:
    path: str
    kind: str  # "topic" | "chapter"
    status: str = "done"  # "done" | "template" | "issues" | "broken"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def finish(self, has_template: bool) -> None:
        if self.errors:
            self.status = "broken"
        elif has_template:
            self.status = "template"
        elif self.warnings:
            self.status = "issues"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def has_template(directory: Path) -> bool:
    return any("TEMPLATE" in read(f) for f in directory.rglob("*")
               if f.is_file() and f.suffix in {".md", ".py"} and "__pycache__" not in f.parts)


def check_stages(unit: Unit, notebook: Path, tests: Path, require_naming: bool) -> set[int]:
    stub_stages = {int(s) for s in STUB_STAGE.findall(read(notebook))}
    test_src = read(tests)
    test_stages = {int(s) for _, s in TEST_STAGE.findall(test_src)}
    for stage in sorted(stub_stages - test_stages):
        unit.errors.append(f"{notebook.name}: stage {stage} has a stub but no test_stage{stage}_* tests")
    if test_stages and test_stages != set(range(1, max(test_stages) + 1)):
        unit.errors.append(f"{tests.name}: stages aren't contiguous from 1: {sorted(test_stages)}")
    if require_naming:
        staged = {name for name, _ in TEST_STAGE.findall(test_src)}
        for name in ANY_TEST.findall(test_src):
            if name not in staged:
                unit.warnings.append(f"{tests.name}: {name} isn't named test_stageN_*, so live checks skip it")
    checked = {int(s) for s in CHECK_CELL.findall(read(notebook))}
    for stage in sorted(test_stages - checked):
        unit.warnings.append(f"{notebook.name}: no live check cell `checks.run({stage}, ...)`")
    return test_stages


def _nb_names(tests_src: str) -> dict[int, set[str]]:
    """{stage: names the stage's tests read as nb.<name>}, following helper functions that take nb."""
    try:
        tree = ast.parse(tests_src)
    except SyntaxError:
        return {}
    funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    direct, calls = {}, {}
    for name, f in funcs.items():
        direct[name] = {n.attr for n in ast.walk(f)
                        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "nb"}
        calls[name] = {n.func.id for n in ast.walk(f)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs}

    def used(name, seen):
        if name in seen:
            return set()
        seen.add(name)
        out = set(direct[name])
        for c in calls[name]:
            out |= used(c, seen)
        return out

    stages: dict[int, set[str]] = {}
    for name in funcs:
        m = re.match(r"test_stage(\d+)_", name)
        if m:
            stages.setdefault(int(m.group(1)), set()).update(used(name, set()))
    return stages


def _check_cell_names(notebook_src: str) -> dict[int, set[str] | None]:
    """{stage: keyword names passed to checks.run(stage, ...)}; None if a call passes **kwargs (can't tell)."""
    try:
        tree = ast.parse(notebook_src)
    except SyntaxError:
        return {}
    out: dict[int, set[str] | None] = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "checks"
                and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, int)):
            stage = n.args[0].value
            if any(k.arg is None for k in n.keywords):
                out[stage] = None
            elif out.get(stage, set()) is not None:
                out[stage] = out.get(stage, set()) | {k.arg for k in n.keywords if not k.arg.startswith("_")}
    return out


def check_live_names(unit: Unit, notebook: Path, tests: Path) -> None:
    """every name a stage's tests read as nb.<name> must be passed to that stage's checks.run(...) cell,
    or the live check crashes on correct code (pytest uses the whole module, so verify can't see this)."""
    passed = _check_cell_names(read(notebook))
    for stage, needed in sorted(_nb_names(read(tests)).items()):
        have = passed.get(stage)
        if have is None:
            continue
        missing = sorted(needed - have)
        if missing:
            unit.errors.append(f"{notebook.name}: checks.run({stage}, ...) doesn't pass {', '.join(missing)}, "
                               f"which test_stage{stage}_* use as nb.<name>: the live check would crash")


THREADS_UP = re.compile(r"torch\.set_num_threads\(\s*([2-9]|[1-9]\d+)\s*\)")


def check_threads(unit: Unit, directory: Path) -> None:
    """a notebook that raises torch's thread count overrides CI's OMP_NUM_THREADS=1 and oversubscribes the runner
    (labs then take 50x longer); cap it with min(N, torch.get_num_threads()) instead."""
    for f in sorted(directory.rglob("*.py")):
        for i, line in enumerate(read(f).splitlines(), 1):
            if THREADS_UP.search(line):
                unit.errors.append(f"{rel(f, directory)}:{i}: raises torch's thread count; use "
                                   "torch.set_num_threads(min(N, torch.get_num_threads()))")


def check_solution(unit: Unit, solution: Path) -> None:
    for i, line in enumerate(read(solution).splitlines(), 1):
        if PLACEHOLDERS.search(line) and "TEMPLATE" not in line:
            unit.errors.append(f"{solution.parent.name}/{solution.name}:{i}: placeholder in reference solution: {line.strip()}")


def audit_topic(topic: Path, root: Path) -> Unit:
    unit = Unit(rel(topic, root), "topic")
    missing = [f for f in TOPIC_FILES if not (topic / f).exists()]
    unit.errors += [f"missing {f}" for f in missing]
    stages = check_stages(unit, topic / "exercise.py", topic / "test_exercise.py", True)
    check_solution(unit, topic / "solutions" / "exercise.py")
    for nb_file in (topic / "exercise.py", topic / "solutions" / "exercise.py"):
        check_live_names(unit, nb_file, topic / "test_exercise.py")
    check_threads(unit, topic)

    lesson = read(topic / "lesson.md")
    if lesson:
        headings = [h.lower() for h in H2.findall(lesson)]
        for section in LESSON_SECTIONS:
            if not any(h.startswith(section) for h in headings):
                unit.warnings.append(f"lesson.md: missing section '## {section}'")
        built = {int(s) for s in BUILD_CALLOUT.findall(lesson)}
        for stage in sorted(stages - built):
            unit.warnings.append(f"lesson.md: no '▶ build: stage {stage}' callout")
        if len(PREDICT_CALLOUT.findall(lesson)) < 2:
            unit.warnings.append("lesson.md: fewer than two '🔮 predict' callouts")
    hints = read(topic / "hints.md")
    if hints:
        hinted = {int(s) for s in HINT_STAGE.findall(hints)}
        for stage in sorted(stages - hinted):
            unit.warnings.append(f"hints.md: no '## stage {stage}' section")
    if (topic / "lab.py").exists() and "ui.predict(" not in read(topic / "lab.py"):
        unit.warnings.append("lab.py: no `ui.predict(...)` prediction gate")
    unit.finish(has_template(topic))
    return unit


def audit_chapter(chapter: Path, root: Path) -> Unit:
    unit = Unit(rel(chapter, root), "chapter")
    unit.errors += [f"missing {f}" for f in CHAPTER_FILES if not (chapter / f).exists()]
    cap = chapter / "capstone"
    if cap.exists():
        check_stages(unit, cap / "capstone.py", cap / "test_capstone.py", False)
        check_solution(unit, cap / "solutions" / "capstone.py")
        for nb_file in (cap / "capstone.py", cap / "solutions" / "capstone.py"):
            check_live_names(unit, nb_file, cap / "test_capstone.py")
        check_threads(unit, cap)
    templated = any("TEMPLATE" in read(chapter / f) for f in CHAPTER_FILES)
    unit.finish(templated)
    return unit


def audit_registry(root: Path) -> list[str]:
    path = root / "mlfs" / "registry.toml"
    if not path.exists():
        return []
    try:
        modules = tomllib.loads(read(path)).get("modules", {})
    except tomllib.TOMLDecodeError as e:
        return [f"mlfs/registry.toml: invalid TOML: {e}"]
    errors = []
    for name, targets in modules.items():
        if name in RESERVED:
            errors.append(f"mlfs.{name}: name is reserved")
        for target in [targets] if isinstance(targets, str) else targets:
            topic = root / target
            if not topic.exists():
                continue  # planned, not written yet
            for f in ("exercise.py", "solutions/exercise.py"):
                if not (topic / f).exists():
                    errors.append(f"mlfs.{name}: {target}/{f} doesn't exist")
    return errors


def coverage(root: Path, curriculum: Path, only: str | None) -> list[dict]:
    rows = []
    for num, chapter in parse_curriculum(curriculum).items():
        if only and num != only:
            continue
        cdir = find_numbered(root, num)
        present = [t for t in chapter.topics if cdir and find_numbered(cdir, t.num)]
        rows.append({"chapter": num, "title": chapter.title, "dir": cdir.name if cdir else None,
                     "planned": len(chapter.topics), "present": len(present)})
    return rows


def marimo_check(root: Path, dirs: list[Path]) -> list[str]:
    files = [str(f) for d in dirs for f in d.rglob("*.py")
             if not f.name.startswith("test_") and "__pycache__" not in f.parts]
    if not files:
        return []
    proc = subprocess.run([sys.executable, "-m", "marimo", "check", "--strict", *files],
                          capture_output=True, text=True, cwd=root)
    return [] if proc.returncode == 0 else [(proc.stdout + proc.stderr).strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--curriculum", type=Path, default=CURRICULUM)
    ap.add_argument("--chapter", help="two-digit chapter number")
    ap.add_argument("--marimo", action="store_true", help="also run `marimo check --strict`")
    ap.add_argument("--strict", action="store_true", help="fail on warnings too")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    only = f"{int(args.chapter):02d}" if args.chapter else None

    chapters = [c for c in chapter_dirs(root) if not only or c.name.startswith(f"{only}-")]
    units: list[Unit] = []
    for chapter in chapters:
        units.append(audit_chapter(chapter, root))
        units += [audit_topic(t, root) for t in topic_dirs(chapter)]
    registry_errors = audit_registry(root)
    marimo_errors = marimo_check(root, chapters) if args.marimo else []
    cov = coverage(root, args.curriculum, only)

    n_err = sum(len(u.errors) for u in units) + len(registry_errors) + len(marimo_errors)
    n_warn = sum(len(u.warnings) for u in units)
    if args.json:
        print(json.dumps({"coverage": cov, "units": [asdict(u) for u in units],
                          "registry": registry_errors, "marimo": marimo_errors}, indent=2))
    else:
        topics = [u for u in units if u.kind == "topic"]
        planned = sum(r["planned"] for r in cov)
        by = {s: sum(u.status == s for u in topics) for s in ("done", "issues", "template", "broken")}
        print(f"topics: {len(topics)}/{planned} planned exist · "
              f"{by['done']} done · {by['issues']} with warnings · "
              f"{by['template']} still templates · {by['broken']} broken")
        print("\nchapter coverage:")
        for r in cov:
            mark = "·" if r["present"] == 0 else ("✓" if r["present"] == r["planned"] else "◐")
            print(f"  {mark} {r['chapter']} {r['title']:<48} {r['present']:>2}/{r['planned']}")
        icon = {"done": "✅", "issues": "⚠️ ", "template": "📝", "broken": "❌"}
        problem_units = [u for u in units if u.errors or u.warnings or u.status != "done"]
        if problem_units:
            print()
        for u in problem_units:
            print(f"{icon[u.status]} {u.path} [{u.status}]")
            for e in u.errors:
                print(f"    error: {e}")
            for w in u.warnings:
                print(f"    warn:  {w}")
        for e in registry_errors:
            print(f"❌ registry: {e}")
        for e in marimo_errors:
            print(f"❌ marimo check:\n{e}")
        print(f"\n{n_err} errors, {n_warn} warnings")

    templates_left = any(u.status == "template" for u in units)
    if n_err or (args.strict and (n_warn or templates_left)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
