"""Verify that the curriculum works: solutions pass, stubs fail, labs run.

    python scripts/verify.py                         # everything
    python scripts/verify.py --chapter 14            # one chapter
    python scripts/verify.py --topic 14-neural-networks-from-scratch/02-scalar-reverse-mode-autodiff
    python scripts/verify.py --fast -j 8             # skip @slow tests, run 8 at a time

For every topic and capstone:
  1. solutions  the tests pass against the reference notebook (MLFS_SOLUTIONS=1)
  2. stubs      every stage fails against the untouched exercise notebook, so no stage
                passes by accident (skip with --skip-stubs, e.g. after doing exercises yourself)
  3. lab        lab.py runs headless end to end against the reference, with datasets in
                offline (synthetic) mode (skip with --skip-labs)

Nothing in the repo is modified.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import chapter_dirs, rel, topic_dirs  # noqa: E402

PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
          "--import-mode=importlib", "--no-header"]


@dataclass
class Target:
    path: Path
    notebook: str  # "exercise" | "capstone"
    lab: bool


@dataclass
class Outcome:
    target: Target
    failures: list[str] = field(default_factory=list)


def env_for(root: Path, solutions: bool) -> dict[str, str]:
    env = dict(os.environ, MLFS_ROOT=str(root), MPLBACKEND="Agg", MLFS_OFFLINE="1")
    env.pop("MLFS_SOLUTIONS", None)
    if solutions:
        env["MLFS_SOLUTIONS"] = "1"
    return env


def tail(text: str, n: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(("      " + line) for line in lines[-n:])


def run_solutions(t: Target, root: Path, fast: bool) -> list[str]:
    tests = t.path / f"test_{t.notebook}.py"
    cmd = PYTEST + (["-m", "not slow"] if fast else []) + [str(tests)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root, env=env_for(root, True))
    if proc.returncode not in (0, 5):  # 5 = no tests collected
        return [f"solutions fail their tests:\n{tail(proc.stdout + proc.stderr)}"]
    if proc.returncode == 5:
        return [f"no tests collected in {tests.name}"]
    return []


def run_stubs(t: Target, root: Path, fast: bool) -> list[str]:
    tests = t.path / f"test_{t.notebook}.py"
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "report.xml"
        cmd = PYTEST + (["-m", "not slow"] if fast else []) + [f"--junitxml={xml}", str(tests)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root, env=env_for(root, False))
        if not xml.exists():
            return [f"couldn't run tests against stubs:\n{tail(proc.stdout + proc.stderr)}"]
        cases = ET.parse(xml).getroot().iter("testcase")
        stages: dict[int, bool] = {}  # stage -> any test failed
        for case in cases:
            m = re.match(r"test_stage(\d+)_", case.get("name", ""))
            if not m:
                continue
            failed = any(child.tag in ("failure", "error") for child in case)
            stage = int(m.group(1))
            stages[stage] = stages.get(stage, False) or failed
    return [f"stage {s} passes against the untouched {t.notebook}.py: its tests don't check anything"
            for s, failed in sorted(stages.items()) if not failed]


def run_lab(t: Target, root: Path, timeout: int) -> list[str]:
    try:
        proc = subprocess.run([sys.executable, str(t.path / "lab.py")], capture_output=True,
                              text=True, cwd=t.path, env=env_for(root, True), timeout=timeout)
    except subprocess.TimeoutExpired:
        return [f"lab.py took longer than {timeout}s"]
    if proc.returncode != 0:
        return [f"lab.py fails against the reference solution:\n{tail(proc.stdout + proc.stderr)}"]
    return []


def verify(t: Target, root: Path, args) -> Outcome:
    out = Outcome(t)
    out.failures += run_solutions(t, root, args.fast)
    if not args.skip_stubs:
        out.failures += run_stubs(t, root, args.fast)
    if t.lab and not args.skip_labs:
        out.failures += run_lab(t, root, args.timeout)
    return out


def targets(root: Path, chapter: str | None, topic: str | None) -> list[Target]:
    if topic:
        p = (root / topic).resolve()
        if (p / "test_capstone.py").exists():
            return [Target(p, "capstone", False)]
        return [Target(p, "exercise", True)]
    found = []
    for c in chapter_dirs(root):
        if chapter and not c.name.startswith(f"{chapter}-"):
            continue
        found += [Target(t, "exercise", True) for t in topic_dirs(c) if (t / "test_exercise.py").exists()]
        if (c / "capstone" / "test_capstone.py").exists():
            found.append(Target(c / "capstone", "capstone", False))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--chapter", help="two-digit chapter number")
    ap.add_argument("--topic", help="path to one topic or capstone directory, relative to the root")
    ap.add_argument("--fast", action="store_true", help="skip tests marked @pytest.mark.slow")
    ap.add_argument("--skip-stubs", action="store_true")
    ap.add_argument("--skip-labs", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="seconds allowed per lab")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    chapter = f"{int(args.chapter):02d}" if args.chapter else None

    todo = targets(root, chapter, args.topic)
    if not todo:
        print("nothing to verify")
        return 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        outcomes = list(pool.map(lambda t: verify(t, root, args), todo))

    bad = 0
    for o in outcomes:
        name = rel(o.target.path, root)
        if o.failures:
            bad += 1
            print(f"❌ {name}")
            for f in o.failures:
                print(f"    {f}")
        else:
            print(f"✅ {name}")
    print(f"\n{len(outcomes) - bad}/{len(outcomes)} verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
