#!/usr/bin/env python3
"""
Verify that reference solutions pass the existing tests without mutating repo-tracked files.

For each topic directory containing both:
  - test_implementation.py
  - solutions/solution.py
we create a temp directory, copy the topic contents, overwrite exercise.py with the solution,
and run pytest in the temp directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Result:
    topic_dir: str
    ok: bool
    returncode: int
    stdout: str
    stderr: str


def _is_topic_dir(p: Path) -> bool:
    return (p / "test_implementation.py").exists() and (p / "solutions" / "solution.py").exists()


def discover_topics(root: Path) -> list[Path]:
    topics: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", ".venv", "__pycache__", ".pytest_cache"} and not d.startswith(".")
        ]
        p = Path(dirpath)
        if _is_topic_dir(p):
            topics.append(p)
    # Keep deepest only
    topics = sorted(set(topics))
    topic_set = set(topics)
    filtered: list[Path] = []
    for t in topics:
        has_child_topic = any((child != t and child.is_relative_to(t)) for child in topic_set)
        if not has_child_topic:
            filtered.append(t)
    return sorted(filtered)


def copy_topic_to_temp(topic_dir: Path, temp_root: Path) -> Path:
    dest = temp_root / topic_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(topic_dir):
        rel = Path(dirpath).relative_to(topic_dir)
        # ignore caches
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".pytest_cache"} and not d.startswith(".")]
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            if fn in {".DS_Store"}:
                continue
            src_file = Path(dirpath) / fn
            dst_file = dest / rel / fn
            shutil.copy2(src_file, dst_file)
    # overwrite exercise.py with solution.py
    sol = dest / "solutions" / "solution.py"
    if not sol.exists():
        raise FileNotFoundError(f"Missing solution in temp copy for {topic_dir}")
    shutil.copy2(sol, dest / "exercise.py")
    return dest


def run_pytest(topic_temp_dir: Path, *, timeout_s: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [sys.executable, "-m", "pytest", "-q", "test_implementation.py"]
    p = subprocess.run(
        cmd,
        cwd=str(topic_temp_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Verify all topics with tests+solutions")
    parser.add_argument("--chapter", help="Verify only a specific chapter (e.g. 05-architectures)")
    parser.add_argument("--topic", help="Verify a specific topic directory path")
    parser.add_argument("--timeout-s", type=int, default=60, help="Per-topic timeout (seconds)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp directories for debugging")
    args = parser.parse_args()

    selected: list[Path] = []
    if args.topic:
        selected = [Path(args.topic).resolve()]
    else:
        topics = discover_topics(ROOT)
        if args.chapter:
            topics = [t for t in topics if t.relative_to(ROOT).parts and t.relative_to(ROOT).parts[0] == args.chapter]
        if args.all or args.chapter:
            selected = topics
    if not selected:
        print("No topics selected. Use --all, --chapter, or --topic.")
        return 2

    base_temp = Path(tempfile.mkdtemp(prefix="mlfs_verify_"))
    results: list[Result] = []
    try:
        for t in selected:
            rel = str(t.relative_to(ROOT))
            try:
                temp_topic = copy_topic_to_temp(t, base_temp)
                rc, out, err = run_pytest(temp_topic, timeout_s=args.timeout_s)
                ok = rc == 0
            except subprocess.TimeoutExpired as e:
                rc = 124
                out = e.stdout or ""
                err = (e.stderr or "") + "\nTIMEOUT"
                ok = False
            except Exception as e:
                rc = 1
                out = ""
                err = repr(e)
                ok = False
            results.append(Result(topic_dir=rel, ok=ok, returncode=rc, stdout=out, stderr=err))
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {rel}")
            if not ok:
                # keep output short; user can rerun per-topic with --topic
                tail = (out + "\n" + err).strip().splitlines()[-30:]
                if tail:
                    print("\n".join("  " + line for line in tail))
                print()

        failed = [r for r in results if not r.ok]
        print(f"Summary: {len(results) - len(failed)}/{len(results)} passed")
        return 0 if not failed else 1
    finally:
        if args.keep_temp:
            print(f"Temp kept at: {base_temp}")
        else:
            shutil.rmtree(base_temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

