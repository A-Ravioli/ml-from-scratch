"""Load marimo notebooks as Python modules, and decide learner vs reference code.

Exercise notebooks define their public API in `@app.function` / `@app.class_definition`
cells, which marimo exposes as plain module attributes. Importing a notebook runs its
setup cell and those definitions, but no other cells, so it's cheap and side-effect free.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_CACHE: dict[tuple[str, float], ModuleType] = {}


def repo_root() -> Path:
    """The curriculum root.

    $MLFS_ROOT if set; else the nearest directory at or above the working directory that
    contains `CURRICULUM.md` and `mlfs/registry.toml` (so a clone or git worktree uses its
    own content); else the checkout this package was installed from.
    """
    env = os.environ.get("MLFS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "CURRICULUM.md").exists() and (d / "mlfs" / "registry.toml").exists():
            return d
    return Path(__file__).resolve().parent.parent


def use_solutions() -> bool:
    """True when MLFS_SOLUTIONS=1: every notebook load resolves to the reference solution."""
    return os.environ.get("MLFS_SOLUTIONS", "") not in ("", "0", "false", "no")


def topic_dir(path: str | Path) -> Path:
    """The topic (or capstone) directory for a file inside it, including files in `solutions/`."""
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    return p.parent if p.name == "solutions" else p


def resolve(directory: str | Path, name: str, *, solution: bool | None = None) -> Path:
    """Path of notebook `name` (e.g. "exercise") in a topic directory.

    Learner code is `<topic>/<name>.py`; the reference is `<topic>/solutions/<name>.py`.
    `solution=None` defers to MLFS_SOLUTIONS.
    """
    base = topic_dir(directory)
    if solution is None:
        solution = use_solutions()
    path = base / "solutions" / f"{name}.py" if solution else base / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"no notebook at {path}")
    return path


def load(path: str | Path, module_name: str | None = None) -> ModuleType:
    """Import a marimo notebook file as a module (cached until the file changes)."""
    path = Path(path).resolve()
    key = (str(path), path.stat().st_mtime)
    if module_name is None and key in _CACHE:
        return _CACHE[key]
    if module_name is None:
        module_name = "_mlfs_nb_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load notebook {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if module_name.startswith("_mlfs_nb_"):
        _CACHE[key] = module
    return module


def load_sibling(name: str, *, solution: bool | None = None) -> ModuleType:
    """From inside a notebook, load another notebook in the same topic.

    Labs use this to import the learner's own exercise:  `ex = load_sibling("exercise")`.
    """
    import marimo as mo

    here = mo.notebook_dir()
    if here is None:
        raise RuntimeError("load_sibling() must be called from inside a marimo notebook")
    return load(resolve(here, name, solution=solution))
