"""The `mlfs` module registry: `from mlfs.linalg import lu_solve` → topics' exercise notebooks.

The registry lives in `mlfs/registry.toml`. A module is built from one topic, or merged from
several (in order; later topics win on name clashes):

    [modules]
    tensor = "14-neural-networks-from-scratch/03-a-tensor-autograd-engine"
    linalg = ["02-linear-algebra/03-solving-linear-systems", "02-linear-algebra/04-orthogonality-and-least-squares"]

A module exposes the functions and classes its notebooks define (their `@app.function` and
`@app.class_definition` cells). By default those come from the learner's `exercise.py`. The
reference `solutions/exercise.py` is used instead when MLFS_SOLUTIONS=1, or when the module is
named in MLFS_REFERENCE (comma-separated; `*` means every module, `none` means none).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType

from mlfs import notebooks

# names that are real submodules of `mlfs` and so can't be registry entries
RESERVED = frozenset({"notebooks", "registry", "checks", "lesson", "ui", "pytest_plugin",
                      "datasets", "contracts"})


def registry_path() -> Path:
    return notebooks.repo_root() / "mlfs" / "registry.toml"


def modules() -> dict[str, list[str]]:
    path = registry_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f).get("modules", {})
    return {name: [v] if isinstance(v, str) else list(v) for name, v in raw.items()}


def wants_reference(name: str) -> bool:
    if notebooks.use_solutions():
        return True
    raw = os.environ.get("MLFS_REFERENCE", "")
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return "*" in names or name in names


def notebooks_for(name: str) -> list[Path]:
    try:
        topics = modules()[name]
    except KeyError:
        raise ModuleNotFoundError(f"mlfs.{name} is not in {registry_path()}") from None
    solution = wants_reference(name)
    root = notebooks.repo_root()
    return [notebooks.resolve(root / t, "exercise", solution=solution) for t in topics]


def public_definitions(nb: ModuleType) -> dict[str, object]:
    """Functions and classes a notebook defines itself (not its imports)."""
    return {k: v for k, v in vars(nb).items()
            if not k.startswith("_") and getattr(v, "__module__", None) == nb.__name__}


class _Loader(importlib.abc.Loader):
    def __init__(self, name: str):
        self.name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        paths = notebooks_for(self.name)
        module.__dict__["__notebooks__"] = [str(p) for p in paths]
        for path in paths:
            module.__dict__.update(public_definitions(notebooks.load(path)))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix, _, name = fullname.partition(".")
        if prefix != "mlfs" or not name or "." in name or name in RESERVED:
            return None
        if name not in modules():
            return None
        return importlib.machinery.ModuleSpec(fullname, _Loader(name))


def install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
