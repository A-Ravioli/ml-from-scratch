"""The `mlfs` module registry: `from mlfs.tensor import Tensor` → a topic's exercise notebook.

The registry lives in `mlfs/registry.toml`:

    [modules]
    tensor = "14-neural-networks-from-scratch/03-a-tensor-autograd-engine"

By default a registered module loads the learner's `exercise.py` for that topic. It loads the
reference `solutions/exercise.py` instead when MLFS_SOLUTIONS=1, or when the module is named
in MLFS_REFERENCE (comma-separated; `*` means every module).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import tomllib
from pathlib import Path

from mlfs import notebooks

# names that are real submodules of `mlfs` and so can't be registry entries
RESERVED = frozenset({"notebooks", "registry", "checks", "lesson", "ui", "pytest_plugin"})


def registry_path() -> Path:
    return notebooks.repo_root() / "mlfs" / "registry.toml"


def modules() -> dict[str, str]:
    path = registry_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return dict(tomllib.load(f).get("modules", {}))


def wants_reference(name: str) -> bool:
    if notebooks.use_solutions():
        return True
    raw = os.environ.get("MLFS_REFERENCE", "")
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return "*" in names or name in names


def notebook_for(name: str) -> Path:
    try:
        rel = modules()[name]
    except KeyError:
        raise ModuleNotFoundError(f"mlfs.{name} is not in {registry_path()}") from None
    topic = notebooks.repo_root() / rel
    return notebooks.resolve(topic, "exercise", solution=wants_reference(name))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix, _, name = fullname.partition(".")
        if prefix != "mlfs" or not name or "." in name or name in RESERVED:
            return None
        if name not in modules():
            return None
        return importlib.util.spec_from_file_location(fullname, notebook_for(name))


def install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
