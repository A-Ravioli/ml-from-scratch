"""Live checks: run a topic's pytest tests inside the marimo notebook as you work.

In an exercise notebook, a check cell looks like

    checks.run(1, mean=mean)

marimo re-runs that cell every time the `mean` cell changes, so feedback is instant. The
checks are the same `test_stageN_*` functions that `pytest` runs: each one takes a single
argument `nb`, which here is a namespace built from the keyword arguments you pass.
"""
from __future__ import annotations

import importlib.util
import inspect
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mlfs import notebooks


@dataclass
class Result:
    name: str
    status: str  # "pass" | "fail" | "error" | "todo" | "skip"
    message: str = ""


class _Impl:
    """Namespace of the definitions passed to `run`, with a helpful error for missing ones."""

    def __init__(self, stage: int, defs: dict[str, Any]):
        self.__dict__.update(defs)
        self._stage = stage

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"this check uses `{name}`, but it wasn't passed in. "
            f"call `checks.run({self._stage}, ..., {name}={name})`"
        )


def _load_tests(path: Path):
    spec = importlib.util.spec_from_file_location(f"_mlfs_checks_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(tests_path: Path, stage: int) -> list[Callable]:
    """The `test_stage{stage}_*` functions in a test file, in definition order."""
    module = _load_tests(tests_path)
    prefix = f"test_stage{stage}_"
    return [fn for name, fn in vars(module).items() if name.startswith(prefix) and callable(fn)]


def _is_slow(fn: Callable) -> bool:
    return any(getattr(m, "name", None) == "slow" for m in getattr(fn, "pytestmark", []))


def _param_sets(fn: Callable) -> list[dict[str, Any]]:
    """Expand @pytest.mark.parametrize marks into keyword-argument sets (cartesian product)."""
    sets: list[dict[str, Any]] = [{}]
    for mark in getattr(fn, "pytestmark", []):
        if getattr(mark, "name", None) != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        rows = []
        for v in values:
            if type(v).__name__ == "ParameterSet":  # pytest.param(...) -> its values
                v = v.values
            if len(names) == 1 and not (isinstance(v, tuple) and len(v) == 1):
                v = (v,)
            rows.append(dict(zip(names, v)))
        sets = [{**a, **b} for a in sets for b in rows]
    return sets


def _call(fn: Callable, impl: _Impl, stage: int, params: dict[str, Any] | None = None) -> Result:
    name = fn.__name__.removeprefix(f"test_stage{stage}_").replace("_", " ")
    if params:
        name += " [" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + "]"
    if _is_slow(fn):
        return Result(name, "skip", "slow check: run it with pytest")
    try:
        params = params or {}
        positional = [p for p in inspect.signature(fn).parameters if p not in params]
        if positional:
            fn(impl, **params)
        else:
            fn(**params)
    except NotImplementedError as e:
        return Result(name, "todo", str(e) or "not implemented yet")
    except AssertionError as e:
        msg = str(e).strip() or traceback.format_exc(limit=-1).strip().splitlines()[-1]
        return Result(name, "fail", msg)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # pytest.skip/fail raise BaseException subclasses
        kind = type(e).__name__
        if kind == "Skipped":
            return Result(name, "skip", str(e))
        if kind == "Failed":
            return Result(name, "fail", str(e))
        return Result(name, "error", f"{kind}: {e}")
    return Result(name, "pass")


def run_stage(directory: str | Path, stage: int, defs: dict[str, Any],
              tests: str = "test_exercise.py") -> list[Result]:
    """Run one stage's checks against `defs`. Usable outside marimo (returns raw results).
    Tests parametrized with @pytest.mark.parametrize run once per parameter set."""
    tests_path = notebooks.topic_dir(directory) / tests
    impl = _Impl(stage, defs)
    return [_call(fn, impl, stage, params) for fn in collect(tests_path, stage)
            for params in _param_sets(fn)]


_ICON = {"pass": "✅", "fail": "❌", "error": "💥", "todo": "⬜", "skip": "⏭️"}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def render(stage: int, results: list[Result]):
    import marimo as mo

    if not results:
        return mo.callout(mo.md(f"no checks found for stage {stage}"), kind="warn")
    statuses = {r.status for r in results}
    passed = sum(r.status == "pass" for r in results)
    if statuses <= {"pass", "skip"}:
        return mo.callout(
            mo.md(f"**stage {stage}: all {passed} checks pass.** move on to stage {stage + 1}."),
            kind="success",
        )
    if statuses <= {"todo", "skip"}:
        return mo.callout(mo.md(f"**stage {stage}: not started.** {results[0].message}"),
                          kind="neutral")
    rows = "\n".join(f"| {_ICON[r.status]} | {_cell(r.name)} | {_cell(r.message)} |"
                     for r in results)
    table = f"| | check | details |\n|---|---|---|\n{rows}"
    kind = "danger" if "error" in statuses else "warn"
    return mo.callout(mo.md(f"**stage {stage}: {passed}/{len(results)} checks pass**\n\n{table}"),
                      kind=kind)


def run(stage: int, /, *, _tests: str = "test_exercise.py", **defs: Any):
    """Run stage `stage`'s checks against the given definitions and render the results.

    Does nothing outside an interactive notebook (pytest covers that case).
    """
    import marimo as mo

    if not mo.running_in_notebook():
        return None
    return render(stage, run_stage(mo.notebook_dir(), stage, defs, _tests))
