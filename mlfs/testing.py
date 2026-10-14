"""Helpers for writing topic tests (test_*.py files), not for learners' notebooks.

    from mlfs.testing import bounded

    out = bounded(nb.decode, bits, code, what="decode")

`bounded` runs a call under a budget of executed Python lines, in the current thread. If the
code under test runs past the budget (an infinite loop, a missing iteration cap), it raises
an AssertionError with a teaching message instead of hanging pytest or freezing a live marimo
check. Unlike a timeout thread, the runaway loop really stops, nothing keeps running in the
background, and the result doesn't depend on machine speed.

Only Python lines count. Time spent inside NumPy calls is free, so a budget of a few hundred
thousand lines is generous for correct code and still stops a runaway loop within a second.
"""
from __future__ import annotations

import sys
from typing import Any, Callable


class _Runaway(Exception):
    pass


def bounded(fn: Callable, *args: Any, budget: int = 500_000, what: str | None = None,
            hint: str = "", **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs), failing with AssertionError after `budget` executed lines.

    The previous trace function (a debugger, coverage) is restored afterwards.
    """
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
            if count > budget:
                raise _Runaway
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return fn(*args, **kwargs)
    except _Runaway:
        name = what or getattr(fn, "__name__", "the function")
        raise AssertionError(
            f"{name} didn't finish after {budget:,} steps: a loop never ends (or runs far longer than "
            f"it should). check that every loop makes progress and has a way out. {hint}".strip()
        ) from None
    finally:
        sys.settrace(old)
