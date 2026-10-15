import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import importlib.util
    import math
    import random

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # testing, debugging & reproducibility · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks your code live as you edit. this topic is about the
    tools that keep everything else honest: controlled randomness, comparisons with
    tolerances, tests that hunt for bugs, and a systematic way to find where a bug came from.

    | stage | you build |
    |---|---|
    | 1 | `as_rng`, `set_seed`: reproducible randomness → `mlfs.utils` |
    | 2 | `assert_close`: compare arrays with tolerances, and say exactly what differs → `mlfs.utils` |
    | 3 | `find_counterexample`: random testing that reproduces its failures |
    | 4 | `first_bad`: find the first broken version with binary search, like `git bisect` |

    `as_rng`, `set_seed` and `assert_close` become part of `mlfs.utils`. every function in the
    course that uses randomness takes an `rng` argument and passes it through `as_rng`, and
    many later tests compare arrays with `assert_close`. you can also run the checks from a
    terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def as_rng(rng: "None | int | np.random.Generator" = None) -> np.random.Generator:
    """stage 1: turn the course's standard `rng` argument into a NumPy random Generator.

    None -> a fresh np.random.default_rng() (unpredictable); an int (Python or NumPy
    integer) -> np.random.default_rng(that int) (reproducible); a Generator -> that SAME
    object, unchanged (so the caller's stream keeps advancing).
    raises TypeError for anything else (including bools and floats).
    """
    # step 1: if rng is None, return np.random.default_rng()
    if rng is None:
        return np.random.default_rng()
    # step 2: if it's already a np.random.Generator, return it as is (not a copy)
    if isinstance(rng, np.random.Generator):
        return rng
    # step 3: if it's an int or np.integer (but NOT a bool), return np.random.default_rng(rng)
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
        return np.random.default_rng(int(rng))
    # step 4: anything else: raise TypeError with a message saying what was passed
    raise TypeError(f"rng must be None, an int seed or a np.random.Generator, got {type(rng).__name__}")


@app.function
def set_seed(seed: int) -> np.random.Generator:
    """stage 1: seed every global random number generator, for scripts and notebooks.

    seeds Python's `random` module, NumPy's legacy global generator (np.random.seed) and,
    if PyTorch is installed, torch (torch.manual_seed). returns np.random.default_rng(seed),
    a fresh Generator to pass around as `rng`.
    """
    # step 1: random.seed(seed) and np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # step 2: if importlib.util.find_spec("torch") is not None, import torch and torch.manual_seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
    # step 3: return np.random.default_rng(seed)
    return np.random.default_rng(seed)


@app.cell(hide_code=True)
def _():
    checks.run(1, as_rng=as_rng, set_seed=set_seed)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def assert_close(actual, expected, *, rtol: float = 1e-06, atol: float = 1e-08,
                 name: str = "value") -> None:
    """stage 2: check that two arrays (or numbers) are equal up to a tolerance; return None if so.

    element i passes if |actual[i] - expected[i]| <= atol + rtol * |expected[i]|, or if both
    are nan, or if both are the same infinity. the shapes must match EXACTLY (no broadcasting).
    real numbers only: raise TypeError if either input is complex (converting a complex
    number to float would silently drop its imaginary part).
    otherwise raise AssertionError with this message (index is a tuple, like (2, 5)):

        {name}: shape mismatch: actual {shape}, expected {shape}
    or
        {name}: {count} of {size} values differ (rtol={rtol:g}, atol={atol:g})
          worst at index {index}: actual {a:.10g}, expected {e:.10g}
          abs error {abs:.3g}, rel error {rel:.3g}

    the worst element is the one with the largest |actual - expected| / (atol + rtol*|expected|)
    (a nan mismatch counts as infinitely bad). rel error is abs error / |expected| (inf if 0).
    """
    # step 1: reject complex input (np.iscomplexobj), convert both with np.asarray(..., dtype=float),
    # step 1: then compare the shapes first
    if np.iscomplexobj(actual) or np.iscomplexobj(expected):
        raise TypeError(f"{name}: assert_close compares real numbers, got complex input")
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shape mismatch: actual {a.shape}, expected {e.shape}")
    # step 2: the tolerance per element, and which elements pass: within tolerance (only
    # step 2: when expected is finite: an infinite expected value would make tol infinite),
    # step 2: or exactly equal (that covers matching infinities), or both nan
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - e)
        tol = atol + rtol * np.abs(e)
        ok = (np.isfinite(e) & (diff <= tol)) | (a == e) | (np.isnan(a) & np.isnan(e))
    # step 3: if every element passes, return None
    if ok.all():
        return None
    # step 4: the badness of each failing element is diff / tol (nan or a zero tol -> inf);
    # step 4: the worst index is np.unravel_index(argmax, shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        badness = np.where(ok, -1.0, diff / tol)
    badness = np.where(~ok & ~(badness >= 0), np.inf, badness)  # nan -> inf
    worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(badness)), a.shape))
    # step 5: build the message from the template in the docstring and raise AssertionError
    av, ev = float(a[worst]), float(e[worst])
    abs_err = abs(av - ev)
    rel_err = abs_err / abs(ev) if ev != 0 else math.inf
    raise AssertionError(
        f"{name}: {int((~ok).sum())} of {a.size} values differ (rtol={rtol:g}, atol={atol:g})\n"
        f"  worst at index {worst}: actual {av:.10g}, expected {ev:.10g}\n"
        f"  abs error {abs_err:.3g}, rel error {rel_err:.3g}")


@app.cell(hide_code=True)
def _():
    checks.run(2, assert_close=assert_close)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def find_counterexample(prop, generate, *, n_trials: int = 100, rng=None):
    """stage 3: random testing. look for an input on which the property `prop` fails.

    rng = as_rng(rng). then, n_trials times: x = generate(rng) (one call per trial, always
    with that same Generator), and check prop(x). return the first x for which prop(x) is
    falsy or raises an exception (a crash is a failure too). return None if every trial passes.
    the same integer seed always finds the same counterexample.
    """
    # step 1: rng = as_rng(rng)
    rng = as_rng(rng)
    for _ in range(n_trials):
        # step 2: x = generate(rng)
        x = generate(rng)
        # step 3: call prop(x) inside try/except Exception: a crash means x is a counterexample
        try:
            passed = prop(x)
        except Exception:
            return x
        # step 4: if the result is falsy, return x
        if not passed:
            return x
    # step 5: after n_trials passes, return None
    return None


@app.cell(hide_code=True)
def _():
    checks.run(3, find_counterexample=find_counterexample)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def first_bad(n: int, is_bad) -> int:
    """stage 4: binary search for the first bad version, like `git bisect`.

    versions are numbered 0 .. n-1 (oldest first). is_bad(i) says whether version i has the
    bug; once a version is bad, every later one is too. the newest version, n-1, must be bad:
    raise ValueError if is_bad(n - 1) is False. returns the smallest i with is_bad(i) True,
    calling is_bad at most ceil(log2(n)) + 1 times, and only with 0 <= i < n.
    """
    # step 1: check is_bad(n - 1); if it's False, raise ValueError
    if not is_bad(n - 1):
        raise ValueError(f"the newest version ({n - 1}) must be bad to bisect")
    # step 2: keep a range [lo, hi] known to contain the first bad version: lo = 0, hi = n - 1
    lo, hi = 0, n - 1
    # step 3: while lo < hi: test the middle mid = (lo + hi) // 2.
    # step 3: if it's bad, the first bad one is at mid or before (hi = mid); else after it (lo = mid + 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if is_bad(mid):
            hi = mid
        else:
            lo = mid + 1
    # step 4: when lo == hi, that's the answer
    return lo


@app.cell(hide_code=True)
def _():
    checks.run(4, first_bad=first_bad)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `as_rng`, `set_seed` and `assert_close` are part of
    `mlfs.utils`, and chapter 00's library is complete. then open the lab:
    `marimo edit lab.py`. it plants bugs that look fine in a plot, hides one from a thousand
    random tests, and asks whether a "better" model is really better.
    """)
    return


if __name__ == "__main__":
    app.run()
