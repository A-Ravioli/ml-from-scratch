"""Checks for exercise.py (testing, debugging & reproducibility).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import importlib.util
import math
import random
import re

import numpy as np
import pytest


def _message(fn, *args, **kwargs):
    """Run fn, expecting an AssertionError, and return its message."""
    with pytest.raises(AssertionError) as info:
        fn(*args, **kwargs)
    return str(info.value)


# ---------------------------------------------------------------- stage 1: seeds and generators


def test_stage1_as_rng_passes_generators_through(nb):
    g = np.random.default_rng(5)
    out = nb.as_rng(g)
    assert out is g, (
        "as_rng(generator) must return that SAME generator object, not a new one or a copy: "
        "callers pass their generator down so that one stream of random numbers flows through "
        "the whole program.")


def test_stage1_as_rng_seeds_are_reproducible(nb):
    a = nb.as_rng(42)
    assert isinstance(a, np.random.Generator), f"as_rng(42) should return a np.random.Generator, got {type(a).__name__}"
    want = np.random.default_rng(42).random(5)
    assert np.array_equal(a.random(5), want), "as_rng(42) should behave exactly like np.random.default_rng(42)"
    assert np.array_equal(nb.as_rng(np.int64(42)).random(5), want), "NumPy integers are valid seeds too"
    assert not np.array_equal(nb.as_rng(43).random(5), want), "different seeds must give different streams"


def test_stage1_as_rng_none_is_fresh(nb):
    a, b = nb.as_rng(None), nb.as_rng()
    assert isinstance(a, np.random.Generator) and isinstance(b, np.random.Generator), (
        "as_rng(None) should return a new np.random.Generator")
    assert a is not b and not np.array_equal(a.random(4), b.random(4)), (
        "two calls to as_rng(None) should give independent, unpredictable generators "
        "(np.random.default_rng() with no seed)")


def test_stage1_as_rng_rejects_other_types(nb):
    for bad in (1.5, "42", True, [1, 2]):
        with pytest.raises(TypeError):
            nb.as_rng(bad)


def test_stage1_set_seed_seeds_everything(nb):
    g1 = nb.set_seed(123)
    r1, n1 = [random.random() for _ in range(3)], np.random.rand(3)
    g2 = nb.set_seed(123)
    r2, n2 = [random.random() for _ in range(3)], np.random.rand(3)
    assert r1 == r2, "after set_seed(123) twice, Python's random module should give the same numbers: random.seed(seed)"
    assert np.array_equal(n1, n2), "after set_seed(123) twice, np.random.rand should give the same numbers: np.random.seed(seed)"
    assert isinstance(g1, np.random.Generator), f"set_seed should return a np.random.Generator, got {type(g1).__name__}"
    assert np.array_equal(g1.random(3), np.random.default_rng(123).random(3)), (
        "set_seed(seed) should return np.random.default_rng(seed)")
    assert np.array_equal(g2.random(3), np.random.default_rng(123).random(3))
    nb.set_seed(124)
    assert random.random() != r1[0], "a different seed should give different numbers"


def test_stage1_set_seed_seeds_torch_if_installed(nb):
    if importlib.util.find_spec("torch") is None:
        return  # nothing to check without PyTorch
    import torch

    nb.set_seed(7)
    a = torch.rand(3)
    nb.set_seed(7)
    b = torch.rand(3)
    assert torch.equal(a, b), "PyTorch is installed, so set_seed should also call torch.manual_seed(seed)"


# ---------------------------------------------------------------- stage 2: assert_close


def test_stage2_passes_within_tolerance(nb):
    assert nb.assert_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None, "assert_close should return None when values match"
    nb.assert_close(np.array([[1.0, 2.0]]), np.array([[1.0 + 1e-9, 2.0 - 1e-7]]))
    nb.assert_close(1000.0005, 1000.0)                 # within rtol * 1000 = 1e-3
    nb.assert_close(1e-9, 0.0)                         # within atol = 1e-8
    nb.assert_close(0.1 + 0.2, 0.3)
    nb.assert_close(3, 3.0, name="ints are fine")


def test_stage2_fails_beyond_tolerance(nb):
    with pytest.raises(AssertionError):
        nb.assert_close(1000.002, 1000.0)              # beyond rtol * 1000
    with pytest.raises(AssertionError):
        nb.assert_close(1e-7, 0.0)                     # beyond atol
    with pytest.raises(AssertionError):
        nb.assert_close([1.0, 2.0], [1.0, 2.1])
    nb.assert_close(1.05, 1.0, rtol=0.1)               # custom tolerances are respected
    with pytest.raises(AssertionError):
        nb.assert_close(1.05, 1.0, rtol=0.01, atol=0.0)


def test_stage2_tolerance_is_relative_to_expected(nb):
    nb.assert_close(1.5, 2.0, rtol=0.3, atol=0.0)      # 0.5 <= 0.3 * |2|
    nb.assert_close(1.5, 2.0, rtol=0.25, atol=0.0)     # exactly AT the tolerance: 0.5 <= 0.25 * 2 passes (<=, not <)
    with pytest.raises(AssertionError):
        # |2 - 1| = 1 > 0.6 * |expected| = 0.6, even though 1 <= 0.6 * |actual| = 1.2
        nb.assert_close(2.0, 1.0, rtol=0.6, atol=0.0)


def test_stage2_nan_and_inf(nb):
    nb.assert_close([1.0, np.nan, np.inf, -np.inf], [1.0, np.nan, np.inf, -np.inf])
    for actual, expected in ((np.nan, 1.0), (1.0, np.nan), (np.inf, -np.inf), (np.inf, 1e308), (1e308, np.inf)):
        with pytest.raises(AssertionError):
            nb.assert_close(actual, expected)


def test_stage2_rejects_complex(nb):
    with pytest.raises(TypeError):
        nb.assert_close(np.array([1 + 2j]), np.array([1 + 0j]))   # float() would drop the 2j and "pass"
    with pytest.raises(TypeError):
        nb.assert_close(1.0, 1 + 1e-3j)


def test_stage2_shapes_must_match(nb):
    msg = _message(nb.assert_close, np.zeros(3), np.zeros((3, 1)), name="weights")
    assert "weights" in msg and "(3,)" in msg and "(3, 1)" in msg, (
        f"a shape mismatch should raise AssertionError naming both shapes, (3,) and (3, 1); got: {msg!r}. "
        "don't let broadcasting compare them: a (3,) and a (3, 1) array broadcast to (3, 3).")
    with pytest.raises(AssertionError):
        nb.assert_close(np.ones(4), 1.0)  # no broadcasting of the expected value either


def test_stage2_message_names_the_worst_element(nb):
    expected = np.arange(12.0).reshape(3, 4) / 4      # entry (2, 1) is 2.25
    actual = expected.copy()
    actual[2, 1] = 2.75                                # off by 0.5: the worst
    actual[0, 3] += 1e-3                               # a smaller violation
    msg = _message(nb.assert_close, actual, expected, name="logits")
    assert msg.startswith("logits"), f"the message should start with the name ('logits'); got: {msg!r}"
    assert re.search(r"\(\s*2\s*,\s*1\s*\)", msg), (
        f"the message should name the worst index as a tuple, (2, 1); got: {msg!r}. "
        "np.unravel_index turns a flat argmax back into a (row, column) index.")
    assert "2.75" in msg and "2.25" in msg, f"the message should show the actual (2.75) and expected (2.25) values; got: {msg!r}"
    assert "0.5" in msg and "0.222" in msg, (
        f"the message should give the abs error (0.5) and rel error (0.5 / 2.25 = 0.222); got: {msg!r}")
    assert re.search(r"\b2 of 12\b", msg), f"the message should say how many values differ: '2 of 12'; got: {msg!r}"


def test_stage2_worst_is_by_tolerance_and_nan_counts(nb):
    expected = np.array([1000.0, 1.0, 5.0])
    actual = np.array([1000.5, 1.01, 5.0])            # abs errors 0.5 and 0.01, but 1.01 is 10,000x out of tolerance
    msg = _message(nb.assert_close, actual, expected, rtol=1e-6, atol=0.0)
    assert re.search(r"\(\s*1\s*,?\s*\)", msg), (
        f"the worst element is the one furthest out of ITS tolerance: index (1,) (error 0.01 against a "
        f"tolerance of 1e-6) beats index (0,) (error 0.5 against a tolerance of 1e-3); got: {msg!r}")
    msg = _message(nb.assert_close, np.array([1.0, np.nan, 3.5]), np.array([1.0, 2.0, 3.0]))
    assert re.search(r"\(\s*1\s*,?\s*\)", msg), f"a nan where a number was expected is the worst mismatch; got: {msg!r}"
    msg = _message(nb.assert_close, 2.0, 1.0)
    assert msg.startswith("value"), f"the default name is 'value'; got: {msg!r}"


# ---------------------------------------------------------------- stage 3: random testing


def test_stage3_no_counterexample(nb):
    calls = []

    def generate(rng):
        calls.append(1)
        return rng.normal(size=3)

    got = nb.find_counterexample(lambda x: np.all(np.isfinite(x)), generate, n_trials=50, rng=0)
    assert got is None, f"the property holds every time, so find_counterexample should return None, got {got!r}"
    assert len(calls) == 50, f"with n_trials=50 the generator should be called 50 times, got {len(calls)}"


def test_stage3_finds_and_reproduces_the_first_failure(nb):
    def generate(rng):
        return int(rng.integers(0, 100))

    got = nb.find_counterexample(lambda x: x < 90, generate, n_trials=500, rng=3)
    stream = np.random.default_rng(3)
    want = next(x for x in (int(stream.integers(0, 100)) for _ in range(500)) if not x < 90)
    assert got == want, (
        f"with rng=3 the first failing input is {want}, got {got!r}. make ONE generator with "
        "as_rng(rng) before the loop, call generate(rng) once per trial, and return the first x that fails.")
    assert nb.find_counterexample(lambda x: x < 90, generate, n_trials=500, rng=3) == got, (
        "the same seed must find the same counterexample")


def test_stage3_uses_the_callers_generator(nb):
    g = np.random.default_rng(11)
    nb.find_counterexample(lambda x: True, lambda rng: rng.random(), n_trials=10, rng=g)
    ref = np.random.default_rng(11)
    ref.random(10)
    assert g.random() == ref.random(), (
        "find_counterexample should draw from the Generator it was given (as_rng returns it unchanged), "
        "advancing it by exactly one draw per trial")


def test_stage3_crashes_count_as_failures(nb):
    seen = []

    def generate(rng):
        seen.append(len(seen))
        return len(seen) - 1  # 0, 1, 2, ...

    got = nb.find_counterexample(lambda x: 1 / (x - 3) != 0, generate, n_trials=10, rng=0)
    assert got == 3, (
        f"the property raises ZeroDivisionError at x = 3; a crash is a failure, so 3 should be returned, got {got!r}. "
        "wrap the call to prop in try/except Exception.")
    assert len(seen) == 4, f"stop at the first failure: the generator should have been called 4 times, not {len(seen)}"


# ---------------------------------------------------------------- stage 4: bisection


class _Counter:
    """is_bad(i) for a bug introduced in version k, that refuses to be called too often, so a
    binary search whose interval stops shrinking fails instead of looping forever."""

    def __init__(self, n, k, limit):
        self.n, self.k, self.limit, self.calls = n, k, limit, []

    def __call__(self, i):
        assert 0 <= i < self.n, f"is_bad was called with version {i}, outside 0..{self.n - 1}"
        self.calls.append(i)
        if len(self.calls) > self.limit:
            raise AssertionError(
                f"is_bad was called more than {self.limit} times for {self.n} versions (last calls: "
                f"{self.calls[-4:]}). does your interval [lo, hi] shrink at every step? when mid is good, "
                "the answer is AFTER it: lo = mid + 1, not lo = mid.")
        return i >= self.k


def test_stage4_finds_every_boundary(nb):
    for n in range(1, 41):
        limit = math.ceil(math.log2(n)) + 1
        for k in range(n):
            is_bad = _Counter(n, k, limit=limit + 20)
            got = nb.first_bad(n, is_bad)
            assert got == k, f"with {n} versions where version {k} introduced the bug, first_bad returned {got}"
            assert len(is_bad.calls) <= limit, (
                f"with n = {n}, first_bad made {len(is_bad.calls)} calls to is_bad; binary search needs at most "
                f"ceil(log2(n)) + 1 = {limit}. halve the range [lo, hi] each time instead of scanning.")


def test_stage4_large_history(nb):
    is_bad = _Counter(1_000_000, 731_042, limit=100)
    assert nb.first_bad(1_000_000, is_bad) == 731_042
    assert len(is_bad.calls) <= 21, f"a million versions should take at most 21 tests, took {len(is_bad.calls)}"


def test_stage4_requires_a_bad_newest_version(nb):
    with pytest.raises(ValueError):
        nb.first_bad(10, _Counter(10, 10, limit=30))   # no version is bad
