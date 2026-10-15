"""Checks for exercise.py (floating point & stability).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import math
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from scipy import special


@contextmanager
def _no_warnings():
    """Turn every warning (overflow, divide by zero, invalid value) into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            yield


class _Runaway(Exception):
    pass


def _bounded(fn, *args, budget=200_000, what=None):
    """Call fn(*args), but give up with a teaching AssertionError after `budget` executed lines,
    so a loop that never ends fails the check instead of freezing it."""
    count = [0]

    def tracer(frame, event, arg):
        count[0] += 1
        if count[0] > budget:
            raise _Runaway
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return fn(*args)
    except _Runaway:
        raise AssertionError(
            f"{what or fn.__name__} didn't finish after {budget:,} steps: a loop never ends. for decompose, "
            "check the input FIRST: 0 can be doubled forever without reaching 1, and inf halved forever "
            "without dropping below 2. for machine_epsilon, make sure eps shrinks every time.") from None
    finally:
        sys.settrace(old)


def _rel(got, want):
    return abs(got - want) / abs(want)


# ---------------------------------------------------------------- stage 1: inside a float


def test_stage1_decompose_hand_examples(nb):
    for x, want in ((6.0, (0, 1.5, 2)), (-0.375, (1, 1.5, -2)), (1.0, (0, 1.0, 0)), (0.1, (0, 1.6, -4))):
        got = _bounded(nb.decompose, x, what=f"decompose({x})")
        assert len(got) == 3, f"decompose should return (sign, mantissa, exponent), got {got!r}"
        s, m, e = got
        assert s == want[0] and e == want[2] and math.isclose(m, want[1], rel_tol=1e-15), (
            f"decompose({x}) should be {want}, got {got!r}. "
            + ("the mantissa must satisfy 1 <= mantissa < 2 (math.frexp uses [0.5, 1): don't copy that)."
               if not 1 <= m < 2 else
               "the sign is 0 for positive numbers and 1 for negative ones." if s != want[0] else ""))


def test_stage1_decompose_reconstructs_exactly(nb):
    rng = np.random.default_rng(0)
    xs = list(rng.normal(size=20) * 10.0 ** rng.integers(-30, 30, size=20)) + [1e300, -3e-300, 5e-324, 2.0, 0.75]
    for x in xs:
        s, m, e = _bounded(nb.decompose, float(x), what=f"decompose({x})")
        assert isinstance(e, int) and s in (0, 1), f"decompose({x}): sign should be 0 or 1 and exponent an int"
        assert 1.0 <= m < 2.0, f"decompose({x}): mantissa {m} is outside [1, 2)"
        fm, fe = math.frexp(float(x))
        assert m == abs(fm) * 2 and e == fe - 1, (
            f"decompose({x}) gave mantissa {m!r}, exponent {e}; expected {abs(fm) * 2!r}, {fe - 1}. "
            "halving and doubling by 2 is exact in binary, so the mantissa should come out EXACT.")
        assert (-1) ** s * m * 2.0 ** e == x or (-1) ** s * math.ldexp(m, e) == x


def test_stage1_decompose_rejects_zero_and_non_finite(nb):
    for bad in (0.0, -0.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _bounded(nb.decompose, bad, what=f"decompose({bad})")


def test_stage1_machine_epsilon(nb):
    for dtype, want in ((np.float64, 2.0 ** -52), (np.float32, 2.0 ** -23), (np.float16, 2.0 ** -10)):
        got = _bounded(nb.machine_epsilon, dtype, what=f"machine_epsilon({dtype.__name__})")
        assert got == want, (
            f"machine_epsilon({dtype.__name__}) should be {want!r} (np.finfo({dtype.__name__}).eps), got {got!r}. "
            + ("that's the float64 value: do the arithmetic in the requested dtype (dtype(1), dtype(2), ...), "
               "not with Python floats." if got == 2.0 ** -52 else
               "off by a factor of 2: stop when 1 + eps/2 == 1, and return the last eps for which 1 + eps > 1."
               if got in (want / 2, want * 2) else ""))
    assert type(_bounded(nb.machine_epsilon, np.float32)) is float, "machine_epsilon should return a Python float"
    assert _bounded(nb.machine_epsilon) == 2.0 ** -52, "the default dtype is float64"


# ---------------------------------------------------------------- stage 2: cancellation


def test_stage2_easy_roots(nb):
    got = nb.quadratic_roots(1.0, -3.0, 2.0)
    assert len(got) == 2 and math.isclose(got[0], 1.0) and math.isclose(got[1], 2.0), (
        f"x^2 - 3x + 2 = (x - 1)(x - 2) has roots (1, 2) (smaller first), got {got!r}")
    got = nb.quadratic_roots(2.0, 0.0, -8.0)
    assert math.isclose(got[0], -2.0) and math.isclose(got[1], 2.0), f"2x^2 - 8 has roots (-2, 2), got {got!r}"
    assert nb.quadratic_roots(1.0, 0.0, 0.0) == (0.0, 0.0), "x^2 = 0 has the double root 0"
    got = nb.quadratic_roots(-1.0, 1.0, 6.0)  # -(x - 3)(x + 2)
    assert math.isclose(got[0], -2.0) and math.isclose(got[1], 3.0), f"-x^2 + x + 6 has roots (-2, 3), got {got!r}"


def test_stage2_tiny_root_is_accurate(nb):
    # x^2 + 1e8 x + 1: roots -1e8 (to 16 digits) and -1e-8 * (1 + 1e-16) ~ -1e-8
    big, tiny = nb.quadratic_roots(1.0, 1e8, 1.0)  # sorted: -1e8 comes first
    want_tiny = -1.0000000000000000e-08
    assert math.isclose(big, -1e8, rel_tol=1e-15), (
        f"the roots of x^2 + 1e8 x + 1, smaller first, are (-1e8, -1e-8); got {(big, tiny)!r}")
    assert _rel(tiny, want_tiny) < 1e-15, (
        f"the tiny root of x^2 + 1e8 x + 1 is -1.0000000000000000e-08, got {tiny!r} "
        f"(relative error {_rel(tiny, want_tiny):.1e}). "
        + ("that's the textbook formula (-b + sqrt(b^2 - 4ac)) / 2a: -b and the square root nearly cancel. "
           "compute q = -(b + sign(b) sqrt(disc)) / 2 and use the root c / q instead."
           if _rel(tiny, want_tiny) > 1e-3 else ""))


def test_stage2_both_signs_of_b(nb):
    small, big = nb.quadratic_roots(1.0, -1e8, 1.0)  # roots 1e-8 and 1e8
    assert _rel(small, 1e-8) < 1e-15 and _rel(big, 1e8) < 1e-15, (
        f"x^2 - 1e8 x + 1 has roots 1e-8 and 1e8, got {(small, big)!r}. with b < 0 the square root must "
        "get b's sign (math.copysign(sqrt(disc), b)), so that b and the root ADD instead of cancelling.")
    small, big = nb.quadratic_roots(3.0, 7e9, 2.0)
    for r in (small, big):
        assert abs(3.0 * r * r + 7e9 * r + 2.0) <= 1e-6 * max(1.0, 7e9 * abs(r)), (
            f"{r!r} isn't a root of 3x^2 + 7e9 x + 2 to working precision")
    assert _rel(small * big, 2.0 / 3.0) < 1e-14, "the product of the roots should be c / a (Vieta)"


def test_stage2_rejects_bad_input(nb):
    with pytest.raises(ValueError):
        nb.quadratic_roots(1.0, 0.0, 1.0)  # x^2 + 1: no real roots
    with pytest.raises(ValueError):
        nb.quadratic_roots(0.0, 2.0, 1.0)  # not a quadratic


# ---------------------------------------------------------------- stage 3: the sigmoid family


def test_stage3_sigmoid_values(nb):
    x = np.linspace(-30, 30, 121)
    got = np.asarray(nb.sigmoid(x))
    assert got.shape == x.shape, f"sigmoid should keep the shape {x.shape}, got {got.shape}"
    assert np.allclose(got, special.expit(x), rtol=1e-13, atol=0), "sigmoid disagrees with 1 / (1 + exp(-x))"
    assert float(np.asarray(nb.sigmoid(0.0))) == 0.5, "sigmoid(0) should be exactly 0.5 (a plain number works too)"
    M = np.array([[-2.0, 0.0], [1.0, 5.0]])
    assert np.asarray(nb.sigmoid(M)).shape == (2, 2), "sigmoid should work elementwise on any shape"


def test_stage3_sigmoid_never_overflows(nb):
    x = np.array([-1e4, -800.0, -40.0, 0.0, 40.0, 800.0, 1e4])
    try:
        with _no_warnings():
            got = np.asarray(nb.sigmoid(x))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(
            f"sigmoid raised an overflow warning ({e}). exp(800) is too big for a float. compute "
            "e = exp(-|x|), which is always <= 1, and use 1/(1+e) for x >= 0, e/(1+e) for x < 0. "
            "careful: np.where evaluates BOTH branches for every element.") from None
    assert np.all((got >= 0) & (got <= 1)), "sigmoid must stay in [0, 1]"
    assert got[0] == 0.0 and got[-1] == 1.0 and got[3] == 0.5
    assert _rel(got[2], 1 / (1 + math.exp(40))) < 1e-14, f"sigmoid(-40) should be ~4.25e-18, got {got[2]!r}"


def test_stage3_softplus(nb):
    x = np.linspace(-30, 30, 121)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = np.asarray(nb.softplus(x))
    assert got.shape == x.shape and np.allclose(got, np.logaddexp(0, x), rtol=1e-13, atol=0), (
        "softplus disagrees with log(1 + exp(x))")
    try:
        with _no_warnings():
            big = np.asarray(nb.softplus(np.array([1000.0, 1e4])))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(f"softplus overflowed ({e}): rewrite it as max(x, 0) + log1p(exp(-|x|))") from None
    assert np.array_equal(big, [1000.0, 1e4]), f"softplus(1000) should be 1000, got {big.tolist()}"
    tiny = float(np.asarray(nb.softplus(-40.0)))
    want = math.log1p(math.exp(-40.0))
    assert tiny > 0 and _rel(tiny, want) < 1e-14, (
        f"softplus(-40) should be {want!r} (about e^-40), got {tiny!r}. "
        + ("np.log(1 + t) loses t entirely when t < 1e-16 (1 + t rounds to 1): use np.log1p(t)."
           if tiny == 0 else ""))


def test_stage3_log_sigmoid(nb):
    x = np.linspace(-30, 30, 121)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = np.asarray(nb.log_sigmoid(x))
    assert got.shape == x.shape and np.allclose(got, special.log_expit(x), rtol=1e-13, atol=0), (
        "log_sigmoid disagrees with log(sigmoid(x))")
    try:
        with _no_warnings():
            ends = np.asarray(nb.log_sigmoid(np.array([-1000.0, -1e4, 1000.0])))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(f"log_sigmoid raised a warning ({e}): compute it as -softplus(-x)") from None
    assert np.array_equal(ends, [-1000.0, -1e4, -0.0]), (
        f"log_sigmoid at (-1000, -1e4, 1000) should be (-1000, -1e4, 0), got {ends.tolist()}. "
        "np.log(sigmoid(-1000)) is log(0) = -inf.")
    tiny = float(np.asarray(nb.log_sigmoid(40.0)))
    want = -math.log1p(math.exp(-40.0))
    assert tiny < 0 and _rel(tiny, want) < 1e-14, (
        f"log_sigmoid(40) should be {want!r}, got {tiny!r}. sigmoid(40) rounds to exactly 1.0, whose log "
        "is 0: work with -softplus(-x), which uses log1p.")


# ---------------------------------------------------------------- stage 4: logsumexp


def test_stage4_matches_scipy(nb):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 5, 3)) * 10
    for axis in (None, 0, 1, 2, -1, (0, 2)):
        for keepdims in (False, True):
            got = np.asarray(nb.logsumexp(x, axis=axis, keepdims=keepdims))
            want = special.logsumexp(x, axis=axis, keepdims=keepdims)
            assert got.shape == np.shape(want), (
                f"logsumexp(x of shape {x.shape}, axis={axis}, keepdims={keepdims}) should have shape "
                f"{np.shape(want)}, got {got.shape}")
            assert np.allclose(got, want, rtol=1e-12), f"logsumexp(axis={axis}, keepdims={keepdims}) is wrong"


def test_stage4_full_reduction_is_a_scalar(nb):
    got = nb.logsumexp(np.log(np.array([1.0, 2.0, 3.0, 4.0])))
    assert np.ndim(got) == 0, f"logsumexp over every entry should return a single number, got shape {np.shape(got)}"
    assert isinstance(got, (float, np.floating)), (
        f"a full reduction should return a plain number (a NumPy float), not a {type(got).__name__}: end with "
        "`return out[()]`, which turns a 0-d array into a number and leaves bigger arrays alone")
    assert math.isclose(float(got), math.log(10.0), rel_tol=1e-14), (
        f"logsumexp(log [1, 2, 3, 4]) should be log 10 = {math.log(10.0):.6f}, got {float(got)!r}")


def test_stage4_huge_and_tiny_inputs(nb):
    try:
        with _no_warnings():
            big = float(nb.logsumexp(np.array([1000.0, 1000.0])))
            small = float(nb.logsumexp(np.array([-1000.0, -1000.0])))
            rows = np.asarray(nb.logsumexp(np.array([[1e4, 0.0], [-1e4, -1e4 - 1.0]]), axis=1))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(
            f"logsumexp raised a warning ({e}). exp(1000) overflows: subtract the max first, "
            "log(sum(exp(x - m))) + m.") from None
    assert math.isclose(big, 1000.0 + math.log(2.0), rel_tol=1e-15), f"logsumexp([1000, 1000]) should be 1000 + log 2, got {big!r}"
    assert math.isclose(small, -1000.0 + math.log(2.0), rel_tol=1e-15), (
        f"logsumexp([-1000, -1000]) should be -1000 + log 2, got {small!r}. exp(-1000) underflows to 0, "
        "and log(0) = -inf: shifting by the max fixes the small end too.")
    want = [1e4, -1e4 + math.log1p(math.exp(-1.0))]
    assert np.allclose(rows, want, rtol=1e-15), (
        f"along axis=1, the max must be taken PER ROW (keepdims=True): expected {want}, got {rows.tolist()}")


def test_stage4_minus_infinity(nb):
    try:
        with _no_warnings():
            all_inf = float(nb.logsumexp(np.array([-np.inf, -np.inf])))
            some_inf = float(nb.logsumexp(np.array([-np.inf, 0.0, 0.0])))
            rows = np.asarray(nb.logsumexp(np.array([[-np.inf, -np.inf], [0.0, -np.inf]]), axis=1))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(
            f"logsumexp raised a warning with -inf inputs ({e}). if every entry is -inf, the max m is "
            "-inf and x - m is -inf - (-inf) = nan. replace a non-finite m by 0 (np.where(np.isfinite(m), m, 0)), "
            "and wrap the final log in np.errstate(divide='ignore'), since log(0) = -inf is the right answer.") from None
    assert all_inf == -np.inf, f"the log of a sum of zeros is -inf; logsumexp([-inf, -inf]) gave {all_inf!r}"
    assert math.isclose(some_inf, math.log(2.0)), f"logsumexp([-inf, 0, 0]) = log(0 + 1 + 1) = log 2, got {some_inf!r}"
    assert rows[0] == -np.inf and rows[1] == 0.0, f"row-wise logsumexp of [[-inf, -inf], [0, -inf]] should be [-inf, 0], got {rows.tolist()}"


# ---------------------------------------------------------------- stage 5: softmax


def test_stage5_softmax_hand_example(nb):
    got = np.asarray(nb.softmax(np.array([1.0, 2.0, 3.0])))
    e = np.exp([1.0, 2.0, 3.0])
    assert got.shape == (3,) and np.allclose(got, e / e.sum(), rtol=1e-14), (
        f"softmax([1, 2, 3]) should be {np.round(e / e.sum(), 6).tolist()}, got {np.round(got, 6).tolist()}")


def test_stage5_softmax_large_logits(nb):
    try:
        with _no_warnings():
            got = np.asarray(nb.softmax(np.array([1000.0, 1001.0, 1002.0])))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(
            f"softmax overflowed on logits around 1000 ({e}). subtracting the max changes nothing "
            "mathematically (it cancels between top and bottom) and keeps every exp <= 1.") from None
    e = np.exp([0.0, 1.0, 2.0])
    assert np.allclose(got, e / e.sum(), rtol=1e-14), (
        f"softmax([1000, 1001, 1002]) should equal softmax([0, 1, 2]) = {np.round(e / e.sum(), 6).tolist()}, "
        f"got {got.tolist()}")


def test_stage5_softmax_axis_and_shift_invariance(nb):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 4)) * 5
    P = np.asarray(nb.softmax(X))
    assert P.shape == X.shape and np.allclose(P.sum(axis=1), 1.0), (
        "by default, softmax works along the LAST axis: every row should sum to 1")
    assert np.allclose(P, special.softmax(X, axis=-1), rtol=1e-13), "softmax(X) disagrees with scipy's softmax along the last axis"
    P0 = np.asarray(nb.softmax(X, axis=0))
    assert np.allclose(P0.sum(axis=0), 1.0) and np.allclose(P0, special.softmax(X, axis=0), rtol=1e-13), (
        "softmax(X, axis=0) should normalize each COLUMN: take the max and the sum along axis 0 with keepdims=True")
    shifted = np.asarray(nb.softmax(X + rng.normal(size=(6, 1)) * 300))
    assert np.allclose(shifted, P, rtol=1e-12), (
        "adding a different constant to each row must not change its softmax. take the max per ROW "
        "(axis=-1, keepdims=True), not over the whole array.")


def test_stage5_softmax_masked_entries(nb):
    got = np.asarray(nb.softmax(np.array([[0.0, -np.inf, 0.0], [5.0, -np.inf, -np.inf]])))
    assert np.array_equal(got, [[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]]), (
        f"-inf logits (masked entries) should get probability exactly 0, got {got.tolist()}")


def test_stage5_log_softmax(nb):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 7)) * 4
    got = np.asarray(nb.log_softmax(X))
    assert got.shape == X.shape and np.allclose(got, special.log_softmax(X, axis=-1), rtol=1e-13), (
        "log_softmax(X) disagrees with log(softmax(X)) along the last axis")
    assert np.allclose(np.asarray(nb.log_softmax(X, axis=0)), special.log_softmax(X, axis=0), rtol=1e-13), (
        "log_softmax(X, axis=0) should normalize each column")
    try:
        with _no_warnings():
            extreme = np.asarray(nb.log_softmax(np.array([0.0, -1000.0])))
    except (RuntimeWarning, FloatingPointError) as e:
        raise AssertionError(f"log_softmax raised a warning ({e}): compute it as x - logsumexp(x, axis, keepdims=True)") from None
    assert np.allclose(extreme, [0.0, -1000.0], rtol=1e-15, atol=1e-300), (
        f"log_softmax([0, -1000]) should be [0, -1000], got {extreme.tolist()}. log(softmax(x)) gives "
        "log(0) = -inf for the second entry: subtract logsumexp instead of taking a log.")
