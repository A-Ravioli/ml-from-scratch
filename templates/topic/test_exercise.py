"""Checks for exercise.py.

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook. Failure messages should say what was expected,
what came back, and the most likely cause.
"""
# TEMPLATE: replace these example checks with this topic's checks.
import numpy as np


def test_stage1_worked_example(nb):
    got = nb.mean(np.array([1.0, 2.0, 3.0, 6.0]))
    assert np.isclose(got, 3.0), f"mean of (1, 2, 3, 6) should be 3.0, got {got}"


def test_stage1_matches_numpy(nb):
    x = np.random.default_rng(0).normal(size=1000)
    got = nb.mean(x)
    assert np.isclose(got, x.mean()), f"expected {x.mean():.6f}, got {got:.6f}"


def test_stage2_worked_example(nb):
    got = nb.variance(np.array([1.0, 2.0, 3.0, 6.0]))
    assert np.isclose(got, 3.5), (
        f"variance of (1, 2, 3, 6) should be 3.5, got {got}. "
        "if you got 4.67, you divided by n-1 instead of n."
    )


def test_stage2_shift_invariant(nb):
    x = np.random.default_rng(1).normal(size=200)
    a, b = nb.variance(x), nb.variance(x + 1e3)
    assert np.isclose(a, b, rtol=1e-6), (
        f"adding a constant shouldn't change the variance: {a:.6f} vs {b:.6f}"
    )
