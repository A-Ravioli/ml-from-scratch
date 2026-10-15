"""Checks for capstone.py (Bayesian optimization).

All objectives here are synthetic and cheap. The GP and expected improvement come from
mlfs.kernels (the reference solutions under pytest); these checks test your proposal step,
your search loops, your cross-validated objective and your comparison harness, plus minimum
optimization quality on a standard test function. Every search runs on an objective that
counts its calls and refuses to be called more often than the budget allows, so a loop that
never stops fails instead of hanging.
"""
import numpy as np
import pytest

from mlfs.testing import bounded

BRANIN_BOX = np.array([[-5.0, 10.0], [0.0, 15.0]])
BRANIN_MAX = -0.397887


def branin(x):
    """the (negated) Branin function: three global maxima of -0.397887 in the box above."""
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return -((x[1] - b * x[0] ** 2 + c * x[0] - 6) ** 2 + 10 * (1 - t) * np.cos(x[0]) + 10)


class Counted:
    """wraps an objective: records every x, and raises once called more than `limit` times."""

    def __init__(self, fn, limit):
        self.fn, self.limit, self.xs = fn, limit, []

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if len(self.xs) >= self.limit:
            raise AssertionError(f"the objective was called more than {self.limit} times: the budget is the total number of "
                                 "evaluations (n_init + n_iter for bayes_opt, n_iter for random_search)")
        assert x.ndim == 1, f"the objective takes one point, a 1-D array (d,), got shape {x.shape}"
        self.xs.append(x.copy())
        return float(self.fn(x))


def _gp(X, y):
    from mlfs.kernels import GaussianProcessRegressor
    return GaussianProcessRegressor(kernel="matern52", lengthscale=0.3, variance=1.0, noise=1e-6, optimize=False).fit(X, y)


# ---------------------------------------------------------------- stage 1: propose


def test_stage1_picks_the_max_ei(nb):
    from mlfs.kernels import expected_improvement
    X = np.array([[0.1], [0.4], [0.5], [0.9]])
    y = np.array([0.2, 0.8, 0.7, -0.1])
    gp = _gp(X, y)
    cand = np.linspace(0, 1, 201)[:, None]
    i, ei = nb.propose(gp, cand, 0.8, xi=0.01)
    mu, sd = gp.predict(cand, return_std=True)
    want = expected_improvement(mu, sd, 0.8, xi=0.01)
    assert np.shape(ei) == (201,) and np.allclose(ei, want), "ei holds expected_improvement(mean, std, best, xi) per candidate"
    assert i == int(np.argmax(want)) and isinstance(i, int), f"the index of the largest EI ({int(np.argmax(want))}), got {i}"


def test_stage1_ties_and_xi(nb):
    gp = _gp(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    i, _ = nb.propose(gp, np.array([[0.5], [0.5]]), 1.0)
    assert i == 0, f"two identical candidates tie: the first one wins (np.argmax), got index {i}"
    far = np.array([[0.5], [5.0]])
    _, ei_small = nb.propose(gp, far, 1.0, xi=0.0)
    _, ei_big = nb.propose(gp, far, 1.0, xi=0.5)
    assert np.all(ei_big <= ei_small + 1e-12), "a larger xi demands more improvement, so EI can only shrink"


# ---------------------------------------------------------------- stage 2: Bayesian optimization


def test_stage2_bookkeeping(nb):
    f = Counted(branin, 12)
    out = bounded(nb.bayes_opt, f, BRANIN_BOX, n_init=4, n_iter=8, rng=0, budget=20_000_000, what="bayes_opt")
    assert set(out) >= {"X", "y", "best_x", "best_y", "trace"}, f"bayes_opt returns a dict with X, y, best_x, best_y, trace; got {sorted(out)}"
    X, y = np.asarray(out["X"]), np.asarray(out["y"])
    assert X.shape == (12, 2) and y.shape == (12,), f"n_init + n_iter = 12 evaluations: X (12, 2), y (12,); got {X.shape}, {y.shape}"
    assert len(f.xs) == 12 and np.allclose(np.array(f.xs), X), "X lists the evaluated points in order (in the box's units)"
    assert np.allclose(y, [branin(x) for x in X]), "y[i] = objective(X[i])"
    assert np.all(X >= BRANIN_BOX[:, 0]) and np.all(X <= BRANIN_BOX[:, 1]), "every evaluated point lies inside the box"
    assert np.allclose(out["trace"], np.maximum.accumulate(y)), "trace[i] = the best value among the first i + 1 evaluations"
    assert np.isclose(out["best_y"], y.max()) and np.allclose(out["best_x"], X[np.argmax(y)]), "best_x, best_y: the best evaluation"
    again = nb.bayes_opt(Counted(branin, 12), BRANIN_BOX, n_init=4, n_iter=8, rng=0)
    assert np.allclose(again["X"], X), "the same rng gives the same search"


def test_stage2_finds_the_global_max_in_1d(nb):
    f1 = lambda x: float(0.8 * np.exp(-(x[0] - 1) ** 2 / 0.5) + 1.2 * np.exp(-(x[0] - 3.7) ** 2 / 0.2)  # noqa: E731
                         + 0.2 * np.sin(5 * x[0]))
    grid = np.linspace(0, 5, 20001)
    top = max(f1([g]) for g in grid)
    hits = 0
    for seed in range(3):
        out = bounded(nb.bayes_opt, Counted(f1, 16), np.array([[0.0, 5.0]]), n_init=4, n_iter=12, rng=seed,
                      budget=20_000_000, what="bayes_opt")
        hits += top - out["best_y"] < 0.01
    assert hits >= 2, (
        f"on a 1-D function with several local maxima, 16 evaluations should get within 0.01 of the global maximum "
        f"({top:.4f}) in at least 2 of 3 runs; got {hits}. does the GP see every evaluation so far, and do you fit it on "
        "the unit cube?")


def test_stage2_beats_random_on_branin(nb):
    bo = [bounded(nb.bayes_opt, Counted(branin, 25), BRANIN_BOX, n_init=5, n_iter=20, rng=seed, budget=40_000_000,
                  what="bayes_opt")["best_y"] for seed in range(5)]
    g = np.random.default_rng(100)
    rs = [max(branin(BRANIN_BOX[:, 0] + g.random(2) * 15.0) for _ in range(25)) for _ in range(50)]
    regret_bo, regret_rs = BRANIN_MAX - np.median(bo), BRANIN_MAX - np.median(rs)
    assert regret_bo < 0.6 and regret_bo < regret_rs, (
        f"with 25 evaluations, Bayesian optimization should get within 0.6 of Branin's maximum (median regret over 5 "
        f"runs {regret_bo:.3f}) and beat 25 random evaluations (median regret over 50 runs {regret_rs:.3f})")


# ---------------------------------------------------------------- stage 3: random search


def test_stage3_random_search(nb):
    box = np.array([[-6.0, 2.0], [10.0, 11.0], [0.0, 1.0]])
    f = Counted(lambda x: -np.sum(x**2), 500)
    out = bounded(nb.random_search, f, box, n_iter=500, rng=0, budget=2_000_000, what="random_search")
    X = np.asarray(out["X"])
    assert X.shape == (500, 3) and len(f.xs) == 500, "n_iter evaluations, X (n_iter, d)"
    assert np.all(X >= box[:, 0]) and np.all(X <= box[:, 1]), "points lie inside the box"
    assert np.allclose(X.mean(axis=0), box.mean(axis=1), atol=0.4) and np.allclose(X.std(axis=0), np.ptp(box, axis=1) / np.sqrt(12), rtol=0.2), (
        "points are uniform in the box: each coordinate has the mean and standard deviation of a uniform distribution")
    assert np.allclose(out["trace"], np.maximum.accumulate(out["y"])) and np.isclose(out["best_y"], np.max(out["y"]))
    assert np.allclose(nb.random_search(Counted(f.fn, 5), box, n_iter=5, rng=3)["X"],
                       nb.random_search(Counted(f.fn, 5), box, n_iter=5, rng=3)["X"]), "the same rng gives the same points"


# ---------------------------------------------------------------- stage 4: the objective


def _toy_regression(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    return X, np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 5.0 + 0.1 * rng.normal(size=n)


def test_stage4_matches_manual_cv(nb):
    from mlfs.kernels import KernelRidge
    from mlfs.model_selection import KFold
    X, y = _toy_regression()
    f = nb.krr_objective(X, y, cv=4, rng=7)
    assert callable(f), "krr_objective returns a function of the hyperparameters"
    params = np.array([-1.0, -0.3])
    sq = 0.0
    for tr, te in KFold(4, shuffle=True, rng=np.random.default_rng(7)).split(X):
        m = KernelRidge(0.1, gamma=10**-0.3).fit(X[tr], y[tr] - y[tr].mean())
        sq += np.sum((m.predict(X[te]) + y[tr].mean() - y[te]) ** 2)
    want = -np.sqrt(sq / len(y))
    got = f(params)
    assert np.isclose(got, want), (
        f"f([log10 alpha, log10 gamma]) should be minus the cross-validated RMSE ({want:.6f}), got {got}. centre y on each "
        "training fold (kernel ridge has no intercept) and pool the squared errors of all folds before the square root")
    assert f(params) == got, "the folds are fixed when the objective is created: the same params give the same value"


def test_stage4_rewards_good_hyperparameters(nb):
    X, y = _toy_regression(seed=1)
    f = nb.krr_objective(X, y, cv=5, rng=0)
    good, too_smooth, too_rough = f([-2.0, -0.5]), f([1.5, -3.0]), f([-6.0, 2.0])
    assert good > too_smooth and good > too_rough, (
        f"a sensible setting should beat an over-regularized one and an overfitting one: {good:.3f} vs {too_smooth:.3f} "
        f"and {too_rough:.3f}")
    assert -0.3 < good < 0, f"on this smooth toy problem the CV RMSE should be small (noise sd 0.1), got {-good:.3f}"


# ---------------------------------------------------------------- stage 5: the comparison


def test_stage5_compare_searches(nb):
    calls = Counted(branin, 3 * 2 * 10)
    out = bounded(lambda: nb.compare_searches(calls, BRANIN_BOX, budget=10, n_init=4, n_seeds=3, rng=0),
                  what="compare_searches", budget=60_000_000)
    for key in ("bo", "random"):
        t = np.asarray(out[key])
        assert t.shape == (3, 10), f"out[{key!r}] is an (n_seeds, budget) = (3, 10) array of traces, got {t.shape}"
        assert np.all(np.diff(t, axis=1) >= 0), f"each {key} trace is a running maximum (never decreases)"
    assert len(out["bo_runs"]) == 3 and len(out["random_runs"]) == 3, "bo_runs and random_runs keep each run's dict"
    assert np.allclose(out["bo"][0], out["bo_runs"][0]["trace"]), "out['bo'][s] is the trace of bo_runs[s]"
    assert len(calls.xs) == 60, f"3 seeds x 2 searches x 10 evaluations = 60 objective calls, got {len(calls.xs)}"
