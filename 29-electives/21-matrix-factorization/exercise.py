import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # matrix factorization · build

    this block is spec-driven (L4): each stage's exact specification is in the lesson's
    **spec** box next to it, the checks below enforce it, and how you structure the code is up
    to you. by the end you can predict a user's rating of a film they haven't seen, two ways,
    and fold a brand-new user into a trained model from a handful of ratings.

    | stage | you build |
    |---|---|
    | 1 | `split_ratings`, `rmse`, `BiasBaseline`: an honest split, the metric, and the baseline to beat |
    | 2 | `ALS`: biased matrix factorization by alternating ridge regressions |
    | 3 | `SGDMF`: the same model fitted by stochastic gradient descent |
    | 4 | `fold_in`, `predict_new_user`: cold start for a new user without retraining |

    there's no MovieLens on this machine and nothing is downloaded: the data comes from
    `make_ratings`, a simulator (given, below) with latent tastes, power-law popularity, and
    people who mostly rate what they chose to watch. `load_movielens` reads the real thing if
    you have a `u.data` file. you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.function
def make_ratings(n_users=600, n_items=500, rank=6, *, mean_ratings=30, popularity_exponent=0.8,
                 selection_strength=1.0, noise=0.5, rng=None):
    """given: a synthetic explicit-ratings world (see the lesson's "the data" box).

    every user u and item i has a true preference on the 1-5 scale,
        pref[u, i] = 3 + user_bias[u] + item_bias[i] + U[u] . V[i]     (U, V: rank-`rank` tastes),
    item i has a power-law exposure weight pop[i] = (1 + its popularity rank) ** -popularity_exponent,
    and user u rates n_u items (lognormal, mean about `mean_ratings`), drawn without replacement with
    probability proportional to pop[i] * exp(selection_strength * pref[u, i]): popular items and items
    the user likes are far more likely to be rated (missing NOT at random). observed rating =
    pref + N(0, noise^2), rounded and clipped to 1..5.

    returns a SimpleNamespace with users, items (int64 (n,)), ratings (float (n,)), n_users, n_items,
    preference (n_users, n_items) the noise-free pref, popularity (n_items,) the exposure weights, and
    sample_random(n, rng=None) -> (users, items, ratings) for n uniformly random user-item pairs rated
    with the same noise: what a randomized experiment would observe.
    """
    rng = as_rng(rng)
    scale = (0.8 / rank) ** 0.25  # so that U[u] . V[i] has standard deviation about 0.9
    U = rng.normal(0.0, scale, (n_users, rank))
    V = rng.normal(0.0, scale, (n_items, rank))
    log_pop = -popularity_exponent * np.log1p(rng.permutation(n_items))
    item_bias = 0.25 * (log_pop - log_pop.mean()) / log_pop.std() + rng.normal(0.0, 0.3, n_items)
    user_bias = rng.normal(0.0, 0.4, n_users)
    pref = 3.0 + user_bias[:, None] + item_bias[None, :] + U @ V.T
    n_per = np.clip(np.round(rng.lognormal(np.log(mean_ratings) - 0.5, 1.0, n_users)), 5, n_items // 2).astype(int)
    logits = log_pop[None, :] + selection_strength * (pref - pref.mean(axis=1, keepdims=True))
    keys = logits + rng.gumbel(size=logits.shape)  # Gumbel top-k = sampling without replacement
    order = np.argsort(-keys, axis=1)
    users = np.concatenate([np.full(n, u) for u, n in enumerate(n_per)]).astype(np.int64)
    items = np.concatenate([order[u, :n] for u, n in enumerate(n_per)]).astype(np.int64)

    def rate(u, i, g):
        return np.clip(np.round(pref[u, i] + g.normal(0.0, noise, len(u))), 1, 5)

    def sample_random(n, rng=None):
        g = as_rng(rng)
        u, i = g.integers(n_users, size=n), g.integers(n_items, size=n)
        return u, i, rate(u, i, g)

    return SimpleNamespace(users=users, items=items, ratings=rate(users, items, rng), n_users=n_users,
                           n_items=n_items, preference=pref, popularity=np.exp(log_pop), sample_random=sample_random)


@app.function
def load_movielens(path):
    """given: read a MovieLens-100K style `u.data` file (tab-separated user, item, rating, timestamp; ids from 1).

    returns (users, items, ratings) with 0-based int64 ids and float ratings. not used by the checks or
    the lab (nothing is downloaded); use it if you have the file.
    """
    raw = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2), ndmin=2)
    return raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 2].astype(float)


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def split_ratings(users, items, ratings, *, test_frac=0.2, rng=None):
    """stage 1: a random train/test split of rating triples: ((u, i, r) train, (u, i, r) test)."""
    raise NotImplementedError("stage1: implement split_ratings")


@app.function
def rmse(pred, target):
    """stage 1: root mean squared error between two equal-length arrays, a float."""
    raise NotImplementedError("stage1: implement rmse")


@app.class_definition
class BiasBaseline:
    """r_hat = mu + b_u + b_i, with ridge-regularized biases fitted by alternating closed-form updates."""

    def __init__(self, lam=5.0, *, n_iters=10):
        """stage 1: store lam and n_iters."""
        raise NotImplementedError("stage1: implement BiasBaseline.__init__")

    def fit(self, users, items, ratings, n_users=None, n_items=None):
        """stage 1: mu_ = mean rating; then n_iters sweeps of item biases, then user biases (0 for unseen ids, even with lam=0); returns self."""
        raise NotImplementedError("stage1: implement BiasBaseline.fit")

    def predict(self, users, items):
        """stage 1: mu_ + user_bias_[users] + item_bias_[items], float (n,)."""
        raise NotImplementedError("stage1: implement BiasBaseline.predict")


@app.cell(hide_code=True)
def _():
    checks.run(1, split_ratings=split_ratings, rmse=rmse, BiasBaseline=BiasBaseline)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ALS:
    """biased MF r_hat = mu + b_u + b_i + p_u . q_i, fitted by alternating ridge regressions."""

    def __init__(self, rank=10, *, lam=1.0, n_iters=10, rng=None):
        """stage 2: store rank, lam, n_iters, rng."""
        raise NotImplementedError("stage2: implement ALS.__init__")

    def fit(self, users, items, ratings, n_users=None, n_items=None):
        """stage 2: init factors N(0, 0.1^2) (users, then items), biases 0; n_iters sweeps (items, then users); returns self."""
        raise NotImplementedError("stage2: implement ALS.fit")

    def predict(self, users, items):
        """stage 2: mu_ + b_u + b_i + p_u . q_i for each pair, float (n,)."""
        raise NotImplementedError("stage2: implement ALS.predict")

    def loss(self, users, items, ratings):
        """stage 2: sum of squared errors + lam * (||P||^2 + ||Q||^2 + ||b_user||^2 + ||b_item||^2), a float."""
        raise NotImplementedError("stage2: implement ALS.loss")


@app.cell(hide_code=True)
def _():
    checks.run(2, ALS=ALS)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SGDMF:
    """the same biased MF model, fitted by stochastic gradient descent over shuffled ratings (Funk / Koren)."""

    def __init__(self, rank=10, *, lr=0.01, lam=0.05, n_epochs=20, rng=None):
        """stage 3: store rank, lr, lam, n_epochs, rng."""
        raise NotImplementedError("stage3: implement SGDMF.__init__")

    def fit(self, users, items, ratings, n_users=None, n_items=None):
        """stage 3: init as ALS; each epoch visits the ratings in rng.permutation order calling step; returns self."""
        raise NotImplementedError("stage3: implement SGDMF.fit")

    def step(self, u, i, r):
        """stage 3: one SGD update on rating r of item i by user u (all from the old values); returns the error r - r_hat."""
        raise NotImplementedError("stage3: implement SGDMF.step")

    def predict(self, users, items):
        """stage 3: mu_ + b_u + b_i + p_u . q_i, float (n,)."""
        raise NotImplementedError("stage3: implement SGDMF.predict")


@app.cell(hide_code=True)
def _():
    checks.run(3, SGDMF=SGDMF, rmse=rmse)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fold_in(model, items, ratings, *, lam=None):
    """stage 4: a new user's (factors (rank,), bias float) by ridge regression on the model's item factors."""
    raise NotImplementedError("stage4: implement fold_in")


@app.function
def predict_new_user(model, factors, bias, items):
    """stage 4: mu_ + bias + item_bias_[items] + item_factors_[items] @ factors, float (n,)."""
    raise NotImplementedError("stage4: implement predict_new_user")


@app.cell(hide_code=True)
def _():
    checks.run(4, fold_in=fold_in, predict_new_user=predict_new_user, ALS=ALS, SGDMF=SGDMF)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it runs your `ALS`,
    `SGDMF` and `fold_in` on the simulated world and asks what rank you should use, and how
    much a low test RMSE really tells you.
    """)
    return


if __name__ == "__main__":
    app.run()
