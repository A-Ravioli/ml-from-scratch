import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # matrix factorization · lab

    every experiment here runs **your** code from `exercise.py` on a world simulated by
    `make_ratings` (600 users, 500 films, true taste rank 6, about 18,000 ratings; see the
    lesson's "the data" box). it's synthetic because no ratings dataset is cached and nothing
    is downloaded, and that turns out to be a feature: the simulator knows every user's true
    taste for every film, so you can check things a real dataset can't tell you. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute.
    """)
    return


@app.cell
def _():
    world = ex.make_ratings(rng=0)
    train, test = ex.split_ratings(world.users, world.items, world.ratings, test_frac=0.2, rng=1)
    shape = dict(n_users=world.n_users, n_items=world.n_items)
    baseline = ex.BiasBaseline(lam=5.0).fit(*train, **shape)
    base_rmse = ex.rmse(baseline.predict(train[0], train[1]), train[2]), ex.rmse(baseline.predict(test[0], test[1]), test[2])
    _counts = np.bincount(world.users, minlength=world.n_users)
    _dist = np.bincount(world.ratings.astype(int), minlength=6)[1:] / len(world.ratings)
    mo.md(f"{len(world.ratings):,} ratings ({len(train[2]):,} train, {len(test[2]):,} test); ratings per user: median "
          f"{np.median(_counts):.0f}, max {_counts.max()}; stars 1–5: "
          + ", ".join(f"{p:.0%}" for p in _dist)
          + f". the bias baseline (μ + b_u + b_i, λ = 5) has test RMSE **{base_rmse[1]:.3f}**.")
    return base_rmse, baseline, shape, test, train, world


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: rank vs overfitting

    ALS with ranks from 1 to 96 and three regularization strengths, 10 iterations each.
    the true tastes have rank 6. half the users have rated 19 films or fewer.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "sketch test RMSE against rank for lam = 0.01 (almost no regularization), lam = 1 and lam = 10. where is "
        "each curve lowest? does the nearly unregularized model keep getting worse as the rank grows?"
    )
    p1
    return (p1,)


@app.cell
def _(base_rmse, p1, shape, test, train):
    ui.gate(p1)
    ranks = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    lams = [0.01, 1.0, 10.0]
    curves = {}
    for _lam in lams:
        _tr, _te = [], []
        for _k in ranks:
            _m = ex.ALS(rank=_k, lam=_lam, n_iters=10, rng=0).fit(*train, **shape)
            _tr.append(ex.rmse(_m.predict(train[0], train[1]), train[2]))
            _te.append(ex.rmse(_m.predict(test[0], test[1]), test[2]))
        curves[_lam] = (_tr, _te)
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for _lam, (_tr, _te) in curves.items():
        _ax[0].plot(ranks, _tr, "o-", ms=3, label=f"λ = {_lam:g}")
        _ax[1].plot(ranks, _te, "o-", ms=3, label=f"λ = {_lam:g}")
    for _a, _title in zip(_ax, ["train RMSE", "test RMSE"]):
        _a.axhline(base_rmse[1], color="k", ls=":", lw=1, label="bias baseline (test)")
        _a.axvline(6, color="gray", lw=0.8, ls="--")
        _a.set(xscale="log", xlabel="rank", title=_title)
    _ax[0].set(ylabel="RMSE", ylim=(0, 1.8))
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {_lam:g} | " + " | ".join(f"{v:.3f}" for v in curves[_lam][1]) + " |" for _lam in lams]
    mo.vstack([_fig, mo.md("test RMSE (dashed line: the true rank 6)\n\n| λ \\ rank | " + " | ".join(map(str, ranks))
                            + " |\n|---|" + "---|" * len(ranks) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does training RMSE keep falling while test RMSE doesn't? for
    $\lambda = 0.01$, test RMSE gets worse up to some rank and then gets *better* again.
    compare the rank where it peaks with the number of ratings a typical user or film has:
    what changes about each ridge regression when it has more unknowns than equations? why
    doesn't $\lambda = 10$ show the bump? use the slider to find the best $\lambda$ at rank 8.
    """)
    return


@app.cell
def _():
    lam_s = mo.ui.slider(0.0, 30.0, value=4.0, step=0.5, label="λ for ALS (rank 8)")
    lam_s
    return (lam_s,)


@app.cell
def _(lam_s, p1, shape, test, train):
    ui.gate(p1)
    _m = ex.ALS(rank=8, lam=max(lam_s.value, 1e-3), n_iters=10, rng=0).fit(*train, **shape)
    mo.md(f"rank 8, λ = {lam_s.value:g}: train RMSE **{ex.rmse(_m.predict(train[0], train[1]), train[2]):.3f}**, "
          f"test RMSE **{ex.rmse(_m.predict(test[0], test[1]), test[2]):.3f}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: ALS vs SGD

    the same model (rank 8) fitted two ways: ALS with $\lambda = 4$, and SGD with
    $\lambda = 0.1$ (the two $\lambda$s are on different scales, see common bugs) at two
    learning rates, 0.01 and 0.05, for 40 epochs. everything is tracked by test RMSE against
    wall-clock time. the SGD epochs are run with your `step` method, so each point costs one
    pass over the training ratings.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which reaches a test RMSE below 0.80 first in wall-clock time, ALS or SGD, and how many times faster? "
        "does SGD's test RMSE fall steadily from the first epoch?"
    )
    p2
    return (p2,)


@app.cell
def _(base_rmse, p2, shape, test, train):
    ui.gate(p2)

    def _als_curve(n_max=12):
        times, errs = [], []
        for _n in range(1, n_max + 1):  # same seed -> the n-iteration fit is a prefix of the (n+1)-iteration one
            _t0 = time.perf_counter()
            _m = ex.ALS(rank=8, lam=4.0, n_iters=_n, rng=0).fit(*train, **shape)
            times.append(time.perf_counter() - _t0)
            errs.append(ex.rmse(_m.predict(test[0], test[1]), test[2]))
        return np.array(times), np.array(errs)

    def _sgd_curve(lr, n_epochs=40):
        _m = ex.SGDMF(rank=8, lr=lr, lam=0.1, n_epochs=0, rng=0).fit(*train, **shape)
        _g = np.random.default_rng(0)
        times, errs, _t = [], [], 0.0
        for _ in range(n_epochs):
            _t0 = time.perf_counter()
            for _j in _g.permutation(len(train[2])):
                _m.step(train[0][_j], train[1][_j], train[2][_j])
            _t += time.perf_counter() - _t0
            times.append(_t)
            errs.append(ex.rmse(_m.predict(test[0], test[1]), test[2]))
        return np.array(times), np.array(errs)

    als_t, als_e = _als_curve()
    sgd = {_lr: _sgd_curve(_lr) for _lr in (0.01, 0.05)}
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(als_t, als_e, "o-", ms=3, label="ALS (λ = 4), one point per iteration")
    for _lr, (_t, _e) in sgd.items():
        _ax.plot(_t, _e, "s-", ms=2, label=f"SGD (lr {_lr:g}, λ = 0.1), one point per epoch")
    _ax.axhline(base_rmse[1], color="k", ls=":", lw=1, label="bias baseline")
    _ax.set(xlabel="seconds", ylabel="test RMSE", ylim=(0.7, 1.0))
    _ax.legend(fontsize=8)

    def _first(t, e, thr=0.80):
        _hit = np.flatnonzero(e < thr)
        return f"{t[_hit[0]]:.2f} s" if len(_hit) else "never"

    _lines = [f"ALS: best test RMSE **{als_e.min():.3f}** (iteration {als_e.argmin() + 1}), {als_t[0]:.2f} s for one "
              f"iteration, below 0.80 after {_first(als_t, als_e)}."]
    for _lr, (_t, _e) in sgd.items():
        _lines.append(f"SGD, lr {_lr:g}: best **{_e.min():.3f}** (epoch {_e.argmin() + 1}), {_t[0]:.2f} s per epoch, "
                      f"below 0.80 after {_first(_t, _e)}; test RMSE after epochs 1, 5, 10: "
                      f"{_e[0]:.3f}, {_e[4]:.3f}, {_e[9]:.3f}.")
    mo.vstack([_fig, mo.md("  \n".join(_lines))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does one ALS iteration make so much more progress than one SGD
    epoch? SGD first sits on a plateau close to the bias baseline: what are the factors
    doing there (look at their initial size, and at the gradient of $p_u \cdot q_i$ when
    both are small)? what does SGD buy you in exchange for being slower here (think about
    models that aren't a dot product, and data that arrive one rating at a time)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what the test set can't see

    take a well-tuned ALS model: rank 8, $\lambda = 4$ (test RMSE 0.755–0.761 across
    initialization seeds; experiment 1's rank 6, $\lambda = 1$ scored 0.738 with seed 0 but
    0.77–0.79 with other seeds). the simulator can also do what no real dataset can: show 20,000
    *uniformly random* user–film pairs to their users and record the ratings they'd give.
    that's what you'd see if the model were deployed and recommended films at random.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the model's RMSE on the held-out test ratings is about 0.76. what is its RMSE on the random pairs? "
        "does it over- or under-predict them, and by how much on average?"
    )
    p3
    return (p3,)


@app.cell
def _(baseline, p3, shape, test, train, world):
    ui.gate(p3)
    best = ex.ALS(rank=8, lam=4.0, n_iters=10, rng=0).fit(*train, **shape)
    ru, ri, rr = world.sample_random(20_000, rng=5)
    _rows = []
    for _name, _m in [("bias baseline", baseline), ("ALS rank 8, λ = 4", best)]:
        _pt, _pr = _m.predict(test[0], test[1]), _m.predict(ru, ri)
        _rows.append(f"| {_name} | {ex.rmse(_pt, test[2]):.3f} | {ex.rmse(_pr, rr):.3f} | {np.mean(_pr - rr):+.2f} |")
    _pop = world.popularity / world.popularity.sum()
    _q = np.quantile(_pop, [0.9])[0]
    _head = _pop[ri] >= _q
    _err = best.predict(ru, ri) - rr
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.8))
    _bins = np.arange(0.5, 6.0, 1.0)
    _ax[0].hist(world.ratings, bins=_bins, alpha=0.6, density=True, label="ratings people chose to give")
    _ax[0].hist(rr, bins=_bins, alpha=0.6, density=True, label="ratings of random pairs")
    _ax[0].set(xlabel="stars", ylabel="share", title="what the data shows vs the truth")
    _ax[0].legend(fontsize=8)
    _ax[1].scatter(best.predict(ru[:2000], ri[:2000]), world.preference[ru[:2000], ri[:2000]], s=3, alpha=0.4)
    _ax[1].plot([1, 5.5], [1, 5.5], "k--", lw=1)
    _ax[1].set(xlabel="predicted rating", ylabel="true preference", title="random pairs (2,000 shown)")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| model | RMSE on held-out ratings | RMSE on random pairs | mean (predicted − actual) on random pairs |\n"
        "|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n\nmean observed rating **{world.ratings.mean():.2f}**, mean rating of random pairs **{rr.mean():.2f}**. "
        f"ALS error on random pairs: head films (top 10% by popularity) {np.sqrt(np.mean(_err[_head] ** 2)):.3f}, "
        f"the rest {np.sqrt(np.mean(_err[~_head] ** 2)):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the test set was a random 20% of the *observed* ratings, so why isn't
    it a fair sample of what the model will face? where does the systematic over-prediction
    come from (think about who chose to rate what)? would more data of the same kind fix
    it? what could you change about how data are collected?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: cold start

    100 users are removed from training entirely. the model (rank 8, $\lambda = 4$) is
    trained on the rest; then each removed user reveals $n$ of their ratings, gets folded in
    with your `fold_in`, and is scored on the ratings they didn't reveal. two fold-in
    strengths: the model's own $\lambda = 4$ and a weak $\lambda = 0.1$.
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "with n = 1, 2, 3 revealed ratings, is folding in better or worse than predicting mu + b_i (n = 0)? "
        "how many ratings until the new user is about as well served as a training user?"
    )
    p4
    return (p4,)


@app.cell
def _(p4, world):
    ui.gate(p4)
    _g = np.random.default_rng(3)
    _counts = np.bincount(world.users, minlength=world.n_users)
    _cold = _g.choice(np.flatnonzero(_counts >= 30), size=100, replace=False)
    _is_cold = np.isin(world.users, _cold)
    _tr = (world.users[~_is_cold], world.items[~_is_cold], world.ratings[~_is_cold])
    _model = ex.ALS(rank=8, lam=4.0, n_iters=10, rng=0).fit(*_tr, n_users=world.n_users, n_items=world.n_items)
    ns = [0, 1, 2, 3, 5, 8, 12, 20]
    cold = {4.0: [], 0.1: []}
    for _lam in cold:
        for _n in ns:
            _err = []
            for _u in _cold:
                _idx = _g.permutation(np.flatnonzero(world.users == _u))
                _shown, _hidden = _idx[:_n], _idx[20:]
                _p, _b = ex.fold_in(_model, world.items[_shown], world.ratings[_shown], lam=_lam)
                _err.append(ex.predict_new_user(_model, _p, _b, world.items[_hidden]) - world.ratings[_hidden])
            cold[_lam].append(float(np.sqrt(np.mean(np.concatenate(_err) ** 2))))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _lam, _v in cold.items():
        _ax.plot(ns, _v, "o-", label=f"fold-in λ = {_lam:g}")
    _ax.set(xlabel="ratings revealed by the new user", ylabel="RMSE on their other ratings", ylim=(0.7, 1.3))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| n revealed | " + " | ".join(map(str, ns)) + " |\n|---|" + "---|" * len(ns) + "\n"
        + "\n".join(f"| λ = {_lam:g} | " + " | ".join(f"{v:.3f}" for v in _v) + " |" for _lam, _v in cold.items())
        + "\n\n(for scale: the same model's RMSE on held-out ratings of users it was trained on is about 0.76, "
        "experiment 1.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** fold-in solves a ridge regression with rank + 1 = 9 unknowns. what does
    that regression do with one equation and almost no regularization? why does $\lambda$
    matter so much more for a new user than for the training users? what would you show a
    brand-new user before they've rated anything?
    """)
    return


if __name__ == "__main__":
    app.run()
