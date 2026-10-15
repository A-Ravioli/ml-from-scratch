import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # implicit feedback · lab

    every experiment here runs **your** metrics and models from `exercise.py` on a world
    simulated by `make_interactions`: 800 users, 600 items, about 23,000 interactions with
    play counts, consumption driven by popularity and taste (see the lesson's "the data"
    box). the simulator also knows every user's true taste for every item, so besides the
    usual held-out evaluation you can ask what a user would *actually* like. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. the whole lab takes about half a minute.
    """)
    return


@app.cell
def _():
    world = ex.make_interactions(rng=0)
    train, test = ex.split_interactions(world.users, world.items, world.counts, world.n_users, world.n_items, rng=1)
    seen = train > 0
    # the truth: each user's 20 favourite items among those they haven't consumed in training
    truth = np.zeros_like(seen)
    np.put_along_axis(truth, np.argsort(-np.where(seen, -np.inf, world.preference), axis=1)[:, :20], True, axis=1)
    head = np.argsort(-seen.sum(axis=0))[: world.n_items // 10]  # the 10% most consumed items
    mo.md(f"{len(world.counts):,} interactions ({int(seen.sum()):,} train, {int((test > 0).sum()):,} test), "
          f"{len(world.counts) / (world.n_users * world.n_items):.1%} of all cells; the top 10% of items account for "
          f"{seen[:, head].sum() / seen.sum():.0%} of training interactions.")
    return head, seen, test, train, truth, world


@app.cell
def _(seen, train):
    def observed_mf(R, rank=8, lam=5.0, n_iters=10, seed=0):
        """an explicit-style model (topic 21's biased ALS) fitted to log(1 + plays) on the OBSERVED cells only."""
        g = np.random.default_rng(seed)
        u, i = np.nonzero(R)
        y = np.log1p(R[u, i])
        n_users, n_items = R.shape
        mu = y.mean()
        P, Q = g.normal(0, 0.1, (n_users, rank)), g.normal(0, 0.1, (n_items, rank))
        bu, bi = np.zeros(n_users), np.zeros(n_items)
        for _ in range(n_iters):
            for rows, other, F, B, OF, OB in ((i, u, Q, bi, P, bu), (u, i, P, bu, Q, bi)):
                for a in range(len(F)):
                    idx = np.flatnonzero(rows == a)
                    X = np.c_[OF[other[idx]], np.ones(len(idx))]
                    w = np.linalg.solve(X.T @ X + lam * np.eye(rank + 1), X.T @ (y[idx] - mu - OB[other[idx]]))
                    F[a], B[a] = w[:-1], w[-1]
        return mu + bu[:, None] + bi[None, :] + P @ Q.T

    models = {
        "random": np.random.default_rng(0).random(train.shape),
        "popularity": ex.popularity_scores(train),
        "observed-only MF": observed_mf(train),
        "implicit ALS (α = 1)": ex.ImplicitALS(rank=16, alpha=1.0, lam=1.0, n_iters=10, rng=0).fit(train).scores(),
        "BPR (30 epochs)": ex.BPR(rank=16, lr=0.05, lam=0.01, n_epochs=30, rng=0).fit(train).scores(),
    }
    tops = {_name: ex.top_k(_s, 10, exclude=seen) for _name, _s in models.items()}
    return models, tops


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the best RMSE, the worst list

    four models and a random scorer. one of them is the explicit-feedback approach of topic
    21 applied naively: biased ALS fitted to $\log(1 + \text{plays})$ on the observed cells,
    exactly as you'd fit star ratings. it's scored on what it was trained for (RMSE on the
    held-out play counts) and on what a recommender is for (how many held-out items land in
    each user's top 10, with already-consumed items excluded).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the observed-only model has by far the lowest RMSE on held-out play counts. rank the five scorers by "
        "recall@10. where does it land relative to plain popularity? and what 'accuracy' does a model that "
        "predicts 'no interaction' for every unseen cell get?"
    )
    p1
    return (p1,)


@app.cell
def _(models, p1, seen, test, train):
    ui.gate(p1)
    _tu, _ti = np.nonzero(test)
    _rows = []
    for _name, _s in models.items():
        _ev = ex.evaluate_topk(_s, train, test, k=10)
        _rmse = np.sqrt(np.mean((_s[_tu, _ti] - np.log1p(test[_tu, _ti])) ** 2)) if _name == "observed-only MF" else np.nan
        _rows.append(f"| {_name} | {'—' if np.isnan(_rmse) else f'{_rmse:.3f}'} | {_ev['recall']:.3f} | {_ev['ndcg']:.3f} |")
    _acc = 1 - (test > 0)[~seen].mean()
    mo.md("| scorer | RMSE on held-out log plays | recall@10 | NDCG@10 |\n|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\npredicting 'no interaction' for every unseen cell is right **{_acc:.1%}** of the time (and its recall@10 is 0).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the observed-only model predicts *how much* a user will play an item
    very well, given that they play it. why is that the wrong question for choosing ten items
    out of 600? which items does it put at the top of a user's list (look at what it has
    never been shown)? why is 99% accuracy meaningless here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: held-out items vs the truth

    the standard offline protocol (experiment 1) counts a recommendation as good if the user
    happened to consume it in the held-out 20%. the simulator can also score each list
    against the user's **20 truly favourite** unconsumed items, by their real taste, whether
    or not they ever came across them. the last column is the share of recommended items
    that come from the 10% most popular.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "popularity scores nearly as well as BPR on held-out items. how does it do against the users' true "
        "favourites? does the ranking of the five scorers change between the two protocols?"
    )
    p2
    return (p2,)


@app.cell
def _(head, models, p2, test, tops, train, truth):
    ui.gate(p2)
    _rows, _std, _unb = [], [], []
    for _name, _s in models.items():
        _ev = ex.evaluate_topk(_s, train, test, k=10)
        _r = float(np.nanmean(ex.recall_at_k(tops[_name], truth)))
        _n = float(np.nanmean(ex.ndcg_at_k(tops[_name], truth)))
        _share = np.isin(tops[_name], head).mean()
        _std.append(_ev["ndcg"])
        _unb.append(_n)
        _rows.append(f"| {_name} | {_ev['recall']:.3f} | {_ev['ndcg']:.3f} | {_r:.3f} | {_n:.3f} | {_share:.0%} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
    _ax.scatter(_std, _unb)
    for _name, _x, _y in zip(models, _std, _unb):
        _ax.annotate(_name, (_x, _y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    _ax.plot([0, 0.55], [0, 0.55], "k:", lw=1)
    _ax.set(xlabel="NDCG@10 against held-out items (standard protocol)", ylabel="NDCG@10 against true favourites",
            xlim=(0, 0.55), ylim=(0, 0.55))
    _ax.set_aspect("equal")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| scorer | held-out recall@10 | held-out NDCG@10 | true-favourite recall@10 | true-favourite NDCG@10 | from top-10% items |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the standard protocol reward popularity so generously? (what
    decided which items ended up in the held-out set, and what does popularity have to do
    with it?) why does the observed-only model look better against the truth than against
    the held-out items? which protocol would you trust to decide what to ship, and what
    could you do when you don't have a simulator?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: AUC vs the top of the list

    BPR maximizes a smooth version of AUC. here you train it for 100 epochs (with your
    `step`, replicating `fit`) and track three numbers after each of several epochs: mean
    AUC on held-out items, NDCG@10 on held-out items, and NDCG@10 against the true
    favourites.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "between epoch 30 and epoch 100, how do AUC, held-out NDCG@10 and true-favourite NDCG@10 change? do they "
        "move together? and what is BPR recommending after only 5 epochs?"
    )
    p3
    return (p3,)


@app.cell
def _(head, p3, seen, test, train, truth):
    ui.gate(p3)
    _m = ex.BPR(rank=16, lr=0.05, lam=0.01, n_epochs=0, rng=0).fit(train)
    _g = np.random.default_rng(0)
    _pu, _pi = np.nonzero(train > 0)
    checkpoints = [1, 3, 5, 10, 20, 30, 50, 75, 100]
    track = {"AUC (held-out)": [], "NDCG@10 (held-out)": [], "NDCG@10 (true favourites)": [], "share from top-10%": []}
    for _epoch in range(1, 101):
        _perm = _g.permutation(len(_pu))
        for _s in range(0, len(_perm), _m.batch_size):
            _b = _perm[_s:_s + _m.batch_size]
            _m.step(_pu[_b], _pi[_b], _g.integers(train.shape[1], size=len(_b)))
        if _epoch in checkpoints:
            _sc = _m.scores()
            _top = ex.top_k(_sc, 10, exclude=seen)
            track["AUC (held-out)"].append(float(np.nanmean(ex.user_auc(_sc, test > 0, exclude=seen))))
            track["NDCG@10 (held-out)"].append(ex.evaluate_topk(_sc, train, test)["ndcg"])
            track["NDCG@10 (true favourites)"].append(float(np.nanmean(ex.ndcg_at_k(_top, truth))))
            track["share from top-10%"].append(float(np.isin(_top, head).mean()))
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax[0].plot(checkpoints, track["AUC (held-out)"], "o-", color="k")
    _ax[0].set(xscale="log", xlabel="epoch", ylabel="mean AUC", title="AUC (held-out items)")
    for _k in ("NDCG@10 (held-out)", "NDCG@10 (true favourites)"):
        _ax[1].plot(checkpoints, track[_k], "o-", label=_k)
    _ax[1].set(xscale="log", xlabel="epoch", ylabel="NDCG@10", title="the top of the list")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| epoch | " + " | ".join(map(str, checkpoints)) + " |\n|---|" + "---|" * len(checkpoints) + "\n"
        + "\n".join(f"| {_k} | " + " | ".join(f"{v:.3f}" if _k != "share from top-10%" else f"{v:.0%}" for v in _v) + " |"
                    for _k, _v in track.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** AUC averages over *every* (relevant, irrelevant) pair, wherever they sit
    in the list. why can it stay flat while the top 10 changes completely? what does BPR
    learn first, and why (think about which parameter gets a gradient from every single
    triple)? when the two NDCG curves move in opposite directions, which one is telling the
    truth, and what is the other one rewarding? try the confidence weight of implicit ALS
    below.
    """)
    return


@app.cell
def _():
    alpha_dd = mo.ui.dropdown(["0.1", "1", "3", "10", "40", "100"], value="40", label="implicit ALS confidence α")
    alpha_dd
    return (alpha_dd,)


@app.cell
def _(alpha_dd, p3, seen, test, train, truth):
    ui.gate(p3)
    _sc = ex.ImplicitALS(rank=16, alpha=float(alpha_dd.value), lam=1.0, n_iters=10, rng=0).fit(train).scores()
    _ev = ex.evaluate_topk(_sc, train, test)
    _top = ex.top_k(_sc, 10, exclude=seen)
    mo.md(f"implicit ALS, α = {alpha_dd.value}: AUC **{np.nanmean(ex.user_auc(_sc, test > 0, exclude=seen)):.3f}**, "
          f"held-out NDCG@10 **{_ev['ndcg']:.3f}**, true-favourite NDCG@10 **{np.nanmean(ex.ndcg_at_k(_top, truth)):.3f}**, "
          f"coverage {_ev['coverage']:.0%}.")
    return


if __name__ == "__main__":
    app.run()
