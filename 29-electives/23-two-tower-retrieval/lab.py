import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03

    torch.set_num_threads(4)
    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # two-tower retrieval · lab

    every experiment here runs **your** towers, loss, index and ranker from `exercise.py` on
    the catalogue simulated by `make_catalog`: 1,500 users, 3,000 items, about 38,000
    interactions, with noisy taste features on both sides plus a user budget and an item
    price (see the lesson's "the data" box). 10% of the items are **cold**: all their
    interactions are removed before training, as if they'd just been added. the simulator
    knows every user's true taste, so lists are scored both against held-out interactions
    and against each user's 20 truly favourite (warm, unseen) items. the last experiment is
    the block's project. the whole lab takes one to two minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    world = ex.make_catalog(rng=0)
    _g = np.random.default_rng(1)
    is_cold = np.zeros(world.n_items, dtype=bool)
    is_cold[_g.choice(world.n_items, size=world.n_items // 10, replace=False)] = True
    _test = _g.random(len(world.users)) < 0.2
    _warm = ~is_cold[world.items]
    tr_u, tr_i = world.users[_warm & ~_test], world.items[_warm & ~_test]
    seen = np.zeros((world.n_users, world.n_items), dtype=bool)
    seen[tr_u, tr_i] = True
    heldout = np.zeros_like(seen)
    heldout[world.users[_warm & _test], world.items[_warm & _test]] = True
    cold_truth = np.zeros_like(seen)  # every interaction with a cold item (none were trained on)
    cold_truth[world.users[~_warm], world.items[~_warm]] = True
    exclude = seen | is_cold[None, :]  # warm recommendations: never a seen item or a cold one
    truth = np.zeros_like(seen)
    np.put_along_axis(truth, np.argsort(-np.where(exclude, -np.inf, world.preference), axis=1)[:, :20], True, axis=1)
    counts = np.bincount(tr_i, minlength=world.n_items)
    head = np.argsort(-counts)[: world.n_items // 10]
    unaffordable = world.item_features[None, :, -1] > world.user_features[:, None, -1]

    def recall(top, relevant):
        """mean truncated recall@k over users with something relevant (your stage-1 metric from topic 22)."""
        hits = np.take_along_axis(relevant, top, axis=1).sum(axis=1)
        n = relevant.sum(axis=1)
        ok = n > 0
        return float(np.mean(hits[ok] / np.minimum(n[ok], top.shape[1])))

    def top_of(scores, k, mask):
        return np.argsort(-np.where(mask, -np.inf, scores), axis=1, kind="stable")[:, :k]

    def train_model(logq=True, use_features=True, seed=0):
        torch.manual_seed(seed)
        _uf = world.user_features if use_features else np.zeros_like(world.user_features)
        _if = world.item_features if use_features else np.zeros_like(world.item_features)
        _m = ex.TwoTower(world.n_users, world.n_items, _uf.shape[1], _if.shape[1])
        _t0 = time.perf_counter()
        _hist = ex.train_two_tower(_m, tr_u, tr_i, _uf, _if, epochs=8, logq_correction=logq, seed=seed)
        _U, _V = ex.embed_all(_m, _uf, _if)
        return _U, _V, _hist, time.perf_counter() - _t0

    _pop_top = top_of(np.broadcast_to(counts.astype(float), seen.shape), 10, exclude)
    pop_row = (recall(_pop_top, heldout), recall(top_of(np.broadcast_to(counts.astype(float), seen.shape), 100, exclude), heldout),
               recall(_pop_top, truth))
    mo.md(f"{len(world.users):,} interactions; {len(tr_u):,} train, {int(heldout.sum()):,} held-out (warm), "
          f"{int(cold_truth.sum()):,} with the {int(is_cold.sum())} cold items. truth: unaffordable items make up "
          f"{unaffordable[truth].mean():.1%} of the users' true favourites but {unaffordable.mean():.0%} of all pairs.")
    return (cold_truth, counts, exclude, head, heldout, is_cold, pop_row, recall, seen, top_of, tr_i, tr_u, train_model,
            truth, unaffordable, world)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what the logQ correction is for

    two identical two-tower models (dimension 32, temperature 0.1, 8 epochs of in-batch
    softmax with batches of 512), one trained with the logQ correction and one without.
    retrieval here is exact: every user is scored against every warm item.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the uncorrected model treats every in-batch item as a fair negative. how will its recall@10 compare with "
        "the corrected model's, and with plain popularity? will its top-10 lists be more or less popular?"
    )
    p1
    return (p1,)


@app.cell
def _(exclude, head, heldout, p1, pop_row, recall, top_of, train_model, truth):
    ui.gate(p1)
    runs = {_logq: train_model(logq=_logq) for _logq in (False, True)}
    _rows = [f"| popularity | — | {pop_row[0]:.3f} | {pop_row[1]:.3f} | {pop_row[2]:.3f} | 100% |"]
    for _logq, (_U, _V, _hist, _secs) in runs.items():
        _S = _U @ _V.T
        _t10, _t100 = top_of(_S, 10, exclude), top_of(_S, 100, exclude)
        _rows.append(f"| two-tower, logQ {'on' if _logq else 'off'} | {_hist[-1]:.2f} ({_secs:.0f} s) | {recall(_t10, heldout):.3f} | "
                     f"{recall(_t100, heldout):.3f} | {recall(_t10, truth):.3f} | {np.isin(_t10, head).mean():.0%} |")
    U, V = runs[True][0], runs[True][1]
    mo.md("| model | final training loss (time) | held-out recall@10 | held-out recall@100 | true-favourite recall@10 | "
          "top-10 items from the head |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + "\n\n(the head is the 10% most interacted-with items.)")
    return U, V


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the in-batch negatives aren't uniform: an item is in the batch as often as
    it's interacted with. what does the uncorrected softmax end up estimating, and why does
    that punish popular items? why is the training loss of the two models not a useful way
    to compare them? (the corrected model is used for the rest of the lab.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how approximate can the index be?

    an IVF index with 55 lists ($\approx\sqrt{3000}$) over the warm item embeddings. for
    several `n_probe` values: the index's recall against exact search (what fraction of the
    true top-100 inner products it returns), how many items it scores per query, and the
    end-to-end held-out recall@10 of recommending the top 10 of its 100 candidates (after
    dropping seen items).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with n_probe = 4 the index scores about 6% of the catalogue. what fraction of the exact top 100 does it "
        "find? and how much end-to-end recall@10 do you lose compared with exact search?"
    )
    p2
    return (p2,)


@app.cell
def _(U, V, exclude, heldout, is_cold, p2, recall, world):
    ui.gate(p2)
    _warm_ids = np.flatnonzero(~is_cold)
    _users = np.arange(world.n_users)
    _exact_ids, _ = ex.exact_search(U, V[_warm_ids], 100)

    def _retrieval_only(index):
        _top = ex.two_stage(_users, U, index, lambda _u, _c: -np.arange(_c.shape[1], dtype=float)[None].repeat(len(_u), 0),
                            n_candidates=100, k=10, exclude=exclude[:, _warm_ids])
        # an empty slot (-1) becomes a cold item, which is never a held-out hit
        return np.where(_top >= 0, _warm_ids[np.maximum(_top, 0)], np.flatnonzero(is_cold)[0])

    probes = [1, 2, 4, 8, 16, 55]
    ann = []
    for _p in probes:
        _idx = ex.IVFIndex(55, n_probe=_p, rng=0).build(V[_warm_ids])
        _t0 = time.perf_counter()
        _ids, _ = _idx.search(U, 100)
        _secs = time.perf_counter() - _t0
        _overlap = np.mean([len(np.intersect1d(a, b)) / 100 for a, b in zip(_ids, _exact_ids)])
        ann.append((_p, _overlap, _idx.last_n_scored_ / len(_warm_ids), _secs, recall(_retrieval_only(_idx), heldout)))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.plot([a[2] for a in ann], [a[1] for a in ann], "o-", label="index recall@100 vs exact")
    _ax.plot([a[2] for a in ann], [a[4] / ann[-1][4] for a in ann], "s-", label="end-to-end recall@10, relative to exact")
    for _p, _o, _f, _s, _r in ann:
        _ax.annotate(f"n_probe={_p}", (_f, _o), textcoords="offset points", xytext=(5, -10), fontsize=7)
    _ax.set(xscale="log", xlabel="fraction of the catalogue scored per query", ylabel="fraction", ylim=(0, 1.05))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| n_probe | index recall@100 vs exact | catalogue scored | search time, 1,500 queries | end-to-end held-out recall@10 |\n"
        "|---|---|---|---|---|\n" + "\n".join(f"| {p} | {o:.3f} | {f:.1%} | {s:.2f} s | {r:.3f} |" for p, o, f, s, r in ann))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the index throws away a large part of the true top 100 at small
    `n_probe`, yet the recommendations barely suffer. which neighbours does it miss, and why
    are those the ones that matter least for the top 10? (the search times are a Python loop
    over queries, so don't read too much into them at this catalogue size: count items scored
    instead.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: items nobody has seen

    the 300 cold items had every interaction removed before training, so their id embeddings
    never received a gradient. for each user, rank only the cold items and check against
    the interactions they really had with them. compared: the corrected model from
    experiment 1, and the same architecture trained with all features set to zero (ids
    only).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for cold items, how does the feature-based model compare with random ranking? and the ids-only model? "
        "which of the two models is better on warm items?"
    )
    p3
    return (p3,)


@app.cell
def _(U, V, cold_truth, exclude, heldout, is_cold, p3, recall, top_of, train_model):
    ui.gate(p3)
    _U0, _V0, _, _ = train_model(use_features=False)
    _not_cold = ~is_cold[None, :] | np.zeros_like(cold_truth)
    _rand = np.random.default_rng(0).random(cold_truth.shape)
    _rows = []
    for _name, _S in [("features + ids", U @ V.T), ("ids only", _U0 @ _V0.T), ("random", _rand)]:
        _cold = recall(top_of(_S, 10, _not_cold), cold_truth)
        _warm = recall(top_of(_S, 10, exclude), heldout)
        _rows.append(f"| {_name} | {_cold:.3f} | {_warm:.3f} |")
    mo.md("| model | cold items: recall@10 among cold items | warm items: held-out recall@10 |\n|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does the item tower compute for an item whose id embedding is still
    at its random initialization? why does the ids-only model fail on cold items, and what
    does the comparison on warm items tell you about what the features are worth when you
    *do* have interactions? how is this the same idea as topic 21's fold-in?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4 (project): a two-stage retrieve-then-rank recommender

    the full system. **stage 1:** the IVF index (`n_probe = 8`) retrieves $N$ candidates per
    user from the corrected two-tower embeddings. **stage 2:** your `Ranker` rescores them
    from `ranker_features`, which, unlike the towers, sees the user and the item
    *together* and so can use the cross feature price − budget. it's trained with your
    `train_ranker` on lists of one training positive and 20 **hard negatives**: unseen items
    the retriever put in that user's top 200. the table compares retrieval alone with
    two-stage at several $N$, ranking the whole warm catalogue, and the same ranker with the
    item's (log) popularity added as a fifth feature; then the whole system is retrained with
    two more seeds to see how much of the gain is noise. note what makes the ranker useful
    here: the simulator has a *hard* step (price above budget costs 3 points of
    preference) and the ranker receives that cross feature exactly. on real data, how much
    a second stage adds depends on what it can see that the towers can't.
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "does the second stage improve held-out recall@10 over retrieval alone, and by how much? how large must N be "
        "before ranking more candidates stops helping? and what happens when the ranker is also given each item's "
        "popularity?"
    )
    p4
    return (p4,)


@app.cell
def _(U, V, counts, exclude, head, heldout, is_cold, p4, recall, seen, tr_i, tr_u, train_model, truth, unaffordable,
      world):
    ui.gate(p4)
    _warm_ids = np.flatnonzero(~is_cold)
    _pad = np.flatnonzero(is_cold)[0]  # stands in for an empty slot (-1): a cold item, never relevant here
    _users = np.arange(world.n_users)
    _uf, _if = world.user_features, world.item_features  # last column of each: budget / price (see make_catalog)
    _logpop = np.log1p(counts).astype(np.float32)

    def _system(U, V, seed):
        """index + rankers (without / with popularity) for one set of tower embeddings."""
        index = ex.IVFIndex(55, n_probe=8, rng=seed).build(V[_warm_ids])
        # stage-2 training lists: [positive, 20 hard negatives from the user's retrieved top 200]
        cand, _ = index.search(U, 200)
        cand = np.where(cand >= 0, _warm_ids[np.maximum(cand, 0)], -1)
        g = np.random.default_rng(seed)
        negs = np.empty((len(tr_u), 20), dtype=np.int64)
        for n, u in enumerate(tr_u):
            c = cand[u][(cand[u] >= 0) & ~seen[u, np.maximum(cand[u], 0)]]
            negs[n] = g.choice(c, size=20, replace=len(c) < 20)
        lists, lu = np.c_[tr_i, negs], np.repeat(tr_u, 21)

        def features(users, items, with_pop):
            x = ex.ranker_features(U[users], V[items], _uf[users], _if[items])
            return np.c_[x, _logpop[items]] if with_pop else x

        rankers = {}
        for with_pop in (False, True):
            X = features(lu, lists.ravel(), with_pop).reshape(len(tr_u), 21, -1)
            torch.manual_seed(seed)
            r = ex.Ranker(X.shape[-1])
            t0 = time.perf_counter()
            hist = ex.train_ranker(r, X, epochs=5, seed=seed)
            rankers[with_pop] = (r, hist, time.perf_counter() - t0)

        def recommend(n_candidates, scorer, index=index):
            out = []
            for s0 in range(0, world.n_users, 250):  # chunks keep the ranker's batch small
                uu = _users[s0:s0 + 250]

                def score(users, cand_local):
                    return scorer(users, _warm_ids[np.maximum(cand_local, 0)])

                top = ex.two_stage(uu, U, index, score, n_candidates=n_candidates, k=10, exclude=exclude[:, _warm_ids])
                out.append(np.where(top >= 0, _warm_ids[np.maximum(top, 0)], _pad))
            return np.concatenate(out)

        def retrieval_score(users, items):
            return np.einsum("qd,qnd->qn", U[users], V[items])

        def ranker_score(with_pop):
            def f(users, items):
                x = features(np.repeat(users, items.shape[1]), items.ravel(), with_pop)
                with torch.no_grad():
                    return rankers[with_pop][0](torch.as_tensor(x)).numpy().reshape(items.shape)
            return f

        return recommend, retrieval_score, ranker_score, rankers

    def _row(name, top, n_scored):
        return (f"| {name} | {n_scored} | {recall(top, heldout):.3f} | {recall(top, truth):.3f} | "
                f"{np.take_along_axis(unaffordable, top, axis=1).mean():.1%} | {np.isin(top, head).mean():.0%} |")

    _recommend, _retrieval_score, _ranker_score, _rankers = _system(U, V, 0)
    _rows = [_row("retrieval only (top 10 of 200 candidates)", _recommend(200, _retrieval_score), "—")]
    for _N in (20, 50, 100, 200, 500):
        _rows.append(_row(f"two-stage, N = {_N}", _recommend(_N, _ranker_score(False)), _N))
    _all = ex.IVFIndex(1, n_probe=1, rng=0).build(V[_warm_ids])
    _rows.append(_row("rank every warm item", _recommend(len(_warm_ids), _ranker_score(False), index=_all), len(_warm_ids)))
    _rows.append(_row("two-stage, N = 200, ranker also sees popularity", _recommend(200, _ranker_score(True)), 200))

    # the gain over training seeds: towers, index and ranker all retrained; paired bootstrap over users
    def _per_user(top):
        hits = np.take_along_axis(heldout, top, axis=1).sum(axis=1)
        n = heldout.sum(axis=1)
        ok = n > 0
        return hits[ok] / np.minimum(n[ok], 10)

    _seed_rows, _gains = [], []
    for _seed in (0, 1, 2):
        if _seed == 0:
            _rec, _rs, _rk = _recommend, _retrieval_score, _ranker_score
        else:
            _Us, _Vs, _, _ = train_model(seed=_seed)
            _rec, _rs, _rk, _ = _system(_Us, _Vs, _seed)
        _a, _b, _c = (_per_user(_rec(200, _rs)), _per_user(_rec(100, _rk(False))), _per_user(_rec(200, _rk(True))))
        _lo, _hi = bootstrap_ci(_b - _a, np.mean, n_boot=2000, rng=_seed)
        _gains.append(np.mean(_b - _a))
        _seed_rows.append(f"| {_seed} | {_a.mean():.3f} | {_b.mean():.3f} | {_gains[-1]:+.3f} [{_lo:+.3f}, {_hi:+.3f}] | "
                          f"{_c.mean():.3f} |")
    mo.md("training seed 0:\n\n| system | items the ranker scores per user | held-out recall@10 | true-favourite recall@10 | "
          "unaffordable share of the top 10 | top 10 from the head |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\nthe users' true favourites are {unaffordable[truth].mean():.1%} unaffordable. ranker training: "
          f"{_rankers[False][2]:.1f} s, list loss {_rankers[False][1][0]:.2f} → {_rankers[False][1][-1]:.2f} "
          f"(with popularity: {_rankers[True][1][-1]:.2f}).\n\nthree training seeds (towers, index, negatives and ranker "
          "all retrained), held-out recall@10:\n\n| seed | retrieval only | two-stage, N = 100 | gain [95% paired-bootstrap CI over users] | "
          "two-stage + popularity, N = 200 |\n|---|---|---|---|---|\n" + "\n".join(_seed_rows)
          + f"\n\nmean gain **{np.mean(_gains):+.3f}** (range {min(_gains):+.3f} to {max(_gains):+.3f}).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what can the ranker do that the towers can't, and how do you see it in
    the unaffordable column? why does ranking more candidates stop helping after a fairly
    small $N$ (what is retrieval's job, and what is ranking's), and why is $N = 20$ *worse*
    than retrieval alone? adding popularity, a feature that obviously predicts
    interactions, made held-out recall *worse* than the same ranker without it, while
    true-favourite recall didn't suffer.
    the ranker's training positives are items the towers were fitted on, and its negatives
    are popular items the towers ranked highly: what does it learn about popularity from
    such lists (see the last column)? how would you train the second stage in a real
    system?
    """)
    return


if __name__ == "__main__":
    app.run()
