import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.utils import image_grid

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(min(2, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)

    N_EPISODES = 400  # training episodes for every configuration (1-shot, 3 queries per class)
    N_TEST = 300  # test episodes (5-way; 1-shot with 5 queries, or 5-shot with 1 query)
    TEST_POOL = 6  # photos per test person that test episodes may use; the other 4 are held back
    SPLITS = (0, 1, 2)  # split s: people split with seed s, training seed s


@app.function
def mlp_embed():
    """1024 pixels -> 256 -> 128."""
    return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 256), nn.ReLU(), nn.Linear(256, 128))


@app.function
def conv_embed():
    """three conv3x3(16)-batchnorm-ReLU-maxpool blocks (a narrower, shorter Conv-4): 32x32 -> 256."""
    layers, c = [], 1
    for _ in range(3):
        layers += [nn.Conv2d(c, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)]
        c = 16
    return nn.Sequential(*layers, nn.Flatten())


@app.function
def test_acc(embed, X, y_eval, test_classes, seed, k_shot=1, metric="sqeuclidean", scale=1.0):
    """5-way k-shot accuracy on test_classes, using only the TEST_POOL photos y_eval keeps (300 seeded episodes)."""
    return ex.evaluate_protonet(embed, X, y_eval, test_classes, n_way=min(5, len(test_classes)), k_shot=k_shot,
                                n_query=TEST_POOL - k_shot if k_shot > 1 else 5, n_episodes=N_TEST, metric=metric,
                                scale=scale, seed=1000 + seed)[0]


@app.function
def run(make, X, y, train_classes, test_classes, y_eval, seed, *, n_way, metric="sqeuclidean", scale=1.0):
    """train a fresh embedding with n_way 1-shot episodes on labels y; return (5-way 1-shot acc, 5-way 5-shot acc,
    seconds) on test_classes, scored with labels y_eval."""
    n_way = min(n_way, len(train_classes))  # only matters on the small offline stand-in data
    torch.manual_seed(seed)
    embed = make()
    t0 = time.perf_counter()
    ex.train_protonet(embed, X, y, train_classes, n_way=n_way, k_shot=1, n_query=3, n_episodes=N_EPISODES,
                      lr=1e-3, metric=metric, scale=scale, seed=seed)
    secs = time.perf_counter() - t0
    return (test_acc(embed, X, y_eval, test_classes, seed, 1, metric, scale),
            test_acc(embed, X, y_eval, test_classes, seed, 5, metric, scale), secs)


@app.function
def summary(values):
    """'mean ± sd' over splits."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


@app.function
def per_split(values):
    """the per-split values, comma-separated."""
    return ", ".join(f"{v:.3f}" for v in values)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # prototypical networks · lab

    every experiment here runs **your** episode sampler, prototypes, loss, training and
    evaluation. the data is the Olivetti faces: 40 people, 10 photos each, downsampled to
    32×32, a few-shot problem by construction. we split the *people*, 28 for meta-training
    and 12 for meta-testing. test episodes are 5-way and use {TEST_POOL} fixed photos of each test
    person: 1 support photo and 5 queries, or 5 support photos and 1 query ({N_TEST} episodes per
    score, so one 1-shot score's 95% interval is about ±0.01). the other 4 photos of each test
    person are held back for experiment 1's cheat. each configuration trains for {N_EPISODES}
    1-shot episodes (3 queries per class) with Adam at $10^{{-3}}$. **which 12 people end up in the test split matters a lot**, so everything runs
    on three splits (split $s$ also uses training seed $s$), and methods are compared split by
    split. the lab takes under a minute on an idle laptop CPU.
    """)
    return


@app.cell
def _():
    _faces, _labels = datasets.olivetti_faces()
    _X = F.avg_pool2d(torch.tensor(_faces, dtype=torch.float32).unsqueeze(1), 2)  # (n, 1, 32, 32)
    X = (_X - _X.mean()) / _X.std()
    y = torch.tensor(_labels)
    _counts = torch.bincount(y)
    people = [c for c in range(len(_counts)) if _counts[c] >= 10]  # all 40 on the real data
    splits = {s: ex.split_classes(people, round(0.3 * len(people)), generator=torch.Generator().manual_seed(s))
              for s in SPLITS}
    # per split: y_eval hides 4 photos of every test person (set to -1, a label no episode asks for); y_leak hides
    # the other TEST_POOL, so a model trained on y_leak sees test people, but never a photo a test episode uses
    y_eval, y_leak = {}, {}
    for _s, (_, _te) in splits.items():
        y_eval[_s], y_leak[_s] = y.clone(), y.clone()
        _g = torch.Generator().manual_seed(100 + _s)
        for _c in _te:
            _idx = torch.nonzero(y == _c).flatten()
            _idx = _idx[torch.randperm(len(_idx), generator=_g)]
            y_eval[_s][_idx[:len(_idx) - TEST_POOL]] = -1
            y_leak[_s][_idx[len(_idx) - TEST_POOL:]] = -1
    mo.md("\n".join(f"- split {s}: test people {list(te)}" for s, (_, te) in splits.items()))
    return X, splits, y, y_eval, y_leak


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does an embedding learned on other people buy?

    the baseline needs no training: prototypes of the raw pixels (`nn.Flatten()` as the
    embedding). against it, an MLP trained with 20-way 1-shot episodes on the 28 training people.
    and one more run that breaks the rules: the same MLP trained on episodes from **all 40**
    people, using the 4 held-back photos of each test person, then scored on the same test
    episodes. it never sees a photo that a test episode uses.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "5-way 1-shot accuracy on unseen people: raw pixels? the trained MLP? the MLP that also trained on 4 other "
        "photos of each test person (never on a photo the test episodes use)?"
    )
    p1
    return (p1,)


@app.cell
def _(X, p1, splits, y, y_eval, y_leak):
    ui.gate(p1)
    pixels = {k: [test_acc(nn.Flatten(), X, y_eval[s], splits[s][1], s, k) for s in SPLITS] for k in (1, 5)}
    mlp20 = [run(mlp_embed, X, y, *splits[s], y_eval[s], s, n_way=20) for s in SPLITS]
    _leak = [run(mlp_embed, X, y_leak[s], tuple(sorted(splits[s][0] + splits[s][1])), splits[s][1], y_eval[s], s,
                 n_way=20) for s in SPLITS]
    _rows = [("raw pixels", "nothing", pixels[1], pixels[5]),
             ("MLP", "28 training people", [r[0] for r in mlp20], [r[1] for r in mlp20]),
             ("MLP", "**all 40 people**, 4 other photos per test person (leak)", [r[0] for r in _leak],
              [r[1] for r in _leak])]
    mo.md("| embedding | trained on | 5-way 1-shot, per split | mean ± sd | 5-way 5-shot, mean ± sd |\n"
          "|---|---|---|---|---|\n"
          + "\n".join(f"| {a} | {b} | {per_split(o)} | {summary(o)} | {summary(f)} |" for a, b, o, f in _rows)
          + "\n\nMLP minus pixels, 1-shot, per split: "
          + per_split(np.array([r[0] for r in mlp20]) - np.array(pixels[1])))
    return mlp20, pixels


@app.cell
def _(X, splits, y):
    ep_pick = mo.ui.slider(0, 9, value=0, label="test episode (split 0)")
    _tr, _te = splits[0]
    torch.manual_seed(0)
    show_embed = mlp_embed()
    ex.train_protonet(show_embed, X, y, _tr, n_way=min(20, len(_tr)), k_shot=1, n_query=3, n_episodes=N_EPISODES,
                      seed=0)
    ep_pick
    return ep_pick, show_embed


@app.cell
def _(X, ep_pick, show_embed, splits, y_eval):
    _te = splits[0][1]
    _w = min(5, len(_te))
    _ep = ex.sample_episode(y_eval[0], _te, _w, 1, 2, generator=torch.Generator().manual_seed(500 + ep_pick.value))
    _order = torch.arange(2 * _w).reshape(_w, 2).T.flatten()  # query-major: column j shows person j's queries
    _ep["query"], _ep["query_labels"] = _ep["query"][_order], _ep["query_labels"][_order]
    show_embed.eval()
    with torch.no_grad():
        _z = show_embed(X[torch.cat([_ep["support"], _ep["query"]])])
        _logits = ex.proto_logits(_z[_w:], ex.prototypes(_z[:_w], _ep["support_labels"], _w))
    _pred = _logits.argmax(1)
    _imgs = torch.cat([X[_ep["support"]], X[_ep["query"]]])[:, 0].numpy()
    _titles = [f"person {j}" for j in range(_w)] + [
        f"→ {int(p)} " + ("ok" if p == t else "WRONG") for p, t in zip(_pred, _ep["query_labels"])]
    _fig, _ax = plt.subplots(figsize=(7, 4.4))
    image_grid((_imgs - _imgs.min()) / (_imgs.max() - _imgs.min()), nrow=_w, ax=_ax, titles=_titles)
    _ax.set_title("top row: each person's support photo; below it: two queries of that person, and the person "
                  "they were matched to", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the MLP never saw any of the test people. on which splits does it beat raw
    pixels, and by how much compared with the split-to-split differences? the leaky model never
    saw a photo that a test episode uses, only 4 *other* photos of each test person: why does
    that inflate the score so much, and what would it mean for a paper that split photos, not
    people? (step through the episodes above: what makes the errors hard?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: episode design or architecture?

    three ways to change the training recipe, one at a time, against the 20-way MLP of
    experiment 1: train with 5-way instead of 20-way episodes; train on only 7 of the 28 people
    (5-way, since there are only 7 classes); swap the MLP for a small convnet (5-way). every
    row trains for the same number of episodes and is tested on the same held-out people.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the three changes by how much they move 5-way 1-shot test accuracy: 5-way training, 7 training people, "
        "a convnet. which is the largest?"
    )
    p2
    return (p2,)


@app.cell
def _(X, mlp20, p2, pixels, splits, y, y_eval):
    ui.gate(p2)
    rows = {"MLP, 20-way (exp. 1)": mlp20}
    rows["MLP, 5-way"] = [run(mlp_embed, X, y, *splits[s], y_eval[s], s, n_way=5) for s in SPLITS]
    rows["MLP, 5-way, 7 people"] = [run(mlp_embed, X, y, splits[s][0][:7], splits[s][1], y_eval[s], s, n_way=5)
                                    for s in SPLITS]
    rows["convnet, 5-way"] = [run(conv_embed, X, y, *splits[s], y_eval[s], s, n_way=5) for s in SPLITS]
    _base = np.array([r[0] for r in mlp20])
    _fig, _ax = plt.subplots(figsize=(6.4, 2.6))
    for _i, (_n, _v) in enumerate(rows.items()):
        _d = np.array([r[0] for r in _v]) - _base
        _ax.scatter(_d, [_i] * len(_d), color=f"C{_i}", s=30, zorder=3)
        _ax.plot([_d.mean()] * 2, [_i - 0.3, _i + 0.3], color=f"C{_i}", lw=2)
    _pix = np.array(pixels[1]) - _base
    _ax.scatter(_pix, [len(rows)] * len(_pix), color="gray", s=30)
    _ax.plot([_pix.mean()] * 2, [len(rows) - 0.3, len(rows) + 0.3], color="gray", lw=2)
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set_yticks(range(len(rows) + 1), list(rows) + ["raw pixels (no training)"])
    _ax.set(xlabel="5-way 1-shot accuracy minus the 20-way MLP's (same split)",
            title="dots: the three splits; bars: their mean")
    _ax.invert_yaxis()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| training recipe | 5-way 1-shot | difference vs 20-way MLP, per split | 5-way 5-shot | training time |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(f"| {n} | {summary([r[0] for r in v])} | {per_split(np.array([r[0] for r in v]) - _base)} | "
                    f"{summary([r[1] for r in v])} | {np.mean([r[2] for r in v]):.1f} s |" for n, v in rows.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which changes are about the *episodes* and which about the *network*, and
    which moved accuracy most? Snell et al. found that training with more ways than you test
    with helps; did 20-way episodes help here, beyond the split-to-split noise? the 7-people row
    differs from the 5-way row in one thing only: compare those two. what does that say about
    where the transferable knowledge comes from, at this training budget?
    is "episode design matters more than architecture" true here, or only half true? and
    compare the size of every effect with experiment 1's split-to-split differences: why must
    methods be compared on the same split?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: which distance?

    the same 20-way MLP, trained and tested with cosine similarity as the logit instead of
    negative squared distance: once as is (logits in $[-1, 1]$), once multiplied by 10.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "cosine similarity ignores the embeddings' lengths, which seems like a good idea for faces. better or worse "
        "than squared distance? and what will the scale of 10 change?"
    )
    p3
    return (p3,)


@app.cell
def _(X, mlp20, p3, splits, y, y_eval):
    ui.gate(p3)
    _cos1 = [run(mlp_embed, X, y, *splits[s], y_eval[s], s, n_way=20, metric="cosine", scale=1.0) for s in SPLITS]
    _cos10 = [run(mlp_embed, X, y, *splits[s], y_eval[s], s, n_way=20, metric="cosine", scale=10.0)
              for s in SPLITS]
    _rows = [("$-\\lVert z - c \\rVert^2$ (exp. 1)", mlp20), ("cosine", _cos1), ("10 × cosine", _cos10)]
    mo.md("| logits | 5-way 1-shot, per split | mean ± sd | 5-way 5-shot, mean ± sd |\n|---|---|---|---|\n"
          + "\n".join(f"| {n} | {per_split([r[0] for r in v])} | {summary([r[0] for r in v])} | "
                      f"{summary([r[1] for r in v])} |" for n, v in _rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with logits confined to $[-1, 1]$ and 20 classes, what is the smallest
    cross-entropy the model can possibly reach? what does that do to the gradients once most
    queries are classified correctly? the scale doesn't change which prototype is nearest: why
    does it change what the network learns?
    """)
    return


if __name__ == "__main__":
    app.run()
