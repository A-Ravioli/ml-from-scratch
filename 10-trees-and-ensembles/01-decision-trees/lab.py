import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.model_selection import KFold, validation_curve  # yours, from ch 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # decision trees · lab

    every experiment here runs **your** trees from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    experiments 1 and 2 use the same 2-D problem: points uniform on $[-1, 1]^2$, class 1
    above the curve $x_2 = 0.5\sin(3x_1)$, and **20% of the labels flipped** at random. so
    no classifier can beat 80% test accuracy (the Bayes rate), and a classifier that gets the
    curve exactly right gets exactly 80%. 1,000 training points, 20,000 test points.
    """)
    return


@app.cell
def _():
    def make_wave(n, rng, flip=0.2):
        X = rng.uniform(-1, 1, size=(n, 2))
        clean = (X[:, 1] > 0.5 * np.sin(3 * X[:, 0])).astype(int)
        y = np.where(rng.random(n) < flip, 1 - clean, clean)
        return X, y, clean

    _rng = np.random.default_rng(0)
    X_tr, y_tr, _ = make_wave(1000, _rng)
    X_te, y_te, clean_te = make_wave(20000, _rng)
    _g = np.linspace(-1, 1, 200)
    grid = np.array(np.meshgrid(_g, _g)).reshape(2, -1).T  # for drawing decision regions

    def draw(ax, model, title):
        ax.imshow(model.predict(grid).reshape(200, 200), origin="lower", extent=[-1, 1, -1, 1],
                  cmap="coolwarm", alpha=0.45, vmin=0, vmax=1)
        ax.plot(_g, 0.5 * np.sin(3 * _g), "k--", lw=1)
        ax.set(title=title, xticks=[], yticks=[], aspect="equal")
    return X_te, X_tr, clean_te, draw, y_te, y_tr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: depth versus overfitting

    we fit your `DecisionTreeClassifier` with `max_depth` from 1 to 20 and unlimited, and
    record training and test accuracy. then we let 5-fold cross-validation (your
    `validation_curve`) choose `ccp_alpha` for a fully grown tree, and prune.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the fully grown tree is 100% accurate on the training set. what's its TEST accuracy, given that the "
        "best possible is 80%? and which depth is best?"
    )
    p1
    return (p1,)


@app.cell
def _(X_te, X_tr, clean_te, draw, p1, y_te, y_tr):
    ui.gate(p1)
    _depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
    _rows = []
    for _d in _depths:
        _m = ex.DecisionTreeClassifier(max_depth=_d).fit(X_tr, y_tr)
        _rows.append((_m.get_n_leaves(), _m.score(X_tr, y_tr), _m.score(X_te, y_te)))
    _rows = np.array(_rows)
    _alphas = [0.0, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2]
    _, _va = validation_curve(ex.DecisionTreeClassifier(), X_tr, y_tr, "ccp_alpha", _alphas,
                              cv=KFold(5, shuffle=True, rng=0))
    _best = _alphas[int(np.argmax(_va.mean(axis=1)))]
    _pruned = ex.DecisionTreeClassifier(ccp_alpha=_best).fit(X_tr, y_tr)
    _full = ex.DecisionTreeClassifier().fit(X_tr, y_tr)
    _d3 = ex.DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    _x = np.arange(len(_depths))
    _axes[0].plot(_x, _rows[:, 1], "o-", label="train")
    _axes[0].plot(_x, _rows[:, 2], "o-", label="test")
    _axes[0].axhline(0.8, color="k", ls=":", label="Bayes rate 0.80")
    _axes[0].axhline(0.68, color="gray", ls="--", label="1 − 2ε(1 − ε) = 0.68")
    _axes[0].set(xticks=_x, xticklabels=[str(d) if d else "∞" for d in _depths], xlabel="max_depth",
                 ylabel="accuracy", ylim=(0.6, 1.02))
    _axes[0].legend(fontsize=8)
    draw(_axes[1], _d3, f"max_depth=3: test {_d3.score(X_te, y_te):.3f}")
    draw(_axes[2], _full, f"fully grown ({_full.get_n_leaves()} leaves): test {_full.score(X_te, y_te):.3f}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"fully grown: train **{_full.score(X_tr, y_tr):.3f}**, test **{_full.score(X_te, y_te):.3f}**, and it "
        f"agrees with the *clean* labels on only {np.mean(_full.predict(X_te) == clean_te):.3f} of the test points. "
        f"best depth: **{_depths[int(np.argmax(_rows[:, 2]))]}** (test {_rows[:, 2].max():.3f}). "
        f"CV picks `ccp_alpha` = **{_best}**: the pruned tree has **{_pruned.get_n_leaves()}** leaves "
        f"(from {_full.get_n_leaves()}) and test accuracy **{_pruned.score(X_te, y_te):.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does the fully grown tree's test accuracy come from? (hint: a test
    point lands in a leaf that holds one training point, whose label was flipped with
    probability $\varepsilon = 0.2$, and its own label is flipped with probability
    $\varepsilon$ too.) why does accuracy against the *clean* labels fall as well? why are
    depths 1 and 2 equally bad here, and what does that say about greedy splitting?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how stable is a tree?

    we refit a tree 20 times, each time on the same training set with a random 5% of the rows
    left out, and measure how often two refits disagree on a test point, for depth 3 and for
    fully grown trees.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "removing 5% of the rows and refitting a fully grown tree: on what fraction of test points do two such "
        "trees disagree? and two depth-3 trees?"
    )
    p2
    return (p2,)


@app.cell
def _(X_te, X_tr, draw, p2, y_tr):
    ui.gate(p2)
    _out = {}
    _trees = {}
    _thr = {}
    for _depth in (3, None):
        _P, _roots = [], set()
        for _r in range(20):
            _keep = np.random.default_rng(100 + _r).random(len(y_tr)) < 0.95
            _m = ex.DecisionTreeClassifier(max_depth=_depth).fit(X_tr[_keep], y_tr[_keep])
            _P.append(_m.predict(X_te))
            _roots.add((int(_m.feature_[0]), round(float(_m.threshold_[0]), 2)))
            _thr.setdefault(_depth, []).append(float(_m.threshold_[0]))
            _trees.setdefault(_depth, []).append(_m)
        _P = np.array(_P)
        _dis = [np.mean(_P[i] != _P[j]) for i in range(20) for j in range(i)]
        _out[_depth] = (np.mean(_dis), np.min(_dis), np.max(_dis),
                        f"{len({f for f, _ in _roots})} feature(s), thresholds {min(_thr[_depth]):.3f} – {max(_thr[_depth]):.3f}")
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
    draw(_axes[0], _trees[None][0], "fully grown, refit A")
    draw(_axes[1], _trees[None][1], "fully grown, refit B")
    _fig.tight_layout()
    _md = "| tree | mean disagreement | min – max over pairs | root splits |\n|---|---|---|---|\n"
    for _depth, (_mean, _lo, _hi, _nr) in _out.items():
        _md += f"| {'depth 3' if _depth else 'fully grown'} | {_mean:.3f} | {_lo:.3f} – {_hi:.3f} | {_nr} |\n"
    mo.vstack([_fig, mo.md(_md)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the root splits barely move between refits (the same feature, thresholds
    within a few hundredths). why do the fully grown trees still disagree so much, and where
    in the picture do they disagree? how does an error
    made high in the tree propagate? what would averaging many such trees do to the
    disagreement, and what to the accuracy? (that's topic 02.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: boundaries that aren't axis-aligned

    no label noise now: 2,000 points uniform on $[-1, 1]^2$, class 1 on one side of a
    straight line through the origin at angle $\theta$. a fully grown tree classifies the
    training set perfectly; we count its leaves and measure test accuracy for $\theta$ from
    0° to 90°.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "a single straight boundary. at theta = 0 the tree needs 2 leaves. how many does it need at 45 degrees, "
        "and what test accuracy does it reach?"
    )
    p3
    return (p3,)


@app.cell
def _():
    theta = mo.ui.slider(0, 90, value=45, step=5, label="θ (degrees)")
    theta
    return (theta,)


@app.cell
def _(p3, theta):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _X = _rng.uniform(-1, 1, size=(2000, 2))
    _Xt = _rng.uniform(-1, 1, size=(20000, 2))

    def _labels(X, deg):
        a = np.deg2rad(deg)
        return (X @ np.array([-np.sin(a), np.cos(a)]) > 0).astype(int)

    _angles = np.arange(0, 91, 5)
    _leaves, _acc = [], []
    for _a in _angles:
        _m = ex.DecisionTreeClassifier().fit(_X, _labels(_X, _a))
        _leaves.append(_m.get_n_leaves())
        _acc.append(_m.score(_Xt, _labels(_Xt, _a)))
    _m = ex.DecisionTreeClassifier().fit(_X, _labels(_X, theta.value))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_angles, _leaves, "o-", color="C0")
    _axes[0].set(xlabel="θ (degrees)", ylabel="leaves", title="leaves of the fully grown tree")
    _ax2 = _axes[0].twinx()
    _ax2.plot(_angles, _acc, "s--", color="C3")
    _ax2.set_ylabel("test accuracy", color="C3")
    _g = np.linspace(-1, 1, 300)
    _G = np.array(np.meshgrid(_g, _g)).reshape(2, -1).T
    _axes[1].imshow(_m.predict(_G).reshape(300, 300), origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm",
                    alpha=0.5)
    _a = np.deg2rad(theta.value)
    _axes[1].plot([-1.5 * np.cos(_a), 1.5 * np.cos(_a)], [-1.5 * np.sin(_a), 1.5 * np.sin(_a)], "k--")
    _axes[1].set(title=f"θ = {theta.value}°: {_m.get_n_leaves()} leaves", aspect="equal", xlim=(-1, 1),
                 ylim=(-1, 1))
    _fig.tight_layout()
    _i45 = int(np.flatnonzero(_angles == 45)[0])
    mo.vstack([_fig, mo.md(
        f"at 45°: **{_leaves[_i45]}** leaves, test accuracy **{_acc[_i45]:.4f}**; at 0° and 90°: "
        f"{_leaves[0]} and {_leaves[-1]} leaves.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a straight diagonal line cost dozens of leaves? what shape does
    the tree draw instead, and where are its errors? how would the number of leaves grow if
    you had 10 times more data? what single feature would you add so that one split suffices,
    and what does that tell you about feature engineering for trees?
    """)
    return


if __name__ == "__main__":
    app.run()
