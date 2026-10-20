import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.base import Classifier, check_X_y  # yours, from topic 01
    from mlfs.utils import pairwise_distances  # yours, from ch 00

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def top_correlated(X, y, k):
    """indices of the k columns of X with the largest |correlation| with the (binary) labels y."""
    t = (np.asarray(y) == np.unique(y)[-1]).astype(float)
    Xc, tc = X - X.mean(0), t - t.mean()
    corr = (Xc.T @ tc) / (np.sqrt((Xc**2).sum(0) * (tc**2).sum()) + 1e-12)
    return np.argsort(-np.abs(corr))[:k]


@app.class_definition
class NearestCentroid(Classifier):
    """predicts the class whose mean training vector is closest (a tiny lab helper)."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        X, _ = check_X_y(X)
        nearest = np.argmin(pairwise_distances(X, self.centroids_), axis=1)
        return np.eye(len(self.classes_))[nearest]


@app.class_definition
class SelectThenCentroid(Classifier):
    """keeps the k features most correlated with y (learned in fit), then a nearest centroid."""

    def __init__(self, k=20):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.columns_ = top_correlated(X, y, self.k)
        self.model_ = NearestCentroid().fit(X[:, self.columns_], y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X, _ = check_X_y(X)
        return self.model_.predict_proba(X[:, self.columns_])


@app.class_definition
class OneNN(Classifier):
    """1-nearest-neighbour: the label of the closest training row (a lab helper; chapter 09 does
    this properly)."""

    def fit(self, X, y):
        self.X_, self.y_ = check_X_y(X, y)
        self.classes_, self.idx_ = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba(self, X):
        X, _ = check_X_y(X)
        nearest = np.argmin(pairwise_distances(X, self.X_), axis=1)
        return np.eye(len(self.classes_))[self.idx_[nearest]]


@app.function
def ar1(n, phi, rng):
    """an AR(1) series x_t = phi x_{t-1} + noise: slowly drifting for phi close to 1."""
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # splitting & cross-validation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. all three experiments use data where the
    honest answer is known: the labels carry **no** usable signal, so the true accuracy of any
    model is 50%.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: preprocessing before splitting

    60 samples (30 per class), $d$ features of pure noise, labels independent of everything.
    the model keeps the $k = 20$ features most correlated with the label, then classifies by
    nearest centroid. three ways to cross-validate it (5-fold stratified, your splitters and
    your `cross_val_score`), each on 10 independent datasets:

    - **honest:** feature selection happens inside `fit`, so each fold selects on its own
      training rows only;
    - **leaky selection:** select the 20 features once on *all* 60 rows, then cross-validate
      the classifier on those columns;
    - **leaky scaling:** standardize every feature with the mean and standard deviation of
      all 60 rows, then cross-validate the honest pipeline.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the true accuracy is 50% for every variant. what CV accuracy do you expect from the "
        "honest one, the leaky selection, and the leaky scaling?"
    )
    p1
    return (p1,)


@app.cell
def _():
    d1 = mo.ui.slider(50, 5000, value=2000, step=50, label="number of noise features d")
    d1
    return (d1,)


@app.cell
def _(d1, p1):
    ui.gate(p1)
    _res = {"honest": [], "leaky selection": [], "leaky scaling": []}
    for _seed in range(10):
        _rng = np.random.default_rng(_seed)
        _X = _rng.normal(size=(60, d1.value))
        _y = np.repeat([0, 1], 30)
        _cv = ex.StratifiedKFold(5, shuffle=True, rng=_seed)
        _res["honest"].append(ex.cross_val_score(SelectThenCentroid(20), _X, _y, cv=_cv).mean())
        _cols = top_correlated(_X, _y, 20)  # selected using ALL the labels
        _res["leaky selection"].append(ex.cross_val_score(NearestCentroid(), _X[:, _cols], _y, cv=_cv).mean())
        _Xs = (_X - _X.mean(0)) / _X.std(0)  # statistics from ALL the rows
        _res["leaky scaling"].append(ex.cross_val_score(SelectThenCentroid(20), _Xs, _y, cv=_cv).mean())
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    for _i, (_name, _v) in enumerate(_res.items()):
        _ax.scatter(np.full(10, _i) + np.linspace(-0.15, 0.15, 10), _v, s=18)
        _ax.plot([_i - 0.25, _i + 0.25], [np.mean(_v)] * 2, "k", lw=2)
    _ax.axhline(0.5, color="gray", ls=":", label="true accuracy (chance)")
    _ax.set_xticks(range(3), list(_res))
    _ax.set(ylabel="5-fold CV accuracy", ylim=(0.0, 1.05), title=f"pure noise, d = {d1.value}, 10 datasets")
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {k} | {np.mean(v):.2f} | {np.min(v):.2f} – {np.max(v):.2f} |" for k, v in _res.items())
    mo.vstack([_fig, mo.md("| variant | mean CV accuracy | range over 10 datasets |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both leaky variants use information from the test folds. why is one of
    them catastrophic and the other harmless here? (what did each one learn from the test
    rows: the labels, or only the inputs?) why does the leaky selection get worse as $d$
    grows? and what would the honest variant look like as code: where must the selection
    live?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: patients, not recordings

    30 patients, each recorded 8 times. every patient has a personal "fingerprint" (their 10
    features vary a lot between people and little between recordings), and each patient is
    labelled sick or healthy at random: **the features say nothing about sickness.** a
    1-nearest-neighbour classifier, cross-validated two ways: shuffled 5-fold over
    recordings (`KFold`), and 5-fold over patients (`GroupKFold`).
    """)
    return


@app.cell
def _():
    p2 = ui.predict("what CV accuracy will 1-NN get with KFold, and with GroupKFold?")
    p2
    return (p2,)


@app.cell
def _():
    within = mo.ui.slider(0.05, 2.0, value=0.3, step=0.05, label="spread between recordings of the same patient")
    within
    return (within,)


@app.cell
def _(p2, within):
    ui.gate(p2)
    _kf, _gkf = [], []
    for _seed in range(10):
        _rng = np.random.default_rng(100 + _seed)
        _patients = np.repeat(np.arange(30), 8)
        _finger = _rng.normal(size=(30, 10))
        _X = _finger[_patients] + within.value * _rng.normal(size=(240, 10))
        _y = _rng.permutation(np.repeat([0, 1], 15))[_patients]
        _kf.append(ex.cross_val_score(OneNN(), _X, _y, cv=ex.KFold(5, shuffle=True, rng=_seed)).mean())
        _gkf.append(ex.cross_val_score(OneNN(), _X, _y, cv=ex.GroupKFold(5), groups=_patients).mean())
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.boxplot([_kf, _gkf])
    _ax.set_xticks([1, 2], ["KFold over recordings", "GroupKFold over patients"])
    _ax.axhline(0.5, color="gray", ls=":")
    _ax.set(ylabel="5-fold CV accuracy (10 datasets)", ylim=(0.0, 1.05))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"mean CV accuracy: KFold **{np.mean(_kf):.2f}**, GroupKFold **{np.mean(_gkf):.2f}** "
                            f"(within-patient spread {within.value})")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what is 1-NN's nearest neighbour for a test recording under each scheme?
    what is the KFold number actually measuring? slide the within-patient spread up to 2:
    why does the gap close? which question does a hospital deploying this model care about,
    "a new recording of a patient we've seen" or "a new patient"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the future leaks into the past

    a time series of 600 days. the two features are slowly drifting random processes
    ($x_t = 0.995\,x_{t-1} + \text{noise}$), and the label is whether a third, *independent*
    process ($z_t = 0.95\,z_{t-1} + \text{noise}$, which forgets faster) is positive. again
    the features carry no information about the label. 1-NN, cross-validated with shuffled
    `KFold` and with your `TimeSeriesSplit`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("what CV accuracy will 1-NN get with shuffled KFold, and with TimeSeriesSplit?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _kf, _ts = [], []
    for _seed in range(10):
        _rng = np.random.default_rng(200 + _seed)
        _X = np.column_stack([ar1(600, 0.995, _rng), ar1(600, 0.995, _rng)])
        _y = (ar1(600, 0.95, _rng) > 0).astype(int)
        _kf.append(ex.cross_val_score(OneNN(), _X, _y, cv=ex.KFold(5, shuffle=True, rng=_seed)).mean())
        _ts.append(ex.cross_val_score(OneNN(), _X, _y, cv=ex.TimeSeriesSplit(5)).mean())
    _rng = np.random.default_rng(200)
    _Xd = np.column_stack([ar1(600, 0.995, _rng), ar1(600, 0.995, _rng)])
    _yd = (ar1(600, 0.95, _rng) > 0).astype(int)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    _a1.plot(_Xd[:, 0], lw=1, label="feature 0")
    _a1.plot(_Xd[:, 1], lw=1, label="feature 1")
    _lim = 1.1 * np.abs(_Xd).max()
    _a1.fill_between(np.arange(600), -_lim, _lim, where=_yd == 1, color="C2", alpha=0.15, label="label = 1")
    _a1.set(xlabel="day", ylim=(-_lim, _lim), title="one of the 10 series")
    _a1.legend(fontsize=8, loc="lower left")
    _a2.boxplot([_kf, _ts])
    _a2.set_xticks([1, 2], ["shuffled KFold", "TimeSeriesSplit"])
    _a2.axhline(0.5, color="gray", ls=":")
    _a2.set(ylabel="5-fold CV accuracy (10 series)", ylim=(0.0, 1.05))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"mean CV accuracy: shuffled KFold **{np.mean(_kf):.2f}**, "
                            f"TimeSeriesSplit **{np.mean(_ts):.2f}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** under shuffled KFold, which training rows are closest to a test day, in
    feature space *and* in time? why does that predict the label even though the features
    and the label are independent processes? TimeSeriesSplit's scores vary a lot between
    series: why is that honest, too?
    """)
    return


if __name__ == "__main__":
    app.run()
