import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, Regressor, check_X_y  # yours, from ch 07
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gradient boosting · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. you'll build an XGBoost-style
    booster: features binned once into small integers, gradient/hessian histograms built with
    one `bincount`, second-order split gains, and shrinkage.

    | stage | you build |
    |---|---|
    | 1 | `quantile_bin_edges`, `bin_data`: every feature as at most `n_bins` integer codes |
    | 2 | `gradient_histograms`, `split_gain`, `leaf_weight`, `best_hist_split`: the second-order split search |
    | 3 | `HistGradientTree`: a depth-limited tree on (codes, gradients, hessians), with histogram subtraction |
    | 4 | `GradientBoostingRegressor`: squared loss, shrinkage, row subsampling → `mlfs.trees` |
    | 5 | `GradientBoostingClassifier`: binary log loss → `mlfs.trees` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def quantile_bin_edges(X, n_bins=256):
    """stage 1: per-feature sorted bin edges (a list of d arrays, each of length < n_bins): midpoints if few distinct values, else unique quantiles."""
    if not 2 <= n_bins <= 256:
        raise ValueError(f"n_bins must be between 2 and 256 (codes are uint8), got {n_bins}")
    X = np.asarray(X, dtype=float)
    edges = []
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        if len(vals) <= n_bins:
            e = (vals[:-1] + vals[1:]) / 2.0          # one bin per distinct value
        else:
            qs = np.quantile(X[:, j], np.arange(1, n_bins) / n_bins)
            e = np.unique(qs)                            # repeated quantiles collapse
        edges.append(e)
    return edges


@app.function
def bin_data(X, edges):
    """stage 1: integer codes (n, d), uint8: code = number of edges strictly below x, so code <= b exactly when x <= edges[b]."""
    X = np.asarray(X, dtype=float)
    codes = np.empty(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        codes[:, j] = np.searchsorted(e, X[:, j], side="left")
    return codes


@app.cell(hide_code=True)
def _():
    checks.run(1, quantile_bin_edges=quantile_bin_edges, bin_data=bin_data)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gradient_histograms(codes, g, h, n_bins):
    """stage 2: (G, H), each (d, n_bins): the sums of g and of h over the rows in each (feature, bin), from one bincount each."""
    n, d = codes.shape
    flat = (codes.astype(np.intp) + np.arange(d) * n_bins).ravel()   # row-major: (row, feature)
    G = np.bincount(flat, weights=np.repeat(g, d), minlength=d * n_bins).reshape(d, n_bins)
    H = np.bincount(flat, weights=np.repeat(h, d), minlength=d * n_bins).reshape(d, n_bins)
    return G, H


@app.function
def split_gain(G_L, H_L, G_R, H_R, reg_lambda=1.0, gamma=0.0):
    """stage 2: 1/2 [G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam)] - gamma, elementwise."""
    G, H = G_L + G_R, H_L + H_R
    return 0.5 * (G_L ** 2 / (H_L + reg_lambda) + G_R ** 2 / (H_R + reg_lambda) - G ** 2 / (H + reg_lambda)) - gamma


@app.function
def leaf_weight(G, H, reg_lambda=1.0):
    """stage 2: the optimal leaf value -G / (H + lambda)."""
    return -G / (H + reg_lambda)


@app.function
def best_hist_split(G_hist, H_hist, *, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
    """stage 2: best "code <= b" split from (d, n_bins) histograms -> (feature, bin, gain); (-1, -1, -inf) if no child-weight-valid split."""
    GL = np.cumsum(G_hist, axis=1)[:, :-1]      # left child = bins 0..b
    HL = np.cumsum(H_hist, axis=1)[:, :-1]
    G, H = G_hist[0].sum(), H_hist[0].sum()     # every feature's bins add up to the node totals
    GR, HR = G - GL, H - HL
    with np.errstate(divide="ignore", invalid="ignore"):  # empty children (masked below) may divide by 0
        gain = split_gain(GL, HL, GR, HR, reg_lambda, gamma)
    ok = (HL >= min_child_weight) & (HR >= min_child_weight) & (HL > 1e-12) & (HR > 1e-12)  # no empty child
    gain = np.where(ok, gain, -np.inf)
    if gain.size == 0 or not np.isfinite(gain).any():
        return -1, -1, -np.inf
    f, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
    return int(f), int(b), float(gain[f, b])


@app.cell(hide_code=True)
def _():
    checks.run(2, gradient_histograms=gradient_histograms, split_gain=split_gain, leaf_weight=leaf_weight,
               best_hist_split=best_hist_split)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class HistGradientTree:
    """one second-order regression tree on binned features: splits "code <= bin", leaves -G/(H + lambda)."""

    def __init__(self, max_depth=3, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0, n_bins=256):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins

    def fit(self, codes, g, h):
        """stage 3: grow depth-first (explicit stack), splitting while depth < max_depth and the best gain > 0; returns self."""
        codes = np.asarray(codes)
        g, h = np.asarray(g, dtype=float), np.asarray(h, dtype=float)
        feature, bin_, left, right, value = [], [], [], [], []
        root = gradient_histograms(codes, g, h, self.n_bins)
        stack = [(np.arange(len(g)), 0, -1, False, root)]
        while stack:
            rows, depth, parent, is_left, (Gh, Hh) = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            G, H = Gh[0].sum(), Hh[0].sum()
            feature.append(-1)
            bin_.append(-1)
            left.append(-1)
            right.append(-1)
            value.append(leaf_weight(G, H, self.reg_lambda))
            if depth >= self.max_depth or len(rows) < 2:
                continue
            f, b, gain = best_hist_split(Gh, Hh, reg_lambda=self.reg_lambda, gamma=self.gamma,
                                         min_child_weight=self.min_child_weight)
            if f < 0 or gain <= 0:
                continue
            feature[node], bin_[node] = f, b
            go_left = codes[rows, f] <= b
            rows_l, rows_r = rows[go_left], rows[~go_left]
            # histogram subtraction: build the smaller child's histograms, get the other as parent - small
            small = rows_l if len(rows_l) <= len(rows_r) else rows_r
            Gs, Hs = gradient_histograms(codes[small], g[small], h[small], self.n_bins)
            other = (Gh - Gs, Hh - Hs)
            hist_l, hist_r = ((Gs, Hs), other) if small is rows_l else (other, (Gs, Hs))
            stack.append((rows_r, depth + 1, node, False, hist_r))
            stack.append((rows_l, depth + 1, node, True, hist_l))
        self.feature_ = np.array(feature, dtype=int)
        self.bin_ = np.array(bin_, dtype=int)
        self.children_left_ = np.array(left, dtype=int)
        self.children_right_ = np.array(right, dtype=int)
        self.value_ = np.array(value, dtype=float)
        return self

    def predict(self, codes):
        """stage 3: the leaf value of every row of codes (n, d), shape (n,)."""
        codes = np.asarray(codes)
        node = np.zeros(len(codes), dtype=int)
        rows = np.arange(len(codes))
        for _ in range(self.max_depth):
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                break
            go_left = codes[rows, np.maximum(f, 0)] <= self.bin_[node]
            node = np.where(inner, np.where(go_left, self.children_left_[node], self.children_right_[node]), node)
        return self.value_[node]


@app.cell(hide_code=True)
def _():
    checks.run(3, HistGradientTree=HistGradientTree, bin_data=bin_data, quantile_bin_edges=quantile_bin_edges)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def subsample_rows(n, fraction, rng=None):
    """stage 4: sorted row indices for the next tree: all n if fraction >= 1, else round(fraction * n) (at least 1) drawn WITHOUT replacement."""
    if fraction >= 1.0:
        return np.arange(n)
    return np.sort(as_rng(rng).choice(n, size=max(1, int(round(fraction * n))), replace=False))


@app.class_definition
class GradientBoostingRegressor(Regressor):
    """gradient boosting for squared loss on binned features (XGBoost-style second-order trees)."""

    def __init__(self, n_estimators=100, *, learning_rate=0.1, max_depth=3, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, n_bins=256, subsample=1.0, rng=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.rng = rng

    def fit(self, X, y):
        """stage 4: bin X, start at the mean, then n_estimators trees on g = F - y, h = 1, each added with learning_rate; returns self."""
        X, y = check_X_y(X, y)
        y = y.astype(float)
        gen = as_rng(self.rng)
        self.bin_edges_ = quantile_bin_edges(X, self.n_bins)
        codes = bin_data(X, self.bin_edges_)
        self.init_ = float(y.mean())
        F = np.full(len(y), self.init_)
        h = np.ones(len(y))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            g = F - y
            rows = subsample_rows(len(y), self.subsample, gen)
            tree = HistGradientTree(self.max_depth, self.min_child_weight, self.reg_lambda, self.gamma,
                                    self.n_bins).fit(codes[rows], g[rows], h[rows])
            F += self.learning_rate * tree.predict(codes)
            self.estimators_.append(tree)
        return self

    def staged_predict(self, X):
        """stage 4: a generator yielding the prediction (n,) after 1, 2, ..., n_estimators trees."""
        X, _ = check_X_y(X)
        codes = bin_data(X, self.bin_edges_)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(codes)
            yield F

    def predict(self, X):
        """stage 4: init_ + learning_rate * the sum of the trees' predictions, (n,)."""
        X, _ = check_X_y(X)
        codes = bin_data(X, self.bin_edges_)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(codes)
        return F


@app.cell(hide_code=True)
def _():
    checks.run(4, GradientBoostingRegressor=GradientBoostingRegressor, subsample_rows=subsample_rows, HistGradientTree=HistGradientTree, bin_data=bin_data, quantile_bin_edges=quantile_bin_edges)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class GradientBoostingClassifier(Classifier):
    """binary gradient boosting with the log loss: the trees add up to the log-odds."""

    def __init__(self, n_estimators=100, *, learning_rate=0.1, max_depth=3, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, n_bins=256, subsample=1.0, rng=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.rng = rng

    def fit(self, X, y):
        """stage 5: init_ = log-odds of the class-1 rate; trees on g = p - y, h = p (1 - p) with p = sigmoid(F); returns self."""
        X, y = check_X_y(X, y)
        self.classes_, codes_y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"GradientBoostingClassifier is binary, got {len(self.classes_)} classes")
        t = codes_y.astype(float)
        gen = as_rng(self.rng)
        self.bin_edges_ = quantile_bin_edges(X, self.n_bins)
        codes = bin_data(X, self.bin_edges_)
        p0 = np.clip(t.mean(), 1e-12, 1 - 1e-12)
        self.init_ = float(np.log(p0 / (1 - p0)))
        F = np.full(len(t), self.init_)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            p = 0.5 * (1.0 + np.tanh(F / 2.0))
            g, h = p - t, p * (1.0 - p)
            rows = subsample_rows(len(t), self.subsample, gen)
            tree = HistGradientTree(self.max_depth, self.min_child_weight, self.reg_lambda, self.gamma,
                                    self.n_bins).fit(codes[rows], g[rows], h[rows])
            F += self.learning_rate * tree.predict(codes)
            self.estimators_.append(tree)
        return self

    def staged_decision_function(self, X):
        """stage 5: a generator yielding the log-odds (n,) after 1, 2, ..., n_estimators trees."""
        X, _ = check_X_y(X)
        codes = bin_data(X, self.bin_edges_)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(codes)
            yield F

    def decision_function(self, X):
        """stage 5: the raw score F(x) = init_ + learning_rate * sum of trees, the log-odds of classes_[1], (n,)."""
        X, _ = check_X_y(X)
        codes = bin_data(X, self.bin_edges_)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(codes)
        return F

    def predict_proba(self, X):
        """stage 5: (n, 2) with P(classes_[1]) = sigmoid(decision_function)."""
        p1 = 0.5 * (1.0 + np.tanh(self.decision_function(X) / 2.0))
        return np.column_stack([1.0 - p1, p1])


@app.cell(hide_code=True)
def _():
    checks.run(5, GradientBoostingClassifier=GradientBoostingClassifier, bin_data=bin_data)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `GradientBoostingRegressor` and `GradientBoostingClassifier`
    are part of `mlfs.trees`, and the capstone races them against scikit-learn's. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
