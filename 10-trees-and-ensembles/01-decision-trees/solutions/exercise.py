import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, Estimator, Regressor, check_X_y  # yours, from ch 07
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # decision trees · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you have CART: the
    tree learner behind random forests, AdaBoost and (in a second-order form) gradient
    boosting.

    | stage | you build |
    |---|---|
    | 1 | `gini`, `entropy`, `squared_error`: how impure a node is, from summed statistics |
    | 2 | `best_split`: the best threshold over every feature, vectorized |
    | 3 | `BaseDecisionTree` (growing, `apply`, depth, leaves) and `DecisionTreeClassifier` → `mlfs.trees` |
    | 4 | `DecisionTreeRegressor` → `mlfs.trees` |
    | 5 | cost-complexity pruning (`ccp_alpha`) and `feature_importances_` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def gini(counts):
    """stage 1: Gini impurity 1 - sum_k p_k^2 of weighted class counts (..., k) -> (...,); 0 where the total is 0."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 1.0 - np.sum(counts * counts, axis=-1) / (total * total)
    return np.where(total > 0, g, 0.0)


@app.function
def entropy(counts):
    """stage 1: entropy -sum_k p_k log2 p_k (with 0 log 0 = 0) of class counts (..., k) -> (...,), in bits."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1)
    safe = np.where(total > 0, total, 1.0)
    p = counts / safe[..., None]
    logp = np.log2(np.where(p > 0, p, 1.0))
    return -np.sum(p * logp, axis=-1)


@app.function
def squared_error(sums):
    """stage 1: weighted variance of y from sums (..., 3) = (sum w, sum w y, sum w y^2) -> (...,); clipped to 0 below 1e-12 * mean square."""
    sums = np.asarray(sums, dtype=float)
    w = sums[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sq = sums[..., 2] / w
        var = mean_sq - (sums[..., 1] / w) ** 2
    return np.where(var > 1e-12 * mean_sq, var, 0.0)  # nan (empty node) compares False -> 0


@app.cell(hide_code=True)
def _():
    checks.run(1, gini=gini, entropy=entropy, squared_error=squared_error)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def best_split(X, stats, w, impurity, *, min_samples_leaf=1, features=None):
    """stage 2: best split of one node -> (feature, threshold, decrease); (-1, nan, -inf) if none is valid.

    X (n, d); stats (n, m) per-sample statistics; w (n,) weights; impurity: gini, entropy or squared_error.
    decrease = W imp(node) - W_L imp(left) - W_R imp(right), W = summed weights; left is x <= threshold.
    """
    X = np.asarray(X, dtype=float)
    stats = np.asarray(stats, dtype=float)
    w = np.asarray(w, dtype=float)
    n = X.shape[0]
    feats = np.arange(X.shape[1]) if features is None else np.asarray(features, dtype=int)
    if n < 2 or len(feats) == 0:
        return -1, float("nan"), -np.inf
    Xf = X[:, feats]                                    # (n, f)
    order = np.argsort(Xf, axis=0, kind="stable")       # sort every feature at once
    xs = Xf[order, np.arange(len(feats))]               # sorted values (n, f)
    left = np.cumsum(stats[order], axis=0)[:-1]         # (n-1, f, m): sums of the first i+1 rows
    total = stats.sum(axis=0)
    right = total - left
    w_left = np.cumsum(w[order], axis=0)[:-1]           # (n-1, f)
    w_total = w.sum()
    w_right = w_total - w_left
    imp_left, imp_right = impurity(np.stack([left, right]))  # one call scores both children
    decrease = w_total * impurity(total) - w_left * imp_left - w_right * imp_right
    n_left = np.arange(1, n)[:, None]                   # rows on the left of each cut
    valid = (xs[1:] > xs[:-1]) & (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
    if not valid.any():
        return -1, float("nan"), -np.inf
    decrease = np.where(valid, decrease, -np.inf)
    flat = int(np.argmax(decrease.T))                   # feature-major: ties -> lowest feature, then position
    j, i = divmod(flat, n - 1)
    lo, hi = xs[i, j], xs[i + 1, j]
    threshold = lo / 2.0 + hi / 2.0
    if threshold >= hi:  # rounding put the midpoint on the upper value
        threshold = lo
    return int(feats[j]), float(threshold), float(decrease[i, j])


@app.cell(hide_code=True)
def _():
    checks.run(2, best_split=best_split, gini=gini, entropy=entropy, squared_error=squared_error)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class BaseDecisionTree(Estimator):
    """the CART machinery shared by DecisionTreeClassifier and DecisionTreeRegressor.

    a fitted tree is a set of parallel node arrays, node 0 the root:
      feature_ (int, -1 at leaves), threshold_ (float), children_left_ / children_right_ (int,
      -1 at leaves), value_ (n_nodes, m) summed statistics, impurity_, n_node_samples_ (int),
      weighted_n_node_samples_ (float). a row goes left when x[feature] <= threshold.
    """

    def _n_features_to_try(self, d):
        """given: how many of the d features each split searches (max_features None|int|float|'sqrt'|'log2')."""
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(d)))
            if mf == "log2":
                return max(1, int(np.log2(d)))
            raise ValueError(f"max_features must be None, an int, a float, 'sqrt' or 'log2', got {mf!r}")
        if isinstance(mf, (float, np.floating)):
            return max(1, min(d, int(mf * d)))
        return max(1, min(d, int(mf)))

    def _grow(self, X, stats, w, impurity):
        """stage 3: grow the tree depth-first with an explicit stack, filling the node arrays; returns self."""
        n, d = X.shape
        max_depth = np.inf if self.max_depth is None else self.max_depth
        k = self._n_features_to_try(d)
        gen = as_rng(self.rng)
        eps = np.finfo(float).eps
        feature, threshold, left, right, value, imp, count, weight = ([] for _ in range(8))
        stack = [(np.arange(n), 0, -1, False)]  # (rows, depth, parent, is_left_child)
        while stack:
            rows, depth, parent, is_left = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = node
            s = stats[rows].sum(axis=0)
            node_imp = float(impurity(s))
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(s)
            imp.append(node_imp)
            count.append(len(rows))
            weight.append(float(w[rows].sum()))
            if (depth >= max_depth or len(rows) < self.min_samples_split
                    or len(rows) < 2 * self.min_samples_leaf or node_imp <= eps):
                continue
            Xn, sn, wn = X[rows], stats[rows], w[rows]
            if k < d:  # random feature subset; if none of them can split, keep looking (like sklearn)
                order = gen.permutation(d)
                f, t, _ = best_split(Xn, sn, wn, impurity, min_samples_leaf=self.min_samples_leaf,
                                     features=order[:k])
                if f < 0:
                    f, t, _ = best_split(Xn, sn, wn, impurity, min_samples_leaf=self.min_samples_leaf,
                                         features=order[k:])
            else:
                f, t, _ = best_split(Xn, sn, wn, impurity, min_samples_leaf=self.min_samples_leaf)
            if f < 0:
                continue
            feature[node], threshold[node] = f, t
            go_left = Xn[:, f] <= t
            stack.append((rows[~go_left], depth + 1, node, False))  # pushed first, popped second
            stack.append((rows[go_left], depth + 1, node, True))
        self.feature_ = np.array(feature, dtype=int)
        self.threshold_ = np.array(threshold, dtype=float)
        self.children_left_ = np.array(left, dtype=int)
        self.children_right_ = np.array(right, dtype=int)
        self.value_ = np.array(value, dtype=float)
        self.impurity_ = np.array(imp, dtype=float)
        self.n_node_samples_ = np.array(count, dtype=int)
        self.weighted_n_node_samples_ = np.array(weight, dtype=float)
        self.n_features_in_ = d
        return self

    def apply(self, X):
        """stage 3: the index of the leaf each row of X lands in, int array (n,)."""
        X, _ = check_X_y(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the tree was fitted with {self.n_features_in_}")
        node = np.zeros(len(X), dtype=int)
        rows = np.arange(len(X))
        for _ in range(len(self.feature_)):  # at most one step per level; a cap, not a count
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                break
            go_left = X[rows, np.maximum(f, 0)] <= self.threshold_[node]
            node = np.where(inner, np.where(go_left, self.children_left_[node], self.children_right_[node]), node)
        return node

    def get_depth(self):
        """stage 3: the number of edges on the longest root-to-leaf path (a lone root has depth 0)."""
        depth = np.zeros(len(self.feature_), dtype=int)
        for node in range(len(self.feature_)):  # parents come before children in the node arrays
            for child in (self.children_left_[node], self.children_right_[node]):
                if child >= 0:
                    depth[child] = depth[node] + 1
        return int(depth.max())

    def get_n_leaves(self):
        """stage 3: the number of leaves."""
        return int(np.sum(self.children_left_ < 0))

    def _prune(self, alpha):
        """stage 5: minimal cost-complexity pruning: collapse weakest links while their alpha_eff <= alpha; returns self."""
        n_nodes = len(self.feature_)
        parent = np.full(n_nodes, -1)
        for node in range(n_nodes):
            for child in (self.children_left_[node], self.children_right_[node]):
                if child >= 0:
                    parent[child] = node
        r_node = self.weighted_n_node_samples_ / self.weighted_n_node_samples_[0] * self.impurity_
        r_branch = r_node.copy()                   # R(T_t): summed over the subtree's leaves
        n_leaves = np.ones(n_nodes, dtype=int)     # |leaves(T_t)|
        for node in range(n_nodes - 1, -1, -1):    # children before parents
            lc, rc = self.children_left_[node], self.children_right_[node]
            if lc >= 0:
                r_branch[node] = r_branch[lc] + r_branch[rc]
                n_leaves[node] = n_leaves[lc] + n_leaves[rc]
        alive = np.ones(n_nodes, dtype=bool)
        inner = self.children_left_ >= 0
        for _ in range(n_nodes):  # each pass removes one internal node
            cand = alive & inner
            if not cand.any():
                break
            g = np.full(n_nodes, np.inf)
            g[cand] = (r_node[cand] - r_branch[cand]) / (n_leaves[cand] - 1)
            t = int(np.argmin(g))  # weakest link (ties: the lowest index, like sklearn)
            if g[t] > alpha:
                break
            d_r, d_l = r_node[t] - r_branch[t], 1 - n_leaves[t]
            a = t
            while a >= 0:  # the subtree's change propagates to every ancestor
                r_branch[a] += d_r
                n_leaves[a] += d_l
                a = parent[a]
            inner[t] = False
            stack = [self.children_left_[t], self.children_right_[t]]
            while stack:  # everything below t disappears
                c = stack.pop()
                alive[c] = False
                if self.children_left_[c] >= 0:
                    stack += [self.children_left_[c], self.children_right_[c]]
        # keep the surviving nodes, renumbered in their original (parent-first) order
        keep = np.flatnonzero(alive)
        new_id = np.full(n_nodes, -1)
        new_id[keep] = np.arange(len(keep))
        lc = np.where(inner[keep], self.children_left_[keep], -1)
        rc = np.where(inner[keep], self.children_right_[keep], -1)
        self.children_left_ = np.where(lc >= 0, new_id[np.maximum(lc, 0)], -1)
        self.children_right_ = np.where(rc >= 0, new_id[np.maximum(rc, 0)], -1)
        self.feature_ = np.where(inner[keep], self.feature_[keep], -1)
        self.threshold_ = np.where(inner[keep], self.threshold_[keep], np.nan)
        self.value_ = self.value_[keep]
        self.impurity_ = self.impurity_[keep]
        self.n_node_samples_ = self.n_node_samples_[keep]
        self.weighted_n_node_samples_ = self.weighted_n_node_samples_[keep]
        return self

    @property
    def feature_importances_(self):
        """stage 5: impurity-based importances (d,): each feature's summed weighted impurity decrease, normalized to sum 1."""
        W, imp = self.weighted_n_node_samples_, self.impurity_
        inner = np.flatnonzero(self.children_left_ >= 0)
        lc, rc = self.children_left_[inner], self.children_right_[inner]
        dec = W[inner] * imp[inner] - W[lc] * imp[lc] - W[rc] * imp[rc]
        out = np.bincount(self.feature_[inner], weights=dec, minlength=self.n_features_in_)
        total = out.sum()
        return out / total if total > 0 else out


@app.class_definition
class DecisionTreeClassifier(BaseDecisionTree, Classifier):
    """CART classification tree (Gini or entropy splits)."""

    def __init__(self, *, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, ccp_alpha=0.0, rng=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.rng = rng

    def fit(self, X, y, sample_weight=None):
        """stage 3: classes_, then grow on stats = w * one-hot(y) with gini or entropy; prune if ccp_alpha > 0; returns self."""
        X, y = check_X_y(X, y)
        impurity = {"gini": gini, "entropy": entropy}.get(self.criterion)
        if impurity is None:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {self.criterion!r}")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        stats = np.zeros((len(y), len(self.classes_)))
        stats[np.arange(len(y)), codes] = w
        self._grow(X, stats, w, impurity)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
        return self

    def predict_proba(self, X):
        """stage 3: (n, k) class probabilities: the weighted class fractions of each row's leaf."""
        counts = self.value_[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)


@app.cell(hide_code=True)
def _():
    checks.run(3, DecisionTreeClassifier=DecisionTreeClassifier, best_split=best_split, gini=gini)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class DecisionTreeRegressor(BaseDecisionTree, Regressor):
    """CART regression tree (squared-error splits, leaf = weighted mean)."""

    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
                 ccp_alpha=0.0, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.rng = rng

    def fit(self, X, y, sample_weight=None):
        """stage 4: grow on stats = (w, w z, w z^2) with z = y - weighted mean of y; prune if ccp_alpha > 0; returns self."""
        X, y = check_X_y(X, y)
        y = y.astype(float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.y_offset_ = float(np.sum(w * y) / np.sum(w))
        z = y - self.y_offset_  # centering keeps the variance formula accurate
        stats = np.column_stack([w, w * z, w * z * z])
        self._grow(X, stats, w, squared_error)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
        return self

    def predict(self, X):
        """stage 4: the weighted mean target of each row's leaf, (n,)."""
        v = self.value_[self.apply(X)]
        return self.y_offset_ + v[:, 1] / v[:, 0]


@app.cell(hide_code=True)
def _():
    checks.run(4, DecisionTreeRegressor=DecisionTreeRegressor, squared_error=squared_error)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, DecisionTreeClassifier=DecisionTreeClassifier, DecisionTreeRegressor=DecisionTreeRegressor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `DecisionTreeClassifier` and `DecisionTreeRegressor` are part
    of `mlfs.trees`: random forests (topic 02) and AdaBoost (topic 03) are built out of them.
    open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
