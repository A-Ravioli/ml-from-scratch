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
    # decision trees · build

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
    raise NotImplementedError("stage1: implement gini")


@app.function
def entropy(counts):
    """stage 1: entropy -sum_k p_k log2 p_k (with 0 log 0 = 0) of class counts (..., k) -> (...,), in bits."""
    raise NotImplementedError("stage1: implement entropy")


@app.function
def squared_error(sums):
    """stage 1: weighted variance of y from sums (..., 3) = (sum w, sum w y, sum w y^2) -> (...,); clipped to 0 below 1e-12 * mean square."""
    raise NotImplementedError("stage1: implement squared_error")


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
    raise NotImplementedError("stage2: implement best_split")


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
        raise NotImplementedError("stage3: implement BaseDecisionTree._grow")

    def apply(self, X):
        """stage 3: the index of the leaf each row of X lands in, int array (n,)."""
        raise NotImplementedError("stage3: implement BaseDecisionTree.apply")

    def get_depth(self):
        """stage 3: the number of edges on the longest root-to-leaf path (a lone root has depth 0)."""
        raise NotImplementedError("stage3: implement BaseDecisionTree.get_depth")

    def get_n_leaves(self):
        """stage 3: the number of leaves."""
        raise NotImplementedError("stage3: implement BaseDecisionTree.get_n_leaves")

    def _prune(self, alpha):
        """stage 5: minimal cost-complexity pruning: collapse weakest links while their alpha_eff <= alpha; returns self."""
        raise NotImplementedError("stage5: implement BaseDecisionTree._prune")

    @property
    def feature_importances_(self):
        """stage 5: impurity-based importances (d,): each feature's summed weighted impurity decrease, normalized to sum 1."""
        raise NotImplementedError("stage5: implement BaseDecisionTree.feature_importances_")


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
        raise NotImplementedError("stage3: implement DecisionTreeClassifier.fit")

    def predict_proba(self, X):
        """stage 3: (n, k) class probabilities: the weighted class fractions of each row's leaf."""
        raise NotImplementedError("stage3: implement DecisionTreeClassifier.predict_proba")


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
        raise NotImplementedError("stage4: implement DecisionTreeRegressor.fit")

    def predict(self, X):
        """stage 4: the weighted mean target of each row's leaf, (n,)."""
        raise NotImplementedError("stage4: implement DecisionTreeRegressor.predict")


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
