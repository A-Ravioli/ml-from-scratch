import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Classifier, Regressor, check_X_y  # yours, from ch 07
    from mlfs.trees import DecisionTreeClassifier, DecisionTreeRegressor  # yours, from topic 01
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bagging & random forests · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. your trees from topic 01 do all the
    splitting; this topic is about what happens when you grow hundreds of them on perturbed
    copies of the data and average.

    | stage | you build |
    |---|---|
    | 1 | `bootstrap_sample`: resample rows with replacement, and find the ones left out |
    | 2 | `RandomForestClassifier`: bootstrap + random feature subsets + averaged probabilities → `mlfs.trees` |
    | 3 | out-of-bag estimates: `oob_decision_function_`, `oob_score_` |
    | 4 | `RandomForestRegressor` with OOB $R^2$, and forest `feature_importances_` → `mlfs.trees` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def bootstrap_sample(n, rng=None):
    """stage 1: (idx, oob): n row indices drawn uniformly with replacement, and a bool mask (n,) of the rows never drawn."""
    gen = as_rng(rng)
    idx = gen.integers(0, n, size=n)
    oob = np.ones(n, dtype=bool)
    oob[idx] = False
    return idx, oob


@app.cell(hide_code=True)
def _():
    checks.run(1, bootstrap_sample=bootstrap_sample)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RandomForestClassifier(Classifier):
    """a random forest: bootstrapped CART trees with random feature subsets, probabilities averaged."""

    def __init__(self, n_estimators=100, *, criterion="gini", max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=True, oob_score=False, rng=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.rng = rng

    def fit(self, X, y):
        """stage 2: grow n_estimators trees on bootstrap samples of (X, codes of y); keep estimators_ and estimators_samples_; returns self."""
        X, y = check_X_y(X, y)
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True needs bootstrap=True: without resampling no row is out of bag")
        gen = as_rng(self.rng)
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.estimators_, self.estimators_samples_ = [], []
        for _ in range(self.n_estimators):
            idx = bootstrap_sample(len(y), gen)[0] if self.bootstrap else np.arange(len(y))
            tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features, rng=gen)
            tree.fit(X[idx], codes[idx])
            self.estimators_.append(tree)
            self.estimators_samples_.append(idx)
        if self.oob_score:
            self._compute_oob(X, codes)
        return self

    def _tree_proba(self, tree, X):
        """stage 2: one tree's probabilities as (n, k) columns over ALL classes (a bootstrap sample can miss a class)."""
        p = np.zeros((len(X), len(self.classes_)))
        p[:, tree.classes_.astype(int)] = tree.predict_proba(X)
        return p

    def predict_proba(self, X):
        """stage 2: (n, k): the average of the trees' class probabilities, columns in classes_ order."""
        X, _ = check_X_y(X)
        return sum(self._tree_proba(t, X) for t in self.estimators_) / len(self.estimators_)

    def _compute_oob(self, X, codes):
        """stage 3: oob_decision_function_ (n, k) from the trees that didn't see each row (nan if none) and oob_score_."""
        n, k = len(X), len(self.classes_)
        total, count = np.zeros((n, k)), np.zeros(n)
        for tree, idx in zip(self.estimators_, self.estimators_samples_):
            oob = np.ones(n, dtype=bool)
            oob[idx] = False
            if oob.any():
                total[oob] += self._tree_proba(tree, X[oob])
                count[oob] += 1
        seen = count > 0
        self.oob_decision_function_ = np.full((n, k), np.nan)
        self.oob_decision_function_[seen] = total[seen] / count[seen, None]
        pred = np.argmax(self.oob_decision_function_[seen], axis=1)
        self.oob_score_ = float(np.mean(pred == codes[seen])) if seen.any() else float("nan")

    @property
    def feature_importances_(self):
        """stage 4: the mean of the trees' impurity importances (d,), renormalized to sum 1."""
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        total = imp.sum()
        return imp / total if total > 0 else imp


@app.cell(hide_code=True)
def _():
    checks.run(2, RandomForestClassifier=RandomForestClassifier, bootstrap_sample=bootstrap_sample)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, RandomForestClassifier=RandomForestClassifier)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class RandomForestRegressor(Regressor):
    """a random forest for regression: bootstrapped CART regression trees, predictions averaged."""

    def __init__(self, n_estimators=100, *, max_depth=None, min_samples_leaf=1, max_features=1.0,
                 bootstrap=True, oob_score=False, rng=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.rng = rng

    def fit(self, X, y):
        """stage 4: grow the trees (like the classifier); if oob_score, set oob_prediction_ (n,) and oob_score_ (R^2); returns self."""
        X, y = check_X_y(X, y)
        y = y.astype(float)
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True needs bootstrap=True: without resampling no row is out of bag")
        gen = as_rng(self.rng)
        n = len(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_, self.estimators_samples_ = [], []
        for _ in range(self.n_estimators):
            idx = bootstrap_sample(n, gen)[0] if self.bootstrap else np.arange(n)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features, rng=gen)
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            self.estimators_samples_.append(idx)
        if self.oob_score:
            total, count = np.zeros(n), np.zeros(n)
            for tree, idx in zip(self.estimators_, self.estimators_samples_):
                oob = np.ones(n, dtype=bool)
                oob[idx] = False
                if oob.any():
                    total[oob] += tree.predict(X[oob])
                    count[oob] += 1
            seen = count > 0
            self.oob_prediction_ = np.full(n, np.nan)
            self.oob_prediction_[seen] = total[seen] / count[seen]
            resid = np.sum((y[seen] - self.oob_prediction_[seen]) ** 2)
            spread = np.sum((y[seen] - y[seen].mean()) ** 2)
            self.oob_score_ = float(1.0 - resid / spread) if spread > 0 else float("nan")
        return self

    def predict(self, X):
        """stage 4: the average of the trees' predictions, (n,)."""
        X, _ = check_X_y(X)
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)

    @property
    def feature_importances_(self):
        """stage 4: the mean of the trees' impurity importances (d,), renormalized to sum 1."""
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        total = imp.sum()
        return imp / total if total > 0 else imp


@app.cell(hide_code=True)
def _():
    checks.run(4, RandomForestRegressor=RandomForestRegressor, RandomForestClassifier=RandomForestClassifier)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `RandomForestClassifier` and `RandomForestRegressor` are part
    of `mlfs.trees`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
