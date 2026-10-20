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
    # bagging & random forests · build

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
    raise NotImplementedError("stage1: implement bootstrap_sample")


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
        raise NotImplementedError("stage2: implement RandomForestClassifier.fit")

    def _tree_proba(self, tree, X):
        """stage 2: one tree's probabilities as (n, k) columns over ALL classes (a bootstrap sample can miss a class)."""
        raise NotImplementedError("stage2: implement RandomForestClassifier._tree_proba")

    def predict_proba(self, X):
        """stage 2: (n, k): the average of the trees' class probabilities, columns in classes_ order."""
        raise NotImplementedError("stage2: implement RandomForestClassifier.predict_proba")

    def _compute_oob(self, X, codes):
        """stage 3: oob_decision_function_ (n, k) from the trees that didn't see each row (nan if none) and oob_score_."""
        raise NotImplementedError("stage3: implement RandomForestClassifier._compute_oob")

    @property
    def feature_importances_(self):
        """stage 4: the mean of the trees' impurity importances (d,), renormalized to sum 1."""
        raise NotImplementedError("stage4: implement RandomForestClassifier.feature_importances_")


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
        raise NotImplementedError("stage4: implement RandomForestRegressor.fit")

    def predict(self, X):
        """stage 4: the average of the trees' predictions, (n,)."""
        raise NotImplementedError("stage4: implement RandomForestRegressor.predict")

    @property
    def feature_importances_(self):
        """stage 4: the mean of the trees' impurity importances (d,), renormalized to sum 1."""
        raise NotImplementedError("stage4: implement RandomForestRegressor.feature_importances_")


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
