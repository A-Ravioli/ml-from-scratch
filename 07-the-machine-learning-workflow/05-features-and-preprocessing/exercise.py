import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, Transformer, check_X_y  # yours, from topic 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # features & preprocessing · build

    work through the stages in order. each stage shows its part of the lesson, gives you a
    class to write, and checks it live as you edit. every class here is a `Transformer` from
    your `mlfs.base` (topic 01; if it isn't finished, launch with
    `MLFS_REFERENCE=base marimo edit exercise.py`), and together they become
    `mlfs.preprocessing`.

    | stage | you build |
    |---|---|
    | 1 | `StandardScaler`, `MinMaxScaler`: put features on comparable scales → `mlfs.preprocessing` |
    | 2 | `OneHotEncoder`: categories to indicator columns → `mlfs.preprocessing` |
    | 3 | `SimpleImputer`: fill in missing values → `mlfs.preprocessing` |
    | 4 | `PolynomialFeatures`: products of features, for nonlinear fits with linear models → `mlfs.preprocessing` |
    | 5 | `Pipeline`, `make_pipeline`: chain transformers and a model into one estimator → `mlfs.preprocessing` |
    | 6 | `Pipeline.get_params`, `Pipeline.set_params`: tune any step's hyperparameters as `"step__param"` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class StandardScaler(Transformer):
    """standardize each feature: z = (x - mean) / standard deviation, with statistics learned
    from the training data in fit."""

    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        """stage 1: learn mean_ and scale_ (float arrays (d,)) from X and return self.

        X = check_X_y(X)[0]. mean_ = the column means if with_mean, else zeros; scale_ = the
        column standard deviations (population, ddof=0) if with_std, else ones. a CONSTANT
        column (max == min: test it exactly like that, because rounding can make the computed
        std of a constant column 1e-17 instead of 0) gets scale 1, so it maps to 0, never NaN
        or 1e16.
        """
        raise NotImplementedError("stage1: implement StandardScaler.fit")

    def transform(self, X):
        """stage 1: (X - mean_) / scale_, with the statistics from fit. a different number of
        columns than in fit raises ValueError."""
        raise NotImplementedError("stage1: implement StandardScaler.transform")

    def inverse_transform(self, X):
        """stage 1: undo transform: X * scale_ + mean_."""
        raise NotImplementedError("stage1: implement StandardScaler.inverse_transform")


@app.class_definition
class MinMaxScaler(Transformer):
    """rescale each feature linearly so its training minimum and maximum map to feature_range."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """stage 1: learn data_min_ and data_max_ (column min and max, float (d,)); return self.
        feature_range (lo, hi) must have lo < hi, else ValueError."""
        raise NotImplementedError("stage1: implement MinMaxScaler.fit")

    def transform(self, X):
        """stage 1: lo + (X - data_min_) / (data_max_ - data_min_) * (hi - lo). a constant
        column (max == min) uses a range of 1, so it maps to lo. new data may fall outside
        [lo, hi]: that's fine, don't clip. a different number of columns raises ValueError."""
        raise NotImplementedError("stage1: implement MinMaxScaler.transform")

    def inverse_transform(self, X):
        """stage 1: undo transform."""
        raise NotImplementedError("stage1: implement MinMaxScaler.inverse_transform")


@app.cell(hide_code=True)
def _():
    checks.run(1, StandardScaler=StandardScaler, MinMaxScaler=MinMaxScaler)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class OneHotEncoder(Transformer):
    """one indicator column per category of each input column."""

    def __init__(self, *, handle_unknown="error", drop=None):
        self.handle_unknown = handle_unknown
        self.drop = drop

    def fit(self, X, y=None):
        """stage 2: learn categories_, a list with one sorted array of distinct values per column.

        X: a 2-D array (n, d) of categories, strings or numbers (np.asarray(X); a 1-D X raises
        ValueError: pass a column of shape (n, 1)). handle_unknown must be "error" or "ignore"
        and drop must be None or "first", else ValueError. returns self.
        """
        raise NotImplementedError("stage2: implement OneHotEncoder.fit")

    def transform(self, X):
        """stage 2: a float array (n, sum of k_j): for each input column j, k_j indicator
        columns in categories_[j] order, with a 1 in the column of the row's category.

        drop="first": leave out each block's first column (a row in the first category is all
        zeros in that block), so there are k_j - 1 columns per input column.
        a category not seen in fit: ValueError if handle_unknown == "error", else ("ignore") an
        all-zero block for that row. a different number of columns raises ValueError.
        """
        raise NotImplementedError("stage2: implement OneHotEncoder.transform")

    def inverse_transform(self, X):
        """stage 2: back to categories: an object array (n, d). in each block, the category
        of the column holding the largest value; with drop="first", an all-zero block is the
        first category; without drop, an all-zero block (an ignored unknown) becomes None."""
        raise NotImplementedError("stage2: implement OneHotEncoder.inverse_transform")


@app.cell(hide_code=True)
def _():
    checks.run(2, OneHotEncoder=OneHotEncoder)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class SimpleImputer(Transformer):
    """replace missing values (NaN) in each column with one statistic learned in fit."""

    def __init__(self, strategy="mean", *, fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """stage 3: learn statistics_ (float (d,)), the value to fill in for each column.

        X = np.asarray(X, dtype=float), 2-D (else ValueError); NaN marks a missing entry.
          "mean" / "median":  the column mean / median of the non-missing values
          "most_frequent":    the most common non-missing value (ties: the smallest)
          "constant":         fill_value, or 0.0 if fill_value is None
        a column with no observed values raises ValueError (except for "constant"); an
        unknown strategy raises ValueError. returns self.
        """
        raise NotImplementedError("stage3: implement SimpleImputer.fit")

    def transform(self, X):
        """stage 3: a float COPY of X with every NaN in column j replaced by statistics_[j]
        (X itself is not modified). a different number of columns raises ValueError."""
        raise NotImplementedError("stage3: implement SimpleImputer.transform")


@app.cell(hide_code=True)
def _():
    checks.run(3, SimpleImputer=SimpleImputer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class PolynomialFeatures(Transformer):
    """all products of the input features up to a total degree."""

    def __init__(self, degree=2, *, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        """stage 4: decide the output columns and return self.

        X = check_X_y(X)[0], d columns. powers_ is an int array (n_output, d): row i holds the
        exponent of each input feature in output column i. the columns are every monomial of
        total degree 0 (the bias column of ones, only if include_bias), then 1, 2, ...,
        degree; within one degree, in the order itertools.combinations_with_replacement(range(d),
        k) lists the features (itertools.combinations if interaction_only, which allows no
        repeated feature, i.e. exponents 0 or 1). this is scikit-learn's order: for d = 2,
        degree 2 it's 1, x0, x1, x0^2, x0 x1, x1^2.
        """
        raise NotImplementedError("stage4: implement PolynomialFeatures.fit")

    def transform(self, X):
        """stage 4: the (n, n_output) array whose column i is prod_j X[:, j] ** powers_[i, j].
        a different number of columns than in fit raises ValueError."""
        raise NotImplementedError("stage4: implement PolynomialFeatures.transform")


@app.cell(hide_code=True)
def _():
    checks.run(4, PolynomialFeatures=PolynomialFeatures)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class Pipeline(Estimator):
    """a chain of steps [(name, estimator), ...]: every step but the last is a Transformer;
    the last can be anything (a model, or another transformer). the pipeline is itself an
    Estimator, so cross_validate and searches treat it as one model, and every learned step
    is refitted inside each training fold."""

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        """(given) {name: estimator} for the steps."""
        return dict(self.steps)

    @property
    def classes_(self):
        """(given) the final step's classes_ (so scorers that need classes_ work on pipelines)."""
        return self.steps[-1][1].classes_

    def transform_before_last(self, X):
        """(given) X passed through the transform of every step except the last."""
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def fit(self, X, y=None):
        """stage 5: fit the steps in order and return self.

        the step names must be distinct (else ValueError) and every step but the last must
        have fit and transform methods (else TypeError). then Xt = X; for each step but the
        last: Xt = step.fit_transform(Xt, y); finally fit the last step on (Xt, y).
        """
        raise NotImplementedError("stage5: implement Pipeline.fit")

    def predict(self, X):
        """stage 5: the last step's predict on the transformed X."""
        raise NotImplementedError("stage5: implement Pipeline.predict")

    def predict_proba(self, X):
        """stage 5: the last step's predict_proba on the transformed X."""
        raise NotImplementedError("stage5: implement Pipeline.predict_proba")

    def decision_function(self, X):
        """(given) the last step's decision_function on the transformed X."""
        return self.steps[-1][1].decision_function(self.transform_before_last(X))

    def transform(self, X):
        """stage 5: X through the transform of EVERY step, the last one included (for a
        pipeline made only of transformers)."""
        raise NotImplementedError("stage5: implement Pipeline.transform")

    def fit_transform(self, X, y=None):
        """(given) fit, then transform: lets a pipeline of transformers be a step itself."""
        return self.fit(X, y).transform(X)

    def score(self, X, y=None):
        """stage 5: the last step's score on the transformed X (and y)."""
        raise NotImplementedError("stage5: implement Pipeline.score")

    def get_params(self):
        """stage 6: {"steps": self.steps} plus, for every step and every one of ITS
        parameters, "name__param": value (step.get_params()). nested pipelines nest
        naturally: "outer__inner__param"."""
        raise NotImplementedError("stage6: implement Pipeline.get_params")

    def set_params(self, **params):
        """stage 6: set parameters by name and return self.

          "steps":        replace the list of steps
          "name":         replace that step's estimator (keeping its position)
          "name__param":  named_steps[name].set_params(param=value) (param may itself
                          contain "__", for nested estimators)
        a name that isn't a step raises ValueError. to replace a step, assign a NEW list to
        self.steps rather than mutating the old one (the caller may still hold that list).
        """
        raise NotImplementedError("stage6: implement Pipeline.set_params")


@app.function
def make_pipeline(*steps):
    """stage 5: Pipeline with automatic names: each step is named by its lowercased class
    name (StandardScaler() -> "standardscaler"). if a name occurs more than once, every
    occurrence gets a suffix "-1", "-2", ... in order (like scikit-learn)."""
    raise NotImplementedError("stage5: implement make_pipeline")


@app.cell(hide_code=True)
def _():
    checks.run(5, Pipeline=Pipeline, make_pipeline=make_pipeline, StandardScaler=StandardScaler)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.cell(hide_code=True)
def _():
    checks.run(6, Pipeline=Pipeline, make_pipeline=make_pipeline, StandardScaler=StandardScaler)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
