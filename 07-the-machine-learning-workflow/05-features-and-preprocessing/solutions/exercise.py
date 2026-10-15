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
    # features & preprocessing · build (reference solution)

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
        X, _ = check_X_y(X)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        constant = X.max(axis=0) == X.min(axis=0)
        self.scale_ = np.where(constant, 1.0, X.std(axis=0)) if self.with_std else np.ones(X.shape[1])
        return self

    def transform(self, X):
        """stage 1: (X - mean_) / scale_, with the statistics from fit. a different number of
        columns than in fit raises ValueError."""
        X, _ = check_X_y(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError(f"fitted on {len(self.mean_)} features, got {X.shape[1]}")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        """stage 1: undo transform: X * scale_ + mean_."""
        X, _ = check_X_y(X)
        return X * self.scale_ + self.mean_


@app.class_definition
class MinMaxScaler(Transformer):
    """rescale each feature linearly so its training minimum and maximum map to feature_range."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """stage 1: learn data_min_ and data_max_ (column min and max, float (d,)); return self.
        feature_range (lo, hi) must have lo < hi, else ValueError."""
        lo, hi = self.feature_range
        if not lo < hi:
            raise ValueError(f"feature_range must be (lo, hi) with lo < hi, got {self.feature_range}")
        X, _ = check_X_y(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X):
        """stage 1: lo + (X - data_min_) / (data_max_ - data_min_) * (hi - lo). a constant
        column (max == min) uses a range of 1, so it maps to lo. new data may fall outside
        [lo, hi]: that's fine, don't clip. a different number of columns raises ValueError."""
        X, _ = check_X_y(X)
        if X.shape[1] != len(self.data_min_):
            raise ValueError(f"fitted on {len(self.data_min_)} features, got {X.shape[1]}")
        lo, hi = self.feature_range
        rng = self.data_max_ - self.data_min_
        rng = np.where(rng == 0, 1.0, rng)
        return lo + (X - self.data_min_) / rng * (hi - lo)

    def inverse_transform(self, X):
        """stage 1: undo transform."""
        X, _ = check_X_y(X)
        lo, hi = self.feature_range
        rng = self.data_max_ - self.data_min_
        rng = np.where(rng == 0, 1.0, rng)
        return (X - lo) / (hi - lo) * rng + self.data_min_


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
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {self.handle_unknown!r}")
        if self.drop not in (None, "first"):
            raise ValueError(f"drop must be None or 'first', got {self.drop!r}")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, d); for one categorical column use shape (n, 1), got {X.shape}")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        """stage 2: a float array (n, sum of k_j): for each input column j, k_j indicator
        columns in categories_[j] order, with a 1 in the column of the row's category.

        drop="first": leave out each block's first column (a row in the first category is all
        zeros in that block), so there are k_j - 1 columns per input column.
        a category not seen in fit: ValueError if handle_unknown == "error", else ("ignore") an
        all-zero block for that row. a different number of columns raises ValueError.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.categories_):
            raise ValueError(f"X must be 2-D with {len(self.categories_)} columns, got shape {X.shape}")
        blocks = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            idx = np.minimum(np.searchsorted(cats, col), len(cats) - 1)
            known = cats[idx] == col
            if not np.all(known) and self.handle_unknown == "error":
                raise ValueError(f"column {j} has categories not seen in fit: {np.unique(col[~known]).tolist()}")
            block = np.zeros((len(X), len(cats)))
            block[np.flatnonzero(known), idx[known]] = 1.0
            blocks.append(block[:, 1:] if self.drop == "first" else block)
        return np.hstack(blocks)

    def inverse_transform(self, X):
        """stage 2: back to categories: an object array (n, d). in each block, the category
        of the column holding the largest value; with drop="first", an all-zero block is the
        first category; without drop, an all-zero block (an ignored unknown) becomes None."""
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.categories_)), dtype=object)
        start = 0
        for j, cats in enumerate(self.categories_):
            width = len(cats) - (1 if self.drop == "first" else 0)
            block = X[:, start:start + width]
            start += width
            empty = ~np.any(block != 0, axis=1)
            if self.drop == "first":
                labels = cats[np.argmax(block, axis=1) + 1] if width else np.full(len(X), cats[0], dtype=object)
                out[:, j] = np.where(empty, cats[0], labels)
            else:
                out[:, j] = cats[np.argmax(block, axis=1)]
                out[empty, j] = None
        return out


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
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], 0.0 if self.fill_value is None else float(self.fill_value))
            return self
        stats = []
        for j in range(X.shape[1]):
            seen = X[~np.isnan(X[:, j]), j]
            if len(seen) == 0:
                raise ValueError(f"column {j} has no observed values to learn a {self.strategy} from")
            if self.strategy == "mean":
                stats.append(seen.mean())
            elif self.strategy == "median":
                stats.append(np.median(seen))
            else:
                values, counts = np.unique(seen, return_counts=True)
                stats.append(values[np.argmax(counts)])
        self.statistics_ = np.array(stats, dtype=float)
        return self

    def transform(self, X):
        """stage 3: a float COPY of X with every NaN in column j replaced by statistics_[j]
        (X itself is not modified). a different number of columns raises ValueError."""
        X = np.array(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.statistics_):
            raise ValueError(f"X must be 2-D with {len(self.statistics_)} columns, got shape {X.shape}")
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = self.statistics_[cols]
        return X


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
        X, _ = check_X_y(X)
        d = X.shape[1]
        combos = itertools.combinations if self.interaction_only else itertools.combinations_with_replacement
        rows = []
        for k in range(0 if self.include_bias else 1, self.degree + 1):
            for c in combos(range(d), k):
                rows.append(np.bincount(np.array(c, dtype=int), minlength=d))
        self.powers_ = np.array(rows, dtype=int).reshape(-1, d)
        return self

    def transform(self, X):
        """stage 4: the (n, n_output) array whose column i is prod_j X[:, j] ** powers_[i, j].
        a different number of columns than in fit raises ValueError."""
        X, _ = check_X_y(X)
        if X.shape[1] != self.powers_.shape[1]:
            raise ValueError(f"fitted on {self.powers_.shape[1]} features, got {X.shape[1]}")
        return np.prod(X[:, None, :] ** self.powers_[None, :, :], axis=2)


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
        names = [name for name, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {names}")
        for name, step in self.steps[:-1]:
            if not (hasattr(step, "fit") and hasattr(step, "transform")):
                raise TypeError(f"step {name!r} must be a transformer (fit + transform), got {type(step).__name__}")
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y)
        self.steps[-1][1].fit(Xt, y)
        return self

    def predict(self, X):
        """stage 5: the last step's predict on the transformed X."""
        return self.steps[-1][1].predict(self.transform_before_last(X))

    def predict_proba(self, X):
        """stage 5: the last step's predict_proba on the transformed X."""
        return self.steps[-1][1].predict_proba(self.transform_before_last(X))

    def decision_function(self, X):
        """(given) the last step's decision_function on the transformed X."""
        return self.steps[-1][1].decision_function(self.transform_before_last(X))

    def transform(self, X):
        """stage 5: X through the transform of EVERY step, the last one included (for a
        pipeline made only of transformers)."""
        return self.steps[-1][1].transform(self.transform_before_last(X))

    def fit_transform(self, X, y=None):
        """(given) fit, then transform: lets a pipeline of transformers be a step itself."""
        return self.fit(X, y).transform(X)

    def score(self, X, y=None):
        """stage 5: the last step's score on the transformed X (and y)."""
        return self.steps[-1][1].score(self.transform_before_last(X), y)

    def get_params(self):
        """stage 6: {"steps": self.steps} plus, for every step and every one of ITS
        parameters, "name__param": value (step.get_params()). nested pipelines nest
        naturally: "outer__inner__param"."""
        params = {"steps": self.steps}
        for name, step in self.steps:
            for key, value in step.get_params().items():
                params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        """stage 6: set parameters by name and return self.

          "steps":        replace the list of steps
          "name":         replace that step's estimator (keeping its position)
          "name__param":  named_steps[name].set_params(param=value) (param may itself
                          contain "__", for nested estimators)
        a name that isn't a step raises ValueError. to replace a step, assign a NEW list to
        self.steps rather than mutating the old one (the caller may still hold that list).
        """
        for key, value in params.items():
            if key == "steps":
                self.steps = value
                continue
            name, sep, rest = key.partition("__")
            names = [n for n, _ in self.steps]
            if name not in names:
                raise ValueError(f"pipeline has no step {name!r}; steps: {names}")
            if sep:
                self.named_steps[name].set_params(**{rest: value})
            else:
                self.steps = [(n, value if n == name else s) for n, s in self.steps]
        return self


@app.function
def make_pipeline(*steps):
    """stage 5: Pipeline with automatic names: each step is named by its lowercased class
    name (StandardScaler() -> "standardscaler"). if a name occurs more than once, every
    occurrence gets a suffix "-1", "-2", ... in order (like scikit-learn)."""
    names = [type(s).__name__.lower() for s in steps]
    counts = {n: names.count(n) for n in names}
    seen = {}
    named = []
    for name, step in zip(names, steps):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name}-{seen[name]}"
        named.append((name, step))
    return Pipeline(named)


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
