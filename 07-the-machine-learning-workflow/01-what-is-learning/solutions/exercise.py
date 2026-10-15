import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import inspect

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # what is learning? · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    classes and functions to write, and checks them live as you edit. this notebook becomes
    `mlfs.base`: every classical model you write from chapter 08 to chapter 13 (linear models,
    kNN, SVMs, trees, boosting, k-means, Gaussian processes) subclasses the classes you build
    here, and every evaluation in the rest of the course calls them through `fit`, `predict`
    and `score`.

    | stage | you build |
    |---|---|
    | 1 | `check_X_y`: turn whatever the caller passed into a clean `(X, y)`, or fail loudly → `mlfs.base` |
    | 2 | `Estimator.get_params`, `Estimator.set_params`, `clone`: hyperparameters as data → `mlfs.base` |
    | 3 | `Regressor.score`, `Classifier.predict`, `Classifier.score`, `Transformer.fit_transform`: the shared behaviour → `mlfs.base` |
    | 4 | `DummyRegressor`, `DummyClassifier`: the baselines every number gets compared with → `mlfs.base` |
    | 5 | `LookupTableClassifier`, `StumpClassifier`: memorization versus empirical risk minimization |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def check_X_y(X, y=None):
    """stage 1: validate and standardize the inputs of fit/predict.

    X: anything np.asarray can turn into floats. a 1-D X (n,) is treated as ONE feature and
    becomes a column (n, 1); a 2-D X stays (n, d); any other number of dimensions, zero rows,
    or a NaN/inf entry raises ValueError. returns X as a float64 ndarray of shape (n, d).
    y: None (returned as None) or labels/targets. a column (n, 1) is flattened to (n,); any
    other 2-D shape raises ValueError, and so does len(y) != n. the dtype of y is kept (class
    labels can be strings), but a float y containing NaN or inf raises ValueError.
    returns (X, y).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D (n_samples, n_features), got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("X has no samples (0 rows)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or inf: impute or drop those entries first")
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D (n_samples,), got shape {y.shape}")
    if len(y) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} entries")
    if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or inf")
    return X, y


@app.cell(hide_code=True)
def _():
    checks.run(1, check_X_y=check_X_y)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class Estimator:
    """the base class of every model, transformer and search in mlfs.

    the protocol:
      * __init__ takes only hyperparameters and stores each one, unchanged, as an attribute of
        the same name. it validates nothing and learns nothing. (Estimator itself has no
        __init__: subclasses write their own.)
      * fit(X, y) learns from data and returns self. everything it learns is stored in
        attributes whose names end in "_" (coef_, classes_, ...), so "fitted state" and
        "settings" never mix.
      * get_params() / set_params(**p) read and write the hyperparameters, which is what lets
        clone, cross-validation and hyperparameter search work with ANY estimator.
    """

    def get_params(self):
        """stage 2: the hyperparameters, as {name: getattr(self, name)}.

        the names are the parameters of type(self).__init__ (read them with
        inspect.signature), skipping self and any *args / **kwargs. a class without its own
        __init__ has no hyperparameters: return {}. if __init__ forgot to store an argument,
        getattr fails: raise an AttributeError that says so.
        """
        sig = inspect.signature(type(self).__init__)
        names = [p.name for p in sig.parameters.values()
                 if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        params = {}
        for name in names:
            if not hasattr(self, name):
                raise AttributeError(
                    f"{type(self).__name__}.__init__ takes {name!r} but doesn't store it as self.{name}")
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        """stage 2: set hyperparameters by name and return self.

        set_params(alpha=0.1) does setattr(self, "alpha", 0.1). a name that isn't one of
        get_params()'s keys raises ValueError (a typo must not pass silently). a nested name
        "step__param" sets param on the sub-estimator stored in hyperparameter "step":
        getattr(self, "step").set_params(param=value).
        """
        valid = self.get_params()
        for key, value in params.items():
            head, sep, rest = key.partition("__")
            if head not in valid:
                raise ValueError(
                    f"{type(self).__name__} has no hyperparameter {head!r}; valid: {sorted(valid)}")
            if sep:
                getattr(self, head).set_params(**{rest: value})
            else:
                setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        raise NotImplementedError(f"{type(self).__name__}: subclasses implement fit")

    def predict(self, X):
        raise NotImplementedError(f"{type(self).__name__}: subclasses implement predict")

    def score(self, X, y=None):
        raise NotImplementedError(f"{type(self).__name__}: no default score (subclass Regressor or Classifier)")

    def __repr__(self):
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        except Exception:  # get_params not written yet: still printable
            args = "..."
        return f"{type(self).__name__}({args})"


@app.function
def clone(estimator):
    """stage 2: a new, UNFITTED estimator with the same hyperparameters.

    type(estimator)(**params) where params = estimator.get_params(), ignoring any
    "name__param" keys (they describe sub-estimators, not constructor arguments). each value is
    copied so the clone shares nothing with the original:
      * an estimator (anything with a get_params method that isn't a class) is cloned
        recursively,
      * a list or tuple is rebuilt with each element copied the same way (so the steps of a
        pipeline, [("scale", est), ...], are cloned too), and likewise a dict's values,
      * anything else is copy.deepcopy'd.
    """

    def copy_value(value):
        if hasattr(value, "get_params") and not isinstance(value, type):
            return clone(value)
        if isinstance(value, (list, tuple)):
            return type(value)(copy_value(v) for v in value)
        if isinstance(value, dict):
            return {k: copy_value(v) for k, v in value.items()}
        return copy.deepcopy(value)

    params = {k: copy_value(v) for k, v in estimator.get_params().items() if "__" not in k}
    return type(estimator)(**params)


@app.cell(hide_code=True)
def _():
    checks.run(2, Estimator=Estimator, clone=clone)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Regressor(Estimator):
    """base class for models that predict real numbers. subclasses write fit and predict."""

    def score(self, X, y):
        """stage 3: the coefficient of determination R^2 of self.predict(X) against y.

        R^2 = 1 - SS_res / SS_tot with SS_res = sum (y - y_pred)^2 and
        SS_tot = sum (y - mean(y))^2. 1 is perfect, 0 is "no better than predicting the mean
        of y", and it can be negative. if SS_tot == 0 (constant y), return 1.0 for a perfect
        prediction and 0.0 otherwise. returns a Python float.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = np.asarray(self.predict(X), dtype=np.float64).ravel()
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot


@app.class_definition
class Classifier(Estimator):
    """base class for classifiers. subclasses write fit (which must set classes_, the sorted
    distinct labels, e.g. np.unique(y)) and predict_proba, whose column j is the probability of
    classes_[j]. predict and score come for free."""

    def predict(self, X):
        """stage 3: the most probable class for each row: classes_[argmax predict_proba(X)].

        returns an array (n,) of labels (same dtype as classes_, so string labels work). ties
        go to the first column, i.e. the smallest label (np.argmax does this).
        """
        proba = np.asarray(self.predict_proba(X))
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__}: classifiers implement predict_proba")

    def score(self, X, y):
        """stage 3: accuracy, the fraction of rows where self.predict(X) == y (a float)."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


@app.class_definition
class Transformer(Estimator):
    """base class for estimators that turn X into a new X' (scalers, encoders, PCA, ...).
    subclasses write fit and transform (and inverse_transform if it exists)."""

    def transform(self, X):
        raise NotImplementedError(f"{type(self).__name__}: transformers implement transform")

    def fit_transform(self, X, y=None):
        """stage 3: fit on X (and y, which most transformers ignore), then transform X."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        raise NotImplementedError(f"{type(self).__name__} has no inverse_transform")


@app.cell(hide_code=True)
def _():
    checks.run(3, Estimator=Estimator, Regressor=Regressor, Classifier=Classifier, Transformer=Transformer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class DummyRegressor(Regressor):
    """predicts one constant, learned from y alone: the baseline every regressor must beat."""

    def __init__(self, strategy="mean"):
        self.strategy = strategy

    def fit(self, X, y):
        """stage 4: learn the constant from y and return self.

        strategy "mean" -> constant_ = mean(y), "median" -> constant_ = median(y), stored as a
        Python float; any other strategy raises ValueError. X is only used for its length:
        len(X) != len(y) raises ValueError. (it may hold anything, even strings.)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} entries")
        if self.strategy == "mean":
            self.constant_ = float(np.mean(y))
        elif self.strategy == "median":
            self.constant_ = float(np.median(y))
        else:
            raise ValueError(f"strategy must be 'mean' or 'median', got {self.strategy!r}")
        return self

    def predict(self, X):
        """stage 4: constant_ for every row of X: a float array (len(X),)."""
        return np.full(len(X), self.constant_)


@app.class_definition
class DummyClassifier(Classifier):
    """ignores X: predicts from the class frequencies in y alone."""

    def __init__(self, strategy="most_frequent", *, rng=None):
        self.strategy = strategy
        self.rng = rng

    def fit(self, X, y):
        """stage 4: learn the class frequencies and return self.

        sets classes_ (sorted distinct labels), class_prior_ (float (k,), the fraction of y
        equal to each class, in classes_ order) and rng_ = as_rng(self.rng) (from mlfs.utils,
        used by the random strategies). strategy must be one of "most_frequent", "prior",
        "uniform", "stratified", else ValueError; len(X) != len(y) raises ValueError.
        """
        if self.strategy not in ("most_frequent", "prior", "uniform", "stratified"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} entries")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.rng_ = as_rng(self.rng)
        return self

    def predict_proba(self, X):
        """stage 4: (len(X), k) class probabilities, columns in classes_ order.

          "most_frequent": a one-hot row on the most frequent class (ties: the smallest label)
          "prior":         class_prior_ in every row
          "uniform":       1/k everywhere
          "stratified":    a one-hot row per sample on a class DRAWN from class_prior_ with
                           rng_ (so predict, the argmax, is a random guess in proportion)
        """
        n, k = len(X), len(self.classes_)
        if self.strategy == "most_frequent":
            proba = np.zeros((n, k))
            proba[:, np.argmax(self.class_prior_)] = 1.0
        elif self.strategy == "prior":
            proba = np.tile(self.class_prior_, (n, 1))
        elif self.strategy == "uniform":
            proba = np.full((n, k), 1.0 / k)
        else:
            proba = np.zeros((n, k))
            proba[np.arange(n), self.rng_.choice(k, size=n, p=self.class_prior_)] = 1.0
        return proba

    def predict(self, X):
        """stage 4: for "uniform", a label drawn uniformly at random per row with rng_ (the
        argmax of a flat row would always pick classes_[0]); every other strategy uses
        Classifier.predict, the argmax of predict_proba."""
        if self.strategy == "uniform":
            return np.asarray(self.classes_)[self.rng_.integers(len(self.classes_), size=len(X))]
        return super().predict(X)


@app.cell(hide_code=True)
def _():
    checks.run(4, DummyRegressor=DummyRegressor, DummyClassifier=DummyClassifier)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class LookupTableClassifier(Classifier):
    """the memorizer: remembers the label of every training row it has seen."""

    def fit(self, X, y):
        """stage 5: build the lookup table and return self.

        X, y = check_X_y(X, y); classes_ = sorted distinct labels. table_ is a dict mapping
        tuple(row) -> the index (into classes_) of that row's label; if the same row appears
        with different labels, keep its most common label (ties: the smallest).
        default_ = the index of the most frequent class overall (ties: the smallest), used for
        rows that aren't in the table.
        """
        X, y = check_X_y(X, y)
        self.classes_, idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        counts = {}
        for row, c in zip(map(tuple, X), idx):
            counts.setdefault(row, np.zeros(k, dtype=int))[c] += 1
        self.table_ = {row: int(np.argmax(cnt)) for row, cnt in counts.items()}
        self.default_ = int(np.argmax(np.bincount(idx, minlength=k)))
        return self

    def predict_proba(self, X):
        """stage 5: a one-hot row per sample, on table_[tuple(row)] if the row was seen in
        training, else on default_. returns (n, k)."""
        X, _ = check_X_y(X)
        rows = [self.table_.get(tuple(r), self.default_) for r in X]
        proba = np.zeros((len(X), len(self.classes_)))
        proba[np.arange(len(X)), rows] = 1.0
        return proba


@app.class_definition
class StumpClassifier(Classifier):
    """a decision stump: one feature, one threshold. predicts classes_[1] where
    sign_ * (x[feature_] - threshold_) > 0, and classes_[0] elsewhere."""

    def fit(self, X, y):
        """stage 5: empirical risk minimization over every stump, for two classes.

        X, y = check_X_y(X, y); classes_ = sorted distinct labels (exactly 2, else ValueError).
        candidates, for each feature j: thresholds -inf and the midpoints between consecutive
        DISTINCT sorted values of X[:, j], each with sign +1 and -1. pick the candidate with
        the fewest training mistakes (ties: smallest j, then smallest threshold, then sign +1)
        and set feature_ (int), threshold_ (float), sign_ (+1 or -1) and train_error_ (the
        fraction of training rows it gets wrong). an O(n log n) scan per feature, not a loop
        over every candidate on every row.
        """
        X, y = check_X_y(X, y)
        self.classes_, idx = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"StumpClassifier needs exactly 2 classes, got {len(self.classes_)}")
        n, d = X.shape
        is_pos = idx == 1
        best = None  # (mistakes, feature, threshold, -sign): the smallest tuple wins every tie
        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs, pos = X[order, j], is_pos[order]
            # split i puts the i smallest values on the left (i = 0: everything on the right)
            pos_left = np.concatenate([[0], np.cumsum(pos)])
            neg_left = np.arange(n + 1) - pos_left
            splits = np.concatenate([[0], np.flatnonzero(xs[1:] > xs[:-1]) + 1])
            thresholds = np.concatenate([[-np.inf], (xs[splits[1:] - 1] + xs[splits[1:]]) / 2])
            err_up = pos_left[splits] + (neg_left[-1] - neg_left[splits])  # sign +1: right is class 1
            err_down = neg_left[splits] + (pos_left[-1] - pos_left[splits])  # sign -1: left is class 1
            for errs, sign in ((err_up, 1), (err_down, -1)):
                i = int(np.argmin(errs))  # the first (smallest) threshold among the best
                cand = (int(errs[i]), j, float(thresholds[i]), -sign)
                if best is None or cand < best:
                    best = cand
        mistakes, self.feature_, self.threshold_, neg_sign = best
        self.sign_ = -neg_sign
        self.train_error_ = mistakes / n
        return self

    def predict_proba(self, X):
        """stage 5: one-hot rows (n, 2): column 1 where sign_ * (x[feature_] - threshold_) > 0."""
        X, _ = check_X_y(X)
        right = self.sign_ * (X[:, self.feature_] - self.threshold_) > 0
        return np.column_stack([~right, right]).astype(float)


@app.cell(hide_code=True)
def _():
    checks.run(5, LookupTableClassifier=LookupTableClassifier, StumpClassifier=StumpClassifier)
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
