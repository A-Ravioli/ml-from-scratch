import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, clone  # yours, from topic 01
    from mlfs.model_selection import KFold, cross_validate  # yours, from topic 02
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hyperparameter search · build

    work through the stages in order. each stage shows its part of the lesson, gives you a
    class or function to write, and checks it live as you edit. they complete
    `mlfs.model_selection`, using your `clone` (topic 01) and your `KFold` and
    `cross_validate` (topic 02); if those aren't finished, launch with
    `MLFS_REFERENCE=base,model_selection marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `ParameterGrid`: every combination of the listed values, in a fixed order → `mlfs.model_selection` |
    | 2 | `BaseSearchCV`, `GridSearchCV`: cross-validate every candidate, pick the best, refit it → `mlfs.model_selection` |
    | 3 | `RandomizedSearchCV`: the same, with candidates sampled at random → `mlfs.model_selection` |
    | 4 | `nested_cross_validate`: an honest estimate of a model *including* its tuning → `mlfs.model_selection` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class ParameterGrid:
    """the candidates of a grid search: every combination of the listed values.

    param_grid: a dict {name: list of values}, or a list of such dicts (a union of grids).
    """

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def grids(self):
        """(given) the grid as a list of dicts, with every value checked to be a non-empty
        list or tuple (a string or a single number is a common mistake: ValueError)."""
        grids = [self.param_grid] if isinstance(self.param_grid, dict) else list(self.param_grid)
        for grid in grids:
            for name, values in grid.items():
                if isinstance(values, str) or not hasattr(values, "__len__") or len(values) == 0:
                    raise ValueError(f"parameter {name!r} needs a non-empty list of values, got {values!r}")
        return grids

    def __iter__(self):
        """stage 1: yield one dict per combination.

        for each dict in grids() in order: sort its keys, and yield
        dict(zip(keys, combo)) for combo in itertools.product of the value lists in that key
        order (row-major: the LAST key varies fastest). an empty dict yields one empty dict {}.
        """
        raise NotImplementedError("stage1: implement ParameterGrid.__iter__")

    def __len__(self):
        """stage 1: the number of combinations: the sum over the dicts of the product of
        their list lengths (an empty dict counts 1)."""
        raise NotImplementedError("stage1: implement ParameterGrid.__len__")


@app.cell(hide_code=True)
def _():
    checks.run(1, ParameterGrid=ParameterGrid)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class BaseSearchCV(Estimator):
    """shared machinery of the searches: cross-validate each candidate setting, keep the best,
    refit it on all the data. subclasses define __init__ and candidate_params()."""

    def candidate_params(self):
        raise TypeError(f"{type(self).__name__} must define candidate_params()")

    def fit(self, X, y=None):
        """stage 2: run the search and return self.

        splits: computed ONCE from self.cv (an int k -> KFold(k); a splitter -> its
        split(X, y); else an iterable of (train, test) pairs), so every candidate is scored on
        the same folds. for each params dict in self.candidate_params(), in order:
        est = clone(self.estimator).set_params(**params), and
        cross_validate(est, X, y, cv=splits, scoring=self.scoring)["test_score"].
        set, as fitted attributes:
          cv_results_  {"params": [...], "mean_test_score": [...], "std_test_score": [...]}
                       (lists, one entry per candidate in order; std with ddof=0)
          best_index_  the index of the highest mean (ties: the first)
          best_params_, best_score_ (a float)
          best_estimator_ = clone(self.estimator).set_params(**best_params_).fit(X, y), only
                       if self.refit (and then classes_, if best_estimator_ has one)
        """
        raise NotImplementedError("stage2: implement BaseSearchCV.fit")

    def best(self):
        """(given) the refitted best estimator, or a clear error if there isn't one."""
        if not getattr(self, "refit", False) or not hasattr(self, "best_estimator_"):
            raise TypeError("predict/score need a search fitted with refit=True")
        return self.best_estimator_

    def predict(self, X):
        """stage 2: best_estimator_.predict(X) (see best())."""
        raise NotImplementedError("stage2: implement BaseSearchCV.predict")

    def predict_proba(self, X):
        """(given) best_estimator_.predict_proba(X)."""
        return self.best().predict_proba(X)

    def score(self, X, y=None):
        """stage 2: self.scoring(best_estimator_, X, y) if a scoring was given, else
        best_estimator_.score(X, y)."""
        raise NotImplementedError("stage2: implement BaseSearchCV.score")


@app.class_definition
class GridSearchCV(BaseSearchCV):
    """exhaustive search over a ParameterGrid."""

    def __init__(self, estimator, param_grid, *, cv=5, scoring=None, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def candidate_params(self):
        """stage 2: every combination of the grid, in ParameterGrid order (a list of dicts)."""
        raise NotImplementedError("stage2: implement GridSearchCV.candidate_params")


@app.cell(hide_code=True)
def _():
    checks.run(2, GridSearchCV=GridSearchCV)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RandomizedSearchCV(BaseSearchCV):
    """search over n_iter settings sampled at random from param_distributions."""

    def __init__(self, estimator, param_distributions, *, n_iter=20, cv=5, scoring=None, refit=True, rng=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.rng = rng

    def candidate_params(self):
        """stage 3: a list of n_iter dicts (n_iter >= 1, else ValueError).

        gen = as_rng(self.rng), made once. for each candidate, for each name in SORTED order:
        if the value is callable, call it as value(gen) (e.g. lambda g: 10 ** g.uniform(-3, 0));
        otherwise it's a list, and pick one element uniformly with gen.integers(len(list)).
        sampling is with replacement, so candidates may repeat.
        """
        raise NotImplementedError("stage3: implement RandomizedSearchCV.candidate_params")


@app.cell(hide_code=True)
def _():
    checks.run(3, RandomizedSearchCV=RandomizedSearchCV)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def nested_cross_validate(search, X, y, *, cv=5, scoring=None):
    """stage 4: the outer test scores (k,) of cross-validating the whole search.

    it's cross_validate(search, X, y, cv=cv, scoring=scoring)["test_score"]: each outer fold
    clones the search, which runs its own inner cross-validation on the outer training rows
    only, refits the winner, and is scored on the outer test rows. (the search must have
    refit=True, so it can predict.)
    """
    raise NotImplementedError("stage4: implement nested_cross_validate")


@app.cell(hide_code=True)
def _():
    checks.run(4, nested_cross_validate=nested_cross_validate, GridSearchCV=GridSearchCV)
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
