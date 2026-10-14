import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import clone  # yours, from topic 01
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # splitting & cross-validation · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions and classes to write, and checks them live as you edit. they join
    `mlfs.model_selection`, which every evaluation in the rest of the course goes through.
    stage 5 uses your `clone` from topic 01 through `mlfs.base` (if topic 01 isn't finished,
    launch with `MLFS_REFERENCE=base marimo edit exercise.py` to borrow the reference).

    | stage | you build |
    |---|---|
    | 1 | `train_test_split`: one random (optionally stratified) holdout split → `mlfs.model_selection` |
    | 2 | `KFold`: $k$ contiguous or shuffled folds → `mlfs.model_selection` |
    | 3 | `StratifiedKFold`, `GroupKFold`: folds that respect class balance and groups → `mlfs.model_selection` |
    | 4 | `TimeSeriesSplit`: folds that never train on the future → `mlfs.model_selection` |
    | 5 | `cross_validate`, `cross_val_score`: fit a fresh clone per fold and score it → `mlfs.model_selection` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None, rng=None):
    """stage 1: split equal-length arrays into a training part and a test part.

    arrays: one or more arrays (or lists) with the same first dimension n, split by the SAME
    rows. test_size: a float in (0, 1) is a fraction (n_test = ceil(test_size * n)); an int is
    a count. the result must leave 1 <= n_test <= n - 1, else ValueError (also for no arrays or
    unequal lengths).
      shuffle=False: the last n_test rows are the test set, in order (no randomness).
      shuffle=True:  a random permutation from as_rng(rng); its first n_test entries are the
                     test rows.
      stratify=labels (n,): each class contributes round(n_test * n_c / n) test rows, using
                     the largest-remainder rule so the counts add up to n_test exactly (ties:
                     classes in sorted order), chosen at random within the class.
                     requires shuffle=True (ValueError otherwise).
    returns a list [a_train, a_test, b_train, b_test, ...] of ndarrays.
    """
    raise NotImplementedError("stage1: implement train_test_split")


@app.cell(hide_code=True)
def _():
    checks.run(1, train_test_split=train_test_split)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class KFold:
    """k-fold cross-validation: split the n rows into k folds; each fold is the test set once.

    unshuffled folds are contiguous blocks, and the first n % k folds are one row larger. with
    shuffle=True the rows are permuted first (as_rng(rng)); an int rng gives the same folds on
    every call to split.
    """

    def __init__(self, n_splits=5, *, shuffle=False, rng=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rng = rng

    def split(self, X, y=None, groups=None):
        """stage 2: yield (train_idx, test_idx) int arrays, one pair per fold, in fold order.

        n = len(X); 2 <= n_splits <= n, else ValueError. fold f's test rows are block f of
        np.arange(n) (or of a random permutation of it, if shuffle), with block sizes
        n // k + 1 for the first n % k blocks and n // k for the rest. both arrays are sorted;
        train_idx is every row not in test_idx. y and groups are ignored.
        """
        raise NotImplementedError("stage2: implement KFold.split")

    def get_n_splits(self):
        """stage 2: the number of folds."""
        raise NotImplementedError("stage2: implement KFold.get_n_splits")


@app.cell(hide_code=True)
def _():
    checks.run(2, KFold=KFold)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class StratifiedKFold:
    """k folds that each keep (almost exactly) the class proportions of y."""

    def __init__(self, n_splits=5, *, shuffle=False, rng=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rng = rng

    def split(self, X, y=None, groups=None):
        """stage 3: yield (train_idx, test_idx) with every class spread evenly over the folds.

        y is required (ValueError if None); 2 <= n_splits <= n, else ValueError. deal the rows
        out like cards: order them by class (stable, so within a class they keep their order;
        with shuffle=True, permute the rows with as_rng(rng) first), then row number i of that
        ordering goes to fold i % k. each class's count then differs by at most 1 between
        folds, and so do the fold sizes. both arrays sorted.
        """
        raise NotImplementedError("stage3: implement StratifiedKFold.split")

    def get_n_splits(self):
        """stage 3: the number of folds."""
        raise NotImplementedError("stage3: implement StratifiedKFold.get_n_splits")


@app.class_definition
class GroupKFold:
    """k folds over GROUPS (patients, users, sessions): a group is never split between train
    and test. no shuffling: groups are assigned greedily to balance fold sizes."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """stage 3: yield (train_idx, test_idx) where test_idx is every row of some groups.

        groups (n,) is required (ValueError if None), and there must be at least n_splits
        distinct groups (else ValueError). assignment: visit the distinct groups LARGEST first,
        in the order np.argsort(sizes, kind="stable")[::-1] (sizes in np.unique order; this is
        scikit-learn's order, so equal sizes go later-group first), and put each one into the
        fold with the fewest rows so far (ties: the lowest fold number). then fold f's test
        set is every row whose group went to fold f. both arrays sorted.
        """
        raise NotImplementedError("stage3: implement GroupKFold.split")

    def get_n_splits(self):
        """stage 3: the number of folds."""
        raise NotImplementedError("stage3: implement GroupKFold.get_n_splits")


@app.cell(hide_code=True)
def _():
    checks.run(3, StratifiedKFold=StratifiedKFold, GroupKFold=GroupKFold)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class TimeSeriesSplit:
    """expanding-window splits for ordered data: always train on the past, test on the next
    block."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """stage 4: yield n_splits pairs (train_idx, test_idx) over rows 0..n-1 in time order.

        test_size = n // (n_splits + 1), and n_splits + 1 > n raises ValueError (as does
        n_splits < 2). the test blocks are the last n_splits * test_size rows, cut into
        n_splits consecutive blocks; for a test block starting at row s, train_idx is
        np.arange(s) (everything before it) and test_idx is np.arange(s, s + test_size).
        """
        raise NotImplementedError("stage4: implement TimeSeriesSplit.split")

    def get_n_splits(self):
        """stage 4: the number of splits."""
        raise NotImplementedError("stage4: implement TimeSeriesSplit.get_n_splits")


@app.cell(hide_code=True)
def _():
    checks.run(4, TimeSeriesSplit=TimeSeriesSplit)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def cross_validate(estimator, X, y=None, *, cv=5, scoring=None, groups=None, return_train_score=False,
                   return_estimators=False):
    """stage 5: estimate how well `estimator` generalizes by k-fold cross-validation.

    cv: an int k (-> KFold(k), unshuffled), a splitter (anything with a split(X, y, groups)
    method), or an iterable of (train_idx, test_idx) pairs used as given.
    for each split: est = clone(estimator) (from mlfs.base; NEVER fit `estimator` itself),
    est.fit(X[train], y[train]) (y[train] is None when y is None), then the test score is
    scoring(est, X[test], y[test]) if scoring is a callable, or est.score(X[test], y[test]) if
    scoring is None. X and y are indexed as ndarrays (np.asarray(X)[idx]).
    returns {"test_score": float array (k,)}, plus "train_score" (k,) (the same scorer on the
    training rows) if return_train_score, and "estimators" (list of the k fitted clones) if
    return_estimators. a string scoring raises TypeError (use mlfs.metrics.get_scorer(name)).
    """
    raise NotImplementedError("stage5: implement cross_validate")


@app.function
def cross_val_score(estimator, X, y=None, *, cv=5, scoring=None, groups=None):
    """stage 5: just the test scores: cross_validate(...)["test_score"]."""
    raise NotImplementedError("stage5: implement cross_val_score")


@app.cell(hide_code=True)
def _():
    checks.run(5, cross_validate=cross_validate, cross_val_score=cross_val_score, KFold=KFold,
               GroupKFold=GroupKFold)
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
