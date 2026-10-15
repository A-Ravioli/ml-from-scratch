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
    # splitting & cross-validation · build (reference solution)

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
    if not arrays:
        raise ValueError("train_test_split needs at least one array")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError(f"all arrays must have the same length, got {[len(a) for a in arrays]}")
    if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
        n_test = int(test_size)
    elif isinstance(test_size, (float, np.floating)) and 0.0 < test_size < 1.0:
        n_test = math.ceil(test_size * n)
    else:
        raise ValueError(f"test_size must be a fraction in (0, 1) or an int count, got {test_size!r}")
    if not 1 <= n_test <= n - 1:
        raise ValueError(f"test_size={test_size!r} gives {n_test} test rows out of {n}: need 1..{n - 1}")
    if stratify is not None:
        if not shuffle:
            raise ValueError("stratify requires shuffle=True")
        gen = as_rng(rng)
        classes, cls = np.unique(np.asarray(stratify), return_inverse=True)
        counts = np.bincount(cls, minlength=len(classes))
        ideal = n_test * counts / n
        take = np.floor(ideal).astype(int)
        extra = n_test - take.sum()  # hand the leftover rows to the largest remainders
        take[np.argsort(-(ideal - take), kind="stable")[:extra]] += 1
        test = np.concatenate([gen.permutation(np.flatnonzero(cls == c))[:t] for c, t in enumerate(take)])
        test = gen.permutation(test)
        mask = np.zeros(n, dtype=bool)
        mask[test] = True
        train = gen.permutation(np.flatnonzero(~mask))
    elif shuffle:
        perm = as_rng(rng).permutation(n)
        test, train = perm[:n_test], perm[n_test:]
    else:
        train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    out = []
    for a in arrays:
        a = np.asarray(a)
        out += [a[train], a[test]]
    return out


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
        n, k = len(X), self.n_splits
        if not 2 <= k <= n:
            raise ValueError(f"n_splits must be between 2 and the number of rows ({n}), got {k}")
        idx = as_rng(self.rng).permutation(n) if self.shuffle else np.arange(n)
        sizes = np.full(k, n // k)
        sizes[: n % k] += 1
        start = 0
        for size in sizes:
            test = np.sort(idx[start:start + size])
            mask = np.ones(n, dtype=bool)
            mask[test] = False
            yield np.flatnonzero(mask), test
            start += size

    def get_n_splits(self):
        """stage 2: the number of folds."""
        return self.n_splits


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
        if y is None:
            raise ValueError("StratifiedKFold needs the labels y")
        y = np.asarray(y)
        n, k = len(y), self.n_splits
        if not 2 <= k <= n:
            raise ValueError(f"n_splits must be between 2 and the number of rows ({n}), got {k}")
        _, cls = np.unique(y, return_inverse=True)
        idx = as_rng(self.rng).permutation(n) if self.shuffle else np.arange(n)
        order = idx[np.argsort(cls[idx], kind="stable")]
        fold = np.empty(n, dtype=int)
        fold[order] = np.arange(n) % k
        for f in range(k):
            yield np.flatnonzero(fold != f), np.flatnonzero(fold == f)

    def get_n_splits(self):
        """stage 3: the number of folds."""
        return self.n_splits


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
        if groups is None:
            raise ValueError("GroupKFold needs groups")
        _, g, sizes = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
        k = self.n_splits
        if k < 2 or len(sizes) < k:
            raise ValueError(f"need 2 <= n_splits <= number of groups ({len(sizes)}), got {k}")
        fold_rows = np.zeros(k, dtype=int)
        fold_of_group = np.empty(len(sizes), dtype=int)
        for grp in np.argsort(sizes, kind="stable")[::-1]:
            f = int(np.argmin(fold_rows))
            fold_of_group[grp] = f
            fold_rows[f] += sizes[grp]
        fold = fold_of_group[g]
        for f in range(k):
            yield np.flatnonzero(fold != f), np.flatnonzero(fold == f)

    def get_n_splits(self):
        """stage 3: the number of folds."""
        return self.n_splits


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
        n, k = len(X), self.n_splits
        if k < 2 or k + 1 > n:
            raise ValueError(f"n_splits must be at least 2 and less than the number of rows ({n}), got {k}")
        test_size = n // (k + 1)
        for start in range(n - k * test_size, n, test_size):
            yield np.arange(start), np.arange(start, start + test_size)

    def get_n_splits(self):
        """stage 4: the number of splits."""
        return self.n_splits


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
    if isinstance(scoring, str):
        raise TypeError(f"scoring must be None or a callable scorer(est, X, y); for {scoring!r} pass "
                        "mlfs.metrics.get_scorer(...)")
    if isinstance(cv, (int, np.integer)):
        cv = KFold(int(cv))
    splits = cv.split(X, y, groups) if hasattr(cv, "split") else cv
    X = np.asarray(X)
    y = None if y is None else np.asarray(y)

    def score(est, rows):
        Xs, ys = X[rows], None if y is None else y[rows]
        return float(scoring(est, Xs, ys) if scoring is not None else est.score(Xs, ys))

    test_scores, train_scores, fitted = [], [], []
    for train, test in splits:
        est = clone(estimator)
        est.fit(X[train], None if y is None else y[train])
        test_scores.append(score(est, test))
        if return_train_score:
            train_scores.append(score(est, train))
        if return_estimators:
            fitted.append(est)
    out = {"test_score": np.array(test_scores)}
    if return_train_score:
        out["train_score"] = np.array(train_scores)
    if return_estimators:
        out["estimators"] = fitted
    return out


@app.function
def cross_val_score(estimator, X, y=None, *, cv=5, scoring=None, groups=None):
    """stage 5: just the test scores: cross_validate(...)["test_score"]."""
    return cross_validate(estimator, X, y, cv=cv, scoring=scoring, groups=groups)["test_score"]


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
