import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.linalg import svd  # yours, from topic 06

    # small CPU config (used offline, by tests and verify) vs full config (real Olivetti faces)
    OFFLINE = datasets.is_offline()
    K_GRID = [1, 2, 5, 10, 20, 40, 60] if OFFLINE else [1, 2, 5, 10, 20, 40, 80, 120, 200]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 · capstone: eigenfaces (reference solution)

    read [`brief.md`](brief.md) first. you'll build face recognition from nothing but the
    linear algebra of this chapter: PCA through **your** `mlfs.linalg.svd`, projections, and a
    nearest-neighbour rule. the data loading, the train/test split, and the evaluation are
    written for you. the four functions below are yours.

    | stage | you build |
    |---|---|
    | 1 | `fit_pca`: mean face, eigenfaces, explained variances |
    | 2 | `project`, `reconstruct`: into face space and back |
    | 3 | `nearest_neighbor`: 1-NN in face space |
    | 4 | `eigenface_classifier`: the whole pipeline |

    when they're green, scroll down to run the experiments and write up what you find.
    """)
    return


@app.function
def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stage 1: PCA of the rows of X (n, d) with k components, via mlfs.linalg.svd.

    returns (mean (d,), components (k, d), explained_variance (k,)):
      mean: the average row; components: the top-k right singular vectors of the CENTERED
      data, one per row (orthonormal); explained_variance: s_i^2 / (n - 1), descending.
    raises ValueError unless 1 <= k <= min(n, d).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not 1 <= k <= min(n, d):
        raise ValueError(f"k must be between 1 and {min(n, d)}, got {k}")
    mean = X.mean(axis=0)
    _, s, Vt = svd(X - mean)
    return mean, Vt[:k], s[:k] ** 2 / (n - 1)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", fit_pca=fit_pca)
    return


@app.function
def project(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """stage 2: coordinates of the rows of X (n, d) in face space: (X - mean) @ components.T, shape (n, k)."""
    return (np.asarray(X, dtype=float) - mean) @ components.T


@app.function
def reconstruct(Z: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """stage 2: images (n, d) rebuilt from face-space coordinates Z (n, k): mean + Z @ components."""
    return mean + np.asarray(Z, dtype=float) @ components


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", fit_pca=fit_pca, project=project, reconstruct=reconstruct)
    return


@app.function
def nearest_neighbor(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray) -> np.ndarray:
    """stage 3: 1-nearest-neighbour labels for the rows of Z_test (m, k), using Euclidean
    distance to the rows of Z_train (n, k). returns y_train[nearest index] for each test row,
    shape (m,). vectorize: no Python loop over test points.
    """
    Z_train = np.asarray(Z_train, dtype=float)
    Z_test = np.asarray(Z_test, dtype=float)
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, for all pairs at once
    d2 = (np.sum(Z_test**2, axis=1)[:, None] - 2 * Z_test @ Z_train.T
          + np.sum(Z_train**2, axis=1)[None, :])
    return np.asarray(y_train)[np.argmin(d2, axis=1)]


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", nearest_neighbor=nearest_neighbor)
    return


@app.function
def eigenface_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    """stage 4: predict a label for each row of X_test (m, d): fit PCA with k components on
    X_train (n, d) ONLY, project both sets, and classify by 1-nearest-neighbour. returns (m,).
    """
    mean, components, _ = fit_pca(X_train, k)
    return nearest_neighbor(project(X_train, mean, components), y_train, project(X_test, mean, components))


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", eigenface_classifier=eigenface_classifier)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    the Olivetti faces: 400 grayscale images, 64 × 64 pixels, 10 per person for 40 people.
    offline, `mlfs.datasets` substitutes a small synthetic stand-in with the same format, so
    everything below still runs. the split keeps 7 images per person for training (fewer if a
    person has fewer images) and tests on the rest.
    """)
    return


@app.cell
def _():
    images, labels = datasets.olivetti_faces()
    side = images.shape[1]
    X_all = images.reshape(len(images), -1)
    _train = np.concatenate([np.flatnonzero(labels == c)[:max(1, int(0.7 * np.sum(labels == c)))]
                             for c in np.unique(labels)])
    _test = np.setdiff1d(np.arange(len(labels)), _train)
    X_train, y_train = X_all[_train], labels[_train]
    X_test, y_test = X_all[_test], labels[_test]
    mo.md(f"{'**synthetic stand-in (offline)**' if OFFLINE else '**real Olivetti faces**'}: "
          f"{len(X_all)} images of {side}×{side} = {X_all.shape[1]} pixels, "
          f"{len(np.unique(labels))} people · train {len(X_train)} · test {len(X_test)}")
    return X_test, X_train, side, y_test, y_train


@app.cell
def _(X_train, side):
    # one PCA with every component we'll need, fitted once and sliced for each k
    pca_mean, pca_components, pca_variance = fit_pca(X_train, min(max(K_GRID), *X_train.shape))
    _fig, _axes = plt.subplots(1, 9, figsize=(13, 1.9))
    _axes[0].imshow(pca_mean.reshape(side, side), cmap="gray")
    _axes[0].set_title("mean face", fontsize=8)
    for _i, _ax in enumerate(_axes[1:]):
        _ax.imshow(pca_components[_i].reshape(side, side), cmap="RdBu_r")
        _ax.set_title(f"eigenface {_i + 1}", fontsize=8)
    for _ax in _axes:
        _ax.axis("off")
    _fig
    return pca_components, pca_mean, pca_variance


@app.cell
def _(X_train, pca_variance):
    _total = np.sum(np.var(X_train, axis=0, ddof=1))  # sum of ALL eigenvalues = total variance
    _ratio = np.cumsum(pca_variance) / _total
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot(np.arange(1, len(_ratio) + 1), _ratio, "-")
    for _target in (0.8, 0.9, 0.95):
        _ax.axhline(_target, color="gray", lw=0.6, ls=":")
    _ax.set(xlabel="number of components k", ylabel="fraction of variance kept", ylim=(0, 1.02))
    _k90 = int(np.searchsorted(_ratio, 0.9) + 1) if _ratio[-1] >= 0.9 else None
    mo.vstack([_fig, mo.md(f"components needed for 90% of the variance: **{_k90}**")])
    return


@app.cell
def _(X_test, pca_components, pca_mean, side):
    _ks = [k for k in (5, 20, 50, 100) if k <= len(pca_components)]
    _fig, _axes = plt.subplots(3, len(_ks) + 1, figsize=(2 * (len(_ks) + 1), 6))
    for _row in range(3):
        _x = X_test[_row]
        _axes[_row, 0].imshow(_x.reshape(side, side), cmap="gray")
        _axes[_row, 0].set_title("test image" if _row == 0 else "", fontsize=8)
        for _j, _k in enumerate(_ks):
            _c = pca_components[:_k]
            _xh = reconstruct(project(_x[None], pca_mean, _c), pca_mean, _c)[0]
            _axes[_row, _j + 1].imshow(_xh.reshape(side, side), cmap="gray")
            if _row == 0:
                _axes[_row, _j + 1].set_title(f"k = {_k}", fontsize=8)
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.suptitle("held-out faces rebuilt from k eigenfaces (trained without them)", fontsize=9)
    _fig
    return


@app.cell
def _(X_test, X_train, pca_components, pca_mean, y_test, y_train):
    # evaluation (given). k is chosen by leave-one-out 1-NN accuracy on the TRAINING images, so
    # the test set is used exactly once, for the final number. (the eigenfaces are fitted once on
    # all training images, so this leave-one-out is a slightly optimistic approximation.)
    _ks = [k for k in K_GRID if k <= len(pca_components)]
    _n_test = len(y_test)
    _loo, _test = [], []
    for _k in _ks:
        _c = pca_components[:_k]
        _Z = project(X_train, pca_mean, _c)
        _d = np.sum(_Z**2, 1)[:, None] - 2 * _Z @ _Z.T + np.sum(_Z**2, 1)[None, :]
        np.fill_diagonal(_d, np.inf)  # a training image may not be its own neighbour
        _loo.append(float(np.mean(y_train[np.argmin(_d, axis=1)] == y_train)))
        _pred = nearest_neighbor(_Z, y_train, project(X_test, pca_mean, _c))
        _test.append(float(np.mean(_pred == y_test)))
    _best = int(np.argmax(_loo))  # ties go to the smallest k
    _k_star, _acc = _ks[_best], _test[_best]
    _se = np.sqrt(_acc * (1 - _acc) / _n_test)
    _pix = float(np.mean(nearest_neighbor(X_train, y_train, X_test) == y_test))
    _vals, _counts = np.unique(y_train, return_counts=True)
    _majority = float(np.mean(y_test == _vals[np.argmax(_counts)]))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.semilogx(_ks, _loo, "s-", label="leave-one-out on train (used to pick k)")
    _ax.semilogx(_ks, _test, "o:", alpha=0.6, label="test (for reference only)")
    _ax.axvline(_k_star, color="C0", lw=0.8)
    _ax.axhline(_pix, color="C1", ls="--", label=f"test, 1-NN on raw pixels ({_pix:.3f})")
    _ax.axhline(_majority, color="gray", ls=":", label=f"test, majority class ({_majority:.3f})")
    _ax.set(xlabel="k", ylabel="accuracy", ylim=(0, 1.02))
    _ax.legend(fontsize=7)
    _rows = "\n".join(f"| {k} | {lo:.3f} | {te:.3f} |" for k, lo, te in zip(_ks, _loo, _test))
    mo.vstack([_fig, mo.md(
        f"**chosen k = {_k_star}** (best leave-one-out accuracy on the training set). "
        f"test accuracy at that k: **{_acc:.3f} ± {_se:.3f}** (binomial standard error, {_n_test} test images). "
        f"baselines on the same test set: raw-pixel 1-NN {_pix:.3f}, majority class {_majority:.3f}.\n\n"
        "| k | leave-one-out (train) | test |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** your test accuracy at the $k$ chosen by leave-one-out on the training set,
       with its standard error. how does it compare with 1-NN on raw pixels and with the
       majority-class baseline? is the difference bigger than the uncertainty? (don't pick $k$
       by looking at the test curve: that turns the test set into a training set.)
    2. **the spectrum.** how many components hold 90% of the variance? what do the first few
       eigenfaces look like, and what variation (lighting, pose, identity) do they seem to
       capture?
    3. **ablation.** pick one: drop the first 1–3 components before classifying, skip
       centering, or standardize pixels. what happens to accuracy, and why?
    4. **failure analysis.** show two misclassified test faces next to their nearest training
       face. what went wrong?
    5. **linear algebra.** where exactly did your `svd` enter, and why is a reduced SVD of the
       $n \times d$ data matrix (with $n \ll d$) so much cheaper than an eigendecomposition of
       the $d \times d$ covariance?
    """)
    return


if __name__ == "__main__":
    app.run()
