import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets

    # small CPU config (offline: the synthetic stand-in, also used by verify) vs full config
    # (the real California housing data). both run in a few seconds on a laptop.
    OFFLINE = datasets.is_offline()
    LR = 0.1          # learning rate for gradient descent on standardized features
    N_STEPS = 500     # gradient descent steps
    SEED = 0          # for the train/test split


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 · capstone: predict house prices (reference solution)

    read [`brief.md`](brief.md) first. you'll predict the median house price of California
    districts from eight numbers about each district, with a linear model
    $\hat y = w \cdot x + b$ trained by **gradient descent with a gradient you derive by
    hand**, and check it against the **exact least-squares answer**. it's topic 06 with
    eight inputs instead of one.

    the data loading, the train/test split, the plots and the evaluation are written for
    you. the five stages below are yours. inside them you may use `@` for matrix products
    (you built it in topic 04).

    | stage | you build |
    |---|---|
    | 1 | `standardize_features`: rescale every feature with the training set's statistics |
    | 2 | `predict`, `mse`: the model and its loss |
    | 3 | `mse_grad`: the hand-derived gradient with respect to $w$ and $b$ |
    | 4 | `fit_gd`: gradient descent, with a loss curve |
    | 5 | `fit_closed_form`: the exact answer from the normal equations |

    when they're green, scroll down to train, evaluate and write up what you find.
    """)
    return


@app.function
def standardize_features(X_train, X_test):
    """stage 1: standardize every feature (column) using the TRAINING set's mean and std only.

    X_train (n, d), X_test (m, d). mean = X_train.mean(axis=0), std = X_train.std(axis=0),
    both shape (d,). raise ValueError if any training column has std 0.
    returns (Z_train, Z_test, mean, std) with Z = (X - mean) / std for both sets.
    (the test set must NOT influence mean or std: in real life you don't have it yet.)
    """
    # step 1: X_train, X_test as float arrays.
    # step 2: mean = X_train.mean(axis=0) and std = X_train.std(axis=0)  (one value per column).
    # step 3: if np.any(std == 0), raise ValueError.
    # step 4: return (X_train - mean) / std, (X_test - mean) / std, mean, std.
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    if np.any(std == 0):
        raise ValueError("a training feature is constant and can't be standardized")
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", standardize_features=standardize_features)
    return


@app.function
def predict(X, w, b):
    """stage 2: predictions y_hat = X w + b for every row of X (n, d); w (d,), b float. returns (n,)."""
    # step 1: return np.asarray(X, dtype=float) @ w + b.
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b


@app.function
def mse(X, y, w, b):
    """stage 2: the mean squared error (1/n) sum_i (x_i . w + b - y_i)^2, as a float."""
    # step 1: r = predict(X, w, b) - y.
    # step 2: return float(np.mean(r ** 2)).
    r = predict(X, w, b) - np.asarray(y, dtype=float)
    return float(np.mean(r**2))


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", predict=predict, mse=mse)
    return


@app.function
def mse_grad(X, y, w, b):
    """stage 3: the gradient of mse with respect to w and b. returns (grad_w (d,), grad_b float).

    with residuals r = X w + b - y (shape (n,)):
        grad_w[j] = (2/n) * sum_i r_i X[i, j]      i.e.  grad_w = (2/n) X^T r
        grad_b    = (2/n) * sum_i r_i
    (topic 06's formulas, once per feature.)
    """
    # step 1: X, y as float arrays; n = len(y); r = predict(X, w, b) - y.
    # step 2: grad_w = 2 * X.T @ r / n   (shape (d,): one partial derivative per weight).
    # step 3: grad_b = 2 * np.sum(r) / n.
    # step 4: return grad_w, float(grad_b).
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    r = predict(X, w, b) - y
    return 2 * X.T @ r / n, float(2 * np.sum(r) / n)


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", mse_grad=mse_grad, mse=mse)
    return


@app.function
def fit_gd(X, y, *, lr=0.1, n_steps=500):
    """stage 4: fit (w, b) by gradient descent on mse, starting from w = 0 (shape (d,)), b = 0.

    every step: compute (grad_w, grad_b) at the current (w, b), then
    w <- w - lr * grad_w and b <- b - lr * grad_b.
    returns (w (d,), b float, losses) where losses is a 1-D array of length n_steps + 1:
    the loss before the first step and after every step.
    """
    # step 1: w = np.zeros(X.shape[1]); b = 0.0; losses = [mse(X, y, w, b)].
    # step 2: repeat n_steps times:
    #           grad_w, grad_b = mse_grad(X, y, w, b)
    #           w = w - lr * grad_w;  b = b - lr * grad_b
    #           losses.append(mse(X, y, w, b))
    # step 3: return w, float(b), np.array(losses).
    X = np.asarray(X, dtype=float)
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = [mse(X, y, w, b)]
    for _ in range(n_steps):
        grad_w, grad_b = mse_grad(X, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
        losses.append(mse(X, y, w, b))
    return w, float(b), np.array(losses)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", fit_gd=fit_gd, mse=mse)
    return


@app.function
def gauss_solve(M, c):
    """given (not a stage): solve the square linear system M theta = c by Gaussian elimination
    with partial pivoting. M (k, k) invertible, c (k,); returns theta (k,).

    a preview of chapter 02, topic 03, where you'll build this properly (as LU factorization)
    and learn why the pivoting matters. use it as a black box here.
    """
    A = np.array(M, dtype=float)
    v = np.array(c, dtype=float)
    k = len(v)
    for col in range(k):
        p = col + int(np.argmax(np.abs(A[col:, col])))  # the biggest pivot, for stability
        A[[col, p]], v[[col, p]] = A[[p, col]], v[[p, col]]
        for row in range(col + 1, k):
            factor = A[row, col] / A[col, col]
            A[row, col:] -= factor * A[col, col:]
            v[row] -= factor * v[col]
    theta = np.zeros(k)
    for row in range(k - 1, -1, -1):  # back substitution, from the last unknown up
        theta[row] = (v[row] - A[row, row + 1:] @ theta[row + 1:]) / A[row, row]
    return theta


@app.function
def fit_closed_form(X, y):
    """stage 5: the exact least-squares (w, b), from the NORMAL EQUATIONS.

    append a column of ones to X, so that b becomes one more weight: A = [X | 1], shape (n, d+1),
    and theta = (w, b). setting the gradient of mse to zero gives  (A^T A) theta = A^T y.
    solve that (d+1)-by-(d+1) system with the given gauss_solve.
    returns (w (d,), b float).
    """
    # step 1: X, y as float arrays; A = np.column_stack([X, np.ones(len(y))]).
    # step 2: M = A.T @ A  (shape (d+1, d+1))  and  c = A.T @ y  (shape (d+1,)).
    # step 3: theta = gauss_solve(M, c).
    # step 4: return theta[:-1], float(theta[-1]).
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([X, np.ones(len(y))])
    theta = gauss_solve(A.T @ A, A.T @ y)
    return theta[:-1], float(theta[-1])


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", fit_closed_form=fit_closed_form, fit_gd=fit_gd,
               standardize_features=standardize_features, predict=predict, mse=mse)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    the California housing data (1990 census): one row per district, 8 features, and the
    district's median house value in units of \$100,000. offline, `mlfs.datasets` gives a
    synthetic stand-in with the same format, so everything below still runs. we keep a
    random 20% of the districts aside as a **test set** and never train on it.
    """)
    return


@app.cell
def _():
    X_all, y_all, feature_names = datasets.california_housing()
    _rng = np.random.default_rng(SEED)
    _idx = _rng.permutation(len(y_all))
    _n_train = int(0.8 * len(y_all))
    X_train, y_train = X_all[_idx[:_n_train]], y_all[_idx[:_n_train]]
    X_test, y_test = X_all[_idx[_n_train:]], y_all[_idx[_n_train:]]
    _rows = "\n".join(f"| {n} | {X_train[:, j].mean():.4g} | {X_train[:, j].std():.4g} |"
                      for j, n in enumerate(feature_names))
    mo.md(f"{'**synthetic stand-in (offline)**' if OFFLINE else '**real California housing data**'}: "
          f"{len(y_all):,} districts · train {len(y_train):,} · test {len(y_test):,}\n\n"
          "| feature | mean (train) | std (train) |\n|---|---|---|\n" + _rows)
    return X_test, X_train, feature_names, y_test, y_train


@app.cell
def _(X_test, X_train, y_train):
    Z_train, Z_test, feat_mean, feat_std = standardize_features(X_train, X_test)
    w_gd, b_gd, losses = fit_gd(Z_train, y_train, lr=LR, n_steps=N_STEPS)
    w_cf, b_cf = fit_closed_form(Z_train, y_train)
    best_train_loss = mse(Z_train, y_train, w_cf, b_cf)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _axes[0].plot(losses, label=f"gradient descent (lr = {LR})")
    _axes[0].axhline(best_train_loss, color="k", ls="--", lw=0.8, label="closed form (best possible)")
    _axes[0].set(xlabel="step", ylabel="training MSE", title="loss curve")
    _axes[0].legend(fontsize=8)
    _axes[1].semilogy(np.maximum(losses - best_train_loss, 1e-16))
    _axes[1].set(xlabel="step", ylabel="training MSE − best", title="distance to the best loss (log scale)")
    _fig.tight_layout()
    _fig
    return Z_test, Z_train, b_cf, b_gd, best_train_loss, losses, w_cf, w_gd


@app.cell
def _(Z_test, b_cf, b_gd, feature_names, w_cf, w_gd, y_test, y_train):
    def _mse_and_se(pred, target):
        """test MSE with its standard error (topic 05: the squared errors are the samples)."""
        _sq = (pred - target) ** 2
        return _sq.mean(), _sq.std(ddof=1) / np.sqrt(len(_sq))

    _results = {
        "baseline: predict the training mean": np.full(len(y_test), y_train.mean()),
        "gradient descent": predict(Z_test, w_gd, b_gd),
        "closed form": predict(Z_test, w_cf, b_cf),
    }
    _var = np.var(y_test)
    _rows = []
    for _name, _pred in _results.items():
        _m, _se = _mse_and_se(_pred, y_test)
        _rows.append(f"| {_name} | {_m:.4f} ± {_se:.4f} | {np.sqrt(_m) * 100:.0f}k | {1 - _m / _var:.3f} |")
    _wrows = "\n".join(f"| {n} | {a:+.4f} | {c:+.4f} |" for n, a, c in zip(feature_names, w_gd, w_cf))
    mo.md("### test-set results\n\n"
          "| model | test MSE (± se) | typical error ($, RMSE) | R² |\n|---|---|---|---|\n" + "\n".join(_rows)
          + "\n\n### weights on the standardized features\n\n"
          f"| feature | gradient descent | closed form |\n|---|---|---|\n{_wrows}\n"
          f"| (intercept b) | {b_gd:+.4f} | {b_cf:+.4f} |")
    return


@app.cell
def _(Z_test, b_gd, w_gd, y_test):
    # the full-config requirement from brief.md (given): test MSE <= 0.55 on the real data
    _m = float(np.mean((predict(Z_test, w_gd, b_gd) - y_test) ** 2))
    if OFFLINE:
        _msg = mo.md(f"offline (synthetic stand-in): test MSE {_m:.4f}. the 0.55 threshold applies to the real data only.")
    else:
        _ok = _m <= 0.55
        _msg = mo.callout(mo.md(f"{'✅' if _ok else '❌'} test MSE of your gradient descent model: **{_m:.4f}** "
                                f"(requirement: ≤ 0.55)"), kind="success" if _ok else "danger")
    _msg
    return


@app.cell
def _(Z_test, b_cf, w_cf, y_test):
    _pred = predict(Z_test, w_cf, b_cf)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _axes[0].scatter(y_test, _pred, s=4, alpha=0.3)
    _lims = [min(y_test.min(), _pred.min()), max(y_test.max(), _pred.max())]
    _axes[0].plot(_lims, _lims, "k--", lw=0.8)
    _axes[0].set(xlabel="true price ($100k)", ylabel="predicted price ($100k)", title="test set: predicted vs true")
    _axes[1].scatter(_pred, _pred - y_test, s=4, alpha=0.3)
    _axes[1].axhline(0, color="k", lw=0.8)
    _axes[1].set(xlabel="predicted price ($100k)", ylabel="residual (predicted − true)", title="residuals")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** your test MSE (with its standard error), RMSE in dollars and $R^2$, next to
       the baseline. how much better than "always predict the average" is the model, and is
       the difference bigger than the uncertainty?
    2. **training.** describe your loss curve. how many steps does gradient descent need to get
       within $10^{-4}$ of the closed form's training loss? try a few learning rates (change
       `LR` in the setup cell): what's the largest that still converges, and why is it *not*
       the topic 06 limit of 1.0, even though the features are standardized? (hint: some
       features move together.)
    3. **ablation: skip standardization.** train `fit_gd` on the raw `X_train` instead. what
       learning rate do you need to avoid divergence, and where is the loss after 500 steps?
       connect it to topic 06's lab.
    4. **the weights.** which standardized features have the largest weights? are any signs
       surprising (look at `total_rooms`, `total_bedrooms`, `population`, `households`)? why
       is it risky to read a weight as "the effect of this feature" when features are
       strongly related?
    5. **failure analysis.** look at the residual plot. the real data caps prices at \$500,001:
       what does that do to the residuals of the most expensive districts? what kinds of
       districts does a linear model get most wrong, and what could fix it?
    """)
    return


if __name__ == "__main__":
    app.run()
