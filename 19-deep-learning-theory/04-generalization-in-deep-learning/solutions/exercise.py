import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # generalization in deep learning · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. this topic is NumPy: random-feature models are linear in their trained weights, so
    every quantity in a double-descent curve can be computed exactly.

    | stage | you build |
    |---|---|
    | 1 | `corrupt_labels`: replace an exact fraction of labels with a wrong class |
    | 2 | `RandomReLUFeatures`: a two-layer network whose first layer is frozen at random |
    | 3 | `min_norm_least_squares`: the minimum-norm (or ridge) solution, on either side of the interpolation threshold |
    | 4 | `double_descent_curve`: test error against the number of features, over seeds |
    | 5 | `gd_least_squares`: gradient descent, and the solution it picks among infinitely many |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def corrupt_labels(y, fraction, n_classes, *, rng=None):
    """stage 1: copy of int labels y with exactly round(fraction * n) positions (chosen without replacement) set to a different class, uniform over the other n_classes - 1."""
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    out = y.copy()
    k = int(round(fraction * len(y)))
    idx = rng.choice(len(y), size=k, replace=False)
    shift = rng.integers(1, n_classes, size=k)
    out[idx] = (y[idx] + shift) % n_classes
    return out


@app.cell(hide_code=True)
def _():
    checks.run(1, corrupt_labels=corrupt_labels)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RandomReLUFeatures:
    """phi(x) = relu(x W) / sqrt(p), W (d, p) with N(0, 1/d) entries drawn at fit time."""

    def __init__(self, n_features, *, rng=None):
        self.n_features = n_features
        self.rng = rng

    def fit(self, X, y=None):
        """stage 2: draw W_ (d, n_features) from N(0, 1/d) with np.random.default_rng(self.rng); returns self."""
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        self.W_ = np.random.default_rng(self.rng).normal(0.0, 1.0 / np.sqrt(d), size=(d, self.n_features))
        return self

    def transform(self, X):
        """stage 2: relu(X @ W_) / sqrt(n_features), shape (n, n_features)."""
        return np.maximum(np.asarray(X, dtype=float) @ self.W_, 0.0) / np.sqrt(self.n_features)


@app.cell(hide_code=True)
def _():
    checks.run(2, RandomReLUFeatures=RandomReLUFeatures)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def min_norm_least_squares(Phi, Y, *, ridge=0.0):
    """stage 3: argmin ||Phi B - Y||^2 + ridge ||B||^2; for ridge = 0 the minimum-norm least-squares solution (pinv). B is (p,) or (p, k)."""
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if ridge == 0:
        return np.linalg.pinv(Phi) @ Y
    n, p = Phi.shape
    if p <= n:
        return np.linalg.solve(Phi.T @ Phi + ridge * np.eye(p), Phi.T @ Y)
    return Phi.T @ np.linalg.solve(Phi @ Phi.T + ridge * np.eye(n), Y)


@app.cell(hide_code=True)
def _():
    checks.run(3, min_norm_least_squares=min_norm_least_squares)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def double_descent_curve(X_train, Y_train, X_test, y_test, widths, *, ridge=0.0, n_seeds=3):
    """stage 4: dict of (len(widths), n_seeds) arrays 'test_error', 'test_mse', 'train_mse' for random-feature regression onto one-hot Y_train; seed s uses rng=s."""
    X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    y_test = np.asarray(y_test)
    Y_test = np.eye(Y_train.shape[1])[y_test]
    out = {k: np.zeros((len(widths), n_seeds)) for k in ("test_error", "test_mse", "train_mse")}
    for i, p in enumerate(widths):
        for s in range(n_seeds):
            feats = RandomReLUFeatures(int(p), rng=s).fit(X_train)
            F_tr, F_te = feats.transform(X_train), feats.transform(X_test)
            B = min_norm_least_squares(F_tr, Y_train, ridge=ridge)
            P_tr, P_te = F_tr @ B, F_te @ B
            out["train_mse"][i, s] = np.mean(np.sum((P_tr - Y_train) ** 2, axis=1))
            out["test_mse"][i, s] = np.mean(np.sum((P_te - Y_test) ** 2, axis=1))
            out["test_error"][i, s] = np.mean(P_te.argmax(axis=1) != y_test)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(4, double_descent_curve=double_descent_curve, RandomReLUFeatures=RandomReLUFeatures,
               min_norm_least_squares=min_norm_least_squares)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gd_least_squares(X, y, *, lr, steps, w0=None):
    """stage 5: w after `steps` gradient steps on (1 / 2n) ||X w - y||^2 from w0 (zeros if None); w0 is not modified."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float, copy=True)
    for _ in range(steps):
        w = w - lr * X.T @ (X @ w - y) / n
    return w


@app.cell(hide_code=True)
def _():
    checks.run(5, gd_least_squares=gd_least_squares)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. the capstone reproduces
    double descent with *trained* networks, where none of this is exact any more.
    """)
    return


if __name__ == "__main__":
    app.run()
