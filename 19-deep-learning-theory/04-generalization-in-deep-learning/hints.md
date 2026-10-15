# generalization in deep learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`rng = np.random.default_rng(rng)`, `out = y.copy()`, `k = int(round(fraction * len(y)))`, `idx = rng.choice(len(y), size=k, replace=False)`. a shift drawn from `rng.integers(1, n_classes, size=k)` (upper bound exclusive: 1 to n_classes - 1) added modulo `n_classes` gives a uniformly random *different* class.

</details>

## stage 2

<details><summary>hint</summary>

in `fit`: `d = X.shape[1]`, `self.W_ = np.random.default_rng(self.rng).normal(0.0, 1.0 / np.sqrt(d), size=(d, self.n_features))`, `return self`. in `transform`: `np.maximum(X @ self.W_, 0) / np.sqrt(self.n_features)`.

</details>

## stage 3

<details><summary>hint</summary>

`ridge == 0`: `np.linalg.pinv(Phi) @ Y` (works for every shape and rank). otherwise, if `p <= n`, `np.linalg.solve(Phi.T @ Phi + ridge * np.eye(p), Phi.T @ Y)`; else `Phi.T @ np.linalg.solve(Phi @ Phi.T + ridge * np.eye(n), Y)`.

</details>

## stage 4

<details><summary>hint</summary>

`Y_test = np.eye(k)[y_test]`. loop over `i, p` and `s in range(n_seeds)`: `f = RandomReLUFeatures(p, rng=s).fit(X_train)`, `F_tr, F_te = f.transform(X_train), f.transform(X_test)`, `B = min_norm_least_squares(F_tr, Y_train, ridge=ridge)`; then `np.mean(np.sum((F_te @ B - Y_test) ** 2, axis=1))`, the same on the training set, and `np.mean((F_te @ B).argmax(1) != y_test)`.

</details>

## stage 5

<details><summary>hint</summary>

start from `np.zeros(p)` or `np.array(w0, dtype=float, copy=True)` (a copy, so the caller's array is untouched), then repeat `w = w - lr * X.T @ (X @ w - y) / n` exactly `steps` times.

</details>
