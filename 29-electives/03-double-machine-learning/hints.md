# double machine learning · hints

open a hint only after you've been stuck for a while. this elective is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

`folds = np.array_split(as_rng(rng).permutation(n), n_folds)`; for each `test` index array, the training rows are `np.setdiff1d(np.arange(n), test)`. create the model *inside* the loop (`make_model().fit(X[train], y[train])`) and write its predictions into `out[test]` of a preallocated `np.empty(n)`. validate `response` before doing any work, and if `folds` is given, use it instead of drawing a permutation.

</details>

## stage 2

<details><summary>hint</summary>

`partialling_out` is four lines: `theta = sum(d * y) / sum(d ** 2)`, `psi = d * (y - theta * d)`, `J = mean(d ** 2)`, `se = sqrt(mean(psi ** 2) / J ** 2 / n)`. `naive_plugin` is a plain loop that alternates the two updates in the spec; don't centre `d` anywhere, and fit a fresh model each iteration.

</details>

## stage 3

<details><summary>hint</summary>

make `g = as_rng(rng)` once. in every repeat, draw one partition `folds = np.array_split(g.permutation(n), n_folds)` and pass the same `folds` to both `cross_fit_predict` calls: the generator advances between repeats, so each repeat gets a new partition and the run is still reproducible from `rng`. collect the per-repeat `theta` and `se` in arrays, then `theta = np.median(thetas)` and `se = np.sqrt(np.median(ses ** 2 + (thetas - theta) ** 2))`.

</details>

## stage 4

<details><summary>hint</summary>

one fold loop, three fits per fold. inside it, split the training indices by arm: `tr0 = train[d[train] == 0]` and `tr1 = train[d[train] == 1]`. fill three preallocated arrays `mu0`, `mu1`, `e` on the test indices; after the loop, clip `e` and compute the AIPW scores exactly as in topic 02.

</details>
