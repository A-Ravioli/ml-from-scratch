# potential outcomes & weighting · hints

open a hint only after you've been stuck for a while. this elective is spec-driven: there's one hint per stage, and it points at the idea, not the code.

## stage 1

<details><summary>hint</summary>

`LogisticRegression(alpha=alpha).fit(X, t)` then `.predict_proba(X)[:, 1]`: the classes are sorted, so column 1 is the probability of `t == 1`. pass `t` as integers. `np.clip(e, clip, 1 - clip)` only when `clip is not None`. the ESS is one line: `w.sum() ** 2 / (w ** 2).sum()`, returned as a float.

</details>

## stage 2

<details><summary>hint</summary>

build the two weight vectors once: `w1 = t / e` (zero for controls) and `w0 = (1 - t) / (1 - e)` (zero for treated). Horvitz–Thompson is `mean(w1 * y) - mean(w0 * y)`; Hájek replaces each mean by `sum(w * y) / sum(w)`. for the ATT, keep the treated mean as it is and give each control the weight `e / (1 - e)` in a self-normalized mean.

</details>

## stage 3

<details><summary>hint</summary>

two separate fits on two subsets: `make_model().fit(X[t == 0], y[t == 0])` and the same for `t == 1`. then call `.predict(X)` on the *full* `X` with each model. if `make_model` is `None`, use the class `LinearRegression` itself as the factory: calling it gives a fresh model.

</details>

## stage 4

<details><summary>hint</summary>

compute the score vector `psi` in one vectorized line from the formula in the lesson; everything else follows: the estimate is `psi.mean()` and the standard error `psi.std(ddof=1) / np.sqrt(len(psi))`. check it on the worked example: the six scores are 3.25, 6, −2, 7, 0, 0.75.

</details>
