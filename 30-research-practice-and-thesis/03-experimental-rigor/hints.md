# experimental rigor · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

call the function for every seed first and collect the raw results; only then decide what shape they have. `np.ndim` is 0 for every kind of scalar, Python or NumPy, which makes "is this a scalar?" one test.

</details>

## stage 2

<details><summary>hint</summary>

the $t$ interval needs three things: the mean, the standard error with $n - 1$ in the variance, and the quantile at $1 - \alpha/2$ with $n - 1$ degrees of freedom. the bootstrap branch is a single call to the function you wrote in chapter 05.

</details>

## stage 3

<details><summary>hint</summary>

paired, everything is about the per-seed differences. unpaired, the bootstrap must resample both groups independently, but your `bootstrap_ci` only resamples one array: the statistic you hand it can do the other resampling itself, from its own generator, every time it's called.

</details>

## stage 4

<details><summary>hint</summary>

build each row as a list of cell strings while collecting the means; the best row can only be decided once all means are known. a symmetric $t$ interval's half-width is half its length, and the Δ column is just stage 2 applied to the differences.

</details>

## stage 5

<details><summary>hint</summary>

on paper problem 3 gives the weight of each sorted score; the whole computation is a weighted sum. for an array of budgets, think of a matrix of weights with one row per $k$.

</details>
