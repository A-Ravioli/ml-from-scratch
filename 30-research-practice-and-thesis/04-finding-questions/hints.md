# finding questions · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

most rules are "this field is a non-empty string": write that once. for numbers, rule out bools before anything else (in Python `True` is an `int`), and let the `VAGUE` regex find every match, not just the first.

</details>

## stage 2

<details><summary>hint</summary>

under the alternative the $t$ statistic follows a noncentral $t$ with $n - 1$ degrees of freedom and noncentrality $\Delta\sqrt n/\sigma_d$; the power is the probability that it lands beyond the critical value (or values, two-sided) of the null $t$. the seed search is the same upward search as topic 02's.

</details>

## stage 3

<details><summary>hint</summary>

both functions are the same identity, $C = 6ND$ = hours × seconds per hour × throughput × utilization, read in opposite directions; the second hands the compute to `compute_optimal`.

</details>

## stage 4

<details><summary>hint</summary>

validate the whole plan before computing anything, so a bad entry never produces a half-built answer; then it's a sum of products, and one comparison with the safety factor applied once.

</details>

## stage 5

<details><summary>hint</summary>

reuse everything: stage 1's problems get a prefix, stage 2 runs only when the hypothesis is clean, and stage 4 runs inside a `try` so that a malformed plan becomes a reported problem instead of an exception.

</details>
