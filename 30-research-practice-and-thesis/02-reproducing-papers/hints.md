# reproducing papers · hints

one hint per stage. at this level the spec in the lesson is the main help; open a hint only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

an equivalence test at level $\alpha$ uses the one-sided quantile, so the interval is a $(1 - 2\alpha)$ one. once you have the interval and the band, the verdict is three comparisons: inside, entirely outside (on either side), or neither.

</details>

## stage 2

<details><summary>hint</summary>

separate the two branches by what they produce: a difference, a standard error and degrees of freedom (paired from the per-seed differences, unpaired from Welch). everything after that is shared. then compute two booleans, "the direction holds" and "disjoint from the band", and the five verdicts follow from them and "inside the band" in the spec's order.

</details>

## stage 3

<details><summary>hint</summary>

the condition only gets easier as $n$ grows, so search upwards from 2 and stop at the first $n$ that satisfies it, or at `max_seeds`.

</details>

## stage 4

<details><summary>hint</summary>

make one pass over the claims that produces three things at once: a summary row, a claim section and a verdict. build the gaps and checklist sections separately, counting answered items with exactly the test you use to tick them, then join all the lines.

</details>
