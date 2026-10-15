# 03 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, with exact (dual-number) Jacobians and Hessians | all pass, plus your own extra checks (e.g. your `hessian` against `numerical_hessian` at random points, Newton on a problem with a known closed-form root, a test that pure Newton reaches the saddle where yours doesn't) |
| **results** | no numbers | iteration counts for some experiments | estimated orders for every problem in experiment A, iteration counts *and* wall-clock times for Newton versus gradient descent, all numbers reproducible from the notebook | the same, plus a cost model: forward passes per Newton step as a function of $n$, checked against your timings for at least two problem sizes |
| **analysis** | none | describes the plots | explains the quadratic and linear orders (simple versus double root) and what each safeguard in `newton_minimize` prevents, with a concrete example | also explains the finite-difference results (why $h = 10^{-6}$ is indistinguishable from exact, why $h = 0.3$ is linear, why a quadratic system doesn't care), the basin boundaries, and where Newton stops being worth its per-step cost |
| **clarity** | hard to follow | readable | labelled figures and a narrative that answers all five prompts | a peer could reproduce every number from the write-up alone (config, tolerances, starting points, runtime) |

## what graders look for

- **exactness, demonstrated.** "my derivatives are exact" should come with evidence: an agreement to $10^{-13}$ with a hand-derived formula, or an experiment where finite differences visibly fail. the honest answer to prompt 5 includes a place where exactness made *no* difference (well-chosen finite differences are very good for Newton's local convergence).
- **orders read correctly.** an order estimate at the rounding floor is noise. a good write-up states which iterations it used and why.
- **iterations are not time.** Newton wins on iterations by orders of magnitude; on wall-clock time it depends on $n$ and on the cost of a dual-number pass. a write-up that only compares iterations misses half the story.
- **safeguards motivated by failures.** the Hessian modification and the line search should each be justified by a start where pure Newton goes wrong (towards a saddle or a maximum, or overshooting so that $f$ increases), not just described.
- **connecting to the chapter.** the best write-ups tie the double-root slowdown to topic 01's $(m-1)/m$, the saddle-seeking to topic 03's quadratic model, the line search to topic 05's descent lemma, and the Hessian to topic 04's nested duals.
