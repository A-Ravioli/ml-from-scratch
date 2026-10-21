# entropic OT & sinkhorn · lab notes

numbers are from the reference solution on an Apple M2 laptop CPU (one thread, as in CI); the whole lab takes about 20–25 s. all data are synthetic and seeded, so everything except the timings reproduces exactly (timings vary by ±20% with machine load).

## experiment 1: how ε trades blur for speed

**what people usually predict:** iterations grow a little as ε shrinks (a log factor, perhaps); the naive version fails with NaN as soon as anything underflows; the plan sharpens steadily.

**what happens** (60 + 60 points, median cost 1, exact $W_2^2 = 0.8683$ from your LP; tolerance $10^{-6}$ on the row marginals):

| ε | iterations | ms | partners per point | cost gap ⟨P, C⟩ − W | naive `sinkhorn` |
|---|---|---|---|---|---|
| 1 | 3–4 | < 1 | 59.6 | 0.1345 | same plan |
| 0.3 | 5–8 | 1 | 56.1 | 0.1058 | same plan |
| 0.1 | 9–16 | 1 | 42.0 | 0.0589 | same plan |
| 0.03 | 33–64 | 5 | 20.7 | 0.0196 | same plan |
| 0.01 | 129–256 | 19 | 9.7 | 0.0057 | same plan |
| 0.003 | 513–1024 | 74 | 4.7 | 0.0014 | same plan |
| 0.001 | more than 32,768 (not converged) | 2,400 | 2.7 | 0.0004 | NaN |

**why:** between ε = 0.1 and 0.003 the iteration count grows a little faster than $1/\varepsilon$ (about 4× per 3.3× decrease of ε), as the convergence analysis suggests: each half-step moves the potentials by amounts of order ε, and they have to travel distances of the order of the cost differences. below that something qualitatively different happens: at ε = 0.001 the plan is almost a permutation (2.7 partners per point), and Sinkhorn slows to a crawl (in this near-permutation regime the row error can shrink as slowly as $1/k$); after 32,768 iterations it still hasn't reached $10^{-6}$. the cost gap tracks the bound of on paper 2: it shrinks roughly in proportion to ε (0.0589 → 0.0057 → 0.0004 per factor 10) and stays well below $\varepsilon(H(a) + H(b)) = 8.2\varepsilon$. the naive version gives exactly the same plans until ε = 0.001, even at ε = 0.003 where 0.75% of the kernel's entries have already underflowed to 0 (they're negligible next to the rest of their row). at ε = 0.001, 67% of the entries underflow, including one entire row (and 16 entire columns) of $K$: for that row $u = a/(Kv)$ is infinite, and $\infty\cdot0$ gives NaN. (the stage 2 checks include a problem where the naive kernel is 0 exactly where the optimal plan lives.) in the dropdown, ε = 1 sends every barycentric image to nearly the same point, the target's mean: with a blurred plan, $T(x_i)$ averages over almost all targets. that's the origin of the bias experiment 3 shows. for *matching* points (topic 17's couplings) you want a small ε relative to the cost spread; for a smooth *loss* a moderate ε, debiased.

## experiment 2: Sinkhorn against the exact LP

**what people usually predict:** Sinkhorn is much faster at every size (that's its selling point), and its answer is a slight overestimate.

**what happens** (uniform points in the unit square; errors relative to the exact $W_2^2$; tolerance $10^{-6}$, at most 3,000 iterations):

| n | exact W2² | LP ms | Sinkhorn ms, ε = 0.01 | ⟨P, C⟩ error | S error | Sinkhorn ms, ε = 0.002 | ⟨P, C⟩ error |
|---|---|---|---|---|---|---|---|
| 50 | 0.02617 | 8 | 18 | +18.8% | −11.2% | 175 | +2.6% |
| 100 | 0.01298 | 38 | 35 | +45.4% | −21.2% | 157 | +5.2% |
| 200 | 0.01036 | 263 | 125 | +68.3% | −17.3% | 700 | +9.5% |
| 400 | 0.00719 | 2,440 | 642 | +110.9% | −16.6% | 3,003 | +16.9% |

**why:** a Sinkhorn iteration costs $O(n^2)$ and the LP's total cost grows faster (here about $9\times$ per doubling of $n$ from 200 to 400, against $5\times$ for Sinkhorn at ε = 0.01), so Sinkhorn wins at large enough $n$, but only at a blur that ruins the number: as $n$ grows, $W_2^2$ between two samples of the *same* square shrinks (it's pure sampling noise, topic 15's experiment 3), while the entropic bias stays of order $\varepsilon\log n$, so the relative error of $\langle P_\varepsilon, C\rangle$ explodes to +111%. sharpening to ε = 0.002 cuts the error to +17% but needs so many iterations that the LP is just as fast at $n = 400$. the debiased divergence errs the other way (−11% to −21%): its self-terms subtract the blur of each sample against itself, which overcorrects a little when the true distance is this small. the honest summary: on a CPU, at a few hundred points, Sinkhorn's advantage isn't speed for a single accurate solve; it's that it is smooth, differentiable, batchable on a GPU, and (for losses) debiasable. for an exact number at this scale, use the LP or the Hungarian algorithm.

## experiment 3: fitting a point cloud: biased vs debiased

**what people usually predict:** both losses pull the blob onto the ring; maybe the biased one converges a bit more slowly.

**what happens** (ε = 1, 60 steps):

| loss | mean radius (target 2) | exact W2 to the ring |
|---|---|---|
| biased $\mathrm{OT}_\varepsilon$ | 1.859 | 0.182 |
| debiased $S_\varepsilon$ | 1.999 | 0.114 |

with the dropdown: at ε = 0.1 the two are closer (radius 1.987 vs 1.999, W2 0.112 vs 0.070); at ε = 3 the biased fit collapses to radius 1.29 (W2 0.716) while the debiased one stays on the ring (1.999, W2 0.177).

**why:** the biased flow's update moves each point towards $T_{XY}(x_i)$, a weighted average of the ring points it's coupled to. with blur, the average of points spread along an arc lies inside the circle, so the fixed point is a *smaller* ring: $\mathrm{OT}_\varepsilon(\,\cdot\,, \nu)$ is minimized by a shrunken $\nu$. the debiased update is $2\big(T_{XY}(x_i) - T_{XX}(x_i)\big)$: it compares the blurred image of $x_i$ in the target with its blurred image in its own cloud, and stops when the two blurs agree, i.e. when the clouds match. that's the practical content of Feydy et al.'s theorem that $S_\varepsilon(\,\cdot\,, \nu)$ is minimized exactly at $\nu$, and why topic 17's Sinkhorn loss is debiased by default.
