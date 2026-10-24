# wasserstein distances · lab notes

numbers are from the reference solution on an Apple M2 laptop CPU (one thread, as in CI); the whole lab takes about 5 s. all data are synthetic and seeded, so the tables below reproduce exactly; only the timings vary with machine load (by about ±20% between runs here).

## experiment 1: Wasserstein vs KL and JS when supports don't overlap

**what people usually predict:** all three divergences grow with the shift, KL fastest; any of them can be minimized by gradient descent, KL perhaps best because it's the maximum-likelihood loss.

**what happens** (half-width $w = 0.5$):

| divergence | θ = 0.5 | θ = 1 | θ = 2 | θ = 4 | θ after 40 descent steps from 3 |
|---|---|---|---|---|---|
| KL(p ‖ q) | ∞ | ∞ | ∞ | ∞ | 3.000 (no step possible: the gradient isn't finite) |
| JS(p, q) | 0.443 | 0.693 | 0.693 | 0.693 | 3.000 (gradient exactly 0) |
| W1(p, q) | 0.500 | 1.000 | 2.000 | 4.000 | 0.163 (reaches 0 at step 12, sits there until step 20, then hops between −0.087 and +0.163) |

**why:** KL is infinite for every $\theta > 0$, not just large ones: $q_\theta$ is zero on the left edge of $p$'s support, so $p\log(p/q)$ blows up there. JS is bounded by $\log 2$ and reaches it exactly once the supports are disjoint (here from $\theta = 2w = 1$ on), so its derivative is 0 and gradient descent never moves. $W_1$ is exactly $\theta$, whatever the shape (a translation's optimal plan moves every grain by $\theta$), so its gradient is $\pm1$ everywhere: descent with step 0.25 walks down 12 steps to exactly 0, where the central difference is symmetric and the gradient is 0. rounding moves θ off 0 by about $10^{-4}$ by step 19, and from step 21 the full-size gradient sends it back and forth across 0 for good, because a constant-magnitude gradient never shrinks. with the slider at $w = 2$ the JS curve rises more slowly and saturates at $\theta = 4$, so a start at 3 would get a (small) gradient; $W_1$ doesn't change at all. that's the WGAN argument (Arjovsky et al., 2017) in one picture: when model and data don't overlap, only a transport distance says which way to move.

## experiment 2: what exactness costs

**what people usually predict:** the LP grows with its size, roughly $4\times$ per doubling of $n$; a hand-written simplex is a constant factor slower.

**what happens** (uniform points in the unit square, squared Euclidean cost; times in ms, best of 2):

| n | LP variables | cost (all solvers agree) | your simplex | your LP (HiGHS) | Hungarian (SciPy) |
|---|---|---|---|---|---|
| 8 | 64 | 0.09731 | 1.5 | 1.1 | < 0.1 |
| 16 | 256 | 0.14181 | 22 | 1.6 | < 0.1 |
| 32 | 1,024 | 0.01731 | 121 | 3.4 | < 0.1 |
| 64 | 4,096 | 0.02041 | 1,400 | 14 | 0.1 |
| 128 | 16,384 | 0.01270 | – | 73 | 0.6 |
| 256 | 65,536 | 0.00499 | – | 410 | 1.7 |

the 1-D closed form handles $10^5 + 10^5$ points in about 27 ms.

**why:** all three solvers agree to five digits at every size, so the differences are pure cost. from 64 to 256, HiGHS grows about $5.5\times$ per doubling (slope ≈ 2.5 on the log–log plot): the LP has $n^2$ variables and each iteration touches many of them. your simplex grows about $12\times$ per doubling (slope ≈ 3.5): Bland's rule takes many small, mostly degenerate pivots (on a uniform $n\times n$ problem, $n - 1$ of the $2n - 1$ basic cells carry zero mass), each costing a full pass over the $n^2$ reduced costs plus a tree walk in Python. the Hungarian algorithm is two orders of magnitude faster than HiGHS on the same problems because it exploits the assignment structure (Birkhoff: the optimum is a permutation) and works with $O(n)$ dual variables in compiled code; its worst case is $O(n^3)$. the 1-D formula is a sort. at the minibatch sizes of topic 17 (a few hundred points, every training step) only the Hungarian algorithm is cheap enough among the exact methods; topic 16's Sinkhorn trades exactness for $O(n^2)$ work per iteration.

## experiment 3: how fast do empirical distances converge?

**what people usually predict:** two samples of 800 points from the same distribution are close, $W_2 \approx 0.05$ or less, in any dimension; the error halves when $n$ quadruples, as for a mean.

**what happens** (exact $W_2$ between two independent samples of $\mathcal N(0, I_d)$, mean of 5 repetitions):

| d | W2 at n = 25 | W2 at n = 800 | shrinks by (32× more points) | log–log slope |
|---|---|---|---|---|
| 1 | 0.405 | 0.079 | 5.1× | −0.48 |
| 2 | 0.771 | 0.221 | 3.5× | −0.36 |
| 5 | 1.879 | 0.922 | 2.0× | −0.20 |
| 10 | 3.156 | 2.098 | 1.5× | −0.11 |

for comparison, the true $W_2$ between $\mathcal N(0, I_d)$ and the same Gaussian shifted by a vector of length 0.5 is 0.5 in every dimension.

**why:** only in 1-D does the empirical distance behave like a mean (slope $-1/2$). for $W_2$ in $d > 4$ dimensions (the condition $d > 2p$ of Fournier & Guillin, 2015) it converges like $n^{-1/d}$, measured here as slopes of $-0.20$ at $d = 5$ and $-0.11$ at $d = 10$. below that the rate is faster and depends on the distribution; $d = 2$ is the critical dimension for the matching problem itself, where uniform samples are matched at a cost of order $\sqrt{\log n / n}$ (Ajtai, Komlós & Tusnády, 1984); the Gaussian samples here, with their unbounded tails, converge more slowly (slope $-0.36$). the reason is geometric: an optimal matching has to send each point of one sample to a point of the other, and $n$ points spread through a $d$-dimensional ball are typically $\sim n^{-1/d}$ apart. at $d = 10$ and $n = 800$ the sampling floor (2.1) is four times the distance between two genuinely different Gaussians (0.5): a raw sample-to-sample $W_2$ can't tell them apart at this sample size. so whenever this block measures quality with an empirical $W_2$ (topic 17's project does, in 2-D), it also reports the floor, the distance between two independent samples of real data at the same $n$, and only differences above it mean anything. debiasing (topic 16's Sinkhorn divergence) and fitting in low-dimensional features (FID's Gaussian fit) are two common answers.

## references

- Fournier & Guillin (2015), "On the rate of convergence in Wasserstein distance of the empirical measure", *Probability Theory and Related Fields* 162. [arXiv:1312.2128](https://arxiv.org/abs/1312.2128)
- Ajtai, Komlós & Tusnády (1984), "On optimal matchings", *Combinatorica* 4(4). [doi:10.1007/BF02579135](https://doi.org/10.1007/BF02579135)
- Arjovsky, Chintala & Bottou (2017), "Wasserstein generative adversarial networks", *ICML*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
