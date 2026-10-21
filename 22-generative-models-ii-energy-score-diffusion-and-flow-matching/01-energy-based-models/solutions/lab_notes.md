# energy-based models · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU (one thread, as in CI) on a shared machine (load average about 5); the whole lab took about 70 s there. the data are synthetic (`mlfs.datasets.eight_gaussians`), so offline runs are the same experiments with the same numbers. the seed spreads below come from the two training seeds the lab runs, plus a few extra runs where noted.

## experiment 1: the price of a finite step

**what people usually predict:** a small bias at most, shrinking with longer chains; MALA "about as good".

**what happens** (20,000 chains, 300 steps from 0):

| h | theory (ULA) | Langevin variance | MALA variance | MALA acceptance |
|---|---|---|---|---|
| 0.1 | 1.026 | 1.025 | 0.994 | 0.998 |
| 0.5 | 1.143 | 1.158 | 0.995 | 0.972 |
| 1.0 | 1.333 | 1.349 | 1.000 | 0.921 |
| 2.0 | 2.000 | 2.014 | 1.000 | 0.783 |
| 3.0 | 4.000 | 3.991 | 1.002 | 0.633 |
| 3.9 | 40.000 | 39.913 | 0.992 | 0.512 |

the measured variances follow $1/(1 - h/4)$ to within the sampling noise of 20,000 chains (about ±1–2%): at $h = 2$ the "samples of $\mathcal{N}(0, 1)$" have variance 2, at $h = 3.9$ forty. MALA stays at 1 for every step size; its acceptance rate drops from 99.8% to 51%.

**why:** a longer chain converges to the *stationary distribution of the discretized chain*, which is not $p$. on a Gaussian the recursion is $x' = (1 - h/2)x + \sqrt h z$; its fixed-point variance is $h/(1 - (1 - h/2)^2) = 1/(1 - h/4)$, and at $h \ge 4$ the factor $|1 - h/2| \ge 1$ makes the recursion unstable, exactly like gradient descent with a learning rate above $2/L$ on a quadratic of curvature $L$ (here $L = 1$ and the drift step is $h/2$). MALA turns the same proposal into an exact sampler by rejecting moves in proportion to how wrong they are; the price is rejections (wasted steps) and one more energy evaluation per step. in $D$ dimensions the acceptance rate of a fixed step decays with $D$ (the optimal step scales like $D^{-1/3}$), so for images MALA needs tiny steps, which is why EBM practitioners mostly accept the bias.

## experiment 2: short-run chains and what they teach the energy

**what people usually predict:** longer chains give better samples, since MCMC "converges"; persistent training should be the more faithful of the two.

**what happens** (2,400 training iterations; 2,000 chains from box noise per model):

| training | seed | sampling steps | modes covered | within 3 std | samples per mode | max / min |
|---|---|---|---|---|---|---|
| short-run | 0 | 20 | 8 | 0.700 | 124 – 219 | 1.8 |
| short-run | 0 | 200 | 8 | 0.879 | 162 – 280 | 1.7 |
| short-run | 0 | 2000 | 8 | 0.898 | 149 – 256 | 1.7 |
| short-run | 1 | 20 | 8 | 0.690 | 141 – 209 | 1.5 |
| short-run | 1 | 200 | 8 | 0.869 | 185 – 262 | 1.4 |
| short-run | 1 | 2000 | 8 | 0.900 | 159 – 294 | 1.8 |
| persistent | 0 | 20 | 8 | 0.755 | 130 – 242 | 1.9 |
| persistent | 0 | 200 | 8 | 0.884 | 181 – 275 | 1.5 |
| persistent | 0 | 2000 | 8 | 0.931 | 99 – 334 | **3.4** |
| persistent | 1 | 20 | 8 | 0.756 | 132 – 243 | 1.8 |
| persistent | 1 | 200 | 8 | 0.892 | 164 – 277 | 1.7 |
| persistent | 1 | 2000 | 8 | 0.916 | 80 – 333 | **4.2** |

both schemes learn all 8 modes. running the chains 10 and 100 times longer than they were trained with makes every mode sharper (within-3-std fraction 0.69–0.76 at 20 steps, 0.90–0.93 at 2,000) but not better balanced: the max/min ratio of samples per mode stays at 1.4–1.8 for the short-run models and grows to **3.4–4.2** for the persistent ones at 2,000 steps (the scatter plot shows the diagonal modes thinned out). 2,000 chains give about 250 ± 15 per mode by chance, so ratios above about 1.3 are not sampling noise (though only the ones within 3 std are counted, so the counts are lower).

an earlier version of this lab trained for only 800 iterations: there the short-run models gave the larger long-run imbalance (16 on seed 0, 3.3 on seed 1) and the persistent models had not yet sharpened (their samples and densities were rings). so which scheme ends up more lopsided depends on the training budget and the seed; what doesn't change is that neither pins the mode weights down.

**why:** contrastive divergence stops when the negative samples match the data (on paper 4). during training no chain, short-run or persistent, travels from one well-separated mode to another (topic 02's experiment 1 shows why), so the gradient never compares the depths of two wells: their relative depths, which set how a long, free-running chain splits its mass, are essentially unconstrained. for a short-run model the negatives are the output of a fixed 20-step sampler from box noise, so the energy only has to make *that* sampler's output match the data (the short-run phenomenon of Nijkamp et al., 2019). for a persistent model the buffer's chains keep whatever share of each mode they first fell into (only the 5% re-initialization moves mass between modes), and contrastive divergence then raises the energy of over-represented modes and lowers that of under-represented ones until the *buffer* looks like the data. the well depths absorb the buffer's accidents, and a fresh 2,000-step chain, which settles by basin and depth rather than by the buffer's history, exposes them.

## experiment 3: is the energy a good density?

**what people usually predict:** a trained EBM is a decent density model; maybe within a few tenths of a nat.

**what happens** (grid-normalized densities, 2,400 iterations):

| training | seed | test NLL | mass within 3 std of a mode |
|---|---|---|---|
| short-run | 0 | 1.271 | 0.936 |
| short-run | 1 | 1.281 | 0.926 |
| persistent | 0 | 1.291 | 0.937 |
| persistent | 1 | 1.300 | 0.930 |
| true mixture | — | 1.111 | 0.989 |

all four models are 0.16–0.19 nats per point worse than the truth, with about 93% of their mass near a mode; the two schemes are within 0.03 nats of each other. the density plots show where the missing nats go: the short-run model puts streaks of mass along the diagonals towards the corners of the plot, outside the box its chains start in; the persistent model is compact but its diagonal wells are dimmer than the axis-aligned ones (the same imbalance experiment 2's long chains found). at 800 iterations the gap to the truth was 0.5–1 nat.

**why:** the NLL needs $Z_\theta$, which is an integral over the whole space. in two dimensions a $351 \times 351$ grid computes it to plenty of accuracy; in 784 dimensions the same grid resolution would need $351^{784}$ points, and importance-sampling estimates of $Z$ (annealed importance sampling) are expensive and high-variance. this is the deep practical drawback of EBMs: you can sample and train them, but you can hardly evaluate them. the corner streaks are regions no negative sample ever visits (they lie outside the bounding box where chains start, and 20 small steps don't reach them), so nothing pushes their energy up; the uneven wells are experiment 2's unconstrained mode weights seen as a density. closing the remaining gap would take more iterations (cheapest), more Langevin steps per iteration (cost grows linearly), a larger step (faster mixing but more bias, experiment 1), or negatives initialized over a wider region.

## references

- Nijkamp, Hill, Zhu & Wu (2019), "Learning non-convergent non-persistent short-run MCMC toward energy-based model", *NeurIPS*. [arXiv:1904.09770](https://arxiv.org/abs/1904.09770)
- Du & Mordatch (2019), "Implicit generation and modeling with energy-based models", *NeurIPS*. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
- Roberts & Rosenthal (1998), "Optimal scaling of discrete approximations to Langevin diffusions", *JRSS B* 60(1). the $D^{-1/3}$ scaling of MALA. [doi:10.1111/1467-9868.00123](https://doi.org/10.1111/1467-9868.00123)
