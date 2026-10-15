# normalizing flows · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 15–20; the lab took 167 s there, and 47 s at a load of about 5). the data are synthetic (`mlfs.datasets.eight_gaussians`, `make_moons`), so offline runs are the same experiments with 600 instead of 2,000 training steps. the lab trains each flow once (seed 0); the seed spreads below come from two extra seeds of the same runs.

## experiment 1: exact likelihoods

**what people usually predict:** a well-trained 8-layer flow gets within a few hundredths of a nat of the truth on such a simple density.

**what happens:**

| data | $\int p(x)\,dx$ on the grid | flow test NLL | true test NLL | gap = KL (nats) |
|---|---|---|---|---|
| 8 gaussians | 1.0107 | −0.409 | −1.086 | **0.677** |
| two moons | 1.0003 | 0.391 | — | — |

with seeds 1 and 2 the 8-gaussian gap was 1.261 and 0.765: **0.90 ± 0.26 nats** over three seeds. both densities integrate to 1 within the grid's accuracy (the 8-gaussian sum is 1% off because the grid spacing, 0.03, is coarse next to the modes' width of 0.05). the density plot shows the 8 modes joined by thin streaks and bridges of low density.

**why:** the density integrates to 1 because of the construction, not the training: $p_X(x) = p_Z(f(x))\,|\det J_f(x)|$ is the exact change of variables for a bijection, and $p_Z$ integrates to 1, so every RealNVP defines a normalized density for any weights (the checks test this at random initialization). the gap to the truth is $\operatorname{KL}(p_\text{data} \| p_\text{flow})$, and it's large here for two reasons visible in the plot: the modes are slightly too wide, and mass leaks onto streaks and bridges between them (experiment 2). maximum likelihood can't make a model "too sharp" in the GAN sense: putting zero density where data occur costs $-\log 0 = \infty$, so it errs on the side of spreading.

## experiment 2: can a flow separate the modes?

**what people usually predict:** a likelihood model covers all 8 modes, and puts almost nothing between them, like the true mixture (1.1%).

**what happens** (5,000 samples, seed 0):

| | modes covered | within 3 std of a mode | between modes |
|---|---|---|---|
| your flow | 8 | 0.699 | **0.301** |
| true mixture | 8 | 0.987 | 0.013 |

all 8 modes are covered (between 299 and 597 of the 5,000 samples each, against 625 expected), but **30% of the samples land between modes**, 23 times the true fraction (seeds 1 and 2: 0.510 and 0.256, so 0.36 ± 0.11 over three seeds). the sample plot shows thin curved bridges joining neighbouring modes, and the latent-space panel shows why: the flow maps the 8 clusters of data to 8 adjacent bands that together fill the Gaussian blob, and latent points near the boundary between two bands, where the map has to jump from one mode to the next, land on the bridges.

**why:** a flow is a continuous bijection, so it can't tear the single connected Gaussian blob into 8 disconnected islands (on paper 3). to put little mass between two modes it must squeeze the latent region between them into a sliver, which needs very large Jacobians on a very small set; a network with bounded scales approximates that with thin bridges, and samples drawn near the band boundaries land on them. compared with topic 05's GAN on the same kind of data, the failure is the mirror image: the GAN dropped modes and kept its samples sharp (high precision, low recall); the flow keeps every mode and pays in precision (high recall, low precision). which you'd rather have depends on the use: for density estimation or compression, never missing data is what matters; for pretty samples, precision is.

## experiment 3: depth

**what people usually predict:** more layers, steadily better: going from 8 to 12 should help noticeably.

**what happens** (seed 0 in the lab; the gap and between-modes fraction with seeds 1 and 2 in brackets):

| layers | parameters | test NLL | gap (nats) | modes covered | between modes | training time |
|---|---|---|---|---|---|---|
| 2 | 34,828 | 2.878 | 3.964 | 3 | 0.952 | 9 s |
| 4 | 69,656 | 0.236 | 1.322 [1.109, 1.247] | 8 [8, 7] | 0.541 [0.453, 0.524] | 15 s |
| 8 | 139,312 | −0.409 | 0.677 [1.261, 0.765] | 8 [8, 8] | 0.301 [0.510, 0.256] | 25 s |
| 12 | 208,968 | −0.305 | 0.781 [0.558, 0.965] | 8 [8, 8] | 0.214 [0.159, 0.316] | 36 s |

two layers are far too few (each coordinate is transformed once, conditioned on the other once; five modes are missing and 95% of the samples are between modes). four layers capture all or nearly all modes, with half the samples still between them. beyond that the returns flatten and the **seed-to-seed spread is as large as the effect of depth**: on seed 0, 12 layers look worse than 8 on the NLL; averaged over three seeds, 12 layers are slightly better (gap 0.77 vs 0.90 nats, between-modes fraction 0.23 vs 0.36), at 1.5 times the training time, with overlapping ranges.

**why:** a coupling layer transforms half the coordinates conditioned on the other half, so in 2-D, a coordinate needs at least two layers to depend on the other one and be transformed by it; complex dependencies need several rounds. after that, the bottleneck is no longer expressiveness but optimization and topology: deeper flows can sharpen the bridges further, but each layer's scale is bounded and the training budget is fixed (2,000 steps), so the extra layers are under-trained. the general lesson matches topic 05's: with one seed per configuration you can't rank configurations whose differences are smaller than the seed spread.

## references

- Dinh, Sohl-Dickstein & Bengio (2017), "Density estimation using Real NVP", *ICLR*. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- Cornish, Caterini, Deligiannidis & Doucet (2020), "Relaxing bijectivity constraints with continuously indexed normalising flows", *ICML*. [arXiv:1909.13833](https://arxiv.org/abs/1909.13833)
