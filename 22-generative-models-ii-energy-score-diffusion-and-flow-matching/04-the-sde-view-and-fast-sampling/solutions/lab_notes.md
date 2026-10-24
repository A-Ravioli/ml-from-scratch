# the SDE view & fast sampling · lab notes

experiments 1 and 3 use synthetic data with the exact noise predictor, so offline runs give the same numbers; they were measured with the reference solution on an Apple M2 laptop CPU (one thread). experiment 2 uses the real MNIST digits on the laptop's GPU (`mps`), shared with other jobs; by default it trains for 1,000 steps (a button runs 4,000); offline it is only a smoke test (40 training steps on the synthetic stand-in, 64 samples, no long samplers), and its numbers mean nothing. sampling noise on a quality fraction from 4,000 samples is about ±0.008.

## experiment 1: quality per network call, with a perfect network

**what people usually predict:** the fourth-order solver wins at any budget; the SDE is about as good as the ODE.

**what happens** (fraction of 4,000 samples within 3 std of a mode; the true mixture: 0.99):

| sampler | 5 NFE | 10 NFE | 20 NFE | 50 NFE | 200 NFE |
|---|---|---|---|---|---|
| DDIM | 0.018 | 0.694 | **0.973** | 0.997 | 0.993 |
| PF-ODE Euler | 0.048 | 0.493 | 0.968 | 0.992 | 0.992 |
| reverse SDE | 0.016 | 0.026 | 0.102 | 0.651 | 0.984 |
| PF-ODE RK4 | — | 0.128 | 0.389 | 0.960 | 0.989 |

ancestral DDPM with 1,000 calls: 0.989 (1 s for 4,000 two-dimensional samples). at 20 calls DDIM and Euler are already at 97%; RK4, the most accurate solver per *step*, reaches only 39% at 20 calls and needs 50; the reverse SDE needs about 200.

**why:** the network is exact, so every gap is discretization error. a deterministic solver's error shrinks like a power of the step size; the SDE adds fresh noise of variance $\beta h$ at every step, and its Euler–Maruyama discretization converges much more slowly (its weak error is first order, and the noise has to "average out" over many steps before the modes come into focus). RK4 spends 4 calls per step, so at 20 calls it takes only 4 steps, and the drift is far from smooth: near $t = 0$ the noisy density has 8 sharp modes of width 0.05, so the score changes over tiny distances, and the piecewise-constant $\beta(t)$ of a discrete schedule has a kink at every grid point; high order only pays when the solution is smooth over a step. DDIM is Euler in the coordinates $(\bar x, \sigma)$ where the PF-ODE is closest to straight, which is why it edges out Euler in $t$. at 20 calls, ship DDIM.

## experiment 2: steps against FID on digits

**what people usually predict:** 20-step DDIM is a little worse than 1,000-step DDPM; more steps are always better.

**what happens** (MNIST, `mps`, shared with other jobs). the default run: 1,000 training steps in 193 s, 500 samples per row; the whole lab took 3.9 minutes. 500 real training digits score 12.5 against the test digits, the floor at this sample size.

| sampler | network calls | FID (course) | sampling 500 digits |
|---|---|---|---|
| DDIM | 10 | 184.5 | 0.4 s |
| DDIM | 20 | 148.6 | 1.5 s |
| DDIM | 50 | 133.0 | 4.8 s |
| PF-ODE Euler | 20 | 148.2 | 2.1 s |
| PF-ODE Euler | 50 | 133.2 | 5.6 s |
| reverse SDE | 50 | **111.0** | 5.9 s |

the longer run (the button: 4,000 training steps in 13 minutes, 1,000 samples per row, the whole lab 18 minutes; floor 8.8 at 1,000 samples):

| sampler | network calls | FID (course) | sampling 1,000 digits |
|---|---|---|---|
| DDIM | 10 | 97.6 | 1.2 s |
| DDIM | 20 | 80.8 | 3.1 s |
| DDIM | 50 | **74.4** | 9.4 s |
| PF-ODE Euler | 20 | 101.9 | 4.1 s |
| PF-ODE Euler | 50 | 80.8 | 10.6 s |
| reverse SDE | 50 | 91.1 | 11.8 s |
| reverse SDE | 200 | 95.2 | 46.1 s |
| ancestral DDPM | 1000 | 83.2 | 221 s |

both models are far from the data, the default one much further. the two deterministic samplers agree closely at equal calls (DDIM and Euler are the same ODE in different coordinates), and every sampler improves from 10 to 50 calls. the ranking of the stochastic sampler flips with training: after 1,000 steps the 50-step reverse SDE is the best row by 22 points; after 4,000 steps it is 17 points behind 50-step DDIM, and 20 DDIM calls match the 1,000-call ancestral sampler (80.8 vs 83.2) at 70 times less sampling time. the pictures show where the gap comes from: in the default run about half of the deterministic samplers' digits are buried in white speckle (the same ones in every DDIM and Euler row, since they start from the same noise and follow nearly the same path), while the reverse SDE draws clean if odd strokes on black; after 4,000 steps the speckle has faded to a grey background with some malformed digits, and the stochastic samplers' cleaner digits no longer score better.

**why:** the network is imperfect, mostly at low noise, where predicting $\epsilon$ is hardest (topic 03's lab): its leftover errors show up as speckle that a deterministic sampler, which never re-randomizes, keeps until the end; stochastic samplers wash such errors out (experiment 3). the worse the network, the more that matters, which is why the SDE wins after 1,000 steps; as the network improves, the ODE's smaller discretization error takes over, and the stochastic samplers' final steps also blur and thicken strokes and may cover fewer of the test set's styles, which the feature network's statistics penalize. one training seed and one sample draw per row; differences of a few FID points between rows are within the noise (FID's sampling noise is a few points for a model this far from the data, more at 500 samples), so the robust conclusions are the large ones: deterministic samplers need about 20–50 calls, not 1,000, and which family of sampler wins depends on how good the network is, as experiment 3 shows with a known error. FIDs are course-internal (chapter 21's `FeatureNet`, not Inception).

## experiment 3: when the network is wrong

**what people usually predict:** more steps always help; the deterministic samplers, being more accurate, stay ahead.

**what happens** (noise predictor scaled by 0.9; fraction within 3 std of a mode):

| network calls | DDIM (η = 0) | DDIM (η = 1) | PF-ODE Euler | reverse SDE |
|---|---|---|---|---|
| 50 | 0.860 | **0.998** | 0.775 | 0.666 |
| 200 | 0.782 | 0.996 | 0.770 | 0.968 |
| 1000 | 0.750 | 0.985 | 0.721 | **0.978** |

the deterministic samplers get *worse* with more steps and settle around 72–75%: the scatter plots show every mode too wide. the stochastic ones recover: DDIM with $\eta = 1$ is at 99–100% at every budget, and the reverse SDE climbs to 98% once it has enough steps.

**why:** a 10% too-small $\epsilon$ is a 10% too-weak score. the ODE follows the drift $-\frac12\beta(x + s)$ exactly as given, so with enough steps it converges to the solution of the *wrong* ODE, whose final distribution is too spread out; more steps only remove the discretization error that happened to partly cancel the model error (hence the slight decline). the stochastic samplers put twice the weight on the score (drift $\beta(\frac12 x + s)$) and inject noise; each step acts like a small Langevin correction towards the current noisy density, so the samples are continually pulled back towards where the (slightly wrong) score is consistent with the noise level, and systematic errors partly cancel instead of accumulating. this is the practical argument of Karras et al. (2022) for some stochasticity: deterministic samplers are best for an excellent network at small budgets (experiment 1); stochastic ones are more forgiving of an imperfect network, if you can afford the steps.

## references

- Song, Meng & Ermon (2021), "Denoising diffusion implicit models", *ICLR*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Karras, Aittala, Aila & Laine (2022), "Elucidating the design space of diffusion-based generative models", *NeurIPS*. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Song et al. (2021), "Score-based generative modeling through stochastic differential equations", *ICLR*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
