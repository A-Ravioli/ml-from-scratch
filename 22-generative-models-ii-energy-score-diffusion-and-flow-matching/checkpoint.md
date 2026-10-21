# 22 · checkpoint

pass this to finish the chapter, or to skip it. work without notes, then check the answers. readiness: at least 80% on parts 1 and 2 combined, and a passing part 3.

## 1. concept check

1. why does Langevin dynamics, and every sampler in this chapter, need the score $\nabla_x\log p(x)$ but never the normalizer of $p$?

<details><summary>answer</summary>

the normalizer is a constant in $x$, so it vanishes from $\nabla_x\log p$. samplers only ever move points along the score (plus noise); MALA additionally uses differences of $\log p$, where the normalizer cancels too.

</details>

2. unadjusted Langevin with step $h$ on $\mathcal{N}(0, 1)$: what stationary variance, and why doesn't running longer fix it?

<details><summary>answer</summary>

$1/(1 - h/4)$. a longer chain converges to the stationary distribution of the *discretized* recursion, which is biased; only a smaller step or a Metropolis correction (MALA) removes the bias.

</details>

3. what's the difference between persistent and short-run contrastive divergence, and how can a short-run EBM have good samples but a bad density?

<details><summary>answer</summary>

persistent chains continue across training iterations and stay near the model's equilibrium; short-run chains restart from noise and run a fixed number of steps. contrastive divergence stops when the negatives match the data, so a short-run model only has to make its fixed-length sampler land on the data; the stationary density $e^{-E}/Z$ (e.g. the relative depths of separated wells) is unconstrained, and longer chains expose it.

</details>

4. denoising score matching regresses on $-(\tilde x - x)/\sigma^2$, a target that depends on the unknown clean point. why does that learn the score of anything?

<details><summary>answer</summary>

the least-squares minimizer is the conditional mean of the target given the input $\tilde x$, $(\mathbb{E}[x \mid \tilde x] - \tilde x)/\sigma^2$, which by Tweedie's formula equals $\nabla\log q_\sigma(\tilde x)$, the score of the blurred data. the loss never reaches zero; its floor is the conditional variance.

</details>

5. Langevin with the *exact* score of a mixture $0.2\,\mathcal{N}(-5, 1) + 0.8\,\mathcal{N}(5, 1)$, started uniformly: what fraction ends near $-5$, and what fixes it?

<details><summary>answer</summary>

about the fraction that *started* on that side (half), because the weights only affect the score in the near-empty region between the modes, and chains don't cross it. annealing through large noise levels, where the modes are connected and the weights visible, fixes it.

</details>

6. in DDPM, why can you train on a random step $t$ without simulating the forward chain?

<details><summary>answer</summary>

$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1 - \bar\alpha_t)I)$ in closed form, and the ELBO is a sum of per-step terms, so a uniformly random step per example gives an unbiased estimate.

</details>

7. at which noise levels is predicting $\epsilon$ hardest, and why?

<details><summary>answer</summary>

the smallest ones: there the noise is a tiny perturbation, indistinguishable from the data's own fine detail; for data on a $d$-dimensional manifold in $D$ dimensions, the loss plateaus near $d/D$ once the noise exceeds the manifold's thickness (the $d$ noise components along the manifold can't be told apart from a shift of $x_0$). at large noise $x_t \approx \epsilon$ and the prediction is easy.

</details>

8. how is DDIM related to the probability-flow ODE?

<details><summary>answer</summary>

with $\bar x = x/\sqrt{\bar\alpha}$ and $\sigma = \sqrt{(1 - \bar\alpha)/\bar\alpha}$, the $\eta = 0$ DDIM update is one Euler step of $d\bar x/d\sigma = \epsilon_\theta(x, t)$, which is the PF-ODE in those coordinates.

</details>

9. even with a perfect network, 10 DDIM steps shrink a Gaussian $\mathcal{N}(0, 0.5^2)$ to a standard deviation of about 0.36. why?

<details><summary>answer</summary>

each step re-noises the posterior mean $\hat x_0 = \mathbb{E}[x_0 \mid x_t]$, which has less variance than $x_0$, using the inferred noise rather than fresh noise; with big steps the lost variance isn't restored (a Cauchy–Schwarz inequality between the two steps' coefficient vectors). it's discretization error, and it vanishes as the steps shrink.

</details>

10. which samplers recover when the network systematically underestimates the score, and why?

<details><summary>answer</summary>

stochastic ones (DDIM with $\eta = 1$, the reverse SDE, ancestral sampling): their drift on the score is twice the ODE's and they inject fresh noise, so each step acts like a Langevin correction at the current noise level. deterministic samplers converge to the wrong ODE's solution however many steps they take.

</details>

11. in flow matching with independent noise and data, why does the loss have a floor, and why do the trajectories curve although every training path is straight?

<details><summary>answer</summary>

many straight lines with different directions pass through the same $(x_t, t)$; the network can only learn their average, the marginal velocity, leaving their variance as an irreducible loss ($\pi/2$ per dimension for standard normal data). the average direction changes along a trajectory (at $t = 0$ it points at the data mean), so trajectories bend.

</details>

12. what does reflow do, and what can't it improve?

<details><summary>answer</summary>

it retrains on (noise, sample) pairs produced by the current flow's ODE, a coupling without crossings, so the new flow's paths are nearly straight and a single Euler step nearly suffices. it can't improve the distribution: the pairs carry the first model's errors.

</details>

13. why must classifier-free guidance be trained with label dropout, and what does $w > 1$ do to samples of overlapping classes?

<details><summary>answer</summary>

guidance needs the unconditional prediction $\epsilon(x, t, \varnothing)$; with no dropout the null label is never trained and the extrapolation is from garbage. with $w > 1$ the target is $\propto p(x)p(y \mid x)^w$: samples concentrate where the class is most certain, sharper than the class and pushed past its centre away from other classes; accuracy rises, distribution distance (FID) worsens.

</details>

14. what is adaLN-Zero, and why zero?

<details><summary>answer</summary>

the conditioning vector sets each block's layer-norm shift and scale and a gate on each residual branch; the layer producing them is initialized to zero, so every gate starts at 0 and every block starts as the identity, and the network grows into its depth during training.

</details>

15. a latent diffusion model is perfect. what limits its samples, and how do you measure that limit?

<details><summary>answer</summary>

the autoencoder: decoded samples have the distribution of the reconstructions $D(E(x))$, so the FID of the reconstructions is the floor for the FID of the samples.

</details>

## 2. by hand

1. **likelihood bookkeeping for a diffusion step.** a schedule with $\beta = (0.1, 0.2)$. (a) compute $\bar\alpha_0, \bar\alpha_1$. (b) a pixel $x_0 = 0.5$ and noise $\epsilon = 1$: what is $x_1$? (c) with a perfect noise prediction, compute the ancestral mean $\mu$ for $x_0$ from $x_1$, and the posterior variance $\tilde\beta_1$. (d) what is the DDIM ($\eta = 0$) jump from step 1 straight to $\hat x_0$?

<details><summary>solution</summary>

(a) $\alpha = (0.9, 0.8)$, $\bar\alpha = (0.9, 0.72)$. (b) $x_1 = \sqrt{0.72}\cdot 0.5 + \sqrt{0.28}\cdot 1 = 0.4243 + 0.5292 = 0.9534$. (c) $\mu = (x_1 - \frac{0.2}{\sqrt{0.28}}\cdot 1)/\sqrt{0.8} = (0.9534 - 0.3780)/0.8944 = 0.6433$; $\tilde\beta_1 = \frac{1 - 0.9}{1 - 0.72}\cdot 0.2 = 0.0714$. (d) $\hat x_0 = (x_1 - \sqrt{0.28}\cdot 1)/\sqrt{0.72} = 0.4243/0.8485 = 0.5$: exact, because the noise prediction is perfect. (the ancestral mean 0.643 is the posterior mean of $x_{0}$ at the discrete step, which here is the noisy level $\bar\alpha_0 = 0.9$, not the clean image.)

</details>

2. **flow matching on a Gaussian.** data $\mathcal{N}(2, 0.5^2)$ in one dimension, noise $\mathcal{N}(0, 1)$, straight paths. (a) write the marginal velocity at $t = 0$ and $t = 1$. (b) where does one Euler step of size 1 from $x_0 = 1$ land, and where does the exact flow send it? (c) compute the irreducible loss density $s^2/((1 - t)^2 + t^2 s^2)$ at $t = 0$, $\frac12$ and 1.

<details><summary>solution</summary>

(a) $v^*(x, t) = \mu + \frac{ts^2 - (1 - t)}{(1 - t)^2 + t^2s^2}(x - t\mu)$: at $t = 0$, $2 - x$; at $t = 1$, $2 + (x - 2) = x$. (b) one Euler step: $1 + (2 - 1) = 2$, the mean (every starting point lands there). the exact flow is the monotone map $x_1 = \mu + s x_0 = 2.5$. (c) $0.25/1 = 0.25$; $0.25/(0.25 + 0.0625) = 0.8$; $0.25/0.25 = 1$.

</details>

3. **guidance in closed form.** $p(x) = \mathcal{N}(0, 2)$ and $p(x \mid y) = \mathcal{N}(1, 0.5)$ (variances). find the mean and variance of $p(x)^{1 - w}p(x \mid y)^w$ for $w = 1$ and $w = 3$.

<details><summary>solution</summary>

precision $P = (1 - w)/2 + w/0.5 = (1 - w)/2 + 2w$; mean $= (w \cdot 1/0.5)/P = 2w/P$. $w = 1$: $P = 2$, mean 1, variance 0.5 (the class itself). $w = 3$: $P = -1 + 6 = 5$, mean $6/5 = 1.2$, variance 0.2: narrower than the class and shifted past its mean, away from the population.

</details>

## 3. blind rebuild

**60 minutes, no notes.** in a fresh copy of `03-diffusion-ddpm/exercise.py`, implement stages 1–4 from memory: `DiffusionSchedule` (linear and cosine, `q_sample`, `to`), `sinusoidal_features`, `TimeEmbedding`, `TimeMLP`, `ddpm_loss`, `fit_ddpm` and `ddpm_sample`. you pass when

```bash
python -m pytest 22-generative-models-ii-energy-score-diffusion-and-flow-matching/03-diffusion-ddpm/test_exercise.py -k "stage1 or stage2 or stage3 or stage4"
```

is green, including the closed-form Gaussian checks (your sampler with the exact noise predictor must reproduce $\mathcal{N}(1.5, 0.5^2)$, and the exact predictor's loss must equal the irreducible floor).

## 4. readiness

move on when parts 1 and 2 together are at least 80% right and part 3 passes. if you're testing out of the chapter, also skim the capstone brief: its DDPM-against-flow-matching table at equal network calls and the guidance tradeoff are the chapter's two practical lessons.
