# markov chain monte carlo

## why this matters

the posterior of almost any interesting model is known only up to a constant: Bayes' rule gives you $p(\theta \mid y) \propto p(y \mid \theta)\,p(\theta)$, and the normalizer $\int p(y \mid \theta)p(\theta)\,d\theta$ is an integral over every parameter at once. for Bayesian linear regression (topic 01) it happened to be a Gaussian integral; for a logistic regression, a hierarchical model, or anything with a few dozen parameters and no conjugacy, it's hopeless to compute, and exact inference (topic 03) is out because the parameters are continuous and densely connected.

**Markov chain Monte Carlo** gets around the normalizer entirely. you build a random walk over parameter space whose long-run distribution is the posterior, using only *ratios* of the unnormalized density, in which the constant cancels. run it long enough and the places it visits are draws from the posterior: average a function over them and you have its posterior expectation. this one idea made Bayesian statistics practical (Stan, PyMC and every probabilistic programming language run on its descendants).

the catch is that the draws are **correlated**, sometimes extremely so, and a chain can look perfectly healthy while exploring only part of the distribution. so this topic builds the samplers *and* the diagnostics that tell you what they're worth: the autocorrelation function, the effective sample size, and $\hat R$. getting the diagnostics right is as important as getting the sampler right.

## warm-up: predict first

1. a random-walk sampler accepts 99% of its proposals. is that good?
2. you run 10,000 MCMC steps and the draws are strongly correlated. roughly how many independent draws are they worth: 10,000, 1,000, or "it depends"?
3. Gibbs sampling updates one coordinate at a time from its exact conditional and never rejects. can it still mix badly?
4. a single chain's trace plot looks like stationary noise and its effective sample size is in the hundreds. can its estimate of the mean still be badly wrong?
5. why would you split each chain in half before comparing chains?

## prerequisites check

- [ ] you know Markov chains, stationary distributions and mixing (→ ch 04 · topic 07)
- [ ] you know the Monte Carlo estimate of an expectation and its standard error $\sigma/\sqrt n$ (→ ch 01, ch 04 · topic 05)
- [ ] you can sample a Gaussian and draw uniform numbers from a `Generator` (→ ch 04 · topic 02)
- [ ] you know the FFT computes circular convolutions (→ ch 01, or just trust `np.fft`)

## core ideas

### idea 1: Metropolis–Hastings

you want a Markov chain with transition kernel $T(x \to x')$ whose stationary distribution is the target $\pi$. a sufficient condition is **detailed balance**: $\pi(x)\,T(x \to x') = \pi(x')\,T(x' \to x)$ for every pair, i.e. in equilibrium as much probability flows from $x$ to $x'$ as back. summing both sides over $x$ shows $\pi$ is stationary.

Metropolis–Hastings builds such a kernel from *any* proposal $q(x' \mid x)$: propose $x'$, then accept it with probability

$$a(x \to x') = \min\left(1,\ \frac{\pi(x')\,q(x \mid x')}{\pi(x)\,q(x' \mid x)}\right),$$

and otherwise **stay where you are** (and record $x$ again). plugging in shows detailed balance holds (on paper 1). the target only enters as a ratio, so $\pi$ can be unnormalized. with a symmetric proposal, like the **random walk** $x' = x + s\,\varepsilon$, $\varepsilon \sim \mathcal N(0, I)$, the $q$'s cancel and you accept with probability $\min(1, \pi(x')/\pi(x))$: always go uphill, sometimes go downhill. in log space: accept if $\log u < \log\pi(x') - \log\pi(x)$ with $u \sim \text{Uniform}(0, 1)$.

the proposal scale $s$ is the one knob, and it's a trade-off. tiny steps are almost always accepted but go nowhere (warm-up 1: 99% acceptance usually means a crawling chain); huge steps land in the tails and are almost always rejected, so the chain sits still. for a Gaussian target in $d$ dimensions the optimum is about $s = 2.38/\sqrt d$ in units of the target's scale, with an acceptance rate of about 0.44 in 1-D falling to 0.234 as $d$ grows (Roberts, Gelman & Gilks, 1997), and the cost of a random walk grows like $d$.

two bookkeeping options. **burn-in** discards the first steps, while the chain travels from its arbitrary start to where the mass is. **thinning** keeps every $k$-th draw to save memory (it never *increases* the information per step).

> **spec.** `metropolis_hastings(log_prob, x0, n_samples, *, proposal_scale=1.0, burn_in=0, thin=1, rng=None)`: `log_prob` maps a float array `(d,)` to a float (it may return `-inf`). run `burn_in + n_samples * thin` steps; at each one draw `z = rng.standard_normal(d)` (rng from `as_rng`), propose `x + proposal_scale * z`, then draw `u = rng.random()` and accept if $\log u <$ the log ratio. evaluate `log_prob` exactly once at `x0` and once per proposal. after burn-in, record the state after every `thin`-th step. return `(samples (n_samples, d), accept_rate)`, where the acceptance rate is over *all* steps.

> **🔮 predict:** a random-walk Metropolis sampler on a standard normal in one dimension. which proposal scale mixes best: 0.1, 2.5 or 20? what fraction of proposals will it accept?

> **▶ build: stage 1.** implement `metropolis_hastings`.

### idea 2: Gibbs sampling

if you can sample each coordinate from its **full conditional** $\pi(x_i \mid x_{-i})$, you don't need proposals at all: sweep through the coordinates, replacing each by a draw from its conditional given the current values of all the others (including the ones you just updated). each update leaves $\pi$ invariant (it's an MH step with acceptance probability exactly 1, on paper 2), so the whole sweep does too.

full conditionals are often easy even when the joint is not. in a graphical model, $\pi(x_i \mid x_{-i})$ depends only on $x_i$'s Markov blanket (topic 02); in conjugate models it's a standard distribution. the classic examples: Gaussian mixtures (sample assignments given means, means given assignments), hierarchical models, topic models, Ising models.

but "never rejects" doesn't mean "mixes well" (warm-up 3). each update moves parallel to a coordinate axis, and when two coordinates are strongly correlated their conditional distributions are narrow, so each step is tiny. for a bivariate normal with correlation $\rho$, the $x_1$ chain is an AR(1) process with coefficient $\rho^2$: at $\rho = 0.99$ you need about a hundred sweeps per independent draw. the lab shows the zig-zag.

> **spec.** `gibbs_sampler(conditionals, x0, n_samples, *, burn_in=0, thin=1, rng=None)`: `conditionals[i](x, rng)` returns a new value for coordinate `i` given the full current state `x` (d,). one sweep updates coordinates `0, 1, ..., d-1` in order, in place in a working copy (so later updates see earlier ones). run `burn_in + n_samples * thin` sweeps and record as in MH. don't modify `x0`. return `samples (n_samples, d)`.

> **▶ build: stage 2.** implement `gibbs_sampler`.

### idea 3: autocorrelation

MCMC draws are correlated, because each one is a small modification of the last. the **autocorrelation function** measures how much: for a stationary chain with variance $\sigma^2$, $\rho_k = \operatorname{Cov}(x_t, x_{t+k})/\sigma^2$. estimate it from one chain of length $n$ with

$$\hat\rho_k = \frac{\sum_{t=1}^{n-k}(x_t - \bar x)(x_{t+k} - \bar x)}{\sum_{t=1}^{n}(x_t - \bar x)^2},$$

dividing by the lag-0 sum, not by $n - k$ (the estimate is biased towards 0 at large lags, but much less noisy, and it's what the ESS estimator below expects).

computing all lags directly costs $O(n^2)$: half an hour for a chain of a million draws. the **Wiener–Khinchin theorem** says the autocovariance is the inverse Fourier transform of the power spectrum $|\hat y(\omega)|^2$, so with the FFT it costs $O(n\log n)$. one trap: the FFT computes *circular* correlations, which wrap the end of the chain around onto the start. zero-pad the centred chain to length at least $2n - 1$ and the wrapped terms multiply zeros.

> **spec.** `autocorrelation(x, max_lag=None)` returns $\hat\rho_0, \dots, \hat\rho_{\text{max\_lag}}$ (default: every lag up to $n - 1$) for a 1-D chain, via `np.fft.rfft`/`irfft` on the centred chain zero-padded to at least $2n - 1$. a chain that never moves has zero variance: return all ones.

> **▶ build: stage 3.** implement `autocorrelation`.

### idea 4: effective sample size

for $n$ independent draws, the Monte Carlo error of a mean is $\sigma/\sqrt n$. for a correlated stationary chain, expanding the variance of the average gives (on paper 3)

$$\operatorname{Var}(\bar x) \approx \frac{\sigma^2}{n}\,\tau, \qquad \tau = 1 + 2\sum_{k=1}^{\infty}\rho_k,$$

where $\tau$ is the **integrated autocorrelation time**. the chain is worth $\text{ESS} = n/\tau$ independent draws (warm-up 2: it depends, entirely, on $\tau$). an AR(1) chain $x_t = \phi x_{t-1} + \varepsilon_t$ has $\rho_k = \phi^k$ and $\tau = (1 + \phi)/(1 - \phi)$: $\phi = 0.9$ means 19 steps per independent draw.

estimating $\tau$ is subtle: summing all $n - 1$ estimated autocorrelations gives an estimator whose variance doesn't shrink, because the far lags are pure noise. **Geyer's initial positive sequence** (1992) truncates in a principled way. for a reversible chain the pair sums $\Gamma_m = \rho_{2m} + \rho_{2m+1}$ are positive and decreasing in theory, so sum them only while the estimates stay positive:

$$\hat\tau = -1 + 2\sum_{m=0}^{M-1}\hat\Gamma_m, \qquad M = \text{the first } m \text{ with } \hat\Gamma_m \le 0.$$

(the $-1$ fixes the double-counted $\rho_0$.) Geyer also proposed an *initial monotone sequence* refinement, which additionally forces the $\hat\Gamma_m$ to be non-increasing (replacing each by the minimum of those before it); Stan uses that version. it changes the estimate only slightly, and our checks use the plain positive-sequence rule. for antithetic chains (negative $\rho_1$, like well-tuned HMC) $\hat\tau$ can be below 1, so ESS can exceed $n$; Stan floors $\hat\tau$ at $1/\log_{10}n$ to keep it sane, and so do we.

> **spec.** `effective_sample_size(chain)`: for a 1-D chain, compute $\hat\rho$ with your `autocorrelation`, accumulate pair sums $\hat\Gamma_m$ for $m = 0, 1, \dots$ while $2m + 1 \le n - 1$, stopping *before* the first $\hat\Gamma_m \le 0$; $\hat\tau = \max(-1 + 2\sum\hat\Gamma_m,\ 1/\log_{10}n)$; return `n / tau` as a float. for an `(n, d)` array, return one ESS per column, shape `(d,)`.

> **🔮 predict:** an AR(1) chain with $\phi = 0.5$. what's $\tau$, and what are the first two pair sums $\Gamma_0$, $\Gamma_1$?

> **▶ build: stage 4.** implement `effective_sample_size`.

### idea 5: R-hat

ESS measures how well a chain explores the region it's in. it can't tell you about regions the chain never found (warm-up 4): a chain stuck in one of two modes looks impeccable. the standard defence is to run **several chains** from dispersed starting points and check that they agree. Gelman & Rubin's **potential scale reduction factor** compares the variance *between* chains to the variance *within* them. with $m$ chains of length $n$:

$$W = \frac1m\sum_j s_j^2, \qquad B = \frac{n}{m - 1}\sum_j(\bar x_j - \bar x)^2, \qquad \widehat{\operatorname{var}}^+ = \frac{n - 1}{n}W + \frac1nB, \qquad \hat R = \sqrt{\frac{\widehat{\operatorname{var}}^+}{W}},$$

where $s_j^2$ is chain $j$'s variance (with $n - 1$) and $\bar x_j$ its mean. if the chains all sample the same distribution, $B/n$ is small and $\hat R \approx 1$; if they disagree, $\widehat{\operatorname{var}}^+$ overestimates the spread and $\hat R > 1$. modern practice (Vehtari et al., 2021) demands $\hat R < 1.01$.

a chain can also be non-stationary in a way all chains share: all of them drifting slowly in the same direction. so compute $\hat R$ on **split chains** (warm-up 5): cut each chain into its first and second half and treat them as $2m$ chains; a drifting chain's halves disagree.

> **spec.** `r_hat(chains)`: `chains` of shape `(m, n)`; split each into its first $\lfloor n/2\rfloor$ and last $\lfloor n/2\rfloor$ draws (dropping the middle draw when $n$ is odd), giving $2m$ chains of length $n' = \lfloor n/2\rfloor$; apply the formulas above with $n'$; return a float. for `(m, n, d)` return one value per coordinate, shape `(d,)`.

> **▶ build: stage 5.** implement `r_hat`.

## worked example

**one Metropolis step.** target $\mathcal N(0, 1)$, so $\log\pi(x) = -x^2/2$ up to a constant. at $x = 1$ you propose $x' = 2$: the log ratio is $-\frac12(4 - 1) = -1.5$, so the acceptance probability is $e^{-1.5} = 0.223$. if $u = 0.1$, $\log u = -2.30 < -1.5$: accept, and the chain moves to 2. if $u = 0.5$, $\log u = -0.69 > -1.5$: reject, and the chain records 1 again. a proposal of $x' = 0.5$ is uphill (log ratio $+0.375$): always accepted.

**τ for AR(1) with $\phi = 0.5$.** $\rho_k = 0.5^k$, so $\tau = 1 + 2(0.5 + 0.25 + \dots) = 3$. with pair sums: $\Gamma_m = 0.5^{2m}(1 + 0.5) = 1.5\cdot0.25^m$, so $\Gamma_0 = 1.5$, $\Gamma_1 = 0.375$, and $\tau = -1 + 2\cdot1.5\cdot\frac{1}{1 - 0.25} = 3$. a chain of 3,000 such draws is worth 1,000 independent ones.

**split R-hat by hand.** two chains of four draws, $(0, 2, 1, 3)$ and $(4, 6, 5, 7)$. split: $(0, 2), (1, 3), (4, 6), (5, 7)$, so $n' = 2$. each half has variance 2, so $W = 2$. the half means are $1, 2, 5, 6$, with variance $17/3$, so $B = 2\cdot\frac{17}{3}$. then $\widehat{\operatorname{var}}^+ = \frac12\cdot2 + \frac{17}{3} = \frac{20}{3}$ and $\hat R = \sqrt{10/3} = 1.826$: the second chain lives somewhere else. your `r_hat` must give this number.

## on paper

1. **MH satisfies detailed balance.** show $\pi(x)\,q(x' \mid x)\,a(x \to x') = \pi(x')\,q(x \mid x')\,a(x' \to x)$ for $x \ne x'$.

<details><summary>solution</summary>

let $r = \frac{\pi(x')q(x \mid x')}{\pi(x)q(x' \mid x)}$. if $r \le 1$: the left side is $\pi(x)q(x' \mid x)\,r = \pi(x')q(x \mid x')$, and the right side is $\pi(x')q(x \mid x')\cdot\min(1, 1/r) = \pi(x')q(x \mid x')$. if $r > 1$, swap the roles. the rejection mass stays at $x$, which doesn't affect balance between distinct states.

</details>

2. **Gibbs is MH with acceptance 1.** take the proposal $q(x' \mid x) = \pi(x'_i \mid x_{-i})$ with $x'_{-i} = x_{-i}$. compute the MH acceptance ratio.

<details><summary>solution</summary>

$\frac{\pi(x')\,\pi(x_i \mid x_{-i})}{\pi(x)\,\pi(x'_i \mid x_{-i})} = \frac{\pi(x'_i \mid x_{-i})\pi(x_{-i})\,\pi(x_i \mid x_{-i})}{\pi(x_i \mid x_{-i})\pi(x_{-i})\,\pi(x'_i \mid x_{-i})} = 1$, using $x'_{-i} = x_{-i}$. every proposal is accepted.

</details>

3. **the variance of a correlated average.** for a stationary chain with variance $\sigma^2$ and autocorrelations $\rho_k$, show $n\operatorname{Var}(\bar x) = \sigma^2\big(1 + 2\sum_{k=1}^{n-1}(1 - k/n)\rho_k\big)$, which tends to $\sigma^2\tau$.

<details><summary>solution</summary>

$\operatorname{Var}(\bar x) = \frac1{n^2}\sum_{s,t}\operatorname{Cov}(x_s, x_t) = \frac{\sigma^2}{n^2}\sum_{s,t}\rho_{|s-t|}$. there are $n$ pairs with $|s - t| = 0$ and $2(n - k)$ with $|s - t| = k$, so this is $\frac{\sigma^2}{n^2}\big(n + 2\sum_k(n - k)\rho_k\big)$. multiply by $n$; as $n \to \infty$ the weights $1 - k/n \to 1$ (when $\sum|\rho_k| < \infty$).

</details>

4. **Gibbs on a correlated Gaussian.** for a standard bivariate normal with correlation $\rho$, show that the $x_1$ chain of systematic-scan Gibbs satisfies $x_1^{(t+1)} = \rho^2x_1^{(t)} + \text{noise}$, and find its ESS per sweep.

<details><summary>solution</summary>

$x_2^{(t)} = \rho x_1^{(t)} + \sqrt{1 - \rho^2}\,\varepsilon$ and $x_1^{(t+1)} = \rho x_2^{(t)} + \sqrt{1 - \rho^2}\,\varepsilon'$, so $x_1^{(t+1)} = \rho^2x_1^{(t)} + $ independent noise: AR(1) with $\phi = \rho^2$. ESS per sweep is $(1 - \rho^2)/(1 + \rho^2)$, about $0.01$ at $\rho = 0.99$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how the proposal scale of random-walk Metropolis trades off acceptance and mixing, in 1 and 20 dimensions.
2. what single-chain diagnostics and $\hat R$ say about chains on a bimodal target.
3. how Gibbs sampling slows down as two coordinates become more correlated.

## check your understanding

1. why can MCMC work with an unnormalized density, when ordinary Monte Carlo needs to sample from the normalized one?

<details><summary>answer</summary>

the acceptance ratio $\pi(x')/\pi(x)$ (and Gibbs's conditionals, which are ratios too) don't depend on the normalizing constant. you pay with correlated draws instead of independent ones.

</details>

2. a rejected proposal: why must the chain record the current state *again*, rather than just trying another proposal?

<details><summary>answer</summary>

the rejection mass is part of the transition kernel; dropping it changes the chain's stationary distribution. states with low acceptance (high density, surrounded by lower density) would be under-represented. "repeat until accepted" is a different, biased kernel.

</details>

3. does thinning a chain improve your estimates?

<details><summary>answer</summary>

no: for a fixed number of steps, keeping every draw always gives a lower-variance estimate than keeping every $k$-th (the extra draws are correlated, but not worthless). thinning saves memory and post-processing time, nothing else.

</details>

4. why does the random-walk optimum scale shrink like $1/\sqrt d$?

<details><summary>answer</summary>

a step of size $s$ in every coordinate has length $s\sqrt d$ and changes the log density by an amount whose typical magnitude grows like $s^2d$; keeping the acceptance rate constant needs $s^2d$ constant. so each coordinate moves by $O(1/\sqrt d)$ per step, and an independent draw takes $O(d)$ steps.

</details>

5. $\hat R = 1.00$ and every ESS is large. are you safe?

<details><summary>answer</summary>

safer, not safe: all the chains could have missed the same mode, or a region that's hard to reach (a funnel, as in the capstone). diagnostics can only detect disagreement, not absence. dispersed starting points, several chains, and sampler-specific warnings (like HMC's divergences) help.

</details>

6. Geyer's truncation stops at the first non-positive pair sum, not the first negative $\hat\rho_k$. why pairs?

<details><summary>answer</summary>

for a reversible chain, individual autocorrelations can legitimately be negative (antithetic chains oscillate), but sums of adjacent pairs are provably positive and decreasing. so a non-positive pair sum signals that noise has taken over, while a single negative $\hat\rho_k$ doesn't.

</details>

## common bugs

- **not recording the state after a rejection.** symptom: means that are wrong in a way more samples don't fix; samples arrays with fewer rows than asked for.
- **comparing $u$ to the log ratio, or the log of $u$ to the ratio.** symptom: acceptance rates near 0 or 1 regardless of the scale. compare $\log u$ with $\log\pi(x') - \log\pi(x)$.
- **forgetting to update the stored log density on acceptance.** symptom: the chain compares every proposal with a stale value and targets the wrong distribution.
- **Gibbs updating from a stale copy.** symptom: a "parallel" update that doesn't leave the target invariant; correlations that come out wrong.
- **FFT without zero padding.** symptom: autocorrelations that rise again at large lags (the end of the chain wrapping around onto the start).
- **summing every estimated autocorrelation.** symptom: an ESS that jumps around wildly between runs. truncate.
- **unsplit R-hat.** symptom: $\hat R \approx 1$ for chains that are all still drifting.

## going deeper

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953), "Equation of state calculations by fast computing machines", *Journal of Chemical Physics* 21(6). [doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
- Hastings (1970), "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57(1). [doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
- Geman & Geman (1984), "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images", *IEEE TPAMI* 6(6). [doi:10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596)
- Geyer (1992), "Practical Markov chain Monte Carlo", *Statistical Science* 7(4). the initial positive sequence estimator. [doi:10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137)
- Gelman & Rubin (1992), "Inference from iterative simulation using multiple sequences", *Statistical Science* 7(4). [doi:10.1214/ss/1177011136](https://doi.org/10.1214/ss/1177011136)
- Roberts, Gelman & Gilks (1997), "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Annals of Applied Probability* 7(1). the 0.234 rule. [doi:10.1214/aoap/1034625254](https://doi.org/10.1214/aoap/1034625254)
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), "Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC", *Bayesian Analysis* 16(2). [arXiv:1903.08008](https://arxiv.org/abs/1903.08008)
- Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed., CRC Press, ch. 11–12.

## connections

- ← builds on: Markov chains and mixing (ch 04 · topic 07), Monte Carlo estimates and their standard errors (ch 01, ch 04), Markov blankets (topic 02), the FFT (ch 01).
- → used in: Hamiltonian Monte Carlo, which replaces the random walk with physics (topic 06), checking variational approximations against long MCMC runs (topic 07), the capstone's PPL diagnostics (ESS and $\hat R$ per parameter), Langevin dynamics and score-based samplers (ch 22), simulated annealing and MCMC for combinatorial problems.
