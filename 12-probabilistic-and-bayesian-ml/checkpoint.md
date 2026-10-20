# 12 · probabilistic & bayesian ML · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. in Bayesian linear regression the posterior mean of the weights equals the ridge solution. for which penalty, and what does the Bayesian treatment give you that ridge doesn't?

<details><summary>answer</summary>

$\lambda = \alpha/\beta$ (prior precision over noise precision). the Bayesian version keeps the whole Gaussian posterior: a covariance $S_N$, hence predictive variances $1/\beta + \phi^\top S_N\phi$ that separate noise from weight uncertainty, posterior samples, and the evidence for choosing $\alpha$, $\beta$ and the basis. (topic 01, ideas 2–3)

</details>

2. why does the evidence $p(y \mid \alpha, \beta)$ prefer a cubic over a degree-9 polynomial for data from a cubic, although degree 9 fits the training data better?

<details><summary>answer</summary>

the evidence is a normalized distribution over datasets; a flexible model spreads its probability over many datasets it could have explained, so it assigns less to the one observed. the Occam factor $\frac d2\log\alpha - \frac12\log|A|$ charges for every direction the data have to pin down. (topic 01, idea 4 and lab)

</details>

3. with Gaussian-bump features centered in $[-1, 1]$, the predictive variance at $x = 3$ is barely above the noise. why, and what does that tell you about "uncertainty"?

<details><summary>answer</summary>

$\phi(3) \approx (1, 0, \dots, 0)$, so $\phi^\top S_N\phi$ is just the (small) posterior variance of the constant weight. predictive uncertainty is uncertainty *within the model's hypothesis space*, which here contains only functions that are flat far from the centers. (topic 01, idea 3 and lab)

</details>

4. $A \to B \leftarrow C$. are $A$ and $C$ independent? given $B$? given a child $D$ of $B$?

<details><summary>answer</summary>

yes; no; no. a collider blocks the path until it or one of its descendants is observed; observing it creates the explaining-away dependence between its parents. (topic 02, idea 4)

</details>

5. how does the "ancestral moral graph" test decide d-separation, and why does it handle colliders correctly?

<details><summary>answer</summary>

keep $X \cup Y \cup Z$ and their ancestors, marry co-parents and drop directions, delete $Z$, and check connectivity. an unobserved collider without observed descendants isn't an ancestor of anything kept, so it's dropped and its path is blocked; a collider that is an ancestor of $Z$ stays, and moralization connects its parents, which opens the path. (topic 02, idea 4)

</details>

6. variable elimination is exact for every order. what does the order change, and what graph quantity bounds the best you can do?

<details><summary>answer</summary>

the size of the intermediate tables: eliminating a variable joins all its current neighbours into one factor (fill-in). the best order's largest table is exponential in the treewidth; on an $n \times n$ grid that's about $2^{n+1}$ for binary variables. (topic 03, idea 2 and lab)

</details>

7. why does sum-product compute every marginal of a tree for about the cost of two eliminations?

<details><summary>answer</summary>

the message from a subtree to its neighbour is the same for every query outside that subtree; a tree has $2E$ directed messages, each computed once and reused, and each marginal is the product of the messages arriving at that variable. (topic 03, idea 4)

</details>

8. why does the unscaled forward algorithm fail on long sequences, and what do the scaling constants $c_t$ mean?

<details><summary>answer</summary>

$\alpha_t = p(x_{1:t}, z_t)$ shrinks geometrically and underflows to 0. normalizing at each step gives $\hat\alpha_t = P(z_t \mid x_{1:t})$, and the normalizers are the one-step predictive probabilities $c_t = p(x_t \mid x_{1:t-1})$, so $\log p(x_{1:T}) = \sum_t\log c_t$. (topic 04, idea 2)

</details>

9. Viterbi and posterior decoding: which maximizes the expected number of correctly labelled time steps, and can the other one produce an impossible path?

<details><summary>answer</summary>

posterior decoding (argmax of each $\gamma_t$) maximizes expected per-step accuracy, but its path can include zero-probability transitions. Viterbi maximizes the probability that the *whole* path is right and always returns a feasible path. (topic 04, idea 3 and lab)

</details>

10. why does random-walk Metropolis keep the current state (and record it again) when it rejects a proposal?

<details><summary>answer</summary>

the rejection probability is part of the transition kernel; without it detailed balance fails and the chain targets the wrong distribution (high-density states with low acceptance would be under-represented). (topic 05, idea 1)

</details>

11. what's the effective sample size, and why does Geyer's estimator truncate the sum of autocorrelations?

<details><summary>answer</summary>

$\text{ESS} = n/\tau$ with $\tau = 1 + 2\sum_k\rho_k$: the number of independent draws with the same Monte Carlo variance for a mean. far lags are pure noise, so summing all of them gives an estimator whose variance doesn't shrink; pair sums $\rho_{2m} + \rho_{2m+1}$ are positive for reversible chains, so summing until the first non-positive pair sum cuts off the noise. (topic 05, idea 4)

</details>

12. four chains, each with ESS around 300 and a stationary-looking trace, give means $-4, -4, +4, +4$. what does split $\hat R$ say, and what's going on?

<details><summary>answer</summary>

$\hat R \gg 1$ (about 4.4 in the lab): the chains disagree. each chain is stuck in one mode of a multimodal target; single-chain diagnostics can't see a mode a chain never visited. (topic 05, idea 5 and lab)

</details>

13. why is HMC's leapfrog proposal accepted with probability $\min(1, e^{-\Delta H})$, with no proposal-density ratio, and why must the step size be below $2\sigma_{\min}$?

<details><summary>answer</summary>

leapfrog followed by a momentum flip is a deterministic, volume-preserving involution, so the proposal densities cancel and only $\pi(q^*, p^*)/\pi(q, p) = e^{-\Delta H}$ remains. on a direction with sd $\sigma$, one leapfrog step is a linear map whose eigenvalues leave the unit circle when $\varepsilon > 2\sigma$: the trajectory diverges exponentially and everything is rejected. (topic 06, ideas 1–2)

</details>

14. why does mean-field VI underestimate posterior variances on a correlated Gaussian, and what are the variances it reports?

<details><summary>answer</summary>

it minimizes $\mathrm{KL}(q\|p)$, an expectation under $q$, which heavily penalizes $q$ putting mass where $p$ has little; an axis-aligned $q$ must shrink to fit inside the tilted ellipse. the optimum has the right mean and variances $1/\Lambda_{ii}$, the conditional variances ($1 - \rho^2$ for unit variances and correlation $\rho$). (topic 07, idea 5 and lab)

</details>

15. in the eight-schools model, what is the funnel, and how does the non-centered parameterization remove it?

<details><summary>answer</summary>

in the centered parameterization $\theta_j \sim \mathcal N(\mu, \tau)$, the conditional spread of the $\theta_j$ shrinks with $\tau$: the posterior in $(\theta_j, \log\tau)$ is wide at large $\tau$ and a narrow neck at small $\tau$, so no single step size works and HMC diverges at (and under-samples) the neck. writing $\theta_j = \mu + \tau\eta_j$ with $\eta_j \sim \mathcal N(0, 1)$ makes the prior geometry of $(\eta, \log\tau)$ independent of $\tau$; with weakly informative data the posterior is nearly as easy as the prior. (capstone)

</details>

## 2. by hand

1. **a conjugate posterior and predictive.** one feature, no intercept: inputs $x = (1, 2)$, targets $y = (2, 3)$, prior precision $\alpha = 1$, noise precision $\beta = 2$. find $m_N$, $S_N$, and the predictive mean and variance at $x_* = 3$.

<details><summary>solution</summary>

$A = \alpha + \beta(1 + 4) = 11$, so $S_N = 1/11$ and $m_N = \beta S_N(1\cdot2 + 2\cdot3) = 16/11 \approx 1.455$. predictive mean $3\cdot16/11 = 48/11 \approx 4.364$; variance $1/\beta + 9/11 = 0.5 + 0.818 = 1.318$. (topic 01, ideas 2–3)

</details>

2. **forward–backward and Viterbi.** a 2-state HMM with $\pi = (0.6, 0.4)$, $A = \begin{pmatrix}0.7 & 0.3\\0.4 & 0.6\end{pmatrix}$, $B = \begin{pmatrix}0.9 & 0.1\\0.2 & 0.8\end{pmatrix}$ (rows: states; columns: symbols 0, 1), and observations $x = (1, 1)$. compute (a) $p(x)$, (b) $P(z_1 \mid x)$, (c) the Viterbi path and its probability.

<details><summary>solution</summary>

(a) $\alpha_1 = (0.6\cdot0.1, 0.4\cdot0.8) = (0.06, 0.32)$; $\alpha_1A = (0.17, 0.21)$; $\alpha_2 = (0.017, 0.168)$; $p(x) = 0.185$. (b) $\beta_1 = A\,(0.1, 0.8)^\top = (0.31, 0.52)$; $\alpha_1\beta_1 = (0.0186, 0.1664)$, divided by 0.185: $(0.101, 0.899)$. (c) $\delta_2(0) = \max(0.06\cdot0.7, 0.32\cdot0.4)\cdot0.1 = 0.0128$ and $\delta_2(1) = \max(0.06\cdot0.3, 0.32\cdot0.6)\cdot0.8 = 0.1536$, both from state 1; the path is $(1, 1)$ with $p(x, z) = 0.1536$. (topic 04, worked example)

</details>

3. **one leapfrog step.** target $\mathcal N(0, 4)$, so $\nabla\log\pi(q) = -q/4$. from $(q, p) = (2, 1)$, take one leapfrog step with $\varepsilon = 1$ (identity mass). find $(q', p')$, the energy error, and the acceptance probability.

<details><summary>solution</summary>

half kick: $p = 1 - 0.5\cdot2/4 = 0.75$; drift: $q' = 2 + 0.75 = 2.75$; half kick: $p' = 0.75 - 0.5\cdot2.75/4 = 0.40625$. $H = q^2/8 + p^2/2$: $H_0 = 0.5 + 0.5 = 1$, $H_1 = 0.9453 + 0.0825 = 1.0278$. $\Delta H = 0.0278$, so the acceptance probability is $e^{-0.0278} = 0.973$. (topic 06, idea 1)

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the MCMC core**: topic 05's `metropolis_hastings`, `autocorrelation` and `effective_sample_size`, and topic 06's `hamiltonian`, `leapfrog` and `hmc`. work in scratch directories, starting from the *untouched* stub notebooks and the topics' checks, so your own solved notebooks stay where they are. from the repo root:

```bash
C=12-probabilistic-and-bayesian-ml
mkdir -p /tmp/rebuild-1205 /tmp/rebuild-1206
git show HEAD:$C/05-mcmc/exercise.py > /tmp/rebuild-1205/exercise.py
git show HEAD:$C/06-hamiltonian-monte-carlo/exercise.py > /tmp/rebuild-1206/exercise.py
cp $C/05-mcmc/test_exercise.py /tmp/rebuild-1205/
cp $C/06-hamiltonian-monte-carlo/test_exercise.py /tmp/rebuild-1206/
```

(if you've committed your own solutions, regenerate the stubs instead: copy each topic's `solutions/` into the scratch directory and run `python scripts/stubify.py /tmp/rebuild-1205 /tmp/rebuild-1206`.) open each `exercise.py` with `marimo edit`, implement the stages from memory, and run the checks against your copies:

```bash
pytest /tmp/rebuild-1205/test_exercise.py -k "stage1 or stage3 or stage4"
pytest /tmp/rebuild-1206/test_exercise.py -k "stage1 or stage2"
```

you pass if every check is green within the hour. (the samplers are checked against exact posteriors with tolerances from the effective sample size, the bookkeeping exactly, the ESS against its precise definition, and the integrator for reversibility, volume preservation and second-order energy error, so "almost right" fails.)

**harder variant** (for testing out of the chapter with confidence): in a further 60 minutes, rebuild topic 04's `HMM` (`sample`, `scaled_messages`, `forward_backward`, `viterbi`, `fit`, `score`) in its own scratch directory and pass:

```bash
pytest /tmp/rebuild-1204/test_exercise.py
```

(run them from the repo root: the notebooks import earlier modules through `mlfs`, which the tests resolve to the reference solutions, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic and idea it comes from.
