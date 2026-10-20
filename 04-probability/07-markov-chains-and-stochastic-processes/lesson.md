# markov chains & stochastic processes

## why this matters

so far every random quantity in this chapter was a single draw, or a batch of independent ones. but most interesting randomness unfolds in time, with memory: tomorrow's weather depends on today's, the next word depends on the last few, a web surfer's next page depends on the current one, a molecule's next position depends on where it is now.

the simplest useful model of randomness with memory is a **Markov chain**: the future depends on the past *only through the present*. that one assumption buys an enormous amount. the n-gram text generator in this chapter's capstone is a Markov chain over characters. PageRank (ch 02's lab) is the long-run behavior of a Markov chain on web pages. Markov chain Monte Carlo (ch 12), the workhorse of Bayesian inference, *builds* a chain whose long-run behavior is the posterior you want to sample. reinforcement learning (ch 23) is built on Markov decision processes. and the forward process of a diffusion model (ch 22) is a Markov chain that gradually turns data into noise, whose continuous-time limit is Brownian motion, the last thing you'll build here.

two questions organize the topic. **where does the chain go in the long run?** (its stationary distribution.) and **how long does it take to get there?** (its mixing time, which, as the lab shows, is controlled by a single eigenvalue.)

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. if today is sunny, tomorrow is sunny with probability 0.9. if today is rainy, tomorrow is rainy with probability 0.5. in the long run, what fraction of days are sunny?
2. in that chain, does it matter whether day 0 was sunny or rainy, if you're asking about day 100?
3. a chain on two states always switches: $0 \to 1 \to 0 \to 1 \to \cdots$. does the distribution of its state converge as $t \to \infty$? does the fraction of time spent in state 0?
4. a random walk takes 10,000 steps of $\pm 1$. how far from the start is it, typically?
5. that random walk is positive (above its start) some fraction of the time. what's the *most likely* value of that fraction: near 1/2, or somewhere else?

## prerequisites check

- [ ] you can condition and use the law of total probability (→ ch 04 · topic 01)
- [ ] you can sample from a categorical distribution with cumulative sums (→ ch 04 · topic 02)
- [ ] you know left and right eigenvectors, and what an eigenvalue's modulus is (→ ch 02 · topic 05)
- [ ] you know the law of large numbers and the CLT (→ ch 04 · topic 05)

## core ideas

### idea 1: transition matrices and evolving distributions

a sequence of random states $X_0, X_1, X_2, \dots$ in a finite set $\{0, \dots, S - 1\}$ is a **Markov chain** if

$$P(X_{t+1} = j \mid X_t = i, X_{t-1}, \dots, X_0) = P(X_{t+1} = j \mid X_t = i) = P_{ij}.$$

the past matters only through the present state, and (for the time-homogeneous chains here) the rule doesn't change over time. all the chain's dynamics live in the **transition matrix** $P$, an $S \times S$ matrix whose row $i$ is the distribution of the next state given the current state $i$. every row is a probability distribution, so $P$ has non-negative entries and rows summing to 1: it's **row-stochastic**. the weather chain from warm-up 1 is

$$P = \begin{pmatrix}0.9 & 0.1\\ 0.5 & 0.5\end{pmatrix} \qquad (\text{state 0 = sunny, state 1 = rainy}).$$

suppose you don't know the state at time $t$ exactly, only its distribution, a row vector $p_t$ with $(p_t)_i = P(X_t = i)$. the law of total probability gives the distribution one step later:

$$(p_{t+1})_j = \sum_i P(X_t = i)\,P(X_{t+1} = j \mid X_t = i) = \sum_i (p_t)_i P_{ij}, \qquad\text{that is,}\qquad p_{t+1} = p_t P.$$

distributions are **row vectors multiplied on the left**. (writing $P\,p$ is the most common bug in this topic: it uses the columns of $P$, which aren't distributions at all.) iterating, $p_t = p_0P^t$, and row $i$ of $P^t$ is the distribution after $t$ steps from state $i$. that identity, $P(X_{s+t} = j \mid X_s = i) = (P^t)_{ij}$, is the **Chapman–Kolmogorov equation**.

starting sunny, $p_0 = (1, 0)$: $p_1 = (0.9, 0.1)$, $p_2 = (0.86, 0.14)$, $p_3 = (0.844, 0.156)$, … and starting rainy, $p_1 = (0.5, 0.5)$, $p_2 = (0.7, 0.3)$, …: both are heading for the same place, which is warm-up 2's answer in the making.

> **▶ build: stage 1.** implement `evolve(p0, P, n_steps)`, which returns the whole history $p_0, \dots, p_n$ as an `(n_steps + 1, S)` array, in `exercise.py` (`marimo edit exercise.py`).

### idea 2: simulating a path

a distribution describes what you know about the chain. a **path** is one particular realization: an actual sequence of states. to simulate one, start at $x_0$ and repeatedly draw the next state from row $P_{x_t}$ of the transition matrix. that's a categorical draw, and you already know how to do it with one uniform (topic 02): take the cumulative sums of the row, and find where $u$ falls with `np.searchsorted(..., side="right")`. as in topic 02, `side="right"` guarantees that a state with probability 0 (an interval of length zero) is never chosen, even if $u$ happens to be exactly 0.

simulation makes one theorem directly visible. for a well-behaved chain, the fraction of time a single long path spends in state $j$ converges to a fixed number:

$$\frac{1}{T}\sum_{t=1}^T\mathbb{1}[X_t = j] \xrightarrow{T \to \infty} \pi_j \quad\text{(with probability 1)}.$$

this is the **ergodic theorem** for Markov chains, the LLN of topic 05 for dependent draws. the limit $\pi$ doesn't depend on where the path started. it's what MCMC relies on: run one chain long enough and its states can be averaged as if they were samples from $\pi$, even though consecutive states are far from independent.

> **▶ build: stage 2.** implement `simulate_markov_chain(P, x0, n_steps, *, rng=None)`. validate $P$, draw one uniform per step, and return the `(n_steps + 1,)` integer path.

### idea 3: the stationary distribution

what is that limit $\pi$? if the chain's distribution ever settles, it settles at a distribution that one step doesn't change:

$$\pi P = \pi, \qquad \sum_i\pi_i = 1, \qquad \pi_i \ge 0.$$

such a $\pi$ is a **stationary distribution**. in linear-algebra terms, it's a *left* eigenvector of $P$ with eigenvalue 1 (a right eigenvector of $P^\top$). every stochastic matrix has eigenvalue 1, because $P\mathbf 1 = \mathbf 1$ (rows sum to 1), but that's the *right* eigenvector, the all-ones vector, which tells you nothing. the left one is the one you want.

for the weather chain, $\pi P = \pi$ reads $0.9\pi_0 + 0.5\pi_1 = \pi_0$, so $0.1\pi_0 = 0.5\pi_1$: in the long run, the probability flowing from sunny to rainy ($\pi_0 \cdot 0.1$) must equal the flow back ($\pi_1 \cdot 0.5$). with $\pi_0 + \pi_1 = 1$, that's $\pi = (5/6, 1/6)$, warm-up 1's answer: about 83% of days are sunny.

**when is $\pi$ unique, and when does $p_t$ converge to it?** two conditions matter:

- **irreducible**: every state can reach every other state. then $\pi$ exists, is unique, and is positive everywhere (the Perron–Frobenius theorem). if the chain splits into separate closed groups, each group has its own stationary distribution and any mixture of them is stationary too.
- **aperiodic**: the chain doesn't cycle with a fixed period. the always-switching chain of warm-up 3 has period 2. its stationary distribution is $(1/2, 1/2)$, but $p_t$ alternates between $(1, 0)$ and $(0, 1)$ forever and never converges. the ergodic theorem still holds, though: the *fraction of time* in state 0 converges to 1/2. that's warm-up 3: the distribution doesn't converge, the time average does.

an irreducible, aperiodic finite chain is called **ergodic**, and for it $p_0P^t \to \pi$ from every starting distribution.

how to compute $\pi$? iterating $p \mapsto pP$ is power iteration (ch 02 · topic 05), which fails for periodic chains and is slow for slowly mixing ones. better: solve the linear system directly. $(P^\top - I)\pi = 0$ is $S$ equations, but they're linearly dependent (they sum to zero, since rows of $P$ sum to 1), so add the normalization $\mathbf 1^\top\pi = 1$ as one more equation and solve the stacked $(S + 1) \times S$ system by least squares. for an irreducible chain it has exactly one solution.

one more idea you'll need in chapter 12: if $\pi_iP_{ij} = \pi_jP_{ji}$ for all $i, j$ (**detailed balance**: the flow from $i$ to $j$ equals the flow back, for every pair), then $\pi$ is stationary (on paper 3). MCMC algorithms like Metropolis–Hastings are designed to satisfy detailed balance with respect to the distribution you want to sample.

> **🔮 predict:** you start the weather chain on a rainy day. after how many days is the forecast within 1 percentage point of the long-run $5/6$?

> **▶ build: stage 3.** implement `stationary_distribution(P)` by stacking the normalization onto $(P^\top - I)$ and solving with `np.linalg.lstsq`.

### idea 4: mixing and the spectral gap

"converges to $\pi$" doesn't say how fast. to measure it, you need a distance between distributions. the **total variation distance**

$$\mathrm{TV}(p, q) = \tfrac12\sum_i|p_i - q_i| = \max_{A}\,|p(A) - q(A)|$$

is the largest disagreement about the probability of any event, between 0 (identical) and 1 (disjoint supports). define the worst-case distance after $t$ steps,

$$d(t) = \max_x\,\mathrm{TV}\big(P^t[x], \pi\big),$$

where $P^t[x]$ is row $x$ of $P^t$, the distribution after $t$ steps from state $x$. the **mixing time** $t_{\text{mix}}(\varepsilon)$ is the first $t$ with $d(t) \le \varepsilon$ (the convention $\varepsilon = 1/4$ is standard). to compute it, evolve every starting state at once: start from the identity matrix and multiply by $P$ until every row is within $\varepsilon$ of $\pi$.

what controls the speed? diagonalize. if $P$ has eigenvalues $1 = \lambda_1, \lambda_2, \dots$ with $|\lambda_2| \ge |\lambda_3| \ge \cdots$, then $P^t = \mathbf 1\pi + \sum_{k \ge 2}\lambda_k^t(\text{fixed matrices})$, so the distance to stationarity decays like $|\lambda_2|^t$. the **spectral gap** $\gamma = 1 - |\lambda_2|$ sets the rate: roughly,

$$t_{\text{mix}} \approx \frac{1}{\gamma}\log\frac{C}{\varepsilon},$$

where $C$ depends on the chain (for reversible chains, $\log(1/\pi_{\min})$ appears). a gap near 0 means slow mixing. a gap of exactly 0 means $|\lambda_2| = 1$: either a second eigenvalue equal to 1 (the chain is reducible) or one on the unit circle like $-1$ (it's periodic). the eigenvalues of a non-symmetric $P$ can be complex, so compare moduli.

for the weather chain, $\lambda_2 = 0.4$ (the trace is $1.4 = 1 + \lambda_2$), so $\gamma = 0.6$, and from rainy the distance is exactly $\tfrac56\,0.4^t$: below $1/4$ after 2 steps, below $0.01$ after 5. a chain with a **bottleneck** (two well-connected clusters joined by a narrow bridge) has a tiny gap: the chain lingers on one side for a long time before crossing. the lab measures this.

> **▶ build: stage 4.** implement `total_variation(p, q)`, `spectral_gap(P)` and `mixing_time(P, eps=0.25, max_steps=10_000)` (which returns `None` if the chain hasn't mixed by `max_steps`).

### idea 5: random walks and Brownian motion

the most famous Markov chain has infinitely many states: the **simple random walk** $S_n = \xi_1 + \cdots + \xi_n$ with independent steps $\xi_i = \pm 1$. it has mean 0 and variance $n$, so it typically sits about $\sqrt n$ from its start: warm-up 4's 10,000 steps put it about 100 away (more precisely, $\mathbb{E}|S_n| \approx \sqrt{2n/\pi} \approx 80$).

zoom out: speed up time by a factor $n$ and shrink space by $\sqrt n$, the scaling that keeps the variance fixed. the rescaled walk $S_{\lfloor nt\rfloor}/\sqrt n$ converges (in distribution, as a whole random function of $t$) to **Brownian motion** $B_t$, by Donsker's theorem, a CLT for entire paths. Brownian motion is characterized by:

1. $B_0 = 0$;
2. independent increments: $B_{t} - B_s$ is independent of everything before time $s$;
3. Gaussian increments: $B_t - B_s \sim \mathcal{N}(0, t - s)$;
4. continuous paths.

it's the continuous-time Markov process at the heart of physics (diffusion of particles), finance (stock prices), and generative modeling: a diffusion model's forward process is a discretized stochastic differential equation driven by Brownian motion (ch 22).

on a computer you simulate it on a time grid $0, \Delta t, 2\Delta t, \dots$: draw independent increments $\mathcal{N}(0, \Delta t)$, standard deviation $\sqrt{\Delta t}$, and take running sums, with a 0 in front. since the variance of $B_t$ is $t$, a path over $[0, 1]$ wanders about $\pm 1$, and over a short interval $\Delta t$ it moves about $\sqrt{\Delta t} \gg \Delta t$. that's why Brownian paths are so jagged: the ratio $|\Delta B|/\Delta t \approx 1/\sqrt{\Delta t}$ blows up as $\Delta t \to 0$, and the path is nowhere differentiable.

> **🔮 predict:** run a Brownian motion for one unit of time. what's the most likely value of the fraction of that time it spends above 0? (warm-up 5. the lab answers it.)

> **▶ build: stage 5.** implement `brownian_motion(n_paths, n_steps, *, dt=0.01, rng=None)` with increments from *your* `mlfs.sampling.normal`.

## worked example

**evolving.** for the weather chain from sunny: $p_1 = (1, 0)P = (0.9, 0.1)$. $p_2 = (0.9 \cdot 0.9 + 0.1 \cdot 0.5,\ 0.9 \cdot 0.1 + 0.1 \cdot 0.5) = (0.86, 0.14)$.

**the stationary distribution.** flow balance $0.1\pi_0 = 0.5\pi_1$ with $\pi_0 + \pi_1 = 1$ gives $\pi = (5/6, 1/6) \approx (0.833, 0.167)$. check: $(5/6)(0.9) + (1/6)(0.5) = 0.75 + 0.0833 = 5/6$. ✓

**the spectral gap.** the eigenvalues of a $2 \times 2$ stochastic matrix are 1 and $\operatorname{tr}P - 1 = 0.4$, so $\gamma = 0.6$.

**the mixing time.** for two states, $p_t - \pi = (p_0 - \pi)\lambda_2^t$. from rainy, $p_0 - \pi = (-5/6, 5/6)$, so the TV distance is $\tfrac56\,0.4^t$: $0.333$ at $t = 1$, $0.133$ at $t = 2$. from sunny it's only $\tfrac16\,0.4^t$. the worst start is rainy, so $t_{\text{mix}}(1/4) = 2$, and $t_{\text{mix}}(0.01) = 5$ (since $\tfrac56 0.4^4 = 0.021$ and $\tfrac56 0.4^5 = 0.0085$). that also answers idea 3's prediction: 5 days.

**total variation.** between $(0.5, 0.5, 0)$ and $(0.2, 0.3, 0.5)$: $\tfrac12(0.3 + 0.2 + 0.5) = 0.5$.

**Brownian increments.** with $\Delta t = 0.01$, each increment has standard deviation $0.1$, and after 100 steps ($t = 1$) the variance is $100 \times 0.01 = 1$.

the tests check every one of these numbers.

## on paper

1. **Chapman–Kolmogorov.** show by induction that $P(X_t = j \mid X_0 = i) = (P^t)_{ij}$, using the Markov property and the law of total probability.

<details><summary>solution</summary>

$t = 1$ is the definition. assume it for $t$. then $P(X_{t+1} = j \mid X_0 = i) = \sum_k P(X_t = k \mid X_0 = i)\,P(X_{t+1} = j \mid X_t = k, X_0 = i) = \sum_k (P^t)_{ik}P_{kj} = (P^{t+1})_{ij}$, where the Markov property removes the conditioning on $X_0$ in the second factor.

</details>

2. **the general two-state chain.** let $P = \begin{pmatrix}1 - a & a\\ b & 1 - b\end{pmatrix}$ with $0 < a, b \le 1$. find $\pi$, the eigenvalues, and $d(t)$. when is the chain periodic, and when does it mix in one step?

<details><summary>solution</summary>

flow balance $a\pi_0 = b\pi_1$ gives $\pi = \big(\tfrac{b}{a+b}, \tfrac{a}{a+b}\big)$. the eigenvalues are 1 and $\operatorname{tr}P - 1 = 1 - a - b$. from state $x$, $p_t - \pi = (e_x - \pi)(1 - a - b)^t$, so $d(t) = \max\big(\tfrac{a}{a+b}, \tfrac{b}{a+b}\big)\,|1 - a - b|^t$. it's periodic when $a = b = 1$ ($\lambda_2 = -1$, gap 0). it mixes in one step when $a + b = 1$ ($\lambda_2 = 0$): then both rows of $P$ equal $\pi$, so the next state doesn't depend on the current one at all.

</details>

3. **detailed balance implies stationarity.** show that if $\pi_iP_{ij} = \pi_jP_{ji}$ for all $i, j$, then $\pi P = \pi$. give an example of a stationary $\pi$ that does *not* satisfy detailed balance.

<details><summary>solution</summary>

$(\pi P)_j = \sum_i\pi_iP_{ij} = \sum_i\pi_jP_{ji} = \pi_j\sum_iP_{ji} = \pi_j$. counterexample: the deterministic 3-cycle $0 \to 1 \to 2 \to 0$ has uniform stationary $\pi$, but $\pi_0P_{01} = 1/3 \ne 0 = \pi_1P_{10}$: probability circulates around the cycle instead of balancing pair by pair. chains with detailed balance are called *reversible*: run backwards in time, they look the same.

</details>

4. **the Brownian scaling.** for the simple random walk, show $\mathbb{E}S_n = 0$ and $\operatorname{Var}S_n = n$. why does $S_{\lfloor nt\rfloor}/\sqrt n$ have variance close to $t$, and why would dividing by $n$ instead of $\sqrt n$ give a boring limit?

<details><summary>solution</summary>

the steps are independent with mean 0 and variance 1, so the sum has mean 0 and variance $n$ (topic 03). then $\operatorname{Var}(S_{\lfloor nt\rfloor}/\sqrt n) = \lfloor nt\rfloor/n \to t$, matching $\operatorname{Var}B_t = t$; by the CLT, each such value is approximately $\mathcal{N}(0, t)$. dividing by $n$ gives variance $\lfloor nt\rfloor/n^2 \to 0$: the path collapses to the constant 0, which is just the law of large numbers again.

</details>

## experiment

`lab.py` runs three experiments on your chain tools:

1. mixing time against the spectral gap, for chains on cycles, complete graphs, and two cliques joined by a single edge.
2. the ergodic theorem: time averages along one path, for an aperiodic chain and a periodic one.
3. Brownian motion: how much of its time a path spends above zero.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. why must distributions be multiplied as $pP$ and not $Pp$?

<details><summary>answer</summary>

$(pP)_j = \sum_ip_iP_{ij}$ sums over *where you came from*, weighting each row (a conditional distribution) by the probability of being there. $Pp$ would weight entries of each row by $p$, mixing up "from" and "to"; for a non-symmetric $P$ the result isn't even a distribution in general.

</details>

2. every stochastic matrix satisfies $P\mathbf 1 = \mathbf 1$. why isn't the all-ones vector (normalized) the stationary distribution?

<details><summary>answer</summary>

it's the right eigenvector, which expresses "rows sum to 1". the stationary distribution is the *left* eigenvector, $\pi P = \pi$. they coincide only for doubly stochastic matrices (columns also sum to 1), whose stationary distribution is uniform.

</details>

3. a chain has two closed groups of states that can't reach each other. what does `stationary_distribution` return, and what's wrong with it?

<details><summary>answer</summary>

the system has a whole family of solutions (any mixture of the two groups' stationary distributions), so least squares returns one of them, the minimum-norm one. it's a valid stationary distribution but not *the* stationary distribution: the long-run behavior depends on which group you start in. irreducibility is what makes $\pi$ unique.

</details>

4. why is the spectral gap computed from $|\lambda_2|$ and not $\lambda_2$?

<details><summary>answer</summary>

the decay rate of $\lambda_2^t$ depends on its modulus. a chain with $\lambda_2 = -0.99$ (nearly periodic) mixes as slowly as one with $\lambda_2 = 0.99$, and complex eigenvalues come in conjugate pairs that rotate as they decay. this is also why "lazy" chains ($\tfrac12(I + P)$) are popular in theory: all their eigenvalues are in $[0, 1]$ for reversible chains, which removes the periodic case.

</details>

5. an MCMC sampler has a spectral gap of $10^{-4}$. what does that mean in practice?

<details><summary>answer</summary>

it takes on the order of $10^4$ steps (times a log factor) to forget its starting point, and consecutive samples are strongly correlated over similar time scales: 100,000 steps carry only about ten independent samples' worth of information. you'd want a better proposal or a reparameterization (ch 12).

</details>

6. Brownian increments have standard deviation $\sqrt{\Delta t}$, not $\Delta t$. what would go wrong with $\Delta t$?

<details><summary>answer</summary>

after $n = T/\Delta t$ steps the variance would be $n\Delta t^2 = T\Delta t \to 0$ as the grid refines: the "Brownian motion" would vanish in the continuous limit. the $\sqrt{\Delta t}$ is what makes the variance add up to $T$ regardless of the grid.

</details>

7. an n-gram language model (the capstone) predicts the next character from the previous $n - 1$. in what sense is it a Markov chain, and on what state space?

<details><summary>answer</summary>

the state is the last $n - 1$ characters. the next state is the last $n - 2$ of them plus the newly drawn character, with probabilities from the model's conditional distribution. so it's a Markov chain on $V^{n-1}$ (all strings of length $n - 1$ over the vocabulary $V$), with a very sparse transition matrix: each state can move to at most $|V|$ others. generating text is `simulate_markov_chain` on that state space.

</details>

## common bugs

- **$Pp$ instead of $pP$.** symptom: "distributions" that don't sum to 1, or a wrong stationary distribution for any non-symmetric $P$.
- **the right eigenvector.** `np.linalg.eig(P)` returns *right* eigenvectors; the stationary distribution is an eigenvector of $P^\top$. symptom: a uniform "stationary distribution" for every chain.
- **power iteration on a periodic chain.** `p = p @ P` repeated 1000 times oscillates forever for a period-2 chain. symptom: an answer that depends on whether you iterated an even or odd number of times.
- **`side="left"` in `searchsorted`.** symptom: very rarely, a transition with probability 0.
- **comparing eigenvalues without `np.abs`.** symptom: a spectral gap of 2 for a periodic chain (eigenvalue $-1$), or a crash sorting complex numbers.
- **forgetting the 0 at the start of a Brownian path**, or cumulative-summing along the wrong axis. symptom: paths that don't start at 0, or variance growing across paths instead of along time.

## going deeper

- Levin & Peres (2017), *Markov Chains and Mixing Times*, 2nd ed., American Mathematical Society, chapters 1, 4 and 12: stationary distributions, total variation, mixing times and the spectral gap. free from the authors.
- Norris (1997), *Markov Chains*, Cambridge University Press, chapter 1: a clean, rigorous introduction to discrete-time chains, including the ergodic theorem.
- Mörters & Peres (2010), *Brownian Motion*, Cambridge University Press, chapters 1 and 5: construction, properties, Donsker's invariance principle and the arcsine laws. free from the authors.
- Diaconis (2009), "The Markov chain Monte Carlo revolution", *Bulletin of the American Mathematical Society* 46(2): 179–205. a tour of why mixing matters, starting from decoding a prison message. [doi:10.1090/S0273-0979-08-01238-X](https://doi.org/10.1090/S0273-0979-08-01238-X)
- Page, Brin, Motwani & Winograd (1999), "The PageRank citation ranking: bringing order to the web", Stanford InfoLab technical report. the stationary distribution of a random surfer.

## connections

- ← builds on: conditioning and total probability (topic 01), categorical sampling and `normal` (topic 02), the LLN and CLT (topic 05), eigenvalues and power iteration (ch 02 · topic 05), PageRank (ch 02's lab).
- → used in: the n-gram text generator (this chapter's capstone), hidden Markov models and MCMC (ch 12), recurrent models as learned state machines (ch 17), diffusion models and SDEs (ch 22), and Markov decision processes in reinforcement learning (ch 23).
