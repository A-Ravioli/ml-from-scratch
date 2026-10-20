# hidden markov models

## why this matters

a casino croupier occasionally swaps a fair die for a loaded one, then swaps back. you see 300 rolls. when was the loaded die in play? a single six tells you almost nothing (sixes happen on fair dice too), but a *run* of sixes does, and so does the fact that the croupier doesn't switch on every roll. the evidence is spread across time, and the thing you care about is hidden.

that pattern is everywhere: the words behind a speech signal, the genes in a DNA sequence, the part of speech of each word, the regime (calm or volatile) of a market, the activity behind an accelerometer trace. the **hidden Markov model** is the simplest model of it: a hidden state that evolves as a Markov chain, and an observation at each step that depends only on the current state. it's a graphical model shaped like a chain, so everything from topic 03 applies, and every question has an exact, linear-time answer. this topic builds the three classic algorithms: **forward–backward** (how likely are the observations, and what was the state at each step?), **Viterbi** (what's the single best explanation?), and **Baum–Welch** (what are the parameters?), the first and still one of the most useful instances of the EM algorithm.

## warm-up: predict first

1. an HMM with 2 states and a sequence of 1,000 observations. how many hidden paths are there? how long does it take to sum over all of them with the forward algorithm?
2. you multiply 1,000 probabilities, each around 0.2, in float64. what do you get?
3. the most probable *sequence* of states and the sequence of most probable *individual* states: are they always the same?
4. Baum–Welch from a random start reaches parameters with a higher likelihood than the true ones. is that a bug?
5. the croupier switches dice on almost every roll. how much does knowing the rolls on either side help you decode a given roll?

## prerequisites check

- [ ] you can simulate a Markov chain from a transition matrix and know its stationary distribution (→ ch 04 · topic 07)
- [ ] you know variable elimination on a chain and the idea of a message (→ topic 03)
- [ ] you can work in log space with `logsumexp` and know why (→ ch 00, ch 01)
- [ ] you know maximum likelihood for a categorical distribution: normalized counts (→ ch 05)

## core ideas

### idea 1: the model

an HMM has $K$ hidden states and $M$ observation symbols, and three tables:

- the **start distribution** $\pi_k = P(z_1 = k)$, `startprob_`, shape $(K,)$;
- the **transition matrix** $A_{ij} = P(z_{t+1} = j \mid z_t = i)$, `transmat_`, shape $(K, K)$, rows summing to 1;
- the **emission matrix** $B_{km} = P(x_t = m \mid z_t = k)$, `emissionprob_`, shape $(K, M)$, rows summing to 1.

the joint probability of a hidden path $z_{1:T}$ and observations $x_{1:T}$ is a product along the chain:

$$p(x_{1:T}, z_{1:T}) = \pi_{z_1}B_{z_1x_1}\prod_{t=2}^{T}A_{z_{t-1}z_t}B_{z_tx_t}.$$

as a graphical model it's a chain $z_1 \to z_2 \to \dots \to z_T$ with a leaf $x_t$ hanging off each $z_t$. two independences carry all the algorithms: given $z_t$, the future is independent of the past, and $x_t$ depends on nothing but $z_t$. sampling is the generative story read left to right: draw $z_1$ from $\pi$; then at each step emit $x_t$ from row $z_t$ of $B$, and move to $z_{t+1}$ drawn from row $z_t$ of $A$.

the dishonest casino in these terms: $K = 2$ (fair, loaded), $M = 6$ (faces), $A = \begin{pmatrix}0.95 & 0.05\\0.1 & 0.9\end{pmatrix}$, the fair row of $B$ uniform, the loaded row $(0.1, 0.1, 0.1, 0.1, 0.1, 0.5)$.

> **spec.** `HMM(n_states, n_symbols, *, max_iters=100, tol=1e-6, rng=None)` is an `mlfs.base.Estimator` whose `__init__` only stores its arguments. the parameters are the attributes `startprob_`, `transmat_`, `emissionprob_`, set by `fit` or directly. `sample(n_steps, *, rng=None)` returns `(obs, states)`, two int arrays of shape `(n_steps,)`, drawing with `as_rng(rng)` (e.g. `rng.choice(K, p=row)` per draw).

> **▶ build: stage 1.** implement `__init__` and `sample`.

### idea 2: forward–backward

how likely are the observations? summing the joint over all $K^T$ paths is hopeless ($2^{1000}$ for warm-up 1), but it's a chain, and topic 03 says: eliminate along it. define the **forward** message

$$\alpha_t(k) = p(x_{1:t}, z_t = k), \qquad \alpha_1(k) = \pi_kB_{kx_1}, \qquad \alpha_{t+1}(j) = \Big(\sum_k\alpha_t(k)A_{kj}\Big)B_{jx_{t+1}},$$

and $p(x_{1:T}) = \sum_k\alpha_T(k)$: $O(TK^2)$ work. the **backward** message runs the other way, $\beta_t(k) = p(x_{t+1:T} \mid z_t = k)$ with $\beta_T = 1$ and $\beta_t(i) = \sum_jA_{ij}B_{jx_{t+1}}\beta_{t+1}(j)$. multiply them and you get the **posterior marginal** of every hidden state:

$$\gamma_t(k) = P(z_t = k \mid x_{1:T}) = \frac{\alpha_t(k)\,\beta_t(k)}{p(x_{1:T})}.$$

this is exactly sum-product on the chain: $\alpha$ is the message arriving at $z_t$ from the left, $\beta$ from the right.

there's a numerical trap (warm-up 2). $\alpha_t$ is a probability of $t$ observations, so it shrinks geometrically: after a few hundred steps it underflows to exactly 0 in float64, and every posterior becomes 0/0. the fix is **scaling**: normalize $\alpha$ at every step and remember the normalizer,

$$\hat\alpha_t = \frac{\alpha_t}{p(x_{1:t})} = P(z_t \mid x_{1:t}), \qquad c_t = \sum_k\big(\hat\alpha_{t-1}A\big)_kB_{kx_t} = p(x_t \mid x_{1:t-1}),$$

so $\log p(x_{1:T}) = \sum_t\log c_t$. scale the backward messages by the same constants, $\hat\beta_t = \beta_t / \prod_{s > t}c_s$, and the posterior is simply $\gamma_t \propto \hat\alpha_t\hat\beta_t$. each $\hat\alpha_t$ is a distribution, so nothing ever underflows. (working with $\log\alpha$ and `logsumexp` works too, but is slower.)

> **spec.** `scaled_messages(obs)` returns `(alpha, beta, c)`: $\hat\alpha$ and $\hat\beta$ of shape $(T, K)$ and $c$ of shape $(T,)$, as above (with $\hat\beta_T = 1$ and $\hat\beta_t = A(B_{\cdot x_{t+1}} \odot \hat\beta_{t+1})/c_{t+1}$). `forward_backward(obs)` returns `(log_likelihood, posteriors)`: a float, and $(T, K)$ with rows $\gamma_t$. both must work on sequences of thousands of steps.

> **🔮 predict:** in the casino, a single six raises the probability that the loaded die is in play. by how much does it raise the posterior at the *neighbouring* rolls, which weren't sixes? not at all, a little, or as much?

> **▶ build: stage 2.** implement `scaled_messages` and `forward_backward`.

### idea 3: Viterbi

the posterior marginals tell you the most probable state at each step, one step at a time. often you want the most probable *path*: one consistent story. replace the sum in the forward recursion with a max (max-product, topic 03), and keep track of which predecessor won:

$$\delta_1(k) = \log\pi_k + \log B_{kx_1}, \qquad \delta_{t+1}(j) = \max_k\big[\delta_t(k) + \log A_{kj}\big] + \log B_{jx_{t+1}},$$

with the back-pointer $\psi_{t+1}(j) = \arg\max_k[\delta_t(k) + \log A_{kj}]$. $\delta_t(k)$ is the log-probability of the best path that ends in state $k$ at time $t$. at the end, take $\arg\max_k\delta_T(k)$ and follow the back-pointers to the start. the max doesn't need normalizing, so log space is the natural fix for underflow here; zero probabilities become $-\infty$, which `max` handles correctly.

the best path and the sequence of best states can differ (warm-up 3): the per-step argmaxes can stitch together a path that is improbable as a whole, or even impossible if some transitions have probability 0. the worked example shows a two-step case. which one you want depends on the loss: Viterbi minimizes the chance of getting *anything* wrong, posterior decoding the expected *number* of wrong steps.

> **spec.** `viterbi(obs)` returns `(states, log_prob)`: an int array of shape $(T,)$ and $\log p(x_{1:T}, z^*_{1:T})$ for the returned path. ties go to the lowest state (`np.argmax`). zero entries in the parameters must not break it.

> **🔮 predict:** on 300 rolls of the casino, which labels more rolls correctly: Viterbi, or "call a roll loaded exactly when it's a six"? by how much?

> **▶ build: stage 3.** implement `viterbi`.

### idea 4: Baum–Welch

where do $\pi$, $A$, $B$ come from? if you knew the hidden states, maximum likelihood would be counting: $A_{ij}$ = (number of $i \to j$ transitions) / (number of transitions out of $i$), and the same for $B$ and $\pi$. you don't know them, but forward–backward gives you a *distribution* over them, so count in expectation. that's the **EM algorithm**:

- **E-step**: with the current parameters, compute $\gamma_t(k)$ and the pairwise posteriors
$$\xi_t(i, j) = P(z_t = i, z_{t+1} = j \mid x) = \frac{\hat\alpha_t(i)\,A_{ij}\,B_{jx_{t+1}}\,\hat\beta_{t+1}(j)}{c_{t+1}}.$$
- **M-step**: the expected counts, normalized:
$$\pi_k \leftarrow \gamma_1(k), \qquad A_{ij} \leftarrow \frac{\sum_t\xi_t(i, j)}{\sum_t\sum_j\xi_t(i, j)}, \qquad B_{km} \leftarrow \frac{\sum_{t:\,x_t = m}\gamma_t(k)}{\sum_t\gamma_t(k)}.$$

with several sequences, add up the expected counts from all of them before normalizing. each iteration provably never decreases the likelihood (EM maximizes a lower bound that touches the likelihood at the current parameters; topic 07 makes this precise), so you can stop when the improvement is below `tol`. it only finds a *local* maximum, though, and which one depends on the starting point: run it from several seeds. two more facts worth knowing. the states come out in an arbitrary order (**label switching**: nothing tells the model that state 0 should be "fair"). and the maximum-likelihood parameters fit the training sequence *better* than the true parameters do (warm-up 4): with little data, they partly fit the noise.

> **spec.** `fit(X, y=None)` takes one int sequence or a list of them. if `startprob_`, `transmat_` and `emissionprob_` are all already set, start from them; otherwise draw each row from a Dirichlet(1) with `as_rng(self.rng)`. each iteration: E-step on every sequence (pooling the counts), append the total log-likelihood to `history_`, then the M-step; stop after `max_iters` iterations or when the log-likelihood improved by less than `tol` since the previous iteration. set `n_iter_` and return `self`. `score(X, y=None)` returns the total log-likelihood of one sequence or a list.

> **▶ build: stage 4.** implement `fit` and `score`.

## worked example

$K = 2$, $M = 2$: $\pi = (0.5, 0.5)$, $A = \begin{pmatrix}0.9 & 0.1\\0.2 & 0.8\end{pmatrix}$, $B = \begin{pmatrix}0.8 & 0.2\\0.3 & 0.7\end{pmatrix}$, observations $x = (0, 1)$.

**forward.** $\alpha_1 = (0.5\cdot0.8, 0.5\cdot0.3) = (0.4, 0.15)$. then $\alpha_1A = (0.4\cdot0.9 + 0.15\cdot0.2,\ 0.4\cdot0.1 + 0.15\cdot0.8) = (0.39, 0.16)$, times the emissions of symbol 1, $(0.2, 0.7)$: $\alpha_2 = (0.078, 0.112)$. so $p(x) = 0.19$ and $\log p(x) = -1.6607$. (scaled: $c_1 = 0.55$, $c_2 = 0.19/0.55$.)

**backward and posteriors.** $\beta_2 = (1, 1)$, $\beta_1 = A\,(0.2, 0.7)^\top = (0.25, 0.6)$. $\gamma_1 \propto (0.4\cdot0.25, 0.15\cdot0.6) = (0.1, 0.09)$, so $\gamma_1 = (0.526, 0.474)$; $\gamma_2 = (0.078, 0.112)/0.19 = (0.411, 0.589)$. posterior decoding says: state 0, then state 1.

**Viterbi.** $\delta_1 = (0.4, 0.15)$ (as probabilities). $\delta_2(0) = \max(0.4\cdot0.9, 0.15\cdot0.2)\cdot0.2 = 0.072$ (from 0), $\delta_2(1) = \max(0.4\cdot0.1, 0.15\cdot0.8)\cdot0.7 = 0.084$ (from 1). the best path ends in 1 and came from 1: $z^* = (1, 1)$ with $p(x, z^*) = 0.084$. the posterior-decoded path $(0, 1)$ has $p(x, z) = 0.4\cdot0.1\cdot0.7 = 0.028$, three times less likely, even though each of its states is individually the most probable. your `forward_backward` and `viterbi` must reproduce these numbers.

## on paper

1. **the forward recursion.** derive $\alpha_{t+1}(j) = \big(\sum_k\alpha_t(k)A_{kj}\big)B_{jx_{t+1}}$ from the definition $\alpha_t(k) = p(x_{1:t}, z_t = k)$.

<details><summary>solution</summary>

$\alpha_{t+1}(j) = \sum_kp(x_{1:t}, z_t = k, z_{t+1} = j, x_{t+1}) = \sum_kp(x_{1:t}, z_t = k)\,p(z_{t+1} = j \mid z_t = k)\,p(x_{t+1} \mid z_{t+1} = j)$, using that $z_{t+1}$ depends only on $z_t$ and $x_{t+1}$ only on $z_{t+1}$. that's $\sum_k\alpha_t(k)A_{kj}B_{jx_{t+1}}$.

</details>

2. **scaling.** show that $\hat\alpha_t = \alpha_t / \prod_{s \le t}c_s$ with $c_s$ the normalizer at step $s$, and that $\prod_{s \le T}c_s = p(x_{1:T})$.

<details><summary>solution</summary>

by induction: $\hat\alpha_1 = \alpha_1/c_1$ with $c_1 = \sum\alpha_1 = p(x_1)$. if $\hat\alpha_t = \alpha_t/p(x_{1:t})$, the unnormalized update gives $(\hat\alpha_tA) \odot B_{\cdot x_{t+1}} = \alpha_{t+1}/p(x_{1:t})$, whose sum is $p(x_{1:t+1})/p(x_{1:t}) = c_{t+1}$, so dividing gives $\alpha_{t+1}/p(x_{1:t+1})$. the product of the $c_s$ telescopes to $p(x_{1:T})$.

</details>

3. **the pairwise posterior.** derive $\xi_t(i, j) = \alpha_t(i)A_{ij}B_{jx_{t+1}}\beta_{t+1}(j)/p(x)$, and check that $\sum_j\xi_t(i, j) = \gamma_t(i)$.

<details><summary>solution</summary>

$p(z_t = i, z_{t+1} = j, x) = p(x_{1:t}, z_t = i)\,A_{ij}\,B_{jx_{t+1}}\,p(x_{t+2:T} \mid z_{t+1} = j)$ by the chain's independences. summing over $j$: $\alpha_t(i)\sum_jA_{ij}B_{jx_{t+1}}\beta_{t+1}(j) = \alpha_t(i)\beta_t(i)$, which over $p(x)$ is $\gamma_t(i)$. in scaled form, $\alpha_t\beta_{t+1}/p(x) = \hat\alpha_t\hat\beta_{t+1}/c_{t+1}$.

</details>

4. **the M-step for $B$.** the expected complete-data log-likelihood contains $\sum_t\sum_k\gamma_t(k)\log B_{kx_t}$. maximize it over each row of $B$ subject to $\sum_mB_{km} = 1$.

<details><summary>solution</summary>

for row $k$ the objective is $\sum_mn_{km}\log B_{km}$ with expected counts $n_{km} = \sum_{t:\,x_t = m}\gamma_t(k)$. a Lagrange multiplier for the constraint gives $n_{km}/B_{km} = \lambda$, so $B_{km} = n_{km}/\sum_{m'}n_{km'}$: normalized expected counts, exactly like the fully observed case.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how well Viterbi, posterior decoding and a memoryless rule recover when the casino's loaded die was in play.
2. what Baum–Welch learns about the casino from scratch, from different random starts, and how it compares with the true parameters.
3. how decoding accuracy changes as the croupier switches dice more and more often.

## check your understanding

1. why is the forward algorithm $O(TK^2)$ and not $O(K^T)$?

<details><summary>answer</summary>

$\alpha_t$ summarizes everything about $x_{1:t}$ that the future needs, in $K$ numbers, because of the Markov property. each step combines $K$ old values for each of $K$ new ones: $K^2$ per step.

</details>

2. what's $\hat\alpha_T$, and how would you use it to track a hidden state *online*, as observations arrive?

<details><summary>answer</summary>

$\hat\alpha_t = P(z_t \mid x_{1:t})$: the **filtering** distribution. online you update it with one forward step per new observation; no backward pass needed. forward–backward computes the **smoothing** distribution $P(z_t \mid x_{1:T})$, which uses the future too.

</details>

3. why can posterior decoding output an impossible path, and why can't Viterbi?

<details><summary>answer</summary>

posterior decoding picks each $z_t$ separately; if $A_{ij} = 0$, nothing stops it choosing $i$ at $t$ and $j$ at $t + 1$. Viterbi maximizes the joint probability of the whole path, and a path with a zero-probability transition has joint probability 0.

</details>

4. Baum–Welch's log-likelihood went *down* between two iterations. what's wrong?

<details><summary>answer</summary>

a bug: EM never decreases it. usual suspects: the M-step uses counts from different parameters than the E-step, the counts aren't pooled across sequences before normalizing, or $\xi$ is missing the $1/c_{t+1}$ scaling.

</details>

5. you fit a 2-state HMM to the casino and state 0 comes out loaded. is the model wrong?

<details><summary>answer</summary>

no: the likelihood is invariant to permuting the state labels (label switching), so either assignment is equally good. identify the states by their emissions after fitting.

</details>

6. what happens to Baum–Welch if a symbol never appears in the training data?

<details><summary>answer</summary>

its expected count is 0 in every state, so its emission probability becomes exactly 0, and any test sequence containing it gets likelihood 0. in practice you add pseudo-counts (a Dirichlet prior: MAP-EM) to avoid this.

</details>

## common bugs

- **underflow.** symptom: `nan` posteriors or a log-likelihood of `-inf` on sequences longer than a few hundred steps. scale the messages (or work in log space).
- **emission row vs column.** symptom: shape errors, or wrong numbers that pass on square examples. $B$ is (states, symbols); the emission likelihoods of an observation sequence are the columns `B[:, obs]`, shape $(K, T)$.
- **transposed transitions.** symptom: forward probabilities that are fine for symmetric $A$ and wrong otherwise. $\alpha_{t+1} = (\alpha_tA) \odot B$: a row vector times $A$, because $A_{ij}$ is from $i$ to $j$.
- **off-by-one in the back-pointers.** symptom: Viterbi paths shifted by one step, or a `log_prob` that doesn't match the returned path. $\psi_{t+1}$ points from step $t + 1$ back to step $t$.
- **re-initializing in `fit`.** symptom: one EM step from preset parameters disagrees with expected counts. start from the current parameters when all three are set.
- **counting $\xi$ for the last step.** symptom: transition counts slightly off. there are $T - 1$ transitions in a sequence of length $T$.

## going deeper

- Rabiner (1989), "A tutorial on hidden Markov models and selected applications in speech recognition", *Proceedings of the IEEE* 77(2). the classic reference, scaling included. [doi:10.1109/5.18626](https://doi.org/10.1109/5.18626)
- Durbin, Eddy, Krogh & Mitchison (1998), *Biological Sequence Analysis*, Cambridge University Press, ch. 3. the dishonest casino and HMMs for DNA. [doi:10.1017/CBO9780511790492](https://doi.org/10.1017/CBO9780511790492)
- Baum, Petrie, Soules & Weiss (1970), "A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains", *Annals of Mathematical Statistics* 41(1). [doi:10.1214/aoms/1177697196](https://doi.org/10.1214/aoms/1177697196)
- Viterbi (1967), "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm", *IEEE Transactions on Information Theory* 13(2). [doi:10.1109/TIT.1967.1054010](https://doi.org/10.1109/TIT.1967.1054010)
- Murphy (2012), *Machine Learning: A Probabilistic Perspective*, MIT Press, ch. 17.
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, §13.2.

## connections

- ← builds on: Markov chains (ch 04 · topic 07), elimination and sum-/max-product on chains (topic 03), maximum likelihood for categorical distributions (ch 05), log-space arithmetic (ch 00), the `Estimator` protocol (ch 07).
- → used in: EM as coordinate ascent on the ELBO (topic 07), Gibbs sampling of hidden states (topic 05), Kalman filters, the linear-Gaussian HMM (ch 29 elective), and RNNs as learned, continuous-state alternatives (ch 17).
