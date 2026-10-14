# bandits

## why this matters

you run a website with three headlines for the same article. each visitor sees one, and either clicks or doesn't. you don't know the click rates. if you show the headline that has done best so far, you might lock onto a lucky early winner and never learn that another one is better. if you keep showing all three equally, you learn the rates perfectly and lose clicks on every visitor who sees a worse headline. every visitor is both a measurement and a cost.

that's the **exploration–exploitation** dilemma, and it's the first thing that makes reinforcement learning different from supervised learning. in supervised learning someone hands you labelled data. here *your own choices decide what data you get*: you only see the reward of the arm you pulled. the bandit is the simplest setting with this feedback loop (no states, no delayed consequences, just repeated choices) and it already needs real ideas: how to measure the cost of learning (regret), why the obvious strategies fail, how optimism and posterior sampling fix them, and a theorem that says how well *any* algorithm can do. every later topic in this chapter inherits the dilemma: Q-learning and SARSA explore with exactly the ε-greedy rule you build here.

## warm-up: predict first

1. you always play the arm with the best average so far (pure greedy). over 10,000 rounds, does your total loss compared with always playing the best arm grow like $\log T$, like $\sqrt T$, or like $T$?
2. ε-greedy with $\varepsilon = 0.1$ on three arms. when it has correctly identified the best arm, how often does it play it: 90%, 93.3%, or 100%?
3. two arms with click rates $0.50$ and $0.51$. is that problem *harder* or *easier* than $0.50$ versus $0.90$, in terms of total clicks lost over 1,000 visitors?
4. UCB has never pulled arm 3. what does it do?
5. can any algorithm have regret that stays bounded as $T \to \infty$ on every Bernoulli bandit?

## prerequisites check

- [ ] you can maintain a running mean without storing the samples, $\bar x_n = \bar x_{n-1} + (x_n - \bar x_{n-1})/n$ (→ ch 04 · topic 03)
- [ ] you know the Beta distribution and that it's the conjugate prior of the Bernoulli (→ ch 04 · topic 02, ch 05 · topic 02)
- [ ] you know the KL divergence between two distributions (→ ch 05 · topic 05)
- [ ] you know Hoeffding's inequality for averages of bounded variables (→ ch 04 · topic 06)
- [ ] you can pass a seed or a `Generator` around with `as_rng` instead of using global random state (→ ch 00 · topic 05)

## core ideas

### idea 1: the testbed and regret

a **$k$-armed bandit** has arms $a = 0, \dots, k - 1$. pulling arm $a$ returns a random reward with unknown mean $\mu_a$. the best arm has mean $\mu^* = \max_a\mu_a$, and its **gap** is zero; every other arm has gap $\Delta_a = \mu^* - \mu_a > 0$. in round $t$ an agent picks $A_t$, sees $R_t$, and updates its beliefs. that's the whole protocol: two methods, `select_arm()` and `update(arm, reward)`, and a loop.

how do you score an agent? the total reward is noisy, and it depends on how generous the bandit happens to be. the standard measure compares the agent with an oracle that always plays the best arm. the **(pseudo-)regret** after $T$ rounds is

$$\bar R_T = \sum_{t=1}^{T}\big(\mu^* - \mu_{A_t}\big) = \sum_a \Delta_a\,N_a(T),$$

where $N_a(T)$ counts the pulls of arm $a$. it uses the *means* of the arms you chose, not the rewards you happened to get, so it isn't affected by reward noise, only by your choices. regret is the price of not knowing the means. an algorithm is good when its regret grows *sublinearly*, $\bar R_T/T \to 0$: eventually it almost always plays the best arm.

the second form, $\sum_a\Delta_aN_a(T)$, is the one to remember: regret is "how many times did you pull each bad arm, weighted by how bad it is". every bound in this topic is a bound on $N_a(T)$.

two testbeds cover most of what you need: **Bernoulli** arms (reward 1 with probability $\mu_a$, else 0: clicks, conversions, wins) and **Gaussian** arms (reward $\mathcal N(\mu_a, \sigma^2)$).

> **spec.** `BernoulliBandit(probs, *, rng=None)` and `GaussianBandit(means, *, std=1.0, rng=None)` store `means` (a float array), `n_arms`, `optimal_arm` (an `int`, the argmax of the means) and `self.rng = as_rng(rng)`. `pull(arm)` returns a Python float: `float(self.rng.random() < means[arm])` for Bernoulli, a normal draw for Gaussian. `run_bandit(bandit, agent, n_steps)` plays `n_steps` rounds of `arm = agent.select_arm(); r = bandit.pull(arm); agent.update(arm, r)` and returns `(rewards, regret)`, both shape `(n_steps,)`: the rewards received and the *cumulative* pseudo-regret after each round.

> **▶ build: stage 1.** implement `BernoulliBandit`, `GaussianBandit` and `run_bandit`.

### idea 2: ε-greedy, and why greedy fails

the natural agent keeps a **sample average** $Q_a$ of each arm's rewards and plays the arm with the largest. update it incrementally, without storing rewards:

$$N_a \leftarrow N_a + 1,\qquad Q_a \leftarrow Q_a + \frac{1}{N_a}\big(R - Q_a\big).$$

pure greedy can fail forever. suppose the best arm pays 0 on its first pull and a worse arm pays 1 on its first. greedy now believes the worse arm is better, keeps playing it, and never collects the evidence that would change its mind: the bad estimate is never corrected because acting on it stops the data that would correct it. that happens with constant probability, and each time it happens the agent loses $\Delta$ on *every* later round. so greedy's expected regret grows linearly in $T$ (warm-up 1).

**ε-greedy** patches this: with probability $\varepsilon$ play a uniformly random arm, otherwise play greedily. now every arm is pulled about $\varepsilon T/k$ times, so every estimate converges and the agent eventually knows the best arm. but it never stops exploring. once it knows the best arm, it plays it with probability $(1 - \varepsilon) + \varepsilon/k$ (the random arm can also be the best one: $0.9 + 0.1/3 = 0.933$ for warm-up 2), and pays $\varepsilon\cdot\frac1k\sum_a\Delta_a$ per round forever. its regret is linear too, just with a much smaller slope. the fix (decay $\varepsilon_t$ like $1/t$) works in theory but needs the gaps to tune.

one more detail matters more than it looks. at the start all $Q_a$ are 0, and with Bernoulli rewards many arms tie for long stretches. `np.argmax` returns the *first* maximum, so a "greedy" agent with that rule always prefers arm 0 among ties: a systematic bias towards low indices. break ties uniformly at random.

> **spec.** `EpsilonGreedy(n_arms, epsilon=0.1, *, rng=None)`: attributes `Q` (float zeros) and `N` (int zeros), both `(n_arms,)`, and `self.rng = as_rng(rng)`. `select_arm()`: draw `self.rng.random()`; if it's below `epsilon`, return a uniformly random arm (`self.rng.integers(n_arms)`, any arm, including the greedy one); otherwise return a uniformly random arm among those with `Q == Q.max()`. return Python ints.

> **🔮 predict:** you run ε-greedy with $\varepsilon = 0$ on three arms with means $(0.2, 0.5, 0.8)$ and *random* tie-breaking. what fraction of runs end up stuck on a bad arm?

> **▶ build: stage 2.** implement `EpsilonGreedy`.

### idea 3: optimism in the face of uncertainty (UCB)

ε-greedy explores blindly: it is as likely to re-try an arm it has pulled 5,000 times as one it has pulled 3 times. a better rule explores where it's *uncertain*. by Hoeffding's inequality, after $N_a$ pulls the sample average of a $[0, 1]$-valued arm is within $\sqrt{\ln(1/\delta)/(2N_a)}$ of the true mean with probability at least $1 - \delta$. so act as if each arm were as good as it *plausibly* could be:

$$A_t = \arg\max_a\left(Q_a + c\sqrt{\frac{\ln t}{N_a}}\right).$$

the bonus is large for arms you've rarely tried and shrinks as $1/\sqrt{N_a}$. the slowly growing $\ln t$ makes sure no arm is abandoned for good: an arm that hasn't been pulled for a long time sees its bonus creep up until it gets another chance. this is the principle of **optimism in the face of uncertainty**: if an arm is bad, playing it optimistically reveals that quickly (its upper bound falls); if it's good, you wanted to play it anyway. either way you learn something or you earn something.

why is the regret logarithmic? a bad arm $a$ is only chosen while its upper bound exceeds the best arm's, which (with high probability) is at least $\mu^*$. that needs the bonus to cover the gap, $c\sqrt{\ln t/N_a} \gtrsim \Delta_a$, i.e. $N_a \lesssim c^2\ln t/\Delta_a^2$. so each bad arm is pulled $O(\ln T/\Delta_a^2)$ times, and $\bar R_T = \sum_a\Delta_aN_a = O\big(\sum_a\ln T/\Delta_a\big)$ (Auer, Cesa-Bianchi & Fischer, 2002, prove this for $c = \sqrt2$). small gaps need many pulls to resolve, but each of those pulls costs little.

> **spec.** `UCB(n_arms, c=2.0)` (no randomness at all): `Q`, `N` as before. `select_arm()`: if some arm has `N == 0`, return the lowest such index. otherwise return the argmax of `Q + c * sqrt(ln t / N)` with **$t$ = the number of pulls so far** (`N.sum()`), ties to the lowest index (`np.argmax`). `update` is the sample average.

> **▶ build: stage 3.** implement `UCB`.

### idea 4: Thompson sampling and the lower bound

a Bayesian keeps a posterior over each arm's mean. for Bernoulli arms with a $\text{Beta}(\alpha_0, \beta_0)$ prior, the posterior after $s$ successes and $f$ failures is $\text{Beta}(\alpha_0 + s, \beta_0 + f)$ (conjugacy, ch 05 · topic 02). which arm should a Bayesian play? **Thompson sampling** (Thompson, 1933) answers with a trick: draw one plausible mean from each posterior, $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$, and play the argmax of the draws. then

$$P(A_t = a) = P\big(a \text{ is the best arm} \mid \text{data so far}\big),$$

so each arm is played exactly as often as the posterior believes it might be the best (**probability matching**). an arm with a wide posterior sometimes draws a high $\theta$ and gets explored; an arm that is confidently bad almost never does. there's no tuning constant, and empirically it's one of the strongest bandit algorithms there is. Agrawal & Goyal (2012) proved its regret is logarithmic, and Kaufmann, Korda & Munos (2012) and Agrawal & Goyal (2013) proved that for Bernoulli arms it achieves the best possible constant. rewards don't even need to be 0/1: adding $r \in [0, 1]$ to $\alpha$ and $1 - r$ to $\beta$ gives a sensible "fractional" update.

how good can *any* algorithm be? Lai & Robbins (1985) proved that for every algorithm that is reasonable on all Bernoulli bandits (regret $o(T^p)$ for every $p > 0$),

$$\liminf_{T\to\infty}\frac{\bar R_T}{\ln T} \;\ge\; \sum_{a:\,\Delta_a > 0}\frac{\Delta_a}{\mathrm{KL}(\mu_a, \mu^*)},\qquad \mathrm{KL}(p, q) = p\ln\frac pq + (1 - p)\ln\frac{1 - p}{1 - q}.$$

the intuition: to be sure arm $a$ isn't the best, you need enough pulls to tell $\mu_a$ apart from $\mu^*$, and the number of samples needed to distinguish two Bernoullis with confidence $1 - 1/T$ is about $\ln T/\mathrm{KL}$. so logarithmic regret is unavoidable (warm-up 5: no), and the constant depends on how hard the arms are to tell apart. note that it's an *asymptotic* statement; the lab shows what it does and doesn't say about the first few thousand rounds.

> **spec.** `ThompsonSampling(n_arms, *, prior=(1.0, 1.0), rng=None)`: float arrays `alpha` and `beta` of shape `(n_arms,)`, filled with `prior[0]` and `prior[1]`. `select_arm()` draws all arms at once with `self.rng.beta(self.alpha, self.beta)` and returns the argmax. `update(arm, r)` adds `r` to `alpha[arm]` and `1 - r` to `beta[arm]`. `lai_robbins_constant(means)` returns $\sum_{a:\Delta_a > 0}\Delta_a/\mathrm{KL}(\mu_a, \mu^*)$ with $0\ln0 = 0$ (arms tied for best contribute nothing).

> **🔮 predict:** two arms with posteriors $\text{Beta}(3, 2)$ and $\text{Beta}(2, 3)$. how often does Thompson sampling play arm 0: always, 3/5 of the time, or something else?

> **▶ build: stage 4.** implement `ThompsonSampling` and `lai_robbins_constant`.

## worked example

**ε-greedy.** with $\varepsilon = 0.1$, three arms and $Q = (0, 1, 0)$, the greedy arm 1 is played with probability $0.9 + 0.1 \cdot \tfrac13 = 0.9333$; each other arm with $0.0333$.

**UCB.** three arms, $c = 2$. the rewards so far: arm 0 paid 1, 1, 0; arm 1 paid 0, 1; arm 2 paid 0. so $Q = (\tfrac23, \tfrac12, 0)$, $N = (3, 2, 1)$ and $t = 6$, $\ln 6 = 1.792$. the upper bounds are

$$\tfrac23 + 2\sqrt{1.792/3} = 2.212,\qquad \tfrac12 + 2\sqrt{1.792/2} = 2.393,\qquad 0 + 2\sqrt{1.792/1} = 2.677,$$

so UCB plays arm 2, the arm with the *worst* average: it has been tried only once. with $c = 0.3$ the bonuses shrink to $0.232, 0.284, 0.402$, the bounds become $0.899, 0.784, 0.402$, and it plays arm 0.

**Thompson.** with posteriors $\text{Beta}(3, 2)$ (mean 0.6) and $\text{Beta}(2, 3)$ (mean 0.4), $P(\theta_0 > \theta_1) = \int_0^1 p_{3,2}(x)\,F_{2,3}(x)\,dx = 53/70 \approx 0.757$. Thompson plays arm 0 about 76% of the time, not 100% (that would be greedy on the posterior mean) and not 60%.

**the lower bound** for means $(0.3, 0.5, 0.7)$: $\mathrm{KL}(0.3, 0.7) = 0.339$ and $\mathrm{KL}(0.5, 0.7) = 0.0872$, so the constant is $0.4/0.339 + 0.2/0.0872 = 1.180 + 2.294 = 3.474$. the arm with the *smaller* gap contributes twice as much regret. your code must reproduce all four calculations.

## on paper

1. **regret decomposition.** show that $\mathbb E\big[\sum_{t}(\mu^* - R_t)\big] = \sum_a\Delta_a\,\mathbb E[N_a(T)]$.

<details><summary>solution</summary>

$\mathbb E[R_t \mid A_t] = \mu_{A_t}$, so $\mathbb E[\mu^* - R_t] = \mathbb E[\mu^* - \mu_{A_t}] = \mathbb E\big[\sum_a\Delta_a\mathbb 1\{A_t = a\}\big]$. summing over $t$ and swapping the sums gives $\sum_a\Delta_a\mathbb E\big[\sum_t\mathbb 1\{A_t = a\}\big] = \sum_a\Delta_a\mathbb E[N_a(T)]$. this is why pseudo-regret $\sum_t(\mu^* - \mu_{A_t})$ has the same expectation as the realized one but less noise.

</details>

2. **greedy is linear.** two Bernoulli arms with means $0.8$ (arm 0) and $0.2$ (arm 1). greedy, with ties broken at random, starts by pulling each arm once. (a) with what probability does arm 0 pay 0 and arm 1 pay 1 on those pulls? (b) argue that in that case greedy never plays arm 0 again. (c) conclude that its expected regret is at least $cT$ for a constant $c$; find one.

<details><summary>solution</summary>

(a) $0.2 \cdot 0.2 = 0.04$. (b) now $Q_0 = 0$ and $Q_1 = 1$. $Q_1$ is an average of 0/1 rewards and stays $> 0$ as long as arm 1 has paid at least once, which it has; $Q_0$ stays 0 because arm 0 is never pulled again. so arm 1 is greedy forever. (c) on that event every later round costs $\Delta = 0.6$, so $\mathbb E[\bar R_T] \ge 0.04 \cdot 0.6\,(T - 2) \approx 0.024\,T$.

</details>

3. **UCB's pull count.** take UCB1's bonus $\sqrt{2\ln t/N}$ (i.e. $c = \sqrt2$) and rewards in $[0, 1]$. suppose that at some round arm $a$ has $Q_a \le \mu_a + \sqrt{2\ln t/N_a}$ and the best arm has $Q^* + \sqrt{2\ln t/N^*} \ge \mu^*$. show that arm $a$ can then only be chosen if $N_a \le 8\ln t/\Delta_a^2$, and sketch why $\mathbb E[N_a(T)] \le 8\ln T/\Delta_a^2 + O(1)$.

<details><summary>solution</summary>

arm $a$ is chosen only if its bound is at least the best arm's: $Q_a + \sqrt{2\ln t/N_a} \ge Q^* + \sqrt{2\ln t/N^*} \ge \mu^*$. using the first assumption, $\mu_a + 2\sqrt{2\ln t/N_a} \ge \mu^*$, so $2\sqrt{2\ln t/N_a} \ge \Delta_a$, i.e. $N_a \le 8\ln t/\Delta_a^2$. for the expectation, count the rounds where an assumption fails. the counts $N_a$ and $N^*$ are random, so take a union over their possible values: for each $s \le t$, Hoeffding gives $P(\bar x_{a,s} > \mu_a + \sqrt{2\ln t/s}) \le e^{-4\ln t} = t^{-4}$, and the same for the best arm; summing over $s$ for both arms gives at most $2t \cdot t^{-4} = 2t^{-3}$ per round, and $\sum_t2t^{-3}$ is a constant. so $\mathbb E[N_a(T)] \le 8\ln T/\Delta_a^2 + O(1)$ (Auer et al., 2002, Theorem 1).

</details>

4. **the Beta posterior.** with a $\text{Beta}(\alpha, \beta)$ prior on $\mu$ and one Bernoulli reward $r$, show the posterior is $\text{Beta}(\alpha + r, \beta + 1 - r)$.

<details><summary>solution</summary>

$p(\mu \mid r) \propto \mu^r(1 - \mu)^{1 - r}\cdot\mu^{\alpha - 1}(1 - \mu)^{\beta - 1} = \mu^{\alpha + r - 1}(1 - \mu)^{\beta + (1 - r) - 1}$, the kernel of $\text{Beta}(\alpha + r, \beta + 1 - r)$.

</details>

## experiment

`lab.py` uses your agents on three questions:

1. regret curves for greedy, ε-greedy, UCB and Thompson sampling on the same bandits: who wins early, who wins late, and what shape is each curve?
2. how regret depends on the gap between two arms.
3. how Thompson sampling's regret compares with the Lai–Robbins bound.

## check your understanding

1. why does the pseudo-regret use $\mu_{A_t}$ rather than the reward $R_t$ you actually got?

<details><summary>answer</summary>

the realized reward adds noise that has nothing to do with your decisions (a good arm can pay 0). pseudo-regret has the same expectation (on paper 1) and measures only the cost of your choices, so curves are smoother and comparisons fairer.

</details>

2. ε-greedy's regret is linear. why is it still often used (it's how Q-learning explores in topic 05)?

<details><summary>answer</summary>

it's simple, needs no model of uncertainty, works with any value estimates (including neural networks, ch 24), and its linear slope $\varepsilon\bar\Delta$ is small for small $\varepsilon$. in practice $\varepsilon$ is decayed over time. UCB-style bonuses need counts, which don't exist for continuous or high-dimensional states.

</details>

3. (warm-up 4.) UCB has never pulled arm 3. what does it do, and why is that consistent with its formula?

<details><summary>answer</summary>

it pulls it (the lowest untried arm first). with $N_a = 0$ the bonus $\sqrt{\ln t/N_a}$ is infinite: no data means the arm could be arbitrarily good.

</details>

4. what happens to UCB if rewards are in $[0, 100]$ but you keep $c = 2$?

<details><summary>answer</summary>

the bonus is tuned for rewards of scale 1. with rewards of scale 100 the differences in $Q$ dwarf the bonus, and UCB becomes nearly greedy: it can lock onto a bad arm for a long time. scale $c$ with the reward range (Hoeffding's bound is proportional to it), or normalize rewards.

</details>

5. Thompson sampling is randomized, UCB is deterministic. name a situation where that difference matters in practice.

<details><summary>answer</summary>

delayed or batched feedback: if you must choose arms for 1,000 visitors before seeing any results, deterministic UCB shows all of them the same arm, while Thompson naturally spreads them across plausible arms. (also: an adversary can predict a deterministic algorithm.)

</details>

6. (the predict callout in idea 4.) why isn't the answer $3/5$, the posterior mean of arm 0?

<details><summary>answer</summary>

Thompson plays arm 0 with the probability that arm 0 is *better than arm 1*, $P(\theta_0 > \theta_1) = 53/70 \approx 0.757$, which depends on both posteriors. the posterior mean of a single arm isn't a probability of being best.

</details>

7. the Lai–Robbins constant is $\sum\Delta_a/\mathrm{KL}(\mu_a, \mu^*)$. why does an arm with a tiny gap contribute *more* than one with a big gap, when it costs less per pull?

<details><summary>answer</summary>

$\mathrm{KL}(\mu_a, \mu^*) \approx \Delta_a^2/(2\mu^*(1 - \mu^*))$ for small gaps, so $\Delta_a/\mathrm{KL} \sim 1/\Delta_a$: the number of pulls needed grows like $1/\Delta_a^2$, faster than the cost per pull shrinks. (but only asymptotically: over a fixed horizon, an arm so close that you can't resolve it costs at most $\Delta_aT$. the lab looks at both effects.)

</details>

## common bugs

- **ties to the lowest index.** symptom: greedy and ε-greedy prefer arm 0 at the start, and results depend on the arm order. break ties at random among `np.flatnonzero(Q == Q.max())`.
- **exploring only the non-greedy arms.** symptom: the greedy arm is played with probability exactly $1 - \varepsilon$ instead of $1 - \varepsilon + \varepsilon/k$. the spec says the random arm is uniform over *all* arms.
- **the realized reward in the regret.** symptom: noisy regret curves that can go down. use $\mu^* - \mu_{A_t}$.
- **dividing by zero in UCB.** symptom: `RuntimeWarning: divide by zero` and NaN bounds. try each arm once before using the formula.
- **Thompson on the posterior mean.** symptom: Thompson behaves exactly like greedy and sometimes locks onto a bad arm. draw $\theta_a$ from the posterior; don't use $\alpha/(\alpha + \beta)$.
- **global random state.** symptom: results change from run to run with the same seed. every random draw goes through `self.rng`.

## going deeper

- Lattimore & Szepesvári (2020), *Bandit Algorithms*, Cambridge University Press, chapters 4–8 (stochastic bandits, UCB) and 16 (lower bounds); free online. [tor-lattimore.com/downloads/book/book.pdf](https://tor-lattimore.com/downloads/book/book.pdf)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapter 2 (multi-armed bandits, the 10-armed testbed). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Auer, Cesa-Bianchi & Fischer (2002), "Finite-time analysis of the multiarmed bandit problem", *Machine Learning* 47. UCB1 and its regret bound. [doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)
- Lai & Robbins (1985), "Asymptotically efficient adaptive allocation rules", *Advances in Applied Mathematics* 6(1). the lower bound. [doi:10.1016/0196-8858(85)90002-8](https://doi.org/10.1016/0196-8858(85)90002-8)
- Russo, Van Roy, Kazerouni, Osband & Wen (2018), "A tutorial on Thompson sampling", *Foundations and Trends in Machine Learning* 11(1). [arXiv:1707.02038](https://arxiv.org/abs/1707.02038)
- Thompson (1933), "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples", *Biometrika* 25(3/4). where Thompson sampling comes from. [doi:10.2307/2332286](https://doi.org/10.2307/2332286)
- Kaufmann, Korda & Munos (2012), "Thompson sampling: an asymptotically optimal finite-time analysis", *ALT*. [arXiv:1205.4217](https://arxiv.org/abs/1205.4217)
- Agrawal & Goyal (2012), "Analysis of Thompson sampling for the multi-armed bandit problem", *COLT*. [arXiv:1111.1797](https://arxiv.org/abs/1111.1797)
- Agrawal & Goyal (2013), "Further optimal regret bounds for Thompson sampling", *AISTATS*. [arXiv:1209.3353](https://arxiv.org/abs/1209.3353)

## connections

- ← builds on: running means (ch 04 · topic 03), Hoeffding's inequality (ch 04 · topic 06), Beta–Bernoulli updating and KL divergence (ch 05 · topics 02 and 05), seeding with `as_rng` (ch 00).
- → used in: `epsilon_greedy` exploration in Monte Carlo and TD control (topics 04–05), the value-estimation updates of every later topic (the incremental mean is TD's update with $\alpha = 1/N$), and UCT's tree policy in Monte Carlo tree search (ch 24 · topic 06).
