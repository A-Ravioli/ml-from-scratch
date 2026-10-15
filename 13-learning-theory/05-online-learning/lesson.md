# online learning

## why this matters

every bound so far rested on one assumption: the data are drawn **iid** from a fixed distribution. drop it and the whole edifice of topics 01–04 collapses. but a spam filter faces spammers who adapt to it, a recommender sees tastes that drift, a trading strategy faces markets that react to it, and a California housing stream sorted by latitude looks nothing like a random sample. what can you promise when the data might be chosen by an adversary who knows your algorithm?

online learning gives a surprising answer: quite a lot, if you change what you promise. you can't promise a small error (an adversary can always make the next example hard). you *can* promise a small **regret**: that over $T$ rounds your total loss is not much worse than that of the best *fixed* strategy you could have picked in hindsight, knowing the whole sequence. and "not much worse" means $O(\sqrt T)$, so the average regret per round vanishes, for **every** sequence, adversarial or not.

this topic builds the two workhorse algorithms that achieve it. **Hedge** (multiplicative weights) spreads its bets over a set of experts and shifts weight exponentially away from those that do badly; **online gradient descent** takes one projected gradient step per round. both are tiny, both have regret bounds that are theorems about every possible sequence, and both are ancestors of things you use daily: boosting is Hedge in disguise (chapter 10), SGD is OGD with random data, and Adam's ancestor AdaGrad came from this literature.

## warm-up: predict first

1. ten experts, and each round every expert's loss is an independent fair coin. no expert is better than any other. what's the best possible regret after 10,000 rounds: about 0, or something growing with $T$?
2. the greedy strategy "follow the leader" plays, each round, the expert with the lowest total loss so far. can an adversary make it lose almost every round while some expert loses only half of them?
3. Hedge's regret bound is $\ln N/\eta + \eta T/8$. what happens to the bound if you take $\eta$ very large? very small?
4. you double the number of experts from 1,000 to 2,000. how much does Hedge's worst-case regret grow?
5. an online learner predicts house prices district by district. can its total loss be *smaller* than that of the best fixed linear model fitted to the whole stream in hindsight?

## prerequisites check

- [ ] you know gradient descent, convexity, and projections onto convex sets (→ ch 06 · topics 02, 06)
- [ ] you know the softmax and how to compute it stably with log-sum-exp (→ ch 00)
- [ ] you know Hoeffding's lemma: for $X \in [0, 1]$, $\ln\mathbb Ee^{s(X - \mathbb EX)} \le s^2/8$ (→ ch 05 · topic 01)
- [ ] you know the PAC framework, so you can see what online learning gives up and gains (→ topic 01)

## core ideas

### idea 1: experts and multiplicative weights

the **experts problem** is the cleanest online game. there are $N$ experts (strategies, models, stocks, anything). in each round $t = 1, \dots, T$: you choose a probability distribution $p_t$ over the experts; then the environment reveals a loss $\ell_{t,i} \in [0, 1]$ for each expert; you suffer the expected loss $p_t \cdot \ell_t$. the environment may be an adversary who has seen $p_t$'s update rule (but not the randomness of a draw from it, which is why you're scored on the expectation).

**Hedge** (Freund & Schapire, 1997), also called multiplicative weights or exponential weights, keeps one weight per expert and multiplies it by $e^{-\eta\ell_{t,i}}$ after each round. equivalently, with $L_{t,i} = \sum_{s \le t}\ell_{s,i}$ the cumulative loss of expert $i$,

$$p_{t+1,i} = \frac{e^{-\eta L_{t,i}}}{\sum_j e^{-\eta L_{t,j}}} = \text{softmax}(-\eta L_t)_i.$$

the learning rate $\eta > 0$ sets how fast it abandons experts that do badly: a gap of $1/\eta$ in cumulative loss costs a factor $e$ in weight. as $\eta \to 0$ it stays uniform; as $\eta \to \infty$ it puts all its weight on the current leader. computing $e^{-\eta L}$ directly underflows after a few thousand rounds (every weight becomes 0 and $0/0$ is NaN), so compute the softmax in log space, subtracting the maximum (chapter 00's log-sum-exp).

> **spec.** `Hedge(n_experts, lr)` keeps the cumulative losses (start: zeros). `predict()` returns a new array $p = \text{softmax}(-\eta L)$ of shape `(n_experts,)`, computed stably. `update(losses)` adds a `(n_experts,)` array of losses in $[0, 1]$ to the cumulative losses (raise `ValueError` on a wrong shape) and returns `self`.

> **▶ build: stage 1.** implement `Hedge`.

### idea 2: online convex optimization and OGD

experts are a special case of a more general game. in **online convex optimization** (OCO; Zinkevich, 2003), you choose a point $x_t$ in a convex set $\mathcal K \subset \mathbb R^d$; the environment reveals a convex loss $f_t$; you suffer $f_t(x_t)$. experts are OCO with $\mathcal K$ the probability simplex and linear losses $f_t(p) = p \cdot \ell_t$. online regression is OCO with $f_t(w) = \frac12(w^\top x_t - y_t)^2$.

**online gradient descent** is the obvious algorithm: after suffering $f_t(x_t)$, take one gradient step and project back onto $\mathcal K$:

$$x_{t+1} = \Pi_{\mathcal K}\big(x_t - \eta_t\nabla f_t(x_t)\big), \qquad \eta_t = \frac{\eta}{\sqrt t}.$$

it looks exactly like the projected gradient descent of chapter 06, and the code is the same. the difference is the setting: there's no fixed function being minimized, each step uses a gradient of a *different* function, chosen after you committed to $x_t$, and the step sizes decay like $1/\sqrt t$ rather than staying constant or following a line search. the projection keeps the iterates in $\mathcal K$, which the analysis needs, because the guarantee is relative to points in $\mathcal K$.

> **spec.** `OnlineGradientDescent(dim, lr=1.0, *, project=None, decay='sqrt')` starts at $x_1 = 0$ with $t = 0$; `decay` must be `'sqrt'` or `None` (else `ValueError`). `predict()` returns a copy of the current $x$. `update(grad)`: increment $t$, step with $\eta_t = $ `lr / sqrt(t)` (or `lr` when `decay=None`), then apply `project` to the result if it's given; return `self`.

> **▶ build: stage 2.** implement `OnlineGradientDescent`.

### idea 3: regret, and why greed fails

the **regret** after $T$ rounds compares your total loss with that of the best fixed choice in hindsight:

$$R_T = \sum_{t=1}^T p_t \cdot \ell_t - \min_i\sum_{t=1}^T\ell_{t,i}, \qquad R_T^{\text{OCO}} = \sum_{t=1}^Tf_t(x_t) - \min_{x \in \mathcal K}\sum_{t=1}^Tf_t(x).$$

the comparator gets to see the whole sequence but must commit to one expert (one point) for all $T$ rounds. an algorithm is **no-regret** if $R_T/T \to 0$ for every sequence. the regret curve $R_t$, $t = 1, \dots, T$, compares at each time with the best expert *up to that time*, which can change.

the natural greedy strategy, **follow the leader** (FTL), plays the expert with the smallest cumulative loss so far. on iid data it's excellent. against an adversary it's a disaster (warm-up 2): with two experts, losses $(\frac12, 0)$, then $(0, 1), (1, 0), (0, 1), \dots$, the leader is always the expert about to lose. FTL suffers loss 1 in almost every round, each expert about $T/2$, so $R_T \approx T/2$: linear regret, no learning at all. the fix is to not commit to the leader completely; Hedge's softmax is exactly a smoothed version of FTL's argmax.

> **spec.** `follow_the_leader(expert_losses)`: `expert_losses` (T, N); in round $t$ play the expert with the smallest cumulative loss over rounds *before* $t$ (round 1: all zero; ties to the lowest index, `np.argmin`), and return the learner's losses, shape (T,). `regret_curve(learner_losses, expert_losses)` returns $R_t = \sum_{s \le t}\text{learner}_s - \min_i\sum_{s \le t}\ell_{s,i}$ for every $t$, shape (T,).

> **🔮 predict:** on the alternating sequence above with $T = 1000$, what's FTL's regret? what's Hedge's with $\eta = \sqrt{8\ln 2/1000}$?

> **▶ build: stage 3.** implement `follow_the_leader` and `regret_curve`.

### idea 4: Hedge's guarantee

**theorem** (Freund & Schapire 1997; Cesa-Bianchi & Lugosi 2006, Theorem 2.2). for any sequence of losses in $[0, 1]$, Hedge with learning rate $\eta$ satisfies

$$R_T \le \frac{\ln N}{\eta} + \frac{\eta T}{8}.$$

the proof (on paper 1) tracks the potential $\Phi_t = \ln\sum_ie^{-\eta L_{t,i}}$. it can't drop below $-\eta\min_iL_{T,i}$ at the end (one term of the sum is that large), it starts at $\ln N$, and each round it drops by at least $\eta\,p_t \cdot \ell_t - \eta^2/8$ by Hoeffding's lemma. comparing the two ends gives the bound. the two terms are a trade-off (warm-up 3): large $\eta$ means fast adaptation but overreaction (the $\eta T/8$ term); small $\eta$ means a slow start (the $\ln N/\eta$ term, the price of the initial uniform spread). balancing them with $\eta^\star = \sqrt{8\ln N/T}$ gives

$$R_T \le \sqrt{\frac T2\ln N}.$$

$\sqrt T$ is sublinear, so the average regret per round goes to 0 like $\sqrt{\ln N/(2T)}$, and $N$ enters through $\sqrt{\ln N}$ (warm-up 4: doubling 1,000 experts adds 5% to the bound). you can't do better: when every loss is an independent fair coin, every algorithm has expected loss $T/2$, but the best expert in hindsight has expected cumulative loss about $T/2 - \sqrt{(T/2)\ln N}$, just by luck, so *every* algorithm has regret of order $\sqrt{T\ln N}$ (warm-up 1). the regret is large because the comparator gets lucky, not because the learner is bad.

$\eta^\star$ needs $T$ in advance. if you don't know it, the **doubling trick** restarts Hedge with $\eta^\star(2^k)$ at rounds $2^k$, losing only a constant factor; a time-varying $\eta_t \propto \sqrt{\ln N/t}$ also works.

> **spec.** `run_hedge(expert_losses, lr)`: run your `Hedge` over the rows of `expert_losses` (T, N), recording $p_t \cdot \ell_t$ with $p_t$ from `predict()` *before* `update`; return shape (T,). `hedge_optimal_lr(T, n_experts)` returns $\sqrt{8\ln N/T}$; `hedge_regret_bound(T, n_experts, lr)` returns $\ln N/\eta + \eta T/8$.

> **▶ build: stage 4.** implement `run_hedge`, `hedge_optimal_lr` and `hedge_regret_bound`.

### idea 5: OGD's guarantee, and back to statistics

**theorem** (Zinkevich 2003; Hazan 2016, Theorem 3.1). let $\mathcal K$ be convex with diameter $D = \max_{x, y \in \mathcal K}\|x - y\|$, the $f_t$ convex with $\|\nabla f_t(x)\| \le G$ on $\mathcal K$, and $x_1 \in \mathcal K$. OGD with $\eta_t = D/(G\sqrt t)$ (that's `lr = D / G`, `decay='sqrt'`) satisfies, for every sequence,

$$R_T \le \frac32GD\sqrt T.$$

the proof (on paper 2) is the gradient-descent analysis of chapter 06 done one round at a time: convexity gives $f_t(x_t) - f_t(x^\star) \le \nabla f_t(x_t)^\top(x_t - x^\star)$, the projection can only bring $x_{t+1}$ closer to $x^\star \in \mathcal K$, and telescoping $\|x_t - x^\star\|^2$ with decreasing step sizes leaves $\frac{D^2}{2\eta_T} + \frac{G^2}2\sum_t\eta_t$. as with Hedge, $\sqrt T$ is the best possible rate for general convex losses (strongly convex losses allow $\log T$).

for online least squares, $f_t(w) = \frac12(w^\top x_t - y_t)^2$ on the ball $\|w\| \le R$ has gradient $(w^\top x_t - y_t)x_t$, so $G \le \|x\|_{\max}(R\|x\|_{\max} + |y|_{\max})$ and $D = 2R$. these worst-case constants make the bound loose on real data (the lab measures by how much).

**online-to-batch.** regret bounds imply statistical bounds. if the $f_t$ are iid draws of a convex loss and $\bar x = \frac1T\sum_tx_t$, Jensen's inequality and a martingale argument give $\mathbb E[F(\bar x)] - \min_{x \in \mathcal K}F(x) \le \mathbb E[R_T]/T$ for the expected loss $F$ (Cesa-Bianchi, Conconi & Gentile, 2004). a $\sqrt T$ regret becomes a $1/\sqrt T$ excess risk: the agnostic PAC rate of topic 01, with no iid assumption needed in the online analysis itself. and when the data *aren't* iid, regret against a fixed comparator can even be negative (warm-up 5): the comparator is fixed, the learner isn't.

> **spec.** `ogd_regret_bound(T, diameter, grad_bound)` returns $\frac32GD\sqrt T$. `online_least_squares(X, y, *, lr=1.0, radius=1.0, decay='sqrt')`: one round per row; run your `OnlineGradientDescent` with `project=lambda w: project_l2_ball(w, radius)` (from `mlfs.optim`); in each round predict $w_t$, record the loss $\frac12(w_t^\top x_t - y_t)^2$, then update with its gradient. return `(losses (T,), final w)`.

> **🔮 predict:** with $T = 400$, $D = 2$, $G = 3$, what's the OGD guarantee? does the bound depend on the dimension $d$?

> **▶ build: stage 5.** implement `ogd_regret_bound` and `online_least_squares`.

## worked example

**one Hedge round.** three experts, $\eta = \ln 2$, first losses $(1, 0, 0)$. weights $e^{-\eta L} = (\frac12, 1, 1)$, so $p_2 = (0.2, 0.4, 0.4)$. a second round with losses $(0, 1, 0.5)$ gives cumulative losses $(1, 1, 0.5)$ and weights $(\frac12, \frac12, 2^{-1/2})$, normalized $(0.293, 0.293, 0.414)$.

**two OGD steps.** $d = 2$, `lr = 1`, decay `'sqrt'`, from $x_1 = 0$: gradient $(1, 0)$ with $\eta_1 = 1$ gives $x_2 = (-1, 0)$; gradient $(0, 2)$ with $\eta_2 = 1/\sqrt2$ gives $x_3 = (-1, -\sqrt2)$.

**follow the leader.** losses $(0.2, 0.1, 0.9), (1, 0, 0), (0, 1, 0), (0, 0, 1)$. round 1: all cumulative losses are 0, play expert 0, lose 0.2. round 2: cumulative $(0.2, 0.1, 0.9)$, play expert 1, lose 0. round 3: $(1.2, 0.1, 0.9)$, expert 1 again, lose 1. round 4: $(1.2, 1.1, 0.9)$, expert 2, lose 1. FTL's losses are $(0.2, 0, 1, 1)$.

**the bounds.** $T = 5000$, $N = 7$: $\eta^\star = \sqrt{8\ln 7/5000} = 0.0558$ and the guarantee is $\sqrt{2500\ln 7} = 69.8$, an average regret of 0.014 per round. OGD with $T = 400$, $D = 2$, $G = 3$: $1.5 \times 3 \times 2 \times 20 = 180$.

## on paper

1. **Hedge's bound.** with $W_t = \sum_ie^{-\eta L_{t,i}}$ ($W_0 = N$), show $\ln\frac{W_t}{W_{t-1}} \le -\eta\,p_t \cdot \ell_t + \frac{\eta^2}8$, and conclude $R_T \le \frac{\ln N}\eta + \frac{\eta T}8$.

<details><summary>solution</summary>

$\frac{W_t}{W_{t-1}} = \sum_ip_{t,i}e^{-\eta\ell_{t,i}} = \mathbb E_{i \sim p_t}e^{-\eta\ell_{t,i}}$. by Hoeffding's lemma for a $[0, 1]$-valued variable, $\ln\mathbb Ee^{-\eta X} \le -\eta\mathbb EX + \eta^2/8$, with $\mathbb EX = p_t \cdot \ell_t$. summing over $t$: $\ln W_T - \ln N \le -\eta\sum_tp_t \cdot \ell_t + \eta^2T/8$. and $\ln W_T \ge -\eta\min_iL_{T,i}$. combine and divide by $\eta$.

</details>

2. **OGD's bound.** with $g_t = \nabla f_t(x_t)$ and any $x^\star \in \mathcal K$, show $\|x_{t+1} - x^\star\|^2 \le \|x_t - x^\star\|^2 - 2\eta_tg_t^\top(x_t - x^\star) + \eta_t^2G^2$, and derive $R_T \le \frac{D^2}{2\eta_T} + \frac{G^2}2\sum_t\eta_t$. then plug in $\eta_t = D/(G\sqrt t)$.

<details><summary>solution</summary>

projection onto a convex set is non-expansive towards points of the set: $\|\Pi(y) - x^\star\| \le \|y - x^\star\|$. expand $\|x_t - \eta_tg_t - x^\star\|^2$. rearranged, $g_t^\top(x_t - x^\star) \le \frac{\|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2}{2\eta_t} + \frac{\eta_tG^2}2$, and convexity bounds $f_t(x_t) - f_t(x^\star)$ by the left side. summing, the first terms telescope with weights $\frac1{2\eta_t} - \frac1{2\eta_{t-1}} \ge 0$, each multiplying $\|x_t - x^\star\|^2 \le D^2$, so they total at most $\frac{D^2}{2\eta_T}$. with $\eta_t = D/(G\sqrt t)$: $\frac{DG\sqrt T}2 + \frac{GD}2\sum_t\frac1{\sqrt t} \le \frac{DG\sqrt T}2 + GD\sqrt T = \frac32GD\sqrt T$, using $\sum_{t \le T}t^{-1/2} \le 2\sqrt T$.

</details>

3. **FTL's linear regret.** for the alternating sequence of idea 3, compute FTL's total loss and the best expert's total loss after $T = 2k + 1$ rounds.

<details><summary>solution</summary>

after round 1 the cumulative losses are $(\frac12, 0)$, so FTL plays expert 1 in round 2 and loses 1; now $(\frac12, 1)$, so it plays expert 0 in round 3 and loses 1; and so on: from round 2 on it loses 1 every round. FTL's total: $\frac12 + 2k$. each expert loses 1 every other round: expert 0 totals $\frac12 + k$, expert 1 totals $k$. the regret is $\frac12 + k \approx T/2$.

</details>

4. **why $\sqrt T$ is unavoidable.** with $N = 2$ experts whose losses are independent fair coins, show that any algorithm's expected loss is $T/2$ and that $\mathbb E\min(L_{T,1}, L_{T,2}) = T/2 - \frac12\mathbb E|L_{T,1} - L_{T,2}| \approx T/2 - \sqrt{T/(4\pi)}$.

<details><summary>solution</summary>

$p_t$ depends only on past losses, and $\ell_t$ is independent of the past with mean $\frac12$ in each coordinate, so $\mathbb E[p_t \cdot \ell_t] = \frac12$ and the expected total is $T/2$. for the minimum, $\min(a, b) = \frac{a + b}2 - \frac{|a - b|}2$ and $\mathbb E(L_{T,1} + L_{T,2})/2 = T/2$. the difference $L_{T,1} - L_{T,2}$ is a sum of $T$ independent terms equal to $-1, 0, 1$ with probabilities $\frac14, \frac12, \frac14$ (variance $\frac12$ each), so it's approximately $\mathcal N(0, T/2)$ and $\mathbb E|L_{T,1} - L_{T,2}| \approx \sqrt{2/\pi}\sqrt{T/2} = \sqrt{T/\pi}$. so the expected regret of *every* algorithm is about $\frac12\sqrt{T/\pi} = \sqrt{T/(4\pi)}$: of order $\sqrt T$.

</details>

## experiment

`lab.py` investigates, with your code:

1. how Hedge's regret grows with the horizon $T$ on three kinds of sequences, against its guarantee, and its log-log slope.
2. follow-the-leader against Hedge on a stochastic and an alternating sequence.
3. online least squares on a stream of California housing districts, shuffled and sorted by latitude, with two step-size schedules.

## check your understanding

1. why is regret measured against the best *fixed* expert rather than the best expert in each round?

<details><summary>answer</summary>

against the best expert in each round (the "oracle" that always picks the round's winner), no algorithm can have sublinear regret: an adversary can give loss 1 to whatever you're most likely to play. a fixed comparator is the strongest benchmark you can still compete with. (shifting comparators that change $k$ times are a middle ground, with regret growing with $k$.)

</details>

2. how is Hedge related to follow-the-leader?

<details><summary>answer</summary>

Hedge plays $\text{softmax}(-\eta L_t)$; FTL plays $\arg\min L_t$, the limit $\eta \to \infty$. Hedge is FTL with a temperature: its smoothing (an entropy regularizer, in the language of "follow the regularized leader") is what makes it stable against adversaries.

</details>

3. Hedge's guarantee is for the *expected* loss $p_t \cdot \ell_t$. what if you have to pick a single expert each round?

<details><summary>answer</summary>

draw $i_t \sim p_t$ with fresh randomness. the expected loss is the same, and the realized loss is within $O(\sqrt{T\ln(1/\delta)})$ of it with probability $1 - \delta$ (Azuma–Hoeffding), so the regret bound survives, as long as the adversary can't see your random draw before choosing the losses.

</details>

4. why does OGD's bound depend on $G$ and $D$ but not on the dimension?

<details><summary>answer</summary>

the analysis only uses distances and gradient norms in $\mathbb R^d$, never coordinates. dimension enters only if it makes $D$ or $G$ large (for example, the diameter of a cube grows like $\sqrt d$). like the $\ell_2$ Rademacher bound of topic 03, it's a geometric statement.

</details>

5. the online-to-batch conversion turns regret into excess risk. why average the iterates instead of taking the last one?

<details><summary>answer</summary>

the regret bounds the *sum* of the losses of all iterates; by Jensen, the loss of the average is at most the average of the losses, so the average inherits the bound directly. the last iterate has no such guarantee in general (it can be bad on a particular round), although for many problems it also works in practice.

</details>

6. regret against a fixed comparator was negative on a sorted stream. does that mean the regret bound is wrong, or the comparator is weak?

<details><summary>answer</summary>

neither: the bound is an upper bound, and negative values are allowed. the comparator is weak *for drifting data*: one fixed predictor for the whole stream can't follow the drift, while an online learner with a non-vanishing step size can. for such data, "dynamic regret" against a slowly changing comparator is the right yardstick.

</details>

## common bugs

- **multiplying raw weights for thousands of rounds.** symptom: NaN predictions after a few thousand rounds. keep cumulative losses and take a stable softmax.
- **updating before predicting.** symptom: regret that's suspiciously negative; Hedge "sees" the loss before playing. record $p_t \cdot \ell_t$ first.
- **FTL using the current round's loss.** symptom: FTL with zero regret on the alternating sequence (it's cheating).
- **`t` starting at 0 in $\eta/\sqrt t$.** symptom: division by zero, or a first step that's too large. increment $t$ before computing the step.
- **projecting before the step.** symptom: iterates leaving the set; the regret bound fails.
- **the best expert at the end only.** symptom: a regret curve that's wrong in the middle. `np.cumsum` over time, then the minimum over experts at each time.

## going deeper

- Littlestone & Warmuth (1994), "The weighted majority algorithm", *Information and Computation* 108(2). [doi:10.1006/inco.1994.1009](https://doi.org/10.1006/inco.1994.1009)
- Freund & Schapire (1997), "A decision-theoretic generalization of on-line learning and an application to boosting", *Journal of Computer and System Sciences* 55(1). Hedge and AdaBoost. [doi:10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
- Zinkevich (2003), "Online convex programming and generalized infinitesimal gradient ascent", *ICML*. online gradient descent. [aaai.org/Papers/ICML/2003/ICML03-120.pdf](https://www.aaai.org/Papers/ICML/2003/ICML03-120.pdf)
- Cesa-Bianchi, Conconi & Gentile (2004), "On the generalization ability of on-line learning algorithms", *IEEE Transactions on Information Theory* 50(9). online-to-batch. [doi:10.1109/TIT.2004.833339](https://doi.org/10.1109/TIT.2004.833339)
- Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and Games*, Cambridge University Press, ch. 2. [doi:10.1017/CBO9780511546921](https://doi.org/10.1017/CBO9780511546921)
- Hazan (2016), "Introduction to online convex optimization", *Foundations and Trends in Optimization* 2(3–4), ch. 1–3 and 5. [arXiv:1909.05207](https://arxiv.org/abs/1909.05207)

## connections

- ← builds on: projected gradient descent and convexity (ch 06), stable softmax (ch 00), Hoeffding's lemma (ch 05), the PAC framework and its iid assumption (topic 01).
- → used in: AdaBoost, which reweights training examples multiplicatively like Hedge (ch 10 · topic 03), SGD and its step-size schedules (ch 15), multi-armed bandits, the partial-information version of the experts problem (ch 23 · topic 01), and the stability view of generalization (topic 06).
