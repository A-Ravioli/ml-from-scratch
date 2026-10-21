# alignment · lab notes

every experiment in this lab is exact or synthetic by design (a known true objective is what makes the failures measurable), so the notebook gives the same numbers offline; it takes about 27 seconds on an Apple M2 laptop (CPU, one thread). experiment 1 averages over 20 random populations; experiment 3's layouts were also redrawn with seeds 1 and 2 outside the notebook.

## experiment 1: optimising a proxy, harder and harder

**what people predict.** that true quality rises with optimisation as long as the proxy is positively correlated with it, perhaps with diminishing returns; few expect the heavy-tailed proxy to be nearly useless.

**what happens.** means over 20 populations of 5,000 responses:

| proxy | best mean true quality (KL-regularised) | true at $\beta = 0.01$ [10%, 90% of populations] | proxy at $\beta = 0.01$ | best of best-of-$n$ | true, best-of-1024 |
|---|---|---|---|---|---|
| noisy ($u + \mathcal{N}(0,1)$) | 2.48 at KL 8.3 | 2.45 [1.78, 3.20] | 5.24 | 2.15 at $n = 1024$ | 2.15 |
| heavy-tailed ($u + 0.3\,t_2$) | **0.18** at KL 0.4 | **−0.06** [−1.25, 1.22] | **20.40** | **1.45 at $n = 32$** | 0.37 |
| spurious ($0.5u$ + noise) | 1.84 at KL 8.8 | 1.81 [0.84, 3.12] | 4.27 | 1.37 at $n = 1024$ | 1.37 |

optimising $u$ itself reaches 3.7 at the same KL. the noisy proxy keeps paying off (slowly, about two-thirds of the oracle). the spurious proxy pays off about half as well and with a wider spread. the heavy-tailed proxy's mean true quality barely moves above 0 under the KL-regularised policy and ends as a lottery (−1.25 to +1.22 across populations) while its proxy value explodes to 20. best-of-$n$ on the same proxy climbs to 1.45 at $n = 32$ and then falls back to 0.37 at $n = 1024$.

**why.** for the noisy proxy, the top proxy scores are a mix of high true quality and lucky noise; selecting them selects some of both (regressional Goodhart), so the true value improves at a discount. for the heavy-tailed proxy, a handful of responses in every population have proxy errors of 5–30, far larger than any true quality; $e^{r/\beta}$ weights them exponentially, so even mild optimisation hands them most of the mass, and their true quality is just a random draw (extremal Goodhart). best-of-$n$ uses only the proxy's ranking, so until $n$ is large enough to draw the outliers (around $n \sim$ 1/their total reference mass) it behaves like optimising a decent proxy; past that it collapses the same way. the red curves are what you'd see in training: in every panel they go up, and nothing in them tells you which panel you're in.

## experiment 2: when does an agent game its reward?

**what people predict.** that only "greedy" (impatient) agents chase the bonus, or that patient agents, valuing the big goal more, are safe.

**what happens.** for a corridor of 10 cells with the bonus at cell 2 and $G = 10$, the regret map matches the derived threshold exactly: the agent loses all of its true value above the curve and none below it. the threshold peaks at **$B = 1.23$ at $\gamma = 0.85$**. at $\gamma = 0.99$ a bonus of 0.189, 1/53 of the goal, makes the agent circle forever; at $\gamma = 0.3$, a bonus of 0.022.

**why.** the impatient agent compares a bonus two steps away with a goal seven steps away that it discounts by $\gamma^7$ (at 0.3, a factor of $2 \times 10^{-4}$), so almost any bonus wins. the patient agent compares a finite goal with an infinite stream of bonuses worth $B\gamma^2/(1-\gamma^2)$, which diverges as $\gamma \to 1$. in between, both effects are moderate, but no discount makes the respawning bonus safe. "does the reward give more for what I want?" is the wrong test: this reward does give more for reaching the goal than for not reaching it, and it's still gamed; what matters is whether some *other* behaviour gets even more. the sliders under the map rerun one corridor for any $(B, \gamma)$ and print the agent's path: at $B = 1$, $\gamma = 0.9$ (threshold 1.12) it walks to the goal; push $B$ past the threshold and the path turns into `[0, 1, 2, 1, 2, 1, ...]`.

## experiment 3: an eval catches what the training reward hides

**what people predict.** that the proxy agent fails occasionally, on the most extreme layouts; that the shaped agent, being paid the same bonus, games it too.

**what happens.** 200 random corridors, pass = reaches the goal within $2L$ steps:

| agent | bonus < 1 | bonus ≥ 1 | overall |
|---|---|---|---|
| proxy | 1.00 [1.00, 1.00] | **0.34** [0.26, 0.41] | 0.53 [0.46, 0.59] |
| true | 1.00 | 1.00 | 1.00 |
| shaped | 1.00 | 1.00 | 1.00 |

paired difference true − agent: proxy +0.475 [+0.405, +0.540], shaped +0.000. with layouts drawn from seeds 1 and 2, the proxy agent passed 0.52 and 0.565 overall (0.30 and 0.33 with large bonuses, 1.00 and 0.99 with small ones).

**why.** the proxy agent fails in about half the corridors, almost all with bonuses above 1, which is where $B$ exceeds the threshold for that layout's distance to the goal. in every failed corridor it collects more proxy reward than the honest agent would, so a training curve of proxy reward would show it as the *better* agent; only measuring the intended behaviour reveals the problem. the shaped agent is paid $\gamma B$ for entering the cell and charged $B$ for leaving it, so a loop costs $B(1-\gamma^2) > 0$ per cycle; shaping changes the value of every policy by the same amount and can't change which one is best (on paper 3). the paired interval is much tighter than the gap between the two separate intervals suggests, because both agents face the same corridors. this eval had it easy: a known true objective, every case checkable by a one-line rule, and no adversary. real evals of capable models have none of these, which is why designing them is a research field.
