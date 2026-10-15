# PAC learning · lab notes

the numbers below are from the reference solution at the shipped seeds. the "measured" column comes from a simulation (2,000 samples per sample size, a grid about 8% apart), so expect a few percent of scatter if your code draws random numbers in a different order.

## experiment 1: theory against simulation, no noise

**what people usually predict:** that a theoretical bound is loose by an order of magnitude or more even in this toy problem, and that the finite-class bound for $2^{32}$ thresholds is astronomically large ("$2^{32}$ is four billion hypotheses").

**what happens:** with $\delta = 0.05$:

| ε | measured | exact | ln(1/δ)/ε | finite, 1,000 | finite, $2^{32}$ |
|---|---|---|---|---|---|
| 0.2 | 13 | 14 | 15.0 | 50 | 126 |
| 0.1 | 31 | 29 | 30.0 | 100 | 252 |
| 0.05 | 61 | 59 | 59.9 | 199 | 504 |
| 0.02 | 156 | 149 | 149.8 | 496 | 1259 |
| 0.01 | 309 | 299 | 299.6 | 991 | 2518 |

the measured sample complexity sits on the exact curve (within the grid spacing and Monte Carlo noise), and $\ln(1/\delta)/\varepsilon$ is within 7.1% of exact at every $\varepsilon$ (the largest gap is at $\varepsilon = 0.2$: 15.0 against 14). the finite-class bound for the 1,000-point grid is about 3.4× the exact value, and for all $2^{32}$ float32 thresholds about 8.5× (9.0× at $\varepsilon = 0.2$, 8.4× at 0.01). the ratio barely moves with $\varepsilon$.

**why:** the threshold argument is tight: failure happens *exactly* when an interval of mass $\varepsilon$ is empty, so the only slack is $1 - \varepsilon \le e^{-\varepsilon}$, a factor $-\ln(1 - \varepsilon)/\varepsilon \approx 1 + \varepsilon/2$ (on paper 3). the finite-class bound pays $\ln|\mathcal H|/\varepsilon$ on top of $\ln(1/\delta)/\varepsilon$: for $2^{32}$ thresholds that's $22.2/\varepsilon$ against $3.0/\varepsilon$, so the ratio is about $(22.2 + 3.0)/3.0 \approx 8.4$ for every $\varepsilon$ (both terms scale as $1/\varepsilon$). four billion hypotheses cost only a factor of 8 because they enter through a logarithm. still, the union bound counts hypotheses that are *never distinguishable* on a sample (two thresholds between the same pair of points behave identically), and that's the slack topic 02 removes: what matters is the number of distinct labelings of a sample, not the number of hypotheses.

## experiment 2: label noise

**what people usually predict:** that label noise makes the problem dramatically harder, following the agnostic bound: quadrupling the sample size each time $\varepsilon$ halves, and thousands of examples for $\varepsilon = 0.02$.

**what happens:** with $\eta = 0.1$:

| ε | measured | agnostic bound (1,000 thresholds) | bound / measured |
|---|---|---|---|
| 0.2 | 14 | 133 | 10× |
| 0.1 | 32 | 530 | 17× |
| 0.05 | 70 | 2,120 | 30× |
| 0.02 | 165 | 13,246 | 80× |

the measured sample sizes are only 5–15% above the *noiseless* ones from experiment 1, and the fitted log-log slope is **−1.07**: the rate is $1/\varepsilon$, not $1/\varepsilon^2$. the gap to the bound grows like $1/\varepsilon$, reaching 80× at $\varepsilon = 0.02$. and the comparison flatters the bound: the agnostic number at $\varepsilon$ guarantees uniform convergence to within $\varepsilon$, hence ERM excess risk at most $2\varepsilon$, while the simulation counts a failure at excess risk $\varepsilon$. a bound that guarantees excess $\varepsilon$ needs uniform convergence to $\varepsilon/2$, four times the sample size, so at matched tolerance the gap is up to 4× larger still (about 320× at $\varepsilon = 0.02$). on the slider (same seeds), $\eta = 0.2$ gives 19, 44, 96, 225; $\eta = 0.3$ gives 70, 131, 331 for $\varepsilon = 0.1, 0.05, 0.02$; $\eta = 0.4$ gives 243 and 775 for $\varepsilon = 0.05, 0.02$. at large $\eta$ and large $\varepsilon$ the answer collapses to the smallest grid value (2), because the excess risk $(1 - 2\eta)|t - t^\star|$ can never exceed $(1 - 2\eta)/2$: with $\eta = 0.3$ even the worst threshold is only 0.2 worse than the best.

**why:** the agnostic bound is a worst case over every distribution with this hypothesis class, and the $1/\varepsilon^2$ rate is forced by distributions in which some hypothesis has error only $\varepsilon$ above the best *while disagreeing with it on a large region*: then telling them apart means estimating two error rates near some $p > 0$ to precision $\varepsilon$. thresholds with label noise are nothing like that. moving $t$ by $\Delta$ away from $t^\star$ changes the risk by $(1 - 2\eta)\Delta$, linearly, and the two hypotheses disagree only on a region of mass $\Delta$. near $t^\star$ each example in the disagreement region shifts the empirical error difference by $\pm1$ with drift only $1 - 2\eta$, so ERM locates $t^\star$ to within $O(1/((1 - 2\eta)^2m))$ (a random walk with drift $1 - 2\eta$ and unit steps), and since the excess risk is $(1 - 2\eta)|t - t^\star|$, the excess risk is $O(1/((1 - 2\eta)m))$: the realizable rate, with a constant that grows as $\eta \to 1/2$. this is the **Massart noise condition** (noise bounded away from 1/2), under which fast $1/\varepsilon$ rates hold (Massart & Nédélec, 2006). when $\eta \to 0.5$ the labels carry no information near the boundary, $1 - 2\eta \to 0$, and the sample complexity blows up. the lesson for the rest of the chapter: a valid bound can be off by orders of magnitude *for your distribution* without being wrong, and measuring is the only way to know.
