# structural causal models & do-calculus · lab notes

numbers below are from the reference solution with the lab's seeds. your own code gives exactly the same numbers if your `SCM.sample` draws its noise as the spec says (they're all simulated data, so offline and online runs agree).

## experiment 1: healthier, and harmed

**what people usually predict:** a positive naive difference, but smaller than the true effect is negative: "confounding nudges the estimate a bit".

**what happens:** with 20,000 people per setting:

| $s$ | naive | backdoor | truth |
|---|---|---|---|
| 0.00 | −0.198 | −0.190 | −0.200 |
| 0.25 | +0.020 | −0.194 | −0.200 |
| 0.50 | +0.257 | −0.189 | −0.200 |
| 0.75 | +0.484 | −0.187 | −0.200 |
| 1.00 | **+0.705** | −0.214 | −0.200 |

at full confounding vitamin-takers are 0.7 points *healthier*, while the vitamin costs everyone who takes it 0.2 points. the backdoor estimate stays within sampling noise (±0.02 or so) of the truth at every $s$.

**why:** the naive difference is $\mathbb E[Y \mid X{=}1] - \mathbb E[Y \mid X{=}0] = -0.2 + 1 \cdot \big(P(Z{=}1 \mid X{=}1) - P(Z{=}1 \mid X{=}0)\big)$, because $Y$'s equation is linear and the noise has mean zero whatever $X$ is. by symmetry $P(X{=}1) = 0.5$, so Bayes' rule gives $P(Z{=}1 \mid X{=}1) = 0.5 + 0.45s$ and $P(Z{=}1 \mid X{=}0) = 0.5 - 0.45s$: naive $= -0.2 + 0.9s$, which is $0.70$ at $s = 1$ and crosses zero at $s = 0.22$. the backdoor formula reweights the two strata to their population shares (half and half) whatever $s$ is, so the confounding cancels. correlation and causation can disagree in sign, not just in size, and nothing in the observed joint distribution of $(X, Y)$ alone tells you which world you're in.

## experiment 2: adjust for everything?

**what people usually predict:** $\{Z\}$ is right, and adding more pre-treatment variables ($B$) is harmless or even helpful ("more controls, less bias"). most people expect the mediator to be bad but don't expect the "adjust for everything" set to be the worst.

**what happens:** the true total effect is 1.1 (simulated: 1.099). mean ± sd over 10 seeds of 50,000 draws:

| adjustment set | criterion | estimate | bias |
|---|---|---|---|
| {} | invalid | 1.509 ± 0.014 | +0.41 |
| {Z} | valid | 1.109 ± 0.010 | +0.01 |
| {Z, B} | invalid | 0.867 ± 0.008 | **−0.23** |
| {Z, B, U1} | valid | 1.108 ± 0.015 | +0.01 |
| {Z, M} | invalid | 0.504 ± 0.015 | −0.60 |
| {Z, C} | invalid | 0.240 ± 0.007 | −0.86 |
| {Z, B, M, C} | invalid | **−0.160 ± 0.012** | −1.26 |

your `is_backdoor_adjustment` verdicts line up with the estimates exactly: every valid set is unbiased (within noise), every invalid one is off.

**why:**
- $\{\}$ leaves the fork $X \leftarrow Z \to Y$ open.
- $\{Z, B\}$: $B$ is a collider on $X \leftarrow U_1 \to B \leftarrow U_2 \to Y$. conditioning on it makes $U_1$ and $U_2$ dependent, which opens a backdoor path. being measured before treatment doesn't protect a variable: what matters is its position in the graph. adding $U_1$ blocks the reopened path again, but $U_1$ is unobserved, so in practice $\{Z\}$ (leave $B$ out) is the only working choice.
- $\{Z, M\}$ blocks the mediated path $X \to M \to Y$, leaving the direct effect: 0.504 ≈ 0.5. you lose $0.6 \times 1.0 = 0.6$, exactly the indirect effect.
- $\{Z, C\}$: $C$ is a common effect of $X$ and $Y$. within a stratum of $C$, a unit with $X = 1$ needs a lower $Y$ to land in the same $C$ stratum, which induces a negative association and drags the estimate down.
- combining all of them stacks the biases and flips the sign. "adjust for everything" is not a conservative default; it's a modelling choice that can be badly wrong.

## experiment 3: the front door

**what people usually predict:** that the front-door estimate is about as precise as the naive one, and that with a direct effect it will "partly" pick it up, landing somewhere between 1.4 and 2.4.

**what happens:** with $\delta = 0$ (truth 1.4; simulated 1.396), mean ± sd over 40 seeds:

| $n$ | front-door | naive |
|---|---|---|
| 500 | 1.362 ± 0.136 | 3.179 ± 0.162 |
| 2,000 | 1.409 ± 0.081 | 3.196 ± 0.078 |
| 8,000 | 1.395 ± 0.035 | 3.203 ± 0.045 |
| 32,000 | 1.403 ± 0.018 | 3.202 ± 0.017 |

the naive estimate converges, to 3.2, which is badly wrong; the front-door estimate converges to the truth with a spread that shrinks like $1/\sqrt n$. at these sizes its sd is comparable to the naive one, not much larger. moving the slider to $\delta = 1$ (truth 2.4), the front-door estimate still converges, to **1.4**: it ignores the direct effect entirely (a check with 200,000 draws gives 1.41 against a truth of 2.40, and 1.41 against 3.40 at $\delta = 2$).

**why:** the naive estimate's tiny error bars are the warning of this experiment: precision says nothing about bias, and no amount of data fixes a wrong adjustment. the front-door precision is decent here because $M$ is binary and strongly driven by $X$, so each cell $(m, x')$ has plenty of data; with many values of $m$, or weak $X \to M$, the cells thin out and the front-door estimate gets much noisier. with a direct edge, $\mathbb E[Y \mid m, x'] = 2m + 3\,\mathbb E[U \mid x'] + \delta x'$, and averaging over $P(x')$ gives $2m + 3\,\mathbb E[U] + \delta P(X{=}1)$: the $\delta$ term doesn't depend on $m$, so it cancels in the difference of the two front-door values. the formula measures only the effect that flows *through* $M$, which is the whole effect only if "$X$ affects $Y$ only through $M$" holds. that assumption is about the graph, and you can't test it from $(X, M, Y)$ alone: the data with an unobserved $U$ and a direct edge are compatible with both stories. the front-door criterion trades one untestable assumption (no unobserved confounding) for others (full mediation, no confounding of $X \to M$).
