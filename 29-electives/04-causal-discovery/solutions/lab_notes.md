# causal discovery · lab notes

numbers below are from the reference solution with the lab's seeds. experiments 1 and 2 are fully simulated, so offline and online runs agree. experiment 3 (the project) uses the **real** California housing covariates from the cached `california_housing.csv`; offline (`MLFS_OFFLINE=1`, as in `verify.py`) the loader returns a 1,000-row synthetic stand-in, and the numbers differ (see the end).

## experiment 1: what the data can tell apart

**what people usually predict:** that PC recovers each true graph, or at least gets the direction of the chain from "which variable comes first"; some expect it to fail on the collider, the "odd" structure.

**what happens:**

| world | true DAG | PC found | true CPDAG |
|---|---|---|---|
| chain | A → B, B → C | A – B, B – C | A – B, B – C |
| reversed chain | B → A, C → B | A – B, B – C | A – B, B – C |
| fork | B → A, B → C | A – B, B – C | A – B, B – C |
| collider | A → B, C → B | A → B, C → B | A → B, C → B |

the three non-collider worlds give *identical* output with no directions at all; the collider is the only world PC fully orients.

**why:** the chain, the reversed chain and the fork all imply exactly one conditional independence, $A \perp C \mid B$, and no others: same skeleton, no v-structure, so they're Markov equivalent (on paper 2) and no observational test can separate them. the collider implies the opposite pattern, $A \perp C$ but $A \not\perp C \mid B$, which is the fingerprint the v-structure rule looks for. to separate a chain from a fork you need something else: an intervention (setting $A$ moves $C$ in the chain but not in the fork), time order, or a distributional assumption such as non-Gaussian noise (LiNGAM).

## experiment 2: how much data does PC need?

**what people usually predict:** the error falls steadily towards zero as $n$ grows, mostly through fewer missing edges; by 10,000 samples PC is essentially perfect.

**what happens:** 30 fixed graphs (7.9 edges on average), SHD to the true CPDAG:

| α | n | SHD | missing | extra | wrong orientation |
|---|---|---|---|---|---|
| 0.01 | 100 | 4.20 | 1.87 | 0.03 | 2.30 |
| 0.01 | 300 | 3.63 | 1.00 | 0.13 | 2.50 |
| 0.01 | 1,000 | 3.10 | 0.83 | 0.17 | 2.10 |
| 0.01 | 3,000 | 1.83 | 0.50 | 0.17 | 1.17 |
| 0.01 | 10,000 | **1.60** | 0.37 | 0.03 | 1.20 |
| 0.05 | 100 | 4.17 | 1.57 | 0.20 | 2.40 |
| 0.05 | 300 | 3.60 | 0.93 | 0.40 | 2.27 |
| 0.05 | 1,000 | 3.27 | 0.67 | 0.33 | 2.27 |
| 0.05 | 3,000 | 2.23 | 0.50 | 0.37 | 1.37 |
| 0.05 | 10,000 | 1.67 | 0.30 | 0.07 | 1.30 |
| — | ∞ (perfect test) | 0.00 | 0.00 | 0.00 | 0.00 |

the standard error of each mean SHD is about 0.3–0.5. with a perfect test your code gets all 30 graphs exactly right; with 10,000 samples it's still off by 1.6 pairs per graph on average, and **orientation errors dominate at every sample size**. the diagnosis: **16 of the 30 graphs** have λ < 0.05, i.e. some pair of adjacent variables has a partial correlation below 0.05 given some conditioning set. at $n = 10{,}000$ ($\alpha = 0.01$) those 16 graphs average an SHD of **2.75**; the other 14 average **0.29**. $\alpha = 0.05$ keeps more edges (fewer missing, more extra) and ends up no better overall.

**why:** PC removes an edge as soon as *one* test fails to reject, so a single near-zero partial correlation (a near-cancellation of paths, on paper 4) removes a true edge, and detecting $|\rho| = 0.03$ at $\alpha = 0.01$ needs $n$ in the tens of thousands. a missing edge is worse than one error: it creates a false unshielded triple, the separating set found for it decides whether that triple becomes a false v-structure, and Meek's rules carry the wrong direction further. that's why orientation errors outnumber skeleton errors. random graphs with random-sign weights produce near-cancellations often (more than half of these small graphs have one), which is Uhler et al.'s (2013) point: faithfulness holds almost surely, but *strong* faithfulness, the version finite samples need, fails for a large fraction of graphs. smaller $\alpha$ isn't "fewer mistakes": it trades extra edges for missing ones, and missing edges are the ones that cascade.

## experiment 3 (project): discover, then estimate

**what people usually predict:** PC gets roughly the right graph but leaves some edges at $T$ undirected; the naive regression is too high; "regress on everything" is close to the truth, since it controls for every confounder.

**what happens** (real covariates, each column shuffled independently, 5,000 districts per draw, $\alpha = 0.01$):

- one draw: PC returns the true CPDAG exactly (SHD 0), every edge directed. the only possible parent set of $T$ is $\{\text{inc}, \text{age}\}$, and IDA gives **0.974** against a truth of 0.96. the naive slope of $Y$ on $T$ is 1.141; "$Y$ on $T$ and everything else" is **0.408**.
- 20 fresh draws: SHD 0 in **20 of 20**; $T$'s parent set is identified uniquely every time; IDA averages **0.960 ± 0.018** (sd over draws). naive averages 1.120, "everything else" 0.410.

with the checkbox ticked (the real covariates keep their dependence: the correlation of income and rooms is 0.55, income and age −0.12, age and rooms −0.22):

- PC keeps the edge inc – rooms in every draw (unoriented) but drops the real income–age and age–rooms dependences. that isn't PC "getting the covariates right": in the population, $\rho(\text{inc}, \text{age} \mid \text{rooms}) = 0.002$ (the real covariates happen to look like age ⊥ inc given rooms), and $\rho(\text{age}, \text{rooms} \mid Y) = -0.008$, a near-cancellation: conditioning on the collider $Y$ almost exactly offsets their real correlation of −0.22.
- SHD over pairs involving a simulated variable is 0 in **10 of 20** draws. in the other 10, PC orients $C \to T$, a false v-structure at $T$, so $C$ lands in $T$'s parent set and IDA adjusts for a collider: estimates of 0.66–0.69. the correct draws give 0.93–0.99. over all 20 draws IDA averages **0.816 ± 0.147**. naive averages 1.262, "everything else" 0.415.

**why:**
- **orientation at $T$.** the v-structures $\text{inc} \to T \leftarrow \text{age}$ (the two roots are independent) and $\text{rooms} \to Y \leftarrow T$, $\text{inc} \to Y \leftarrow M$ fix the edges into $T$ and $Y$; Meek's R1 then orients $T \to M$ (from $\text{inc} \to T - M$ with inc and $M$ non-adjacent) and $T \to C$, $Y \to C$ in the same way. with every edge at $T$ directed, $T$'s parents are known and the backdoor set is $\{\text{inc}, \text{age}\}$.
- **the naive slope** is confounded by income: bias $= 0.4\,\mathrm{Cov}(\text{inc}, T)/\mathrm{Var}(T) = 0.4 \times 0.8/2 = 0.16$, so $0.96 + 0.16 = 1.12$, as measured.
- **"everything else" makes two mistakes.** it conditions on the mediator $M$, which removes the indirect effect $0.9 \times 0.4 = 0.36$ and leaves the direct 0.6; and it conditions on $C$, a common effect of $T$ and $Y$, which induces a negative association between them and pulls the estimate down further, to 0.41. it's the worst of the three, and it's what "control for everything you have" produces.
- **with the real dependence**, the failure is a linear near-violation of faithfulness, the same mechanism as experiment 2, not a problem with nonlinearity or heavy tails. the diagnosis: compute population partial correlations from the covariance implied by the real root correlations and the SEM weights. with independent roots, $\rho(\text{inc}, C \mid Y) = 0.085$; with the real root correlations it shrinks to **0.037**, and λ for the simulated edges drops from 0.15 to 0.09. at $n = 5{,}000$ a partial correlation of 0.037 gives $z \approx \sqrt{4996}\,\operatorname{atanh}(0.037) = 2.6$, so the p-value of "inc ⊥ C | Y" sits right around α = 0.01. in about half the draws PC accepts it, records the sepset $(Y,)$ without $T$, and the unshielded triple inc – $T$ – $C$ becomes the false v-structure inc → $T$ ← $C$; $C$ then enters $T$'s parent set. the correspondence is exact in the 20 draws: all 10 bad draws have sepset(inc, $C$) = $(Y,)$ with p between 0.016 and 0.36, and all 10 good draws have sepset $(T, Y)$ because p(inc ⊥ C | Y) was below 0.004. replacing the real covariates by Gaussian roots with the **same** correlation matrix reproduces the failure (12 bad draws of 20), which rules out nonlinearity and heavy tails as the cause. the general lesson is the one the project is built to show: discovery is only as good as its independence tests and its assumptions, and a correlation structure you didn't design can create near-cancellations that a perfectly linear world would still fail on at this $n$. a single wrong orientation at the treatment turns a valid adjustment set into one with a collider in it. report the discovered graph, check its stability over resamples (here a 50/50 split between two answers is a clear warning), and treat the effect estimate as conditional on the graph.

**weights and faithfulness, a note on how this world was built.** a first version used $Y = 0.7M + 0.5T + \dots$ and $C = 0.8T + 0.8Y$. with those weights the population partial correlation of $T$ and $Y$ given $C$ is −0.004: conditioning on the collider almost exactly cancels the direct dependence, and PC deleted the true edge $T - Y$ in every draw. a search over weights for which every adjacent pair keeps $|\rho| > 0.15$ under every conditioning set (λ = 0.15 for the final weights) fixed it. that's experiment 2's lesson showing up while building the project.

**offline stand-in:** with the 1,000-row synthetic covariates, PC returns the true CPDAG in only 3 of 20 draws (mean SHD 1.25), usually misorienting edges at $C$; IDA averages 0.43 ± 0.49 because a wrong parent set (often including $Y$, giving an effect of 0) is read off. the same world at $n = 2{,}000$ with Gaussian roots is right about 55% of the time and at $n = 5{,}000$ about 90%: the project needs its 5,000 districts.
