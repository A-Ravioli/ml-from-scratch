# causal discovery

## why this matters

every method in this block so far started from a causal graph you were handed: "stone size affects treatment and recovery", "these 180 covariates are the confounders". in practice someone has to draw that graph, and often nobody knows it. biologists measuring a dozen signalling proteins in thousands of single cells want to know which protein activates which (Sachs et al., 2005, recovered most of a known signalling network this way). an analyst with a table of variables wants to know which ones to adjust for before estimating an effect, and topic 01 showed that adjusting for the wrong one, a collider or a mediator, can flip the answer.

**causal discovery** asks how much of the graph the data can reveal on its own. the surprising answer is "a lot, but not all". conditional independences in the data rule out many graphs; a specific pattern, the collider $A \to B \leftarrow C$, leaves a fingerprint that fixes arrow directions; and a few logical rules then propagate those directions. what's left is a **Markov equivalence class** of graphs the data can't tell apart, summarized as a partly directed graph. this topic builds the classic algorithm that finds it, the **PC algorithm** (Spirtes & Glymour, 1991), and then the block's project: discover a graph from semi-synthetic data built on real covariates, read off what to adjust for, and estimate an effect, with **IDA**, which handles the arrows the data couldn't orient.

## warm-up: predict first

1. data come from either $A \to B \to C$ or $A \leftarrow B \to C$. with unlimited data, can you tell which?
2. data come from $A \to B \leftarrow C$. can you tell that apart from the chain?
3. two variables are strongly correlated and nothing else is measured. which way does the arrow point?
4. $A$ affects $C$ directly and through $B$, and the two paths cancel exactly. what does a test of "$A \perp C$" say, and what does that do to an algorithm that removes edges between independent variables?
5. PC tests independence given subsets of neighbours. with 20 variables, roughly how many conditioning sets would you need if you tried every subset of the *other* variables instead?

## prerequisites check

- [ ] you can decide d-separation and know the three junctions (chain, fork, collider) (→ ch 29 · topic 01)
- [ ] you know that in a multivariate Gaussian, $X_i \perp X_j \mid X_S$ exactly when the partial correlation is zero (→ ch 04 · topic 04)
- [ ] you can compute regression residuals and know the Frisch–Waugh–Lovell idea (→ ch 08 · topic 01, ch 29 · topic 03)
- [ ] you know what a p-value and a significance level are, and why many tests produce false rejections (→ ch 05 · topic 04)
- [ ] you know that any valid backdoor set gives the effect by regression adjustment in a linear model (→ ch 29 · topic 01)

## core ideas

### idea 1: from independences to graphs, and a test for them

topic 01 went from a graph to independences: d-separation in the graph implies conditional independence in the data (the **causal Markov condition**). discovery runs the other way, and needs the converse: every conditional independence in the data comes from d-separation in the graph, not from a coincidence of parameters. that's **faithfulness**. it rules out exact cancellations: if $A \to C$ directly with weight $+1$ and $A \to B \to C$ with weights $1$ and $-1$, then $A$ and $C$ are uncorrelated despite the edge (warm-up 4). such cancellations have probability zero for randomly chosen parameters, so faithfulness is usually considered mild, but near-cancellations are not rare at all, and at finite sample sizes they behave like the real thing. the lab measures how often.

with Markov and faithfulness, discovery reduces to answering many questions of the form "is $X_i \perp X_j \mid X_S$?". for linear-Gaussian data the answer is a **partial correlation**: regress $X_i$ and $X_j$ on $X_S$ and correlate the residuals. equivalently, with $P$ the inverse of the correlation matrix of $(X_i, X_j, X_S)$,

$$\rho_{ij\cdot S} = -\frac{P_{12}}{\sqrt{P_{11}P_{22}}}.$$

to turn an estimate $r$ into a test, use **Fisher's z-transform**: $z = \tfrac12\ln\frac{1 + r}{1 - r} = \operatorname{atanh}(r)$ is approximately normal with variance $1/(n - |S| - 3)$ when the true partial correlation is zero. so the two-sided p-value is

$$p = 2\big(1 - \Phi(\sqrt{n - |S| - 3}\,|z|)\big) = \operatorname{erfc}\!\Big(\sqrt{n - |S| - 3}\,|z|\,\big/\sqrt2\Big),$$

and the algorithm treats "$p > \alpha$" as independence. note the asymmetry: a test can only *fail to reject* independence, and with little data it will often do so for dependent pairs. PC turns those failures into missing edges.

partial correlation detects only *linear* dependence. for non-Gaussian or nonlinear data you'd swap in a kernel or permutation-based conditional independence test; the rest of PC doesn't change.

> **spec.** `partial_correlation(data, i, j, cond=())` returns the partial correlation of columns `i` and `j` of `data` (shape `(n, p)`) given the columns in `cond`, as a float (with `cond = ()`, the ordinary correlation). `fisher_z_test(data, i, j, cond=())` returns the p-value above. clip $r$ to $\pm(1 - 10^{-7})$ before the transform so perfectly correlated columns give $p \approx 0$ instead of an error.

> **▶ build: stage 1.** implement `partial_correlation` and `fisher_z_test`.

### idea 2: the PC skeleton

start from the complete undirected graph and delete the edge $i - j$ whenever you find a set $S$ with $X_i \perp X_j \mid X_S$; record $S$ as the **separating set** of the pair. which sets $S$ do you need to try? a key fact (on paper 1): if $i$ and $j$ are not adjacent in the true DAG, they're d-separated by the parents of one of them, and parents are neighbours. so it's enough to search subsets of the *current neighbours* of $i$ (or of $j$), which shrink as edges disappear.

PC organizes the search by the size $\ell$ of the conditioning set. at level $\ell = 0$ it tests every pair marginally; at $\ell = 1$, every remaining pair given each single neighbour; and so on, until no node has enough neighbours left to form a set of size $\ell$. sparse graphs finish after a few levels, so the number of tests grows polynomially with the number of variables rather than exponentially (warm-up 5: testing every subset of 18 other variables would mean $2^{18} \approx 262{,}000$ sets per pair).

the original PC updates neighbour sets as soon as an edge is removed, so its output depends on the order in which pairs are visited. the **stable** variant (Colombo & Maathuis, 2014) fixes this: at the start of each level, take a snapshot of every node's neighbours and draw conditioning sets only from the snapshot, while still removing edges immediately. with a perfect test any order gives the true skeleton; with finite data, the stable version gives the same answer for any variable ordering, which you want from a scientific tool.

> **spec.** `pc_skeleton(n_vars, ci_test, *, alpha=0.05, max_cond=None)`, where `ci_test(i, j, S)` returns a p-value and `S` is a tuple of ints. start complete. for $\ell = 0, 1, 2, \dots$ (up to `max_cond` if given): snapshot each node's neighbours; for each node $i$ and each $j$ in its snapshot that is still adjacent to $i$, try the subsets $S$ of size $\ell$ of the snapshot of $i$ minus $j$ (in `itertools.combinations` order); at the first $S$ with $p > \alpha$, remove $i - j$ and store `sepsets[(min(i, j), max(i, j))] = S`. stop when, at some level, no adjacent pair has enough other snapshot neighbours to test. return `(adj, sepsets)`: a symmetric boolean `(n_vars, n_vars)` matrix with a false diagonal, and the dict.

> **🔮 predict:** with a perfect independence test and the chain $0 \to 1 \to 2 \to 3$, at which level is the edge $0 - 3$ removed, and with which separating set?

> **▶ build: stage 2.** implement `pc_skeleton`.

### idea 3: orienting edges

the skeleton says which variables are adjacent, not which way the arrows point. the first clue is the collider. take an **unshielded triple** $i - k - j$ with $i$ and $j$ not adjacent. in a chain or a fork through $k$, the middle node $k$ blocks the path, so $k$ must be in *every* set that separates $i$ and $j$. in a collider $i \to k \leftarrow j$, $k$ opens the path, so $k$ is in *no* separating set. so:

> if $k \notin \text{sepset}(i, j)$, orient $i \to k \leftarrow j$ (a **v-structure**).

that's the only place where the data alone directs edges (warm-up 2: yes, the collider differs from the chain). the rest follows by logic: some orientations of the remaining undirected edges would create a new v-structure (which the data would have shown) or a directed cycle (which a DAG can't have). **Meek's rules** (Meek, 1995) apply those constraints until nothing changes:

- **R1** (no new v-structure): if $a \to b - c$ and $a$, $c$ are not adjacent, orient $b \to c$.
- **R2** (no cycle): if $a \to c \to b$ and $a - b$, orient $a \to b$.
- **R3**: if $a - c \to b$, $a - d \to b$, $a - b$, and $c$, $d$ are not adjacent, orient $a \to b$.

(a fourth rule is needed only when you add background knowledge, such as a known time order.) the output, edges directed where every compatible DAG agrees and undirected where they disagree, is a **CPDAG** (completed partially directed acyclic graph).

with finite data the v-structure step can conflict: one triple says $i \to k$, another says $k \to i$. a simple, deterministic convention: orient v-structures one at a time and only turn edges that are still undirected, so the first decision wins.

> **spec.** graphs are `(p, p)` integer matrices: `G[i, j] = 1, G[j, i] = 0` is $i \to j$; both 1 is $i - j$. `apply_meek_rules(G)` returns a new matrix with R1–R3 applied until no rule fires (it never modifies `G`). `orient(adj, sepsets)` starts from the skeleton with every edge undirected; for each node $k$ (in increasing order) and each pair $i < j$ of its neighbours that are not adjacent to each other, if $k$ is not in `sepsets[(i, j)]` (a missing key counts as the empty set), it orients $i \to k$ and $j \to k$, each only if that edge is still undirected; then it returns `apply_meek_rules` of the result. `pc(data, *, alpha=0.05, max_cond=None)` runs `pc_skeleton` with `fisher_z_test` on `data` and returns `orient(...)`.

> **▶ build: stage 3.** implement `apply_meek_rules`, `orient` and `pc`.

### idea 4: identifiability and the CPDAG

when do two DAGs imply exactly the same conditional independences, so that no amount of observational data separates them? Verma & Pearl (1990) answered it: **two DAGs are Markov equivalent if and only if they have the same skeleton and the same v-structures.** the chain, the reversed chain and the fork on three variables share the skeleton $A - B - C$ and have no v-structure, so they're equivalent (warm-up 1: no); the collider has a v-structure and stands alone. on two variables, $X \to Y$ and $Y \to X$ are equivalent (warm-up 3: the data can't say). on paper 2 you'll check the three-variable case directly from the factorizations.

so the CPDAG of the true DAG is the **best possible answer** from observational data with these assumptions, and the right target to score PC against. you can compute it from a DAG with the same tools: keep the skeleton, orient exactly the DAG's v-structures, undirect everything else, and apply Meek's rules (Meek, 1995, proved this gives the CPDAG). to compare graphs, use the **structural Hamming distance**: the number of node pairs whose status (no edge, $i \to j$, $j \to i$, or $i - j$) differs.

two things can push you past the equivalence class, both outside this topic. **interventions**: an experiment that sets one variable breaks the symmetry between equivalent graphs. **stronger assumptions**: with linear models and *non-Gaussian* noise, the direction of every edge is identifiable (LiNGAM; Shimizu et al., 2006), and similar results hold for some nonlinear models. on the other side, the finite-sample version of faithfulness is demanding: Uhler et al. (2013) showed that "no partial correlation is close to zero" (λ-strong faithfulness) fails for a large share of graphs as they grow, which is what the lab's experiment 2 measures.

> **spec.** `dag_to_cpdag(dag)` takes a DAG matrix and returns its CPDAG matrix (skeleton, the DAG's v-structures $i \to k \leftarrow j$ with $i$, $j$ non-adjacent, everything else undirected, then `apply_meek_rules`). `structural_hamming_distance(G1, G2)` returns the number of pairs $i < j$ with `(G1[i, j], G1[j, i]) != (G2[i, j], G2[j, i])`, as an int.

> **🔮 predict:** in the DAG $0 \to 2 \leftarrow 1$, $2 \to 3$, which edges of the CPDAG are directed?

> **▶ build: stage 4.** implement `dag_to_cpdag` and `structural_hamming_distance`.

### idea 5: effects from a partially oriented graph (IDA)

now use the graph. in a DAG, the parents of $x$ always form a valid backdoor set for the effect of $x$ on any $y$ (they block every path into $x$, and none of them is a descendant of $x$). in a linear model the effect is then the coefficient of $x$ in the regression of $y$ on $x$ and its parents.

a CPDAG doesn't fix the parents when some edges at $x$ are undirected. **IDA** ("intervention calculus when the DAG is absent"; Maathuis, Kalisch & Bühlmann, 2009) answers with a *set* of possible effects. take $x$'s definite parents (edges directed into $x$) and its undirected neighbours. a subset $S$ of the undirected neighbours is a **possible extra parent set** if turning $S \to x$ (and every other undirected neighbour into a child) creates no new v-structure at $x$: every node of $S$ must be adjacent to every other node of $S$ and to every definite parent. for each such set, adjust for (definite parents ∪ $S$) and record the effect. if $y$ itself is in the parent set, $x$ can't cause $y$ in that DAG, so that effect is 0. this "local" check is cheap and gives the same set of distinct values as enumerating the whole equivalence class, though the multiplicities can differ.

the output is honest about what the data can't say. if the set is a single number, the effect is identified. if it's $\{0.2, 1.1\}$, the data are compatible with both, and a bound like $\min|\text{effect}|$ is the most you can claim without more assumptions or an experiment.

> **spec.** `possible_parent_sets(cpdag, x)` returns a list of sorted tuples, each the definite parents of `x` plus an admissible subset $S$ of its undirected neighbours (every node of $S$ adjacent to every other node of $S$ and of the definite parents), sorted by length and then lexicographically. `ida_effects(data, x, y, cpdag)` returns a list with one float per parent set, in the same order: `0.0` if `y` is in the set, otherwise the OLS coefficient of column `x` in a regression of column `y` on an intercept, `x`, and the set's columns.

> **▶ build: stage 5.** implement `possible_parent_sets` and `ida_effects`.

## worked example

**PC with a perfect test** on the DAG $0 \to 2 \leftarrow 1$, $2 \to 3$:

- **level 0.** the only marginal independence is $0 \perp 1$ (the path $0 \to 2 \leftarrow 1$ is blocked at the collider). remove $0 - 1$ with sepset $()$.
- **level 1.** $0 \perp 3 \mid 2$ and $1 \perp 3 \mid 2$ (the chains through 2). remove both, sepsets $(2,)$. the skeleton is $0 - 2$, $1 - 2$, $2 - 3$.
- **v-structures.** unshielded triples: $0 - 2 - 1$ (sepset $()$, doesn't contain 2: orient $0 \to 2 \leftarrow 1$); $0 - 2 - 3$ and $1 - 2 - 3$ (sepsets $(2,)$, contain 2: no v-structure).
- **Meek.** R1 on $0 \to 2 - 3$ with 0 and 3 non-adjacent: orient $2 \to 3$.

the CPDAG is fully directed: this DAG is alone in its equivalence class.

**IDA on a chain.** in the CPDAG $0 - 1 - 2$ (from any of the three DAGs of warm-up 1), node 1 has no definite parents and undirected neighbours $\{0, 2\}$. the admissible parent sets are $()$, $(0,)$ and $(2,)$; $(0, 2)$ is not, because 0 and 2 aren't adjacent and $0 \to 1 \leftarrow 2$ would be a v-structure the data didn't show. so `ida_effects(data, 1, 2, cpdag)` returns three numbers: the plain regression slope (DAG $0 \to 1 \to 2$ or $0 \leftarrow 1 \to 2$), the slope adjusted for 0, and $0.0$ (DAG $0 \leftarrow 1 \leftarrow 2$, where 2 is a parent of 1).

**SHD.** if the truth is $0 \to 1$, $1 \to 2$, $2 - 3$ and an estimate has $1 \to 0$, $1 \to 2$, $2 \to 3$, the SHD is 2: one reversed edge, one undirected edge that became directed. your code must reproduce all three examples.

## on paper

1. **why neighbours suffice.** let $i$ and $j$ be non-adjacent in a DAG, with $j$ not a descendant of $i$. show that $\mathrm{pa}(i)$ d-separates $i$ and $j$. why does this mean PC only ever needs subsets of neighbours?

<details><summary>solution</summary>

take any path from $i$ to $j$. $j \notin \mathrm{pa}(i)$, since they aren't adjacent. **case 1:** the path starts $i \leftarrow p$. then $p \in \mathrm{pa}(i)$ is conditioned on, and it's not a collider on the path (the edge $p \to i$ points away from $p$), so the path is blocked. **case 2:** the path starts $i \to c$. it can't be directed all the way to $j$, because $j$ isn't a descendant of $i$. so there is a first node $w$ where two arrowheads meet, $\dots \to w \leftarrow \dots$: a collider, and a descendant of $i$. for the path to be open, $w$ or one of its descendants would have to be in $\mathrm{pa}(i)$, making a parent of $i$ a descendant of $i$: a directed cycle. so the path is blocked. in a DAG two non-adjacent nodes can't each be a descendant of the other, so $\mathrm{pa}(i)$ or $\mathrm{pa}(j)$ separates them, and parents are neighbours: searching subsets of the current neighbours of $i$ and of $j$ is enough.

</details>

2. **Markov equivalence by factorization.** show that $A \to B \to C$, $A \leftarrow B \leftarrow C$ and $A \leftarrow B \to C$ can represent exactly the same joint distributions, and that $A \to B \leftarrow C$ cannot represent a distribution in which $A \not\perp C$.

<details><summary>solution</summary>

the chain gives $P(a)P(b \mid a)P(c \mid b)$. by Bayes' rule $P(a)P(b \mid a) = P(b)P(a \mid b)$, so it equals $P(b)P(a \mid b)P(c \mid b)$, the fork's factorization; and $P(b)P(c \mid b) = P(c)P(b \mid c)$ turns that into $P(c)P(b \mid c)P(a \mid b)$, the reversed chain. the collider's factorization $P(a)P(c)P(b \mid a, c)$ sums over $b$ to $P(a)P(c)$: $A \perp C$ always, while the other three graphs generally have $A \not\perp C$.

</details>

3. **why R1.** in $a \to b - c$ with $a$ and $c$ non-adjacent, suppose the edge $b - c$ were oriented $c \to b$. what would the data have shown, and why does that force $b \to c$?

<details><summary>solution</summary>

$a \to b \leftarrow c$ with $a$, $c$ non-adjacent is a v-structure, so $b$ would be in no separating set of $(a, c)$ and PC would already have oriented it as one. since it didn't (so $b$ is in the separating set), $c \to b$ is impossible, and $b \to c$ is the only orientation left.

</details>

4. **a faithfulness violation.** in the linear SEM $A = \varepsilon_A$, $B = A + \varepsilon_B$, $C = A - B + \varepsilon_C$ with independent standard normal noises, compute $\mathrm{Cov}(A, C)$. what skeleton does PC (with a perfect test) return, and which of its conclusions is wrong?

<details><summary>solution</summary>

$C = A - (A + \varepsilon_B) + \varepsilon_C = -\varepsilon_B + \varepsilon_C$, so $\mathrm{Cov}(A, C) = 0$ even though $A \to C$ is an edge. at level 0 PC removes $A - C$ with sepset $()$. the remaining triple $A - B - C$ is unshielded with $B \notin ()$, so PC orients the v-structure $A \to B \leftarrow C$: it gets the $A - C$ edge wrong *and* reverses $B \to C$. one exact cancellation produces two errors, and a near-cancellation does the same at finite $n$.

</details>

## experiment

`lab.py` runs your PC and IDA code:

1. four three-variable worlds (chain, reversed chain, fork, collider): what PC can and can't tell apart.
2. random 8-variable graphs at sample sizes from 100 to 10,000, against a perfect test: how fast PC converges, which errors remain, and why.
3. **the block's project.** a semi-synthetic world built on real California housing covariates, with a simulated treatment, mediator, outcome and a common effect. discover the graph, estimate the effect with IDA, and compare with two regressions people actually run. then repeat with the real covariates' own dependence left in.

## check your understanding

1. (the predict callout in idea 2.) when is $0 - 3$ removed in the chain $0 \to 1 \to 2 \to 3$?

<details><summary>answer</summary>

at level 1. nothing is marginally independent in a chain, so level 0 removes nothing. at level 1, node 0's snapshot neighbours are $\{1, 2, 3\}$, and the subsets of size 1 of $\{1, 2\}$ are tried in order: $(1,)$ separates 0 and 3, so the sepset is $(1,)$. ($(2,)$ would too.)

</details>

2. (the predict callout in idea 4.) in $0 \to 2 \leftarrow 1$, $2 \to 3$, which CPDAG edges are directed?

<details><summary>answer</summary>

all three: the v-structure fixes $0 \to 2 \leftarrow 1$, and R1 then forces $2 \to 3$ (the worked example). a single collider can orient a whole downstream chain.

</details>

3. why is "$p > \alpha$" a risky way to decide independence, and how do errors propagate in PC?

<details><summary>answer</summary>

failing to reject isn't evidence of independence; with little data or weak edges, real edges get removed. a wrongly removed edge creates false unshielded triples, which can create false v-structures, which Meek's rules then propagate. a wrongly kept edge can hide a true v-structure (the triple becomes shielded).

</details>

4. two DAGs have the same skeleton but different v-structures. can observational data distinguish them?

<details><summary>answer</summary>

yes, in principle: by Verma & Pearl's theorem they're not Markov equivalent, so some conditional independence differs (whether $k$ separates $i$ and $j$ in the triple).

</details>

5. why are the parents of $x$ always a valid adjustment set, and why doesn't that solve the problem when the graph is unknown?

<details><summary>answer</summary>

they block every backdoor path (each starts with an edge into $x$, through a parent that is a conditioned-on non-collider) and contain no descendant of $x$. but the parents are exactly what a CPDAG may leave ambiguous, which is why IDA returns one effect per possible parent set.

</details>

6. in the project, why does "regress $Y$ on $T$ and every other variable" fail even though it includes all the confounders?

<details><summary>answer</summary>

"everything" includes the mediator $M$, which removes the indirect part of the effect, and the common effect $C$ of $T$ and $Y$, which opens a collider path and biases the rest. a discovered graph tells you which variables are descendants of $T$, which a regression can't.

</details>

## common bugs

- **using $n - 3$ instead of $n - |S| - 3$.** symptom: p-values slightly off the checks, mostly with large conditioning sets.
- **taking conditioning sets from the live adjacency.** symptom: results change when you relabel the variables. stable PC draws them from the level's snapshot.
- **forgetting to store or symmetrize sepsets.** symptom: every unshielded triple becomes a v-structure. key separating sets by `(min, max)`.
- **orienting a v-structure over an already directed edge.** symptom: edges that point both ways or vanish. only turn edges that are still undirected.
- **applying Meek's rules once.** symptom: some orientations missing. one rule can enable another; loop until nothing changes.
- **adjusting for every neighbour in IDA.** symptom: estimates near zero or biased. children of $x$ are not adjustment variables; only the possible parent sets are.

## going deeper

- Spirtes, Glymour & Scheines (2000), *Causation, Prediction, and Search*, 2nd ed., MIT Press, chapters 3 and 5 (faithfulness, the PC algorithm). [doi:10.7551/mitpress/1754.001.0001](https://doi.org/10.7551/mitpress/1754.001.0001)
- Spirtes & Glymour (1991), "An algorithm for fast recovery of sparse causal graphs", *Social Science Computer Review* 9(1). the PC algorithm. [doi:10.1177/089443939100900106](https://doi.org/10.1177/089443939100900106)
- Verma & Pearl (1990), "Equivalence and synthesis of causal models", *UAI*. the skeleton + v-structures characterization.
- Meek (1995), "Causal inference and causal explanation with background knowledge", *UAI*. the orientation rules and their completeness.
- Colombo & Maathuis (2014), "Order-independent constraint-based causal structure learning", *JMLR* 15. stable PC. [jmlr.org/papers/v15/colombo14a.html](https://jmlr.org/papers/v15/colombo14a.html)
- Kalisch & Bühlmann (2007), "Estimating high-dimensional directed acyclic graphs with the PC-algorithm", *JMLR* 8. consistency of PC with Fisher-z tests. [jmlr.org/papers/v8/kalisch07a.html](https://jmlr.org/papers/v8/kalisch07a.html)
- Maathuis, Kalisch & Bühlmann (2009), "Estimating high-dimensional intervention effects from observational data", *Annals of Statistics* 37(6A). IDA. [doi:10.1214/09-AOS685](https://doi.org/10.1214/09-AOS685)
- Uhler, Raskutti, Bühlmann & Yu (2013), "Geometry of the faithfulness assumption in causal inference", *Annals of Statistics* 41(2). [doi:10.1214/12-AOS1080](https://doi.org/10.1214/12-AOS1080)
- Shimizu, Hoyer, Hyvärinen & Kerminen (2006), "A linear non-Gaussian acyclic model for causal discovery", *JMLR* 7. LiNGAM. [jmlr.org/papers/v7/shimizu06a.html](https://jmlr.org/papers/v7/shimizu06a.html)
- Sachs, Perez, Pe'er, Lauffenburger & Nolan (2005), "Causal protein-signaling networks derived from multiparameter single-cell data", *Science* 308(5721). [doi:10.1126/science.1105809](https://doi.org/10.1126/science.1105809)
- Peters, Janzing & Schölkopf (2017), *Elements of Causal Inference*, MIT Press (open access), chapters 4 and 7.

## connections

- ← builds on: d-separation, backdoor adjustment and colliders (topic 01), partial correlation and the multivariate Gaussian (ch 04 · topic 04), regression adjustment (ch 08 · topic 01), hypothesis tests and multiple testing (ch 05 · topic 04).
- → used in: the block's project (this lab), which closes the loop of the block: discover a graph, choose an adjustment set from it, and estimate an effect, which you could hand to topic 02's AIPW or topic 03's DML when the relationships aren't linear.
