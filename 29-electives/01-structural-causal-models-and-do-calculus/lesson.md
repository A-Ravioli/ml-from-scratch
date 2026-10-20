# structural causal models & do-calculus

## why this matters

in 1986 a urology team compared two treatments for kidney stones on 700 patients (Charig et al., 1986). open surgery (A) worked for 273 of its 350 patients (78%). the newer keyhole procedure (B) worked for 289 of 350 (83%). B looks better. now split the patients by stone size. for small stones A succeeds 93% of the time and B 87%. for large stones A succeeds 73% and B 69%. **A is better for small stones and better for large stones, but worse overall.**

no arithmetic is wrong here. doctors gave the harder cases (large stones) to open surgery more often, and large stones fail more often whatever you do. the overall comparison mixes "which treatment" with "how hard the case was". so which number answers the question a patient actually asks: *if I get treatment A, how likely am I to recover?*

statistics alone can't settle it. the same table, with different labels, can call for the opposite answer: if the third variable were something the treatment *caused* (say, blood pressure after surgery), splitting on it would be the mistake. what decides it is the causal story behind the data: what causes what. this topic turns that story into mathematics you can compute with: a **structural causal model** (SCM) says how each variable is produced; the **do-operator** says what an intervention means; **d-separation** reads independences off a graph; and the **backdoor** and **front-door** formulas turn observational data into answers about interventions, when that's possible at all. it's the foundation of every other topic in this elective.

## warm-up: predict first

1. in a simulated world, people who take vitamins are healthier on average. can vitamins still be *harmful* in that world?
2. you have a variable measured *before* treatment. can adjusting for it ever make a causal estimate worse?
3. $X$ and $Y$ are independent coin flips, and $C = X + Y$. if you only look at cases with $C = 1$, are $X$ and $Y$ still independent?
4. smoking and lung cancer share an unmeasured genetic cause. you can't measure the gene. can you still estimate the causal effect of smoking from observational data?
5. is $P(Y \mid X = x)$ the same as $P(Y \mid \mathrm{do}(X = x))$ when $X$ has no parents in the graph?

## prerequisites check

- [ ] you can apply the law of total probability, $P(y) = \sum_z P(y \mid z)P(z)$, and Bayes' rule (→ ch 04 · topic 01)
- [ ] you know what conditional independence $X \perp Y \mid Z$ means (→ ch 04 · topic 01, ch 12 · topic 01)
- [ ] you know that a regression slope picks up correlation, not causation (→ ch 08 · topic 01)
- [ ] you can write a depth-first search over a graph stored as a dict of lists (→ ch 00 · topic 01)
- [ ] you pass randomness around as a `Generator` with `as_rng` (→ ch 00 · topic 05)

## core ideas

### idea 1: causal graphs

draw an arrow $A \to B$ when $A$ is one of the things $B$ *listens to*: if you could reach in and change $A$, $B$ might change. collect these arrows for a set of variables and you get a directed graph. we require it to have no directed cycles (nothing causes itself, even indirectly), which makes it a **DAG**. the kidney-stone story is

$$S \to T,\qquad S \to R,\qquad T \to R:$$

stone size $S$ influences which treatment $T$ a doctor picks, and both influence recovery $R$.

the vocabulary you need: the **parents** $\mathrm{pa}(V)$ are the nodes with an arrow into $V$. $A$ is an **ancestor** of $B$ (and $B$ a **descendant** of $A$) if there's a directed path $A \to \dots \to B$. a **topological order** lists every node after all of its parents; every DAG has at least one, and a graph has one exactly when it has no directed cycle. Kahn's algorithm finds one: repeatedly take any node whose parents have all been listed. if at some point no node qualifies but nodes remain, those nodes contain a cycle.

in code a graph is a dict mapping each node to the list of its parents, e.g. `{"S": [], "T": ["S"], "R": ["S", "T"]}`. storing parents rather than children is deliberate: an SCM computes each node *from its parents*, so that's the direction you'll look up most.

the graph is an assumption, not a fact you read off the data. the arrows you draw are claims about the world; the ones you *don't* draw are stronger claims still (no effect at all). everything that follows is "if this graph is right, then…". topic 04 asks how much of the graph data can reveal.

> **spec.** `topological_order(parents)` returns a list containing every node once, each after all its parents (any valid order). it raises `ValueError` if the graph has a directed cycle (including a self-loop) or lists a parent that isn't a key. `ancestors(parents, node)` and `descendants(parents, node)` return sets that never include `node` itself.

> **▶ build: stage 1.** implement `topological_order`, `ancestors` and `descendants`.

### idea 2: structural causal models and the do-operator

a graph says *who* listens to whom; a structural causal model says *how*. each node $V$ gets a **structural equation**

$$V := f_V\big(\mathrm{pa}(V),\, U_V\big),$$

where $U_V$ is an exogenous noise term: everything else that affects $V$ that we don't model. the noises are mutually independent. the "$:=$" is an assignment, not an equation: it says $V$ is *computed* from its parents, not that the two sides are interchangeable. an SCM is a little program, and running it in topological order with fresh noise gives one sample of the **observational distribution**, the world as it is.

an **intervention** $\mathrm{do}(X = x)$ changes the program: delete $X$'s equation and replace it with $X := x$, leave every other equation alone, and run it again. in the graph, that's **graph surgery**: cut every arrow *into* $X$, because $X$ no longer listens to its parents. the result is the **interventional distribution** $P(Y \mid \mathrm{do}(X = x))$, the world in which everyone is assigned $X = x$, as in an ideal randomized experiment.

compare it with conditioning. $P(Y \mid X = x)$ looks at the units that *happened* to have $X = x$. in the kidney data those are disproportionately large-stone patients, so their outcomes carry the stone-size effect along with the treatment's. $P(Y \mid \mathrm{do}(X = x))$ gives *everyone* $X = x$, so the stone sizes keep their population distribution. the two agree only when nothing else is pushing $X$ and $Y$ around together, for instance when $X$ has no parents (warm-up 5: yes).

there's a one-line formula for the interventional distribution of a model whose variables are all observed. the SCM implies the observational distribution factorizes along the graph, $P(v_1, \dots, v_n) = \prod_i P(v_i \mid \mathrm{pa}_i)$. an intervention removes exactly one factor, which gives the **truncated factorization**

$$P(v \mid \mathrm{do}(X = x)) = \prod_{i:\, V_i \neq X} P(v_i \mid \mathrm{pa}_i)\,\Big|_{X = x}.$$

this tells you that, in principle, interventional questions about a fully observed model reduce to conditional probabilities. the hard part, which ideas 3 and 4 handle, is that real problems have **unobserved** variables.

for your simulator, fixing *how* the noise is drawn makes every run reproducible and makes interventions comparable: an intervention changes the equations but not the noise, so `sample(n, rng=s)` and `sample(n, do=..., rng=s)` describe the *same* units in two worlds. that's what lets you compute an exact causal effect in simulation by subtracting two runs.

> **spec.** `SCM(parents, mechanisms)`: `mechanisms[v]` is a function `f(pa, u)` where `pa` is a dict containing exactly `v`'s parents' value arrays and `u` is `v`'s noise, an array of shape `(n,)`. store `self.parents`, `self.nodes = list(parents)` (dict order), `self.mechanisms`, and `self.order = topological_order(parents)`. `sample(n, *, do=None, rng=None)` draws **all** the noise first as `as_rng(rng).normal(size=(n, len(nodes)))`, column `k` belonging to `nodes[k]` (even for intervened nodes), evaluates nodes in `self.order`, and returns a dict `node -> float array (n,)` (broadcast scalar outputs). `do` maps node names to a scalar or an array `(n,)`; an unknown node raises `ValueError`. `intervene_graph(parents, nodes)` returns a new dict with every edge into `nodes` (a name or a collection) removed, without modifying its input.

> **🔮 predict:** in the SCM $Z := U_Z$, $X := Z + U_X$, $Y := X + 3Z + U_Y$, the causal effect of $X$ on $Y$ is 1. what slope will a regression of $Y$ on $X$ find?

> **▶ build: stage 2.** implement `SCM` and `intervene_graph`.

### idea 3: d-separation and the backdoor criterion

which independences does a graph imply? every path between two nodes is a possible channel for association, and each node along a path is one of three junctions:

| junction | shape | open when the middle node is… |
|---|---|---|
| chain | $A \to B \to C$ | *not* conditioned on |
| fork | $A \leftarrow B \to C$ | *not* conditioned on |
| collider | $A \to B \leftarrow C$ | conditioned on, **or any descendant of $B$ is** |

chains and forks behave the way intuition says: a common cause makes its effects correlated, and knowing the common cause screens them off. colliders are the odd ones. two independent causes of a common effect are independent, until you condition on the effect: among patients admitted to hospital (admission caused by disease A *or* disease B), having A makes B less likely, because something had to explain the admission (warm-up 3: no, $X$ and $Y$ become perfectly anti-correlated given $C = 1$). conditioning on a descendant of the collider leaks the same information. this is **collider bias**, also called selection bias, and it's behind a lot of spurious findings.

a path is **blocked** by a set $Z$ if some junction on it is closed. $X$ and $Y$ are **d-separated** by $Z$, written $X \perp_G Y \mid Z$, if *every* path between them is blocked. the central theorem (Verma & Pearl, 1988; Lauritzen et al., 1990): if $X$ and $Y$ are d-separated by $Z$ in the graph of an SCM, then $X \perp Y \mid Z$ in every distribution the SCM can produce.

checking every path is exponential. an equivalent test is simpler to code (Lauritzen et al., 1990): (1) keep only $X \cup Y \cup Z$ and their ancestors; (2) **moralize**: connect every pair of parents of a common child, then drop all arrow directions; (3) delete $Z$. then $X$ and $Y$ are d-separated by $Z$ exactly when they're disconnected in what's left. moralizing is what handles colliders: a collider's parents get a direct link that only survives if the collider (or a descendant, kept as an ancestor of $Z$) is in the ancestral set.

now the payoff. a **backdoor path** from $X$ to $Y$ is a path that starts with an arrow *into* $X$, like $T \leftarrow S \to R$. these paths carry association that isn't causation. a set $Z$ satisfies the **backdoor criterion** relative to $(X, Y)$ if

1. no node in $Z$ is a descendant of $X$, and
2. $Z$ blocks every backdoor path from $X$ to $Y$.

condition 2 is d-separation in the graph with $X$'s *outgoing* edges removed (all that's left between $X$ and $Y$ are the backdoor paths). condition 1 stops you from adjusting for mediators, which would remove part of the effect you want, and for colliders downstream of $X$, which would open new paths.

watch out for one classic trap, **M-bias**: $X \leftarrow U_1 \to M \leftarrow U_2 \to Y$. $M$ is measured before treatment and is correlated with both $X$ and $Y$, so it looks like a confounder. but $M$ is a collider on that path: the empty set is a valid adjustment, and adjusting for $M$ *opens* a backdoor path (warm-up 2: yes).

> **spec.** `d_separated(parents, xs, ys, zs=())` takes node sets (or single names) and returns a bool. `is_backdoor_adjustment(parents, x, y, zs)` returns `True` exactly when `zs` contains no descendant of `x` (and neither `x` nor `y`) and d-separates `x` from `y` once `x`'s outgoing edges are deleted.

> **▶ build: stage 3.** implement `d_separated` and `is_backdoor_adjustment`.

### idea 4: adjustment formulas and the do-calculus

if $Z$ satisfies the backdoor criterion, the effect is **identified** from observational data by the **backdoor adjustment formula**

$$P(y \mid \mathrm{do}(x)) = \sum_z P(y \mid x, z)\,P(z),\qquad \mathbb E[Y \mid \mathrm{do}(x)] = \sum_z \mathbb E[Y \mid x, z]\,P(z).$$

read it as: within each stratum $z$ the treated and untreated are comparable, so use the stratum's conditional mean; then average the strata with their *population* weights $P(z)$, not with $P(z \mid x)$, which is what the naive conditional mean $\mathbb E[Y \mid x] = \sum_z\mathbb E[Y \mid x, z]P(z \mid x)$ uses. the difference between those two weightings is the whole of confounding. the formula needs **positivity**: every stratum must contain units with $X = x$, otherwise $\mathbb E[Y \mid x, z]$ isn't defined. topic 02 rewrites the same formula as reweighting.

sometimes no backdoor set exists, because the confounder is unobserved. Pearl's example: smoking $X$, tar in the lungs $M$, cancer $Y$, and an unobserved genotype $U$ that affects smoking and cancer. if smoking affects cancer *only* through tar, if nothing unobserved affects both smoking and tar, and if the backdoor paths from tar to cancer are blocked by smoking, the **front-door formula** still identifies the effect:

$$P(y \mid \mathrm{do}(x)) = \sum_m P(m \mid x)\sum_{x'} P(y \mid m, x')\,P(x').$$

the logic is two backdoor adjustments chained together. the effect of $X$ on $M$ has no open backdoor path (the only one, $X \leftarrow U \to Y \leftarrow M$, is blocked at the collider $Y$), so $P(m \mid \mathrm{do}(x)) = P(m \mid x)$. the effect of $M$ on $Y$ has the backdoor path $M \leftarrow X \leftarrow U \to Y$, which $X$ blocks, so $P(y \mid \mathrm{do}(m)) = \sum_{x'}P(y \mid m, x')P(x')$. chain them: smoking moves tar, and tar moves cancer (warm-up 4: yes, under these assumptions).

backdoor and front-door are special cases of a general calculus. write $G_{\overline X}$ for the graph with arrows into $X$ removed and $G_{\underline X}$ for arrows out of $X$ removed. the three rules of the **do-calculus** (Pearl, 1995) are:

1. **insert/delete observations:** $P(y \mid \mathrm{do}(x), z, w) = P(y \mid \mathrm{do}(x), w)$ if $Y \perp Z \mid X, W$ in $G_{\overline X}$.
2. **exchange action and observation:** $P(y \mid \mathrm{do}(x), \mathrm{do}(z), w) = P(y \mid \mathrm{do}(x), z, w)$ if $Y \perp Z \mid X, W$ in $G_{\overline X\underline Z}$.
3. **insert/delete actions:** $P(y \mid \mathrm{do}(x), \mathrm{do}(z), w) = P(y \mid \mathrm{do}(x), w)$ if $Y \perp Z \mid X, W$ in $G_{\overline{X}\,\overline{Z(W)}}$, where $Z(W)$ is the set of $Z$-nodes that aren't ancestors of any $W$-node in $G_{\overline X}$.

an effect is identified when the rules can rewrite $P(y \mid \mathrm{do}(x))$ into an expression with no $\mathrm{do}$ left. the rules are **complete** (Shpitser & Pearl, 2006; Huang & Valtorta, 2006): if they can't remove every $\mathrm{do}$, no method can, and the effect is not identified without further assumptions. you'll use rule 2 to derive the backdoor formula, and all three for the front-door one, on paper.

> **spec.** `backdoor_adjust(data, x, y, zs, x_value)`: `data` is a dict of equal-length arrays, `zs` a list of names (possibly empty) of **discrete** variables. return the float $\sum_z \hat P(z)\,\hat{\mathbb E}[y \mid x = x_\text{value}, z]$ over the observed combinations $z$ of the `zs` columns (with `zs = []`, the plain mean of `y` where `x == x_value`). raise `ValueError` if some stratum has no row with `x == x_value`. `frontdoor_adjust(data, x, y, m, x_value)`: discrete `x` and `m`; return $\sum_m \hat P(m \mid x_\text{value})\sum_{x'}\hat P(x')\,\hat{\mathbb E}[y \mid m, x']$, with the outer sum over the values of `m` seen among rows with `x == x_value`; raise `ValueError` when a needed cell $(m, x')$ is empty.

> **🔮 predict:** in the front-door table of the worked example, the naive difference $\mathbb E[y \mid x{=}1] - \mathbb E[y \mid x{=}0]$ is 0.35. will the front-door estimate of the effect be larger, smaller, or the same?

> **▶ build: stage 4.** implement `backdoor_adjust` and `frontdoor_adjust`.

## worked example

**backdoor, kidney stones.** with $T = 1$ for open surgery, $S = 1$ for small stones and $R = 1$ for success, the 700 patients are

| | small stones | large stones | all |
|---|---|---|---|
| A (open) | 81/87 = 0.931 | 192/263 = 0.730 | 273/350 = 0.780 |
| B (keyhole) | 234/270 = 0.867 | 55/80 = 0.688 | 289/350 = 0.826 |

stone size satisfies the backdoor criterion in $S \to T$, $S \to R$, $T \to R$. the stratum weights are $P(S{=}1) = 357/700 = 0.51$ and $P(S{=}0) = 343/700 = 0.49$, so

$$\mathbb E[R \mid \mathrm{do}(A)] = 0.51 \times 0.9310 + 0.49 \times 0.7300 = 0.8325,$$
$$\mathbb E[R \mid \mathrm{do}(B)] = 0.51 \times 0.8667 + 0.49 \times 0.6875 = 0.7789.$$

A is better by 5.4 points, while the naive comparison said B was better by 4.6. the naive mean for A weights its strata by $P(S \mid A) = (87, 263)/350$, three quarters large stones; the adjustment gives every treatment the same 51/49 mix.

**front-door.** 400 units with binary $x$, $m$, $y$:

| | $m = 0$ | $m = 1$ |
|---|---|---|
| $x = 0$ (200 units) | 160 units, $\bar y = 0.25$ | 40 units, $\bar y = 0.50$ |
| $x = 1$ (200 units) | 40 units, $\bar y = 0.45$ | 160 units, $\bar y = 0.70$ |

$P(x') = 0.5$ for both values, so the inner sums are $0.5(0.25) + 0.5(0.45) = 0.35$ for $m = 0$ and $0.5(0.50) + 0.5(0.70) = 0.60$ for $m = 1$. with $P(m{=}1 \mid x{=}1) = 0.8$ and $P(m{=}1 \mid x{=}0) = 0.2$:

$$\mathbb E[y \mid \mathrm{do}(x{=}1)] = 0.2(0.35) + 0.8(0.60) = 0.55,\qquad \mathbb E[y \mid \mathrm{do}(x{=}0)] = 0.8(0.35) + 0.2(0.60) = 0.40.$$

the front-door effect is $0.15$, less than half the naive $0.65 - 0.30 = 0.35$. your `backdoor_adjust` and `frontdoor_adjust` must reproduce all four numbers.

## on paper

1. **backdoor from the truncated factorization.** for the graph $Z \to X$, $Z \to Y$, $X \to Y$ with everything observed, derive $P(y \mid \mathrm{do}(x)) = \sum_z P(y \mid x, z)P(z)$. then derive it again with do-calculus rule 2.

<details><summary>solution</summary>

the factorization is $P(z)P(x \mid z)P(y \mid x, z)$. the truncated factorization drops $P(x \mid z)$ and fixes $x$: $P(z, y \mid \mathrm{do}(x)) = P(z)P(y \mid x, z)$. summing over $z$ gives the formula.

with the calculus: $P(y \mid \mathrm{do}(x)) = \sum_z P(y \mid \mathrm{do}(x), z)P(z \mid \mathrm{do}(x))$. first, $P(z \mid \mathrm{do}(x)) = P(z)$ by rule 3: in $G_{\overline X}$, $X$ has no parents and $Z$ isn't its descendant, so $Z \perp X$. second, $P(y \mid \mathrm{do}(x), z) = P(y \mid x, z)$ by rule 2: in $G_{\underline X}$ the only path from $X$ to $Y$ is $X \leftarrow Z \to Y$, blocked by $Z$. this second step is the backdoor criterion in disguise.

</details>

2. **confounded regression.** in the SCM $Z := U_Z$, $X := Z + U_X$, $Y := X + 3Z + U_Y$ with standard normal noises, compute the population slope of $Y$ on $X$ and the causal effect $\frac{d}{dx}\mathbb E[Y \mid \mathrm{do}(x)]$. which one does regressing $Y$ on $X$ and $Z$ recover?

<details><summary>solution</summary>

$\mathrm{Var}(X) = 2$, $\mathrm{Cov}(X, Y) = \mathrm{Var}(X) + 3\,\mathrm{Cov}(X, Z) = 2 + 3 = 5$, so the slope is $2.5$. under $\mathrm{do}(X = x)$, $Y = x + 3U_Z + U_Y$, so the effect is 1. regressing on $X$ and $Z$ gives coefficient 1 on $X$: in linear models, backdoor adjustment is "add the adjustment set as covariates".

</details>

3. **the front-door formula.** for $U \to X$, $U \to Y$, $X \to M \to Y$ with $U$ unobserved, derive the front-door formula with the three rules. use the steps (a) $P(m \mid \mathrm{do}(x)) = P(m \mid x)$, (b) $P(y \mid \mathrm{do}(m)) = \sum_{x'}P(y \mid m, x')P(x')$, (c) $P(y \mid \mathrm{do}(x)) = \sum_m P(y \mid \mathrm{do}(m))P(m \mid \mathrm{do}(x))$.

<details><summary>solution</summary>

(a) rule 2 with $Z = X$: in $G_{\underline X}$, the only path between $X$ and $M$ is $X \leftarrow U \to Y \leftarrow M$, blocked at the collider $Y$.

(b) $P(y \mid \mathrm{do}(m)) = \sum_{x'}P(y \mid \mathrm{do}(m), x')P(x' \mid \mathrm{do}(m))$. rule 3: $P(x' \mid \mathrm{do}(m)) = P(x')$, since in $G_{\overline M}$ $M$ has no path to $X$ that isn't blocked by the collider $Y$. rule 2: $P(y \mid \mathrm{do}(m), x') = P(y \mid m, x')$, since in $G_{\underline M}$ the path $M \leftarrow X \leftarrow U \to Y$ is blocked by $X$.

(c) $P(y \mid \mathrm{do}(x)) = \sum_m P(y \mid \mathrm{do}(x), m)P(m \mid \mathrm{do}(x))$. rule 2 turns $P(y \mid \mathrm{do}(x), m)$ into $P(y \mid \mathrm{do}(x), \mathrm{do}(m))$ (in $G_{\overline X\underline M}$, $M$'s only remaining path to $Y$ is out of $M$, removed). rule 3 deletes $\mathrm{do}(x)$: in $G_{\overline X\,\overline M}$ nothing connects $X$ to $Y$. so the inner term is $P(y \mid \mathrm{do}(m))$. substitute (a) and (b).

</details>

4. **reading the graph.** in $A \to B \to D \leftarrow C$, $D \to E$, is $A \perp C$? $A \perp C \mid E$? $A \perp C \mid \{E, B\}$?

<details><summary>solution</summary>

the only path is $A \to B \to D \leftarrow C$, with a collider at $D$. unconditioned: blocked at $D$, so yes. given $E$ (a descendant of $D$): the collider opens and $B$ isn't conditioned on, so no. given $\{E, B\}$: the collider is open but the chain is blocked at $B$, so yes.

</details>

## experiment

`lab.py` uses your SCM simulator and adjustment code in three simulated worlds:

1. a world where vitamin-takers are healthier: how far can the correlation and the causal effect disagree, and does adjustment fix it?
2. "adjust for everything you measured before treatment": a graph with a confounder, a mediator, a collider and an M-structure. which adjustment sets work?
3. an unmeasured confounder with a front-door path: does the front-door estimate converge, and what happens when its assumptions fail?

## check your understanding

1. why does an intervention cut arrows *into* $X$ but leave arrows *out of* $X$ alone?

<details><summary>answer</summary>

$\mathrm{do}(X = x)$ replaces how $X$ is *produced*: it no longer listens to its parents. everything downstream still listens to $X$ (that's the effect you want to measure), so its equations are unchanged.

</details>

2. two variables are correlated in data from an SCM. what are the possible graph structures behind it?

<details><summary>answer</summary>

any open path: one causes the other (chain), a common cause (fork), or selection on a common effect (a collider you conditioned on, e.g. by how the sample was collected). correlation tells you some path is open, not which.

</details>

3. (warm-up 1.) can vitamins be harmful in a world where vitamin-takers are healthier?

<details><summary>answer</summary>

yes: if health-conscious people both take vitamins and are healthier for other reasons, the backdoor path through health-consciousness can carry a positive association larger than a negative causal effect. the lab builds exactly this.

</details>

4. why does the backdoor formula weight strata by $P(z)$ and not by $P(z \mid x)$?

<details><summary>answer</summary>

$P(z \mid x)$ is the stratum mix among those who *chose* $x$; it differs between treatment groups when $Z$ confounds. an intervention assigns $x$ to everyone, so the strata keep their population frequencies $P(z)$.

</details>

5. if $X$ has no parents, show that $P(y \mid \mathrm{do}(x)) = P(y \mid x)$.

<details><summary>answer</summary>

with no arrows into $X$ there are no backdoor paths, so the empty set satisfies the backdoor criterion and the formula reduces to $P(y \mid x)$. that's why randomized experiments work: randomization makes the coin flip $X$'s only parent.

</details>

6. what goes wrong if you adjust for a mediator $M$ on $X \to M \to Y$?

<details><summary>answer</summary>

you block the causal path itself, so the estimate measures only the direct effect (here zero), not the total effect. and if $M$ shares an unobserved cause with $Y$, conditioning on it also opens $X \to M \leftarrow U \to Y$.

</details>

7. the front-door formula never uses $U$. where did the assumption "no unobserved confounder of $X$ and $M$" enter?

<details><summary>answer</summary>

step (a), $P(m \mid \mathrm{do}(x)) = P(m \mid x)$, needs no open backdoor path from $X$ to $M$. an unobserved $U' \to X$, $U' \to M$ would open one and break that step.

</details>

## common bugs

- **weighting by $P(z \mid x)$.** symptom: `backdoor_adjust` returns the naive conditional mean. the stratum weights are frequencies over *all* rows.
- **drawing noise in topological order.** symptom: your samples are valid but don't match the checks, and `do` changes the noise of other nodes. draw the full `(n, len(nodes))` matrix first, column by `nodes` index.
- **forgetting colliders' descendants.** symptom: d-separation says "blocked" when you condition on a child of a collider. keep ancestors of $Z$ when you build the moral graph; they're how descendants of colliders reopen paths.
- **removing incoming edges for the backdoor test.** symptom: everything looks like a valid adjustment set. the backdoor test deletes $X$'s *outgoing* edges; graph surgery for $\mathrm{do}(X)$ deletes the incoming ones.
- **adjusting for descendants of $X$.** symptom: estimates shrink towards zero (mediators) or pick up bias (colliders). check condition 1 of the criterion.
- **an infinite loop on cyclic graphs.** symptom: `topological_order` hangs. if a full pass places no node, raise.

## going deeper

- Pearl (2009), *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, chapters 1, 3 (d-separation, backdoor, front-door, do-calculus). [doi:10.1017/CBO9780511803161](https://doi.org/10.1017/CBO9780511803161)
- Pearl (1995), "Causal diagrams for empirical research", *Biometrika* 82(4). the do-calculus and the front-door criterion. [doi:10.1093/biomet/82.4.669](https://doi.org/10.1093/biomet/82.4.669)
- Pearl, Glymour & Jewell (2016), *Causal Inference in Statistics: A Primer*, Wiley. a gentle version of this topic, chapters 2–3.
- Peters, Janzing & Schölkopf (2017), *Elements of Causal Inference*, MIT Press (open access), chapters 3 and 6. [mitpress.mit.edu/9780262037310](https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/)
- Lauritzen, Dawid, Larsen & Leimer (1990), "Independence properties of directed Markov fields", *Networks* 20(5). the moralization criterion for d-separation. [doi:10.1002/net.3230200503](https://doi.org/10.1002/net.3230200503)
- Charig, Webb, Payne & Wickham (1986), "Comparison of treatment of renal calculi by open surgery, percutaneous nephrolithotomy, and extracorporeal shockwave lithotripsy", *BMJ* 292. the kidney-stone data. [doi:10.1136/bmj.292.6524.879](https://doi.org/10.1136/bmj.292.6524.879)
- Shpitser & Pearl (2006), "Identification of joint interventional distributions in recursive semi-Markovian causal models", *AAAI*. completeness of the do-calculus.
- Hernán & Robins (2020), *Causal Inference: What If*, Chapman & Hall/CRC, chapters 6–8 (graphs, confounding, selection bias). [free online](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

## connections

- ← builds on: conditional probability and the law of total probability (ch 04 · topic 01), graphical models and conditional independence (ch 12), regression slopes (ch 08 · topic 01), seeded sampling with `as_rng` (ch 00 · topic 05).
- → used in: potential outcomes and weighting, where the backdoor formula becomes inverse propensity weighting (topic 02); double machine learning, the backdoor formula with flexible models (topic 03); causal discovery, which tries to learn the graph you assumed here (topic 04).
