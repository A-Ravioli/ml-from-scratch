# graphical models

## why this matters

a medical diagnosis model over 30 yes/no variables (symptoms, test results, diseases, risk factors) has a joint distribution with $2^{30} \approx 10^9$ probabilities. you can't store that table, you certainly can't estimate it from data, and no doctor could tell you most of its entries. but a doctor *can* tell you the structure: smoking affects lung cancer, lung cancer affects the X-ray, the X-ray doesn't care about smoking once you know about the cancer. a handful of such local statements pin the whole distribution down with a few hundred numbers.

**graphical models** turn that structure into mathematics. a graph says which variables interact directly; the distribution is a product of small tables (**factors**), one per local interaction. the graph then does three jobs at once: it makes the model compact, it makes inference tractable (topic 03), and it lets you *read conditional independences off the picture*, without computing anything. that last one is subtle. two causes that have nothing to do with each other become dependent the moment you observe a common effect. it's the reason a good alarm system makes you *less* sure there was a burglary once you hear about the earthquake, and the reason selected samples (hospital patients, published papers, famous people) show correlations that aren't there in the population.

## warm-up: predict first

1. three binary variables in a chain $A \to B \to C$. how many numbers do you need to specify the joint distribution with the chain's structure, and how many without any structure?
2. two independent coins; a bell rings if at least one shows heads. you hear the bell. are the coins still independent?
3. in an undirected graph, when are two sets of nodes conditionally independent given a third?
4. which nodes do you need to know to make a variable independent of *everything* else in a Bayes net: its parents only, or more?
5. variable elimination and belief propagation are exact on trees. is the factor graph of "cloudy → sprinkler, cloudy → rain, sprinkler + rain → wet grass" a tree?

## prerequisites check

- [ ] you know joint, marginal and conditional distributions, the chain rule, and conditional independence (→ ch 04 · topic 01)
- [ ] you can multiply and sum NumPy arrays along chosen axes (`np.einsum`, broadcasting, `axis=`) (→ ch 00)
- [ ] you know breadth-first search on a graph (→ ch 00)
- [ ] you know mutual information (→ ch 05 · topic 05)

## core ideas

### idea 1: factors

a **factor** $\phi(X_1, \dots, X_k)$ is a non-negative table indexed by the values of some discrete variables: a probability table, a conditional probability table (CPT) like $P(S \mid C)$, or an unnormalized "compatibility score". a graphical model is a distribution written as a **product of factors**:

$$p(x_1, \dots, x_n) = \frac{1}{Z}\prod_{a}\phi_a(x_{S_a}),$$

where each factor only touches a small subset $S_a$ of the variables and $Z$ makes it sum to one. a Bayes net is the special case where the factors are CPTs $P(x_i \mid \text{parents}_i)$, so $Z = 1$ automatically. the chain $A \to B \to C$ needs $1 + 2 + 2 = 5$ numbers instead of $2^3 - 1 = 7$ (warm-up 1); for 30 variables with at most 3 parents each it's at most $30\cdot 8 = 240$ instead of a billion.

the one operation that builds everything is the **factor product**. $(\phi_1\cdot\phi_2)(x_{S_1 \cup S_2}) = \phi_1(x_{S_1})\,\phi_2(x_{S_2})$: the result has one axis per variable in the union, and each entry multiplies the two entries that agree on the shared variables. it's a broadcasted multiply after lining up the axes, or a single `np.einsum`. in code a factor is three things: a tuple of variable names, a tuple of cardinalities, and an array with one axis per variable in that order.

> **spec.** `Factor(variables, cardinalities, values)` stores `variables` and `cardinalities` as tuples and `values` as a float array of shape `cardinalities` (a flat array of the right size is reshaped; repeated variables, or the wrong number of values, raise `ValueError`). `f * g` returns a new `Factor` whose variables are `f`'s, then `g`'s new ones in `g`'s order; a shared variable with different cardinalities raises `ValueError`.

> **▶ build: stage 1.** implement `Factor.__init__` and `Factor.__mul__`.

### idea 2: inference by brute force

two more operations finish the toolkit. **marginalize** sums a variable out: $\sum_{x_j}\phi(\dots, x_j, \dots)$ removes an axis. **reduce** conditions on evidence: keep the slice where $X_j$ equals its observed value, and drop the axis. with those, any query is

$$P(Q \mid E = e) = \frac{\sum_{\text{rest}}\prod_a\phi_a(\cdot)\big|_{E = e}}{\sum_{Q, \text{rest}}\prod_a\phi_a(\cdot)\big|_{E = e}}:$$

multiply every factor into the joint, reduce by the evidence, sum out everything except the query, and **normalize** (divide by the total). the normalizer is $P(E = e)$, so you never need to compute it separately. it's exact and simple, and it costs time and memory exponential in the number of variables, because the first step builds the full joint. topic 03 is about never building it.

> **spec.** `marginalize(variables)` takes one name or a list and returns a new factor over the rest (summing out everything gives a factor with `variables == ()`); `reduce(evidence)` takes a dict name → value index, drops those variables, and ignores names the factor doesn't have; `normalize()` returns a factor whose values sum to 1. none of them modify `self`. `joint_query(factors, query, evidence=None)` returns the normalized factor over `query` (a name, or a tuple of names in that order).

> **🔮 predict:** in the sprinkler network of the worked example the grass is wet. then you learn it rained. does the probability that the sprinkler was on go up, down, or stay the same?

> **▶ build: stage 2.** implement `marginalize`, `reduce`, `normalize` and `joint_query`.

### idea 3: undirected graphs and separation

a **Markov random field** (MRF) draws an undirected edge between every two variables that share a factor. the factors (potentials) are arbitrary non-negative scores, so $Z$ has to be computed: an Ising model, an image denoising prior where neighboring pixels like to agree, a conditional random field over the words of a sentence.

the beautiful fact about MRFs is that independence is plain graph connectivity. **separation**: if every path from $X$ to $Y$ passes through a node in $Z$, then $X \perp Y \mid Z$ (warm-up 3). the reason fits in a line: once $Z$ is fixed, the factors split into those touching $X$'s side and those touching $Y$'s side, and the distribution factorizes into a function of $X$ times a function of $Y$. (the converse, "every independence shows up as a separation", holds for generic potentials, though not for every special choice of numbers.)

you can always turn a Bayes net into an MRF that describes the same factorization: drop the arrow directions and **marry the parents** (connect every two parents of a common child), because the CPT $P(x_i \mid \text{pa}_i)$ is a factor over the child *and all its parents* together. the result is the **moral graph**. it can only lose independences, never invent them. one consequence: in a Bayes net a node's **Markov blanket**, the smallest set that makes it independent of everything else, is its neighbors in the moral graph: parents, children, and the children's other parents (warm-up 4). Gibbs sampling (topic 05) resamples each variable given exactly its blanket.

> **spec.** `moralize(edges)` takes a DAG as `(parent, child)` pairs and returns a set of `frozenset({a, b})` undirected edges. `markov_blanket(edges, node)` returns a set. `separated(undirected_edges, x, y, given=())` takes edges as 2-element sets (or pairs), and `x`, `y` as single nodes or sets; it returns `True` if no path from `x` to `y` avoids `given` (breadth-first search that never enters a `given` node). nodes that appear in no edge are separated from everything.

> **▶ build: stage 3.** implement `moralize`, `markov_blanket` and `separated`.

### idea 4: Bayes nets and d-separation

in a directed graph, connectivity isn't enough. there are three ways a path can pass through a middle node $B$:

- **chain** $A \to B \to C$: information flows through $B$; observing $B$ blocks it ($A \perp C \mid B$).
- **fork** $A \leftarrow B \to C$: a common cause makes $A$ and $C$ correlated; observing it blocks the path.
- **collider** $A \to B \leftarrow C$: the opposite. $A$ and $C$ are independent, until you observe $B$, *or any descendant of $B$*. then they become dependent: if the bell rang and the first coin shows tails, the second must show heads (warm-up 2). this is **explaining away**.

a path is **blocked** by $Z$ if it contains a chain or fork node in $Z$, or a collider that is not in $Z$ and has no descendant in $Z$. $X$ and $Y$ are **d-separated** by $Z$ if every path between them is blocked, and then $X \perp Y \mid Z$ in every distribution that factorizes over the graph.

checking every path is exponential, but there's a slick equivalent (Lauritzen et al., 1990): keep only $X \cup Y \cup Z$ and their **ancestors**, moralize that subgraph, and test ordinary separation. why does it work? a collider with no observed descendant isn't an ancestor of anything observed, so it's dropped, and its blocking job is done by its absence; a collider that *is* an ancestor of $Z$ stays, and moralization marries its parents, which opens exactly the path that explaining away opens. you've already written both pieces.

> **spec.** `ancestors(edges, nodes)` returns `nodes` together with all their ancestors. `d_separated(edges, x, y, given=())` takes a DAG as `(parent, child)` pairs, `x` and `y` as single nodes or sets, and returns a bool, by the ancestral moral graph test. raise `ValueError` if `x`, `y` and `given` overlap.

> **🔮 predict:** in the sprinkler network (cloudy → sprinkler, cloudy → rain, sprinkler → wet, rain → wet), are sprinkler and rain independent given cloudy? given cloudy *and* wet grass?

> **▶ build: stage 4.** implement `ancestors` and `d_separated`.

### idea 5: factor graphs

Bayes nets and MRFs both hide something: an MRF triangle could be one factor over three variables or three pairwise factors, and the graph can't tell the difference. a **factor graph** makes the factors explicit: a bipartite graph with one node per variable, one node per factor, and an edge whenever a factor mentions a variable. it's the data structure inference algorithms run on.

the property that matters most: is the factor graph a **tree** (no cycles)? on a tree, every variable splits the graph into independent pieces, and you can compute every marginal exactly by passing messages along the edges (belief propagation, topic 03), in time linear in the number of factors. a cycle means some information arrives by two routes, and exact inference has to work harder. warm-up 5: the sprinkler network's factor graph has the cycle $C - P(S \mid C) - S - P(W \mid S, R) - R - P(R \mid C) - C$, even though its Bayes net looks innocent. a graph with $V$ nodes, $E$ edges and $k$ connected components is a forest exactly when $E = V - k$; union–find detects the first edge that closes a cycle.

> **spec.** `factor_graph(factors)` returns a dict mapping each variable to the list of indices of the factors that mention it. `is_tree(factors)` returns `True` if the bipartite variable–factor graph has no cycle (a forest counts). two factors over the same pair of variables form a cycle.

> **▶ build: stage 5.** implement `factor_graph` and `is_tree`.

## worked example

the sprinkler network, every variable binary (1 = yes): $P(C{=}1) = 0.5$; $P(S{=}1 \mid C) = 0.5$ if $C = 0$, $0.1$ if $C = 1$; $P(R{=}1 \mid C) = 0.2$ if $C = 0$, $0.8$ if $C = 1$; and $P(W{=}1 \mid S, R) = 0, 0.9, 0.9, 0.99$ for $(S, R) = (0,0), (0,1), (1,0), (1,1)$.

**$P(R{=}1 \mid W{=}1)$ by brute force.** multiply all four factors, reduce to $W = 1$, and sum out $C$. that gives the unnormalized table $P(S, R, W{=}1)$:

| | $R = 0$ | $R = 1$ |
|---|---|---|
| $S = 0$ | 0 | 0.369 |
| $S = 1$ | 0.189 | 0.0891 |

for example, the $(S{=}0, R{=}1)$ entry is $\sum_c P(c)P(S{=}0 \mid c)P(R{=}1 \mid c)\cdot0.9 = 0.5(0.5\cdot0.2 + 0.9\cdot0.8)\cdot0.9 = 0.369$. the total is $P(W{=}1) = 0.6471$. summing out $S$ and normalizing: $P(R{=}1 \mid W{=}1) = (0.369 + 0.0891)/0.6471 = 0.7079$; summing out $R$ instead, $P(S{=}1 \mid W{=}1) = 0.2781/0.6471 = 0.4298$.

**explaining away.** now add $R = 1$: keep the $R = 1$ column, $P(S{=}1 \mid W{=}1, R{=}1) = 0.0891/(0.369 + 0.0891) = 0.1945$. the sprinkler went from 0.43 to 0.19: rain already explains the wet grass. your `joint_query` must reproduce all three numbers.

## on paper

1. **separation implies independence.** in an MRF $p(x) \propto \prod_a\phi_a(x_{S_a})$, suppose $Z$ separates $X$ from $Y$. show that $p(x, y \mid z)$ factorizes as $f(x, z)\,g(y, z)$.

<details><summary>solution</summary>

remove $Z$ from the graph; let $A$ be the set of nodes connected to $X$, and $B$ the rest (it contains $Y$). no factor touches both $A$ and $B$ (it would create an edge between them), so each factor's variables lie in $A \cup Z$ or in $B \cup Z$. group them: $p(x_A, x_B, z) \propto F(x_A, z)\,G(x_B, z)$. summing out the parts of $A$ and $B$ that aren't $X$ or $Y$ keeps the product form, so $p(x, y \mid z) \propto f(x, z)g(y, z)$, which is conditional independence.

</details>

2. **explaining away with numbers.** $A, C$ are independent fair coins (1 = heads) and $B = A \lor C$. compute $P(A{=}1)$, $P(A{=}1 \mid B{=}1)$ and $P(A{=}1 \mid B{=}1, C{=}1)$.

<details><summary>solution</summary>

$P(A{=}1) = 1/2$. $B = 1$ in 3 of 4 equally likely outcomes, $A = 1$ in 2 of them: $P(A{=}1 \mid B{=}1) = 2/3$. given also $C = 1$, $B = 1$ is guaranteed whatever $A$ is, so $P(A{=}1 \mid B{=}1, C{=}1) = 1/2$: $C$ explained the bell away, and $A$ falls back to its prior.

</details>

3. **why the ancestral graph is enough.** in the collider $A \to B \leftarrow C$ with $Z = \emptyset$, what graph does the ancestral moral test build, and what does it conclude? and with $Z = \{D\}$ for an extra edge $B \to D$?

<details><summary>solution</summary>

with $Z = \emptyset$ the ancestral set of $\{A, C\}$ is $\{A, C\}$ alone: no edges, so they're separated. with $Z = \{D\}$ the ancestral set is $\{A, B, C, D\}$; moralizing marries $A$ and $C$ (parents of $B$), so the edge $A - C$ connects them directly and no conditioning set can separate them: dependent, as the collider-with-observed-descendant rule says.

</details>

4. **a Bayes net's Markov blanket.** show that $p(x_i \mid x_{-i})$ depends only on $x_i$'s parents, children and co-parents.

<details><summary>solution</summary>

$p(x_i \mid x_{-i}) \propto p(x) = \prod_j P(x_j \mid \text{pa}_j)$. as a function of $x_i$, only the factors that contain $x_i$ matter: its own CPT $P(x_i \mid \text{pa}_i)$ and the CPTs of its children $P(x_c \mid \text{pa}_c)$, which also mention the children's other parents. every other factor cancels in the normalization.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how hearing an alarm, and then about an earthquake, changes the probability of a burglary.
2. what selecting a sample on a common effect does to the correlation between two independent causes.
3. which conditioning sets make the sprinkler and the rain independent, exactly and as measured from samples.

## check your understanding

1. why is a Bayes net's normalizing constant always 1, while an MRF's must be computed?

<details><summary>answer</summary>

each CPT sums to 1 over its child, so summing the product over variables in reverse topological order removes one CPT at a time, each contributing 1. MRF potentials have no such constraint, and $Z$ is a sum over exponentially many configurations (computing it is #P-hard in general).

</details>

2. two different DAGs, $A \to B \to C$ and $A \leftarrow B \leftarrow C$. do they imply the same independences?

<details><summary>answer</summary>

yes: both say exactly $A \perp C \mid B$. they're **Markov equivalent**. from observational data alone you can't tell them apart; $A \to B \leftarrow C$, the collider, implies different independences and *can* be distinguished.

</details>

3. a factor's values don't have to sum to 1. why does `joint_query` still return probabilities?

<details><summary>answer</summary>

it normalizes at the end, and conditional probabilities don't depend on a global constant: any scaling of the factors cancels between numerator and denominator.

</details>

4. you condition on a variable that is a descendant of a collider but not the collider itself. why does that open the path?

<details><summary>answer</summary>

the descendant carries information about the collider (it's a noisy measurement of it), so observing it is like partially observing the collider, which creates the explaining-away dependence between its parents.

</details>

5. a dataset shows that $X$ and $Y$ are independent. can you conclude there's no edge between them?

<details><summary>answer</summary>

not quite: a special choice of parameters can cancel two paths exactly (a violation of *faithfulness*). for generic parameters, independence does mean d-separation, which is what constraint-based structure learning (the PC algorithm) assumes.

</details>

6. why can't you represent the independences of $A \to B \leftarrow C$ with an undirected graph?

<details><summary>answer</summary>

an undirected graph with no $A - C$ edge says $A \perp C \mid B$ (wrong: observing $B$ couples them), and adding the edge loses $A \perp C$. moralization picks the second: it keeps the factorization correct and gives up the marginal independence.

</details>

## common bugs

- **multiplying factors without aligning axes.** symptom: products that look plausible but are wrong whenever the shared variable isn't the first axis of both. transpose each factor to the union order (with size-1 axes for missing variables) before multiplying, or use `np.einsum`.
- **forgetting the co-parents.** symptom: Markov blankets and moral graphs missing the edge between two parents of a common child.
- **treating colliders like chains.** symptom: `d_separated` says "dependent" for two independent causes when nothing is observed, or "independent" once their common effect is observed.
- **dropping descendants of colliders.** symptom: observing a child of the collider doesn't open the path. the ancestral set must include the ancestors of the *evidence*, which brings the collider back in.
- **modifying factors in place.** symptom: a second query gives a different answer than the first. return new factors from every operation.

## going deeper

- Koller & Friedman (2009), *Probabilistic Graphical Models: Principles and Techniques*, MIT Press, ch. 3–4 (Bayes nets, MRFs, d-separation, I-maps) and ch. 9 (factors). [mitpress.mit.edu/9780262013192](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- Pearl (1988), *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann. d-separation, explaining away and the alarm example.
- Lauritzen, Dawid, Larsen & Leimer (1990), "Independence properties of directed Markov fields", *Networks* 20(5). the ancestral moral graph criterion. [doi:10.1002/net.3230200503](https://doi.org/10.1002/net.3230200503)
- Kschischang, Frey & Loeliger (2001), "Factor graphs and the sum-product algorithm", *IEEE Transactions on Information Theory* 47(2). [doi:10.1109/18.910572](https://doi.org/10.1109/18.910572)
- Shachter (1998), "Bayes-ball: the rational pastime", *Proceedings of UAI 1998*. a linear-time path-following algorithm for d-separation.
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, ch. 8.

## connections

- ← builds on: conditional independence and the chain rule (ch 04 · topic 01), array axes and `einsum` (ch 00), graph search (ch 00), mutual information (ch 05 · topic 05).
- → used in: variable elimination and belief propagation on your `Factor` (topic 03), HMMs as a chain-structured graphical model (topic 04), Gibbs sampling over Markov blankets (topic 05), the hierarchical model in the capstone, causal discovery and the PC algorithm (ch 29 · causal inference elective), conditional random fields for sequence labelling.
