# exact inference

## why this matters

topic 02 answered every question by brute force: multiply all the factors into the joint table, then sum. that's fine for the sprinkler network and hopeless for anything real. a chain of 40 binary variables has a joint with $2^{40}$ entries, about 8 terabytes, yet the question "what's the distribution of the last variable?" only involves passing a 2-vector down the chain 40 times. the joint was never needed. you just summed in a silly order.

exact inference is the art of summing in a good order. two algorithms do it: **variable elimination** answers one query by pushing each sum as far inside the product as it will go; **belief propagation** (the sum-product algorithm) answers *every* single-variable query on a tree at the cost of about two eliminations, by caching the intermediate results as **messages**. they're the same idea seen twice, and you've met it before: the forward algorithm for HMMs (next topic), the Kalman filter, Viterbi decoding, dynamic programming in RL, and even backpropagation are all instances of "distribute the sum over the product". you'll also see where it stops: on a grid or any graph with many loops, even the best order produces tables that grow exponentially with the graph's width, and that's why the rest of the chapter is about approximate inference.

## warm-up: predict first

1. $\sum_a\sum_b f(a)g(a, b)h(b)$ with $a, b$ each taking $k$ values. how many multiplications does the double sum take as written? can you do it in $O(k^2)$ with fewer?
2. a star: a hub $h$ connected by pairwise factors to 20 binary leaves. you want $P(h)$. what goes wrong if you sum out the hub first?
3. is there always an elimination order whose intermediate tables stay small?
4. you need the marginal of all 100 variables of a tree. is that 100 times the work of one marginal?
5. belief propagation is exact on trees. what do you think happens if you run it on a graph with loops anyway?

## prerequisites check

- [ ] you can multiply, marginalize and reduce `Factor`s, and know what `joint_query` computes (→ topic 02, ideas 1–2)
- [ ] you know factor graphs and what makes one a tree (→ topic 02, idea 5)
- [ ] you can reason about the cost of NumPy operations in terms of array sizes (→ ch 00)

## core ideas

### idea 1: push the sums inside

the whole topic rests on the distributive law, $ab + ac = a(b + c)$. take a chain $A \to B \to C$ with $p(a, b, c) = P(a)P(b \mid a)P(c \mid b)$. the marginal of $C$ written naively is

$$P(c) = \sum_a\sum_b P(a)\,P(b \mid a)\,P(c \mid b),$$

which visits every $(a, b, c)$ triple. but $P(c \mid b)$ doesn't depend on $a$, so it can come out of the inner sum:

$$P(c) = \sum_b P(c \mid b)\underbrace{\sum_a P(a)\,P(b \mid a)}_{\tau(b)}.$$

the inner sum produces a new factor $\tau(b)$, a function of $b$ alone, and $a$ is gone for good. that's **eliminating** a variable: take every factor that mentions it, multiply just those, sum the variable out, and put the result back in the list. the other factors don't change. repeat until only the query variables are left; their product is the (unnormalized) answer.

nothing is approximated: eliminating in any order gives exactly the same result as brute force, because it's the same sum, rearranged. what changes is the cost, and it changes enormously.

> **spec.** `eliminate(factors, var)` returns a new list: the factors that don't mention `var` (in their original order), followed by one new factor, the product of those that do with `var` summed out. if no factor mentions `var`, return the list unchanged. use your `Factor`'s `*` and `marginalize`.

> **▶ build: stage 1.** implement `eliminate`.

### idea 2: the order decides the cost

eliminating $v$ creates a table over all of $v$'s current **neighbours** (the variables that share a factor with it). those neighbours are now joined in one factor, so in the interaction graph they've become a clique: elimination adds **fill-in** edges. the cost of an elimination order is dominated by the largest table it ever creates, which is the product of the cardinalities of the largest such neighbourhood (plus $v$ itself).

the star from warm-up 2 shows how much this matters. eliminate the hub first and its 20 neighbours become one factor with $2^{21}$ entries. eliminate the leaves first and every step multiplies one $2 \times 2$ table into the hub. same answer; two million times less work.

for a general graph the minimum over all orders of (largest neighbourhood) is the graph's **treewidth** (plus one), and finding the best order is NP-hard. but simple greedy heuristics are usually good. **min-degree** eliminates, at each step, a variable with the fewest current neighbours, then updates the graph with its fill-in. on a tree it peels off leaves and never creates anything bigger than the original factors; on an $n \times n$ grid it gets close to the best possible, a frontier of about one row, $2^{n+1}$ entries for binary variables. no order beats the treewidth, and a $100 \times 100$ grid has treewidth 100: exact inference on an image-sized MRF is out of reach, however cleverly you sum (warm-up 3).

> **spec.** `min_degree_order(factors, variables)` returns the variables in `variables` in greedy min-degree order: build the interaction graph from the factors' scopes; repeatedly pick the variable with the fewest neighbours (ties go to the one that comes first in `variables`), connect its neighbours to each other, remove it. `elimination_cost(factors, order)` simulates eliminating `order` on the scopes alone (no arithmetic) and returns the number of entries of the largest product table formed, counting the original factors too.

> **🔮 predict:** a $4 \times 4$ grid of binary variables, eliminated row by row (left to right, top to bottom). what's the largest table, in entries?

> **▶ build: stage 2.** implement `min_degree_order` and `elimination_cost`.

### idea 3: variable elimination

the full algorithm for $P(Q \mid E = e)$:

1. **reduce** every factor by the evidence: the observed variables disappear, and the factors now describe $p(\cdot, E = e)$ up to a constant.
2. the **hidden** variables are everything left except the query. choose an order for them (min-degree unless one is given).
3. **eliminate** them one at a time.
4. **multiply** what's left (factors over the query variables, plus constants) and **normalize**. the normalizer is $P(E = e)$, which you get for free.

the cost is linear in the number of factors times the size of the largest table: on a chain of $n$ binary variables it's $O(n)$ instead of $O(2^n)$. the price of its simplicity is that it answers one query. ask for $P(x_1)$ and then $P(x_2)$ and it redoes almost all the work, because the elimination order for $x_2$ shares most of its intermediate tables with the order for $x_1$.

> **spec.** `variable_elimination(factors, query, evidence=None, order=None)` returns a normalized `Factor` over `query` (one name, or a tuple of names in that order). `order` lists the variables to eliminate; names that aren't hidden are skipped; if it misses a hidden variable, raise `ValueError`. when `order` is `None`, use your `min_degree_order` on the reduced factors' hidden variables (in order of first appearance).

> **▶ build: stage 3.** implement `variable_elimination`.

### idea 4: belief propagation on trees

on a tree-shaped factor graph you can share that work. picture eliminating towards a variable $x$: every factor $f$ next to $x$ ends up sending $x$ a summary of everything on its side of the tree, a function of $x$ alone. call it a **message** $m_{f \to x}(x)$. the summary only depends on the subtree behind $f$, not on which variable you were eliminating towards, so each message can be computed once and reused for every query. two kinds of messages, defined recursively:

$$m_{x \to f}(x) = \prod_{g \in N(x) \setminus f} m_{g \to x}(x), \qquad m_{f \to x}(x) = \sum_{x_{N(f) \setminus x}} f(x_{N(f)})\prod_{y \in N(f) \setminus x} m_{y \to f}(y),$$

where $N(\cdot)$ is the set of neighbours in the factor graph. a variable passes on the product of what its other factors told it; a factor multiplies in what its other variables told it and sums them out. a leaf variable sends all-ones, a leaf factor sends itself. the **belief** at $x$ is the product of all its incoming messages, and normalized it's the exact marginal:

$$p(x) \propto \prod_{f \in N(x)} m_{f \to x}(x).$$

on a tree with $E$ edges there are $2E$ messages in total, one per edge in each direction, so every marginal costs about twice one elimination (warm-up 4). schedule them as two passes: pick a root, list the nodes of the (bipartite) tree in breadth-first order from it, compute the messages towards the root in *reverse* order (each node's message to its parent needs only the messages from its children, which are already done), then the messages away from the root in forward order. computing each message on demand with recursion and a memo dict also works on small trees, but its recursion depth grows with the length of the tree, and Python gives up (`RecursionError`) on a chain of a few hundred variables. normalize each message as you go: it doesn't change the beliefs and keeps long chains from underflowing. evidence works exactly as in elimination: reduce the factors first.

on a graph with loops there's no such order, because a message would (indirectly) depend on itself. iterating the same updates anyway is **loopy belief propagation**: often a good approximation (it decodes turbo codes and LDPC codes in every phone), sometimes oscillating or overconfident, never guaranteed (warm-up 5). your `sum_product` should refuse loops.

> **spec.** `sum_product(factors, evidence=None)` reduces the factors by `evidence` (dropping factors with no variables left), checks that the factor graph is a forest (#edges = #nodes − #components; raise `ValueError` otherwise), computes the messages with an explicit two-pass schedule (no recursion: it must handle a chain of 600 variables), and returns a dict mapping every unobserved variable to its normalized marginal, a 1-D array of length equal to its cardinality.

> **🔮 predict:** in the worked example's chain, observing $C$ changes the belief at $A$. through which message does the information arrive, and does $A$'s message to $B$ change too?

> **▶ build: stage 4.** implement `sum_product`.

## worked example

the chain $A \to B \to C$, all binary: $P(A) = (0.6, 0.4)$, $P(B \mid A) = \begin{pmatrix}0.7 & 0.3\\0.2 & 0.8\end{pmatrix}$, $P(C \mid B) = \begin{pmatrix}0.9 & 0.1\\0.4 & 0.6\end{pmatrix}$ (rows: the parent's value).

**$P(C)$ by elimination, order $A$ then $B$.** eliminating $A$: $\tau(b) = \sum_aP(a)P(b \mid a) = (0.6\cdot0.7 + 0.4\cdot0.2,\ 0.6\cdot0.3 + 0.4\cdot0.8) = (0.5, 0.5)$. eliminating $B$: $P(c) = \sum_b\tau(b)P(c \mid b) = (0.5\cdot0.9 + 0.5\cdot0.4,\ 0.5\cdot0.1 + 0.5\cdot0.6) = (0.65, 0.35)$. the largest table was $2 \times 2$; brute force would have built $2^3$.

**$P(A \mid C{=}1)$, as messages.** reducing $P(C \mid B)$ to $C = 1$ leaves the factor $(0.1, 0.6)$ over $B$: that's the message from the reduced factor to $B$. $B$ forwards it to the factor $P(B \mid A)$, which sends $A$ the message $m(a) = \sum_bP(b \mid a)\,(0.1, 0.6)_b = (0.7\cdot0.1 + 0.3\cdot0.6,\ 0.2\cdot0.1 + 0.8\cdot0.6) = (0.25, 0.5)$. the belief at $A$ multiplies in the message from its prior factor, $(0.6, 0.4)$: $(0.15, 0.2)$, normalized $(0.4286, 0.5714)$. observing $C = 1$ raised $P(A{=}1)$ from 0.4 to 0.57.

**$P(B \mid C{=}1)$.** $B$'s incoming messages are $\tau(b) = (0.5, 0.5)$ from above and $(0.1, 0.6)$ from below: belief $(0.05, 0.3)$, normalized $(0.143, 0.857)$. your `variable_elimination` and `sum_product` must reproduce these numbers.

## on paper

1. **cost of a chain.** for a chain of $n$ variables with $k$ states each, count the multiplications done by brute force and by eliminating from one end.

<details><summary>solution</summary>

brute force builds the joint ($k^n$ entries, each a product of $n$ factor values: about $nk^n$ multiplications) and sums it. eliminating $x_1$, then $x_2$, ...: each step multiplies a $k$-vector into a $k \times k$ table and sums, $k^2$ multiplications, so $(n - 1)k^2$ in total. for $n = 40$, $k = 2$: about 44 trillion versus 156.

</details>

2. **fill-in on a cycle.** four variables in a cycle $a - b - c - d - a$ (pairwise factors). eliminate $a$ first. which edge is added, and what's the largest table? is there an order that adds no fill-in?

<details><summary>solution</summary>

$a$'s neighbours are $b$ and $d$, so eliminating it creates $\tau(b, d)$ and adds the edge $b - d$; the product table is over $\{a, b, d\}$: $k^3$ entries. every order adds one fill-in edge on a 4-cycle (the first eliminated variable always has two non-adjacent neighbours), so the treewidth is 2. a cycle is the smallest graph that isn't a tree, and it already costs more.

</details>

3. **messages are eliminations.** in the chain $A - B - C$ (factors $P(A)$, $P(B \mid A)$, $P(C \mid B)$), show that the message from $P(B \mid A)$ to $B$ is exactly $\tau(b)$ from eliminating $A$.

<details><summary>solution</summary>

$A$'s message to $P(B \mid A)$ is the product of its other incoming messages: just $m_{P(A) \to A}(a) = P(a)$ (a leaf factor sends itself). the factor then multiplies that in and sums $A$ out: $m(b) = \sum_aP(b \mid a)P(a) = \tau(b)$. elimination towards $B$ computes exactly this message, and so does elimination towards $C$: that's the reuse.

</details>

4. **why loops break exactness.** in a triangle $a - b - c - a$ of pairwise factors, write the message from the factor $\phi(a, b)$ to $b$ in terms of other messages, and follow the dependencies.

<details><summary>solution</summary>

$m_{\phi_{ab} \to b} = \sum_a\phi(a, b)\,m_{a \to \phi_{ab}}(a)$, and $m_{a \to \phi_{ab}} = m_{\phi_{ca} \to a}$, which needs $m_{c \to \phi_{ca}} = m_{\phi_{bc} \to c}$, which needs $m_{b \to \phi_{bc}} = m_{\phi_{ab} \to b}$: the message depends on itself. there's no leaf to start from; information about $a$ reaches $b$ along two routes and loopy BP counts it twice.

</details>

## experiment

`lab.py` investigates three things with your code:

1. how the cost of variable elimination on a $6 \times 6$ grid depends on the elimination order.
2. how brute-force inference and variable elimination scale on a chain.
3. the cost of every marginal of a tree, by repeated elimination and by one call to sum-product.

## check your understanding

1. why does variable elimination give *exactly* the brute-force answer, whatever the order?

<details><summary>answer</summary>

it computes the same sum of the same products; it only uses the distributive law to factor out terms that don't depend on the summation variable. no term is dropped or approximated. (floating-point rounding differs slightly between orders.)

</details>

2. why reduce by the evidence *before* eliminating, rather than at the end?

<details><summary>answer</summary>

reducing removes the observed variables from every factor, so the tables get smaller and the observed variables never need eliminating. at the end you'd have carried them through every product for nothing (and they'd have to be among the "query" variables to be reducible at all).

</details>

3. the min-degree order on a tree never creates a table bigger than the original factors. why?

<details><summary>answer</summary>

a tree always has a leaf (degree 1 in the interaction graph, if the factors are pairwise). eliminating a leaf multiplies its factor(s) into a table over the leaf and its single neighbour, adds no fill-in, and leaves a tree. so every step stays at pairwise size.

</details>

4. two variables of a tree are observed. does sum-product still need only $2E$ messages?

<details><summary>answer</summary>

yes: reducing the factors by the evidence only changes their values (and removes the observed variables), not the tree structure. some messages become constant (from fully observed factors), which costs nothing.

</details>

5. what's the relationship between variable elimination on a chain-shaped model and the forward algorithm of an HMM?

<details><summary>answer</summary>

they're the same computation: eliminating $z_1, z_2, \dots$ in order passes the vector $\alpha_t(z_t) = p(x_{1:t}, z_t)$ forward, multiplying by the transition and emission factors at each step. the backward pass of forward–backward is the messages flowing the other way, and together they're sum-product on the chain.

</details>

6. "sum-product" has a sibling, "max-product". what does replacing every sum by a max compute?

<details><summary>answer</summary>

the probability of the most likely joint assignment (and, with back-pointers, the assignment itself): MAP inference. on a chain it's the Viterbi algorithm (next topic). it works because max distributes over products of non-negative numbers just like sum does.

</details>

## common bugs

- **eliminating a variable from only one of its factors.** symptom: wrong answers that look plausible. every factor mentioning the variable has to be multiplied together *before* summing it out.
- **forgetting fill-in when estimating cost.** symptom: `elimination_cost` and `min_degree_order` that think a grid is as cheap as a tree. eliminating a variable connects all its neighbours.
- **normalizing too early or not at all.** symptom: answers that don't sum to 1, or that are off by $P(E = e)$. normalize the final factor (and, for stability, each BP message).
- **running sum-product on a loopy graph.** symptom: infinite recursion or answers that don't match brute force. check for cycles first.
- **dropping observed variables from the output... or not.** symptom: a missing key or an extra key in `sum_product`'s dict. observed variables are omitted.
- **recursion on long chains.** symptom: `RecursionError` on a chain of a few hundred variables (Python's default limit is 1,000 frames, and each message costs two). schedule the messages iteratively, leaves to root and back; the checks include a 600-variable chain.

## going deeper

- Koller & Friedman (2009), *Probabilistic Graphical Models*, MIT Press, ch. 9 (variable elimination, orderings, treewidth) and ch. 10 (clique trees, belief propagation). [mitpress.mit.edu/9780262013192](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)
- Zhang & Poole (1994), "A simple approach to Bayesian network computations", *Proceedings of the Canadian Conference on AI*. variable elimination as presented here.
- Kschischang, Frey & Loeliger (2001), "Factor graphs and the sum-product algorithm", *IEEE Transactions on Information Theory* 47(2). [doi:10.1109/18.910572](https://doi.org/10.1109/18.910572)
- Aji & McEliece (2000), "The generalized distributive law", *IEEE Transactions on Information Theory* 46(2). one algorithm behind FFTs, Viterbi, BP and more. [doi:10.1109/18.825794](https://doi.org/10.1109/18.825794)
- Yedidia, Freeman & Weiss (2003), "Understanding belief propagation and its generalizations", in *Exploring Artificial Intelligence in the New Millennium*, Morgan Kaufmann. why loopy BP works when it does.
- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, §8.4.

## connections

- ← builds on: factors, factor graphs and trees (topic 02), the cost of array operations (ch 00).
- → used in: the forward–backward and Viterbi algorithms for HMMs (topic 04), the conjugate models whose exact posteriors check MCMC and VI (topics 05–07), the motivation for approximate inference when the treewidth is large (topics 05–07), Kalman filtering and smoothing (ch 29 elective), message passing in graph neural networks (ch 20).
