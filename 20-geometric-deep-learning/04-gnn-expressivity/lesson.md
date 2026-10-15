# GNN expressivity

## why this matters

cyclohexane is a ring of six carbons. two cyclopropane molecules are two rings of three. a graph model that only sees atoms and bonds (all carbons, every carbon bonded to two others) receives two inputs that look locally identical: every atom has two neighbours, each of which has two neighbours, and so on forever. a message-passing network will give them *exactly* the same representation, however many layers it has and however long you train it. yet they're different molecules with different properties.

that's not a quirk of one example. there is a precise characterization of which graphs message-passing networks can tell apart, and it comes from a graph-isomorphism heuristic from 1968: the **Weisfeiler–Leman test**. every GNN of topic 03 is at most as powerful as it, and some are strictly weaker: in this topic's lab, a GCN fails to tell 3-regular from 4-regular graphs, a distinction about as basic as graph structure gets, and sits at 50% accuracy where another network gets 100%. that network is **GIN**, designed by asking what an aggregation must look like to match the test. this topic builds the test, builds GIN, and maps the boundary: what's possible within it, and what it costs to go beyond.

## warm-up: predict first

1. two graphs have the same number of nodes and every node in both has exactly 3 neighbours. can a message-passing GNN with no node features tell them apart?
2. one node has neighbours with features $\{a\}$; another has $\{a, a, a\}$. which of sum, mean and max aggregation give them different results?
3. a GNN gives two graphs the same embedding. does that mean they're isomorphic? and if it gives different embeddings, are they certainly not isomorphic?
4. can a message-passing GNN count the triangles in a graph?
5. you give every node a random feature vector. does the GNN become more powerful? what do you lose?

## prerequisites check

- [ ] you can write a message-passing layer with scatter operations and batch graphs (→ topic 03)
- [ ] you know why the sum is the only one of sum/mean/max that sees multiplicities (→ topic 02)
- [ ] you can test permutation invariance with relabelled graphs (→ topic 01)
- [ ] you remember that an MLP with one hidden layer can approximate any continuous function on a compact set (→ ch 14 · topic 05)

## core ideas

### idea 1: colour refinement

two graphs are **isomorphic** if one is a relabelling of the other. deciding this is famously hard in general (no polynomial algorithm is known, though Babai's 2016 algorithm is quasi-polynomial), but a cheap heuristic settles most cases: give every node a **colour**, then repeatedly recolour each node by its own colour together with the *multiset* of its neighbours' colours.

$$c^{(0)}_i = 0, \qquad c^{(t+1)}_i = \operatorname{hash}\Big(c^{(t)}_i,\ \{\!\{\, c^{(t)}_j : j \to i \,\}\!\}\Big),$$

where $\{\!\{\cdot\}\!\}$ is a multiset (order doesn't matter, repeats do). round 1 separates nodes by degree. round 2 separates them by the multiset of their neighbours' degrees. round $t$ encodes, roughly, the shape of the tree of walks of length $t$ starting at each node. that's the **1-dimensional Weisfeiler–Leman algorithm** (1-WL; Weisfeiler & Leman, 1968).

two details make it a correct test rather than a sketch:

- **one palette for all graphs.** the "hash" must assign colour ids consistently across the graphs you compare: the same (colour, neighbour multiset) signature gets the same new id in every graph. refine graphs separately and colour 3 in one graph has nothing to do with colour 3 in the other. an easy way: collect the signatures of all nodes of all graphs, sort the distinct ones, and number them.
- **keep your own colour.** including $c^{(t)}_i$ in the signature makes it obvious that each round's colouring *refines* the previous one (nodes split, never merge). so the partition stabilizes after at most $N$ rounds, and a difference between graphs, once it appears, persists. (a curiosity: when every node starts with the *same* colour, as here, dropping the own colour gives exactly the same partitions in every round, because a node's multiset of neighbour colours at round $t$ already encodes its own colour at round $t - 1$; on paper 5. with node labels, it doesn't: two differently labelled nodes with the same neighbours would merge.)

> **spec · stage 1.** `wl_colors(graphs, n_iters=3)`: `graphs` is a list of `(edge_index, num_nodes)`, `edge_index` `(2, E)` int64 (tensor or array; rows source, target). returns, per graph, an int64 tensor of shape `(n_iters + 1, num_nodes)` whose row $t$ is the colouring after $t$ rounds: row 0 all equal; in round $t + 1$ two nodes (in any of the graphs) get the same colour iff they had the same colour in round $t$ and the same multiset of in-neighbour colours. colour ids are shared across all the graphs. exactly `n_iters` rounds.

> **▶ build: stage 1.** implement `wl_colors` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the WL test, and where it fails

to compare two graphs, refine them together and compare the **colour histograms** (how many nodes have each colour) after every round. if a histogram ever differs, the graphs are certainly not isomorphic: an isomorphism would map nodes to nodes with identical colours. if the histograms agree in every round, the test is **inconclusive**: the graphs *may* be isomorphic. the test is one-sided.

how good is inconclusive? for random graphs, excellent: Babai, Erdős & Selkow (1980) showed that colour refinement identifies almost every large random graph uniquely. but it fails systematically on graphs with a lot of local regularity:

- **regular graphs of the same degree and size.** every node of a $d$-regular graph sees $d$ neighbours of its own colour in every round, so nothing ever splits. a 6-cycle and two triangles are both 2-regular on 6 nodes: 1-WL can't tell them apart.
- **decalin and bicyclopentyl**, two bicyclic molecules (two fused hexagons; two pentagons joined by a bond) with ten carbons and eleven bonds each, the textbook pair.
- anything that requires **counting cycles**: since $C_6$ and $C_3 + C_3$ are indistinguishable, no function computed by colour refinement can count triangles.

> **spec · stage 2.** `wl_test(graph1, graph2, *, n_iters=3)`: `False` if the graphs have different numbers of nodes or 1-WL separates them within `n_iters` rounds (their colour histograms differ in some round $\le$ `n_iters`); otherwise `True` ("indistinguishable", not "isomorphic"). this is `mlfs.gnn.wl_test`.

> **▶ build: stage 2.** implement `wl_test`.

> **🔮 predict:** the path on 6 nodes and a 4-cycle plus a separate edge have the same degree sequence $(1, 1, 2, 2, 2, 2)$. after how many rounds does 1-WL tell them apart?

### idea 3: GNNs are at most as powerful as WL, and GIN reaches it

compare colour refinement with a message-passing layer: $c_i \leftarrow \operatorname{hash}(c_i, \{\!\{c_j\}\!\})$ versus $h_i \leftarrow \phi(h_i, \bigoplus_j \psi(h_j))$. a GNN layer is colour refinement with a *learned, continuous* hash. that gives the central theorem (Xu et al., 2019; Morris et al., 2019):

1. **upper bound.** if two nodes have the same WL colour after $t$ rounds, then *every* $t$-layer message-passing network gives them the same embedding (when all nodes start with the same features). so if 1-WL can't distinguish two graphs, no GNN with a permutation-invariant readout can (on paper 1). this holds for any weights, any training, any width.
2. **matching it.** a GNN is exactly as powerful as 1-WL if its update is **injective** on (own state, multiset of neighbour states): different inputs, different outputs, so no distinction WL makes is lost.

which aggregations are injective on multisets? the mean maps $\{a\}$ and $\{a, a\}$ to the same thing, and the max maps $\{a, b\}$ and $\{a, a, b\}$ to the same thing (topic 02). only the **sum** can be injective: for multisets of bounded size from a countable set, there is a $\phi$ such that $\sum_j \phi(x_j)$ is unique for every multiset (topic 02, on paper 1). so the injective update is "$\phi$, then sum, then another function", and to combine the node's own state with the sum injectively, Xu et al. show that $(1 + \epsilon)\phi(h_i) + \sum_j \phi(h_j)$ works for all but countably many $\epsilon$. collapsing the two $\phi$'s into the next layer's MLP gives the **graph isomorphism network**:

$$h_i' = \operatorname{MLP}\Big((1 + \epsilon)\, h_i + \sum_{j \to i} h_j\Big).$$

the MLP must have a hidden layer (a single linear layer followed by a ReLU can't approximate the injective functions the argument needs); $\epsilon$ can be fixed at 0 or learned. GCN fails the test twice: its mean-like normalized aggregation isn't injective, and its self-loop mixes $h_i$ into the same average as its neighbours. GAT's softmax weights make it a weighted mean, which isn't injective either.

> **spec · stage 3.** `GINConv(in_dim, out_dim, *, eps=0.0, train_eps=False)`: `forward(x, edge_index)` → $\operatorname{MLP}((1 + \epsilon)x_i + \sum_{j \to i} x_j)$, `(N, out_dim)`, the MLP `Linear(in_dim, out_dim) → ReLU → Linear(out_dim, out_dim)`, the sum over incoming edges with no normalization; $\epsilon$ is one learnable scalar parameter if `train_eps`, otherwise a constant (not a parameter). this is `mlfs.gnn.GINConv`.

> **▶ build: stage 3.** implement `GINConv`.

### idea 4: graph-level GIN, and beyond WL

for whole-graph tasks, Xu et al. read out the graph after *every* layer, not just the last: early layers capture local structure (degrees), later layers larger patterns, and a graph's class may depend on any of them. with sum readouts, $h_G = \operatorname{concat}\big(\sum_i h_i^{(0)}, \sum_i h_i^{(1)}, \dots, \sum_i h_i^{(L)}\big)$, followed by a linear layer. the sum readout matters for the same reason as the sum aggregation: a mean readout can't tell a graph from two copies of it.

the result is a network whose graph embeddings separate exactly the pairs that $L$ rounds of 1-WL separate (for suitable weights), and give *identical* embeddings to pairs that 1-WL can't separate, for all weights. that last property is testable to rounding error, and the stage 4 checks test it.

if your task needs more than 1-WL (counting triangles, telling one big ring from two small ones), the options all pay for it somehow:

- **random node features** (Sato et al., 2021; Abboud et al., 2021): append a few random numbers to every node's features. nodes become distinguishable, so the network can, for example, notice that the node two hops away to its left is the same node as two hops to its right. the price: the output is now random, and invariance holds only in distribution, not for each forward pass.
- **structural features**: precompute counts of triangles, cycles or other substructures per node and feed them in (cheap and effective when you know what matters).
- **higher-order networks** that pass messages between pairs or triples of nodes, matching the $k$-WL hierarchy, at $O(N^k)$ cost.
- **geometry**: if the nodes have coordinates, distances break most symmetries. topic 05 builds networks on them.

> **spec · stage 4.** `GIN(in_dim, hidden_dim, out_dim, *, n_layers=3)`: `n_layers` `GINConv` layers (`in_dim → hidden_dim → … → hidden_dim`, ReLU between them); `forward(x, edge_index, batch, num_graphs=None)` concatenates the per-graph sum of the input features and of every layer's output, and maps it linearly to `(num_graphs, out_dim)`. exactly invariant to relabelling; graphs in a batch don't interact. name the layers `self.convs` (an `nn.ModuleList` of `GINConv`) and the final linear map `self.head` (input width `in_dim + n_layers * hidden_dim`, readouts in layer order): the checks swap in stub layers to verify the readout layout and the ReLUs.

> **▶ build: stage 4.** implement `GIN`.

> **🔮 predict:** a GCN with a *sum* readout classifies 3-regular versus 4-regular graphs on 12 nodes (all node features equal). what accuracy?

## worked example

run 1-WL on the path $P_6$ (nodes $0 - 1 - 2 - 3 - 4 - 5$) and on $C_4 + K_2$ (a 4-cycle $0 - 1 - 2 - 3 - 0$ plus an edge $4 - 5$), jointly.

- **round 0:** every node colour $a$. histograms: 6 × $a$ in both.
- **round 1:** signatures are $(a, \{\!\{a\}\!\})$ for degree-1 nodes and $(a, \{\!\{a, a\}\!\})$ for degree-2 nodes. number them $b$ and $c$. $P_6$: ends $0, 5$ get $b$, the rest $c$. $C_4 + K_2$: nodes $4, 5$ get $b$, the cycle $c$. histograms: 2 × $b$, 4 × $c$ in both. still indistinguishable.
- **round 2:** in $P_6$, the ends have signature $(b, \{\!\{c\}\!\})$, nodes 1 and 4 have $(c, \{\!\{b, c\}\!\})$, nodes 2 and 3 have $(c, \{\!\{c, c\}\!\})$. in $C_4 + K_2$, nodes 4 and 5 have $(b, \{\!\{b\}\!\})$ and the cycle nodes $(c, \{\!\{c, c\}\!\})$. the signature $(b, \{\!\{c\}\!\})$ occurs in $P_6$ but not in $C_4 + K_2$: the histograms differ, and `wl_test(..., n_iters=1)` is `True` while `wl_test(..., n_iters=2)` is `False`. both are stage 2 checks.

and the GCN failure from the lab, by hand: on a $d$-regular graph every node has $\tilde d = d + 1$ (with its self-loop), so every entry of $\hat A$ on an edge or self-loop is $1/(d + 1)$ and every row of $\hat A$ sums to $(d + 1)/(d + 1) = 1$. if all nodes carry the same $x$, then $\hat A x = x$ exactly, whatever $d$ is: a GCN can't even see the degree. GIN computes $(1 + \epsilon)x + d\,x$, which can.

## on paper

1. prove the upper bound: suppose all nodes of graphs $G$ and $H$ start with the same feature vector, and $h^{(t)}$ is computed by $t$ message-passing layers $h_i^{(s+1)} = \phi_s(h_i^{(s)}, \bigoplus_{j \to i}\psi_s(h_j^{(s)}))$. show by induction that $c_i^{(t)} = c_k^{(t)} \Rightarrow h_i^{(t)} = h_k^{(t)}$ for any nodes $i, k$ of $G$ or $H$. conclude that if 1-WL can't distinguish $G$ and $H$, a GNN with a sum (or mean, or max) readout gives them the same embedding.

<details><summary>solution</summary>

$t = 0$: all colours and all features are equal. step: if $c_i^{(t+1)} = c_k^{(t+1)}$, then by the definition of refinement $c_i^{(t)} = c_k^{(t)}$ and the multisets $\{\!\{c_j^{(t)} : j \to i\}\!\}$ and $\{\!\{c_l^{(t)} : l \to k\}\!\}$ are equal. by the induction hypothesis, equal colours mean equal features, so $h_i^{(t)} = h_k^{(t)}$ and the multisets of neighbour features are equal; then the messages, their aggregate (any multiset function) and $\phi_t$'s output are equal. if 1-WL can't distinguish the graphs, their colour histograms agree in every round, so there's a colour-preserving bijection between their nodes, hence between their final features, and any multiset readout gives the same result.

</details>

2. show that GCN's update $h_i' = \sum_{j \in \mathcal N(i) \cup \{i\}} h_j / \sqrt{\tilde d_i \tilde d_j}$ (with $W = I$) is not injective on (own state, neighbour multiset): find two nodes in different graphs with different WL colours after one round but equal GCN outputs.

<details><summary>solution</summary>

any two regular graphs of different degrees with constant features: in a $d$-regular graph $h_i' = \sum_{j} x/(d + 1) = x$ for every $d$ (the worked example). the nodes have different degrees, so different colours after one round, but identical outputs. the symmetric normalization turns the sum into something mean-like, which forgets the multiset's size.

</details>

3. with node colours from a finite palette $\{1, \dots, K\}$ and at most $m$ neighbours, give an explicit $\phi$ that makes $\sum_j \phi(c_j)$ injective on neighbour multisets. how does a GIN layer approximate this with continuous features?

<details><summary>solution</summary>

$\phi(c) = e_c$, the one-hot vector of colour $c$: the sum is the vector of colour counts, which *is* the multiset. (a scalar version: $\phi(c) = (m+1)^{c}$, counts as digits in base $m + 1$.) GIN's input features play the role of $\phi$ for the first layer, and each layer's MLP must produce features for the next layer whose sum stays injective; by universal approximation an MLP can approximate such a $\phi$ on the finite set of feature values that actually occur.

</details>

4. one round of 1-WL on a graph with $N$ nodes and $E$ edges: how much work does it take? why is the full test (until stable) at most $N$ rounds?

<details><summary>solution</summary>

building every node's signature reads each edge once, $O(N + E)$, sorting each neighbour multiset costs $O(E \log \Delta)$ in total, and numbering the distinct signatures (by sorting or hashing) $O(N \log N)$. so roughly $O((N + E)\log N)$ per round. each round either splits at least one colour class or changes nothing (and then nothing will ever change again, since the update depends only on the partition); a partition of $N$ nodes can be split at most $N - 1$ times.

</details>

5. colour refinement *without* the own colour: $\tilde c^{(t+1)}_i = \operatorname{hash}\{\!\{\tilde c^{(t)}_j : j \to i\}\!\}$, from uniform colours $\tilde c^{(0)} = c^{(0)}$. show by induction that $\tilde c^{(t)}$ and $c^{(t)}$ induce the same partition for every $t$, for directed graphs too. why does the argument break when the initial colours are node labels?

<details><summary>solution</summary>

it's enough to show that $\tilde c^{(t)}_i$ determines $c^{(t)}_i$ (the converse is clear: $c^{(t)}$ contains $\tilde c^{(t)}$'s information plus more). induction hypothesis: $\tilde c^{(t)}$ and $c^{(t)}$ have the same partition. then $\tilde c^{(t+1)}_i$ is (equivalent to) the multiset $\{\!\{c^{(t)}_j\}\!\}$ over in-neighbours. each $c^{(t)}_j$ contains $c^{(t-1)}_j$, so this multiset determines $\{\!\{c^{(t-1)}_j\}\!\}$, which together with $c^{(t-1)}_i$ gives $c^{(t)}_i$. and $c^{(t-1)}_i$ is in turn determined by $\{\!\{c^{(t-2)}_j\}\!\}$ and $c^{(t-2)}_i$, and so on down to $c^{(0)}_i$, which is the same constant for every node. so the in-neighbour multiset determines the node's whole colour history, including $c^{(t)}_i$, and $c^{(t+1)}_i = (c^{(t)}_i, \{\!\{c^{(t)}_j\}\!\})$ follows. the last step uses that $c^{(0)}$ is uniform: with labels, two nodes with different labels and the same in-neighbours get the same $\tilde c$ forever.

</details>

## experiment

the lab runs your `wl_test` and `GIN` against your GCN from topic 03 on graphs with no node features. you'll compare graph embeddings on five pairs of non-isomorphic graphs (and count their triangles), train GCNs with mean and sum readouts and GIN to tell 3-regular from 4-regular graphs, and see whether random node features let GIN tell one long cycle from two short ones, and what they cost. predict first.

## check your understanding

1. `wl_test` returns `True`. what can you conclude about the two graphs?

<details><summary>answer</summary>

only that 1-WL can't tell them apart within `n_iters` rounds. they may be isomorphic (the test never says `False` for isomorphic graphs) or not ($C_6$ and $C_3 + C_3$). `False`, on the other hand, is a proof of non-isomorphism.

</details>

2. why must colour refinement compare graphs with a shared palette?

<details><summary>answer</summary>

colour ids are only meaningful relative to the signatures that produced them. refined separately, two graphs could both number their nodes' colours $0, 1, 2$ with completely different meanings, and equal histograms would mean nothing (or different numbering of the same signatures would make isomorphic graphs look different).

</details>

3. a GIN layer uses a mean instead of a sum. which pairs from this topic can it no longer distinguish?

<details><summary>answer</summary>

any where the difference is in multiplicities, not proportions: a node with neighbours $\{a\}$ versus $\{a, a\}$; with constant features, regular graphs of different degrees (the mean of $d$ copies of $x$ is $x$). with the mean, GIN becomes as weak as GCN on the lab's experiment 2.

</details>

4. why does a 3-layer GNN give the same embedding to $C_{12}$ and $C_4 + C_4 + C_4$, even though it only sees 3 hops and both look like paths at that range?

<details><summary>answer</summary>

that's exactly why: within 3 hops every node of either graph sees a path of 7 nodes centred on itself, the same rooted tree. 1-WL (and a GNN) only ever sees these local unrollings, and both graphs are 2-regular, so they agree at every depth. even with 100 layers they'd agree.

</details>

5. GAT can weight neighbours differently. does that make it more expressive than GIN?

<details><summary>answer</summary>

no. the attention weights are a softmax, so GAT's aggregation is a weighted *mean*, which can't count. it's bounded by 1-WL like every message-passing network, and strictly weaker than GIN on multiplicity distinctions (identical neighbours get identical weights, and the mean of identical vectors forgets how many there were).

</details>

6. random node features make GIN able to distinguish $C_6$ from $C_3 + C_3$. why doesn't everyone use them?

<details><summary>answer</summary>

the network's output becomes a random variable: the same molecule gets different predictions on different forward passes (the lab measures how much), and permutation invariance holds only in distribution. you need averaging over many draws at test time, training gets noisier, and generalization to larger graphs suffers. structural features or geometry give the extra power deterministically.

</details>

## common bugs

- **refining each graph with its own palette**: symptom: isomorphic graphs come out as distinguishable (or the shared-palette check fails). collect all graphs' signatures before numbering them.
- **neighbour colours as a set instead of a multiset**: symptom: the path's ends and middles merge, or star-versus-path pairs pass. sort the list of neighbour colours; don't deduplicate it.
- **dropping the node's own colour from the signature**: symptom: none here, surprisingly (on paper 5), but the same code silently merges differently labelled nodes as soon as you start from node labels. keep it.
- **a "while not stable" loop without a cap**: symptom: the check hangs, then fails with a "didn't finish" message. run exactly `n_iters` rounds.
- **GIN with a mean or a normalized sum**: symptom: the multiplicity check and the regular-graph check fail. the sum is unnormalized: `index_add` of the raw neighbour features.
- **eps as a `Parameter` when `train_eps=False`**: symptom: the parameter count is off by one, and an optimizer moves an $\epsilon$ that should be fixed. use `register_buffer`.

## going deeper

- Xu, Hu, Leskovec & Jegelka (2019), "How powerful are graph neural networks?", *ICLR*. the WL bound and GIN. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Morris, Ritzert, Fey, Hamilton, Lenssen, Rattan & Grohe (2019), "Weisfeiler and Leman go neural: higher-order graph neural networks", *AAAI*. the same bound, and $k$-WL networks. [arXiv:1810.02244](https://arxiv.org/abs/1810.02244)
- Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt (2011), "Weisfeiler–Lehman graph kernels", *JMLR* 12: 2539–2561. WL as a feature extractor, before GNNs. [jmlr.org/papers/v12/shervashidze11a](https://jmlr.org/papers/v12/shervashidze11a.html)
- Babai, Erdős & Selkow (1980), "Random graph isomorphism", *SIAM Journal on Computing* 9(3): 628–635. colour refinement works for almost all graphs. [doi:10.1137/0209047](https://doi.org/10.1137/0209047)
- Chen, Chen, Villar & Bruna (2020), "Can graph neural networks count substructures?", *NeurIPS*. [arXiv:2002.04025](https://arxiv.org/abs/2002.04025)
- Sato, Yamada & Kashima (2021), "Random features strengthen graph neural networks", *SDM*. [arXiv:2002.03155](https://arxiv.org/abs/2002.03155)

## connections

- ← builds on: multiset functions and why sums are injective (topic 02), message passing, GCN and batching (topic 03), invariance (topic 01).
- → used in: topic 05 escapes the WL bound with geometry (distances separate graphs that look identical combinatorially); the capstone compares GIN with GCN and EGNN on molecules; the same "what can this architecture distinguish" question returns for transformers and positional encodings (ch 18).
