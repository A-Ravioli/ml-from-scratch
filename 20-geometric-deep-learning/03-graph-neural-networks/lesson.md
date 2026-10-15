# graph neural networks

## why this matters

a bank wants to flag fraudulent accounts. each account has features (age, balance, activity), and accounts are linked by transfers. fraudsters cluster: an account that sends money mostly to flagged accounts is suspicious even if its own features look normal. a model that sees each account in isolation can't use that; it needs the **graph**. the same is true of a protein's residues and their contacts, a paper and its citations, a molecule's atoms and bonds, a road network's junctions.

a graph has no natural order of its nodes. number the accounts differently and you have the same graph, so a model's per-node predictions should be relabelled along with the nodes (permutation equivariance), and its per-graph predictions shouldn't change at all (invariance). nodes also have different numbers of neighbours, from one to thousands. an MLP on the adjacency matrix fails both requirements: it depends on the numbering and on the graph's size.

graph neural networks meet both by generalizing the set networks of topic 02: each node collects **messages** from its neighbours, aggregates them with a symmetric operation, and updates its own state. stack a few such layers and each node's representation summarizes its neighbourhood a few hops out. this topic builds the engine (scatter operations over an edge list), three classic layers (GCN, GraphSAGE, GAT), and graph-level readouts. the lab then finds the two ways GNNs famously fail: too many layers, and neighbours that aren't like you.

## warm-up: predict first

1. a GCN layer averages each node's features with its neighbours'. apply it (without weights) 100 times to a connected graph. what do the node features converge to?
2. a node has 1,000 neighbours and another has 2. if a layer **sums** neighbour messages, what goes wrong? what if it takes the **mean**?
3. in a graph stored as an edge list `(source, target)`, a layer computes `out[target] += message[source]`. for an undirected graph, how must the edge list be stored so that information flows both ways?
4. a GCN with 2 layers: how many hops away can information come from? and with 32 layers, does more information mean better predictions?
5. on a dating graph, most edges connect people of *different* types. does averaging your neighbours' features help or hurt when predicting your type?

## prerequisites check

- [ ] you can test permutation equivariance with `equivariance_error` (→ topic 01)
- [ ] you know DeepSets' sum/mean/max pooling and the masked attention softmax (→ topic 02)
- [ ] you can write a matrix's eigendecomposition and know what repeated multiplication by a matrix does to a vector (→ ch 02 · topic 05, the power method)
- [ ] you can train a PyTorch module with Adam (→ ch 15 · topic 07)
- [ ] you remember why deep plain networks are hard to train (→ ch 15 · topic 01) and what residual connections do (→ ch 16)

## core ideas

### idea 1: message passing on an edge list

a graph with $N$ nodes and $E$ directed edges is stored as two tensors: node features `x` of shape `(N, F)`, and `edge_index` of shape `(2, E)`, where column $e$ is the edge from `edge_index[0, e]` (the **source**) to `edge_index[1, e]` (the **target**). messages flow from source to target. an undirected edge is stored as two directed edges, one each way. this is the convention of every modern graph library, and it scales: storage is $O(N + E)$, not $O(N^2)$ like an adjacency matrix.

a **message-passing layer** (Gilmer et al., 2017) computes, for every node $i$,

$$x_i' = \phi\Big(x_i,\ \bigoplus_{j \to i} \psi(x_i, x_j)\Big),$$

where $\psi$ computes a message along each edge, $\bigoplus$ is a symmetric aggregation over $i$'s incoming edges (sum, mean or max: exactly DeepSets' choices, applied to each node's set of neighbours), and $\phi$ updates the node. because $\psi$ and $\phi$ are shared by all nodes and edges and $\bigoplus$ doesn't care about order, the layer is permutation-equivariant (on paper 1).

in code, the aggregation is a **scatter**: compute one message per edge, a tensor of shape `(E, F)`, then add each row into the row of its target node. `index_add`, `scatter_reduce` and `bincount` do this without a Python loop, and they're differentiable. three variants cover everything in this chapter:

- `scatter_sum(src, index, dim_size)`: $\text{out}[n] = \sum_{e:\ \text{index}[e] = n} \text{src}[e]$.
- `scatter_mean`: the same divided by the group's size.
- `scatter_max`: the elementwise maximum over the group.

nodes that receive no messages (no incoming edges) get 0 from all three, by convention. for the max that needs care: start from $-\infty$ (not 0, or a group of negative values would return 0) and then replace empty groups.

> **spec · stage 1.** `scatter_sum(src, index, dim_size)`, `scatter_mean(...)`, `scatter_max(...)`: reduce the rows of `src` `(E, ...)` into `(dim_size, ...)` by the int64 `index` `(E,)`; empty groups give 0; differentiable; no Python loops over edges or nodes (the checks time you against `index_add_`); `src` is not modified. these are `mlfs.gnn`'s scatter operations.

> **▶ build: stage 1.** implement the three scatters in `exercise.py` (`marimo edit exercise.py`).

### idea 2: GCN, normalized neighbourhood averaging

the graph convolutional network of Kipf & Welling (2017) is the simplest useful message-passing layer. in matrix form, with $A$ the adjacency matrix ($A_{ij} = 1$ for an edge $j \to i$):

$$X' = \hat A X W + b, \qquad \hat A = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}, \qquad \tilde D_{ii} = \textstyle\sum_j (A + I)_{ij}.$$

read it right to left. $XW$ transforms every node's features with one shared weight matrix. $A + I$ adds a **self-loop** to every node, so a node's own features are part of its update (without it, $x_i$ would be replaced by its neighbours' average and lose itself). the **symmetric normalization** divides the message along edge $j \to i$ by $\sqrt{\tilde d_i \tilde d_j}$: high-degree nodes don't receive sums that grow with their degree, and high-degree senders don't dominate. as messages:

$$x_i' = \sum_{j \in \mathcal N(i) \cup \{i\}} \frac{1}{\sqrt{\tilde d_i\,\tilde d_j}}\, W^\top x_j + b,$$

with $\tilde d$ counting incoming edges *including* the self-loop. (the original derivation is a first-order approximation of a spectral filter on the graph Laplacian; the message form is all you need.)

the normalization has a consequence that you'll measure in the lab. $\hat A$ is symmetric with eigenvalues in $(-1, 1]$, and on a connected graph its top eigenvalue is exactly 1 with eigenvector $\tilde D^{1/2}\mathbf 1$ (on paper 2). so repeated propagation, $\hat A^k X$, converges to the projection of $X$ onto that one direction: every feature becomes proportional to $\sqrt{\tilde d_i}$, and nodes become indistinguishable except by their degree. this is **oversmoothing**, and it's why a GCN stack that's too deep loses the information it was built to collect. how fast depends on the gap between the top eigenvalue and the next ones.

> **spec · stage 2.**
> - `add_self_loops(edge_index, num_nodes)`: `edge_index` with columns $(i, i)$ appended for $i = 0, \dots, N - 1$ (inputs have no self-loops).
> - `degree(index, num_nodes, dtype=torch.float32)`: how many times each node occurs in `index`.
> - `GCNConv(in_dim, out_dim)`: `forward(x, edge_index)` → $\hat A X W + b$ as above, `(N, out_dim)`, with $W$ a weight without bias and $b$ a learnable bias added after aggregation (so exactly `in_dim * out_dim + out_dim` parameters). this is `mlfs.gnn.GCNConv`.

> **▶ build: stage 2.** implement `add_self_loops`, `degree` and `GCNConv`.

> **🔮 predict:** on a connected graph, after $\hat A$ is applied 256 times, feature 0 of each node is a function of one property of the node. which one?

### idea 3: GraphSAGE, a separate weight for yourself

GCN treats a node's own features as just one more neighbour (the self-loop), averaged in with everyone else's. GraphSAGE (Hamilton et al., 2017) keeps them apart:

$$x_i' = W_1^\top x_i + W_2^\top \operatorname*{mean}_{j \to i} x_j + b.$$

two weight matrices: one for "me", one for "my neighbourhood". the model can now learn how much to trust the neighbours, including not at all, or "the opposite of my neighbours". a node with no incoming edges just gets $W_1^\top x_i + b$. GraphSAGE was designed for **inductive** learning on large graphs: the same weights apply to nodes and graphs never seen in training, and the paper samples a fixed number of neighbours per node to bound the cost (we use all of them). the mean aggregator makes a node's update independent of its degree; the paper's other choices, max and an LSTM over a random neighbour order, trade that off differently.

> **spec · stage 3.** `SAGEConv(in_dim, out_dim)`: `forward(x, edge_index)` → $x_i W_1 + \operatorname{mean}_{j \to i} x_j\, W_2 + b$ (`scatter_mean` over incoming edges, no self-loops; one bias), `(N, out_dim)`. this is `mlfs.gnn.SAGEConv`.

> **▶ build: stage 3.** implement `SAGEConv`.

### idea 4: GAT, attention over neighbours

GCN's weights $1/\sqrt{\tilde d_i \tilde d_j}$ depend only on the graph's structure. the graph attention network (Veličković et al., 2018) **learns** how much each neighbour matters, from the features. with $h_j = W^\top x_j$ and learned vectors $a_\text{src}, a_\text{dst}$:

$$e_{ij} = \operatorname{LeakyReLU}_{0.2}\big(a_\text{dst}^\top h_i + a_\text{src}^\top h_j\big), \qquad \alpha_{ij} = \frac{\exp e_{ij}}{\sum_{k \in \mathcal N(i)\cup\{i\}}\exp e_{ik}}, \qquad x_i' = \sum_{j \in \mathcal N(i)\cup\{i\}} \alpha_{ij} h_j + b.$$

the softmax runs over the edges *into* $i$ (self-loop included), which is a **scatter softmax**: subtract each group's maximum (for stability, as in any softmax), exponentiate, and divide by the group's scatter-sum. it's topic 02's attention pooling, applied to every node's neighbourhood, with an additive scoring function instead of a dot product. **multi-head** attention runs $H$ independent copies (each with its own $W$ slice and attention vectors) and concatenates their outputs (hidden layers) or averages them (an output layer). **dropout** on the attention coefficients during training randomly hides neighbours, a strong regularizer on small graphs.

notice what $\alpha$ is: a set of non-negative weights summing to 1 over the neighbourhood. so a GAT layer's output is a *weighted mean* of the transformed neighbours. it can focus on the relevant ones, but, like every mean, it can't count them. topic 04 shows why that matters.

> **spec · stage 4.**
> - `scatter_softmax(src, index, dim_size)`: the softmax of `src` `(E, ...)` within each group of equal `index`, numerically stable (adding a constant to all scores changes nothing).
> - `GATConv(in_dim, out_dim, *, heads=1, concat=True, dropout=0.0)`: `forward(x, edge_index)` → the layer above with self-loops added, $H$ heads, output `(N, heads * out_dim)` if `concat` else the mean over heads `(N, out_dim)`, plus a bias; dropout with probability `dropout` on the coefficients $\alpha$ in training mode only. so that the checks can compare your scores with the formula, use PyTorch Geometric's parameter names: `lin` (an `nn.Linear(in_dim, heads * out_dim, bias=False)`; head $k$ is output columns $k \cdot$ `out_dim` to $(k+1) \cdot$ `out_dim` $- 1$), `att_src` and `att_dst` (`(heads, out_dim)` parameters), and `bias`. this is `mlfs.gnn.GATConv`.

> **▶ build: stage 4.** implement `scatter_softmax` and `GATConv`.

> **🔮 predict:** on a graph where most of each node's neighbours belong to a *different* class, which layer do you expect to do best: GCN, GraphSAGE or GAT?

### idea 5: graph-level readouts and batching

to predict a property of a whole graph (a molecule's energy, a protein's function), follow the GDL blueprint: message-passing layers, then a permutation-invariant **readout** over the nodes: `global_add_pool` (the sum of node features per graph) or `global_mean_pool`. which one matters, for the same reasons as in topic 02: the sum sees the graph's size, the mean doesn't.

training on many small graphs needs batching, and the edge-list format makes it elegant: a batch is one big graph made of disjoint copies. concatenate the node features, offset each graph's edge indices by the number of nodes before it (so graph 2's node 0 becomes node $N_0 + N_1$), and record each node's graph id in a `batch` vector. message passing never crosses between graphs because no edge does, and the readout is a scatter over `batch`. no padding, no masks.

> **spec · stage 5.**
> - `global_add_pool(x, batch, num_graphs=None)`, `global_mean_pool(...)`: `(N, F)` → `(num_graphs, F)`, summing or averaging the nodes of each graph; `num_graphs=None` means `batch.max() + 1`; graphs with no nodes give 0. these are `mlfs.gnn`'s pools.
> - `batch_graphs(graphs)`: a list of `(x, edge_index)` → `(x, edge_index, batch)` for the disjoint union, node ids offset per graph, `batch` int64.

> **▶ build: stage 5.** implement the pools and `batch_graphs`.

## worked example

take the path $0 - 1 - 2$, stored as the four directed edges `[[0, 1, 1, 2], [1, 0, 2, 1]]`. with self-loops the degrees are $\tilde d = (2, 3, 2)$, and

$$\hat A = \begin{pmatrix} \frac12 & \frac{1}{\sqrt6} & 0 \\ \frac{1}{\sqrt6} & \frac13 & \frac1{\sqrt6} \\ 0 & \frac1{\sqrt6} & \frac12 \end{pmatrix} \approx \begin{pmatrix} 0.5 & 0.408 & 0 \\ 0.408 & 0.333 & 0.408 \\ 0 & 0.408 & 0.5 \end{pmatrix}.$$

a one-feature GCN with $W = 1$, $b = 0$ maps $x = (1, 0, 0)$ to $\hat A x = (0.5, 0.408, 0)$: the end node keeps half its value, its neighbour receives $1/\sqrt{2 \cdot 3}$, and node 2, two hops away, receives nothing yet. one more layer gives $\hat A^2 x = (0.417, 0.340, 0.167)$: information travels one hop per layer. the rows of $\hat A$ don't sum to 1 (row 1 sums to 1.15): the symmetric normalization is not an average, which is why the fixed point is $\sqrt{\tilde d}$ and not a constant. the stage 2 checks recover your layer's $\hat A$ on this path and compare it with the matrix above.

and a scatter by hand: messages `[[1, 2], [3, 4], [5, 6]]` to targets `[0, 2, 0]` with 4 nodes give the sum `[[6, 8], [0, 0], [3, 4], [0, 0]]`, the mean `[[3, 4], [0, 0], [3, 4], [0, 0]]` and the max `[[5, 6], [0, 0], [3, 4], [0, 0]]`.

## on paper

1. show that a message-passing layer $x_i' = \phi(x_i, \bigoplus_{j \to i}\psi(x_i, x_j))$ is permutation-equivariant: relabel the nodes by a permutation $\pi$ (node $i$ of the new graph is node $\pi(i)$ of the old one, and edges are relabelled accordingly). show that the new output at $i$ is the old output at $\pi(i)$.

<details><summary>solution</summary>

in the relabelled graph, node $i$ has features $x_{\pi(i)}$, and $j \to i$ is an edge iff $\pi(j) \to \pi(i)$ was one. so the multiset of messages arriving at new node $i$ is $\{\psi(x_{\pi(i)}, x_{\pi(j)}) : \pi(j) \to \pi(i)\} = \{\psi(x_{\pi(i)}, x_k) : k \to \pi(i)\}$, exactly the messages old node $\pi(i)$ received. $\bigoplus$ depends only on the multiset, and $\phi$ is the same function everywhere, so the new $x_i'$ equals the old $x_{\pi(i)}'$. the proof needs the aggregation to be order-independent and the functions to be shared, nothing else.

</details>

2. let $\hat A = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}$ for an undirected graph. show that $v = \tilde D^{1/2}\mathbf 1$ is an eigenvector with eigenvalue 1. using that the random-walk matrix $P = \tilde D^{-1}(A + I)$ is similar to $\hat A$, argue that all eigenvalues of $\hat A$ lie in $[-1, 1]$, and conclude what $\hat A^k X$ converges to on a connected graph.

<details><summary>solution</summary>

$\hat A v = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}\tilde D^{1/2}\mathbf 1 = \tilde D^{-1/2}(A + I)\mathbf 1 = \tilde D^{-1/2}\tilde d = \tilde D^{1/2}\mathbf 1 = v$, since $(A + I)\mathbf 1$ is the vector of degrees. $P = \tilde D^{-1/2}\hat A\tilde D^{1/2}$ is similar to $\hat A$ and is a stochastic matrix (non-negative rows summing to 1), so its eigenvalues have $|\lambda| \le 1$; $\hat A$ has the same eigenvalues, and being symmetric, they're real. on a connected graph the eigenvalue 1 is simple (Perron–Frobenius), and the self-loops make the graph aperiodic so $-1$ isn't an eigenvalue. writing $X$ in $\hat A$'s orthonormal eigenbasis, $\hat A^k X = \sum_m \lambda_m^k u_m u_m^\top X \to u_1 u_1^\top X$ with $u_1 = v/\|v\|$: every column of $X$ collapses onto $\sqrt{\tilde d}$, at the rate $|\lambda_2|^k$.

</details>

3. what does one GCN layer cost in time, in terms of $N$, $E$, $F_\text{in}$ and $F_\text{out}$? compare with multiplying by a dense $N \times N$ matrix.

<details><summary>solution</summary>

the transform $XW$ costs $O(NF_\text{in}F_\text{out})$; gathering, scaling and scattering one message per edge (plus $N$ self-loops) costs $O((E + N)F_\text{out})$. total $O(NF_\text{in}F_\text{out} + EF_\text{out})$. the dense product $\hat A (XW)$ costs $O(N^2F_\text{out})$, which for a sparse graph ($E \ll N^2$, e.g. a social network with $N = 10^6$ and average degree 100) is thousands of times more, and the matrix wouldn't fit in memory. (doing $W$ first matters too when $F_\text{out} < F_\text{in}$: transform, then propagate.)

</details>

4. a GAT head has $e_{ij} = \operatorname{LeakyReLU}(a_\text{dst}^\top h_i + a_\text{src}^\top h_j)$. show that without the LeakyReLU, the ranking of $i$'s neighbours by attention is the same for every node $i$. (this is the "static attention" observation of Brody et al., 2022, and it partly holds even with the LeakyReLU.)

<details><summary>solution</summary>

without the nonlinearity, $e_{ij} = c_i + s_j$ with $c_i = a_\text{dst}^\top h_i$ and $s_j = a_\text{src}^\top h_j$. inside $i$'s softmax, $c_i$ is a constant that cancels: $\alpha_{ij} = e^{s_j}/\sum_k e^{s_k}$. every node ranks its neighbours by the same score $s_j$; only which neighbours are present differs. the LeakyReLU is monotone, so it only changes things when $c_i + s_j$ changes sign across neighbours, which is why the ranking is still "mostly static". GATv2 applies the nonlinearity before the dot product with $a$ to fix this.

</details>

## experiment

the lab trains your layers on stochastic block models: graphs with 3 communities whose nodes are more (or less) likely to link inside their community, and node features that carry only a weak hint of the community. you'll watch repeated GCN averaging collapse the features and find what they collapse to; compare a linear classifier on $\hat A^k X$ with trained GCNs of depth 1 to 32 (default and kaiming initialization, and with residual connections) against MLPs of the same depth as controls; and compare MLP, GCN, GraphSAGE and GAT when most neighbours belong to other communities. predict first: the control in experiment 2 changes the story most people tell about deep GCNs.

## check your understanding

1. why does GCN add self-loops instead of just averaging the neighbours?

<details><summary>answer</summary>

without them, $x_i'$ is built only from the neighbours, so the node's own features are lost after one layer (and on a bipartite graph the two sides swap information back and forth forever). the self-loop makes the node one of its own neighbours. GraphSAGE solves the same problem differently, with a separate weight for the node itself.

</details>

2. a node has no incoming edges. what do GCN, SAGE and GAT output for it?

<details><summary>answer</summary>

GCN: with the self-loop its degree is 1, so $W^\top x_i + b$. SAGE: the neighbour mean is 0 by convention, so $W_1^\top x_i + b$. GAT: the softmax is over the self-loop alone, $\alpha_{ii} = 1$, so $W^\top x_i + b$. all three reduce to a per-node linear layer, which is how the checks recover your weights.

</details>

3. why does the scatter max start from $-\infty$, and what would go wrong with GAT's softmax if you didn't subtract the group max?

<details><summary>answer</summary>

starting from 0 would return 0 for any group whose values are all negative, a value that isn't in the group. without subtracting the max, a score of 1,000 overflows `exp` to `inf` and the softmax becomes `nan`; subtracting any per-group constant leaves the softmax unchanged, and the max makes the largest exponent $e^0 = 1$.

</details>

4. you batch 32 molecules into one big graph. why can't a GCN layer mix information between molecules?

<details><summary>answer</summary>

messages only flow along edges, and `batch_graphs` creates no edges between graphs (it offsets each graph's indices into its own range). the normalization only uses degrees, which are per node. the readout scatters by `batch`, so each graph is pooled separately. the batched computation equals running each graph alone.

</details>

5. a 2-layer GCN on a graph with diameter 20: which nodes can influence node $i$'s output?

<details><summary>answer</summary>

only nodes within 2 hops (its receptive field). to see further you need more layers, or a different graph (adding long-range edges, a global "virtual node" connected to everything), and more layers bring oversmoothing and the usual depth problems.

</details>

6. in GraphSAGE, why the mean and not the sum?

<details><summary>answer</summary>

the mean keeps the update's scale independent of the degree, which matters in graphs with heavy-tailed degrees (a node with 10,000 neighbours would otherwise receive a sum 10,000 times larger than a leaf). the price is that the mean can't tell a node with 1 neighbour of type A from one with 5 neighbours of type A, the distinction topic 04 builds a test for.

</details>

## common bugs

- **messages flowing the wrong way**: symptom: directed-graph checks fail, undirected ones pass. gather from `edge_index[0]` (sources), scatter into `edge_index[1]` (targets).
- **degree without the self-loop, or from the wrong end**: symptom: the path example gives $1/\sqrt2$ where $1/\sqrt6$ should be. count degrees at the target, after adding self-loops.
- **normalizing by one endpoint only**: symptom: rows of your $\hat A$ sum to 1 (that's the random-walk normalization $\tilde D^{-1}(A + I)$, a different layer). GCN uses $1/\sqrt{\tilde d_i \tilde d_j}$.
- **the softmax over all edges instead of per target node**: symptom: attention weights tiny everywhere, GAT behaves like a scaled-down GCN. use `scatter_softmax` grouped by the target.
- **forgetting the offsets when batching**: symptom: graph 2's edges point at graph 1's nodes; outputs change with the batch composition.
- **a Python loop over nodes**: symptom: correct, and 1,000× slower than it needs to be. the stage 1 timing check catches it.

## going deeper

- Gilmer, Schoenholz, Riley, Vinyals & Dahl (2017), "Neural message passing for quantum chemistry", *ICML*. the message-passing framework. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
- Kipf & Welling (2017), "Semi-supervised classification with graph convolutional networks", *ICLR*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Hamilton, Ying & Leskovec (2017), "Inductive representation learning on large graphs", *NeurIPS*. GraphSAGE. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
- Veličković, Cucurull, Casanova, Romero, Liò & Bengio (2018), "Graph attention networks", *ICLR*. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Brody, Alon & Yahav (2022), "How attentive are graph attention networks?", *ICLR*. static attention and GATv2. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
- Li, Han & Wu (2018), "Deeper insights into graph convolutional networks for semi-supervised learning", *AAAI*. GCN as Laplacian smoothing, and oversmoothing. [arXiv:1801.07606](https://arxiv.org/abs/1801.07606)
- Cong, Ramezani & Mahdavi (2021), "On provable benefits of depth in training graph convolutional networks", *NeurIPS*. deep GCNs are expressive; their trouble is training and generalization, not just oversmoothing. [arXiv:2110.15174](https://arxiv.org/abs/2110.15174)
- Zhang, Sheng, Yin, Jiang, Xia, Gao, Yang & Cui (2022), "Model degradation hinders deep graph neural networks", *KDD*. propagation depth versus transformation depth. [arXiv:2206.04361](https://arxiv.org/abs/2206.04361)
- Zhu, Yan, Zhao, Heimann, Akoglu & Koutra (2020), "Beyond homophily in graph neural networks: current limitations and effective designs", *NeurIPS*. separating ego and neighbour embeddings. [arXiv:2006.11468](https://arxiv.org/abs/2006.11468)
- Hamilton (2020), *Graph Representation Learning*, Morgan & Claypool, chapters 5–7. [cs.mcgill.ca/~wlh/grl_book](https://www.cs.mcgill.ca/~wlh/grl_book/)

## connections

- ← builds on: set pooling and attention (topic 02), equivariance testing (topic 01), eigenvectors and the power method (ch 02), deep-network trainability (ch 15) and residual connections (ch 16).
- → used in: topic 04 asks which graphs these layers can tell apart and builds GIN on the same kind of sum aggregation; topic 05 adds coordinates to the messages (EGNN); the capstone compares GCN, GIN and EGNN on molecules with these pools; attention over a graph's edges is a transformer with a sparsity pattern (ch 18).
