# 20 · geometric deep learning

> build symmetry into architectures, for sets, graphs, geometry, and continuous depth.

a molecule's energy doesn't change when you rotate it, translate it, reflect it or renumber its atoms. a point cloud's class doesn't depend on the order of its points; a social network's communities don't depend on how its users were numbered. every model so far had to *learn* such facts from data, approximately and expensively. this chapter builds models that have them by construction, exactly, at no extra cost, and finds out when that's a gift and when it's a trap. you'll write a tester that checks symmetries to floating-point precision and catch a "translation-invariant" CNN changing one prediction in twelve when the digit moves by one pixel; build DeepSets and attention pooling, and see learned attention fail to count; build a dependency-free graph library on scatter operations (GCN, GraphSAGE, GAT) and use same-depth MLP controls to tell oversmoothing, bad initialization and graph-specific training failure apart; implement the Weisfeiler–Leman test and GIN, and watch a GCN score exactly 50% on telling 3-regular from 4-regular graphs; build the E(n)-equivariant EGNN, 15–21 times more accurate than an augmented MLP on molecular energies and completely blind to chirality; and finish with neural ODEs, whose adjoint method saves 25,000× the memory and can return gradients wrong by a factor of $3 \times 10^7$. the capstone compares GCN, GIN and EGNN on a molecular benchmark you generate yourself, with confidence intervals and an invariance audit.

**you will build:** `random_rotation`/`random_orthogonal`/`random_permutation`, the tester `equivariance_error`, a circular convolution and group averaging (topic 01); masked pooling, `DeepSets`, the permutation-equivariant layer and pooling by attention (02); `scatter_sum`/`scatter_mean`/`scatter_max`/`scatter_softmax`, `GCNConv`, `SAGEConv`, `GATConv`, graph pooling and batching (03); 1-WL colour refinement, `wl_test`, `GINConv` and a graph-level GIN (04); radius graphs, `EGNNLayer` and an invariant `EGNN` (05); Euler/midpoint/RK4 steps, `odeint`, `odeint_adjoint` and `NeuralODE` (06). modules: **`mlfs.gnn`** (topics 01–05), the graph and equivariance toolkit, and **`mlfs.ode`** (topic 06), whose `odeint` chapter 22's probability-flow sampler integrates with. everything is plain PyTorch: no graph library.
**prerequisites:** chapter 15 (PyTorch, `mlfs.torchnn.MLP`, training loops, deep-network trainability), chapter 02 (orthogonal matrices, QR, eigenvectors, the matrix exponential), chapter 03 (Taylor expansions) and chapter 14 (reverse-mode autodiff, vector-Jacobian products), chapter 05 (`bootstrap_ci`, for the capstone), and `as_rng` (ch 00). convolutions (ch 16) and attention (ch 18) help but the lessons are self-contained.
**time · scaffolding:** about 70 hours · L4 (spec-driven): each stage's exact specification is in its lesson's **spec** box and the checks enforce it; the exercise notebooks give you contract signatures and one-line docstrings, and the internal structure (which layers, which attributes) is yours. one hint per stage. the capstone gives you a brief, the constants and the checks; the code and the experiment are yours.
**newly unlocked library components:** from chapter 21 on you may use PyTorch Geometric and its relatives: `torch_geometric.nn`'s `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, `global_add_pool`/`global_mean_pool`, `radius_graph` and `Batch`/`DataLoader` batching, `torch_scatter`'s scatter operations, the WL kernel (`torch_geometric.nn.WLConv`) and set-pooling layers (03–04, 02); EGNN and similar $E(n)$-equivariant layers from libraries such as `egnn-pytorch` (05); and `torchdiffeq`'s `odeint`/`odeint_adjoint` with the fixed-grid methods `euler`, `midpoint` and `rk4` (06). adaptive-step solvers (e.g. `dopri5`) and spherical-harmonic networks (`e3nn`) are not unlocked: you built neither step-size control nor tensor-product layers here. inside `mlfs`, later chapters use your code through `mlfs.gnn` and `mlfs.ode` per [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [symmetry & equivariance](01-symmetry-and-equivariance/) | `random_rotation`, `random_orthogonal`, `random_permutation`, `permutation_matrix`, `equivariance_error`, `translate`, `circular_conv2d`, `c4_rotations`, `group_average` → `mlfs.gnn` | a CNN with two stride-2 layers is exactly invariant to 4-pixel shifts and changes 7–8% of its MNIST predictions under 1-pixel shifts; blurring halves that, augmentation doesn't help; a $C_4$-averaged classifier loses 14–22 points on upright digits |
| 02 | [sets](02-sets/) | `masked_pool`, `DeepSets`, `PermEquivariant`, `masked_attention`, `AttentionPool` → `mlfs.gnn` | max pooling can't count and mean pooling can't sum; a sum-pooled net that learned the max fails on bigger sets while max pooling extrapolates perfectly; learned attention pooling fails the sum and the count exactly like the mean |
| 03 | [graph neural networks](03-graph-neural-networks/) | `scatter_sum`/`mean`/`max`/`softmax`, `add_self_loops`, `degree`, `GCNConv`, `SAGEConv`, `GATConv`, `global_add_pool`, `global_mean_pool`, `batch_graphs` → `mlfs.gnn` | GCN averaging converges to $\sqrt{\text{degree}}$, not a constant; with default init an 8-layer GCN collapses together with an 8-layer MLP (kaiming init fixes both), but at 16–32 layers the GCN fails to train where the MLP still can; on a heterophilous graph GCN and GAT are no better than an MLP while GraphSAGE beats it |
| 04 | [GNN expressivity](04-gnn-expressivity/) | `wl_colors`, `wl_test`, `GINConv`, `GIN` → `mlfs.gnn` | GIN's embeddings of WL-equivalent graphs agree to 0.0; a GCN scores exactly 50% at telling 3-regular from 4-regular graphs (GIN: 100%); random features get GIN to 77% on one cycle vs two, at the cost of predictions that change every forward pass |
| 05 | [equivariant networks](05-equivariant-networks/) | `relative_geometry`, `radius_graph`, `EGNNLayer`, `EGNN` → `mlfs.gnn` | on molecular energies the EGNN's error is 15–21× smaller than an augmented MLP's (seeds 0–2), and its 50-molecule model beats the MLP trained on 3,200; on a chiral target it gets $R^2 = 0.000$ |
| 06 | [neural ODEs](06-neural-odes/) | `euler_step`, `midpoint_step`, `rk4_step`, `odeint`, `odeint_adjoint`, `NeuralODE` → `mlfs.ode` | the adjoint stores 25,000× less and runs 2× slower; on a stiff system its float32 gradient is off by a factor of $3 \times 10^7$; a 2-D neural ODE reaches 99.2% on concentric circles by stretching the outer ring around the inner one |

## capstone

*molecular property prediction*, comparing GCN, GIN and EGNN, plus a GIN that sees raw coordinates. QM9 isn't available in this course (no download, no graph library), so you generate the benchmark yourself: Lennard-Jones clusters of 5–12 atoms of two species, grown by random sequential addition, with exact truncated-and-shifted Lennard-Jones energies. the target is invariant to rotations, reflections, translations and relabelling by construction, which makes the comparison and the invariance audit exact. you report test MAE with bootstrap intervals and seed spreads against a per-atom-mean baseline, learning curves over 250–4,000 training molecules, and an invariance audit of every model. see [`capstone/brief.md`](capstone/brief.md) for the reference results.

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a GCN and a GIN step on a star, 1-WL on two trees, an EGNN update under rotation and an RK4 step), and a 60-minute blind rebuild of the scatter operations, GCN and the EGNN layer.

## notation

| symbol | meaning |
|---|---|
| $G$, $g$, $\rho(g)$ | a group, an element of it, its action on inputs |
| $SO(n)$, $O(n)$, $E(n)$, $S_n$ | rotations; rotations and reflections; $O(n)$ plus translations; permutations of $n$ items |
| $R$, $t$, $P_\pi$ | an orthogonal matrix, a translation, the permutation matrix with $(P_\pi x)_i = x_{\pi(i)}$ |
| $X \in \mathbb{R}^{N \times F}$, $x_i$ | node (or set-element) features; also positions $x_i \in \mathbb{R}^3$ in topic 05 |
| `edge_index` $(2, E)$, $j \to i$ | edges as (source, target) columns; messages flow from $j$ to $i$ |
| $\mathcal N(i)$, $d_i$, $\tilde d_i$ | the in-neighbours of $i$, its in-degree, its degree including a self-loop |
| $A$, $\hat A$ | adjacency ($A_{ij} = 1$ for an edge $j \to i$), GCN's $\tilde D^{-1/2}(A + I)\tilde D^{-1/2}$ |
| $\bigoplus$ | a permutation-invariant aggregation (sum, mean, max) |
| $\phi$, $\rho$ (DeepSets), $\phi_e$, $\phi_x$, $\phi_h$ (EGNN) | the learned per-element, readout, message, position and feature functions |
| $\alpha_{ij}$ | attention weight of edge $j \to i$ (a softmax over $i$'s incoming edges) |
| $c_i^{(t)}$, $\{\!\{\cdot\}\!\}$ | node $i$'s 1-WL colour after $t$ rounds, a multiset |
| $\epsilon$ | GIN's weight on the node itself, $(1 + \epsilon)x_i$ |
| $m_{ij}$, $h_i$ | an EGNN message and a node's invariant features |
| $f(t, y)$, $h$, $t_0, \dots, t_K$ | a vector field, a solver step, the output times |
| $a(t)$, $a_\theta$ | the adjoint $\partial L/\partial y(t)$, and the accumulated parameter gradient |
| $\sigma_{ab}$, $\varepsilon_{ab}$, $r_c$ | Lennard-Jones size and well depth for species $a, b$; the cutoff (capstone) |

## how this connects

- builds on: PyTorch training (ch 15), orthogonal matrices, eigenvectors and the matrix exponential (ch 02), Taylor series (ch 03) and reverse-mode autodiff (ch 14), the bootstrap (ch 05), softmax (ch 08), attention (ch 18) and convolution (ch 16).
- used later in: generative models on graphs and molecules (ch 21–22), where `EGNNLayer` is the denoiser of molecular diffusion and `mlfs.ode.odeint` integrates the probability-flow ODE; continuous normalizing flows and flow matching (ch 22); the "what can this architecture distinguish" question for transformers and positional encodings (ch 18 · 27); data augmentation as approximate group averaging in self-supervised learning (ch 25).

## core references

- Bronstein, Bruna, Cohen & Veličković (2021), *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
- Hamilton (2020), *Graph Representation Learning*, Morgan & Claypool. [cs.mcgill.ca/~wlh/grl_book](https://www.cs.mcgill.ca/~wlh/grl_book/)
- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov & Smola (2017), "Deep sets", *NeurIPS*. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- Xu, Hu, Leskovec & Jegelka (2019), "How powerful are graph neural networks?", *ICLR*. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Satorras, Hoogeboom & Welling (2021), "E(n) equivariant graph neural networks", *ICML*. [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
- Chen, Rubanova, Bettencourt & Duvenaud (2018), "Neural ordinary differential equations", *NeurIPS*. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
