# 20 · geometric deep learning · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. define invariance and equivariance of $f$ under a group $G$. which one do you want for a molecule's energy, its atomic forces, a graph's per-node labels, and a graph's class?

<details><summary>answer</summary>

equivariant: $f(\rho_\text{in}(g)x) = \rho_\text{out}(g)f(x)$ for all $g$; invariant: the special case $\rho_\text{out}(g) = $ identity. energy: invariant to $E(3)$ and permutations. forces: equivariant to rotations ($F \mapsto RF$), invariant to translations, equivariant to permutations. per-node labels: permutation-equivariant. graph class: permutation-invariant. (topic 01, idea 2)

</details>

2. why does `np.linalg.qr` of a Gaussian matrix not give a uniformly random orthogonal matrix, and what fixes it?

<details><summary>answer</summary>

QR is unique only up to the signs of $R$'s diagonal, and LAPACK's sign convention depends on the input, which biases $Q$ (in 3-D its diagonal averages about $-0.5$). multiplying each column of $Q$ by the sign of $R_{jj}$ makes the factorization unique, and then $Q$ inherits the rotation invariance of the Gaussian: Haar on $O(n)$. flip one column when $\det Q = -1$ for $SO(n)$. (topic 01, idea 1)

</details>

3. a CNN of circular convolutions with two stride-2 layers ends in a global pool. to which shifts is it exactly invariant, and why does a 1-pixel shift change its output by as much as the output's size?

<details><summary>answer</summary>

to shifts by multiples of 4 (each stride-2 layer is equivariant only to even shifts, which it turns into shifts by 1). a 1-pixel shift makes each strided layer sample the other sub-grid, so the network computes a genuinely different function of the image; nothing forces the sub-grids to agree. blurring before subsampling reduces but doesn't remove the effect. (topic 01, idea 3 and lab)

</details>

4. why is group averaging exact, and why don't we use it for $SO(3)$ or for permutations of 20 atoms?

<details><summary>answer</summary>

applying $h$ to the input permutes the terms $f(gx)$ of the average, because $g \mapsto gh$ is a bijection of the group; so the average is invariant for any $f$. but it costs $|G|$ forward passes: $20! \approx 2.4 \times 10^{18}$ for permutations, infinitely many for $SO(3)$ (sampling gives only approximate invariance). equivariant layers build the symmetry in at no extra cost. (topic 01, idea 4)

</details>

5. which of sum, mean and max pooling can count the elements of a set? which can tell $\{a, b\}$ from $\{a, a, b\}$? and attention pooling?

<details><summary>answer</summary>

only the sum does both. mean sees the distribution (so it can't tell a set from the set doubled); max sees the support (not multiplicities). attention pooling is a softmax-weighted mean, so it gives the same output for a set and the set listed twice: it can't count either. (topic 02, ideas 1, 2, 4 and on paper 2–3)

</details>

6. what is the most general permutation-equivariant linear map $\mathbb{R}^{n \times F} \to \mathbb{R}^{n \times F'}$?

<details><summary>answer</summary>

$y_i = x_i\Lambda + (\sum_j x_j)\Gamma$ (plus a bias): one matrix for the element itself, one for the sum of all elements, $2FF'$ parameters whatever $n$ is. for one channel it's $\lambda I + \gamma\mathbf{1}\mathbf{1}^\top$. (topic 01, on paper 4; topic 02, idea 3)

</details>

7. write GCN's propagation rule as messages over edges. why the self-loops, and why $1/\sqrt{\tilde d_i\tilde d_j}$?

<details><summary>answer</summary>

$x_i' = \sum_{j \in \mathcal N(i) \cup \{i\}} W^\top x_j/\sqrt{\tilde d_i \tilde d_j} + b$ with degrees counted including the self-loop. self-loops keep the node's own features in its update; the symmetric normalization keeps sums from growing with degree and makes the propagation matrix symmetric with eigenvalues in $(-1, 1]$. (topic 03, idea 2)

</details>

8. what do node features converge to under repeated GCN propagation $\hat A^k X$ on a connected graph, and how fast?

<details><summary>answer</summary>

to the projection onto $\hat A$'s top eigenvector $\tilde D^{1/2}\mathbf 1$: every feature column becomes proportional to $\sqrt{\tilde d_i}$, so nodes differ only by degree. the other components decay like $|\lambda_2|^k$. (topic 03, on paper 2 and lab)

</details>

9. a 16-layer plain GCN fails on a node-classification task. what control experiment tells you whether oversmoothing is to blame?

<details><summary>answer</summary>

train a same-depth MLP on the same features (no graph), and make sure the control is itself trainable (kaiming initialization, zero biases). if a properly initialized MLP fails too, it's ordinary deep-network trainability; if the MLP trains and the GCN doesn't, something graph-specific is going on (in the lab: at 8 layers the default-init failure is shared by the MLP and fixed by kaiming init; at 16–32 layers the kaiming GCN can't fit its training nodes while the MLP can). a linear classifier on $\hat A^{16}X$ shows how much information propagation alone leaves, but it standardizes the features, so it's an upper bound. (topic 03, lab)

</details>

10. on a heterophilous graph, why can GraphSAGE beat an MLP while GCN is no better than it?

<details><summary>answer</summary>

GCN averages the node's own features into the same pool as its neighbours', which on a heterophilous graph are mostly other classes: its own signal is diluted. GraphSAGE has a separate weight for the node itself and can use the neighbour mean as extra evidence ("my neighbours look like B and C, so I'm probably A"). (topic 03, idea 3 and lab)

</details>

11. state the relationship between message-passing GNNs and the 1-WL test. which architectural choice lets GIN reach the bound?

<details><summary>answer</summary>

if 1-WL can't distinguish two graphs (with uniform initial features), no message-passing GNN with an invariant readout can, for any weights; nodes with equal WL colours get equal embeddings. GIN reaches the bound with an injective update: an unnormalized **sum** over neighbours, $(1 + \epsilon)$ on the node itself, and an MLP with a hidden layer, plus sum readouts. (topic 04, idea 3)

</details>

12. why does a GCN give exactly the same embedding to every 3-regular and every 4-regular graph on 12 nodes when all node features are equal?

<details><summary>answer</summary>

on a $d$-regular graph every row of $\hat A$ is $1/(d+1)$ on $d + 1$ entries, summing to 1, so $\hat A x = x$ for constant $x$: the degree never enters. every layer maps every node to the same vector, and a readout over the same number of identical vectors is identical. (topic 04, worked example and lab)

</details>

13. which quantities can an EGNN layer feed into its message MLP, and along which directions may it move atoms? why exactly those?

<details><summary>answer</summary>

invariants only: node features, squared distances, edge attributes. it moves atom $i$ along the relative vectors $x_i - x_j$, scaled by invariant scalars $\phi_x(m_{ij})$ (averaged over neighbours). invariant inputs make the messages invariant; relative vectors rotate with the molecule and cancel translations, so the update is $E(n)$-equivariant. (topic 05, idea 2)

</details>

14. an EGNN is trained to predict the signed volume of four atoms. what does it predict, and why?

<details><summary>answer</summary>

a constant (about 0), $R^2 = 0$. it's invariant to reflections, so it gives a molecule and its mirror image the same output, while the target flips sign; the best such function under squared error is 0. a chiral task needs $SE(3)$ invariance and a reflection-odd feature like a triple product. (topic 05, idea 3 and lab)

</details>

15. what does the adjoint method store, what does it cost, and when can its gradient be badly wrong?

<details><summary>answer</summary>

it stores only the solution at the output times (memory independent of the number of steps), and pays about twice the compute of backprop (each backward stage evaluates $f$ and a vector-Jacobian product). it recomputes $y(t)$ by integrating backwards, which amplifies errors in fast-decaying modes by $e^{kT}$: on stiff systems the parameter gradients can be wrong by orders of magnitude, especially in float32. (topic 06, idea 3 and lab)

</details>

## 2. by hand (no computer)

1. **one GCN step.** a star graph: node 0 connected to nodes 1, 2, 3 (undirected), one feature $x = (1, 0, 0, 0)$, $W = 1$, $b = 0$. compute $\hat A x$ with self-loops and symmetric normalization. then compute one GIN step $h_i = (1 + \epsilon)x_i + \sum_{j \to i} x_j$ with $\epsilon = 0$, before the MLP.

<details><summary>solution</summary>

degrees with self-loops: $\tilde d_0 = 4$, $\tilde d_{1,2,3} = 2$. node 0: $x_0/\tilde d_0 = 1/4$ (its leaves carry 0). each leaf: $x_0/\sqrt{\tilde d_i\tilde d_0} = 1/\sqrt 8 \approx 0.354$. so $\hat A x = (0.25, 0.354, 0.354, 0.354)$. GIN: node 0 gets $x_0 + 0 = 1$, each leaf gets $0 + x_0 = 1$: $(1, 1, 1, 1)$.

</details>

2. **WL by hand.** run 1-WL on the path $0 - 1 - 2 - 3 - 4$ (5 nodes) and the "spider" $K_{1,3}$ with one leg extended (centre 0 joined to 1, 2, 3, and 3 joined to 4). after how many rounds do the colour histograms differ?

<details><summary>solution</summary>

round 1 (degrees): the path has degrees $(1, 2, 2, 2, 1)$, two 1s and three 2s; the spider has $(3, 1, 1, 2, 1)$, three 1s, one 2, one 3. the histograms already differ after one round: different degree sequences.

</details>

3. **EGNN equivariance and RK4 by hand.** (a) atoms at $x_0 = (0, 0, 0)$, $x_1 = (2, 0, 0)$, one edge each way, $\phi_x = 0.25$ on both edges. compute both updated positions, then redo it after rotating the input by $90°$ about $z$ and check equivariance. (b) one RK4 step of $dy/dt = -2y$ from $y = 1$ with $h = 0.5$: compute $k_1, \dots, k_4$ and $y_1$, and compare with $e^{-1}$.

<details><summary>solution</summary>

(a) $x_0' = x_0 + 0.25(x_0 - x_1) = (-0.5, 0, 0)$; $x_1' = x_1 + 0.25(x_1 - x_0) = (2.5, 0, 0)$. rotated input: $x_1 = (0, 2, 0)$, the distance is unchanged so $\phi_x$ is still 0.25, giving $x_0' = (0, -0.5, 0)$ and $x_1' = (0, 2.5, 0)$, which are the rotated outputs. (b) $k_1 = -2$, $k_2 = -2(1 - 0.5) = -1$, $k_3 = -2(1 - 0.25) = -1.5$, $k_4 = -2(1 - 0.75) = -0.5$; $y_1 = 1 + \tfrac{0.5}{6}(-2 - 2 - 3 - 0.5) = 1 - 0.625 = 0.375$, against $e^{-1} = 0.368$ (one large step, $h\lambda = -1$, error 0.007).

</details>

## 3. blind rebuild

with no notes, in 60 minutes: copy `03-graph-neural-networks/exercise.py` and `05-equivariant-networks/exercise.py`, each with its `test_exercise.py`, into two scratch directories (say `rebuild/gnn/` and `rebuild/egnn/`), delete your implementations of the scatter operations, `add_self_loops`, `degree` and `GCNConv` (topic 03, stages 1–2) and of `relative_geometry`, `radius_graph` and `EGNNLayer` (topic 05, stages 1–2), and rebuild them from memory. then run the checks against your copies:

```bash
pytest rebuild/gnn/test_exercise.py -k "stage1 or stage2"
pytest rebuild/egnn/test_exercise.py -k "stage1 or stage2"
```

## 4. readiness

move on if you got at least 80% of sections 1–2 and passed section 3. otherwise, revisit the topics you missed. the same checkpoint works as a placement test: if you can do all of it before starting the chapter, skim the lessons and go straight to the capstone.
