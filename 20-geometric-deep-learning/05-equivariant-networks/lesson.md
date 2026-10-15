# equivariant networks

## why this matters

a molecule is a set of atoms with positions in 3-D space. its energy doesn't depend on where the molecule sits, how it's turned, or how you numbered its atoms: it's invariant to translations, rotations, reflections and permutations. the forces on its atoms are vectors, and turning the molecule turns the forces with it: they're **equivariant**. any model of molecules that gets these facts wrong predicts different energies for the same molecule seen from two angles, and forces that don't turn with the molecule, which in a simulation means energy that appears from nowhere.

the obvious model, an MLP on the list of coordinates, gets every one of these wrong. you can teach it the symmetries approximately by showing it many rotated copies of every molecule (augmentation), at a cost in data and accuracy you'll measure in the lab: on our small benchmark, the augmented MLP's error is 15 to 21 times the error of a network that has the symmetries built in.

topic 03's GNNs handle permutations, but they ignore geometry: a graph of bonds can't tell a bent molecule from a straight one, and topic 04 showed how much combinatorial structure alone leaves ambiguous. this topic builds a network that uses the coordinates *and* respects every symmetry exactly: the **E(n)-equivariant graph neural network** (EGNN) of Satorras, Hoogeboom & Welling (2021). its trick is short enough to fit in a sentence: compute messages only from quantities that don't change under rotations (distances), and move atoms only along directions that rotate with the molecule (relative vectors). the lab then shows what that buys, and one thing it costs that almost nobody expects.

## warm-up: predict first

1. which of these are invariant to rotating and translating a point cloud: the positions $x_i$, the differences $x_i - x_j$, the distances $\|x_i - x_j\|$, the centroid $\bar x$?
2. a network outputs a new position for each atom. if the input molecule is rotated by $R$ and shifted by $t$, how should the output positions change?
3. you feed a network the squared distances between atoms and nothing else. can it tell a molecule from its mirror image?
4. two atoms, and a layer that moves each atom along the bond between them. can the two atoms ever leave the line they're on?
5. an MLP on coordinates is trained on molecules that are all stored in one standard orientation. you test it on randomly rotated copies. better, the same, or worse than on standard-orientation test molecules?

## prerequisites check

- [ ] you can draw Haar-random rotations and reflections and test equivariance with them (→ topic 01)
- [ ] you know message passing on an edge list, scatter sums and batched graphs (→ topic 03)
- [ ] you know what an orthogonal matrix does to dot products and norms (→ ch 02 · topic 04)
- [ ] you can build and train MLPs in PyTorch (→ ch 15 · topic 07)

## core ideas

### idea 1: the Euclidean group and its invariants

the symmetries of a point cloud $X = (x_1, \dots, x_N) \in \mathbb{R}^{N \times 3}$ form the **Euclidean group** $E(3)$: maps $x \mapsto Rx + t$ with $R$ orthogonal ($R^\top R = I$, so rotations *and* reflections) and $t$ a translation. together with relabelling the points ($S_N$), that's the full symmetry of a molecule's energy. a function $f$ is

- **invariant** if $f(RX + t) = f(X)$ (energies, scalar properties, node features),
- **equivariant** if $f(RX + t) = R f(X) + t$ for position outputs (updated coordinates) or $R f(X)$ for vector outputs (forces, velocities, dipoles).

(writing $RX + t$ for "apply to every point".) which raw quantities transform how?

| quantity | under $x \mapsto Rx + t$ | type |
|---|---|---|
| $x_i$ | $Rx_i + t$ | neither: translation breaks it |
| $x_i - x_j$ | $R(x_i - x_j)$ | equivariant vector |
| $\|x_i - x_j\|^2$ | unchanged | invariant scalar |
| $(x_i - x_j)\cdot(x_k - x_l)$ | unchanged | invariant scalar |
| $\det[x_1 - x_0, x_2 - x_0, x_3 - x_0]$ | multiplied by $\det R = \pm1$ | invariant under rotations only |

relative vectors are the equivariant building block, and dot products of relative vectors (distances are a special case) the invariant one. a classical result says the invariants are enough in a strong sense: the matrix of all pairwise distances determines a point cloud up to $E(3)$. but notice the last row. the signed volume flips under a reflection, and no function of distances can see it (on paper 3). keep it in mind for the lab.

the graph of who talks to whom should be invariant too. the **radius graph** connects every ordered pair of atoms closer than a cutoff: distances don't change under $E(3)$, so neither does the graph, and relabelling the atoms relabels its edges. for small molecules a fully connected graph (cutoff $\infty$) is common; for large systems a cutoff of a few ångströms keeps the cost linear in the number of atoms.

> **spec · stage 1.**
> - `relative_geometry(pos, edge_index)`: for each edge $j \to i$ (column $(j, i)$ of `edge_index`), `diff` $= x_i - x_j$ `(E, 3)` (target minus source) and the squared distance `(E, 1)`.
> - `radius_graph(pos, cutoff, batch=None)`: every ordered pair $(j, i)$, $i \ne j$, in the same graph (`batch`; `None` means one graph) with $\|x_i - x_j\| < $ `cutoff`, as a `(2, E)` int64 `edge_index` sorted by source then target.

> **▶ build: stage 1.** implement `relative_geometry` and `radius_graph` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the EGNN layer

each node carries invariant features $h_i \in \mathbb{R}^H$ and a position $x_i \in \mathbb{R}^3$. one EGNN layer updates both:

$$m_{ij} = \phi_e\big(h_i,\ h_j,\ \|x_i - x_j\|^2,\ a_{ij}\big),$$

$$x_i' = x_i + \frac{1}{|\mathcal N(i)|}\sum_{j \to i} (x_i - x_j)\,\phi_x(m_{ij}),$$

$$h_i' = h_i + \phi_h\Big(h_i,\ \sum_{j \to i} m_{ij}\Big).$$

$\phi_e$, $\phi_x$, $\phi_h$ are small MLPs; $\phi_x$ outputs one number per edge; $a_{ij}$ are optional edge attributes (bond types, say). read the equations as three rules:

1. **messages see only invariants.** $m_{ij}$ depends on the features and the squared distance, never on raw coordinates. so $m_{ij}$, and everything computed from it, is invariant.
2. **positions move along relative vectors, by invariant amounts.** each neighbour pushes or pulls atom $i$ along the line $x_i - x_j$, with a strength $\phi_x(m_{ij})$ that's an invariant scalar. rotate the input and every $x_i - x_j$ rotates, the scalars don't change, so the displacement rotates: equivariance (on paper 1). translations cancel in the differences, and $x_i$ itself shifts by $t$: $x_i' \mapsto x_i' + t$.
3. **features update from invariant messages**, with a residual connection, as in any message-passing layer.

the average over neighbours (rather than a sum) keeps displacements from growing with the number of neighbours; a node without neighbours stays put. Satorras et al. initialize $\phi_x$'s last layer with tiny weights so positions start almost still and the network learns to move them.

nothing about the equations is specific to 3-D: the same layer is equivariant to $E(n)$ for any $n$, which is where the name comes from. and it's cheap: compared with a plain message-passing layer, it adds one scalar MLP per edge and a scatter of 3-vectors. other equivariant architectures (tensor field networks, NequIP, MACE) send richer messages (vectors and higher-order tensors built from spherical harmonics) at a higher cost; EGNN's restriction to scalar messages is what keeps it simple.

> **spec · stage 2.** `EGNNLayer(hidden_dim, *, edge_dim=0)`: `forward(h, pos, edge_index, edge_attr=None)` returns `(h', pos')` by the three equations above: $\phi_e$ takes $(h_i, h_j, \|x_i - x_j\|^2, a_{ij})$ for each edge $j \to i$; the position update is the *mean* over incoming edges (nodes with none keep their position); $h' = h + \phi_h(h, \sum_j m_{ij})$; `edge_attr` `(E, edge_dim)` when `edge_dim > 0`. $h'$ is invariant and `pos'` equivariant under $E(3)$, both equivariant under relabelling. this is `mlfs.gnn.EGNNLayer`.

> **▶ build: stage 2.** implement `EGNNLayer`.

> **🔮 predict:** five atoms on a straight line go through an EGNN layer. where can they end up?

### idea 3: invariant models, and what E(3) can't see

for a molecule-level property, follow the blueprint: embed the atom types, run a few EGNN layers, apply an MLP to each atom's final $h_i$ and **sum over the atoms** of each molecule. the output is invariant to $E(3)$ and to relabelling, exactly, for any weights. the positions are updated along the way and then discarded; their job was to let later layers see geometry that earlier layers computed. (to predict *forces*, you'd either use the final position displacements, which are equivariant, or differentiate the predicted energy with respect to the input positions, which gives equivariant forces automatically and conserves energy.)

exact invariance has a consequence that's easy to miss. $E(3)$ contains reflections, so an EGNN gives a molecule and its mirror image **identical** outputs. most molecular properties don't care (a molecule and its mirror image have the same energy in isolation), but some do: chirality decides how a drug fits a protein. a target that changes sign under reflection, such as the signed volume of four atoms, is something an $E(3)$-invariant model can't represent at all, however long you train it. if your task has only rotational symmetry, the right group is $SE(3)$ (rotations and translations, no reflections), and the model needs a feature that sees handedness, such as triple products $(x_i - x_k)\cdot\big((x_j - x_k) \times (x_l - x_k)\big)$ in the messages. building in more symmetry than the task has is as wrong as building in less (topic 01's experiment 3 made the same point with digits).

> **spec · stage 3.** `EGNN(in_dim, hidden_dim, out_dim, *, n_layers=4, edge_dim=0)`: a linear embedding `in_dim → hidden_dim`, `n_layers` `EGNNLayer`s, a per-node MLP to `out_dim`, and a sum over each molecule's nodes. `forward(x, pos, edge_index, batch, num_graphs=None, edge_attr=None)` → `(num_graphs, out_dim)`, exactly invariant to rotations, reflections, translations and relabelling; molecules in a batch don't interact.

> **▶ build: stage 3.** implement `EGNN`.

> **🔮 predict:** an EGNN and an augmented MLP are trained to predict the signed volume of four atoms. what test $R^2$ does each reach?

## worked example

two atoms at $x_0 = (0, 0, 0)$ and $x_1 = (1, 1, 0)$, connected both ways, in a layer whose $\phi_x$ happens to output $0.1$ on edge $1 \to 0$ and $-0.2$ on edge $0 \to 1$. relative vectors: for edge $1 \to 0$, $x_0 - x_1 = (-1, -1, 0)$; for edge $0 \to 1$, $x_1 - x_0 = (1, 1, 0)$. each node has one neighbour, so

$$x_0' = (0, 0, 0) + 0.1\cdot(-1, -1, 0) = (-0.1, -0.1, 0), \qquad x_1' = (1, 1, 0) - 0.2\cdot(1, 1, 0) = (0.8, 0.8, 0).$$

both atoms moved along the bond, atom 0 away from atom 1 and atom 1 toward atom 0, and both stayed on the line through them. now rotate the input by $90°$ about the $z$ axis, $R(a, b, c) = (-b, a, c)$: $x_1 = (-1, 1, 0)$. the squared distance is still 2 and the features are unchanged, so $\phi_x$ returns the same $0.1$ and $-0.2$, and $x_1' = (-1, 1, 0) - 0.2(-1, 1, 0) = (-0.8, 0.8, 0) = R(0.8, 0.8, 0)$: the output rotated with the input. the stage 2 checks test exactly this, for random rotations and reflections of random molecules, to $10^{-10}$.

## on paper

1. prove that the EGNN layer is $E(n)$-equivariant: if $(h', x')$ is the output for $(h, x)$, then the output for $(h, Rx + t)$ is $(h', Rx' + t)$ for every orthogonal $R$ and translation $t$.

<details><summary>solution</summary>

with inputs $y_i = Rx_i + t$: $\|y_i - y_j\|^2 = \|R(x_i - x_j)\|^2 = \|x_i - x_j\|^2$ because $R$ preserves norms, so every $m_{ij}$ is unchanged, hence $h_i'$ is unchanged (it depends only on $h_i$ and the $m_{ij}$). for the positions, $y_i' = y_i + \frac{1}{|\mathcal N(i)|}\sum_j (y_i - y_j)\phi_x(m_{ij}) = Rx_i + t + \frac{1}{|\mathcal N(i)|}\sum_j R(x_i - x_j)\phi_x(m_{ij}) = R x_i' + t$, by linearity of $R$ and because the $\phi_x(m_{ij})$ are unchanged scalars. relabelling: every term is a sum over edges with shared functions, so it's permutation-equivariant as in topic 03.

</details>

2. a "simpler" layer updates positions by $x_i' = x_i + \phi_x(h_i)\,x_i$. which symmetry does it break, and how? what about $x_i' = x_i + \sum_j \phi(m_{ij})\, (x_i \times x_j)$?

<details><summary>solution</summary>

the first is rotation-equivariant (a scalar times $x_i$) but not translation-equivariant: shifting the input by $t$ shifts the update term by $\phi_x(h_i)t$, which depends on $t$, so the output isn't $x' + t$. only relative vectors make translations cancel. the second uses cross products of absolute positions: not translation-equivariant either, and under a reflection a cross product picks up an extra factor $\det R = -1$ (it's a pseudovector), so even the relative version $(x_i - x_k)\times(x_j - x_k)$ is only $SO(3)$-equivariant, not $O(3)$.

</details>

3. show that any function of the pairwise distances alone gives the same value for a point cloud $X$ and its mirror image $X' = XM^\top$ with $M = \operatorname{diag}(-1, 1, 1)$, while the signed volume $V = \det[x_1 - x_0, x_2 - x_0, x_3 - x_0]$ flips sign. what's the best constant predictor of $V$ under mean squared error, if $V$ is symmetric around 0?

<details><summary>solution</summary>

$M$ is orthogonal, so $\|Mx_i - Mx_j\| = \|x_i - x_j\|$: same distances, same function value. $V' = \det(M[x_1 - x_0, \dots]) = \det M \cdot V = -V$. an $E(3)$-invariant model has $f(X) = f(X')$, so on a mirror pair it must predict the same number $c$ for targets $V$ and $-V$; the squared error $\frac12[(V - c)^2 + (V + c)^2] = V^2 + c^2$ is minimized at $c = 0$. so the best invariant predictor of a sign-symmetric pseudoscalar is 0 everywhere, with $R^2 = 0$.

</details>

4. compare the cost of one EGNN layer on a fully connected molecule of $N$ atoms with a radius graph that keeps $k$ neighbours per atom on average. which matters for a protein with $N = 5{,}000$?

<details><summary>solution</summary>

the work is per edge: $\phi_e$ and $\phi_x$ cost $O(H^2)$ each per edge, the scatters $O(H)$ and $O(3)$. fully connected: $N(N-1)$ edges, $O(N^2H^2)$. radius graph: $Nk$ edges, $O(NkH^2)$. for $N = 5{,}000$, $N^2 = 2.5 \times 10^7$ edges against $Nk = 1.5 \times 10^5$ for $k = 30$: a factor of about 170. (building the radius graph by brute force is itself $O(N^2)$ distance computations, cheap per pair; neighbour lists or cell lists make it linear.)

</details>

## experiment

the lab trains your `EGNN` and two MLP baselines on synthetic six-atom molecules whose energy is a smooth function of interatomic distances, stored (like QM9) in a standard orientation. you'll test them on standard-orientation and randomly rotated molecules; measure test error against the number of training molecules; and train both kinds of model on the signed volume of four atoms, then feed them mirror images. predict before you run.

## check your understanding

1. why does EGNN feed $\|x_i - x_j\|^2$ to $\phi_e$ rather than $\|x_i - x_j\|$?

<details><summary>answer</summary>

both are invariant, so either works for symmetry. the squared distance is a polynomial in the coordinates, smooth everywhere, while the norm has an infinite derivative at 0 (and a `nan` gradient in autograd when two points coincide). many models then expand the distance in radial basis functions, which a raw scalar input to an MLP approximates less efficiently.

</details>

2. the position update divides by the number of incoming edges. what would go wrong with a plain sum on a large fully connected molecule?

<details><summary>answer</summary>

the displacement would grow linearly with the number of atoms, so the same weights would move atoms ten times further in a 100-atom molecule than in a 10-atom one, and deeper stacks would explode. the mean keeps displacements on the scale of $\phi_x$'s outputs times interatomic distances. (Satorras et al. use $1/(N-1)$ for fully connected graphs, the same thing there.)

</details>

3. is an EGNN's output invariant to relabelling the atoms if the atom types are one-hot features? and if you feed it the atom *index* as a feature?

<details><summary>answer</summary>

with one-hot types, yes: relabelling permutes nodes and their features together, and the layer is permutation-equivariant, the readout invariant. with the atom index as a feature, no: relabelling changes which features sit at which position in space, so the input describes a different (mislabelled) molecule. features must be properties of the atom, not of its position in your list.

</details>

4. data augmentation with random rotations makes an MLP *approximately* invariant. name two costs that the lab measures.

<details><summary>answer</summary>

accuracy and data: the network spends capacity representing the same function in every orientation, so for a fixed training set its error is much higher than the invariant model's (15 to 21 times, in experiment 1), and more data helps it only slowly (experiment 2). a third cost the lab doesn't measure: its predictions on two rotated copies of a molecule still differ.

</details>

5. an EGNN predicts an energy $E(X)$. you compute forces as $F_i = -\partial E/\partial x_i$ with autograd. are the forces equivariant?

<details><summary>answer</summary>

yes. if $E(RX + t) = E(X)$ for all $R, t$, differentiating with the chain rule gives $\nabla E(RX + t) = R\,\nabla E(X)$: the gradient of an invariant function is equivariant. this is how most machine-learned interatomic potentials produce forces, and it also guarantees the forces are conservative.

</details>

6. you want a model for chiral molecules that respects rotations but not reflections. what minimal change to the EGNN layer would let it tell mirror images apart?

<details><summary>answer</summary>

add an $SO(3)$-invariant (but reflection-odd) scalar to the messages, e.g. the triple product $(x_i - x_j)\cdot\big((x_k - x_j)\times(x_l - x_j)\big)$ for some canonical choice of neighbours $k, l$, or the signed volume of each atom's local frame. distances alone are blind to handedness, so the information has to come from somewhere that changes sign.

</details>

## common bugs

- **raw coordinates in the messages**: symptom: features change under translations or rotations; the stage 2 invariance check reports $O(1)$ errors. only squared distances (and edge attributes) go into $\phi_e$.
- **moving atoms along absolute positions**: symptom: rotations pass, translations fail. use $x_i - x_j$, never $x_i$.
- **dividing by zero for isolated atoms**: symptom: `nan` positions for atoms with no neighbours inside the cutoff. clamp the neighbour count at 1 (the sum is 0 anyway).
- **a batch-level mean instead of a per-node mean**: symptom: molecules in a batch influence each other's positions; the "molecules don't interact" check fails. count incoming edges per node with a scatter.
- **forgetting reflections**: symptom: a model with cross products passes the rotation checks and fails the reflection ones. the spec asks for $E(3)$, which includes mirrors.
- **testing in float32**: symptom: errors of $10^{-6}$ you can't interpret. the checks use float64, where an exact layer shows $10^{-14}$.

## going deeper

- Satorras, Hoogeboom & Welling (2021), "E(n) equivariant graph neural networks", *ICML*. [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
- Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko & Müller (2017), "SchNet: a continuous-filter convolutional neural network for modeling quantum interactions", *NeurIPS*. invariant messages from distances. [arXiv:1706.08566](https://arxiv.org/abs/1706.08566)
- Thomas, Smidt, Kearnes, Yang, Li, Kohlhoff & Riley (2018), "Tensor field networks: rotation- and translation-equivariant neural networks for 3D point clouds". [arXiv:1802.08219](https://arxiv.org/abs/1802.08219)
- Batzner, Musaelian, Sun, Geiger, Mailoa, Kornbluth, Molinari, Smidt & Kozinsky (2022), "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", *Nature Communications* 13: 2453. [doi:10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5)
- Puny, Atzmon, Ben-Hamu, Misra, Grover, Smith & Lipman (2022), "Frame averaging for invariant and equivariant network design", *ICLR*. the group-averaging idea of topic 01, made cheap for $E(3)$. [arXiv:2110.03336](https://arxiv.org/abs/2110.03336)
- Ramakrishnan, Dral, Rupp & von Lilienfeld (2014), "Quantum chemistry structures and properties of 134 kilo molecules", *Scientific Data* 1: 140022. QM9. [doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)

## connections

- ← builds on: rotations, reflections and the equivariance tester (topic 01), message passing and scatters (topic 03), why combinatorial structure alone is ambiguous (topic 04).
- → used in: the capstone compares EGNN with GCN and GIN on molecular energies, including an invariance audit; equivariant networks are the denoisers of molecular diffusion models (ch 22), where `EGNNLayer`'s position update becomes a score; gradients of invariant energies give equivariant forces (ch 03's autodiff, put to work).
