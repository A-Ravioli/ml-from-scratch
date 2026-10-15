# symmetry & equivariance

## why this matters

take a digit classifier that is 90% accurate on MNIST and slide every test image one pixel to the right. no human would see a difference, and the digit is still entirely inside the frame. yet in this topic's lab, about one prediction in twelve changes. slide by four pixels instead and *none* change. the network is built from convolutions, which everyone calls "translation-equivariant", and it ends in a global pool, which everyone calls "translation-invariant". so where did the symmetry go, and why does it come back at exactly four pixels?

the same question shows up wherever data has structure: a molecule's energy ignores rotations and atom labels, a point cloud's class ignores the order of its points. a model that has to *learn* such facts wastes capacity and samples on them, and learns them only approximately. a model that has them *built in* gets them exactly, for free, on inputs it has never seen.

this chapter is about building such models: for sets (topic 02), graphs (03–04), point clouds in 3-D (05), and, in a different sense, continuous depth (06). this topic builds the vocabulary and the tools: what a symmetry is, the difference between invariance and equivariance, how to test for them to floating-point precision, why convolution is *the* translation-equivariant linear layer, and the one trick that makes any model invariant to a finite group. the tester you write here checks every layer in the rest of the chapter.

## warm-up: predict first

1. a function $f$ on lists of numbers satisfies $f(\text{shuffled } x) = f(x)$ for every shuffle. a second function $g$ satisfies $g(\text{shuffled } x) = \text{shuffled } g(x)$. which of `sum`, `sort`, `x - mean(x)`, `x[0]` is like $f$, which like $g$, and which like neither?
2. you compose two layers that are each translation-equivariant. is the composition translation-equivariant? what if you then flatten the feature map and apply a linear layer?
3. a convolution with stride 2 is applied to an image shifted by 1 pixel. by how much is the output shifted?
4. you average a network's output over the 4 rotations of its input by 0°, 90°, 180°, 270°. is the result invariant to rotations by 90°? by 45°?
5. you want a uniformly random rotation of 3-D space. is "draw three Euler angles uniformly from $[0, 2\pi)$" a good recipe?

## prerequisites check

- [ ] you can check whether a matrix is orthogonal and compute its determinant (→ ch 02 · topics 02 and 04)
- [ ] you know what the QR decomposition computes (→ ch 02 · topic 04)
- [ ] you can write a 2-D convolution as a sum over a kernel window, and know what padding and stride do (→ ch 16 · topic 01, or just the formula in idea 3)
- [ ] you can build and train a small PyTorch module (→ ch 15 · topic 07)
- [ ] you know `as_rng` for the course's `rng` argument (→ ch 00 · topic 05)

## core ideas

### idea 1: symmetries are groups acting on your data

a **symmetry** of a problem is a transformation of the input that you know, in advance, doesn't change the answer (or changes it in a known way). rotating a molecule, shifting an image, reordering the points of a point cloud. the set of all such transformations of a problem always has the same algebraic structure, a **group**: a set $G$ with a composition $gh$ such that

- composing two symmetries gives a symmetry (closure),
- doing nothing is a symmetry (an identity $e$),
- every symmetry can be undone (an inverse $g^{-1}$ with $gg^{-1} = e$),
- and composition is associative.

that's all a group is. the groups this chapter uses:

| group | elements | acts on | example |
|---|---|---|---|
| $\mathbb{Z}_H \times \mathbb{Z}_W$ | circular shifts $(d_y, d_x)$ | images $H \times W$ | CNNs |
| $C_4$ | rotations by $0°, 90°, 180°, 270°$ | square images | this topic's lab |
| $S_n$ | permutations of $n$ items | sets, nodes of graphs | topics 02–04 |
| $SO(n)$ | rotations: $R^\top R = I$, $\det R = +1$ | points in $\mathbb{R}^n$ | |
| $O(n)$ | rotations and reflections: $R^\top R = I$ | points in $\mathbb{R}^n$ | |
| $E(n)$ | $x \mapsto Rx + t$, $R \in O(n)$ | point clouds | topic 05 |

the group itself is abstract; what touches your data is its **action**: for each $g$, a concrete map $x \mapsto \rho(g)x$ on inputs, compatible with composition, $\rho(gh) = \rho(g)\rho(h)$. the same group can act in different ways. a permutation $\pi \in S_n$ acts on a set of $n$ feature vectors $X \in \mathbb{R}^{n \times F}$ by reordering rows, $X \mapsto X[\pi]$, which is the matrix $P_\pi X$ with $(P_\pi)_{i,\pi(i)} = 1$. it acts on a graph's adjacency matrix by $A \mapsto P_\pi A P_\pi^\top$ (rows *and* columns). a rotation acts on a point cloud $X \in \mathbb{R}^{n \times 3}$ by $X \mapsto XR^\top$ (each point $x_i \mapsto Rx_i$).

to *test* a symmetry you need random group elements, and for a continuous group "random" should mean uniform: the **Haar measure**, the one distribution unchanged by composing with any fixed element. for $O(n)$: take a matrix $Z$ of independent standard normals (its distribution is rotation-invariant) and its QR decomposition $Z = QR$. QR is unique only up to the signs of $R$'s diagonal, and LAPACK's sign convention biases $Q$ (in 3-D its diagonal averages about $-0.5$). multiply column $j$ of $Q$ by $\operatorname{sign}(R_{jj})$ and $Q$ is Haar on $O(n)$, half rotations and half reflections (Mezzadri, 2007); flip one column whenever $\det Q = -1$ to get a rotation. (uniform Euler angles are *not* uniform on $SO(3)$: like uniform latitude and longitude on a globe, they crowd the poles.)

> **spec · stage 1.**
> - `random_orthogonal(dim, *, rng=None)`: a Haar-random element of $O(\text{dim})$, a `torch.float64` tensor of shape `(dim, dim)`, drawn with `as_rng(rng)`.
> - `random_rotation(dim, *, rng=None)`: a Haar-random element of $SO(\text{dim})$: as above, with $\det = +1$.
> - `random_permutation(n, *, rng=None)`: a uniformly random permutation of $0, \dots, n-1$, an `int64` tensor `(n,)`.
> - `permutation_matrix(perm)`: the `float64` matrix $P$ with $P x = x[\text{perm}]$, i.e. $P_{i, \text{perm}[i]} = 1$.

> **▶ build: stage 1.** implement the four functions in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** in 3-D, what is the average of the entry $R_{11}$ over many Haar-random rotations? and the average of $R_{11}^2$? (hint: $R e_1$ is a uniformly random unit vector.)

### idea 2: invariance, equivariance, and how to test them

a function $f$ is **equivariant** to a group $G$ (acting by $\rho_\text{in}$ on inputs and $\rho_\text{out}$ on outputs) if transforming the input transforms the output in the matching way:

$$f(\rho_\text{in}(g)\,x) = \rho_\text{out}(g)\,f(x) \quad \text{for all } g \in G,\ x.$$

it is **invariant** if $\rho_\text{out}(g)$ is the identity: $f(\rho_\text{in}(g)x) = f(x)$. invariance is the special case of equivariance where the output doesn't move.

which one you want depends on the output. a molecule's energy is a number, and a rotated molecule has the same energy: invariance. the forces on its atoms are vectors, and a rotated molecule has rotated forces: equivariance, with $\rho_\text{out}(R) = R$. a per-node prediction on a graph moves with the nodes: permutation equivariance. the class of the whole graph doesn't: permutation invariance.

two facts make equivariance a design tool and not just a property:

- **composition.** if $f_1$ is equivariant from $\rho_0$ to $\rho_1$ and $f_2$ from $\rho_1$ to $\rho_2$, then $f_2 \circ f_1$ is equivariant from $\rho_0$ to $\rho_2$ (on paper 1). so a deep network built from equivariant layers is equivariant, however deep.
- **equivariant, then invariant, is invariant.** finish the stack with an invariant layer (a sum over the nodes, a global pool over the image) and the whole network is invariant.

an equivariant layer satisfies the equation exactly, which makes it testable: draw a random input and group element, compute both sides, compare. exact layers agree to about $10^{-12}$ in `float64` ($10^{-5}$ in `float32`); broken ones show errors the size of their outputs, so one number tells you which case you're in:

$$\text{err}(f, x, g) = \max \big|\, f(\rho_\text{in}(g)\,x) - \rho_\text{out}(g)\,f(x) \,\big|.$$

> **spec · stage 2.** `equivariance_error(f, x, transform_in, transform_out)` returns the error above as a Python `float`, for callables `transform_in` ($x \mapsto \rho_\text{in}(g)x$) and `transform_out` ($y \mapsto \rho_\text{out}(g)y$); `transform_out=None` means the identity (an invariance test). it runs `f` without recording gradients and doesn't modify `x`. this is `mlfs.gnn.equivariance_error`.

> **▶ build: stage 2.** implement `equivariance_error`.

### idea 3: translations, convolution, and what stride breaks

an image (or a feature map) is a function on a grid of pixels. to have a group with no edge effects, treat the grid as a torus: shifting by $(d_y, d_x)$ moves pixel $(i, j)$ to $(i + d_y, j + d_x)$ modulo the size. the action on an image $x$ is

$$(T_{d}\, x)[i, j] = x[i - d_y,\ j - d_x],$$

so the content moves *down* by $d_y$ and *right* by $d_x$ (`torch.roll`).

a **circular convolution** (strictly, a cross-correlation, the convention of every deep-learning library) with a $k \times k$ kernel, $k$ odd, $p = \lfloor k/2 \rfloor$, is

$$y[o, i, j] = b_o + \sum_{c}\sum_{a=0}^{k-1}\sum_{b=0}^{k-1} w[o, c, a, b]\; x[c,\ (i + a - p) \bmod H,\ (j + b - p) \bmod W].$$

because every output pixel uses the *same* weights on its own neighbourhood, shifting the input shifts the output by the same amount: $\operatorname{conv}(T_d x) = T_d \operatorname{conv}(x)$. the converse is the deep fact: **every linear map on images that commutes with all circular shifts is a circular convolution** (on paper 3). weight sharing isn't one clever design choice among many; it's what translation equivariance forces on a linear layer.

now add a **stride** $s$: keep only every $s$-th output row and column, $y_s[i, j] = y[si, sj]$. shift the input by $s$ and the full-resolution output shifts by $s$, which is exactly one step on the subsampled grid: $y_s$ shifts by 1. shift by 1, and the full-resolution output moves by one pixel, which on the subsampled grid is half a step. there's no half-pixel shift of a grid, so no equation of the form $y_s(T_1 x) = T_? y_s(x)$ holds. strided layers are equivariant only to the **subgroup** of shifts by multiples of $s$, and two strided layers in a row only to multiples of $s^2$. a global pool at the end turns this into invariance to those same shifts, and nothing more. this is the effect you met in "why this matters", and it's a form of **aliasing**: subsampling a signal that contains frequencies above half the new sampling rate. blurring before subsampling (Zhang, 2019) removes most of those frequencies and makes the network *much* less sensitive, but not exactly invariant.

> **spec · stage 3.**
> - `translate(x, shift)`: circularly shift the last two axes by `shift = (dy, dx)`: `out[..., i, j] = x[..., i - dy, j - dx]`.
> - `circular_conv2d(x, weight, bias=None, *, stride=1)`: the cross-correlation above, `x` `(B, C, H, W)`, `weight` `(O, C, k, k)` with `k` odd (`ValueError` otherwise), circular padding of `k // 2` on each side, then stride: output `(B, O, ceil(H/s), ceil(W/s))`. you may use `F.pad(..., mode="circular")` and `F.conv2d`; it must be differentiable.

> **▶ build: stage 3.** implement `translate` and `circular_conv2d`.

> **🔮 predict:** a network has two stride-2 layers and ends in a global pool. for which shifts $(0, d)$, $d = 0, \dots, 8$, is it exactly invariant?

### idea 4: invariance by averaging, and the GDL blueprint

there is one completely general way to make any function invariant to a finite group: average it over the group,

$$\bar f(x) = \frac{1}{|G|}\sum_{g \in G} f(\rho(g)\,x).$$

$\bar f$ is invariant because applying $h$ to the input only reorders the terms of the sum: $\bar f(\rho(h)x) = \frac{1}{|G|}\sum_g f(\rho(gh)x)$, and as $g$ runs over $G$, so does $gh$ (on paper 2). this is the **Reynolds operator** of invariant theory, and it's exact: no learning, no approximation. averaging any network over $C_4$ gives a network that's invariant to quarter turns down to rounding error. averaging over all $H \times W$ circular shifts makes even a fully connected network shift-invariant. and if you *train* the averaged network, it learns the best invariant function it can.

so why isn't this the end of the chapter? cost: $\bar f$ takes $|G|$ forward passes, 4 for $C_4$ but $n!$ for the permutations of $n$ atoms ($2.4 \times 10^{18}$ for $n = 20$), and infinitely many for a continuous group like $SO(3)$, where sampling gives only approximate invariance. the alternative is to build equivariance into each layer at no extra cost: convolutions for shifts, sums over neighbours for permutations (topics 02–04), distances and relative vectors for rotations (topic 05).

that alternative has a common shape, which Bronstein et al. (2021) call the **geometric deep learning blueprint**:

1. **equivariant layers** $B_1, B_2, \dots$: each commutes with the group (convolutions, message passing).
2. **pointwise nonlinearities** $\sigma$ in between: applying the same function to every pixel, node or point keeps equivariance.
3. **local pooling** (optional): coarsening that keeps a smaller version of the symmetry (strided convolutions, graph coarsening). as idea 3 showed, this is where exactness can quietly leak away.
4. **a global invariant pooling** $P$ at the end for invariant tasks: sum, mean or max over all positions or nodes.

$$f = P \circ B_L \circ \sigma \circ \cdots \circ \sigma \circ B_1.$$

every architecture in this chapter is an instance: DeepSets ($B$ = shared per-element MLP, $P$ = sum), GNNs ($B$ = message passing), EGNN ($B$ = message passing on distances and relative positions). a symmetry is only worth building in if it really is a symmetry of the task, and the lab's experiment 3 shows what happens when it isn't.

> **spec · stage 4.**
> - `c4_rotations()`: a list of 4 functions; element $k$ rotates the last two axes by $k \cdot 90°$, exactly `torch.rot90(x, k, dims=(-2, -1))`.
> - `group_average(f, x, group)`: $\frac{1}{|G|}\sum_{g} f(g(x))$ for `group` a list of functions; keeps `f`'s output shape and stays differentiable (so you can train through it).

> **▶ build: stage 4.** implement `c4_rotations` and `group_average`.

> **🔮 predict:** you train a digit classifier inside a $C_4$ average. on upright test digits, will it be more accurate, equally accurate, or less accurate than the same network trained without the average?

## worked example

take three points in the plane, $x = \begin{pmatrix} 1 & 2 \\ 3 & 5 \\ -4 & 0.5 \end{pmatrix}$, and the permutation $\pi = (1, 2, 0)$, so $x[\pi] = \begin{pmatrix} 3 & 5 \\ -4 & 0.5 \\ 1 & 2 \end{pmatrix}$.

- $f(x) = \sum_i x_i = (0, 7.5)$. $f(x[\pi]) = (0, 7.5)$ too: invariant, error 0.
- $f(x) = x^2$ elementwise. squaring then permuting equals permuting then squaring: equivariant, error 0.
- $f(x) = x - x_0$ (subtract the first point from every point). then $f(x[\pi]) = x[\pi] - x_1$ (the new first point is the old $x_1$), while $f(x)[\pi] = x[\pi] - x_0$. their difference is $x_0 - x_1 = (1 - 3,\ 2 - 5) = (-2, -3)$ in every row, so the error is $\max(2, 3) = 3$. "the first point" is not a permutation-equivariant notion; "the mean point" would be.

your `equivariance_error` must return exactly 0, 0 and 3 here (the stage 2 checks use this example).

now a $C_4$ average by hand. on $2 \times 2$ images $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, let $f$ read the top-left pixel: $f = a$. `torch.rot90` turns the image counter-clockwise, so the four rotations put $a, b, d, c$ in the top-left corner, and $\bar f = (a + b + c + d)/4$, the mean pixel, which is visibly rotation-invariant. averaging a "read one pixel" function gave the "read all pixels, equally" function: the average throws away exactly the information the group can move around.

## on paper

1. show that if $f_1$ is equivariant from $\rho_0$ to $\rho_1$ and $f_2$ is equivariant from $\rho_1$ to $\rho_2$, then $f_2 \circ f_1$ is equivariant from $\rho_0$ to $\rho_2$. deduce that a stack of equivariant layers followed by an invariant layer is invariant.

<details><summary>solution</summary>

$f_2(f_1(\rho_0(g)x)) = f_2(\rho_1(g) f_1(x)) = \rho_2(g) f_2(f_1(x))$, using $f_1$'s equivariance, then $f_2$'s. by induction a stack of $L$ equivariant layers is equivariant from the input action to the last layer's action. an invariant layer is equivariant with $\rho_\text{out} = $ identity, so the composition is equivariant to the identity action: invariant.

</details>

2. show that the group average $\bar f$ is invariant, and that if $f$ is already invariant then $\bar f = f$. where does the proof use that $G$ is a group and not just a set of transformations?

<details><summary>solution</summary>

$\bar f(\rho(h)x) = \frac{1}{|G|}\sum_g f(\rho(g)\rho(h)x) = \frac{1}{|G|}\sum_g f(\rho(gh)x)$. the map $g \mapsto gh$ is a bijection of $G$ onto itself (its inverse is $g \mapsto gh^{-1}$), so the last sum is the same sum in a different order: $\bar f(x)$. if $f$ is invariant, every term equals $f(x)$. the proof needs closure ($gh \in G$) and inverses (for the bijection). averaging over, say, rotations by $0°$ and $90°$ only is not invariant to $90°$: the $90°$ input produces the terms for $90°$ and $180°$.

</details>

3. let $L: \mathbb{R}^n \to \mathbb{R}^n$ be linear and commute with the cyclic shift $(Sx)_i = x_{i-1 \bmod n}$. show that $L$ is a circular convolution: $(Lx)_i = \sum_j w_{j}\, x_{(i - j) \bmod n}$ for some vector $w$.

<details><summary>solution</summary>

let $w = L e_0$, the response to a unit impulse at position 0. the impulse at position $j$ is $e_j = S^j e_0$, so $L e_j = L S^j e_0 = S^j L e_0 = S^j w$, i.e. $(L e_j)_i = w_{(i - j) \bmod n}$. by linearity $Lx = \sum_j x_j L e_j$, so $(Lx)_i = \sum_j x_j w_{(i-j) \bmod n}$, a circular convolution of $x$ with $w$ (substitute $j \to i - j$ for the form above). the same argument in 2-D, applied per input and output channel, gives the convolutional layer.

</details>

4. which linear maps $L: \mathbb{R}^n \to \mathbb{R}^n$ commute with *every* permutation matrix? (this is the preview of DeepSets' equivariant layer in topic 02.)

<details><summary>solution</summary>

$P L P^\top = L$ for every permutation $P$ means $L_{\pi(i)\pi(j)} = L_{ij}$ for all $\pi$. any diagonal entry can be moved to any other diagonal entry, so they're all equal, say $a$; any off-diagonal pair $(i, j)$, $i \neq j$, can be moved to any other off-diagonal pair, so they're all equal, say $b$. hence $L = aI + b\mathbf{1}\mathbf{1}^\top$: "a multiple of yourself plus a multiple of the sum of everyone". two parameters, whatever $n$ is.

</details>

## experiment

the lab builds small CNNs out of your `circular_conv2d` and measures their symmetries with your `equivariance_error`. you'll measure how a random network's output changes under shifts of 0 to 8 pixels for different strides; train strided networks on MNIST (with and without blurring before subsampling, and with shift augmentation) and count how often a small shift changes the predicted class; and train a network inside a $C_4$ group average, then test it on upright, quarter-turned and 45°-turned digits. write down your predictions first: most people get at least one of the three wrong.

## check your understanding

1. is `sort` permutation-invariant, permutation-equivariant, or neither? and `argsort`?

<details><summary>answer</summary>

`sort` is invariant: shuffling the input doesn't change the sorted output. `argsort` is neither: it returns positions, which change when the input is shuffled, and not by the same permutation (it changes by the *inverse* permutation's relabelling of indices). (`rank`, the position each element would take in the sorted list, *is* equivariant.)

</details>

2. a CNN uses zero padding instead of circular padding. is a stride-1 convolution still exactly shift-equivariant?

<details><summary>answer</summary>

no: a shifted image brings new content in on one side and pushes content out on the other, and the padding zeros don't match either, so outputs near the edges differ. it stays equivariant in the interior. circular padding makes the group exact. for MNIST, whose border is blank, a circular shift of a few pixels is the same as an ordinary shift, which is why the lab can use circular layers on real digits.

</details>

3. why does `equivariance_error` use the *maximum* absolute difference and not the mean?

<details><summary>answer</summary>

a function can be exact almost everywhere and broken on a few entries (a border pixel, one special node); the mean would hide that, the max reports the worst case, which is what "for all $x$ and $g$" asks for. the lab divides by the output scale to compare networks.

</details>

4. a test passes with an error of $3 \times 10^{-6}$. is the function equivariant?

<details><summary>answer</summary>

it depends on the dtype and scale. in `float32` with outputs of size ~1, $10^{-6}$ is rounding error, so yes. in `float64` it's a billion times larger than rounding error, so something is slightly off (a non-equivariant term with a tiny weight, an interpolation). run the test in `float64` whenever you can: the gap between "exact" and "broken" is much wider there.

</details>

5. a network averaged over $C_4$ is exactly invariant to quarter turns. why isn't it invariant to 45° rotations, and could you fix that by averaging over $C_8$?

<details><summary>answer</summary>

45° is not in $C_4$, and on a pixel grid it isn't even exact: it needs interpolation. $C_8$ averaging wouldn't fix it, because two interpolated 45° turns don't compose to the exact 90° one: on a pixel grid, rotations by other angles don't form a group.

</details>

## common bugs

- **QR without the sign fix**: symptom: your rotations are orthogonal with $\det = 1$, but the stage 1 Haar check reports entries averaging around $\pm 0.5$. fix: multiply each column of $Q$ by the sign of the matching diagonal entry of $R$.
- **late binding in lambdas**: symptom: all four elements of your $C_4$ list rotate by 270°. a lambda created in a loop looks up `k` when it's *called*, after the loop ended. fix: bind it at creation, `lambda x, k=k: ...`.
- **shift direction**: symptom: `translate` is off by a sign, so equivariance tests fail only for asymmetric shifts. `out[i] = x[i - d]` moves content *forward*; that's what `torch.roll(x, d)` does.
- **comparing the strided output with the wrong shift**: symptom: a stride-2 convolution "isn't equivariant" for a shift of 2. the output moves by $2/s = 1$ pixel, not 2.

## going deeper

- Bronstein, Bruna, Cohen & Veličković (2021), *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. the blueprint, and everything in this chapter in one framework. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
- Cohen & Welling (2016), "Group equivariant convolutional networks", *ICML*. convolutions over $C_4$ and $D_4$ as well as shifts. [arXiv:1602.07576](https://arxiv.org/abs/1602.07576)
- Zhang (2019), "Making convolutional networks shift-invariant again", *ICML*. aliasing in strided layers, and blur pooling. [arXiv:1904.11486](https://arxiv.org/abs/1904.11486)
- Azulay & Weiss (2019), "Why do deep convolutional networks generalize so poorly to small image transformations?", *JMLR* 20(184). [arXiv:1805.12177](https://arxiv.org/abs/1805.12177)
- Mezzadri (2007), "How to generate random matrices from the classical compact groups", *Notices of the AMS* 54(5). the QR sign fix. [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050)
- Kondor & Trivedi (2018), "On the generalization of equivariance and convolution in neural networks to the action of compact groups", *ICML*. equivariant linear layers are convolutions, in general. [arXiv:1802.03690](https://arxiv.org/abs/1802.03690)

## connections

- ← builds on: orthogonal matrices, determinants and QR (ch 02), convolution (ch 16), PyTorch modules and training (ch 15 · topic 07), `as_rng` (ch 00).
- → used in: every topic of this chapter tests its layers with `equivariance_error` and these group elements: permutations for DeepSets (02) and GNNs (03–04), rotations, reflections and translations for EGNN (05), and all three in the capstone. data augmentation as approximate group averaging returns in self-supervised learning (ch 25).
