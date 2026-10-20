# universal approximation

## why this matters

you've built networks and trained them, and they work. but why should a stack of matrix products and ReLUs be able to represent the function you care about at all, whether it's the map from pixels to digits or from a board position to a move? the **universal approximation theorem** is the classic answer: a network with a single hidden layer can approximate any continuous function on a bounded region as closely as you like, if it's wide enough. it's a reassuring theorem and a frequently misread one. it says nothing about *how* wide (possibly absurdly), nothing about whether gradient descent will find the weights, and nothing about why practitioners use deep networks instead of one enormous hidden layer.

this topic makes all three points concrete in one dimension, where you can see everything. you'll prove the theorem for ReLU networks by *constructing* the approximating network, a few lines of NumPy. you'll build a deep, narrow network that computes a sawtooth with $2^d$ teeth using only $2d$ units, which any one-hidden-layer network needs exponentially many units to match: a clean example of **depth beating width**. and you'll measure what typical random networks and gradient descent actually do, which is less than the theorems suggest, and more interesting.

## warm-up: predict first

1. a one-hidden-layer ReLU network on a 1-D input has $k$ hidden units. at most how many linear pieces can its output have?
2. you want to approximate $f(x) = \sin x$ on $[0, 2\pi]$ to within $0.01$ with a piecewise-linear function through evenly spaced samples. roughly how many samples do you need: 10, 100 or 10,000?
3. the tent map $T(x) = 2\min(x, 1 - x)$ on $[0, 1]$ has 2 linear pieces. how many does $T(T(T(x)))$ have?
4. a network with 48 ReLU units stacked as 12 layers of 4 and one with a single layer of 48: which typically has more linear pieces at random initialization?
5. if a network can represent a function exactly, will gradient descent find it from a random start?

## prerequisites check

- [ ] you know what continuity and uniform continuity mean, and the Lipschitz condition (→ ch 03 · topics 01 and 05)
- [ ] you can build and train networks with `mlfs.nn` (`Sequential`, `Linear`, `ReLU`, `fit`) (→ ch 14 · topic 04)
- [ ] you know why stacked linear layers collapse and what a hidden layer does (→ ch 14 · topic 01)
- [ ] you can explain the sup norm $\|f - g\|_\infty = \max_x|f(x) - g(x)|$ versus the mean squared error (→ ch 03 · topic 07)

## core ideas

### idea 1: universal approximation, by construction

on an interval, a continuous function can be approximated uniformly by a **piecewise-linear** one: join samples $(x_0, f(x_0)), \dots, (x_{k-1}, f(x_{k-1}))$ by straight lines. if $f$ is $L$-Lipschitz and the knots are $h$ apart, the error is at most $Lh/2$ (on paper 1), so it goes to zero as the knots get denser; uniform continuity gives the same conclusion for any continuous $f$. (warm-up 2: $\sin$ has $L = 1$, so $h = 0.02$, about 300 samples, is guaranteed to do. that's the worst case for a Lipschitz function: for a smooth one the error shrinks like $h^2$, and on paper 1 shows that about 23 samples already suffice for $\sin$.)

and every piecewise-linear function of one variable is exactly a one-hidden-layer ReLU network. sort the knots, let $s_i$ be the slope of segment $i$ (between $x_i$ and $x_{i+1}$), and write

$$g(x) = y_0 + s_0\big(\operatorname{relu}(x - x_0) - \operatorname{relu}(x_0 - x)\big) + \sum_{i=1}^{k-2}(s_i - s_{i-1})\operatorname{relu}(x - x_i).$$

the first pair of units is the line through $(x_0, y_0)$ with slope $s_0$, extending in both directions ($\operatorname{relu}(t) - \operatorname{relu}(-t) = t$); each later unit is zero to the left of its knot and bends the slope from $s_{i-1}$ to $s_i$ at it. so $k$ points need exactly $k$ hidden units, and to the right of the last knot the last segment simply continues. as a network: $W_1 = (1, -1, 1, \dots, 1)^\top$, $b_1 = (-x_0, x_0, -x_1, \dots, -x_{k-2})$, $W_2 = (s_0, -s_0, s_1 - s_0, \dots, s_{k-2} - s_{k-3})$, $b_2 = y_0$.

together: **for any continuous $f$ on $[a, b]$ and any $\varepsilon > 0$, some one-hidden-layer ReLU network is within $\varepsilon$ of $f$ everywhere on $[a, b]$.** that's the universal approximation theorem, in one dimension, for ReLU. the general statements (Cybenko, 1989, for sigmoids; Hornik, 1991; Leshno et al., 1993: any activation that isn't a polynomial) hold in $d$ dimensions on compact sets, with proofs that are less constructive. note what the theorem *doesn't* say: how many units (in $d$ dimensions, the number needed by constructions like this one grows exponentially with $d$), and how to find the weights.

> **▶ build: stage 1.** implement `set_linear(layer, W, b)` (copy weights into one of your `Linear` layers in place) and `relu_interpolant(knots, values)`, returning `Sequential(Linear(1, k), ReLU(), Linear(k, 1))` through the points.

### idea 2: depth multiplies pieces

a one-hidden-layer network with $k$ ReLU units on a 1-D input has at most $k + 1$ linear pieces: each unit adds at most one kink, where its input crosses zero (warm-up 1). a deep network can do much better, because composing functions **multiplies** their pieces.

the tent map $T(x) = 2\operatorname{relu}(x) - 4\operatorname{relu}(x - \tfrac12)$ is $2x$ on $[0, \tfrac12]$ and $2 - 2x$ on $[\tfrac12, 1]$: it maps $[0, 1]$ onto itself twice, once going up and once coming back down. so $T(T(x))$ runs through the whole tent twice, with 4 pieces, and $T^{\circ d}$ is a sawtooth with $2^{d-1}$ teeth and $2^d$ pieces (on paper 3; warm-up 3: 8). each tent costs only 2 ReLU units, so a network of depth $d$ and width 2, with $2d$ hidden units, computes $2^d$ pieces. in the network, the output of one tent ($2h_1 - 4h_2$) feeds both units of the next ($\operatorname{relu}(t)$ and $\operatorname{relu}(t - \tfrac12)$), and those two linear maps merge into a single $2 \times 2$ layer:

$$W = \begin{pmatrix}2&-4\\2&-4\end{pmatrix},\qquad b = (0, -\tfrac12).$$

a shallow network needs at least $2^d - 1$ units just to have enough pieces. Telgarsky (2016) turns this into a theorem: there are functions computed by deep networks of size $O(d)$ that no network with much less depth can approximate without exponentially many units. depth isn't just "more parameters"; it's a different, sometimes exponentially more efficient, way of using them.

> **🔮 predict:** your shallow `relu_interpolant` approximates the depth-5 sawtooth through $k$ evenly spaced samples. at $k = 17$ (one sample every $1/16$), what does the approximation look like?

> **▶ build: stage 2.** implement `sawtooth_net(depth)`: `Linear(1, 2)`, `ReLU`, then `depth - 1` blocks of `Linear(2, 2)` and `ReLU`, then `Linear(2, 1)`, with the weights above.

### idea 3: counting the pieces

to see these numbers in real networks, count the pieces. a ReLU network is linear wherever no unit switches between on and off, so along a line, its linear pieces correspond to runs of points with the same **activation pattern**: the vector of on/off states ($\text{output} > 0$) of every ReLU unit in every layer. `count_linear_regions(model, x)` runs the sorted points through the `Sequential` layer by layer, records the pattern after every `ReLU`, and returns 1 plus the number of times the pattern changes between consecutive points. (two neighbouring regions can happen to have the same slope, so strictly this counts regions, an upper bound on distinct pieces; and a region narrower than the grid spacing is missed. on a fine grid both effects are negligible here.)

the theory of piece counts says that a ReLU network with $L$ layers of width $w$ *can* have a number of pieces exponential in $L$, as the sawtooth does. what's typical is another matter. Hanin & Rolnick (2019) showed that at initialization (and, empirically, through training) the expected number of pieces along a line grows roughly like the *total number of units*, not exponentially in depth. the exponential count needs carefully aligned weights: each layer's kinks have to land in every piece the previous layers produced. the lab measures this (warm-up 4).

> **🔮 predict:** 48 ReLU units with random weights, arranged as one layer of 48 or as 12 layers of 4. which has more linear pieces along $[-3, 3]$, and is either close to its maximum possible count?

> **▶ build: stage 3.** implement `count_linear_regions(model, x)`.

### idea 4: approximating isn't learning

the theorem promises that weights exist. in practice we look for them by gradient descent on a sampled loss, and that's a different problem. `fit_function(model, f, ...)` trains a network on $n$ evenly spaced samples of $f$ with Adam and the mean squared error, using your `fit`; `max_error(model, f, ...)` measures the sup-norm error on a fine grid, which is what the theorem is about (MSE can be small while the max error is large, on one bad spike).

three things can go wrong even when a good solution exists. the optimization can stall in a poor local minimum or on a plateau (topic 01's lab saw this on XOR). ReLU units can **die**: a unit whose input is negative for every training point outputs 0 and gets zero gradient, so nothing ever revives it; deep, narrow networks lose units this way easily, sometimes all of a layer's, which cuts the network in two. and high-frequency functions are learned slowly: networks fit smooth, low-frequency components first (the "spectral bias"), so a sawtooth with many teeth looks like noise around its mean for a long time. the lab trains a deep and a shallow network with the same number of parameters on sawtooths, and the outcome is the opposite of what the representation results suggest (warm-up 5).

> **▶ build: stage 4.** implement `fit_function(model, f, *, lo=0.0, hi=1.0, n_points=256, epochs=500, lr=0.01, batch_size=64, rng=None)` and `max_error(model, f, *, lo=0.0, hi=1.0, n_points=2001)`.

## worked example

the network through $(0, 1)$, $(1, 3)$ and $(3, 2)$. the slopes are $s_0 = (3 - 1)/1 = 2$ and $s_1 = (2 - 3)/2 = -\tfrac12$, so

$$g(x) = 1 + 2\big(\operatorname{relu}(x) - \operatorname{relu}(-x)\big) - \tfrac52\operatorname{relu}(x - 1),$$

i.e. $W_1 = (1, -1, 1)^\top$, $b_1 = (0, 0, -1)$, $W_2 = (2, -2, -\tfrac52)$, $b_2 = 1$. check the points: $g(0) = 1$; $g(1) = 1 + 2 = 3$; $g(3) = 1 + 6 - 5 = 2$. beyond the knots: $g(-2) = 1 - 4 = -3$ (the first segment's line, slope 2), and $g(4) = 1 + 8 - \tfrac{15}{2} = 1.5$, $g(5) = 1 + 10 - 10 = 1$ (the last segment's slope, $-\tfrac12$). the network has 3 units for 3 points and, along the real line, kinks only at $x = 0$ (where two units switch together) and $x = 1$: 3 linear pieces. the tests check all of these values, and that your interpolant passes exactly through 8 random points.

## on paper

1. **the interpolation error.** $f$ is $L$-Lipschitz on $[x_i, x_i + h]$ and $p$ is the line through $(x_i, f(x_i))$ and $(x_i + h, f(x_i + h))$. show that $|f(x) - p(x)| \le Lh/2$ on the interval. how many evenly spaced knots does $f(x) = \sin x$ on $[0, 2\pi]$ need for error $0.01$?

<details><summary>solution</summary>

write $x = x_i + th$ with $t \in [0, 1]$; then $p(x) = (1 - t)f(x_i) + tf(x_i + h)$, so $p(x) - f(x) = (1 - t)(f(x_i) - f(x)) + t(f(x_i + h) - f(x))$. by the Lipschitz bound this is at most $(1 - t)Lth + tL(1 - t)h = 2Lht(1 - t) \le Lh/2$ (maximal at $t = \tfrac12$). for $\sin$, $L = 1$: $h \le 0.02$, so about $2\pi/0.02 \approx 315$ intervals, 316 knots. (for twice-differentiable $f$ the error is $O(h^2)$, so far fewer would do: $h^2\max|f''|/8 \le 0.01$ gives $h \approx 0.28$, about 23 knots. the Lipschitz bound is for the worst case.)

</details>

2. **pieces of a shallow network.** show that $x \mapsto \sum_{j=1}^k a_j\operatorname{relu}(w_jx + b_j) + c$ has at most $k + 1$ linear pieces on $\mathbb R$.

<details><summary>solution</summary>

unit $j$ is linear on each side of $x = -b_j/w_j$ (if $w_j \ne 0$; otherwise it's constant). a sum of functions that are all linear on an interval is linear there, so the output is linear on each interval between consecutive kinks. $k$ units give at most $k$ distinct kink locations, which cut the line into at most $k + 1$ intervals.

</details>

3. **the sawtooth.** prove by induction that $T^{\circ d}$ is linear on each interval $[j/2^d, (j + 1)/2^d]$, alternating between going up from 0 to 1 and down from 1 to 0, so it has $2^d$ pieces on $[0, 1]$.

<details><summary>solution</summary>

$d = 1$: $T$ goes up on $[0, \tfrac12]$ and down on $[\tfrac12, 1]$. step: $T^{\circ(d+1)} = T^{\circ d} \circ T$. on $[0, \tfrac12]$, $T(x) = 2x$ maps $[j/2^{d+1}, (j + 1)/2^{d+1}]$ linearly onto $[j/2^d, (j + 1)/2^d]$, where $T^{\circ d}$ is a full up or down piece; on $[\tfrac12, 1]$, $T(x) = 2 - 2x$ does the same in reverse order. so $T^{\circ(d+1)}$ has $2^d$ pieces on each half, $2^{d+1}$ in total, alternating.

</details>

4. **composition and pieces.** if $f: \mathbb R \to \mathbb R$ has $p$ linear pieces and $g$ has $q$, show that $g \circ f$ has at most $pq$ pieces, and explain why the tent map achieves the bound.

<details><summary>solution</summary>

on each of $f$'s $p$ pieces, $f$ is linear, hence monotone, so it crosses each of $g$'s $q - 1$ kinks at most once: that piece of $f$ is split into at most $q$ pieces of $g \circ f$. total $\le pq$. the tent map achieves it because each of its pieces maps onto the *whole* interval $[0, 1]$, so every kink of the next tent is crossed by every piece: $2^d \cdot 2 = 2^{d+1}$.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. the depth-$d$ sawtooth against shallow interpolants with $k$ units: the max error as a function of $k$, and the units each architecture needs.
2. the number of linear pieces of random ReLU networks with 48 units, stacked from 1 to 12 layers.
3. a shallow and a deep network with the same 73 parameters, trained on sawtooths of depth 2 and 3 from 5 random initializations.

## check your understanding

1. the universal approximation theorem holds for one hidden layer. why use deep networks?

<details><summary>answer</summary>

because "wide enough" can mean exponentially wide. some functions (like the sawtooth, and plausibly many functions with compositional structure) need exponentially fewer units when computed by a deep network. depth also seems to help learning and generalization in practice, for reasons that are still being worked out (chapter 19).

</details>

2. why does the theorem need a *non-linear* activation?

<details><summary>answer</summary>

with a linear activation, the whole network is one linear map (topic 01), and linear functions can't approximate, say, $|x|$ on $[-1, 1]$ better than a constant error. any non-polynomial activation is enough (Leshno et al., 1993); polynomials of bounded degree fail because sums of them stay polynomials of that degree.

</details>

3. your shallow interpolant of the depth-5 sawtooth with 17 evenly spaced knots is identically 0. why?

<details><summary>answer</summary>

the knots $j/16$ are exactly the sawtooth's troughs, where it's 0. sampling a signal at a rate that matches its period sees only one phase of it: aliasing. with 33 knots (every $1/32$) the samples hit every peak and trough, and the interpolant is exact.

</details>

4. a ReLU network is linear on each region of constant activation pattern. why?

<details><summary>answer</summary>

if the on/off state of every unit is fixed, each ReLU acts as either the identity or zero, so every layer is a fixed linear (affine) map, and their composition is affine.

</details>

5. why can't gradient descent revive a ReLU unit that's off for every training input?

<details><summary>answer</summary>

its output is 0 for every input, and so is its derivative, so the loss's gradient with respect to its incoming weights and bias is exactly 0. nothing in the update moves it.

</details>

6. `fit_function` reports a small final MSE, but `max_error` is 0.3. how can both be true?

<details><summary>answer</summary>

MSE averages the squared error, so a narrow region with a large error (a missed spike, a tooth of a sawtooth, a sharp corner) contributes little. the sup norm looks only at the worst point. the theorem is about the sup norm, which is the stricter measure.

</details>

## common bugs

- **unsorted knots.** symptom: the interpolant's slopes are computed between the wrong neighbours and it misses most points. sort the knots and reorder the values with them.
- **forgetting the left extension.** symptom: correct between the knots and to the right, but constant to the left of the first knot. the pair $\operatorname{relu}(x - x_0) - \operatorname{relu}(x_0 - x)$ is what makes the first segment's line continue in both directions.
- **separate layers for "tent output" and "next tent input".** symptom: correct values, but twice as many `Linear` layers as needed, with width-1 bottlenecks between them. merge them into one $(2, 2)$ layer.
- **counting from the first layer only.** symptom: a deep sawtooth reports 2 pieces. record the pattern after *every* `ReLU`.
- **grids that hit the kinks exactly.** symptom: off-by-a-few piece counts. at an exact kink a unit's input is 0 and counts as off, which can create a region of width zero. use grids that avoid special points (the tests use `np.linspace(0, 1, 10_000)`, whose interior points are never dyadic fractions).

## going deeper

- Cybenko (1989), "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals and Systems* 2: 303–314. the classic universal approximation theorem. [doi:10.1007/BF02551274](https://doi.org/10.1007/BF02551274)
- Hornik (1991), "Approximation capabilities of multilayer feedforward networks", *Neural Networks* 4(2): 251–257. [doi:10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)
- Leshno, Lin, Pinkus & Schocken (1993), "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function", *Neural Networks* 6(6): 861–867. [doi:10.1016/S0893-6080(05)80131-5](https://doi.org/10.1016/S0893-6080(05)80131-5)
- Telgarsky (2016), "Benefits of depth in neural networks", *COLT*. the sawtooth separation between deep and shallow networks. [arXiv:1602.04485](https://arxiv.org/abs/1602.04485)
- Montúfar, Pascanu, Cho & Bengio (2014), "On the number of linear regions of deep neural networks", *NeurIPS*. [arXiv:1402.1869](https://arxiv.org/abs/1402.1869)
- Hanin & Rolnick (2019), "Complexity of linear regions in deep networks", *ICML*. typical networks have far fewer regions than the maximum. [arXiv:1901.09021](https://arxiv.org/abs/1901.09021)
- Rahaman et al. (2019), "On the spectral bias of neural networks", *ICML*. why low frequencies are learned first. [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)

## connections

- ← builds on: `mlfs.nn` layers and `fit` (topic 04), the hidden layer as a learned representation (topic 01), Lipschitz continuity and uniform convergence (ch 03 · topics 01, 05 and 07).
- → used in: initialization and dead units (ch 15 · topics 01 and 05), depth and residual connections (ch 16), the theory of deep learning: expressivity, the neural tangent kernel and scaling (ch 19).
