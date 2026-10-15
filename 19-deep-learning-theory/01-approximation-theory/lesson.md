# approximation theory

## why this matters

before asking why deep networks generalize or why gradient descent trains them, there's a more basic question: what can they represent at all? the classical answer, the **universal approximation theorem**, says that a network with a single hidden layer can approximate any continuous function on a bounded domain as closely as you like, given enough units. that answers "can it?" and leaves open everything that matters in practice: how many units, how the number grows with the dimension and the roughness of the target, and whether depth buys anything that width can't.

depth does buy something, and you can see exactly what in one dimension. a ReLU network is a piecewise-linear function, so its power is how many linear pieces it can make per weight. a shallow network adds pieces one unit at a time; a deep one can *compose* folds and double the count with every layer. Telgarsky (2016) turned this into a theorem: there are functions that a deep, narrow network computes with a few dozen weights and that any shallow network needs exponentially many units to approximate. this topic builds both networks by hand, writes a counter for linear pieces, and then runs the experiment people usually skip: training. you'll find a function that depth makes cheap to represent and that gradient descent still can't learn, a different function that depth makes easy to learn at a fixed budget, and that random deep networks don't have anything like the exponential number of pieces that the theory allows.

## warm-up: predict first

1. a ReLU network with one hidden layer of $m$ units and a 1-D input: at most how many linear pieces can its output have?
2. you compose the "tent" map $t(x) = 1 - |2x - 1|$ with itself $k$ times. how many linear pieces does $t^k$ have on $[0, 1]$?
3. a function on $[0, 1]^d$ is only known to be Lipschitz. roughly how many units does a shallow network need to approximate it to accuracy $\varepsilon$, as a function of $d$?
4. a deep network with 8 layers of 6 units, randomly initialized: how many linear pieces does it have on an interval, compared with a one-layer network of 48 units?
5. if a network of some shape can represent a function exactly, will gradient descent from a random start find it?

## prerequisites check

- [ ] you know what a ReLU MLP computes and can build one with `nn.Sequential` (→ ch 14 · topic 01, ch 15 · topic 07)
- [ ] you can write a piecewise-linear interpolant and know `np.interp` (→ ch 01)
- [ ] you know the Fourier transform and why smooth functions have rapidly decaying spectra (→ ch 03 · topic 06)
- [ ] you can count parameters of a linear layer, weights plus biases (→ ch 14 · topic 01)
- [ ] you know Adam and cosine learning-rate schedules (→ ch 15 · topic 04)

## core ideas

### idea 1: one hidden layer can fit anything, one unit per kink

a ReLU unit $\text{relu}(wx + b)$ of a scalar input is zero on one side of the point $-b/w$ and linear on the other: a single kink. a sum of $m$ such units plus a constant is piecewise linear with at most $m$ kinks, so at most $m + 1$ pieces. conversely, any piecewise-linear function with $m$ kinks can be written that way, and that is a complete, constructive universal approximation theorem in 1-D. given knots $x_0 < x_1 < \dots < x_{n-1}$ with values $y_i$, the interpolant has slopes $s_j = (y_{j+1} - y_j)/(x_{j+1} - x_j)$ and

$$f(x) = y_0 + s_0\,\text{relu}(x - x_0) + \sum_{j=1}^{n-2}(s_j - s_{j-1})\,\text{relu}(x - x_j).$$

each unit switches on at a knot and adds the *change* of slope there. that's $n - 1$ hidden units for $n$ knots, and you can't do with fewer, because each kink needs a unit. left of $x_0$ every unit is off and $f = y_0$; right of $x_{n-1}$ the last slope continues. since continuous functions on an interval are uniformly approximated by interpolants on fine enough grids, one hidden layer approximates any continuous function of one variable.

in $d$ dimensions the same is true (Cybenko, 1989, for sigmoids; Hornik, 1991; Leshno et al., 1993, for any non-polynomial activation), but the count changes character. for a target known only to be Lipschitz, you need about $\varepsilon^{-d}$ units for accuracy $\varepsilon$: the curse of dimensionality, no better than a grid. Barron (1993) found a class that escapes it: functions whose Fourier transform has a finite first moment, $C_f = \int\lVert\omega\rVert\,|\hat f(\omega)|\,d\omega < \infty$. for them a shallow network with $n$ units reaches squared error at most about $C_f^2/n$ (times the squared radius of the domain), with *no* dependence on $d$ in the rate (on paper 4 shows why). Barron spaces are the modern name for this class, and they say which functions are "easy" for shallow networks: smooth ones without high-frequency structure in many directions.

> **spec.** `relu_interpolant(knots_x, knots_y)` returns `nn.Sequential(nn.Linear(1, n-1), nn.ReLU(), nn.Linear(n-1, 1))` whose output equals the piecewise-linear interpolant on $[x_0, x_{n-1}]$, is constant $y_0$ left of $x_0$ and continues the last segment right of $x_{n-1}$. it raises `ValueError` unless there are $n \ge 2$ knots with strictly increasing `knots_x` and as many `knots_y`.

> **▶ build: stage 1.** implement `relu_interpolant`.

### idea 2: depth composes, and composition multiplies

the tent map $t(x) = 1 - |2x - 1|$ maps $[0, 1]$ onto itself twice: up from 0 to 1 on the left half, down on the right half. composing it with itself, each half of $t$ sweeps the whole interval, so $t \circ t$ has 4 pieces, and $t^k$ has $2^k$: a sawtooth with $2^{k-1}$ teeth. on $[0, 1]$ the tent is two ReLUs, $t(x) = 2\,\text{relu}(x) - 4\,\text{relu}(x - \tfrac12)$, so $t^k$ is a network with $k$ hidden layers of width 2: layer 1 computes $(\text{relu}(x), \text{relu}(x - \frac12))$, each later layer computes the same pair from $t$ of the previous output, $t = 2h_1 - 4h_2$, and the output layer forms the last $t$. that's $4 + 6(k-1) + 3 = 6k + 1$ weights for $2^k$ pieces.

a one-hidden-layer network needs $2^k - 1$ units for those $2^k$ pieces, about $3 \cdot 2^k$ weights: exponentially more. Telgarsky (2016) proved that no shallow network does much better *approximately* either: a network with fewer than about $2^k$ pieces has to miss a constant fraction of the sawtooth's teeth, so its error stays bounded away from zero (on paper 3). more generally, there are functions computable by networks of depth $k^3$ and polynomial size that need exponential size at depth $k$. depth separations are now known for many targets; Eldan & Shamir (2016) gave one in the dimension rather than the number of pieces, a radial function on $\mathbb R^d$ that a 3-layer network of polynomial width approximates and a 2-layer network needs width exponential in $d$ for, and the lab trains networks on a target of that kind.

> **spec.** `sawtooth_net(k)` returns an `nn.Sequential` of $k$ hidden `Linear` + `ReLU` layers of width 2 and a `Linear(2, 1)` output ($6k + 1$ parameters) computing $t^k$ on $[0, 1]$; `ValueError` for $k < 1$.

> **🔮 predict:** `sawtooth_net(8)` has 49 weights. how many weights does `relu_interpolant` need for the same function on $[0, 1]$?

> **▶ build: stage 2.** implement `sawtooth_net`.

### idea 3: counting the pieces of a network

to compare networks you want to count the linear pieces they actually have on an interval, not the theoretical maximum. a numerical counter: evaluate $f$ on a fine grid and look at the **second differences** $v_{i+1} - 2v_i + v_{i-1}$. on a linear piece they're zero (up to rounding); at a kink they're about $h \cdot \Delta s$, the grid step times the change of slope. two details make the count right:

- **a kink between grid points bends two second differences**, the two on either side, so count *runs* of consecutive large second differences, not points. pieces = runs + 1.
- **"large" is relative.** compare $|v_{i+1} - 2v_i + v_{i-1}|$ with `tol` times the largest first difference, so the answer doesn't change when you multiply the function by $10^5$. and evaluate in float64: in float32 a nearly flat function's outputs are rounded to a staircase whose every step looks like a kink. for an `nn.Module`, run a float64 copy (`copy.deepcopy(f).double()`) so the original is untouched.

the count is a **lower bound**, for two reasons. kinks less than about three grid steps apart light up overlapping second differences and merge into one run; and the tolerance is effectively between `tol` and `2·tol` of the steepest slope, because a kink that falls between grid points splits its second difference over two points (anything with a smaller slope change is missed). on the sawtooth with $2^k \le 2^{14}$ pieces on a grid of $2^{14} + 1$ points it's exact (every kink is a grid point, and kinks are $2^{14-k}$ steps apart).

how many pieces *can* a deep network have? in 1-D, each layer of width $w$ multiplies the number of pieces by at most about $2w$, so $L$ layers give at most $(2w)^L$ (Telgarsky, 2016; Montúfar et al., 2014, for regions in $d$ dimensions). the sawtooth gets $2^L$ pieces from width 2, far below the bound's $4^L$ but exponential in depth, which is what matters against a shallow network's linear count. whether typical networks come close is a separate question, which the lab answers.

> **spec.** `count_linear_pieces(f, lo=0.0, hi=1.0, *, n_grid=16385, tol=1e-4)`: evaluate `f` on `torch.linspace(lo, hi, n_grid, dtype=torch.float64)` reshaped to `(n_grid, 1)` (an `nn.Module` is evaluated on a float64 copy; `f` itself is not modified), take the second differences of the outputs, mark the points where their absolute value exceeds `tol` times the largest absolute first difference, and return 1 + the number of maximal runs of consecutive marked points (1 for a function with no change of slope).

> **▶ build: stage 3.** implement `count_linear_pieces`.

### idea 4: comparing architectures at an equal budget

"deep beats shallow" only means something at an equal cost. the cleanest cost is the parameter count. an MLP with input dimension $d_{in}$, $L$ hidden layers of width $w$ and $d_{out}$ outputs has

$$P(w) = (d_{in} + 1)\,w + (L - 1)(w + 1)\,w + (w + 1)\,d_{out}$$

parameters (weights and biases), increasing in $w$. for a budget $B$, pick the width whose count is closest to $B$. at 3,000 parameters and $d_{in} = 4$, a one-layer network gets 500 units and a four-layer one 30 per layer: that's the comparison the lab runs. parameter count isn't the only fair measure (compute per example and memory differ), but it's the one approximation theorems count.

> **spec.** `mlp_param_count(width, depth, *, in_dim=1, out_dim=1)` returns $P(w)$ above. `width_for_budget(budget, depth, *, in_dim=1, out_dim=1)` returns the width $w \ge 1$ minimizing $|P(w) - \text{budget}|$, the smaller width on a tie.

> **▶ build: stage 4.** implement `mlp_param_count` and `width_for_budget`.

### idea 5: representing is not learning

a theorem that a network of some shape *can* compute $f$ says nothing about whether training will find those weights. to find out, you need a training loop you can trust and reuse: minibatch mean squared error, Adam, a cosine learning-rate decay (from `lr` at step 0 towards 0 at the end), batches drawn with replacement from a seeded generator so that two runs with the same seed are identical, the model in train mode while training and in eval mode afterwards. nothing about it is specific to approximation theory; what's specific is what you feed it.

the lab's answer is uncomfortable for depth-separation results. the sawtooth that `sawtooth_net` computes with 37 weights is, for gradient descent from a random start, essentially unlearnable: deep and shallow networks alike end up predicting the mean. its loss surface is dominated by plateaus (most random settings of a deep width-2 network produce a nearly constant function, and folding the input six times requires every layer to be just right at once). other targets separate depth where it counts, in learnability at an equal budget: the radial ripple of the lab, which deep networks fit to a few percent error and shallow ones not at all. and random deep networks have far fewer pieces than the $(2w)^L$ bound: Hanin & Rolnick (2019) proved that at initialization the expected number of pieces along a line grows with the *total* number of neurons, not exponentially with depth. depth's exponential expressivity exists, but only on a thin set of carefully tuned weights.

> **spec.** `fit_regressor(model, x, y, *, steps=2000, lr=3e-3, batch_size=256, seed=0)` trains `model` with `torch.optim.Adam` on the minibatch mean squared error between `model(x[idx])` and `y[idx]` (`y` may be `(n,)` or `(n, 1)`), for `steps` steps; at step $t = 0, \dots, \text{steps} - 1$ the learning rate is $\text{lr}\cdot\frac12\big(1 + \cos(\pi t/\text{steps})\big)$; each minibatch is `min(batch_size, n)` indices drawn with replacement using a `torch.Generator` seeded with `seed`. the model is in train mode during training and eval mode afterwards. returns the list of the `steps` minibatch losses (floats).

> **🔮 predict:** at 3,000 parameters, how much lower do you expect the test error of a 4-layer network to be than a 1-layer network's on a radial ripple in 4 dimensions: 10% lower, 2× lower, 10× lower?

> **▶ build: stage 5.** implement `fit_regressor`.

## worked example

the interpolant through $(0, 0), (1, 2), (3, 0), (4, 4)$: slopes $s = (2, -1, 4)$, so the output weights are $s_0 = 2$, $s_1 - s_0 = -3$, $s_2 - s_1 = 5$, the hidden biases are $0, -1, -3$, and

$$f(x) = 2\,\text{relu}(x) - 3\,\text{relu}(x - 1) + 5\,\text{relu}(x - 3).$$

check: $f(2) = 4 - 3 = 1$ (halfway down from 2 to 0), $f(3.5) = 7 - 7.5 + 2.5 = 2$ (halfway up from 0 to 4), $f(-1) = 0$ (flat on the left), $f(5) = 10 - 12 + 10 = 8$ (the last slope continues). three hidden units, three kinks, four pieces. your `relu_interpolant` must produce these values (the stage 1 check), and `count_linear_pieces` must find exactly $2^k$ pieces in `sawtooth_net(k)`.

## on paper

1. **the sawtooth.** show that $t^k$ has exactly $2^k$ linear pieces on $[0, 1]$, all of slope $\pm 2^k$, with kinks at the multiples of $2^{-k}$.

<details><summary>solution</summary>

by induction: $t^k$ is linear of slope $\pm 2^k$ on each interval $[j2^{-k}, (j+1)2^{-k}]$ and maps it onto $[0, 1]$. applying $t$ (slope $\pm 2$, kink at $\frac12$) to a linear map onto $[0, 1]$ splits the interval at the preimage of $\frac12$, its midpoint, and doubles the slope. so $t^{k+1}$ has $2 \cdot 2^k$ pieces of slope $\pm 2^{k+1}$ with kinks at the multiples of $2^{-(k+1)}$.

</details>

2. **pieces of a deep network.** a 1-D input passes through layers of width $w$. if the input to a layer is piecewise linear with $p$ pieces (in each coordinate, with shared breakpoints), show that after the layer (affine, then ReLU) the output has at most $p(w + 1)$ pieces. conclude a bound of $(w + 1)^L$ for $L$ layers.

<details><summary>solution</summary>

on each of the $p$ pieces, each of the $w$ pre-activations is linear in $x$, so it crosses zero at most once: at most $w$ new breakpoints per piece, splitting it into at most $w + 1$. after the ReLU every coordinate is linear between consecutive breakpoints. so $p \to p(w + 1)$, and $L$ layers starting from $p = 1$ give at most $(w + 1)^L$ pieces. the output layer is affine and adds none. (Telgarsky's count, $(2w)^L$, is looser but more general: it covers inputs whose coordinates have different breakpoints. either way the point is the exponent $L$.)

</details>

3. **why few pieces can't approximate many teeth.** let $g$ have at most $m$ linear pieces on $[0, 1]$ and let $t^k$ have $2^k$ pieces, alternating between 0 and 1. show that if $m < 2^{k-1}$, then $g$ has error $\int_0^1 |g - t^k| \ge c$ for a constant $c > 0$ independent of $k$.

<details><summary>solution</summary>

$t^k$ consists of $2^k$ monotone pieces, each of length $2^{-k}$ and sweeping from 0 to 1 or back. $g - \frac12$ is piecewise linear with $m$ pieces, so it changes sign at most $m$ times, and at most $m$ of the sawtooth's pieces contain a sign change. on each of the remaining $2^k - m$ pieces, $g$ stays on one side of $\frac12$, say above it; then on the half of the piece where $t^k < \frac12$, $|g - t^k| \ge \frac12 - t^k$, which integrates to $\frac12 \cdot 2^{-k} \cdot \frac14 = 2^{-k}/8$. so $\int_0^1|g - t^k| \ge (2^k - m)\,2^{-k}/8$, which is at least $\frac1{16}$ when $m \le 2^{k-1}$, for every $k$. a shallow network with $u$ units has $u + 1$ pieces, so getting below error $\frac1{16}$ needs $u \ge 2^{k-1}$ units: exponentially many, against $6k + 1$ weights for the deep network.

</details>

4. **why Barron's rate is dimension-free (Maurey's sampling argument).** suppose $f(x) = \mathbb E_{\theta \sim \mu}[g(x; \theta)]$ is an average of "atoms" $g(x; \theta) = c\,\sigma(w\cdot x + b)$ with $|g| \le B$ for all $x$ in the domain. draw $\theta_1, \dots, \theta_n$ independently from $\mu$ and set $f_n = \frac1n\sum_j g(\cdot\,; \theta_j)$, a network with $n$ units. show that $\mathbb E\,\lVert f - f_n\rVert^2_{L^2(P)} \le B^2/n$ for any input distribution $P$, and conclude that some network with $n$ units achieves it. where would the dimension enter?

<details><summary>solution</summary>

for each fixed $x$, $f_n(x)$ is an average of $n$ i.i.d. variables with mean $f(x)$ and variance at most $B^2$, so $\mathbb E[(f(x) - f_n(x))^2] \le B^2/n$. integrate over $x \sim P$ and swap the expectations: $\mathbb E\lVert f - f_n\rVert^2 \le B^2/n$. an average over random draws below $B^2/n$ means at least one draw is below it: a network with $n$ units and squared error at most $B^2/n$ exists. nothing in the argument mentions $d$: it's a Monte Carlo average. the dimension hides in $B$, i.e. in how large the weights of the representing distribution $\mu$ must be; Barron showed that $C_f = \int\lVert\omega\rVert|\hat f(\omega)|d\omega$ controls it (for a domain of radius $r$, $B \approx 2rC_f$), and for oscillating radial functions $C_f$ grows exponentially with $d$.

</details>

## experiment

the lab tabulates weights and counted pieces of `sawtooth_net(k)` and of the shallow interpolant for $k = 1$ to 8, then trains deep and shallow networks on the sawtooth from random starts. it compares one- to four-layer networks at a fixed budget of 3,000 parameters on a radial ripple in four dimensions, three seeds each. finally it counts the pieces of randomly initialized networks with 48 units arranged at different depths.

## check your understanding

1. the universal approximation theorem guarantees a shallow network approximates any continuous function. why isn't that a reason to use shallow networks?

<details><summary>answer</summary>

it says nothing about the number of units, which can grow exponentially with the dimension (Lipschitz targets) or with the number of oscillations (the sawtooth). a theorem of the form "enough units suffice" is about existence; practical architecture choices are about the rate at which error falls with size, and that's where depth, structure (convolutions, attention) and the target's regularity matter.

</details>

2. your piece counter reports 17 pieces for a function you built with 16. what are two likely bugs?

<details><summary>answer</summary>

counting marked points instead of runs (a kink between grid points marks two consecutive second differences), or a threshold so small that rounding noise counts (especially in float32, where a flat region's outputs form a staircase). the opposite error, too few pieces, comes from a grid coarser than the spacing of kinks, or a tolerance that hides small slope changes.

</details>

3. a deep width-2 network *can* represent the 64-piece sawtooth. why does training it from a random start fail?

<details><summary>answer</summary>

with random weights, a deep narrow ReLU network usually produces a nearly constant function: units die (always negative) and the few active paths compose into something simple, so the gradient carries little information about the six folds the target needs. the target's structure is all-or-nothing (a fold that's slightly wrong in layer 2 scrambles everything downstream), so there's no gradual path from a random start to the solution. width and good initialization help; exact recovery from random starts is still rare.

</details>

4. Hanin & Rolnick showed the expected number of pieces of a random network grows with the number of neurons, not exponentially with depth. does that contradict Telgarsky?

<details><summary>answer</summary>

no. Telgarsky's result is about the maximum over weights, and the sawtooth's weights are a measure-zero, finely tuned choice. Hanin & Rolnick's is about typical weights at initialization (and they found training doesn't change it much). together: depth makes some functions cheap, but a typical deep network doesn't use that capacity, which is one reason the practical benefits of depth are thought to come from elsewhere (composition of features, optimization).

</details>

5. the radial ripple is learnable by deep networks and the sawtooth isn't, though both are "deep" functions. what's different?

<details><summary>answer</summary>

the radial target is a composition of smooth, low-frequency pieces: a sum of per-coordinate functions ($\lVert x\rVert^2 = \sum x_i^2$) followed by a 1-D function with a couple of oscillations, and each stage gives the gradient a useful signal on its own. the sawtooth's frequency doubles at every stage, so the target has $2^{k-1}$ teeth and no stage can be fitted partially. learnability depends on how gradually the function can be approached, not just on how compactly it can be written.

</details>

## common bugs

- **using the slopes instead of the slope changes** as the interpolant's output weights; the pieces after the first come out with doubled slopes.
- **one hidden unit too many** (a unit at the last knot): harmless for the values but wrong for the count, which is the point of the construction.
- **folding with the wrong weights** in the sawtooth: the next layer needs $2h_1 - 4h_2$, the whole tent, not $h_1 - h_2$.
- **counting points instead of runs**, or a float32 grid: both inflate the piece count.
- **comparing architectures at different budgets**, or at equal width rather than equal parameters.
- **a training loop that never calls `model.train()`**: fine for these models, wrong for anything with dropout or batch norm.

## going deeper

- Cybenko (1989), "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals and Systems* 2(4): 303–314. [doi:10.1007/BF02551274](https://doi.org/10.1007/BF02551274)
- Hornik (1991), "Approximation capabilities of multilayer feedforward networks", *Neural Networks* 4(2): 251–257. [doi:10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)
- Leshno, Lin, Pinkus & Schocken (1993), "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function", *Neural Networks* 6(6): 861–867. [doi:10.1016/S0893-6080(05)80131-5](https://doi.org/10.1016/S0893-6080(05)80131-5)
- Barron (1993), "Universal approximation bounds for superpositions of a sigmoidal function", *IEEE Transactions on Information Theory* 39(3): 930–945. [doi:10.1109/18.256500](https://doi.org/10.1109/18.256500)
- Telgarsky (2016), "Benefits of depth in neural networks", *COLT*. [arXiv:1602.04485](https://arxiv.org/abs/1602.04485)
- Eldan & Shamir (2016), "The power of depth for feedforward neural networks", *COLT*. [arXiv:1512.03965](https://arxiv.org/abs/1512.03965)
- Montúfar, Pascanu, Cho & Bengio (2014), "On the number of linear regions of deep neural networks", *NeurIPS*. [arXiv:1402.1869](https://arxiv.org/abs/1402.1869)
- Hanin & Rolnick (2019), "Complexity of linear regions in deep networks", *ICML*. [arXiv:1901.09021](https://arxiv.org/abs/1901.09021)
- Safran & Shamir (2017), "Depth-width tradeoffs in approximating natural functions with neural networks", *ICML*. [arXiv:1610.09887](https://arxiv.org/abs/1610.09887)
- E, Ma & Wu (2022), "The Barron space and the flow-induced function spaces for neural network models", *Constructive Approximation* 55. [arXiv:1906.08039](https://arxiv.org/abs/1906.08039)

## connections

- ← builds on: MLPs and ReLUs (ch 14), PyTorch modules and `MLP` (ch 15 · 07), Adam and cosine schedules (ch 15 · 04), Fourier transforms (ch 03).
- → used in: topic 03 (infinitely wide shallow networks as kernels), topic 06 (why depth needs careful initialization to propagate signal at all), CNNs and transformers as structured ways to spend depth (ch 16, 18), and the expressivity of graph networks (ch 20 · 04).
