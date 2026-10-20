# loss landscapes

## why this matters

training a network means walking downhill on a surface over its weights: 26,506 dimensions for the small MNIST network of this topic's lab, billions for a language model. almost every story about why deep learning works is a story about that surface's shape. "SGD finds flat minima, and flat minima generalize." "large batches find sharp minima, which is why they generalize worse" (Keskar et al., 2017). "there are no bad local minima in big networks." "all the good solutions are connected." some of these are true, some are half true, and one of them depends on a quantity that you can multiply by five, or by anything you like, without changing what the network computes. this topic builds the instruments to check such claims on any PyTorch model: meaningful pictures of the surface, the straight path between two solutions, and exact sharpness without forming the Hessian.

## warm-up: predict first

1. you plot the loss along one random direction $d$ through a trained network's weights $\theta$: $L(\theta + \alpha d)$ for $\alpha \in [-1, 1]$. the first layer's weights are 10 times larger than the last layer's. which layer does the curve mostly tell you about?
2. gradient descent with learning rate $\eta$ on the 1-D quadratic $\tfrac{\lambda}{2} w^2$. for which $\lambda$ does it converge?
3. you have a function that returns the gradient of the loss. can you compute the product of the Hessian with a vector $v$ without building the Hessian? at about what cost?
4. two networks, trained from different random initializations, both get 90% test accuracy. you average their weights. what accuracy do you expect?
5. a minimum is "sharp" if the loss rises quickly around it. can you make a minimum sharper without changing the function the network computes?

## prerequisites check

- [ ] you know what the Hessian is and why its eigenvalues describe curvature (→ ch 03 · topic 04)
- [ ] you can run power iteration and know it finds the eigenvalue of largest magnitude (→ ch 02 · topic 05)
- [ ] you know why gradient descent on a quadratic diverges when the step is too large (→ ch 06 · topic 02)
- [ ] you can train a PyTorch model and read and load a `state_dict` (→ ch 15 · topic 07)
- [ ] you know what reverse-mode autodiff computes and what it costs (→ ch 14 · topic 02)

## core ideas

### idea 1: a slice through a million dimensions, at the right scale

you can't plot a function of 26,506 variables, but you can plot it along a line or a plane. pick the trained weights $\theta^*$ and a direction $d$ (a vector with one entry per weight), and plot

$$f(\alpha) = L(\theta^* + \alpha d), \qquad \alpha \in [-1, 1].$$

with two directions you get a surface $L(\theta^* + \alpha d_1 + \beta d_2)$ that you can draw as contours. random directions are the natural choice: in high dimension two random Gaussian vectors are almost orthogonal (their cosine is about $1/\sqrt{n}$), so a random plane is a fair, unbiased sample of the neighborhood. (it is an *average* view, though: check your understanding 1.)

the problem is scale. suppose layer 1's weights have norm 50 and layer 2's have norm 0.5. a Gaussian direction perturbs both by similar absolute amounts, which is a 1% change to layer 1 and a 100% change to layer 2: the plot is mostly about layer 2. worse, ReLU networks have a built-in scale symmetry. for $c > 0$, $\text{relu}(cz) = c\,\text{relu}(z)$, so multiplying one layer's weights and biases by $c$ and the next layer's weights by $1/c$ changes nothing the network computes. the "same" minimum can sit at weights of any scale, and a fixed-size perturbation looks huge at small scale and tiny at large scale. a plot that changes when the function doesn't is a plot of the parameterization.

**filter normalization** (Li et al., 2018) fixes the scale by measuring every perturbation relative to the weights it perturbs. draw a Gaussian direction $d$ of the same shape as the weights, then rescale it **filter by filter**: for each filter $i$ (a row of a linear layer's weight, an output channel of a convolution: the slice along dimension 0),

$$d_i \leftarrow \frac{d_i}{\lVert d_i \rVert}\,\lVert \theta_i \rVert.$$

now $\alpha = 1$ means "perturb every filter by a vector as long as the filter itself", whatever the layer and whatever its scale. biases and normalization parameters (1-D tensors) get a zero direction: Li et al. leave them at their trained values, because their scale is tied to the filters they belong to. the layer-wise rescaling above multiplies every filter of one layer by $c$ and of the next by $1/c$, and the normalized directions scale the same way, so the filter-normalized picture doesn't change (on paper 2). the invariance is only layer-wise: rescaling a single unit (its incoming weights by $c$, its outgoing weights by $1/c$) also leaves the function alone, but it changes one *column* of the next layer, whose filters are rows, so their norms change unevenly and the picture changes.

> **spec.** `filter_normalized_direction(model, *, generator=None)` returns a dict `{name: tensor}` with one entry per `model.named_parameters()`, each the parameter's shape, dtype and device. for a parameter with `dim() >= 2`, draw `torch.randn` with `generator`, then scale each slice `d[i]` (along dim 0) to have the norm of `p[i]`. parameters with `dim() <= 1` get zeros. the model is only read.

> **🔮 predict:** the lab's network has two weight matrices of very different norms. if you skip filter normalization and use a plain Gaussian direction scaled to the *whole model's* norm, which layer's perturbation will dominate the plot?

> **▶ build: stage 1.** implement `filter_normalized_direction`.

### idea 2: drawing the surface

given two directions, the surface is a grid of losses:

$$Z_{ij} = L(\theta^* + \alpha_i d_1 + \beta_j d_2), \qquad \alpha_i, \beta_j \in \text{linspace}(-s, s, r).$$

each grid point is one evaluation of the loss on the data: a 21 × 21 grid costs 441 forward passes over the evaluation set, so keep the evaluation set small (the lab uses 2,000 images). three details decide whether the picture is right:

- **every point starts from $\theta^*$.** setting the weights to $\theta^* + \alpha_i d_1 + \beta_j d_2$ means copying from a saved $\theta^*$, not adding $\alpha_i d_1$ to whatever the weights currently are. adding offsets cumulatively produces a plausible-looking surface around a point that drifts across the grid.
- **the weights come back.** when you're done, or when the evaluation raises halfway through, the model must hold $\theta^*$ again; otherwise the next experiment silently runs on a perturbed network. that's `try`/`finally`.
- **the indexing is fixed.** $Z_{ij}$ is the loss at $\alpha_i$ along $d_1$ and $\beta_j$ along $d_2$. matplotlib's `contour(x, y, Z)` expects `Z[row = y, column = x]`, so you plot `contour(alphas, betas, Z.T)`. a transposed surface still looks like a valley, which is exactly why you should get this right in the code and not by eye.

a direction may leave out some parameters (a missing key means "don't move it"), which lets you slice through one layer only.

> **spec.** `loss_surface_2d(model, eval_loss, d1, d2, *, span=1.0, resolution=21)`: `eval_loss(model) -> float`. `alphas = betas = np.linspace(-span, span, resolution)`; `Z[i, j] = eval_loss` with every named parameter set to $\theta + \alpha_i d_1 + \beta_j d_2$ ($\theta$ = the weights when the function is called; a name missing from `d1` or `d2` contributes nothing). `eval_loss` is called exactly `resolution**2` times. the weights are restored afterwards, even if `eval_loss` raises. returns `(alphas, betas, Z)`, `Z` an ndarray `(resolution, resolution)`.

> **▶ build: stage 2.** implement `loss_surface_2d`.

### idea 3: the straight path between two solutions

a different slice asks a global question: train two networks $\theta_A$ and $\theta_B$ and evaluate the loss along the segment between them,

$$L\big((1-\alpha)\,\theta_A + \alpha\,\theta_B\big), \qquad \alpha \in [0, 1].$$

if the loss stays low all the way, the two solutions are **linearly mode connected**: they sit in one convex-looking basin. if it rises in the middle, there is a hill. the size of the hill is the **barrier**: how far the path rises above the straight line joining the two end losses,

$$B = \max_\alpha \Big[L\big((1-\alpha)\theta_A + \alpha\theta_B\big) - \big((1-\alpha)L(\theta_A) + \alpha L(\theta_B)\big)\Big].$$

measuring against the line rather than against the average of the ends matters when the two ends differ: a path from loss 1 to loss 3 that passes through 2 has no barrier at all.

what should you expect? a hidden layer has no natural order: permute its units (and the matching rows and columns of the weights around it) and the network computes the same function with different weights. two independently trained networks almost certainly learned their features in different slots, so averaging them averages unrelated features, and the midpoint is a bad network. Frankle et al. (2020) found that two copies of a network trained from the *same* initialization, with different minibatch orders, are linearly connected once they share even a short stretch of training (for small MNIST networks, from the very start), and they used that stability to explain when lottery-ticket pruning works. Garipov et al. (2018) and Draxler et al. (2018) showed that even independent solutions are connected by simple *curved* paths of low loss, and Ainsworth et al. (2023) showed that permuting one network's units to match the other's removes most of the straight-line barrier. the lab measures the first two of these.

a trap: `model.state_dict()` returns the model's *own* tensors, not copies. if you pass `model.state_dict()` as `state_a` and then load mixtures into `model`, `state_a` changes under your feet, and the "straight line" bends towards wherever the model was last. copy both end states before the loop. buffers matter too: a BatchNorm layer's running mean and variance are floating-point and get interpolated like weights, but its `num_batches_tracked` counter is an integer and can't be averaged; take it from `state_a`.

> **spec.** `linear_interpolation(model, state_a, state_b, eval_loss, *, n_points=21)`: for each $a$ in `np.linspace(0, 1, n_points)`, load the state whose floating-point entries are $(1-a)\,$`state_a`$ + a\,$`state_b` and whose other entries come from `state_a`, and record `eval_loss(model)`. `state_a` and `state_b` may be live `state_dict()`s (copy them first) and are never modified; the model's own state is restored afterwards. returns `(alphas, losses)`, two ndarrays of length `n_points`. `loss_barrier(losses)` (equally spaced points from $\alpha = 0$ to 1) returns $B$ above as a float, 0 if the path never rises above the line.

> **▶ build: stage 3.** implement `linear_interpolation` and `loss_barrier`.

### idea 4: sharpness, exactly, without the Hessian

a slice shows a typical direction. sharpness asks about the worst one: the largest eigenvalue $\lambda_{\max}$ of the Hessian $H = \nabla^2 L(\theta)$. the Hessian of the lab's small network has $26{,}506^2 \approx 700$ million entries; a billion-parameter model's would have $10^{18}$, about as many as there are grains of sand on Earth. you never need it, because two facts combine.

**Hessian-vector products are cheap.** for any fixed vector $v$, $\nabla_\theta\big(\nabla_\theta L(\theta) \cdot v\big) = H v$: the gradient of the scalar "gradient dotted with $v$" is the Hessian times $v$. autograd computes it with one extra backward pass through the graph of the first backward pass (`torch.autograd.grad(loss, params, create_graph=True)`, dot with $v$, then `torch.autograd.grad` again). this is Pearlmutter's (1994) trick, and it costs about as much as two gradient evaluations.

**power iteration only needs products.** start from a random unit vector $v_0$ and repeat $v_{k+1} = Hv_k / \lVert Hv_k\rVert$. writing $v_0 = \sum_i c_i u_i$ in the eigenvectors of $H$, after $k$ steps the component along $u_i$ is proportional to $\lambda_i^k c_i$, so the eigenvalue of largest *magnitude* takes over at the rate $|\lambda_2/\lambda_1|^k$. read the eigenvalue off with the Rayleigh quotient $v_k^\top H v_k$, which keeps its sign. (the norm $\lVert H v_k\rVert$ converges to $|\lambda_1|$: if the dominant eigenvalue is negative, at a saddle, the norm reports a positive number and hides it.) at a minimum all eigenvalues are $\ge 0$ and the two agree.

why should training care about $\lambda_{\max}$? near a minimum the loss is roughly the quadratic $\tfrac12 (\theta - \theta^*)^\top H (\theta - \theta^*)$, and a gradient-descent step with learning rate $\eta$ multiplies the component along eigenvector $u_i$ by $1 - \eta\lambda_i$. if any $\lambda_i > 2/\eta$, that factor is below $-1$ and the component grows, oscillating: gradient descent can't stay in a region sharper than $2/\eta$. Cohen et al. (2021) found that full-batch gradient descent on neural networks does something stranger than staying away: the sharpness *rises* during training ("progressive sharpening") until it reaches $2/\eta$, and then hovers there while the loss keeps going down, non-monotonically. they called it the **edge of stability**. a network that starts *above* $2/\eta$ (as the lab's does) shows the other side of the same instability: within a few steps it is thrown out to a region flatter than $2/\eta$ (the "catapult", Lewkowycz et al., 2020), then sharpens progressively back up to the edge. minibatch noise changes the picture again, and the lab measures how.

> **spec.** `hessian_vector_product(loss_fn, params, vector)`: `loss_fn()` returns a scalar loss computed from `params` (a list of tensors with `requires_grad`); `vector` is a list shaped like `params`; returns the list $H v$ (detached), by double backpropagation, with zeros for parameters the loss doesn't use. `top_hessian_eigenvalue(model, loss_fn, *, n_iters=20, generator=None)`: power iteration over the parameters with `requires_grad=True`, from a random start drawn with `generator`, using `n_iters` Hessian-vector products; returns the Rayleigh quotient $v^\top H v$ (a float, with its sign) of the last normalized iterate it multiplied. each product evaluates `loss_fn()` once (so `loss_fn` is called exactly `n_iters` times); a trainable parameter the loss doesn't use contributes zero curvature (`allow_unused=True`, `None` treated as zeros). it changes neither the parameters nor their `.grad`.

> **🔮 predict:** you train the same network with batch size 16 and with full-batch gradient descent at $\eta = 0.2$. which one ends sharper, and is there a number you'd expect the sharper one to stop at?

> **▶ build: stage 4.** implement `hessian_vector_product` and `top_hessian_eigenvalue`.

## worked example

power iteration on the quadratic $L(w) = \tfrac12 w^\top A w$ with $A = \operatorname{diag}(1, 2, -5)$, whose Hessian is $A$ everywhere. start from $v_0 = (1, 1, 1)/\sqrt3$.

| step | $v$ (unnormalized direction) | Rayleigh quotient $v^\top A v / v^\top v$ |
|---|---|---|
| 0 | $(1, 1, 1)$ | $(1 + 2 - 5)/3 = -0.667$ |
| 1 | $A(1,1,1) = (1, 2, -5)$ | $(1 + 8 - 125)/30 = -3.867$ |
| 2 | $A(1,2,-5) = (1, 4, 25)$ | $(1 + 32 - 3125)/642 = -4.816$ |
| 3 | $(1, 8, -125)$ | $(1 + 128 - 78125)/15690 = -4.971$ |

the iterate flips sign every step (a negative eigenvalue), the Rayleigh quotient converges to $-5$ at the rate $(2/5)^{2k}$, and the norm $\lVert A v\rVert/\lVert v\rVert$ would converge to $+5$. your `top_hessian_eigenvalue` must return $-5$ on this function (the stage 4 check), and your `hessian_vector_product` must give $Av$ for any $v$.

for the barrier: a path with losses $(1, 4, 3, 3, 3)$ at $\alpha = 0, \tfrac14, \tfrac12, \tfrac34, 1$ has the line $1 + 2\alpha$ under it, which is $1.5$ at $\alpha = \tfrac14$, so $B = 4 - 1.5 = 2.5$ (not $4 - 2 = 2$ against the average of the ends, and not $4 - 3 = 1$ against the larger end). `loss_barrier` must return 2.5.

## on paper

1. **gradient descent on a quadratic.** for $L(\theta) = \tfrac12(\theta - \theta^*)^\top H(\theta - \theta^*)$ with $H$ symmetric positive definite, show that gradient descent with step $\eta$ converges from every start iff $\eta\lambda_{\max} < 2$, and that the slowest-converging direction is the one with the smallest eigenvalue when $\eta = 1/\lambda_{\max}$.

<details><summary>solution</summary>

$\theta_{t+1} - \theta^* = (I - \eta H)(\theta_t - \theta^*)$. in the eigenbasis of $H$ the $i$-th coordinate is multiplied by $1 - \eta\lambda_i$ every step, so it converges iff $|1 - \eta\lambda_i| < 1$, i.e. $0 < \eta\lambda_i < 2$. that holds for every $i$ iff $\eta\lambda_{\max} < 2$. with $\eta = 1/\lambda_{\max}$ the factors are $1 - \lambda_i/\lambda_{\max} \in [0, 1)$, largest for the smallest $\lambda_i$: the flat directions are the slow ones, which is why the condition number $\lambda_{\max}/\lambda_{\min}$ sets the number of steps.

</details>

2. **sharpness is not a property of the function.** take $f(x) = w_2\,\text{relu}(w_1 x)$ with scalars $w_1, w_2 > 0$ and the loss $L = \tfrac12 (f(x) - y)^2$ at a point where $f(x) = y$ (a global minimum, $x > 0$). compute the Hessian with respect to $(w_1, w_2)$. now rescale to $(cw_1, w_2/c)$: same function. what happens to the Hessian's largest eigenvalue as $c \to 0$ or $c \to \infty$? what happens to a filter-normalized direction?

<details><summary>solution</summary>

at a zero-residual point only the Gauss–Newton term survives: $H = g g^\top$ with $g = \nabla f = (w_2 x,\ w_1 x)$ (for $w_1 x > 0$). its only nonzero eigenvalue is $\lVert g\rVert^2 = x^2(w_1^2 + w_2^2)$. after rescaling it is $x^2(c^2 w_1^2 + w_2^2/c^2)$, which goes to infinity in both limits: you can make this minimum as sharp as you like without changing a single prediction (Dinh et al., 2017). a filter-normalized direction scales each filter's perturbation by that filter's norm: $\delta w_1 \propto c\,|w_1|$, $\delta w_2 \propto |w_2|/c$, so $\theta + \alpha d$ is exactly the rescaled version of the original $\theta + \alpha d$, and $L$ along it is unchanged. the lab does this to a real network.

</details>

3. **the Hessian-vector product.** show that $\nabla_\theta\big(\nabla_\theta L \cdot v\big) = Hv$ for a constant $v$, and count the cost relative to one gradient. why does the first `autograd.grad` call need `create_graph=True`?

<details><summary>solution</summary>

$\nabla L \cdot v = \sum_j v_j\,\partial_j L$, so $\partial_i(\nabla L \cdot v) = \sum_j v_j\,\partial_i\partial_j L = (Hv)_i$. computing $\nabla L$ is one forward and one backward pass; the dot product is cheap; differentiating it is a backward pass through the graph that computed $\nabla L$, about the cost of a forward plus a backward pass again. in total, two to four times one gradient, independent of the number of parameters. `create_graph=True` makes the first backward pass record its own operations, so there is a graph to differentiate the second time; without it the gradient comes back as a constant and the second call fails.

</details>

## experiment

the lab measures the sharpness of SGD solutions at four batch sizes (and during training) against $2/\eta$, draws filter-normalized surfaces before and after a function-preserving rescaling, and walks the straight line between pairs of networks trained from the same and from different initializations.

## check your understanding

1. why do random directions in a 26,000-dimensional space tell you about typical curvature but almost nothing about the largest eigenvalue?

<details><summary>answer</summary>

a random unit vector has squared component about $1/n$ along any fixed direction, including the top eigenvector, so the curvature along it is $\sum_i \lambda_i c_i^2 \approx \frac1n \operatorname{tr} H$: the average eigenvalue. if a few eigenvalues are large and most are near zero (as they are for networks), a random slice looks much flatter than the sharpest direction.

</details>

2. why does filter normalization zero out biases instead of normalizing them?

<details><summary>answer</summary>

a bias is a single number per unit whose scale is tied to its filter (rescaling a filter by $c$ rescales its bias by $c$ too), and normalizing a 1-element "filter" to its own norm would perturb each bias by $\pm 100\%$. Li et al. chose to keep them fixed; the picture then shows the weights only. it's a convention, and worth stating whenever you show such a plot.

</details>

3. two networks are linearly connected with barrier 0. does that mean they compute the same function?

<details><summary>answer</summary>

no. it means every network on the segment has (about) the same loss. the end points can still disagree on many individual examples, and in Frankle et al.'s experiments they do; they are different points in one wide basin. connectivity is about the loss along the path, not the function.

</details>

4. why might two copies trained from the same initialization be connected even if they never shared a training step?

<details><summary>answer</summary>

the initialization fixes which feature each hidden unit will develop: early gradients are dominated by the data, not the minibatch order, so both copies grow the same features in the same slots, and different minibatch orders only add small, nearly independent noise. their difference is then small and mostly in flat directions, so the segment stays in the basin. for larger networks and harder data (Frankle et al.'s ResNets on ImageNet) this only holds after some shared training.

</details>

5. if sharpness can be multiplied fivefold (or by any factor) without changing the function, is it useless as a generalization measure?

<details><summary>answer</summary>

not useless, but it needs a scale-invariant definition to be meaningful: sharpness measured relative to the weights (filter- or norm-normalized), or in function space, or as an average over a neighborhood (the PAC-Bayes view of ch 13 measures the loss under noise scaled to the weights). Jiang et al. (2020) compared dozens of measures on thousands of trained networks and found some sharpness-based ones among the most predictive, but only in such normalized forms. Foret et al.'s (2021) SAM optimizer, which explicitly seeks flat regions, improves generalization in practice.

</details>

## common bugs

- **`state_dict()` aliasing.** `model.state_dict()` returns the model's own tensors. interpolating towards it while loading into the same model bends the path; the symptom is a barrier that peaks off-center and endpoints that don't reproduce the trained losses. clone first.
- **cumulative perturbations.** `p.add_(alpha * d)` inside the grid loop moves the center with every point. the surface still looks smooth. copy from a saved $\theta$ each time.
- **weights not restored.** the next cell evaluates a perturbed model and everything downstream is wrong by a little. use `try`/`finally`.
- **the norm instead of the Rayleigh quotient** in power iteration: the sign of a negative curvature direction disappears. and calling `loss.backward()` to get gradients pollutes `.grad`; use `torch.autograd.grad`.

## going deeper

- Li, Xu, Taylor, Studer & Goldstein (2018), "Visualizing the loss landscape of neural nets", *NeurIPS*. filter normalization. [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- Hochreiter & Schmidhuber (1997), "Flat minima", *Neural Computation* 9(1): 1–42. [doi:10.1162/neco.1997.9.1.1](https://doi.org/10.1162/neco.1997.9.1.1)
- Keskar, Mudigere, Nocedal, Smelyanskiy & Tang (2017), "On large-batch training for deep learning: generalization gap and sharp minima", *ICLR*. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
- Dinh, Pascanu, Bengio & Bengio (2017), "Sharp minima can generalize for deep nets", *ICML*. [arXiv:1703.04933](https://arxiv.org/abs/1703.04933)
- Cohen, Kaur, Li, Kolter & Talwalkar (2021), "Gradient descent on neural networks typically occurs at the edge of stability", *ICLR*. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
- Lewkowycz, Bahri, Dyer, Sohl-Dickstein & Gur-Ari (2020), "The large learning rate phase of deep learning: the catapult mechanism". [arXiv:2003.02218](https://arxiv.org/abs/2003.02218)
- Pearlmutter (1994), "Fast exact multiplication by the Hessian", *Neural Computation* 6(1): 147–160. [doi:10.1162/neco.1994.6.1.147](https://doi.org/10.1162/neco.1994.6.1.147)
- Garipov, Izmailov, Podoprikhin, Vetrov & Wilson (2018), "Loss surfaces, mode connectivity, and fast ensembling of DNNs", *NeurIPS*. [arXiv:1802.10026](https://arxiv.org/abs/1802.10026)
- Draxler, Veschgini, Salmhofer & Hamprecht (2018), "Essentially no barriers in neural network energy landscape", *ICML*. [arXiv:1803.00885](https://arxiv.org/abs/1803.00885)
- Frankle, Dziugaite, Roy & Carbin (2020), "Linear mode connectivity and the lottery ticket hypothesis", *ICML*. [arXiv:1912.05671](https://arxiv.org/abs/1912.05671)
- Ainsworth, Hayase & Srinivasa (2023), "Git Re-Basin: merging models modulo permutation symmetries", *ICLR*. [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)
- Jiang, Neyshabur, Mobahi, Krishnan & Bengio (2020), "Fantastic generalization measures and where to find them", *ICLR*. [arXiv:1912.02178](https://arxiv.org/abs/1912.02178)
- Foret, Kleiner, Mobahi & Neyshabur (2021), "Sharpness-aware minimization for efficiently improving generalization", *ICLR*. [arXiv:2010.01412](https://arxiv.org/abs/2010.01412)

## connections

- ← builds on: power iteration (ch 02 · 05), the Hessian and Taylor expansion (ch 03 · 04), gradient descent's stability on quadratics (ch 06 · 02), reverse-mode autodiff (ch 14 · 02), PyTorch training and state dicts (ch 15 · 07), PAC-Bayes and margins (ch 13 · 06).
- → used in: topic 04 (does flatness explain generalization?), topic 06 (learning-rate stability across widths), the capstone (the landscape of the interpolating networks); weight averaging and model merging (ch 25, 27).
