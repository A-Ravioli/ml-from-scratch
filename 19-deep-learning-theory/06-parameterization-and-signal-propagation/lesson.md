# parameterization & signal propagation

## why this matters

two practical questions hide behind every deep network. the first is about depth: why does a 20-layer tanh network initialized with one weight scale train in a few hundred steps, while the same network initialized with a scale 30% smaller or larger sits at chance? the second is about width: you tune the learning rate of a small model carefully, then make it 16 times wider for the real run, and the tuned learning rate is now wrong; the bigger the model, the more expensive the sweep to find the new one. both are questions about **parameterization**: how the scale of each layer's initialization, its output multiplier and its learning rate should depend on the network's shape.

the first has a clean answer from treating a wide random network as a dynamical system on its inputs (Poole et al., 2016; Schoenholz et al., 2017). layer by layer, the size of the signal converges to a fixed point, and two inputs either converge to each other (the *ordered* phase, where the network forgets what it was shown) or fly apart (the *chaotic* phase, where it forgets which inputs were similar). on the boundary between them, the **edge of chaos**, information and gradients travel through many layers, and that's where deep networks train. He initialization for ReLU networks is exactly this boundary. the second question has a newer answer: the **maximal-update parameterization**, μP (Yang & Hu, 2021; Yang et al., 2022), scales each layer so that every layer's features change by the same amount per step at every width, and learning rates tuned on a small model then transfer to a large one ("μTransfer"). this topic builds both theories in code and tests them.

## warm-up: predict first

1. a 50-layer tanh network has weights $\mathcal N(0, \sigma_w^2/\text{fan-in})$. if $\sigma_w$ is small, what happens to the outputs for two different inputs? if it's large?
2. why is He initialization ($\text{Var}(W) = 2/\text{fan-in}$) the right scale for ReLU networks, while Xavier's ($1/\text{fan-in}$) works for tanh?
3. Adam moves every weight by about the learning rate per step. if a hidden layer becomes 16 times wider, how much more does one unit's input change per step?
4. under the standard parameterization, should the best learning rate grow or shrink with width?
5. can the same learning rate be optimal for a 64-wide and a 1,024-wide network?

## prerequisites check

- [ ] you know Xavier and He initialization and the variance argument behind them (→ ch 15 · topic 01)
- [ ] you know the central limit theorem: a sum of many independent terms is approximately Gaussian (→ ch 04 · topic 05)
- [ ] you can compute an expectation by numerical quadrature (→ ch 03 · topic 05)
- [ ] you know Adam's update and per-parameter-group learning rates in PyTorch (→ ch 15 · topic 04)
- [ ] you know the NTK and mean-field parameterizations and the lazy regime (→ topic 03)

## core ideas

### idea 1: the size of a signal, layer by layer

take a deep network $h^{l+1} = W^{l+1}\phi(h^l) + b^{l+1}$ with $W_{ij} \sim \mathcal N(0, \sigma_w^2/N)$ ($N$ the fan-in) and $b_i \sim \mathcal N(0, \sigma_b^2)$, and follow one input through it. each pre-activation $h^{l+1}_i$ is a sum of $N$ independent random terms, so for a wide layer it's approximately Gaussian, with variance

$$q^{l+1} = \sigma_w^2\,\frac1N\sum_j\phi(h^l_j)^2 + \sigma_b^2 \;\approx\; \sigma_w^2\,\mathbb E_{z\sim\mathcal N(0,1)}\big[\phi(\sqrt{q^l}\,z)^2\big] + \sigma_b^2 \;=:\; \mathcal V(q^l).$$

this **variance map** $\mathcal V$ is deterministic: in the infinite-width limit, the size of the signal at layer $l$ is $\mathcal V$ applied $l$ times to the first layer's $q^1 = \sigma_w^2\lVert x\rVert^2/d + \sigma_b^2$, whatever the random draw. for ReLU it's exact and linear: $\mathbb E[\text{relu}(u)^2] = q/2$, so $\mathcal V(q) = \sigma_w^2 q/2 + \sigma_b^2$. for tanh it has no closed form, and you compute it by quadrature.

**Gauss–Hermite quadrature** is the natural tool: `np.polynomial.hermite_e.hermegauss(n)` returns nodes $z_k$ and weights $w_k$ with $\sum_k w_k g(z_k) = \int g(z)e^{-z^2/2}dz$ exactly for polynomials $g$ of degree up to $2n - 1$, so $\mathbb E[g(z)] \approx \sum_k w_k g(z_k)/\sqrt{2\pi}$. use an **even** number of nodes: an odd rule puts a node at $z = 0$, exactly where ReLU's derivative jumps, and a step function evaluated at its own jump gets a large, wrong weight. with an even rule the nodes are symmetric around the kink and half of them see each side, which makes ReLU's expectations exact. the rule is accurate only while $\sqrt q\,z$ stays in the range where the nodes resolve the integrand: for tanh with 100 nodes it's fine up to $q \approx 5$, but $\mathbb E[\tanh'(\sqrt q z)^2]$ (which $\chi_1$ needs, a narrow peak at 0 for large $q$) comes out 2%, 15% and 46% too low at $q = 10$, 20 and 40. cross-check large-$q$ values against a Monte Carlo average over a few million standard normal draws (`np.mean(f(np.sqrt(q) * rng.normal(size=4_000_000)))`); the lab's largest fixed point, $q^* \approx 6.4$ at $\sigma_w = 3$, is within about 1%.

> **spec.** `gaussian_expectation(f, q, *, n_points=100)` returns $\mathbb E[f(\sqrt q\,z)]$, $z \sim \mathcal N(0, 1)$, by Gauss–Hermite quadrature with `hermegauss(n_points)` (divide by $\sqrt{2\pi}$); `f` acts on NumPy arrays. `variance_map(q, sigma_w, sigma_b, activation)` returns $\mathcal V(q)$ for `activation` in `'tanh'`, `'relu'` (the given helper `activation_fn(name)` returns $\phi$ and $\phi'$).

> **▶ build: stage 1.** implement `gaussian_expectation` and `variance_map`.

### idea 2: order, chaos, and the edge between them

the variance map has a fixed point $q^* = \mathcal V(q^*)$, which deep layers approach geometrically. at the fixed point the interesting question is what happens to *two* inputs. their pre-activations at a layer have correlation $c^l$, and the same Gaussian argument gives a **correlation map** $c^{l+1} = \mathcal C(c^l)$, with $c = 1$ (identical inputs) always a fixed point. its slope there,

$$\chi_1 = \frac{\partial\mathcal C}{\partial c}\Big|_{c=1} = \sigma_w^2\,\mathbb E\big[\phi'(\sqrt{q^*}\,z)^2\big],$$

decides everything (on paper 2):

- **$\chi_1 < 1$, ordered.** $c = 1$ is attracting: two different inputs get more and more correlated with depth until the network maps everything to the same output. gradients shrink by a factor of about $\chi_1$ per layer on the way back: they vanish.
- **$\chi_1 > 1$, chaotic.** $c = 1$ is repelling: nearby inputs decorrelate, small input changes produce unrelated outputs, and gradients grow by about $\chi_1$ per layer: they explode.
- **$\chi_1 = 1$, the edge of chaos.** correlations change slowly, over a depth scale that diverges at the boundary ($\xi \approx -1/\ln\chi_1$ when $\chi_1 < 1$), and gradients neither vanish nor explode. Schoenholz et al. (2017) showed that networks are trainable roughly up to depths of a few times $\xi$, so very deep networks train only near the edge.

the same quantity governs the forward correlation and the backward gradient because $\chi_1$ is the mean squared singular value of the layer's Jacobian $\partial h^{l+1}/\partial h^l = W\operatorname{diag}(\phi'(h^l))$: it is how much a small perturbation grows per layer, in either direction.

for ReLU, $\mathbb E[\text{relu}'(u)^2] = \frac12$, so $\chi_1 = \sigma_w^2/2$, and the edge is $\sigma_w^2 = 2$: **He initialization**, derived in chapter 15 from "keep the variance constant", is the edge of chaos. (ReLU's correlation map is degenerate: correlations still converge to 1 slowly even at the edge, which is one reason residual connections and normalization help so much.) for tanh with $\sigma_b = 0$, $q^* = 0$ when $\sigma_w \le 1$ and $\tanh'(0) = 1$, so the edge is $\sigma_w = 1$: Xavier initialization. with a bias, the edge moves to larger $\sigma_w$ (the lab finds $\sigma_w \approx 1.40$ for $\sigma_b = 0.3$).

> **spec.** `q_fixed_point(sigma_w, sigma_b, activation, *, q0=1.0, n_iter=500)` applies `variance_map` `n_iter` times starting from `q0` and returns the result. `chi1(q, sigma_w, activation)` returns $\sigma_w^2\,\mathbb E[\phi'(\sqrt q\,z)^2]$.

> **🔮 predict:** for tanh with $\sigma_b = 0.3$, is the critical $\sigma_w$ above or below 1? what do you expect a 20-layer network to do after 200 SGD steps at $\sigma_w = 0.6$?

> **▶ build: stage 2.** implement `q_fixed_point` and `chi1`.

### idea 3: checking the theory on a real network

the maps are infinite-width statements. to see how well they describe a finite network, simulate one: draw the weights of every layer (a fresh $\mathcal N(0, \sigma_w^2/N)$ matrix and $\mathcal N(0, \sigma_b^2)$ bias each), push a batch of inputs through, and record, at every layer, the mean squared pre-activation of each input and the cosine similarity between the pre-activation vectors of two of them. at width 2,000, the measured $q^l$ follows the variance map within a few percent; at width 50 it fluctuates by tens of percent from layer to layer.

note what's measured: pre-activations $h^l$, before $\phi$, since the maps are about them, and the correlation is the cosine of the vectors $h^l(x_a)$ and $h^l(x_b)$ across the layer's units (the empirical version of $c^l$).

> **spec.** `propagate(x, depth, width, *, sigma_w, sigma_b, activation, generator=None)`: `x` an `(n, d)` array or tensor with $n \ge 2$; `depth` layers of `width` units; layer $l$'s weight `(fan_in, width)` has entries $\mathcal N(0, \sigma_w^2/\text{fan-in})$ and its bias entries $\mathcal N(0, \sigma_b^2)$, all drawn with `torch.randn(..., generator=generator)`; the first layer acts on `x`, later ones on $\phi$ of the previous pre-activations. returns `{'q': (depth, n) array of mean squared pre-activations per input, 'corr': (depth,) array of the cosine similarity of inputs 0 and 1's pre-activation vectors}`. `ValueError` for an unknown activation.

> **▶ build: stage 3.** implement `propagate`.

### idea 4: a parameterization is a choice of three scales per layer

for each layer, three things can depend on the width $n$: the initialization variance, a constant multiplier in the forward pass, and the learning rate. the **standard parameterization** (SP) is what PyTorch does by default: initialization variance $\propto 1/\text{fan-in}$ (PyTorch's default is uniform on $\pm 1/\sqrt{\text{fan-in}}$, variance $1/(3\,\text{fan-in})$; this topic uses $\mathcal N(0, 1/\text{fan-in})$), no multipliers, one learning rate. topic 03 met two others: the NTK parameterization (lazy, features frozen as $n \to \infty$) and mean field (features move). Yang & Hu (2021) asked which choices make *every* layer's features change by $\Theta(1)$ per step as $n \to \infty$, neither vanishing (lazy) nor blowing up, and called the unique answer the maximal-update parameterization.

for Adam, the argument is a counting exercise. Adam normalizes each gradient entry, so a step moves every weight by about the learning rate $\eta$. consider a hidden layer, $n \times n$: the change of one unit's input is $\sum_{j=1}^{n}\Delta W_{ij}\,h_j$, a sum of $n$ terms of size $\eta$. they are not independent: the update is (nearly) low-rank and aligned with the activations it was computed from, so they add coherently, to $\Theta(\eta n)$. at a fixed $\eta$ the change grows linearly with width; to keep it $\Theta(1)$, the hidden layer's learning rate must scale as $1/n$. the input layer's fan-in is the input dimension, fixed, so it keeps $\eta$; so do biases (one number per unit). the readout's input dimension grows, so its contribution must be scaled down too: μP multiplies the readout's input by $1/m$ ($m$ = width / base width), which turns the coherent $\Theta(\eta n)$ change into $\Theta(\eta)$, and initializes the readout at zero, so that the initial output (which would otherwise be $O(1/n)$ of noise: an $O(1)$ sum, times $1/m$) is exactly 0 (on paper 4).

in code, relative to a base width $n_0$ where μP and SP agree up to the readout initialization: with $m = n/n_0$,

| | init variance | forward multiplier | Adam learning rate |
|---|---|---|---|
| input layer | $1/\text{fan-in}$ | 1 | $\eta$ |
| hidden layers ($n \times n$) | $1/\text{fan-in}$ | 1 | $\eta/m$ |
| readout | 0 (zero init) | input scaled by $1/m$ | $\eta$ |
| biases | 0 | 1 | $\eta$ |

the readout multiplier is applied to the readout's *input*, `readout(h / m)`, so it scales the readout's weights' effect and not its bias, as the `mup` package does.

> **spec.** `MuMLP(in_dim, width, out_dim, *, depth=2, parameterization='sp', base_width=64)` has `hidden`, an `nn.ModuleList` of `depth` `nn.Linear` layers (`in_dim` → `width`, then `width` → `width`), and `readout`, an `nn.Linear(width, out_dim)`; every weight is $\mathcal N(0, 1/\text{fan-in})$ and every bias 0, except that `'mup'` zero-initializes `readout.weight`; attributes `width_mult` $= $ `width / base_width` and `parameterization`; `ValueError` for another parameterization. `activations(x)` returns the list of the `depth` hidden activations (after ReLU) followed by the output, which is `readout(h)` in SP and `readout(h / width_mult)` in μP; `forward(x)` returns the output. `param_groups(model, lr)` returns optimizer parameter groups: SP, one group with every parameter at `lr`; μP, the weights of `hidden[1:]` at `lr / width_mult` and every other parameter at `lr`, each parameter in exactly one group.

> **▶ build: stage 4.** implement `MuMLP` and `param_groups`.

### idea 5: the coordinate check, and μTransfer

you can test a parameterization without any theory: train networks of increasing width for a few steps with the same learning rate and measure how much each layer's activations change (the RMS change per entry). Yang et al. (2022) call this the **coordinate check**. under SP with Adam, the hidden activations' and especially the outputs' changes grow with width: wide SP networks take ever larger steps in function space at the same learning rate, which is why their best learning rate shrinks with width. under μP, the changes are flat in width. flat curves are the definition of a correct μP implementation, and a fast bug-finder (a missing $1/m$ shows up as a slope).

the payoff is **μTransfer**: if every layer's dynamics are width-independent, then so is the best learning rate (and other hyperparameters like the initialization scale and the schedule), so you can tune them on a small model and copy them to a large one. Yang et al. tuned a 40M-parameter proxy, transferred its hyperparameters to a 6.7B-parameter GPT-3 and outperformed the published model, with a tuning cost of about 7% of one pretraining run. transfer is approximate: it holds across width (and, with care, depth), not across changes of data, batch size or training length, and finite-width effects shift the optimum slightly. the lab measures how much.

> **spec.** `activation_deltas(model, x, y, *, lr, steps=1)` records `model.activations(x)` (no gradients), trains `model` in place for `steps` full-batch steps of `torch.optim.Adam(param_groups(model, lr))` on `F.cross_entropy(model(x), y)` in train mode, records the activations again, and returns the list of RMS changes $\sqrt{\text{mean}((a_{\text{after}} - a_{\text{before}})^2)}$, one per entry of `activations`.

> **🔮 predict:** at the same Adam learning rate, how many times more does a 2,048-wide SP network's output change in three steps than a 64-wide one's? and a μP network's?

> **▶ build: stage 5.** implement `activation_deltas`.

## worked example

ReLU with $\sigma_w = 1$, $\sigma_b = 0.5$: the variance map is $\mathcal V(q) = q/2 + 0.25$.

- one step from $q = 7$: $3.5 + 0.25 = 3.75$. the fixed point solves $q = q/2 + 0.25$: $q^* = 0.5$, approached geometrically with ratio $\frac12$.
- $\chi_1 = \sigma_w^2\,\mathbb E[\text{relu}'^2] = 1 \cdot \frac12 = 0.5 < 1$: ordered. with $\sigma_w^2 = 2$ (He), $\chi_1 = 1$ exactly, and with $\sigma_b = 0$ the variance map is the identity, $q^l = q^1$ for every layer.
- quadrature check: with an even number of Gauss–Hermite nodes, $\mathbb E[\text{relu}'(u)^2]$ is computed as exactly $\frac12$ (half of the symmetric nodes are positive); with 101 nodes the middle node sits at 0 and carries 12% of the total weight, which $\text{relu}'(0) = 0$ throws away: the estimate drops to 0.44.

the stage 1–2 checks use these numbers: `q_fixed_point(1.0, 0.5, "relu")` is 0.5, one iteration from 7 gives 3.75, `chi1(q, sqrt(2), "relu")` is 1 for any $q$.

## on paper

1. **the variance map.** for $h_i = \sum_{j=1}^N W_{ij}a_j + b_i$ with independent $W_{ij} \sim \mathcal N(0, \sigma_w^2/N)$, $b_i \sim \mathcal N(0, \sigma_b^2)$ and fixed $a$, compute $\text{Var}(h_i)$. then explain why, for $a_j = \phi(h^{prev}_j)$ with $h^{prev}_j$ approximately $\mathcal N(0, q)$ and $N$ large, $\frac1N\sum_j a_j^2 \approx \mathbb E[\phi(\sqrt q\,z)^2]$.

<details><summary>solution</summary>

$\text{Var}(h_i) = \sum_j a_j^2\,\sigma_w^2/N + \sigma_b^2 = \sigma_w^2\cdot\frac1N\sum_j a_j^2 + \sigma_b^2$. the previous layer's pre-activations are (in the wide limit) independent across units given the layer before, each approximately $\mathcal N(0, q)$, so $\frac1N\sum_j\phi(h_j)^2$ is an average of $N$ i.i.d. terms and converges to $\mathbb E[\phi(\sqrt q z)^2]$ by the law of large numbers. each $h_i$ is a sum of many independent terms, hence Gaussian by the CLT, which makes the recursion close.

</details>

2. **$\chi_1$ is the slope of the correlation map.** the correlation map is $\mathcal C(c) = \frac{1}{q^*}\big(\sigma_w^2\,\mathbb E[\phi(u_1)\phi(u_2)] + \sigma_b^2\big)$ with $u_1 = \sqrt{q^*}z_1$, $u_2 = \sqrt{q^*}(cz_1 + \sqrt{1 - c^2}z_2)$. show that $\mathcal C'(1) = \sigma_w^2\,\mathbb E[\phi'(u)^2]$. (hint: for jointly Gaussian $u_1, u_2$ with covariance $\Sigma_{12}$, $\partial\,\mathbb E[f(u_1)g(u_2)]/\partial\Sigma_{12} = \mathbb E[f'(u_1)g'(u_2)]$.)

<details><summary>solution</summary>

the covariance of $u_1, u_2$ is $q^*c$, so $\partial/\partial c = q^*\,\partial/\partial\Sigma_{12}$. by the hint (Price's theorem), $\mathcal C'(c) = \frac{1}{q^*}\sigma_w^2\,q^*\,\mathbb E[\phi'(u_1)\phi'(u_2)] = \sigma_w^2\,\mathbb E[\phi'(u_1)\phi'(u_2)]$. at $c = 1$, $u_1 = u_2 = u$, giving $\sigma_w^2\,\mathbb E[\phi'(u)^2] = \chi_1$. if $\chi_1 < 1$ the fixed point $c = 1$ is stable (inputs converge); if $\chi_1 > 1$ it's unstable (they separate).

</details>

3. **ReLU's phases.** for ReLU, find $q^*$ as a function of $\sigma_w, \sigma_b$, and show that a finite positive fixed point exists only when $\sigma_w^2 < 2$. what happens to $q^l$ when $\sigma_w^2 > 2$, and when $\sigma_w^2 = 2$, $\sigma_b = 0$?

<details><summary>solution</summary>

$\mathcal V(q) = \sigma_w^2 q/2 + \sigma_b^2$ is linear with slope $\sigma_w^2/2$. for slope below 1, $q^* = \sigma_b^2/(1 - \sigma_w^2/2)$ attracts every start. for slope above 1, $q^l$ grows geometrically (by $\sigma_w^2/2$ per layer): the signal explodes. at slope 1 with $\sigma_b = 0$, $q^l = q^1$ forever: the variance is preserved exactly, which is He initialization's defining property, and $\chi_1 = \sigma_w^2/2 = 1$: the variance condition and the edge-of-chaos condition coincide for ReLU.

</details>

4. **why μP divides the hidden learning rate by the width.** a hidden layer $h' = Wh$ with $h \in \mathbb R^n$, entries of order 1. one Adam step changes $W$ by $\Delta W \approx -\eta\,\text{sign}(g\,h^\top)$ for a backpropagated vector $g$ (Adam's normalization, roughly). show that $(\Delta W h)_i$ is of order $\eta n$, and conclude the learning-rate scaling that keeps it of order 1. what changes for the input layer, whose fan-in is a fixed $d$?

<details><summary>solution</summary>

$(\Delta W h)_i = -\eta\sum_j \text{sign}(g_i h_j)\,h_j = -\eta\,\text{sign}(g_i)\sum_j |h_j|$, a sum of $n$ positive terms of order 1: $\Theta(\eta n)$. the update is aligned with the very vector it multiplies, so nothing cancels (an independent random update would give $\Theta(\eta\sqrt n)$). keeping it $\Theta(1)$ needs $\eta \propto 1/n$, i.e. $\eta/m$ relative to a base width. for the input layer the sum has $d$ terms, independent of the width, so $\eta$ stays constant. the same counting applied to the readout (input dimension $n$) gives a change of $\Theta(\eta n)$ in each output, which the multiplier $1/m$ brings back to $\Theta(\eta)$.

</details>

## experiment

the lab maps the edge of chaos for a tanh network with bias 0.3: $\chi_1$ from your maps, the correlation of two inputs after 30 layers of a real network, and the training loss of 20-layer networks initialized on both sides of the edge. it runs the coordinate check for SP and μP from width 64 to 2,048 with your `activation_deltas`. finally it sweeps the Adam learning rate for three widths in both parameterizations to see whether the optimum transfers.

## check your understanding

1. a network at the edge of chaos still has $c = 1$ as a stable fixed point for ReLU. why does that matter less with residual connections?

<details><summary>answer</summary>

a residual block computes $x + f(x)$, so each layer only adds a (small) perturbation to the identity, whose Jacobian has all singular values near 1; correlations and gradients then change slowly with depth for any reasonable initialization of $f$. that's why ResNets and pre-LN transformers train at depths where plain networks need exact criticality.

</details>

2. under SP the best Adam learning rate shrinks with width. explain with the coordinate check.

<details><summary>answer</summary>

at a fixed learning rate a wider SP network's hidden activations and outputs move more per step (in the lab, the output change grows 27-fold from width 64 to 2,048). the largest stable step in function space is roughly width-independent, so the learning rate has to shrink to keep the actual change at that size: the lab's SP optimum moves from $2^{-6}$ to $2^{-10}$ over a 16-fold width increase.

</details>

3. why does μP zero-initialize the readout?

<details><summary>answer</summary>

with the $1/m$ multiplier, a random readout's contribution at initialization shrinks with width, so its exact value doesn't matter in the limit, and zero removes a source of width-dependent noise in the initial function (and the initial gradient of the hidden layers). Yang et al. recommend it; with it, the initial output is exactly the readout bias, 0.

</details>

4. in the lab, the best SP network at width 1,024 (with its own tuned learning rate) reaches a lower training loss than the best μP network of the same width. does that contradict μP?

<details><summary>answer</summary>

no: μP's claim is that the optimum *transfers*, not that μP's optimum is better at every width and budget. at a short budget, SP's larger effective steps in the readout can make faster progress, and μP's zero readout starts slower. μP's advantage is that you didn't have to sweep at width 1,024 to find a good learning rate; and Yang et al. report that at large scale μTransferred models match or beat SP models tuned directly, which a 200-step MNIST experiment can't test.

</details>

## common bugs

- **an odd number of quadrature nodes** with ReLU: the node at 0 lands on the derivative's jump and biases $\chi_1$.
- **measuring after the activation** in `propagate`: the maps describe pre-activations.
- **scaling the readout's output instead of its input**: the bias then gets the multiplier too, and its learning dynamics change with width.
- **dividing the input layer's learning rate by the width multiplier**: its fan-in doesn't grow, so it would learn slower and slower in wider models.
- **a parameter in two optimizer groups**, or one forgotten: PyTorch raises for the first; the second silently freezes a layer.
- **comparing parameterizations at different base widths**: μP and SP agree only at `width == base_width`; the base width is part of the parameterization.

## going deeper

- Poole, Lahiri, Raghu, Sohl-Dickstein & Ganguli (2016), "Exponential expressivity in deep neural networks through transient chaos", *NeurIPS*. [arXiv:1606.05340](https://arxiv.org/abs/1606.05340)
- Schoenholz, Gilmer, Ganguli & Sohl-Dickstein (2017), "Deep information propagation", *ICLR*. [arXiv:1611.01232](https://arxiv.org/abs/1611.01232)
- He, Zhang, Ren & Sun (2015), "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification", *ICCV*. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Pennington, Schoenholz & Ganguli (2017), "Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice", *NeurIPS*. [arXiv:1711.04735](https://arxiv.org/abs/1711.04735)
- Hanin (2018), "Which neural net architectures give rise to exploding and vanishing gradients?", *NeurIPS*. [arXiv:1801.03744](https://arxiv.org/abs/1801.03744)
- Yang & Hu (2021), "Tensor programs IV: feature learning in infinite-width neural networks", *ICML*. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)
- Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen & Gao (2022), "Tensor programs V: tuning large neural networks via zero-shot hyperparameter transfer". [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Everett et al. (2024), "Scaling exponents across parameterizations and optimizers", *ICML*. [arXiv:2407.05872](https://arxiv.org/abs/2407.05872)
- Roberts, Yaida & Hanin (2022), *The Principles of Deep Learning Theory*, chapters 3–5 (criticality). [arXiv:2106.10165](https://arxiv.org/abs/2106.10165)

## connections

- ← builds on: initialization and its variance argument (ch 15 · 01), Adam and parameter groups (ch 15 · 04), the CLT (ch 04), quadrature (ch 03), the NTK and mean-field limits (topic 03).
- → used in: the capstone's μTransfer option; residual networks and normalization as ways to stay near criticality (ch 16, 18); hyperparameter transfer for the scaling studies of topic 05, ch 26 and ch 27.
