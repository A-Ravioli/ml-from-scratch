# infinite width: NTK & mean field

## why this matters

a neural network is a nonlinear function of its weights, its loss is non-convex, and gradient descent on it comes with almost no guarantees. yet very wide networks train reliably to zero training loss, from any random start. in 2018 Jacot, Gabriel and Hongler found a clean reason: as the width goes to infinity (with the right scaling), training a network by gradient descent becomes *exactly* kernel regression with a fixed kernel, the **neural tangent kernel** (NTK), which depends on the architecture and not on the data. a nonlinear, non-convex problem turns into the linear, convex problem you solved in chapter 09. that result gave the first convergence proofs for deep networks and a precise sense in which "a wide network is a kernel machine".

it also exposed a gap. if a wide network is a kernel method, it never learns features: its "features" are frozen at their random initial values, and everything chapter 16 said about convolutional filters learning edges would be false in this regime. the way out is that the NTK limit depends on a choice of scaling. scale the output layer by $1/m$ instead of $1/\sqrt m$ and the infinite-width limit is a different object, the **mean-field** limit, in which the features move as much as they like. this topic builds both: the empirical NTK of any PyTorch model, the infinite-width kernel of a two-layer ReLU network in closed form, the dynamics of the linearized network, and the measurements (how far the weights move, how much the kernel changes) that tell you which regime a network is in.

## warm-up: predict first

1. a two-layer network with $m$ hidden units is trained to fit 16 points. as $m$ grows from 64 to 16,384, does the *relative* change of its weights during training grow, shrink, or stay the same?
2. you linearize a network around its initial weights, $f(x;\theta) \approx f(x;\theta_0) + \nabla_\theta f(x;\theta_0)\cdot(\theta - \theta_0)$, and train the linear model instead. is that a convex problem?
3. the output of a randomly initialized network $f(x) = \frac{1}{\sqrt m}\sum_j a_j\,\text{relu}(w_j\cdot x)$, with standard-normal $a_j, w_j$: as $m \to \infty$, does $f(x)$ go to 0, blow up, or converge to something random?
4. kernel regression with ridge 0 interpolates its training data. does gradient descent on a wide network, run to convergence, interpolate too? where would the two disagree?
5. which do you expect to be learned first by gradient descent on a wide network: the smooth part of a target or its fine, high-frequency details?

## prerequisites check

- [ ] you know kernel ridge regression, Gram matrices and why $K$ must be positive semi-definite (→ ch 09 · topics 03, 05)
- [ ] you can compute per-example gradients with `torch.autograd.grad` (→ ch 14 · topic 02, ch 15 · topic 07)
- [ ] you know that a sum of many independent terms concentrates around its mean at rate $1/\sqrt n$ (→ ch 04 · topic 05)
- [ ] you can diagonalize a symmetric matrix and use $f(K) = V f(\Lambda) V^\top$ (→ ch 02 · topic 05)
- [ ] you know Xavier/He initialization and why weights are scaled by $1/\sqrt{\text{fan-in}}$ (→ ch 15 · topic 01)

## core ideas

### idea 1: the tangent kernel

take any model $f(x;\theta)$ with a scalar output and train it on $n$ points with the squared loss $L = \frac12\sum_i (f(x_i;\theta) - y_i)^2$. one step of gradient descent changes the weights by $\Delta\theta = -\eta\sum_i (f(x_i) - y_i)\,\nabla_\theta f(x_i)$. how does that change the *prediction* at some input $x$? to first order,

$$\Delta f(x) = \nabla_\theta f(x)\cdot\Delta\theta = -\eta\sum_i \underbrace{\nabla_\theta f(x)\cdot\nabla_\theta f(x_i)}_{\Theta(x,\,x_i)}\,\big(f(x_i) - y_i\big).$$

the quantity in the brace is the **neural tangent kernel**,

$$\Theta(x, x') = \big\langle \nabla_\theta f(x;\theta),\ \nabla_\theta f(x';\theta) \big\rangle,$$

the inner product of the two inputs' gradients. it says how much a step that improves the prediction at $x'$ drags the prediction at $x$ along. on the training set, stacking the gradients as rows of the Jacobian $J \in \mathbb{R}^{n \times P}$, the kernel matrix is $K = JJ^\top$, and the predictions evolve as

$$f_{t+1} = f_t - \eta K (f_t - y).$$

this is exact for a model that is linear in its weights and a first-order approximation otherwise. two things make it a kernel: it's an inner product of feature vectors (the gradients), so $K$ is positive semi-definite, and the features are $P$-dimensional, often millions. for a linear model $f(x) = w\cdot x + b$ the gradient is $(x, 1)$ and $\Theta(x, x') = x\cdot x' + 1$: the linear kernel plus a constant.

computing it is mechanical: one gradient per example (one backward pass for each output), flattened and concatenated over the parameters you train, then a matrix product. parameters with `requires_grad=False` are not trained, so they are not part of the kernel.

> **spec.** `empirical_ntk(model, x1, x2=None)`: `model(x)` returns `(n,)` or `(n, 1)` outputs. returns the `(n1, n2)` tensor $K_{ij} = \langle \partial f(x^{(1)}_i)/\partial\theta,\ \partial f(x^{(2)}_j)/\partial\theta\rangle$, over the parameters with `requires_grad=True`; `x2=None` means $x_2 = x_1$. it doesn't touch any `.grad` (use `torch.autograd.grad`).

> **▶ build: stage 1.** implement `empirical_ntk`.

### idea 2: if the kernel stays fixed, training is kernel regression

suppose $K$ doesn't change during training. then $f_{t+1} - y = (I - \eta K)(f_t - y)$, a linear recursion, and with $\eta < 2/\lambda_{\max}(K)$ the training residual goes to zero along every eigendirection with $\lambda > 0$. the predictions elsewhere follow the same kernel: summing the updates,

$$f_\infty(x) = f_0(x) + \Theta(x, X)\,K^{-1}\big(y - f_0(X)\big).$$

that is **kernel regression** with kernel $\Theta$, ridge 0, applied to the residual $y - f_0$ of the initial function: exactly the estimator of chapter 09's `KernelRidge`, with a kernel you didn't choose. two consequences already. gradient descent run to convergence interpolates the training data (if $K$ is invertible), like kernel regression. and between the training points, the answer is decided by $\Theta$: which functions the kernel finds "simple" is the network's inductive bias in this regime.

you'll use kernel regression on many kernels in this chapter, so build it once on Gram matrices: $\hat y = K_{*X}(K_{XX} + \lambda I)^{-1}y$, by a linear solve rather than an inverse.

> **spec.** `kernel_regression_predict(K_train, y_train, K_test_train, *, ridge=0.0)` returns `K_test_train @ solve(K_train + ridge * I, y_train)`. `y_train` is `(n,)` or `(n, k)`. inputs may be tensors or ndarrays; the result is a tensor when `K_test_train` is a tensor and an ndarray otherwise.

> **▶ build: stage 2.** implement `kernel_regression_predict`.

### idea 3: infinite width, and a kernel you can write down

the empirical NTK depends on the random initialization. as the network gets wider, it stops depending on it. take the simplest case, a two-layer ReLU network without biases in the **NTK parameterization**:

$$f(x) = \frac{1}{\sqrt m}\sum_{j=1}^m a_j\,\text{relu}(w_j\cdot x), \qquad a_j \sim \mathcal N(0, 1),\ w_j \sim \mathcal N(0, I).$$

the weights are standard normal and the scale $1/\sqrt m$ is in the forward pass instead of the initialization. (at initialization this is the same network as a $1/\sqrt{m}$-scaled initialization; the difference shows up in the gradients, idea 5.) the gradients are $\partial f/\partial a_j = \text{relu}(w_j\cdot x)/\sqrt m$ and $\partial f/\partial w_j = a_j\,\mathbb 1[w_j\cdot x > 0]\,x/\sqrt m$, so

$$\Theta(x, x') = \frac1m\sum_j \text{relu}(w_j\cdot x)\,\text{relu}(w_j\cdot x') + x\cdot x'\,\frac1m\sum_j a_j^2\,\mathbb 1[w_j\cdot x > 0]\,\mathbb 1[w_j\cdot x' > 0].$$

both terms are averages of $m$ independent random variables. by the law of large numbers they converge to their expectations, with fluctuations of order $1/\sqrt m$. the expectations are over a pair of jointly Gaussian variables $(u, v) = (w\cdot x, w\cdot x')$ with variances $\lVert x\rVert^2, \lVert x'\rVert^2$ and correlation $\cos\vartheta$, $\vartheta$ the angle between $x$ and $x'$. both have closed forms (Cho & Saul's arc-cosine kernels, 2009, and on paper 2):

$$\Theta_\infty(x, x') = \underbrace{\frac{\lVert x\rVert\lVert x'\rVert}{2\pi}\big(\sin\vartheta + (\pi - \vartheta)\cos\vartheta\big)}_{\text{from the output layer}} + \underbrace{x\cdot x'\ \frac{\pi - \vartheta}{2\pi}}_{\text{from the hidden layer}}.$$

check it: for $x' = x$, $\vartheta = 0$ and $\Theta_\infty = \lVert x\rVert^2/2 + \lVert x\rVert^2/2 = \lVert x\rVert^2$; for opposite inputs, $\vartheta = \pi$ and both terms vanish. the formula is deterministic. a randomly initialized network of infinite width has a fixed tangent kernel, the same for every seed.

numerically, compute $\cos\vartheta$ as $x\cdot x'/(\lVert x\rVert\lVert x'\rVert)$ and clamp it to $[-1, 1]$ before the arccosine: for identical inputs rounding can give $1.0000000002$, and `arccos` returns NaN.

the **mean-field parameterization** replaces $1/\sqrt m$ by $1/m$. at initialization its output goes to 0 as $m$ grows, and its tangent kernel is exactly $1/m$ times the NTK-parameterized one. idea 5 shows why that small change is the difference between a kernel machine and a feature learner.

> **spec.** `NTKNet(in_dim, width, *, parameterization='ntk')` is an `nn.Module` with exactly two parameters, `W` `(width, in_dim)` and `a` `(width,)`, both drawn from `torch.randn` (the global generator), attributes `width` and `parameterization`, and `forward(x)` mapping `(n, in_dim)` to `(n,)`: $a\cdot\text{relu}(Wx)$ times $1/\sqrt{\text{width}}$ (`'ntk'`) or $1/\text{width}$ (`'mf'`); any other parameterization raises `ValueError`. `relu_ntk(X1, X2=None)` returns the float64 tensor $\Theta_\infty$ above, `(n1, n2)`, with no NaN for identical or zero-angle inputs.

> **🔮 predict:** the relative error of the empirical NTK against $\Theta_\infty$ at width 1,024 is about 5%. what do you expect at width 16,384?

> **▶ build: stage 3.** implement `NTKNet` and `relu_ntk`.

### idea 4: the whole training run, in closed form

with a fixed kernel the recursion $f_{t+1} - y = (I - \eta K)(f_t - y)$ solves itself. diagonalize $K = V\Lambda V^\top$. in the eigenbasis each component of the residual shrinks independently, by a factor $1 - \eta\lambda_k$ per step:

$$f_t(X) = y - V(I - \eta\Lambda)^t V^\top\big(y - f_0(X)\big).$$

at a test input the prediction moves by $-\eta\,\Theta(x, X)(f_s - y)$ at every step $s$; summing the geometric series,

$$f_t(x) = f_0(x) + \Theta(x, X)\, V\,\operatorname{diag}\!\Big(\frac{1 - (1 - \eta\lambda_k)^t}{\lambda_k}\Big)V^\top\big(y - f_0(X)\big).$$

as $t \to \infty$ the diagonal becomes $1/\lambda_k$ and you get idea 2's kernel regression. at finite $t$ it is kernel regression with every eigendirection only partly fitted. the eigenbasis also handles a singular $K$ (more points than features, or duplicated inputs): the factor $(1 - (1 - \eta\lambda)^t)/\lambda$ has the finite limit $\eta t$ as $\lambda \to 0$, so compute it with that limit for (numerically) zero eigenvalues instead of dividing $0$ by $0$.

the formula explains **spectral bias**: directions with large $\lambda_k$ are fitted in a few steps, directions with small $\lambda_k$ take $\sim 1/(\eta\lambda_k)$ steps. for the NTKs of real networks the large-eigenvalue directions are smooth, low-frequency functions of the input, so wide networks learn the coarse shape of a target first and its fine detail last (Rahaman et al., 2019; Basri et al., 2019). early stopping is then a smoothness prior.

> **spec.** `ntk_gd_predictions(K_train, K_test_train, y_train, f0_train, f0_test, *, lr, steps)` returns `(f_train, f_test)`, the predictions after `steps` gradient-descent steps of size `lr` on $\frac12\sum_i (f(x_i) - y_i)^2$ for the model linearized at initialization (the two formulas above). inputs may be tensors or ndarrays; `K_train` may be singular; `steps=0` returns `(f0_train, f0_test)`.

> **▶ build: stage 4.** implement `ntk_gd_predictions`.

### idea 5: lazy training, and why it can't be the whole story

why does the kernel stay fixed when the network is wide? look at the sizes. in the NTK parameterization, each weight's gradient carries a factor $1/\sqrt m$, so to change the output by $O(1)$ (enough to fit the data), each weight only needs to move by $O(1/\sqrt m)$. the kernel is built from the gradients, which depend on the weights through $\text{relu}'(w_j\cdot x)$ and $a_j$, which barely move: the kernel changes by $O(1/\sqrt m)$ too, and in the limit the linearization is exact. the relative change of the weight vector, $\lVert\theta_T - \theta_0\rVert / \lVert\theta_0\rVert$, is the practical diagnostic: it shrinks like $1/\sqrt m$ in the NTK regime. Chizat, Oyallon & Bach (2019) called this **lazy training** and pointed out that it comes from the scaling, not from width as such: any model whose output is scaled up enough relative to how fast its gradients change will train lazily.

lazy training is a regime in which nothing is learned except a linear readout of fixed random features. the mean-field parameterization escapes it. with $f = \frac1m\sum_j a_j\,\text{relu}(w_j\cdot x)$, the output is an *average* over units, so fitting the data requires each unit to change by $O(1)$, and the features $w_j$ rotate towards useful directions. as $m \to \infty$ the network becomes a distribution over units evolving by a nonlinear transport equation (Mei, Montanari & Nguyen, 2018; Chizat & Bach, 2018; Rotskoff & Vanden-Eijnden, 2018), not a kernel method. the price is that its tangent kernel is $m$ times smaller, so it needs a learning rate about $m$ times larger to move at all; topic 06 shows how μP (Yang & Hu, 2021) chooses the scalings of every layer so that features keep learning at every width.

which regime describes the networks that work? not the NTK, mostly. kernel regression with the NTK of a convolutional network gets about 77% on CIFAR-10 (Arora et al., 2019), well below a trained ResNet; finite networks with the standard learning rates move their kernels a lot, and the kernel after training is a much better kernel than the one before (Fort et al., 2020). the NTK is a precise theory of one limit and a useful baseline, not an explanation of deep learning.

> **spec.** `gd_train(model, x, y, *, lr, steps)` runs `steps` plain full-batch gradient-descent steps on $\frac12\sum_i (f(x_i) - y_i)^2$ (outputs and `y` flattened to `(n,)`), updating the trainable parameters in place, and returns the list of `steps + 1` losses: before each step, then after the last. `relative_change(model, initial_state)` returns $\lVert\theta - \theta_0\rVert_2 / \lVert\theta_0\rVert_2$, one global norm over `model.named_parameters()`, with $\theta_0$ looked up by name in `initial_state` (buffers don't count).

> **🔮 predict:** in the lab, an NTK-parameterized and a mean-field network of the same width both fit the same 16 points. by how much does each one's tangent kernel change during training?

> **▶ build: stage 5.** implement `gd_train` and `relative_change`.

## worked example

the infinite-width kernel between $x = (3, 4)$ and $x' = (0, 2)$: $\lVert x\rVert = 5$, $\lVert x'\rVert = 2$, $x\cdot x' = 8$, so $\cos\vartheta = 0.8$, $\vartheta = \arccos 0.8 = 0.6435$, $\sin\vartheta = 0.6$, $\pi - \vartheta = 2.4981$.

- output-layer term: $\frac{5 \cdot 2}{2\pi}\big(0.6 + 2.4981 \times 0.8\big) = 1.5915 \times 2.5985 = 4.1356$.
- hidden-layer term: $8 \times \frac{2.4981}{2\pi} = 3.1807$.
- $\Theta_\infty(x, x') = 7.3163$.

also $\Theta_\infty(x, x) = 25$, and $\Theta_\infty(x, -x) = 0$. your `relu_ntk` must give these three numbers (the stage 3 check), and `empirical_ntk` of a 20,000-unit `NTKNet` must agree with `relu_ntk` within a few percent.

## on paper

1. **from one step to the closed form.** starting from $f_{t+1}(X) - y = (I - \eta K)(f_t(X) - y)$ and $f_{t+1}(x) = f_t(x) - \eta\,\Theta(x, X)(f_t(X) - y)$, derive idea 4's two formulas.

<details><summary>solution</summary>

the first recursion gives $f_t(X) - y = (I - \eta K)^t (f_0(X) - y)$, and $(I - \eta K)^t = V(I - \eta\Lambda)^tV^\top$. for the test input, summing the updates, $f_t(x) = f_0(x) - \eta\,\Theta(x, X)\sum_{s=0}^{t-1}(I - \eta K)^s(f_0(X) - y)$. in the eigenbasis the geometric sum is $\sum_s (1 - \eta\lambda_k)^s = \frac{1 - (1 - \eta\lambda_k)^t}{\eta\lambda_k}$ (or $t$ if $\lambda_k = 0$). multiplying by $\eta$ gives the diagonal $\frac{1 - (1 - \eta\lambda_k)^t}{\lambda_k}$, with limit $\eta t$ at $\lambda_k = 0$, and the sign flips because of $f_0 - y = -(y - f_0)$.

</details>

2. **the hidden-layer term.** for $w \sim \mathcal N(0, I)$, show that $P(w\cdot x > 0,\ w\cdot x' > 0) = \frac{\pi - \vartheta}{2\pi}$, $\vartheta$ the angle between $x$ and $x'$. (hint: only the projection of $w$ onto the plane of $x$ and $x'$ matters, and its direction is uniform on the circle.)

<details><summary>solution</summary>

project $w$ onto the plane spanned by $x$ and $x'$; the projection is a 2-D standard Gaussian, whose direction is uniform on the circle and independent of its length. $w\cdot x > 0$ is a half-plane (a half-circle of directions); so is $w\cdot x' > 0$. the two half-circles are rotated against each other by $\vartheta$, so they overlap on an arc of length $\pi - \vartheta$, a fraction $\frac{\pi - \vartheta}{2\pi}$ of the circle. with $E[a^2] = 1$ independent of $w$, the hidden-layer term converges to $x\cdot x'\,\frac{\pi - \vartheta}{2\pi}$. (the output-layer term needs one more integral: Cho & Saul, 2009, compute $E[\text{relu}(u)\,\text{relu}(v)] = \frac{\lVert x\rVert\lVert x'\rVert}{2\pi}(\sin\vartheta + (\pi - \vartheta)\cos\vartheta)$; check it at $\vartheta = 0$, where it must equal $E[\text{relu}(u)^2] = \lVert x\rVert^2/2$.)

</details>

3. **why the weights barely move.** in the NTK parameterization, gradient flow fits the data with a total change $\Delta f = O(1)$ at each training point. show that the weight change of a single unit is $O(1/\sqrt m)$, that $\lVert\Delta\theta\rVert$ stays $O(1)$ while $\lVert\theta_0\rVert = \Theta(\sqrt m)$, and conclude how the relative change scales with $m$.

<details><summary>solution</summary>

every component of $\nabla_\theta f$ carries a factor $1/\sqrt m$, and the residuals are $O(1)$, so $d\theta_j/dt = O(1/\sqrt m)$ for each of the $O(m)$ weights, over an $O(1)$ amount of training time (the NTK, and so the speed of convergence in function space, is $O(1)$). each weight moves $O(1/\sqrt m)$; there are $O(m)$ of them, so $\lVert\Delta\theta\rVert = O(\sqrt{m}\cdot 1/\sqrt m) = O(1)$. the initial weights are standard normal, $\lVert\theta_0\rVert \approx \sqrt{(d + 1)m}$. the ratio is $O(1/\sqrt m)$: the lab measures 0.75, 0.18 and 0.04 at widths 64, 1,024 and 16,384, a factor of about 4 per 16-fold width.

</details>

4. **the mean-field kernel.** show that with the same weights, the mean-field network's tangent kernel is the NTK network's divided by $m$. if you train it with learning rate $\eta m$, what is the size of each weight's update compared with its initial value?

<details><summary>solution</summary>

the mean-field output is the NTK output divided by $\sqrt m$, so every gradient is divided by $\sqrt m$ and their inner products by $m$. with learning rate $\eta m$, a weight's update is $\eta m \cdot O(1/m) \cdot O(1) = O(\eta)$ per step: of the same order as the weight itself ($O(1)$). nothing is small, the linearization fails, and the units genuinely move, which is the point of the mean-field regime and why its learning rate needs care (the lab uses $\eta m/3$; $\eta m$ is unstable).

</details>

## experiment

the lab works on the circle, where the NTK is a function of one angle you can plot. it measures how fast the empirical NTK of a random network approaches `relu_ntk` as the width grows from 16 to 16,384, with three networks per width. it trains networks of three widths to fit 16 points and compares each with the prediction its own linearization made before training started, and measures how far the weights moved. finally, at one width, it trains an NTK-parameterized and a mean-field network on the same task and measures how much their tangent kernels and their hidden units change.

## check your understanding

1. the empirical NTK of a finite network changes during training. does the first-order formula $\Delta f(x) = -\eta\,\Theta(x, X)(f - y)$ still hold at every step?

<details><summary>answer</summary>

yes, to first order in $\eta$, with the kernel evaluated at the current weights. what fails at finite width is the assumption that this kernel is the same at every step; the closed form of idea 4 uses the kernel at initialization for the whole run. how well that works is what the lab's "trained vs linearized" column measures.

</details>

2. why can't a network in the NTK regime learn a better kernel for the task, even if trained forever?

<details><summary>answer</summary>

its kernel is (in the limit) fixed at initialization and independent of the data: training only fits the coefficients of a kernel expansion. the "features" are the gradient vectors at initialization, which are random. learning features means changing $\Theta$, which requires leaving the lazy regime.

</details>

3. a trained infinite-width NTK network and `KernelRidge` with $\Theta_\infty$ and ridge 0: do they make the same predictions?

<details><summary>answer</summary>

not quite: the network predicts $f_0(x) + \Theta(x, X)K^{-1}(y - f_0(X))$, and $f_0$ is a random function (a Gaussian process sample) that doesn't vanish as $m \to \infty$. averaged over initializations, or with an initialization trick that makes $f_0 = 0$ (two copies with opposite output weights), the predictions coincide with kernel regression.

</details>

4. Arora et al. computed the NTK of an infinitely wide CNN and found it worse on CIFAR-10 than a finite trained CNN. which of this topic's measurements would you use on the trained CNN to see why?

<details><summary>answer</summary>

the relative change of its weights and, more directly, the change of its empirical NTK over training (the lab's "kernel change"). a finite CNN trained with standard learning rates moves its kernel substantially and ends with a kernel that is much better aligned with the labels; that improvement is exactly what the infinite-width limit throws away.

</details>

## common bugs

- **differentiating the sum of the outputs.** `autograd.grad(out.sum(), params)` gives the sum of all examples' gradients, and every row of $K$ comes out the same. take one gradient per output.
- **including frozen parameters**, or leaving `.grad` filled by calling `backward()` inside the kernel computation; the next optimizer step then uses stale gradients.
- **NaN on the diagonal** of `relu_ntk`: the cosine of a vector with itself rounds to slightly more than 1. clamp.
- **the mean instead of the sum** in the loss. the NTK dynamics $f_{t+1} = f_t - \eta K(f_t - y)$ assume $\frac12\sum$; with a mean the effective learning rate is $\eta/n$ and nothing matches the closed form.
- **comparing a trained network with kernel regression from zero.** the linearized network starts at $f_0$, not at 0; compare with $f_0 + \Theta K^{-1}(y - f_0)$ or remove $f_0$ by construction.
- **a learning rate above $2/\lambda_{\max}(K)$**: the linearized dynamics diverge (topic 02's edge of stability, in function space).

## going deeper

- Jacot, Gabriel & Hongler (2018), "Neural tangent kernel: convergence and generalization in neural networks", *NeurIPS*. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein & Pennington (2019), "Wide neural networks of any depth evolve as linear models under gradient descent", *NeurIPS*. [arXiv:1902.06720](https://arxiv.org/abs/1902.06720)
- Cho & Saul (2009), "Kernel methods for deep learning", *NeurIPS*; extended as "Analysis and extension of arc-cosine kernels for large margin classification" (2011). the arc-cosine kernels. [arXiv:1112.3712](https://arxiv.org/abs/1112.3712)
- Chizat, Oyallon & Bach (2019), "On lazy training in differentiable programming", *NeurIPS*. [arXiv:1812.07956](https://arxiv.org/abs/1812.07956)
- Mei, Montanari & Nguyen (2018), "A mean field view of the landscape of two-layer neural networks", *PNAS* 115(33). [arXiv:1804.06561](https://arxiv.org/abs/1804.06561)
- Chizat & Bach (2018), "On the global convergence of gradient descent for over-parameterized models using optimal transport", *NeurIPS*. [arXiv:1805.09545](https://arxiv.org/abs/1805.09545)
- Arora, Du, Hu, Li, Salakhutdinov & Wang (2019), "On exact computation with an infinitely wide neural net", *NeurIPS*. [arXiv:1904.11955](https://arxiv.org/abs/1904.11955)
- Rahaman, Baratin, Arpit, Draxler, Lin, Hamprecht, Bengio & Courville (2019), "On the spectral bias of neural networks", *ICML*. [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)
- Basri, Jacobs, Kasten & Kritchman (2019), "The convergence rate of neural networks for learned functions of different frequencies", *NeurIPS*. [arXiv:1906.00425](https://arxiv.org/abs/1906.00425)
- Fort, Dziugaite, Paul, Kharaghani, Roy & Ganguli (2020), "Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel", *NeurIPS*. [arXiv:2010.15110](https://arxiv.org/abs/2010.15110)
- Yang & Hu (2021), "Tensor programs IV: feature learning in infinite-width neural networks", *ICML*. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)

## connections

- ← builds on: kernels, kernel ridge regression and Gram matrices (ch 09 · 03, 05), per-example gradients (ch 14, 15), initialization scales (ch 15 · 01), concentration of averages (ch 04), eigendecomposition (ch 02).
- → used in: topic 04 (double descent in kernel and random-feature models), topic 06 (μP: the parameterization between the two limits), the capstone's NTK-convergence option; Gaussian processes as infinite-width networks (ch 12 · GP connection), and the "lazy vs rich" question in every later chapter that trains wide models.
