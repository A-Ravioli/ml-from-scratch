# MAML

## why this matters

give a person ten points from a sine wave and they'll sketch the rest of the curve. give a freshly initialized network the same ten points and it will fit them, and nothing else: between and beyond the points it draws whatever its random weights suggest. the person isn't smarter about these ten points. they've seen many waves before and know what kind of function they're looking at, so ten points are enough to pin down the two numbers that matter, amplitude and phase.

the usual way to give a network that experience is **pretraining**: fit one network to data from many related tasks, then fine-tune it on the new one. for sine waves this fails in an instructive way. the tasks contradict each other (at $x = 1$, one wave is at $+3$ and another at $-2$), so the best single function for all of them is a flat compromise, and fine-tuning from that compromise with ten points and a few gradient steps barely moves it. the pretrained weights are good at *predicting*, not at *being adapted*.

**model-agnostic meta-learning** (MAML; Finn, Abbeel & Levine, 2017) asks for the second thing directly. it searches for an initialization $\theta$ such that, for a task drawn from the family, a few gradient steps from $\theta$ on that task's few examples give a good model for the task. it treats the fine-tuning procedure as part of the model and differentiates through it. in this topic you build MAML from its parts: a functional network, the inner loop that adapts it, the meta-gradient through that loop (checked against a closed form you derive), and the outer loop that trains the initialization. the lab then asks the question in the curriculum: how many inner steps does adaptation actually need?

## warm-up: predict first

1. a network is pretrained on thousands of sine tasks, then given 10 points of a new wave and one gradient step. how much better than its pretrained prediction will it be: 5%, 50%, 90%?
2. the meta-gradient of MAML involves second derivatives of the loss. for a network with 1,700 weights, will one second-order meta-step cost about 2× a first-order one, 100×, or 1,700×?
3. MAML is trained with one inner step. at test time you take ten. better than one step, the same, or worse?
4. the second derivative of ReLU is zero almost everywhere. is MAML's second-order term therefore zero for a ReLU network?
5. what does MAML do if you set the number of inner steps to zero?

## prerequisites check

- [ ] you can compute gradients with `torch.autograd.grad`, and know what `create_graph=True` does (→ ch 15 · topic 07, ch 14 · topic 03 for what autograd stores)
- [ ] you know gradient descent, its step-size limit, and the Hessian as curvature (→ ch 06 · topics 02 and 05)
- [ ] you can apply the chain rule to vector functions with Jacobians (→ ch 03 · topic 02, ch 02 · topic 07)
- [ ] you know why a train/test split must be honest, and how to report a mean with its spread over seeds (→ ch 07 · topics 02 and 03)
- [ ] you've trained an MLP with Adam in PyTorch (→ ch 15)

## core ideas

### idea 1: tasks, and models as functions of their weights

the data of meta-learning is a **distribution over tasks** $p(\mathcal T)$. each task comes with a small **support set** $\mathcal D^s_\mathcal T$ (the examples you adapt on) and a **query set** $\mathcal D^q_\mathcal T$ (the examples you're judged on). in Finn et al.'s regression benchmark, a task is a sine wave $y = A \sin(x - \varphi)$ with amplitude $A \sim U[0.1, 5]$ and phase $\varphi \sim U[0, \pi]$, inputs $x \sim U[-5, 5]$, and $K = 10$ support points: **10-shot regression**. every quantity is drawn independently, and support and query points of a task are separate draws from the same wave. new tasks at test time are new waves, never seen in training.

meta-learning has two levels of training data, and keeping them apart is the main source of honesty here: the meta-training tasks are the "training set" of the outer problem, and each task's support set is the "training set" of the inner one. a model may look at a test task's support points, never at its query points.

MAML needs one unusual thing from the model: its weights must be *inputs*, not internal state. adaptation produces new weights $\theta'$ that are a differentiable function of the old ones, and the loss of $\theta'$ must be differentiable back to $\theta$. an `nn.Module` updated by an optimizer breaks that chain (`optimizer.step()` writes new values in place, outside autograd). so in this topic a model is a list of tensors plus a function `forward(params, x)`. adapted weights are ordinary tensors computed from the originals, and autograd records how.

one more convenience makes everything fast. a meta-batch has $T$ tasks, and after adaptation each task has its own weights. if the forward function accepts weights with a leading task dimension, a $W$ of shape $(T, d_\text{in}, d_\text{out})$ and inputs $(T, n, d_\text{in})$, then `x @ W` is a batched matrix product and all $T$ adapted networks run in one call. biases of shape $(T, d_\text{out})$ need one extra axis to broadcast over the $n$ points.

> **spec · stage 1.** `init_mlp(sizes, *, generator=None)`: a list `[W1, b1, ..., WL, bL]` of float32 leaf tensors with `requires_grad=True`, `W_l` of shape `(sizes[l], sizes[l+1])`, `b_l` of shape `(sizes[l+1],)`, every entry $U(-1/\sqrt{\text{fan\_in}}, 1/\sqrt{\text{fan\_in}})$ with fan_in `= sizes[l]` (`nn.Linear`'s default range), drawn from `generator`. `mlp_forward(params, x)`: `h = h @ W + b` per layer with ReLU between layers and none after the last; `x` is `(..., n, d_in)`; weights may be shared (`W (d_in, d_out)`, `b (d_out,)`) or per-task (`W (T, d_in, d_out)`, `b (T, d_out)` with `x (T, n, d_in)`); returns `(..., n, d_out)`. `sample_sine_tasks(n_tasks, n_support, n_query, *, amp_range=(0.1, 5.0), phase_range=(0.0, math.pi), x_range=(-5.0, 5.0), generator=None)`: a dict of float32 tensors `xs (T, n_support, 1)`, `ys`, `xq (T, n_query, 1)`, `yq`, `amp (T,)`, `phase (T,)`, with $y = A \sin(x - \varphi)$ and every amplitude, phase and input drawn independently and uniformly from its range using `generator`.

> **▶ build: stage 1.** implement `init_mlp`, `mlp_forward` and `sample_sine_tasks` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the inner loop

adaptation is plain gradient descent on the support loss, starting from the shared initialization:

$$\theta'_\mathcal T = \theta - \alpha \nabla_\theta \mathcal L(\theta; \mathcal D^s_\mathcal T), \qquad \text{repeated } k \text{ times},$$

with the **inner learning rate** $\alpha$ and the loss $\mathcal L$ a mean over the support points (here, mean squared error). every task gets its own copy of the weights and takes its own steps.

two implementation points decide whether this is MAML or something else.

**per-task gradients from one backward pass.** expand the shared weights to $T$ copies, `p.expand(T, *p.shape)`, run the batched forward, and sum the $T$ task losses. task $t$'s loss depends only on copy $t$, so the gradient of the sum with respect to the expanded tensor is, slice by slice, each task's own gradient. averaging instead of summing would scale every task's step by $1/T$; taking the gradient with respect to the *unexpanded* weights would give every task the same, batch-averaged step, which is pretraining in disguise.

**keeping the graph.** MAML's outer loss is a function of $\theta'$, and $\theta'$ is a function of $\theta$ through the gradient $\nabla \mathcal L(\theta)$. to differentiate through that gradient, autograd must record how the gradient itself was computed: `torch.autograd.grad(loss, params, create_graph=True)`. the returned gradients are then tensors with their own history, and $\theta - \alpha g$ remembers both paths back to $\theta$, the direct one and the one through $g$. without `create_graph` the gradient is a constant, and only the direct path survives. that's exactly **first-order MAML**, and it's a useful approximation, not a bug, as long as it's chosen deliberately.

> **spec · stage 2.** `task_mse(pred, y)`: per-task mean squared error over points and outputs, `(T, n, k)` → `(T,)`. `adapt(params, xs, ys, *, forward=None, loss_fn=None, inner_lr=0.01, n_steps=1, first_order=False)`: `forward` defaults to `mlp_forward`, `loss_fn` (per-task losses `(T,)`) to `task_mse`. starting from the shared `params`, every task $t$ of `xs (T, n, d)`, `ys` takes `n_steps` steps $\theta_t \leftarrow \theta_t - \alpha \nabla_{\theta_t} \texttt{loss\_fn}(\ldots)_t$. returns a list of per-task tensors of shape `(T, *p.shape)` (for `n_steps=0`, the shared values repeated). with `first_order=False` the result is differentiable with respect to `params` through every inner step (second order); with `first_order=True` the inner gradients are constants. the input params are not modified, and params with `requires_grad=False` work too (evaluation).

> **▶ build: stage 2.** implement `task_mse` and `adapt`. the worked example below is one of stage 2's checks.

### idea 3: the meta-objective and its gradient

MAML's objective is the query loss *after* adaptation, averaged over tasks:

$$\min_\theta\ \mathbb E_{\mathcal T \sim p(\mathcal T)} \Big[ \mathcal L\big(\theta'_\mathcal T(\theta);\ \mathcal D^q_\mathcal T\big) \Big].$$

this is a **bilevel** problem: the inner problem (adapt to the task) sits inside the outer one (choose the starting point). MAML doesn't solve the inner problem to optimality; it fixes it to $k$ gradient steps, which makes $\theta'$ an explicit, differentiable function of $\theta$.

the chain rule gives the meta-gradient. for one inner step, $\theta' = \theta - \alpha \nabla \mathcal L_s(\theta)$, so the Jacobian is $\partial\theta'/\partial\theta = I - \alpha \nabla^2 \mathcal L_s(\theta)$, symmetric, and

$$\nabla_\theta\, \mathcal L_q(\theta') = \big(I - \alpha H_s(\theta)\big)\, \nabla \mathcal L_q(\theta'), \qquad H_s = \nabla^2 \mathcal L_s.$$

for $k$ steps the Jacobians multiply: $\prod_{j=k-1}^{0} (I - \alpha H_s(\theta_j))$ applied to $\nabla\mathcal L_q(\theta_k)$ (on paper 1). nobody forms these matrices. autograd computes the product as a chain of **Hessian-vector products**, each about the cost of one extra backward pass, so a second-order meta-gradient costs a small constant factor more than a first-order one, plus the memory to keep every inner step's graph alive. the lab measures the factor.

**first-order MAML** (FOMAML) drops the Hessian terms: the meta-gradient is just $\nabla \mathcal L_q(\theta'_\mathcal T)$, the query gradient at the adapted weights, applied to $\theta$. why should that work? the dropped term $-\alpha H_s \nabla\mathcal L_q$ is second order in a small step size, and for ReLU networks part of the curvature is zero (warm-up 4: not all of it, because the output is a product of several layers' weights). Finn et al. found FOMAML nearly as good on their benchmarks; Nichol, Achiam & Schulman (2018) analysed it and proposed **Reptile**, an even simpler first-order method that just moves $\theta$ towards the adapted weights, $\theta \leftarrow \theta + \epsilon(\theta'_\mathcal T - \theta)$, and showed that its expected update contains the same "make the support and query gradients agree" term that MAML's Hessian term produces.

for a linear model with squared loss everything is explicit, which gives you an oracle. with $\hat y = X\theta$ and $\mathcal L_s(\theta) = \frac1n \lVert X_s\theta - y_s \rVert^2$, the Hessian is constant, $H_s = \frac2n X_s^\top X_s$, every inner step is the affine map $\theta \mapsto (I - \alpha H_s)\theta + \frac{2\alpha}{n} X_s^\top y_s$, and

$$\nabla_\theta\, \mathcal L_q(\theta_k) = (I - \alpha H_s)^k\ \nabla \mathcal L_q(\theta_k), \qquad \nabla\mathcal L_q(\theta) = \tfrac{2}{m} X_q^\top (X_q \theta - y_q).$$

> **spec · stage 3.** `maml_loss(params, batch, *, forward=None, loss_fn=None, inner_lr=0.01, n_steps=1, first_order=False)`: `batch` is a dict with `xs, ys, xq, yq` (leading task dimension); adapt on the support set with `adapt` (same options), then return the **mean over tasks** of `loss_fn` on the query set at the adapted weights, a scalar tensor differentiable with respect to `params`. `loss_fn` is used for both inner and outer losses. `linear_meta_gradient(theta, Xs, ys, Xq, yq, *, inner_lr, n_steps=1, first_order=False)`: NumPy, float64, one task, the model $X\theta$ (no bias), mean-squared losses; returns the meta-gradient $(I - \alpha H_s)^k \nabla\mathcal L_q(\theta_k)$, shape `(d,)`, or $\nabla\mathcal L_q(\theta_k)$ when `first_order`.

> **▶ build: stage 3.** implement `maml_loss` and `linear_meta_gradient`. the checks compare your meta-gradient with autograd through an unrolled loop, with finite differences, and with your closed form.

> **🔮 predict:** in the worked example below, the first-order meta-gradient for the bias is $-2.2$: increase the bias. what sign will the full second-order gradient have?

### idea 4: the outer loop, and what it's compared with

the outer loop is ordinary stochastic optimization over tasks: sample a meta-batch of tasks, compute `maml_loss`, take an optimizer step on $\theta$, repeat. Finn et al. use Adam, meta-batches of 25 tasks, one inner step with $\alpha = 0.01$, and 70,000 meta-iterations; the lab uses 3,000.

a meta-learning result is only as good as its baseline, and the natural one comes for free: set `n_steps=0`. then `maml_loss` is the plain query loss at $\theta$ averaged over tasks, and training it is **multi-task pretraining**: one network fitted to all tasks at once. at test time both inits get the same treatment: $k$ gradient steps with the same $\alpha$ on the task's $K$ support points, then scored on its query points. on paper 4 works out what pretraining converges to for sine waves; it explains why the baseline barely improves with fine-tuning.

**evaluating honestly** means new tasks (new amplitudes and phases), many of them (the lab uses 1,000), more query points than support points (100 vs 10), and several meta-training seeds, reporting the spread. an init that looks good on one seed's tasks can be lucky.

> **spec · stage 4.** `train_maml(params, sample_batch, *, n_iters, forward=None, loss_fn=None, meta_lr=1e-3, inner_lr=0.01, n_steps=1, first_order=False, optimizer="adam", seed=0)`: trains a **copy** of `params` (fresh leaf tensors with `requires_grad=True`; the inputs are untouched). creates one `torch.Generator` seeded with `seed` and, for each of `n_iters` iterations, draws `batch = sample_batch(generator)`, computes `maml_loss` with the given options, and takes one step of `torch.optim.Adam(lr=meta_lr)` (default betas) or `torch.optim.SGD(lr=meta_lr)` for `optimizer="sgd"`, created once before the loop; other optimizers: `ValueError`. returns `(params, history)` where `history[i]` is iteration $i$'s meta-loss as a float, computed before that iteration's update.

> **▶ build: stage 4.** implement `train_maml`. its last check meta-trains a small sine network for 400 iterations on three seeds and compares it with pretraining.

> **🔮 predict:** both inits start from the same random weights and see the same tasks. after training, which one has the lower query loss *before* any adaptation ($k = 0$)?

## worked example

one task, the model $\hat y = w x + b$ from $(w, b) = (1, 0)$, support points $(1, 3)$ and $(2, 5)$, query point $(3, 7)$, $\alpha = 0.1$, one inner step, mean squared error.

- **inner step.** predictions $(1, 2)$, errors $\hat y - y = (-2, -3)$, support loss $\frac{4 + 9}{2} = 6.5$. gradients $\partial \mathcal L_s/\partial w = \frac{2}{2}((-2)(1) + (-3)(2)) = -8$ and $\partial \mathcal L_s/\partial b = \frac22(-2 - 3) = -5$. so $w' = 1 + 0.8 = 1.8$ and $b' = 0 + 0.5 = 0.5$.
- **query loss.** $\hat y = 1.8 \cdot 3 + 0.5 = 5.9$, error $-1.1$, $\mathcal L_q = 1.21$; at the adapted weights its gradient is $2(-1.1)(3, 1) = (-6.6, -2.2)$. that's the first-order meta-gradient.
- **Hessian.** with features $(x, 1)$, $H_s = \frac{2}{2}\big[(1,1)^\top(1,1) + (2,1)^\top(2,1)\big] = \begin{pmatrix} 5 & 3 \\ 3 & 2\end{pmatrix}$, so $I - \alpha H_s = \begin{pmatrix} 0.5 & -0.3 \\ -0.3 & 0.8 \end{pmatrix}$.
- **MAML meta-gradient.** $(I - \alpha H_s)(-6.6, -2.2) = (-3.3 + 0.66,\ 1.98 - 1.76) = (-2.64,\ 0.22)$.

the bias gradient flips sign. first-order MAML says "raise $b$": the adapted model undershoots the query point. full MAML knows that the inner step will raise $b$ by itself, and that a larger initial $b$ shrinks the inner step's correction of $w$, which is what the query point actually needs; it asks for a slightly *lower* starting $b$ and a higher $w$. your `adapt` must give $(1.8, 0.5)$, and your `linear_meta_gradient` must give $(-2.64, 0.22)$ and, first-order, $(-6.6, -2.2)$.

## on paper

1. derive the $k$-step meta-gradient for a general model: with $\theta_{j+1} = \theta_j - \alpha \nabla \mathcal L_s(\theta_j)$, show $\nabla_{\theta_0} \mathcal L_q(\theta_k) = \big(\prod_{j=k-1}^{0} (I - \alpha H_s(\theta_j))\big)^{\!\top} \nabla \mathcal L_q(\theta_k)$, and explain why the transpose can be dropped.

<details><summary>solution</summary>

each step has Jacobian $J_j = \partial\theta_{j+1}/\partial\theta_j = I - \alpha \nabla^2\mathcal L_s(\theta_j)$. by the chain rule $\partial\theta_k/\partial\theta_0 = J_{k-1} J_{k-2} \cdots J_0$, and the gradient of the scalar $\mathcal L_q(\theta_k)$ with respect to $\theta_0$ is $(\partial\theta_k/\partial\theta_0)^\top \nabla\mathcal L_q(\theta_k) = J_0^\top \cdots J_{k-1}^\top \nabla \mathcal L_q(\theta_k)$. each $J_j$ is symmetric (a Hessian is), so this is $J_0 J_1 \cdots J_{k-1} \nabla\mathcal L_q(\theta_k)$: reverse-mode autograd applies $J_{k-1}$ first, then $J_{k-2}$, down to $J_0$, one Hessian-vector product each. the transpose matters only in the order of the factors, which are evaluated at different points, so in general they don't commute; for a linear model they're all the same matrix.

</details>

2. for the linear model of idea 3, show that $\theta_k = (I - \alpha H_s)^k \theta_0 + c$ with $c$ independent of $\theta_0$, and conclude the closed form of `linear_meta_gradient`. for which $\alpha$ does the inner loop converge, and to what?

<details><summary>solution</summary>

$\theta_{j+1} = \theta_j - \alpha(H_s \theta_j - \tfrac2n X_s^\top y_s) = (I - \alpha H_s)\theta_j + \tfrac{2\alpha}{n} X_s^\top y_s$. unrolling, $\theta_k = (I-\alpha H_s)^k \theta_0 + \sum_{j<k} (I - \alpha H_s)^j \tfrac{2\alpha}n X_s^\top y_s$, so $\partial\theta_k/\partial\theta_0 = (I - \alpha H_s)^k$ (symmetric) and the meta-gradient is $(I - \alpha H_s)^k \nabla \mathcal L_q(\theta_k)$. the iteration converges iff every eigenvalue of $I - \alpha H_s$ has modulus $\le 1$, i.e. $0 < \alpha < 2/\lambda_{\max}(H_s)$ (ch 06). components in the null space of $X_s$ (eigenvalue 0 of $H_s$) never change, so $\theta_\infty$ is the least-squares solution closest to $\theta_0$: with $\theta_0 = 0$, the minimum-norm solution. remember this in topic 11.

</details>

3. cost. with $P$ weights, a forward-backward pass costs $c$. estimate the time and memory of one meta-gradient with $k$ inner steps and $T$ tasks, second-order and first-order.

<details><summary>solution</summary>

first order: $k$ forward-backward passes per task for the inner loop plus one for the query, about $(k+1)cT$, and memory for one graph at a time, $O(P)$ per task beyond the activations of one pass. second order: the same forward work, but each inner step's backward pass must itself be recorded, and the outer backward pass then runs back through all $k$ inner gradients, each a Hessian-vector product costing a small multiple of $c$; total roughly $2$–$4\times$ the first-order time, and memory for all $k$ inner graphs at once, $O(kP)$ per task plus activations. memory, not time, is what limits second-order MAML with many inner steps on large networks, which motivated implicit MAML (Rajeswaran et al., 2019).

</details>

4. pretraining's target. with `n_steps=0`, the population objective is $\mathbb E_{A,\varphi,x}[(f(x) - A\sin(x - \varphi))^2]$. show that the best $f$ is $m(x) = -\frac{2\bar A}{\pi}\cos x$ with $\bar A = 2.55$, and that its loss on $x \sim U[-5, 5]$ is about $3.0$, against $4.25$ for predicting zero.

<details><summary>solution</summary>

pointwise, the minimizer is the conditional mean $m(x) = \mathbb E[A]\,\mathbb E_\varphi[\sin(x - \varphi)]$, and $\frac1\pi\int_0^\pi \sin(x - \varphi)\,d\varphi = \frac1\pi[\cos(x - \varphi)]_0^\pi = \frac{\cos(x - \pi) - \cos x}{\pi} = -\frac{2\cos x}{\pi}$. so $m(x) = -1.62\cos x$. the loss of predicting zero is $\mathbb E[A^2]\,\mathbb E[\sin^2(x-\varphi)] = \frac{5^3 - 0.1^3}{3 \cdot 4.9} \cdot \frac12 = 4.25$ (half a period of $\sin^2$ averages to $\frac12$ whatever $x$ is). the loss of $m$ is that minus $\mathbb E[m(x)^2] = 1.62^2\,\mathbb E[\cos^2 x] = 2.63 \cdot (\tfrac12 + \tfrac{\sin 10}{20}) = 1.25$, about $3.0$. this flat, low-amplitude compromise is what pretraining converges to, and a few small gradient steps from it on ten points can't turn it into a sine of the right amplitude and phase.

</details>

## experiment

the lab meta-trains your MAML on sine waves (three seeds) and compares it with pretraining under identical test-time fine-tuning, for 0 to 10 inner steps. then it varies the number of inner steps *during training*, with second- and first-order gradients, and times each. last, it adapts from points on only the left half of each wave and scores predictions on the right half. predict before each.

## check your understanding

1. why does `adapt` take the gradient with respect to the *expanded* per-task weights rather than the shared ones?

<details><summary>answer</summary>

the gradient of the summed loss with respect to the shared weights is the sum of the task gradients: every task would take the same step, the batch-average direction. with respect to the expanded tensor, slice $t$ receives only task $t$'s gradient, because no other task's loss depends on it. that's what makes each task's adaptation its own.

</details>

2. what exactly does `create_graph=True` add, and what happens to the meta-gradient without it?

<details><summary>answer</summary>

it records the operations that compute the gradient, so the gradient is a differentiable function of the weights. without it the gradient is a constant tensor; $\theta' = \theta - \alpha g$ then depends on $\theta$ only through the identity, the Jacobian $I - \alpha H$ becomes $I$, and the meta-gradient is the first-order one, $\nabla\mathcal L_q(\theta')$.

</details>

3. MAML's init and the pretrained init can have the same loss before adaptation. how can they then differ so much after one step?

<details><summary>answer</summary>

the loss at $\theta$ measures where you are; adaptation depends on the gradient field around $\theta$, which says where one step takes you. MAML's objective shapes that field: it rewards weights from which each task's support gradient points towards that task's solution, with the right magnitude for $\alpha$. two inits with equal loss can have completely different gradients for the same ten points.

</details>

4. why must a meta-learning paper report results on tasks that were never used for meta-training, and why isn't it enough to hold out query *points* of training tasks?

<details><summary>answer</summary>

the claim is about adapting to new tasks. on a training task, the init may already encode that particular wave (memorized across meta-iterations), so good query loss says nothing about learning from the support set. held-out tasks, like held-out classes in topic 10, are the meta-level test set.

</details>

5. in the linear closed form, what happens to the meta-gradient when $\alpha \lambda_{\max}(H_s) > 2$?

<details><summary>answer</summary>

$I - \alpha H_s$ has an eigenvalue below $-1$, so $(I - \alpha H_s)^k$ grows geometrically with $k$: the inner loop diverges, the query loss explodes, and the meta-gradient is huge and oscillates in sign. the inner learning rate has to respect the curvature of every task's support loss, which is why it is a sensitive hyperparameter (Antoniou et al., 2019, even learn it per layer and step).

</details>

6. Reptile updates $\theta \leftarrow \theta + \epsilon(\theta' - \theta)$ and computes no query loss at all. why is that not just pretraining?

<details><summary>answer</summary>

with one inner step, $\theta' - \theta = -\alpha \nabla\mathcal L_s(\theta)$, so Reptile *is* pretraining (SGD on the task losses). with $k \ge 2$ steps on different minibatches, a Taylor expansion of the expected update shows an extra term, of order $\alpha^2$, that moves $\theta$ in the direction that *increases* the inner product $\nabla \mathcal L_{s_1} \cdot \nabla \mathcal L_{s_2}$ between gradients on different batches of the same task: it rewards weights where a step on one batch also helps another. MAML's Hessian term produces the same kind of term (Nichol et al., 2018). the inner steps are what make it meta-learning.

</details>

## common bugs

- **in-place inner updates**: `p.data -= lr * g` or `optimizer.step()` inside the inner loop. symptom: the meta-gradient is zero or first-order, and the shared params change during adaptation. build new tensors: `p - lr * g`.
- **`create_graph` missing (or only on the last step)**: symptom: the second-order checks fail by a few percent, and the multi-step check fails while the one-step check might pass.
- **one step for the whole batch**: the gradient taken with respect to the unexpanded weights, or a mean over tasks inside the inner loss. symptom: the per-task check fails; MAML behaves like pretraining.
- **scoring on the support set**: symptom: suspiciously low meta-loss that keeps falling while test-time adaptation doesn't improve. the outer loss must use `xq, yq`.
- **inner learning rate too large**: symptom: losses of $10^3$ and NaNs after a few meta-iterations. $\alpha\lambda_{\max}(H_s) < 2$ for every task; for the sine net $\alpha = 0.01$ is safe.
- **adapting the evaluation weights with `torch.no_grad()`**: symptom: `RuntimeError: element 0 of tensors does not require grad`. the inner loop needs gradients even at test time; only the final prediction can run without them.

## going deeper

- Finn, Abbeel & Levine (2017), "Model-agnostic meta-learning for fast adaptation of deep networks", *ICML*. MAML, the sine benchmark and FOMAML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)
- Nichol, Achiam & Schulman (2018), "On first-order meta-learning algorithms". FOMAML analysed, and Reptile. [arXiv:1803.02999](https://arxiv.org/abs/1803.02999)
- Antoniou, Edwards & Storkey (2019), "How to train your MAML", *ICLR*. the instabilities and the fixes (per-step learning rates, multi-step loss). [arXiv:1810.09502](https://arxiv.org/abs/1810.09502)
- Raghu, Raghu, Bengio & Vinyals (2020), "Rapid learning or feature reuse? Towards understanding the effectiveness of MAML", *ICLR*. most of MAML's adaptation happens in the last layer. [arXiv:1909.09157](https://arxiv.org/abs/1909.09157)
- Rajeswaran, Finn, Kakade & Levine (2019), "Meta-learning with implicit gradients", *NeurIPS*. the meta-gradient without storing the inner loop. [arXiv:1909.04630](https://arxiv.org/abs/1909.04630)
- Grant, Finn, Levine, Darrell & Griffiths (2018), "Recasting gradient-based meta-learning as hierarchical Bayes", *ICLR*. MAML's init as a learned prior. [arXiv:1801.08930](https://arxiv.org/abs/1801.08930)
- Hospedales, Antoniou, Micaelli & Storkey (2021), "Meta-learning in neural networks: a survey", *IEEE TPAMI*. [arXiv:2004.05439](https://arxiv.org/abs/2004.05439)

## connections

- ← builds on: gradient descent and curvature (ch 06), the chain rule with Jacobians (ch 02, ch 03), autograd and higher-order gradients (ch 14, ch 15), honest evaluation over seeds (ch 07).
- → used in: topic 10 replaces "adapt by gradient steps" with "adapt by averaging embeddings"; topic 11 loads your `train_maml` and `adapt` and pits them against a transformer that adapts in context, with on paper 2's minimum-norm limit as the reason MAML can match least squares; the elective-portfolio capstone reports the block's project.
