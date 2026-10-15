# regularization approaches

## why this matters

topic 12 showed what fine-tuning does to an old task: the gradient of the new loss moves every weight, and nothing in it knows the old task exists. but most of those weights probably didn't matter much for the old task. a network with 200,000 weights solving 0-vs-1 relies heavily on a few thousand of them; the rest could change a lot without hurting. so here's a tempting fix: estimate how much each weight matters for the old task, and make the important ones expensive to move. the unimportant ones stay free to learn the new task.

that's the whole family of **regularization approaches**, and its two best-known members are this topic. **elastic weight consolidation** (EWC; Kirkpatrick et al., 2017) measures importance with the **Fisher information**, which you met as a property of estimators in chapter 05 and which here becomes the curvature of the old task's loss. **synaptic intelligence** (SI; Zenke, Poole & Ganguli, 2017) measures it from the training trajectory itself: how much each weight's movement contributed to lowering the loss. both add a quadratic penalty to the new task's loss, and both cost no stored data, only a copy of the weights and one importance number per weight.

they also come with a lesson about evaluation that took the field years to absorb. on the benchmark they were introduced with, they work very well. in the lab you'll measure them in the other scenario from topic 12, and see what a penalty on weights can and can't protect.

## warm-up: predict first

1. the Fisher information of a model is defined with labels drawn from the *model's own* predictions, not the true labels. if you use the true labels instead, do you get a larger or smaller importance for a well-trained weight?
2. you compute the gradient of the mean loss over a batch of 200 examples and square it. is that the same as averaging the squares of the 200 per-example gradients? which is bigger?
3. EWC's penalty strength $\lambda$ multiplies the importance. a Bayesian derivation suggests $\lambda \approx 1$ for two tasks of equal size. people use values like $10^2$ to $10^4$. why might that be?
4. a weight didn't move at all while learning task A. how important does SI think it is?
5. on split MNIST, fine-tuning keeps about 20% average accuracy when the model isn't told the task. how much do you expect EWC to improve on that?

## prerequisites check

- [ ] you've built topic 12's harness: tasks, `train_task` with its `penalty` and `after_step` hooks, `run_sequence` and the metrics (→ ch 29 · topic 12)
- [ ] you know the Fisher information $I(\theta) = \mathbb{E}[\nabla \log p\, \nabla \log p^\top]$ and that it equals the expected negative Hessian of the log-likelihood (→ ch 05 · topic 06)
- [ ] you know the Laplace approximation: a Gaussian at the posterior mode with the Hessian as precision (→ ch 12 · topic 01)
- [ ] you know what curvature does to gradient descent's step-size limit (→ ch 06 · topics 02 and 05)
- [ ] you can call `torch.func.vmap` / `jacrev`, or are ready to loop over examples (→ ch 15 · topic 07 for PyTorch basics)

## core ideas

### idea 1: importance as curvature, and the Fisher

why should importance be curvature? near a minimum $\theta^*_A$ of task A's loss, a second-order Taylor expansion says that moving the weights by $\delta$ raises the loss by about $\tfrac12 \delta^\top H \delta$, where $H$ is the Hessian. a weight whose diagonal curvature $H_{ii}$ is large is one where small moves cost a lot: important. one where $H_{ii} \approx 0$ can wander without consequence.

there's a cleaner, Bayesian way to say the same thing, which is how EWC was derived. after task A, everything you know about the weights is the posterior $p(\theta \mid A)$. learning B properly means maximizing $p(\theta \mid A, B) \propto p(B \mid \theta)\, p(\theta \mid A)$: the old posterior becomes the new prior. you can't keep the exact posterior of a neural network, but the **Laplace approximation** replaces it by a Gaussian centred at the mode $\theta^*_A$ with precision equal to the Hessian of the negative log-likelihood, $N_A$ times the per-example curvature. keep only its diagonal, take logs, and the new objective is task B's loss plus a quadratic pull towards $\theta^*_A$ weighted by each weight's curvature (on paper 2).

computing a Hessian diagonal is awkward, but for a model that outputs a probability distribution there's a substitute that only needs first derivatives. the **Fisher information** of the model at $\theta$ is

$$F = \mathbb{E}_{x \sim \text{data}}\ \mathbb{E}_{y \sim p_\theta(y \mid x)} \Big[ \nabla_\theta \log p_\theta(y \mid x)\, \nabla_\theta \log p_\theta(y \mid x)^\top \Big],$$

and it equals the expected Hessian of $-\log p_\theta(y \mid x)$ when $y$ is drawn from the model (on paper 1). its diagonal is a mean of squared gradients:

$$F_{ii} = \frac{1}{N}\sum_{n=1}^{N} \sum_{c} p_\theta(c \mid x_n) \left(\frac{\partial \log p_\theta(c \mid x_n)}{\partial \theta_i}\right)^2.$$

three details decide whether you've computed the right thing:

- **per-example gradients.** the square goes *inside* the average over examples. squaring the gradient of the averaged loss gives $(\frac1N\sum_n g_n)^2$, which near a minimum is close to zero, because the per-example gradients point every which way and cancel. that's the most common EWC bug, and it silently turns EWC into fine-tuning.
- **which $y$.** the definition draws $y$ from the model. with $C$ classes you can take the expectation exactly, weighting each class's squared gradient by $p_\theta(c \mid x)$ (the **exact** Fisher, $C$ gradients per example), or sample one $y$ per example. replacing $y$ by the true label gives the **empirical Fisher**, which is cheaper and often used, but is a different matrix: it isn't the expected Hessian, and for a model that fits its data well it differs systematically from the Fisher (Kunstner et al., 2019). it's usually *smaller*: the true label is the class the model already finds likely, whose log-probability has a small gradient, while the model's own samples sometimes pick unlikely classes, whose gradients are large (warm-up question 1).
- **which classes.** the Fisher belongs to the distribution the model was trained to output. task-incrementally, that's the softmax over the task's own classes, so restrict the logits to those classes before the log-softmax.

per-example gradients are cheap with `torch.func`: `vmap(jacrev(f))` over a chunk of examples returns the Jacobian of every example's log-probabilities in one call. a plain loop over examples (and over classes) is slower but equally correct.

> **spec · stage 1.** `fisher_diagonal(model, X, *, y=None, classes=None, mode="exact", chunk_size=64)`: returns a dict over the model's trainable parameters (`requires_grad`, names from `named_parameters()`) mapping each name to a detached tensor of the parameter's shape: the mean over the rows of `X` of the squared per-example gradient of $\log p_\theta(y \mid x)$. `mode="exact"`: sum over classes $c$ weighted by $p_\theta(c \mid x)$; `mode="empirical"`: $y$ the given label (`ValueError` if `y` is `None`). `classes`: if given, the log-softmax runs over those columns only (labels then must lie in `classes`). other modes: `ValueError`. the model's parameters must not change. `chunk_size` is a hint for vectorized implementations.

> **▶ build: stage 1.** implement `fisher_diagonal` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: elastic weight consolidation

with the Fisher in hand, EWC is a few lines. after finishing task $k$, **consolidate**: compute $F^{(k)}$ on task $k$'s data at the current weights and store a copy of those weights as the **anchor** $\theta^{(k)}$. while training any later task, add

$$\Omega_{\text{EWC}}(\theta) = \frac{\lambda}{2} \sum_{k} \sum_i F^{(k)}_i \big(\theta_i - \theta^{(k)}_i\big)^2$$

to the loss. each weight is held near its old value by a spring whose stiffness is its Fisher; the "elastic" in the name. the penalty is zero at the anchor and its gradient, $\lambda F_i(\theta_i - \theta^{(k)}_i)$, pulls each weight back in proportion to how far it has strayed and how much it matters.

two design questions. **how many quadratics?** Kirkpatrick et al. keep one per task, so memory grows linearly with the number of tasks (two copies of the network per task). **online EWC** (Schwarz et al., 2018; see also Huszár, 2018) keeps a single running Fisher, $F \leftarrow \gamma F + F^{(k)}$, anchored at the latest weights, with $\gamma \le 1$ letting old tasks fade. (Schwarz et al. apply $\gamma$ inside the penalty rather than to the stored Fisher; the two differ by a constant factor, which $\lambda$ absorbs.) with $\gamma = 1$ it's the natural recursive version of the Bayesian argument: the posterior after tasks $0..k$ is the prior for task $k+1$.

**how big is $\lambda$?** on paper 2 shows that, with the training loss written as a mean over task B's $N_B$ examples, the Bayesian value is $\lambda = N_A / N_B$: about 1 for equal-sized tasks. in practice the best values are orders of magnitude larger. part of the reason is that the diagonal Laplace approximation is poor (a network's posterior is far from a Gaussian with independent coordinates), part that one epoch of training doesn't reach a mode, so the Fisher there underestimates the curvature that matters. treat $\lambda$ as a hyperparameter, and remember it buys stability at the price of plasticity: the stiffer the springs, the less room for the new task.

one practical trap: a stiff quadratic limits the step size. plain gradient descent on $\tfrac{\lambda}{2} F_i \theta_i^2$ diverges once $\eta \lambda F_i > 2$ (on paper 3), and the Fishers of output-layer weights are the largest. with SGD at $\eta = 0.05$ and $\lambda = 100$, the limit is $F = 2/(\eta\lambda) = 0.4$; the largest output-layer Fisher entries of the first three permuted-MNIST tasks were 0.16, 0.22 and 0.27, which sum to 0.65 where they fall on the same weight, and the run went to NaN on its fourth task, the first one trained against three consolidated Fishers. Adam rescales each weight's step by its gradient's magnitude and doesn't have this cliff, which is one reason this block's labs use it.

> **spec · stage 2.** `EWC(lam, *, mode="exact", n_samples=1000, online=False, gamma=1.0)`. `consolidate(model, task, *, scenario="class")`: computes `fisher_diagonal` on the first `n_samples` examples of `task` (all if `None`), with `classes=task["classes"]` only when `scenario == "task"`, labels `task["y"]` and the stored `mode`; stores it with a detached copy of the current trainable weights as the anchor. without `online`, each call adds one (anchor, Fisher) pair; with `online`, there is at most one pair: $F \leftarrow \gamma F + F_{\text{new}}$ and the anchor becomes the current weights. `penalty(model)`: $\tfrac{\lambda}{2} \sum_{\text{pairs}} \sum_i F_i(\theta_i - \text{anchor}_i)^2$ as a differentiable scalar tensor, $0$ before the first `consolidate`.

> **▶ build: stage 2.** implement `EWC`.

> **🔮 predict:** with EWC on permuted MNIST, as $\lambda$ goes from $0$ to $10^5$, how does the accuracy on each *new* task change, and how does the final average accuracy change?

### idea 3: synaptic intelligence

EWC estimates importance once, at the end of a task, from a separate pass over data. SI asks a different question, during training: **how much of the drop in loss is each weight responsible for?**

the loss changes along the training trajectory $\theta(t)$ by

$$L(\theta(t_1)) - L(\theta(t_0)) = \int_{t_0}^{t_1} \nabla L(\theta(t)) \cdot \theta'(t)\, dt = \sum_i \int_{t_0}^{t_1} g_i(t)\, \theta_i'(t)\, dt,$$

a path integral that splits exactly into one term per weight. weight $i$'s share of the *decrease* is $\omega_i = -\int g_i\, d\theta_i$. with discrete optimizer steps, approximate it by summing, over steps, minus the gradient that produced the step times the step itself:

$$\omega_i \approx -\sum_{\text{steps}} g_i \,\Delta\theta_i.$$

for plain SGD, $\Delta\theta_i = -\eta g_i$, so every term is $\eta g_i^2 \ge 0$; with Adam or momentum a term can be negative (the weight moved uphill for a while). a weight that moved far while barely reducing the loss gets a small $\omega_i$ relative to its movement. to turn the contribution into a stiffness, SI divides by how far the weight travelled over the task, damped by $\xi > 0$ so weights that didn't move don't divide by zero:

$$\Omega_i \mathrel{+}= \frac{\omega_i}{(\Delta_i)^2 + \xi}, \qquad \Delta_i = \theta_i^{\text{end}} - \theta_i^{\text{start}},$$

accumulated over tasks. the penalty is $c \sum_i \Omega_i (\theta_i - \theta^{*}_i)^2$, anchored at the weights at the end of the last task. on a quadratic loss, $\Omega_i$ recovers half the curvature (on paper 4), so SI's $c$ plays the role of EWC's $\lambda$, in different units: importance per squared distance travelled, not per example.

SI's cost is one extra copy of the weights for the running sums, and no extra pass over data: everything comes from gradients the optimizer computed anyway. in the harness, that's the `after_step` hook. one simplification: Zenke et al. accumulate the gradient of the *unregularized* task loss; the harness's hook sees `.grad` after `backward` of the full loss, penalty included. the penalty's own gradient adds roughly $-c\,\Omega_{\text{old}}\Delta^2$ to $\omega$, so after normalizing, about $-c\,\Omega_{\text{old}}\Delta^2/(\Delta^2 + \xi)$ to $\Omega$, which is small because a well-anchored weight moves little ($\Delta^2 \ll \xi$); in exchange SI stays a pure plug-in.

> **spec · stage 3.** `SI(c, *, xi=0.1)`. `begin_task(model)`: records the current trainable weights as the task's start point and as $\theta_{\text{prev}}$, resets the running $\omega$ to zero (creating zero importances $\Omega$ the first time). `after_step(model)`: for every trainable parameter with a gradient, $\omega \mathrel{+}= -\text{grad} \cdot (\theta - \theta_{\text{prev}})$; then $\theta_{\text{prev}} \leftarrow \theta$ for every parameter. `consolidate(model)`: $\Omega \mathrel{+}= \omega / ((\theta - \theta_{\text{start}})^2 + \xi)$ and the anchor becomes a copy of the current weights. `penalty(model)`: $c \sum_i \Omega_i (\theta_i - \text{anchor}_i)^2$, a differentiable scalar, $0$ before the first `consolidate`. the attribute `xi` holds $\xi$.

> **▶ build: stage 3.** implement `SI`. the worked example below is stage 3's first check.

> **🔮 predict:** the Fisher measures curvature at the end of training; SI's $\Omega$ measures contribution along the way. if you rank 180,000 weights by each, how well will the rankings agree?

## worked example

SI on a two-weight model, by hand. start at $\theta = (1, -2)$, $c = 2$, $\xi = 0.1$.

- step 1: gradient $g = (0.5, -1.0)$, the optimizer moves to $(0.8, -1.5)$, so $\Delta\theta = (-0.2, 0.5)$ and $\omega \mathrel{+}= -g \cdot \Delta\theta = (0.1, 0.5)$.
- step 2: $g = (0.2, 0.4)$, move to $(0.7, -1.7)$, $\Delta\theta = (-0.1, -0.2)$, $\omega \mathrel{+}= (0.02, 0.08)$. total $\omega = (0.12, 0.58)$.
- consolidate: the weights travelled $\Delta = (0.7, -1.7) - (1, -2) = (-0.3, 0.3)$, so $\Omega = \omega / (0.09 + 0.1) = (0.6316,\ 3.0526)$, anchored at $(0.7, -1.7)$.
- penalty at $\theta = (1, -1)$: the offsets from the anchor are $(0.3, 0.7)$, so $c \sum_i \Omega_i \delta_i^2 = 2\,(0.6316 \cdot 0.09 + 3.0526 \cdot 0.49) = 3.1053$.

both weights moved the same total distance, but the second contributed almost five times as much loss decrease, so it's almost five times as stiff. your `SI` must give $3.10526$.

## on paper

1. show that the Fisher equals the expected Hessian of the negative log-likelihood when $y \sim p_\theta(y \mid x)$: $\mathbb{E}_y[-\nabla^2 \log p_\theta] = \mathbb{E}_y[\nabla \log p_\theta\, \nabla\log p_\theta^\top]$. why does the identity fail for the empirical Fisher?

<details><summary>solution</summary>

$\nabla^2 \log p = \frac{\nabla^2 p}{p} - \frac{\nabla p\, \nabla p^\top}{p^2} = \frac{\nabla^2 p}{p} - \nabla\log p\,\nabla\log p^\top$. take the expectation over $y \sim p_\theta$: $\mathbb{E}_y[\nabla^2 p / p] = \sum_y \nabla^2 p_\theta(y \mid x) = \nabla^2 \sum_y p_\theta(y\mid x) = \nabla^2 1 = 0$. so $\mathbb{E}_y[-\nabla^2 \log p] = \mathbb{E}_y[\nabla\log p\,\nabla\log p^\top]$. the cancellation used the model's own distribution; with the data's labels the first term is $\sum_y q(y \mid x)\nabla^2 p_\theta / p_\theta$, which isn't zero unless the model matches the data distribution exactly.

</details>

2. derive EWC from the Laplace approximation. task A has $N_A$ examples and its posterior is approximated by $\mathcal{N}(\theta^*_A, (N_A \operatorname{diag} F)^{-1})$, where $F$ is the per-example Fisher. show that maximizing $\log p(B \mid \theta) + \log p(\theta \mid A)$ is equivalent to minimizing task B's *mean* loss plus $\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta^*_{A,i})^2$ with $\lambda = N_A/N_B$.

<details><summary>solution</summary>

$\log p(B \mid \theta) = -N_B\, \bar L_B(\theta)$ with $\bar L_B$ the mean negative log-likelihood. the Gaussian gives $\log p(\theta \mid A) = -\frac{N_A}{2}\sum_i F_i (\theta_i - \theta^*_{A,i})^2 + \text{const}$. negate and divide by $N_B$: minimize $\bar L_B(\theta) + \frac{N_A}{2N_B}\sum_i F_i(\theta_i - \theta^*_{A,i})^2$. for equal-sized tasks that's $\lambda = 1$.

</details>

3. gradient descent with step size $\eta$ on the penalty alone, $\frac{\lambda}{2}F(\theta - a)^2$ for one weight: show it converges iff $\eta\lambda F < 2$. what does that imply for EWC with SGD as tasks accumulate?

<details><summary>solution</summary>

the update is $\theta - a \leftarrow (\theta - a) - \eta\lambda F(\theta - a) = (1 - \eta\lambda F)(\theta - a)$, which shrinks iff $|1 - \eta\lambda F| < 1$, i.e. $0 < \eta\lambda F < 2$. beyond that the weight oscillates with growing amplitude. without the online variant the stiffnesses of all consolidated tasks add up, so the largest summed Fisher grows with every task, and a $(\lambda, \eta)$ pair that was stable early can blow up later. that's the permuted-MNIST NaN in idea 2.

</details>

4. a single weight with loss $L = \frac{h}{2}(\theta - m)^2$ is trained by gradient flow from $\theta_0$ until it reaches $m$. compute SI's $\omega$ and, with $\xi = 0$, $\Omega$. compare SI's penalty with EWC's for this weight.

<details><summary>solution</summary>

$\omega = -\int g\,d\theta = L(\theta_0) - L(m) = \frac h2(\theta_0 - m)^2$, and $\Delta = m - \theta_0$, so $\Omega = \omega/\Delta^2 = h/2$. SI's penalty is $c\,\frac h2 (\theta - m)^2$; EWC's is $\frac\lambda2 F(\theta - m)^2$, and for a well-specified model the Fisher is the curvature, $F = h$. so the two coincide with $c = \lambda$, in this ideal case. in practice SI's $\Omega$ is measured on a trajectory that doesn't reach the minimum and EWC's $F$ is per example, which is why the useful ranges of $c$ and $\lambda$ differ.

</details>

## experiment

the lab plugs your `EWC` and `SI` into your topic-12 harness. first a sweep of $\lambda$ and $c$ over five orders of magnitude on five permuted-MNIST tasks, recording final accuracy, backward transfer and how well each new task is learned. then both methods on split MNIST, in the task-incremental and the class-incremental scenario, three seeds each. last, you compare the two importance measures directly on one trained network, and test them by adding noise to the weights each one ranks highest. predict before each; the second experiment is the one this topic is remembered for.

## check your understanding

1. why does squaring the gradient of the *batch mean* loss give a nearly useless importance estimate near a minimum?

<details><summary>answer</summary>

at a minimum the mean gradient is (close to) zero, by definition, even though individual examples' gradients are large and point in different directions. the Fisher is the *variance-like* quantity $\mathbb{E}[g g^\top]$ of the per-example gradients; the square of their mean is $(\mathbb{E} g)^2 \approx 0$. the two differ by exactly the variance, which is the part that measures curvature.

</details>

2. EWC needs the Fisher of task A, computed with task A's data, at the end of task A. could you compute it later, say at the end of task B, from a few stored examples of A?

<details><summary>answer</summary>

you could, but it would describe curvature at the wrong point: the weights have moved, and the penalty's quadratic is anchored at $\theta^*_A$. you'd also be storing data, at which point replay (topic 14) is usually a better use of those examples. EWC's appeal is that after consolidating it never touches task A's data again.

</details>

3. what does EWC store, in floats, for a network with $P$ weights after $T$ tasks? online EWC? SI?

<details><summary>answer</summary>

EWC: an anchor and a Fisher per task, $2PT$. online EWC: $2P$ (one running Fisher, one anchor). SI: $\Omega$, the anchor, plus the running $\omega$, start point and previous weights during training, about $5P$ at peak and $2P$ between tasks. for a 784–200–200–10 network ($P \approx 2 \times 10^5$), even $2P$ is the memory of about 500 MNIST images, a fact topic 14's project will make you weigh.

</details>

4. SI's importance can be negative for some weights. how, and is it a problem?

<details><summary>answer</summary>

$\omega_i$ sums $-g_i\Delta\theta_i$; with momentum or Adam, or with a noisy minibatch gradient, a weight can move in a direction that raises the (minibatch) loss for a while, contributing negative terms. a negative $\Omega_i$ would make the penalty *reward* moving the weight away from its anchor. in practice these are small and few, but robust implementations clip $\Omega$ at zero; the spec doesn't, so the checks are exact.

</details>

5. in the class-incremental scenario, a weight-space penalty is anchored to weights that produce *no* outputs for future classes. why does that matter?

<details><summary>answer</summary>

after task 0, the network's output units for classes 2–9 have never been trained to fire; the Fisher of task 0 says nothing about them (task 0's loss doesn't depend on how they behave on 0s and 1s, beyond keeping them low). when task 1 trains them to fire, nothing in the penalty stops them from also firing on 0s, which is exactly how class-incremental forgetting happens. EWC protects the *function on task 0's outputs*, not the competition between old and new outputs.

</details>

## common bugs

- **square of the mean instead of mean of squares**: symptom: the Fisher check fails by orders of magnitude, and EWC behaves like fine-tuning. compute per-example gradients.
- **the anchor is a reference, not a copy**: symptom: the penalty is always zero (the anchor moves with the weights). store `p.detach().clone()`.
- **Fisher over all classes in the task-incremental scenario**: symptom: the restricted-Fisher check fails. restrict the logits to `task["classes"]` before the log-softmax when `scenario="task"`.
- **NaN after a few tasks with SGD**: symptom: losses become `nan` once several tasks are consolidated. $\eta\lambda F_{\max} > 2$: lower $\lambda$ or $\eta$, or use Adam.
- **SI using the gradient of the wrong step**: symptom: the hand example is off. after a step, `.grad` holds the gradient computed *before* the step, which is the one that produced $\Delta\theta$; read it in `after_step`, then update $\theta_{\text{prev}}$.
- **forgetting `begin_task`**: symptom: SI's second task divides by the distance travelled since the very beginning. reset the start point and $\omega$ for every task.

## going deeper

- Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in neural networks", *PNAS* 114(13). EWC. [doi:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114) · [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- Zenke, Poole & Ganguli (2017), "Continual learning through synaptic intelligence", *ICML*. SI. [arXiv:1703.04200](https://arxiv.org/abs/1703.04200)
- Huszár (2018), "Note on the quadratic penalties in elastic weight consolidation", *PNAS* 115(11). why the Bayesian recursion gives one penalty, not one per task. [doi:10.1073/pnas.1717042115](https://doi.org/10.1073/pnas.1717042115)
- Schwarz et al. (2018), "Progress & compress: a scalable framework for continual learning", *ICML*. online EWC. [arXiv:1805.06370](https://arxiv.org/abs/1805.06370)
- Aljundi, Babiloni, Elhoseiny, Rohrbach & Tuytelaars (2018), "Memory aware synapses: learning what (not) to forget", *ECCV*. importance from output sensitivity, without labels. [arXiv:1711.09601](https://arxiv.org/abs/1711.09601)
- Kunstner, Hennig & Balles (2019), "Limitations of the empirical Fisher approximation for natural gradient descent", *NeurIPS*. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)
- van de Ven & Tolias (2019), "Three scenarios for continual learning". the scenario-dependence of EWC and SI. [arXiv:1904.07734](https://arxiv.org/abs/1904.07734)
- MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, chapter 27 (Laplace's method). [free online](https://www.inference.org.uk/itila/book.html)

## connections

- ← builds on: topic 12's harness (`penalty` and `after_step` hooks), the Fisher information (ch 05 · topic 06), the Laplace approximation (ch 12), curvature and step-size limits (ch 06), PyTorch and `torch.func` (ch 15).
- → used in: topic 14 compares memory budgets against EWC's $2P$ floats; the information-geometry elective (topics 18–20) treats the Fisher as a metric and builds natural gradient and K-FAC on the same per-example gradients; the elective-portfolio capstone reports these methods against fine-tuning.
