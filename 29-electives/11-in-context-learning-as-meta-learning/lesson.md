# in-context learning as meta-learning

## why this matters

give a large language model a prompt like "sea otter → loutre de mer, cheese → fromage, peppermint →" and it continues "menthe poivrée". nobody fine-tuned it on translation; its weights didn't change. it read two examples and did the task. Brown et al. (2020) called this **in-context learning** and showed it improves with model size, and it raised an awkward question: is the model *learning* from the examples in its prompt, or recognizing a task it memorized during pretraining?

Garg, Tsipras, Liang & Valiant (2022) found a way to ask the question cleanly. train a transformer on sequences $(x_1, y_1, x_2, y_2, \dots, x_n, y_n)$ where each sequence comes from its own random linear function $y = w^\top x$, and ask it to predict each $y_i$ from what came before. a new sequence is a new $w$, never seen in training, so memorization is impossible: to predict well, the network must infer $w$ from the examples in its context. it does, and its errors track those of least squares. the network has learned a learning algorithm, and it runs it in its forward pass.

seen from this block, that's meta-learning with the inner loop moved inside the model. MAML (topic 09) adapts by gradient steps; prototypical networks (topic 10) adapt by averaging embeddings; a transformer adapts by attention over the support set, with whatever algorithm its training found. in this topic you build the task family and the classical answer (least squares), show that one layer of linear attention can implement a step of gradient descent exactly (von Oswald et al., 2023), build and train the transformer, and in the lab's project compare it with least squares and with your own MAML, fairly tuned, in and out of the training distribution.

## warm-up: predict first

1. with 3 noiseless examples of a random linear function in 5 dimensions, can a well-trained transformer beat least squares?
2. a single layer of *linear* self-attention is trained on this task. what algorithm do you expect its weights to implement, if any?
3. least squares with 5 noisy examples in 5 dimensions: about as good as with 10 examples, a bit worse, or catastrophically worse?
4. at test time the inputs are twice as large as in training. which breaks: least squares, MAML, or the transformer?
5. MAML on a linear model, with enough inner steps, compared with least squares: worse, equal, or better?

## prerequisites check

- [ ] you can solve least squares and ridge regression, and know the minimum-norm solution for underdetermined systems (→ ch 02 · topic 04, ch 08 · topics 01 and 02)
- [ ] you know the posterior mean of Bayesian linear regression with a Gaussian prior (→ ch 12 · topic 01)
- [ ] you've built multi-head attention, causal masks and `TransformerBlock` (→ ch 18 · topics 01 and 03)
- [ ] you've trained a GPT with pre-shifted targets and know why the loss at every position trains the model at once (→ ch 18 · topic 04)
- [ ] you've built MAML's `adapt` and `train_maml` (→ ch 29 · topic 09), whose on paper 2 shows that gradient descent from 0 converges to the minimum-norm least-squares solution

## core ideas

### idea 1: the task family and its optimal learner

a **task** is a weight vector $w \sim \mathcal N(0, I_d)$; its data are inputs $x \sim \mathcal N(0, I_d)$ and labels $y = w^\top x + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2)$ ($\sigma = 0$ unless stated). a sequence of $n$ pairs from one task is a prompt; the prediction at position $i$ uses the **context** $(x_1, y_1, \dots, x_{i-1}, y_{i-1})$ (here indexed from 0: prediction $i$ uses the $i$ pairs before it) and the query $x_i$. scoring every position at once gives an **error curve**: test error as a function of the number of in-context examples.

because the task distribution is known, the best possible predictor is known too. the posterior over $w$ given the context is Gaussian, and the Bayes-optimal prediction is the posterior mean, which is **ridge regression** with $\lambda = \sigma^2$ (penalty $\lambda\lVert w\rVert^2$ against the *summed* squared error). as $\sigma \to 0$ ridge becomes the **minimum-norm least-squares** solution, $\hat w = X^+ y$: with fewer than $d$ examples, the component of $w$ outside the examples' span is unknown, and its posterior mean is 0. so noiseless least squares is the benchmark no learner can beat on average, and its error curve has a simple shape (on paper 1): $d - i$ with $i < d$ examples, exactly 0 from $i = d$ on.

with noise the picture changes in an instructive way. least squares interpolates the noise, and with $i$ near $d$ the system is nearly square and badly conditioned, so the noise is amplified without bound: the expected error is infinite at $i = d$ and $i = d + 1$ (on paper 4). ridge regularizes and stays smooth. this "interpolation peak" is the simplest form of double descent (Hastie et al., 2022), and you'll see it in the lab.

> **spec · stage 1.** `sample_linear_tasks(n_tasks, n_points, dim, *, noise_std=0.0, x_scale=1.0, generator=None)`: a dict of float32 tensors `xs (B, n, d)` with entries $\mathcal N(0, \text{x\_scale}^2)$, `w (B, d)` with entries $\mathcal N(0, 1)$, and `ys (B, n)` $= $ `xs @ w` plus independent $\mathcal N(0, \text{noise\_std}^2)$ noise; all draws from `generator`. `least_squares_predictions(xs, ys, *, ridge=0.0)`: `(B, n)` in the input's dtype; entry $i$ is $x_i^\top \hat w_i$ where $\hat w_i$ fits the first $i$ pairs: minimum-norm least squares when `ridge == 0`, otherwise $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$ with summed (not averaged) errors; entry 0 is 0. `position_mse(preds, ys)`: `(n,)`, the mean over tasks of the squared error at each position.

> **▶ build: stage 1.** implement `sample_linear_tasks`, `least_squares_predictions` and `position_mse` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: one step of gradient descent is one layer of linear attention

how could a transformer possibly implement regression? von Oswald et al. (2023) gave a construction that fits in a few lines. take gradient descent on the context loss $L(w) = \frac12 \sum_j (w^\top x_j - y_j)^2$ from $w_0 = 0$. its gradient at 0 is $-\sum_j y_j x_j$, so one step with step size $\eta$ gives $w_1 = \eta \sum_j y_j x_j$ and the prediction

$$\hat y_q = w_1^\top x_q = \eta \sum_j y_j\, x_j^\top x_q.$$

now look at **linear self-attention** (attention without the softmax): with tokens $e_j = (x_j, y_j) \in \mathbb R^{d+1}$ and a query token $e_q = (x_q, 0)$, one layer updates

$$e_q \leftarrow e_q + W_{PV} \sum_j e_j\, \big(e_j^\top W_{KQ}\, e_q\big),$$

where $W_{KQ}$ plays the role of $W_K^\top W_Q$ (keys against queries) and $W_{PV}$ of the value and output projections. the attention score $e_j^\top W_{KQ} e_q$ becomes $x_j^\top x_q$ if $W_{KQ}$ has the identity in its $x$-block and zeros elsewhere; the value $W_{PV} e_j$ becomes $(0, \dots, 0, \eta y_j)$ if $W_{PV}$ has $\eta$ in its bottom-right corner and zeros elsewhere. then the last coordinate of the updated query token is exactly $\eta \sum_j y_j x_j^\top x_q$: one GD step (on paper 2). stacking layers gives more steps (with a learned preconditioner), and von Oswald et al. showed that linear-attention layers *trained* on this task converge to this construction. the lab trains one and looks.

two lessons hide here. first, an attention layer is a natural place for a learning algorithm: it computes a weighted sum over the context, and gradients of squared losses are weighted sums over the data. second, one GD step is a weak learner: with 10 examples in 5 dimensions its best error is still $d(d+1)/(n+d+1) \approx 1.9$, against least squares' 0 (on paper 3). a transformer that matches least squares must be doing something better than one step: more steps, preconditioning, or something closer to a direct solve; Akyürek et al. (2023) show that transformers can represent GD and closed-form ridge, and that trained ones behave like ridge or least squares depending on depth and noise.

> **spec · stage 2.** `gd_predictions(xs, ys, *, lr, n_steps=1)`: `(B, n)`; entry $i$ is $x_i^\top w$ after `n_steps` of gradient descent from $w = 0$ on $\frac12\sum_{j<i}(w^\top x_j - y_j)^2$ with step size `lr`; entry 0 is 0. `lsa_predict(xs, ys, x_query, W_kq, W_pv)`: `xs (B, n, d)`, `ys (B, n)` the context, `x_query (B, d)`, weights `(d+1, d+1)`; returns `(B,)`, the last coordinate of $e_q + W_{PV}\sum_j e_j(e_j^\top W_{KQ} e_q)$ with $e_j = (x_j, y_j)$, $e_q = (x_q, 0)$. `gd_lsa_weights(dim, lr)`: a pair `(W_kq, W_pv)` of `(dim+1, dim+1)` tensors for which `lsa_predict` equals one step of `gd_predictions` with step size `lr`.

> **▶ build: stage 2.** implement `gd_predictions`, `lsa_predict` and `gd_lsa_weights`. the worked example below is one of stage 2's checks.

> **🔮 predict:** in the worked example, one GD step predicts 0.5 and two steps predict 0. what does least squares predict, and how many GD steps would you need to get close to it?

### idea 3: a transformer that reads a regression problem

the model has to turn a regression problem into tokens. following Garg et al., each pair contributes two tokens: an $x$-token $(x_i, 0)$ and a $y$-token $(0, \dots, 0, y_i)$, both in $\mathbb R^{d+1}$, interleaved as $x_1, y_1, x_2, y_2, \dots$. a linear **read-in** maps tokens to $d_\text{model}$ dimensions, learned **positional embeddings** are added, a stack of causal `TransformerBlock`s (yours, from chapter 18) processes the sequence, and a final layer norm and a linear **read-out** give one number per token.

where you read the prediction is the whole game. the prediction for $y_i$ is read at the $x_i$ token: with a causal mask that position sees $x_1, y_1, \dots, x_{i-1}, y_{i-1}, x_i$, which is exactly the context plus the query. read it at the $y_i$ token instead and the model sees the answer; read it at $y_{i-1}$ and it hasn't seen the query yet. the checks perturb individual inputs and verify that each prediction depends on exactly what it should.

one forward pass scores every context length at once, like a GPT that predicts every next token: the loss averaged over positions trains the model to regress from 0, 1, 2, ... examples simultaneously.

> **spec · stage 3.** `interleave(xs, ys)`: `(B, 2n, d+1)`, position $2i$ is $(x_i, 0)$ and position $2i+1$ is $(0, \dots, 0, y_i)$. `ICLTransformer(dim, *, max_points=40, d_model=64, n_layers=4, n_heads=4)`: an `nn.Module` with attributes `dim` and `max_points` and submodules `read_in` (`nn.Linear(dim + 1, d_model)`), `pos` (a learned `nn.Parameter` of shape `(2 * max_points, d_model)`), `blocks` (an `nn.ModuleList` of `n_layers` `mlfs.transformer.TransformerBlock(d_model, n_heads)`, run with `is_causal=True`), `ln_f` (`nn.LayerNorm`) and `read_out` (`nn.Linear(d_model, 1)`). `forward(xs, ys)` returns `(B, n)`, the read-out at the $x$-token positions $0, 2, 4, \dots$, so that prediction $i$ depends on $x_1, y_1, \dots, x_i$ only; `ValueError` if the input dimension isn't `dim` or $n >$ `max_points`.

> **▶ build: stage 3.** implement `interleave` and `ICLTransformer`.

### idea 4: training as meta-learning, and what to claim

training is the outer loop of a meta-learner: every step draws a batch of **fresh** tasks (new $w$, new $x$), runs the model, and minimizes the squared error averaged over tasks and positions. no task is ever seen twice, so there is nothing to memorize except the procedure. Garg et al. add a **curriculum** (fewer dimensions and shorter prompts first) and train for 500,000 steps; the lab trains for 2,500, which is enough to see the effect and not enough to reach least squares, and the training run itself turns out to be one of the largest sources of variance.

evaluating such a model honestly means three things. compare with the right baselines at every context length: least squares (Bayes-optimal here), the best single GD step, predicting 0, and other meta-learners. report uncertainty: the error at one context length is a mean over test tasks, with a standard error, and the *training run* is random too (the lab notes report five training seeds). and test what the model learned by moving off the training distribution: label noise, inputs of a different scale, more examples than it was trained with. least squares is an algorithm, so it works at any scale; if the transformer is "running least squares", it should too. the lab's project checks.

> **spec · stage 4.** `train_icl(model, *, n_points, n_steps, batch_size=64, lr=1e-3, noise_std=0.0, seed=0)`: puts the model in train mode, creates `torch.optim.Adam(model.parameters(), lr=lr)` and one `torch.Generator` seeded with `seed`; each of `n_steps` steps draws `sample_linear_tasks(batch_size, n_points, model.dim, noise_std=noise_std, generator=...)` and takes one step on the mean squared error of `model(xs, ys)` against `ys` (over tasks and positions); returns the list of losses (floats) and leaves the model in eval mode.

> **▶ build: stage 4.** implement `train_icl`. its slow check trains a small transformer in 2 dimensions for 800 steps and asks that its error drop far below its one-example error once the context passes the dimension, as least squares' does.

> **🔮 predict:** a transformer trained only on noiseless prompts meets prompts with label noise. near $i = d$, will it behave like least squares (and blow up) or like ridge?

## worked example

$d = 2$, context $(1, 0) \mapsto 2$ and $(1, 1) \mapsto 1$, query $x_q = (0, 1)$ (whose true label in the check is $-1$).

- **one GD step**, $\eta = 0.5$: $w_1 = 0.5\,(2\,(1, 0) + 1\,(1, 1)) = (1.5, 0.5)$, prediction $w_1^\top x_q = 0.5$.
- **as attention:** scores $x_j^\top x_q = 0$ and $1$, values $\eta y_j = 1$ and $0.5$, so the update to the query's last coordinate is $0 \cdot 1 + 1 \cdot 0.5 = 0.5$. the same number, as it must be.
- **a second step:** residuals $w_1^\top x_j - y_j = (1.5 - 2, 2 - 1) = (-0.5, 1)$, gradient $-0.5\,(1, 0) + 1\,(1, 1) = (0.5, 1)$, $w_2 = (1.5, 0.5) - 0.5\,(0.5, 1) = (1.25, 0)$, prediction $0$.
- **least squares:** two equations in two unknowns, $w_1 = 2$ and $w_1 + w_2 = 1$, so $w = (2, -1)$ and the prediction is $-1$, exactly right.

gradient descent is heading towards $(2, -1)$, but slowly: the context's Hessian $X^\top X = \begin{pmatrix} 2 & 1 \\ 1 & 1\end{pmatrix}$ has eigenvalues $2.62$ and $0.38$, a condition number of 7, so with $\eta = 0.5$ the slow direction shrinks by a factor $0.81$ per step and it takes about 20 steps to get within 1% of the answer. your `gd_predictions` must give $0.5$ and $0$, and your `lsa_predict` with `gd_lsa_weights(2, 0.5)` must give $0.5$.

## on paper

1. show that noiseless minimum-norm least squares with $i < d$ examples ($x, w \sim \mathcal N(0, I)$) has expected error $d - i$ on a new query, and that no predictor can do better.

<details><summary>solution</summary>

the $i$ examples determine $Pw$ exactly, where $P$ projects onto their span (dimension $i$ almost surely); the minimum-norm solution is $\hat w = Pw$. the error on $x_q$ is $(x_q^\top (I - P) w)^2$, with expectation $\mathbb E[w^\top (I-P) x_q x_q^\top (I-P) w] = \mathbb E[w^\top (I-P) w] = \operatorname{tr}(I - P) = d - i$. since $w \sim \mathcal N(0, I)$ and $(I - P)w$ is independent of the data, its posterior is $\mathcal N(0, I - P)$ and its posterior mean is 0: least squares *is* the posterior mean, hence optimal in squared error.

</details>

2. verify the construction of idea 2: with $W_{KQ} = \begin{pmatrix} I_d & 0 \\ 0 & 0\end{pmatrix}$ and $W_{PV} = \begin{pmatrix} 0 & 0 \\ 0 & \eta\end{pmatrix}$, show that linear self-attention returns the one-step GD prediction. is the construction unique?

<details><summary>solution</summary>

$e_j^\top W_{KQ} e_q = (x_j, y_j)^\top (x_q, 0) $ restricted to the $x$-block, $= x_j^\top x_q$. $W_{PV} e_j = (0, \dots, 0, \eta y_j)$. so the update is $(0, \dots, 0, \eta\sum_j y_j x_j^\top x_q)$ and the last coordinate of $e_q$ (initially 0) becomes the GD prediction. not unique: $W_{KQ}$'s $x$-block can be $cA$ and $W_{PV}$'s corner $\eta/c$ for any $c \ne 0$, and $A = I$ can be replaced by any matrix that acts as a preconditioner (one step of preconditioned GD), and more. writing $c = W_{PV}[y, y]$, $u = W_{PV}[y, x]$, $A = W_{KQ}[x, x]$ and $b = W_{KQ}[y, x]$, a general layer predicts $\sum_j (u^\top x_j + c\,y_j)(x_j^\top A x_q + y_j\, b^\top x_q)$. the two terms linear in $y_j$ are $\sum_j y_j x_j^\top (cA + ub^\top) x_q$: one GD step preconditioned by $cA + ub^\top$, the identifiable quantity, which the lab plots. the term $u^\top x_j\, x_j^\top A x_q$ contains no $y$ and the term $c\,y_j^2\, b^\top x_q$ is even in $w$, so neither correlates with the target $w^\top x_q$: they only add variance, which is why training drives $u$ and $b$ (separately) towards 0.

</details>

3. for a single GD step from 0 with $n$ examples (noiseless, $x, w \sim \mathcal N(0, I_d)$), find the best step size and its error. use the Wishart moment $\mathbb E[(X^\top X)^2] = n(n + d + 1) I$.

<details><summary>solution</summary>

$w_1 = \eta X^\top X w$, so the error on a fresh query is $\mathbb E\lVert \eta X^\top X w - w \rVert^2 = \mathbb E[w^\top(\eta^2 (X^\top X)^2 - 2\eta X^\top X + I)w] = d\,(\eta^2 n(n+d+1) - 2\eta n + 1)$. minimizing over $\eta$: $\eta^* = \frac{1}{n + d + 1}$, error $d\big(1 - \frac{n}{n + d + 1}\big) = \frac{d(d+1)}{n+d+1}$. for $n = 10$, $d = 5$: $\eta^* = 1/16$ and error $30/16 = 1.875$, against least squares' 0.

</details>

4. with label noise $\sigma^2$ and $i \ge d$ examples, show that least squares' expected error is $\sigma^2\big(1 + \frac{d}{i - d - 1}\big)$ for $i \ge d + 2$, using $\mathbb E[(X^\top X)^{-1}] = \frac{I}{i - d - 1}$. what happens at $i = d$?

<details><summary>solution</summary>

$\hat w - w = (X^\top X)^{-1}X^\top \varepsilon$, so the prediction error on $x_q$ is $\sigma^2$ (the query's own noise) plus $\mathbb E[x_q^\top (X^\top X)^{-1} X^\top \varepsilon\varepsilon^\top X (X^\top X)^{-1} x_q] = \sigma^2\,\mathbb E\operatorname{tr}((X^\top X)^{-1}) = \sigma^2 \frac{d}{i - d - 1}$. for $d = 5$, $\sigma = 0.5$: $0.56$ at $i = 10$. the inverse-Wishart mean is infinite for $i \le d + 1$: with $i = d$ a square $X$ has a smallest singular value near 0 often enough that the expected error diverges, and any sample mean you compute is dominated by a few tasks. ridge with $\lambda = \sigma^2$ caps the amplification, which is why it is Bayes-optimal here.

</details>

## experiment

the lab first trains a single layer of linear self-attention and compares its weights with the gradient-descent construction. then it trains your transformer and plots its error curve against least squares, one GD step and predicting 0. the **project** pits the transformer against least squares and your topic-09 MAML (meta-trained and tuned per context length on validation tasks) on the training distribution, with label noise, and with inputs twice as large, with standard errors and a compute budget. predict before each.

## check your understanding

1. why is least squares, not ridge, the Bayes-optimal benchmark in the noiseless lab setting?

<details><summary>answer</summary>

the posterior mean under $w \sim \mathcal N(0, I)$ and noise $\sigma^2$ is ridge with $\lambda = \sigma^2$; with $\sigma = 0$ that's $\lambda \to 0$, the minimum-norm least-squares solution (on paper 1). any $\lambda > 0$ shrinks the known component of $w$ for no reason.

</details>

2. the transformer is trained on fresh tasks at every step. why does that make "memorizing the training set" impossible, and what can it still overfit to?

<details><summary>answer</summary>

no $w$ repeats, so knowing training tasks doesn't help on test tasks; the only useful thing to learn is how to infer $w$ from the context. it can still overfit to the *distribution* of tasks and inputs: the input scale, the noise level (none), the prompt length, the dimension. those are exactly the shifts the project tests.

</details>

3. why must the prediction for $y_i$ be read at $x_i$'s token, and what would each wrong choice do to the training loss?

<details><summary>answer</summary>

at $x_i$'s token a causal model has seen the context and the query, nothing more. reading at $y_i$'s token, the model sees $y_i$ in its own input: training loss goes to 0 by copying, and test predictions are meaningless. reading at $y_{i-1}$'s token, the model hasn't seen $x_i$: it can only predict the average label, 0, and the loss stays near $d$.

</details>

4. a trained linear-attention layer converges to $\eta^* I$ in its $x$-block. why doesn't it learn to approximate least squares instead?

<details><summary>answer</summary>

one layer of linear attention can only compute predictions of the form $\sum_j (\text{linear in } e_j)(\text{bilinear in } e_j, e_q)$, which is one step of (preconditioned) GD at best; least squares needs $(X^\top X)^{-1}$, a nonlinear function of the whole context. within its function class, the best it can do is the best single step, which is what it finds.

</details>

5. MAML on a linear model and least squares can make identical predictions. when, and why doesn't that make MAML pointless?

<details><summary>answer</summary>

GD from the origin converges to the minimum-norm least-squares solution (topic 09, on paper 2), MAML's learned init for this task family is close to the prior mean 0, so with enough inner steps and a stable step size MAML reproduces least squares. that's a sanity check, not a weakness: MAML is model-agnostic and gets the right answer on a family where the right answer is known; on a family without a closed form (topic 09's sines) its init carries the prior that least squares lacks.

</details>

6. what evidence would you need before saying "this transformer implements least squares"?

<details><summary>answer</summary>

matching least squares' predictions (not only its average error) on held-out tasks, at every context length; robustness where least squares is robust (input scale, prompt length within the positional range, any $w$ direction); and ideally a mechanism, weights or activations that compute the solution. agreement in distribution alone is equally consistent with a learned approximation that only works on the training distribution, which the input-scale test can reveal.

</details>

## common bugs

- **reading at the wrong token**: symptom: the causality check fails, or training loss collapses to near 0 (reading at $y_i$) or stalls near $d$ (reading at $y_{i-1}$).
- **context includes the query's own pair**: fitting least squares or GD on the first $i + 1$ pairs. symptom: perfect-looking baselines (error 0 from $i = d - 1$), and the oracle checks fail.
- **mean instead of sum in ridge or GD**: the losses here are sums over the context. symptom: the ridge check is off by a factor of $i$ in $\lambda$, the GD worked example gives the wrong numbers.
- **forgetting the causal mask**: `block(h)` without `is_causal=True`. symptom: every prediction depends on later $y$'s; training loss falls far below least squares' curve.
- **reusing a batch of tasks**: sampling once outside the training loop. symptom: training loss falls while test error stays near $d$: the model memorized a few $w$'s.
- **solving least squares in float32 near $i = d$**: symptom: noisy, inconsistent baseline values at $i = d$; compute in float64.

## going deeper

- Garg, Tsipras, Liang & Valiant (2022), "What can transformers learn in-context? A case study of simple function classes", *NeurIPS*. [arXiv:2208.01066](https://arxiv.org/abs/2208.01066)
- von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov & Vladymyrov (2023), "Transformers learn in-context by gradient descent", *ICML*. [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
- Akyürek, Schuurmans, Andreas, Ma & Zhou (2023), "What learning algorithm is in-context learning? Investigations with linear models", *ICLR*. [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)
- Brown et al. (2020), "Language models are few-shot learners", *NeurIPS*. in-context learning at scale. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Raventós, Paul, Chen & Ganguli (2023), "Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression", *NeurIPS*. with too few distinct training tasks, transformers memorize them instead. [arXiv:2306.15063](https://arxiv.org/abs/2306.15063)
- Xie, Raghunathan, Liang & Ma (2022), "An explanation of in-context learning as implicit Bayesian inference", *ICLR*. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080)
- Hastie, Montanari, Rosset & Tibshirani (2022), "Surprises in high-dimensional ridgeless least squares interpolation", *Annals of Statistics* 50(2). the interpolation peak. [arXiv:1903.08560](https://arxiv.org/abs/1903.08560)

## connections

- ← builds on: least squares, ridge and the minimum-norm solution (ch 02, ch 08), Bayesian linear regression (ch 12), attention, causal masks and `TransformerBlock` (ch 18), GPT-style training on every position (ch 18 · topic 04), topic 09's MAML (loaded in the lab) and its minimum-norm limit.
- → used in: the elective-portfolio capstone reports the project's comparison; chapter 30's research practice starts from exactly this kind of question (does the model do what we think it does?), and the mechanistic-interpretability ideas behind "looking at the weights" in experiment 1.
