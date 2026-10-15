# 14 · neural networks from scratch · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. why can't any number of stacked linear layers compute XOR, and what's the smallest change that fixes it?

<details><summary>answer</summary>

stacked linear layers compose to one linear (affine) map, $X(W_L\cdots W_1)^\top + b$, so the decision boundary is a line, and XOR isn't linearly separable (the midpoint argument). a nonlinearity between the layers fixes it: two hidden ReLU units are enough.

</details>

2. what does a node's `_backward` (or `vjp`) compute, and why must it *add* into its parents' gradients?

<details><summary>answer</summary>

given $\bar y = \partial L/\partial y$ for its output, it computes each input's share $J^\top\bar y$ (local derivative times upstream gradient). a value can feed several operations, and by the multivariable chain rule its gradient is the sum over all of them, so each contribution must be added, not assigned.

</details>

3. why must the backward pass process nodes in reverse topological order, and why should the topological sort not be recursive?

<details><summary>answer</summary>

a node may pass its gradient on only when it's complete, i.e. after every node that uses it has contributed; reverse topological order guarantees that. a recursive depth-first search needs one Python stack frame per step of the longest path, and Python's default limit (about 1,000) is exceeded by merely long graphs.

</details>

4. a bias $b$ of shape $(3,)$ is added to activations of shape $(128, 3)$. what's the shape and value of $b$'s gradient, given the output gradient $G$?

<details><summary>answer</summary>

shape $(3,)$: the column sums of $G$, $\sum_i G_{ij}$. broadcasting copied $b$ into 128 rows, and the transpose of copying is summing (unbroadcast).

</details>

5. give the VJPs of $Y = AB$.

<details><summary>answer</summary>

$\bar A = \bar YB^\top$, $\bar B = A^\top\bar Y$ (shapes force it: $\bar A$ has $A$'s shape).

</details>

6. what's the gradient of $\max(2, 5, 5)$, and why that convention?

<details><summary>answer</summary>

$(0, \tfrac12, \tfrac12)$: the gradient is split evenly between tied maxima. it's the limit of the gradient of $\tau\operatorname{logsumexp}(x/\tau)$ as $\tau \to 0$, and it's what PyTorch's `amax` does.

</details>

7. why does `gradcheck` reduce the output with fixed random weights instead of `.sum()`?

<details><summary>answer</summary>

with `.sum()` every output gets weight 1, and a bug that permutes or mixes gradients between outputs (a reversal whose VJP forgets to reverse back) gives the same total. random weights distinguish every output.

</details>

8. what does `no_grad` change, and what goes wrong if `evaluate` doesn't use it?

<details><summary>answer</summary>

inside it, operations record no graph and results don't require grad. without it, evaluation gives the same numbers but builds a graph for every batch, keeping all its intermediate arrays alive: wasted memory and time.

</details>

9. you forget `optimizer.zero_grad()`. what happens, and how would you notice?

<details><summary>answer</summary>

gradients accumulate across steps, so every update uses the sum of all gradients so far: an erratic, overshooting kind of momentum. training rarely crashes; the loss curve oscillates and stalls above where it should be. the one-batch overfitting test may still pass.

</details>

10. why do losses take logits, and how do you compute cross-entropy from them?

<details><summary>answer</summary>

softmax followed by log underflows ($\log 0$) once a probability drops below about $10^{-308}$; computing $\operatorname{logsumexp}(z) - z_y$ directly is exact for any logits, and its gradient is softmax minus one-hot.

</details>

11. a 10-class classifier starts training at a loss of 18. what's wrong?

<details><summary>answer</summary>

an untrained classifier should be near $\log 10 \approx 2.30$. 18 means huge initial logits: weights initialized at too large a scale, or unnormalized inputs (pixels 0–255).

</details>

12. why does a one-hidden-layer ReLU network with $k$ units on a 1-D input have at most $k + 1$ linear pieces, and how does a deep network beat that?

<details><summary>answer</summary>

each unit contributes one kink, so $k$ kinks cut the line into at most $k + 1$ pieces. composing piecewise-linear functions multiplies pieces (up to $pq$), so the tent map composed $d$ times has $2^d$ pieces from $2d$ units.

</details>

13. a network can represent a function exactly. does gradient descent find it?

<details><summary>answer</summary>

not necessarily. the deep sawtooth network exists but is almost never found from a random start; dead ReLUs, plateaus and poor local minima intervene. representation and learning are different questions.

</details>

14. for $f: \mathbb R^n \to \mathbb R^m$, when is forward mode cheaper than reverse mode for the full Jacobian?

<details><summary>answer</summary>

when $n < m$ (roughly): forward needs $n$ JVP passes, reverse needs $m$ VJP passes (and memory for the graph). for a scalar loss ($m = 1$) reverse wins by a factor of about $n$.

</details>

15. what does gradient checkpointing trade, and what's the best number of segments for $L$ layers?

<details><summary>answer</summary>

memory for compute: segments keep only their outputs during the forward pass and recompute their insides during backward, at the cost of one extra forward pass. memory is about $s + cL/s$ activations, minimized at $s \approx \sqrt{cL}$, giving $O(\sqrt L)$ memory.

</details>

## 2. by hand (no computer)

1. **backprop through a scalar graph.** for $L = \tanh(ab + a)$ at $a = 1$, $b = 0$, compute every node's value and gradient. (use $\tanh 1 \approx 0.7616$.)

<details><summary>solution</summary>

forward: $c = ab = 0$, $d = c + a = 1$, $L = \tanh 1 = 0.7616$. backward: $\bar d = 1 - 0.7616^2 = 0.4200$; the add passes it to $c$ and $a$: $\bar c = 0.4200$, $a$ gets $0.4200$; the product gives $\bar a \mathrel{+}= b\bar c = 0$ and $\bar b = a\bar c = 0.4200$. so $\partial L/\partial a = 0.4200$ (only the direct path contributes, since $b = 0$) and $\partial L/\partial b = 0.4200$.

</details>

2. **a linear layer with softmax cross-entropy.** one example $x = (1, 2)$, weights $W = \begin{pmatrix}1 & 0\\0 & 1\\1 & -1\end{pmatrix}$ (torch layout, 3 classes), bias 0, true class $y = 2$ (the third). compute the logits, the loss, and the gradients $\bar W$ and $\bar b$.

<details><summary>solution</summary>

logits $z = Wx = (1, 2, -1)$. $\operatorname{logsumexp}(z) = \log(e + e^2 + e^{-1}) = \log(2.7183 + 7.3891 + 0.3679) = \log 10.4752 = 2.3490$, so the loss is $2.3490 - (-1) = 3.3490$. $p = \operatorname{softmax}(z) = (0.2595, 0.7054, 0.0351)$, and $\bar z = p - e_2 = (0.2595, 0.7054, -0.9649)$. then $\bar W = \bar z x^\top = \begin{pmatrix}0.2595 & 0.5190\\0.7054 & 1.4108\\-0.9649 & -1.9298\end{pmatrix}$ and $\bar b = \bar z$.

</details>

3. **unbroadcast and a ReLU interpolant.** (a) a gradient $G = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$ flows to an operand of shape $(1, 2)$, and to another of shape $()$. what are their gradients? (b) write a one-hidden-layer ReLU network (weights and biases) through the points $(0, 0)$, $(1, 1)$, $(2, 0)$ that's linear outside $[0, 2]$.

<details><summary>solution</summary>

(a) $(1, 2)$: sum over the rows, keeping the axis: $(9, 12)$. shape $()$: sum everything: $21$.

(b) slopes $s_0 = 1$, $s_1 = -1$. $g(x) = 0 + 1\cdot(\operatorname{relu}(x) - \operatorname{relu}(-x)) + (-1 - 1)\operatorname{relu}(x - 1)$: hidden weights $(1, -1, 1)$, biases $(0, 0, -1)$, output weights $(1, -1, -2)$, output bias 0. check: $g(1) = 1$, $g(2) = 2 - 2 = 0$, $g(-1) = -1$ (slope 1 continues), $g(3) = 3 - 4 = -1$ (slope $-1$ continues).

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the core of the tensor engine** (topic 03, stages 1–3): `unbroadcast`, `Tensor.from_op`, `Tensor.backward`, the arithmetic operators with broadcasting, `sum`, and the elementwise functions and reductions. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=14-neural-networks-from-scratch/03-a-tensor-autograd-engine
mkdir -p /tmp/rebuild-14
git show HEAD:$T/exercise.py > /tmp/rebuild-14/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-14/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-14/solutions/` and run `python scripts/stubify.py /tmp/rebuild-14`.) open `/tmp/rebuild-14/exercise.py` with `marimo edit`, implement the stage 1–3 code from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-14/test_exercise.py -k "stage1 or stage2 or stage3"
```

you pass if every check is green within the hour.

**harder variant** (for testing out of the chapter with confidence): within 90 minutes in total, also finish stage 4 of the same notebook (`@`, `reshape`, `transpose`, indexing, `concatenate`, `stack`, `where`), and rebuild topic 04's `Module`, `Linear` and `fit` in their own scratch directory the same way:

```bash
pytest /tmp/rebuild-14/test_exercise.py -k stage4
pytest /tmp/rebuild-14-nn/test_exercise.py -k "stage1 or stage2 or stage4"
```

(run them from the repo root: the `mlfs.nn` notebook imports `mlfs.tensor`, which the tests resolve to the reference solution, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the idea, and the topic lessons are organized by those ideas.
