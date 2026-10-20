# a tensor autograd engine

## why this matters

your `Value` from topic 02 can differentiate anything, one number at a time. a modest MNIST network has about 200,000 weights and sees 128 images per step: tens of millions of Python objects per step as scalar `Value`s. the fix is a bigger unit: let every node hold a whole array, and every operation push an *array* of gradients back through itself. that's what PyTorch, JAX and TensorFlow do, and what you build here.

two new problems appear with arrays, and both are where real autograd bugs live. **broadcasting**: a bias of shape $(3,)$ added to a batch of shape $(128, 3)$ was silently copied 128 times, so its gradient must be *summed* back over the copies. **shape bookkeeping**: every backward rule must return a gradient with exactly its input's shape. `gradcheck` keeps you honest, comparing your engine with finite differences on any function you can write. `mlfs.tensor.Tensor` then runs every network in this chapter, and chapter 15 checks it against PyTorch.

## warm-up: predict first

1. $b$ has shape $(3,)$ and $x$ has shape $(2, 3)$. after $L = (x + b)$`.sum()`, what's $\partial L/\partial b$? what shape?
2. $y = A x$ with $A$ of shape $(m, n)$. you know $g = \partial L/\partial y$ (shape $(m,)$). what's $\partial L/\partial x$, and why is it not $A g$?
3. what's the gradient of $\max(1, 3, 3)$ with respect to its three inputs?
4. you train with a bias gradient that's accidentally divided by the batch size. does the loss still go down?
5. to check a function with a vector output, you could `.sum()` its outputs and compare gradients. what kind of bug would that miss?

## prerequisites check

- [ ] you can build a scalar computational graph and backpropagate through it in topological order (→ ch 14 · topic 02)
- [ ] you know NumPy's broadcasting rules and `keepdims` (→ ch 00 · topic 02)
- [ ] you can derive $\nabla_W\|XW - Y\|_F^2$ with the trace trick (→ ch 02 · topic 07)
- [ ] you have `numerical_gradient` from `mlfs.numgrad` and know why a gradient check uses relative error (→ ch 02 · topic 07)
- [ ] you can compute logsumexp and sigmoid without overflow (→ ch 00 · topic 03)

## core ideas

### idea 1: broadcasting has an adjoint

when NumPy evaluates `x + b` with `x` of shape $(2, 3)$ and `b` of shape $(3,)$, it behaves as if `b` were first copied into a $(2, 3)$ array $B$ with $B_{ij} = b_j$. so every entry $b_j$ influences two entries of the result, and by the multivariable chain rule its gradient is the **sum** of theirs:

$$\frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial B_{ij}}.$$

in general broadcasting (1) prepends size-1 axes until the ranks match and (2) stretches every size-1 axis to the other operand's size. going backward, undo both: sum the gradient over the axes that were prepended, then over the axes that were stretched (keeping them, with size 1). that's `unbroadcast(grad, shape)`, and it's the single most important helper in the engine: every binary operation calls it on both operands.

a cleaner view: broadcasting is a linear map $\mathcal B$, and the backward pass multiplies by its *transpose*, defined by $\langle G, \mathcal B x\rangle = \langle \mathcal B^\top G, x\rangle$. on paper 3 shows that $\mathcal B^\top$ is "sum over the copies"; the tests check this identity directly.

> **▶ build: stage 1.** implement `unbroadcast(grad, shape)`: sum away the extra leading axes, then sum (with `keepdims=True`) every axis where `shape` has a 1 but `grad` doesn't, and return an array of exactly `shape`.

### idea 2: tensors, vector-Jacobian products and the backward pass

a `Tensor` is a float64 array `.data` plus three things for autograd: `.requires_grad` (should gradients flow here?), `.grad` (filled in by `backward`), and a record of how it was made: its `_parents` and a function `_vjp`. the first two are exactly what `mlfs.optim`'s optimizers look for (ch 06: anything with `.data`, updated in place, and `.grad`), so a `Tensor` with `requires_grad=True` is already a trainable parameter.

**every operation is a vector-Jacobian product.** if $y = f(x)$ and the loss $L$ depends on $y$, the chain rule says

$$\bar x = J_f(x)^\top\,\bar y, \qquad \text{where } \bar y = \frac{\partial L}{\partial y},\ \bar x = \frac{\partial L}{\partial x}.$$

(the bar notation for "gradient of the final loss with respect to this" is standard in the autodiff literature.) we never form $J_f$: for elementwise $y = \tanh(x)$ the Jacobian is a huge diagonal matrix, but its VJP is just $\bar x = \bar y \odot (1 - y^2)$. so each operation supplies a small function `vjp(g)` that maps $\bar y$ (an array of $y$'s shape) to one $\bar x$ per input (arrays of the inputs' shapes). for the arithmetic operations, with $\odot$ elementwise:

| op | forward | $\bar a$ | $\bar b$ |
|---|---|---|---|
| add | $a + b$ | $\bar y$ | $\bar y$ |
| sub | $a - b$ | $\bar y$ | $-\bar y$ |
| mul | $a \odot b$ | $\bar y \odot b$ | $\bar y \odot a$ |
| div | $a / b$ | $\bar y / b$ | $-\bar y \odot a / b^2$ |
| pow | $a^p$ ($p$ a scalar) | $\bar y \odot p\,a^{p-1}$ | |

each of these is then passed through `unbroadcast` to its operand's shape. (for $p = 0$, return zeros: $0 \cdot a^{-1}$ would be `nan` at $a = 0$.)

all operations create their output through one helper, `Tensor.from_op(data, parents, vjp)`. it records the parents and the VJP only if gradients are wanted: some parent has `requires_grad`, and recording is switched on (idea 5). otherwise the result is a plain constant. this is the only place the engine decides whether a graph exists, which makes `no_grad` a one-line feature later. non-Tensor operands (arrays, Python floats) are wrapped as constants with `Tensor._lift`, and `__array_ufunc__ = None` tells NumPy to hand `ndarray + Tensor` over to `Tensor.__radd__`.

**the backward pass** is topic 02's, with arrays. `backward(grad)` starts from $\bar y$ = `grad` (default: ones, allowed only when the output has one element: a loss), then visits the nodes so that each node is processed only after **every** node that uses it, the reverse of a topological order. at each node it calls the node's `vjp` and *adds* the results into its parents' running gradients (a node used twice gets two contributions). leaves, the tensors you created yourself, accumulate into `.grad`: the first gradient a leaf receives is stored as a fresh **copy** (VJPs can hand out shared or read-only arrays: `add` gives both parents the same `g`, `sum` returns a broadcast view), and later ones are added with `leaf.grad + g`; intermediate results don't keep theirs. accumulation is a feature: it's how PyTorch sums gradients over several backward calls, and why every training loop calls `zero_grad()` first.

build the topological order **iteratively**, with an explicit stack and a `seen` set: a recursive search hits Python's recursion limit (about 1,000 frames) on a merely long graph, and without `seen` a node shared by two children is expanded twice, so the cost doubles with every level of sharing. keep the running gradients in a dictionary keyed by `id(node)`, so nothing stale survives from a previous call.

> **🔮 predict:** $a = 3$, $b = 2a$, $c = a + b$, $d = b \cdot c$, so $\partial d/\partial a = 12a = 36$. suppose your backward visits $d, b, c, a$: each node right after the first node that sends it a gradient (a depth-first order, but not a topological one). what comes out for $\partial d/\partial a$?

> **▶ build: stage 2.** implement `from_op`, `backward`, `+`, `-`, `*`, `/`, `**` (scalar exponents) and unary `-`, plus `sum(axis, keepdims)` (idea 3), which you need to turn anything into a scalar loss. the reflected operators (`__radd__` etc.) are given.

### idea 3: elementwise functions and reductions

**elementwise functions** have diagonal Jacobians, so their VJPs are elementwise products with the derivative, which you can often write in terms of the output: $\exp$ gives $\bar y \odot y$, $\tanh$ gives $\bar y \odot (1 - y^2)$, $\sigma$ gives $\bar y \odot y(1 - y)$, $\log$ gives $\bar y / x$, and ReLU gives $\bar y \odot [x > 0]$. the derivative of ReLU at exactly 0 doesn't exist; like PyTorch, use 0. compute $\sigma(x)$ as $\exp(-\operatorname{logaddexp}(0, -x))$ so that $x = \pm 1000$ never overflows.

**reductions** go the other way: many inputs, fewer outputs. `sum` copies one number into every entry it summed, so its VJP **broadcasts** $\bar y$ back to the input's shape. that's the mirror image of idea 1: summing and broadcasting are each other's transposes. the only fiddly part is `keepdims=False`: the reduced axes have disappeared from $\bar y$, so put them back as size-1 axes (`np.expand_dims`) before broadcasting. the given helper `Tensor._expand(g, shape, axis, keepdims)` does exactly this. `mean` is `sum` divided by the number of entries each output averaged.

`max` passes the gradient to the entry that won. with ties the function has a kink and no unique gradient; the convention here (and PyTorch's for `amax`) is to **split it evenly** among the tied entries, so the gradient of $\max(1, 3, 3)$ is $(0, \tfrac12, \tfrac12)$ (warm-up 3). build a mask `x == max` (with `keepdims=True` so it broadcasts) and divide it by its count.

`logsumexp` is the workhorse of classification losses (ch 00 · topic 03). compute it shifted, $m + \log\sum_i e^{x_i - m}$ with $m = \max_i x_i$, and notice that its gradient is the softmax:

$$\frac{\partial}{\partial x_j}\log\sum_i e^{x_i} = \frac{e^{x_j}}{\sum_i e^{x_i}} = \operatorname{softmax}(x)_j,$$

which you can compute as $\exp(x - \operatorname{lse})$ without ever overflowing.

> **▶ build: stage 3.** implement `exp`, `log`, `tanh`, `sigmoid`, `relu`, `mean`, `max` and `logsumexp`.

### idea 4: matrix products, shapes and indexing

**matmul.** for $Y = AB$ with $A$ of shape $(n, k)$ and $B$ of shape $(k, m)$, the differential is $dY = dA\,B + A\,dB$, and the trace trick (on paper 1) gives

$$\bar A = \bar Y B^\top \quad (n \times m \cdot m \times k = n \times k), \qquad \bar B = A^\top \bar Y \quad (k \times n \cdot n \times m = k \times m).$$

the shapes alone almost force the answer (warm-up 2: for $y = Ax$, $\bar x = A^\top\bar y$ has $x$'s shape, while $A\bar y$ doesn't even type-check unless $A$ is square, and then it's wrong). `np.matmul` also accepts 1-D operands (a vector is promoted to a $1 \times k$ row or a $k \times 1$ column, and that axis is dropped from the result) and stacks of matrices with broadcast batch axes. handle the first by promoting in the VJP too, and the second with `np.swapaxes(·, -1, -2)` for the transposes plus `unbroadcast` for the batch axes: a weight matrix shared across a batch of sequences gets the sum of every sequence's gradient.

**reshape and transpose** move numbers without changing them, so their VJPs move the gradient back: reshape $\bar y$ to the old shape; transpose $\bar y$ by the *inverse* permutation (`np.argsort(axes)`).

**indexing** `x[idx]` selects entries; its VJP scatters $\bar y$ back into a zero array of $x$'s shape. with integer arrays an entry can be selected twice (`x[[0, 0, 2]]`), and then its gradient must be the sum of both copies'. `full[idx] += g` silently keeps only one of them (NumPy's buffered assignment); `np.add.at(full, idx, g)` adds every copy. this is how a cross-entropy picks `logits[np.arange(n), labels]`, and how an embedding table gets its gradient in chapter 17.

finally, three functions that combine tensors: `concatenate` (VJP: `np.split` the gradient at the same boundaries), `stack` (VJP: take slice $i$ along the new axis), and `where(cond, x, y)` (VJP: the gradient goes to $x$ where `cond` holds and to $y$ elsewhere, each unbroadcast).

> **▶ build: stage 4.** implement `@`, `reshape`, `transpose`, `__getitem__`, `concatenate`, `stack` and `where`.

### idea 5: switching the tape off, and checking everything

**no_grad.** evaluating a model, or updating its weights, needs no graph, and recording one wastes memory: each node keeps its inputs alive until the graph is dropped (topic 06 measures this). `with no_grad():` flips a global switch that `from_op` consults; inside the block every result is a constant. make it a context manager that remembers the previous state in `__enter__` and restores it in `__exit__`, so that nested blocks and exceptions leave the switch as they found it.

**gradcheck** is your engine's test suite in one function. given $f$ and some inputs, it compares backward's gradients with central differences (your `numerical_gradient` from ch 02 · topic 07) for every input that requires grad. two details matter:

- **reduce with random weights.** $f$ may return an array, and `backward` needs a scalar. reducing with `.sum()` weights every output equally, and a bug that only permutes or mixes gradients between outputs is invisible under equal weights (warm-up 5): the gradient of `sum(reverse(x))` is all ones whether or not your VJP un-reverses. use $L = \sum_i w_i f_i$ with fixed random $w$, which distinguishes every output.
- **compare with a mixed tolerance**, `np.allclose(analytic, numeric, atol, rtol)`: the absolute part handles gradients near zero, the relative part large ones. in float64 with $\varepsilon = 10^{-6}$, correct gradients agree to about $10^{-9}$; bugs are usually off by $10^{-1}$ or more.

leave no trace: restore each input's data after nudging it and its `.grad` afterwards. and remember what gradcheck can't do: it verifies that backward differentiates *the function you wrote*. if you wrote the wrong loss (the lab has a classic), it will happily confirm the wrong gradient.

> **🔮 predict:** a bias gradient that's mistakenly averaged over the batch instead of summed is $n$ times too small. with gradient descent on a linear regression, will the loss still go down? will the bias reach its true value?

> **▶ build: stage 5.** implement `no_grad.__enter__`/`__exit__` and `gradcheck`.

## worked example

$x = \begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$, $b = (1, 0, -1)$, $c = 2$, and $L = \operatorname{sum}\big((x \odot b + c)^2\big)$. both $b$ (shape $(3,)$) and $c$ (shape $()$) are broadcast to $(2, 3)$.

**forward.** $x \odot b = \begin{pmatrix}1&0&-3\\4&0&-6\end{pmatrix}$, $u = x \odot b + c = \begin{pmatrix}3&2&-1\\6&2&-4\end{pmatrix}$, and $L = 9 + 4 + 1 + 36 + 4 + 16 = 70$.

**backward.** $\bar L = 1$; sum's VJP broadcasts it to ones of shape $(2, 3)$; the square's VJP gives $\bar u = 2u = \begin{pmatrix}6&4&-2\\12&4&-8\end{pmatrix}$. then:

- $\bar c$: the add sends $\bar u$ to $c$, and `unbroadcast` to shape $()$ sums everything: $6 + 4 - 2 + 12 + 4 - 8 = 16$.
- the add sends $\bar u$ to $x \odot b$ too; the multiply gives $x$ the gradient $\bar u \odot b = \begin{pmatrix}6&0&2\\12&0&8\end{pmatrix}$ (same shape, nothing to sum).
- and $b$ gets $\bar u \odot x = \begin{pmatrix}6&8&-6\\48&20&-48\end{pmatrix}$, unbroadcast to $(3,)$ by summing the rows: $\bar b = (54, 28, -54)$.

the tests check $L = 70$, $\bar b = (54, 28, -54)$, $\bar c = 16$ and $\bar x$. (without `unbroadcast`, $b$'s "gradient" would have shape $(2, 3)$.)

## on paper

1. **the matmul VJP.** with $Y = AB$ and $dL = \operatorname{tr}(\bar Y^\top dY)$, derive $\bar A = \bar Y B^\top$ and $\bar B = A^\top \bar Y$.

<details><summary>solution</summary>

$dY = dA\,B + A\,dB$, so $dL = \operatorname{tr}(\bar Y^\top dA\,B) + \operatorname{tr}(\bar Y^\top A\,dB)$. cycle the first trace: $\operatorname{tr}(B\bar Y^\top dA) = \operatorname{tr}\big((\bar Y B^\top)^\top dA\big)$, so $\bar A = \bar YB^\top$. the second is already in the form $\operatorname{tr}\big((A^\top\bar Y)^\top dB\big)$, so $\bar B = A^\top\bar Y$. for a linear layer $Y = XW^\top + b$ (torch layout, $W$ of shape (out, in)): $\bar X = \bar YW$, $\bar W = \bar Y^\top X$ (a sum of outer products over the batch), $\bar b = \sum_i \bar y_i$ (unbroadcast).

</details>

2. **softmax as a gradient, and the max convention.** show that $\nabla\operatorname{logsumexp}(x) = \operatorname{softmax}(x)$, and that $\operatorname{logsumexp}(x/\tau)\cdot\tau \to \max(x)$ as $\tau \to 0^+$. what does the gradient converge to when $x$ has two tied maxima?

<details><summary>solution</summary>

$\partial_j \log S = e^{x_j}/S$ with $S = \sum_i e^{x_i}$. for the limit, $\tau\log\sum_i e^{x_i/\tau} = m + \tau\log\sum_i e^{(x_i - m)/\tau}$; the sum tends to the number of maxima $k$, so the second term is $\tau\log k \to 0$. the gradient of $\tau\operatorname{lse}(x/\tau)$ is $\operatorname{softmax}(x/\tau)$, which tends to $1/k$ on each of the $k$ maxima and 0 elsewhere: exactly the "split evenly" convention. it's the natural choice, not an arbitrary one.

</details>

3. **unbroadcast is the transpose of broadcast.** let $\mathcal B$ copy $b \in \mathbb R^{3}$ into $B \in \mathbb R^{2\times 3}$ with $B_{ij} = b_j$. show that $\langle G, \mathcal Bb\rangle = \langle \mathcal B^\top G, b\rangle$ with $(\mathcal B^\top G)_j = \sum_i G_{ij}$.

<details><summary>solution</summary>

$\langle G, \mathcal Bb\rangle = \sum_{ij}G_{ij}b_j = \sum_j\big(\sum_iG_{ij}\big)b_j$. this holds for every $b$, so the column sums are the transpose applied to $G$. the same argument works for any broadcast: each entry of the small array appears in several places of the big one, and its transpose collects them.

</details>

4. **the cheap gradient principle.** a forward pass costs $C$ flops. explain why a reverse sweep costs at most a small constant times $C$ for an MLP, whatever the number of parameters, while finite differences cost $2n\,C$ for $n$ parameters.

<details><summary>solution</summary>

each forward operation gets one VJP in the backward pass, and every VJP here costs about as much as its operation, or twice as much for a matmul (two products, $\bar YB^\top$ and $A^\top\bar Y$, each the size of the forward one). so backward costs about $2C$, and forward plus backward about $3C$, independent of $n$ (Griewank & Walther bound it by a small constant for any program). finite differences need two forward passes per parameter. for 200,000 parameters that's 400,000 forward passes against about 3. this is reverse mode's advantage for "many inputs, one output" (topic 02's lab; topic 06 has the mirror image).

</details>

## experiment

`lab.py` uses your engine for three experiments:

1. three versions of a bias-add operation, two with plausible gradient bugs, train a linear regression. does the loss still go down with a wrong gradient? what does your `gradcheck` say?
2. a regression whose loss has a shape bug (a $(n, 1)$ prediction minus a $(n,)$ target). what does the model learn, and does `gradcheck` notice?
3. the time of a backward pass against a forward pass, as the network grows.

## check your understanding

1. why does `from_op` drop the graph when no parent requires grad, instead of always recording it?

<details><summary>answer</summary>

nothing could ever use that graph: gradients only flow to tensors that require them, and an operation on constants has no path to any. recording it would keep every intermediate array alive for nothing. it's also what makes constants (data, targets) free.

</details>

2. why must leaves accumulate (`+=`) into `.grad` rather than assign?

<details><summary>answer</summary>

so that gradients from several backward calls add up (gradient accumulation over micro-batches, or two losses backpropagated separately). assigning would keep only the last one. the cost is that you must reset gradients yourself: `optimizer.zero_grad()` before each step, or the gradients of every previous step pile up.

</details>

3. your bias has shape $(1, 3)$ and the batch is $(128, 3)$. which axis does `unbroadcast` sum, and does it keep it?

<details><summary>answer</summary>

axis 0, with `keepdims`, so the gradient has shape $(1, 3)$ like the bias. if the bias were $(3,)$, the leading axis would be summed away instead, giving $(3,)$.

</details>

4. why do we need `np.add.at` in the VJP of indexing, but not in the VJP of slicing `x[1:3]`?

<details><summary>answer</summary>

a slice never selects the same entry twice, so `full[1:3] = g` is correct. integer-array indices can repeat (`x[[0, 0]]`), and buffered assignment `full[idx] += g` writes each repeated entry once, losing the other contributions. `np.add.at` is unbuffered and adds them all.

</details>

5. `gradcheck` passes. list two things that could still be wrong.

<details><summary>answer</summary>

(a) the function itself: a wrong loss (wrong shapes, mean over the wrong axis, a missing term) with a correct gradient passes. (b) behaviour at points you didn't test: a ReLU kink, ties in `max`, overflow for large inputs, or a shape (1-D operands, batched matmul) you didn't try. gradcheck is only as good as its test points.

</details>

6. why is the backward of `logsumexp` computed as $\exp(x - \operatorname{lse})$ and not as `exp(x) / exp(x).sum()`?

<details><summary>answer</summary>

`exp(1000)` overflows to `inf` and `inf / inf` is `nan`. subtracting the (already computed, stable) logsumexp first keeps every exponent at most 0, so the softmax comes out exactly.

</details>

7. (the predict callout in idea 2.) with the visiting order $d, b, c, a$, what comes out, and why?

<details><summary>answer</summary>

24. $d$ sends $\bar b = c = 9$ and $\bar c = b = 6$. $b$ is visited next and passes $2 \cdot 9 = 18$ to $a$. then $c$ passes $6$ to $a$ *and* $6$ to $b$, but $b$ has already been processed, so that contribution ($2 \cdot 6 = 12$ for $a$) is lost: $18 + 6 = 24$ instead of $36$. a node may only pass its gradient on once it's complete, i.e. after every node that uses it. the reverse of a topological order guarantees exactly that.

</details>

## common bugs

- **forgetting `unbroadcast`.** symptom: a bias gradient of shape (batch, out) instead of (out,); the optimizer crashes on `p.data -= lr * p.grad`, or worse, broadcasts the bias into a matrix. sum back to the operand's shape in *every* binary op, both operands.
- **assigning instead of accumulating.** `node.grad = g`, or `grads[parent] = g` when a node has two children. symptom: `x * x` has gradient $x$ instead of $2x$; `gradcheck` fails on any function that reuses a value.
- **recursive topological sort.** symptom: `RecursionError` on a long chain (thousands of operations), even though the gradient code is correct.
- **storing a leaf's first gradient without copying it.** symptom: after `(a + b).sum().backward()`, `a.grad` and `b.grad` are the same array (clipping one in place clips both), or a second backward fails with "output array is read-only" because the stored gradient is a broadcast view.
- **`full[idx] += g` for indexing.** symptom: correct gradients for distinct indices, too-small ones when an index repeats (a label appearing twice in a batch, a word twice in a sentence).
- **reducing a vector output with `.sum()` in gradcheck.** symptom: permutation and mixing bugs pass the check.
- **`keepdims` mismatches.** symptom: the VJP of `sum(axis=1)` fails to broadcast against the input (`(4,)` against `(4, 5)`) or broadcasts along the wrong axis. put the reduced axes back with `np.expand_dims` first.

## going deeper

- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- Griewank & Walther (2008), *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed., SIAM, chapters 3–4 (reverse mode and its cost bound). [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
- Paszke et al. (2017), "Automatic differentiation in PyTorch", NIPS Autodiff Workshop. the design you've rebuilt. [openreview.net/forum?id=BJJsrmfCZ](https://openreview.net/forum?id=BJJsrmfCZ)
- Karpathy (2020), *micrograd*: topic 02's model. [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Maclaurin, Duvenaud & Adams (2015), *Autograd*: NumPy autodiff with an "unbroadcast" helper like yours. [github.com/HIPS/autograd](https://github.com/HIPS/autograd)

## connections

- ← builds on: the scalar engine `Value` (topic 02), forward-mode `Dual` (ch 03 · topic 04: tangents pushed forward, where this pulls cotangents back), matrix gradients and `mlfs.numgrad` (ch 02 · topic 07), broadcasting (ch 00 · topic 02), stable logsumexp and sigmoid (ch 00 · topic 03), the `.data`/`.grad` optimizer protocol (ch 06 · topic 04).
- → used in: `mlfs.nn` modules, losses and `fit` (topic 04), universal approximation experiments (topic 05), VJPs and gradient checkpointing (topic 06), the MNIST capstone, initialization and normalization layers with hand-written backward passes (ch 15), and the gradient-parity test against PyTorch (ch 15 · topic 07).
