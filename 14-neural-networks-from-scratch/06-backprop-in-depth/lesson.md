# backprop in depth

## why this matters

you've built backpropagation and used it. this topic looks at it from the outside and asks two engineering questions that decide what you can train. **what exactly does one backward pass compute?** not "the gradient", in general: it computes one vector-Jacobian product, and seeing it that way explains why deep learning uses reverse mode at all, when forward mode (your `Dual` from chapter 03) is simpler, and when it would be the wrong choice. **what does it cost?** in time, a small constant times the forward pass (topic 03's lab). in memory, much more than people expect: to run backward, the engine must keep *every intermediate activation* of the forward pass alive, and for deep networks on large batches that's what fills a GPU, long before the weights do.

the fix for the memory problem is one of the neat ideas of deep learning systems: **gradient checkpointing**. forget most intermediates during the forward pass and recompute them during the backward pass, exactly when they're needed. done well, it cuts activation memory from $O(L)$ to $O(\sqrt L)$ for an $L$-layer network at the cost of about one extra forward pass. large language models are routinely trained this way. you'll add `checkpoint` to your engine, measure the trade-off, and find where the sweet spot is.

## warm-up: predict first

1. $f: \mathbb R^{1000} \to \mathbb R^{10}$. how many forward-mode passes does its full Jacobian take? how many reverse-mode passes?
2. a 50-layer MLP with width 1,000 trains on batches of 512 in float32. which is bigger: the memory for its weights or for the activations backward needs?
3. checkpointing recomputes the forward pass of a segment during backward. if every one of $L$ layers is its own segment, how much memory do you save?
4. checkpointing a network with dropout: what could go wrong in the recomputation?
5. is the minimum of `s + L/s` (over integers $s$) at $s = L/2$, $s = \sqrt L$ or $s = \log L$?

## prerequisites check

- [ ] you've built forward-mode autodiff with dual numbers and know what a JVP is (→ ch 03 · topic 04)
- [ ] your `Tensor` engine works, with `no_grad`, `from_op` and the internal `_parents`/`_vjp` record (→ ch 14 · topic 03)
- [ ] you can explain why a backward pass costs about two forward passes (→ ch 14 · topic 03)
- [ ] you know Jacobians and the chain rule $J_{g\circ f} = J_gJ_f$ (→ ch 02 · topic 07)
- [ ] you know `Module`s and how parameters are closed over by layers (→ ch 14 · topic 04)

## core ideas

### idea 1: one pass, one product

for $f: \mathbb R^n \to \mathbb R^m$ with Jacobian $J$ ($m \times n$), there are two natural products:

- a **JVP** (Jacobian-vector product) $u \mapsto Ju$: how the output moves when the input moves in direction $u$. forward mode computes it in one pass, carrying tangents alongside values. that's your `Dual` from chapter 03: seed $x + \varepsilon u$, read off $Ju$ from the dual parts.
- a **VJP** (vector-Jacobian product) $v \mapsto J^\top v$ (or $v^\top J$ as a row): how sensitive a weighted combination of the outputs, $v^\top f$, is to each input. reverse mode computes it in one forward pass plus one backward pass. that's `out.backward(v)` in your engine: `v` is the gradient you seed the output with.

the chain rule gives both for free, because $J = J_L\cdots J_2J_1$ for a composition: a JVP multiplies right to left, $J_L(\cdots(J_1u))$, in the same order as the computation; a VJP multiplies left to right, $(\cdots(v^\top J_L)\cdots)J_1$, so it has to run *after* the forward pass, backwards. neither ever forms a Jacobian, only matrix-vector products with each layer's.

the **full** Jacobian then costs $n$ JVPs (one per input direction $e_j$: column $j$) or $m$ VJPs (one per output $e_i$: row $i$). so the shape decides: a tall Jacobian (few inputs, many outputs) wants forward mode; a wide one (many inputs, few outputs) wants reverse. training is the extreme case: $n$ is the number of parameters (millions or billions) and $m = 1$, the loss. one VJP with $v = 1$ gives the whole gradient; forward mode would need a million passes (warm-up 1: 10 reverse passes against 1,000 forward ones). that asymmetry is the entire reason backpropagation, not forward-mode autodiff, runs deep learning.

there's a neat consistency check between the two, the **dot-product test**: for any $u$ and $v$, $v^\top(Ju) = (J^\top v)^\top u$, both sides being one number (on paper 3).

> **▶ build: stage 1.** implement `vjp(f, x, v)`, returning `(f(x), J^T v)` from one forward and one backward pass, and `jacobian_reverse(f, x)`, the $(m, n)$ Jacobian built row by row from $m$ backward passes through *one* forward graph. (gradients accumulate: reset the input's `.grad` before each row.)

### idea 2: what backward has to remember

look at any VJP from topic 03. `tanh` needs its output $y$ ($\bar x = \bar y \odot (1 - y^2)$); `mul` needs both inputs; `matmul` needs both matrices; `relu` needs to know where $x > 0$. so the backward pass needs, for every operation, values computed in the forward pass. your engine keeps them in the graph: every node holds its `.data`, and its `_vjp` closure captures whatever arrays it needs. as long as the loss tensor is alive, so is all of that.

how much is it? for an MLP with $L$ layers of width $d$ on a batch of $B$, each layer produces a few $B \times d$ arrays (the matrix product, the bias add, the activation), so activation memory grows like $c \cdot L \cdot B \cdot d$ numbers for a small constant $c$, against $L \cdot d^2$ numbers of weights. the ratio is about $cB/d$: once the batch is larger than the width, **activations dominate** (warm-up 2: weights $50 \cdot 10^6$ floats = 200 MB; activations about $3 \cdot 50 \cdot 512 \cdot 1000$ floats = 300 MB, plus the same again for gradients during backward). for convolutional networks and transformers on long sequences the ratio is worse, which is why activation memory, not parameter count, usually limits the batch size or sequence length you can train.

to measure it, walk the graph: `graph_nbytes(out)` adds up `.data.nbytes` over every **non-leaf** node reachable from `out` through `_parents`, each once. leaves (your inputs and weights) don't count: they exist whether or not you train. it's a slight underestimate (closures can capture extra arrays, like `max`'s tie mask), but it grows exactly the way the true activation memory does.

> **🔮 predict:** your graph for `L` identical layers holds $M$ bytes. what does it hold with `2L` layers? with the batch size doubled? under `no_grad`?

> **▶ build: stage 2.** implement `graph_nbytes(out)`. walk the graph with an explicit stack and a `seen` set, like your topological sort.

### idea 3: forget, then recompute

a **checkpointed** segment `y = checkpoint(fn, x)` makes a different trade. in the forward pass, run `fn` under `no_grad`: nothing inside it is recorded, so its intermediates are freed as soon as they're used, and the result is a single graph node holding only `y` and a reference to its inputs. in the backward pass, when that node's VJP is called with $\bar y$, recompute: run `fn` again with grad recording on, on fresh copies of the inputs, and backpropagate $\bar y$ through the freshly built little graph. return the fresh inputs' gradients as the node's VJP. the recomputed graph lives only while its segment is being differentiated, then it's dropped.

three details make it correct:

- **parameters closed over by `fn`.** a layer's weights aren't usually passed as inputs; `fn` just uses them. during the recomputation they're ordinary leaves of the fresh graph, so `y.backward(g)` accumulates their gradients straight into their `.grad`. that's correct, and it means the checkpointed node must exist in the graph **even if no input requires grad** (the input is often plain data). `from_op` would drop it ("no parent requires grad"), so build the node yourself: `requires_grad = True`, `_parents = inputs`, `_vjp = your recompute function`. (PyTorch's original checkpoint had exactly this gotcha, and warned about it.)
- **`no_grad` outside.** if grad recording is off when `checkpoint` is called, it's just `fn(*inputs)` as a constant.
- **determinism.** the recomputation must produce the same values as the first run. for layers with randomness (dropout, chapter 15) the random state must be saved and restored (warm-up 4); here every function is deterministic.

the gradients are *identical* to the plain call, not just close: the recomputation repeats the same floating-point operations in the same order.

> **▶ build: stage 3.** implement `checkpoint(fn, *inputs)`: forward under `no_grad`, a hand-built node whose VJP recomputes `fn` once and backpropagates through it.

### idea 4: $\sqrt L$ checkpointing

checkpointing the whole network as one segment saves the activations during the forward pass, but the backward pass then recomputes the entire network with a full graph: the peak is no better (slightly worse) than not checkpointing at all. checkpointing every layer separately keeps one output per layer: that saves the intermediates *inside* each layer (a factor of $c$, the number of arrays per layer), but memory still grows linearly with $L$ (warm-up 3). the trick is in between: split the $L$ layers into $s$ segments of $L/s$ layers each, and checkpoint each segment. then the memory has two parts:

- the $s$ segment outputs, kept for the whole backward pass: about $s$ activations;
- one segment's full graph, rebuilt while it's being differentiated: about $c\,L/s$ activations.

the total $s + cL/s$ is minimized at $s = \sqrt{cL}$, giving $2\sqrt{cL}$ (warm-up 5, on paper 2): memory $O(\sqrt L)$ instead of $O(L)$. the price is one extra forward pass per segment, so about one extra forward pass in total: training time grows by roughly a third (forward + backward ≈ 3 forward passes, plus 1). Chen et al. (2016) popularized this for deep learning; Griewank's earlier "revolve" schedule nests checkpoints recursively and gets $O(\log L)$ memory for $O(L\log L)$ compute.

`checkpoint_sequential(functions, x, segments)` applies a list of layer functions in order, splitting them into `segments` contiguous chunks the way `np.array_split` does (sizes differ by at most one, larger chunks first; never more segments than layers), and checkpoints each chunk.

> **🔮 predict:** you checkpoint a 16-layer network with 1, 2, 4, 6, 8 and 16 segments. which gives the lowest peak memory during backward, and is 1 segment better or worse than no checkpointing?

> **▶ build: stage 4.** implement `checkpoint_sequential(functions, x, segments)`.

## worked example

a small graph, counted by hand. $x$ is $10 \times 4$ and $W$ is $4 \times 4$, both leaves with `requires_grad`. then

```python
h = x @ W            # (10, 4): 40 float64 = 320 bytes
a = h.tanh()         # (10, 4): 320 bytes
out = (a * a).sum()  # a * a: 320 bytes; the sum: one float, 8 bytes
```

the non-leaf nodes are `h`, `a`, `a * a` and `out`, so `graph_nbytes(out)` is $320 + 320 + 320 + 8 = 968$ bytes. $x$ and $W$ are leaves (they'd exist anyway), and $a$ is used twice but stored once. now wrap the middle in a checkpoint: `c = checkpoint(lambda t: (t @ W).tanh(), x)`, then `out2 = (c * c).sum()`. the graph holds `c` (320), `c * c` (320) and the sum (8): 648 bytes. the 320 bytes of `h` are gone; during backward they're recomputed for a moment, then freed again. on this toy the saving is a third; on the lab's 16-layer network, 8 checkpointed segments cut the memory held after the forward pass from 50 activations to 9, and the peak during backward from 54 to 21. the tests check the 968 bytes, and that a checkpointed segment adds exactly its output's bytes.

## on paper

1. **Jacobian costs.** for $f: \mathbb R^n \to \mathbb R^m$ built from operations whose JVPs and VJPs each cost about as much as the operation, compare the cost of the full Jacobian by forward and by reverse mode, in units of one evaluation of $f$. when are they equal? what about the gradient of a scalar loss of $10^6$ parameters?

<details><summary>solution</summary>

forward: $n$ JVPs, each about 1–3 evaluations, so $O(n)$ evaluations. reverse: one forward pass plus $m$ backward passes, about $O(m)$ evaluations (each backward pass costs about 2 evaluations), plus the memory for the graph. they're equal (up to constants) when $n \approx m$. for a scalar loss, reverse needs 1 backward pass and forward needs $10^6$ JVPs: a millionfold difference.

</details>

2. **the best number of segments.** with memory $M(s) = s + cL/s$, find the minimizing $s$ and the minimal memory. for $L = 16$ and $c = 3$, which integer $s$ is best, and how does the minimum compare with the $\approx cL = 48$ activations of the plain graph?

<details><summary>solution</summary>

$M'(s) = 1 - cL/s^2 = 0$ at $s^* = \sqrt{cL}$, where $M = 2\sqrt{cL}$. for $L = 16$, $c = 3$: $s^* = \sqrt{48} \approx 6.9$; $M(6) = 6 + 8 = 14$, $M(7) = 7 + 6.9 = 13.9$, $M(8) = 8 + 6 = 14$, so anything from 6 to 8 is best, with about 14 activations against 48: a 3.4× saving. (measured peaks are higher because the gradients themselves need memory too; the lab finds about 2.6×.)

</details>

3. **the dot-product test.** show that $v^\top(Ju) = (J^\top v)^\top u$ for any $u \in \mathbb R^n$, $v \in \mathbb R^m$, and explain how you'd use it to test a reverse-mode engine against a forward-mode one without forming any Jacobian.

<details><summary>solution</summary>

both sides equal $\sum_{i,j}v_iJ_{ij}u_j$. so draw random $u$ and $v$, compute $Ju$ with one JVP (a `Dual` pass seeded with $u$) and $J^\top v$ with one VJP (`vjp(f, x, v)`), and compare the two scalars. a bug in either one makes them disagree for almost every random choice. it costs two passes, regardless of $n$ and $m$.

</details>

4. **why recomputation must be deterministic.** a segment contains dropout: $y = x \odot m$ with a random mask $m$, drawn fresh on every call. what gradient does a checkpointed backward compute, and what should it be?

<details><summary>solution</summary>

the forward pass used mask $m_1$, so the loss depends on $y = x \odot m_1$ and the correct gradient is $\bar x = \bar y \odot m_1$. the recomputation draws a new mask $m_2$ and returns $\bar y \odot m_2$: the gradient of a different function, wrong on every entry where the masks differ. the fix is to save the random generator's state when the segment first runs and restore it before recomputing, so $m_2 = m_1$ (PyTorch's checkpoint does this by default).

</details>

## experiment

`lab.py` uses your engine, and your forward-mode `jacobian` from chapter 03, for two experiments:

1. the full Jacobian of a two-layer function in forward and in reverse mode, for shapes from very wide ($n = 400$ inputs, $m = 1$ output) to very tall ($n = 1$, $m = 400$).
2. the memory/compute trade-off of `checkpoint_sequential` on a 16-layer network: peak memory, memory held after the forward pass, layer evaluations and time, for 1 to 16 segments and without checkpointing.

## check your understanding

1. what does `out.backward(v)` compute when `out` isn't a scalar?

<details><summary>answer</summary>

the VJP $J^\top v$ for every leaf: each leaf's `.grad` receives the gradient of the scalar $\sum_i v_i\,\text{out}_i$ with respect to that leaf. with $v = e_i$ that's row $i$ of the Jacobian.

</details>

2. why does reverse mode need memory proportional to the depth of the computation, while forward mode doesn't?

<details><summary>answer</summary>

forward mode carries tangents along with values and can forget each value as soon as the next one is computed. reverse mode runs backward after the forward pass is finished, and every VJP needs values from its forward operation, so they all have to be stored until backward reaches them.

</details>

3. why does checkpointing the entire network as one segment not reduce the peak memory?

<details><summary>answer</summary>

during backward, the single segment is recomputed with its whole graph, which holds as many intermediates as the plain forward pass did, plus the segment's input and output. the forward pass got cheaper, but the peak (during backward) didn't.

</details>

4. a checkpointed block's input is plain data (`requires_grad=False`). why must the checkpoint node still be part of the graph?

<details><summary>answer</summary>

the block's weights, closed over by `fn`, need gradients. they only get them when the node's VJP runs and backpropagates through the recomputed graph. if the node were a constant, backward would never visit it and the weights' gradients would silently stay `None`.

</details>

5. how much extra compute does $\sqrt L$ checkpointing cost, roughly?

<details><summary>answer</summary>

every layer is evaluated twice in the forward direction (once in the forward pass, once in the recomputation), so one extra forward pass. with forward + backward ≈ 3 forward passes, the total goes from 3 to 4: about 33% more time.

</details>

6. in `jacobian_reverse`, why can you reuse one forward graph for all $m$ backward passes?

<details><summary>answer</summary>

backward doesn't change the graph: it only reads the stored values and calls the VJPs. so the same graph can be swept with $e_1$, then $e_2$, and so on. you only have to reset the input's `.grad` in between, because leaves accumulate.

</details>

7. (the predict callout in idea 2.) how does `graph_nbytes` change with depth, batch size, and under `no_grad`?

<details><summary>answer</summary>

doubling the depth doubles it (each layer adds the same nodes). doubling the batch nearly doubles it (the activations are $B \times d$; only the weight-sized nodes like `W.T` stay the same). under `no_grad` it's 0: nothing is recorded.

</details>

## common bugs

- **forgetting to reset `.grad` between rows of the Jacobian.** symptom: row $i$ comes out as the sum of rows $1..i$.
- **counting leaves in `graph_nbytes`.** symptom: the "activation memory" of a network includes its weights and inputs, and doesn't go to zero under `no_grad`.
- **building the checkpoint node with `from_op`.** symptom: correct gradients when the input requires grad, but `None` gradients for the block's weights when it's plain data (the first block of every network).
- **running the checkpointed forward pass with grad on and keeping its output.** symptom: the gradients are right, but if the node holds on to the inner result (say, to reuse its data), it keeps the whole inner graph alive, and memory is what it was without checkpointing. run the forward pass under `no_grad` and keep only the data.
- **recomputing with the original input tensors.** symptom: gradients flow out of the recomputed graph into the *outer* graph's input nodes, and get added twice. recompute on fresh copies, and hand their gradients back through the VJP.

## going deeper

- Chen, Xu, Zhang & Guestrin (2016), "Training deep nets with sublinear memory cost". the $\sqrt L$ checkpointing schedule, applied to deep networks. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Griewank & Walther (2000), "Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation", *ACM TOMS* 26(1): 19–45. optimal recursive checkpointing. [doi:10.1145/347837.347846](https://doi.org/10.1145/347837.347846)
- Baydin, Pearlmutter, Radul & Siskind (2018), "Automatic differentiation in machine learning: a survey", *JMLR* 18(153): 1–43, §3 (forward and reverse mode, JVPs and VJPs). [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
- Gruslys, Munos, Danihelka, Lanctot & Graves (2016), "Memory-efficient backpropagation through time", *NeurIPS*. checkpointing for recurrent networks, with a dynamic-programming schedule. [arXiv:1606.03401](https://arxiv.org/abs/1606.03401)
- Korthikanti et al. (2022), "Reducing activation recomputation in large transformer models". how activation memory and selective recomputation shape large-model training. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)

## connections

- ← builds on: forward-mode `Dual` and `jacobian` (ch 03 · topic 04), your `Tensor` engine and its `no_grad` (topic 03), layers closed over by functions and modules (topic 04), Jacobians and the chain rule (ch 02 · topic 07).
- → used in: memory budgets and recomputation in large models (ch 26 · ML systems), backpropagation through time for recurrent networks (ch 17), the adjoint method for neural ODEs, which is reverse mode with recomputation in continuous time (ch 20), and every "out of memory" error you'll ever debug.
