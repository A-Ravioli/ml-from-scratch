# model parallelism & memory

## why this matters

a 7-billion-parameter model is 14 GB in 16-bit. that sounds like it fits on an 80 GB GPU, until you train it. the gradients are another 14 GB; AdamW's float32 master weights and two moments are 84 GB more. that's 112 GB before a single activation is stored, and activations for a long sequence at a useful batch size can be larger still. data parallelism (topic 04) doesn't help: every rank holds a full copy of everything.

this topic is about the memory side of scale. first, accounting: exactly how many bytes training needs, and where they go, predicted and then measured. then the four ways out. **ZeRO** shards the optimizer state (then the gradients, then the weights) across data-parallel ranks, so each holds $1/p$ of it. **activation checkpointing** stores a fraction of the activations and recomputes the rest during the backward pass. **tensor parallelism** splits each layer's matrices across devices. **pipeline parallelism** splits the layers. large training runs combine all four ("3D parallelism"); you'll build a small working version of each and learn what each one costs in communication and compute.

## warm-up: predict first

1. training a 1-billion-parameter model in mixed precision with AdamW: how many GB for weights, gradients and optimizer state together?
2. a GPT forward pass at batch 64 and context 128: which is larger, the model's 7 MB of parameters or the activations saved for the backward pass?
3. activation checkpointing recomputes the forward pass during backward. how much slower does a training step get?
4. a pipeline of 4 GPUs processes a batch as 4 micro-batches. what fraction of the time is each GPU idle?

## prerequisites check

- [ ] you know what autograd saves during the forward pass and why (→ ch 14 · 03)
- [ ] you know AdamW's state (two moments per parameter) and mixed precision's master weights (→ ch 15 · 04, topic 03)
- [ ] you can explain all-reduce, reduce-scatter and all-gather, and their cost (→ topic 04)
- [ ] you know a transformer block's MLP as `fc2(gelu(fc1(x)))` (→ ch 18 · 03)

## core ideas

### idea 1: where training memory goes

per parameter, mixed-precision training with AdamW keeps:

| what | bytes |
|---|---|
| 16-bit weights (used in forward and backward) | 2 |
| 16-bit gradients | 2 |
| float32 master weights | 4 |
| float32 first moment $m$ | 4 |
| float32 second moment $v$ | 4 |

16 bytes per parameter: Rajbhandari et al. (2020) call this $2\Psi + 2\Psi + K\Psi$ with $K = 12$ for the "optimizer states". in plain float32 training there's no separate master copy (the weights are already float32), so it's $4 + 4 + 8 = 16$ too; SGD with momentum keeps one float32 buffer instead of two, plain SGD none.

**ZeRO** (the Zero Redundancy Optimizer) notices that in data parallelism all $p$ ranks hold identical copies of all of this, and shards it instead:

- **stage 1** partitions the optimizer state: rank $r$ keeps master weights and moments only for its $1/p$ slice of the (flattened) parameters, and updates only that slice. after the update the ranks all-gather the new weights. per rank: $2 + 2 + 12/p$ bytes per parameter.
- **stage 2** also partitions gradients: each rank needs only the gradient of its own slice, so the all-reduce becomes a reduce-scatter. $2 + (2 + 12)/p$.
- **stage 3** also partitions the 16-bit weights, all-gathering each layer's weights just before it's used, in the forward and again in the backward pass. $16/p$. PyTorch's FSDP (fully sharded data parallel) implements this.

stages 1 and 2 cost no more communication than plain data parallelism (a reduce-scatter plus an all-gather *is* an all-reduce, topic 04); stage 3 adds one more all-gather of the weights per pass, about 1.5× the traffic. with $p = 64$, stage 3 trains a 7.5B model in 1.9 GB per rank instead of 120 GB.

> **spec.** `training_memory(n_params, *, bytes_per_param=2, optimizer='adamw', zero_stage=0, world_size=1)` returns a dict `{'params', 'grads', 'optimizer', 'total'}` of bytes per rank. params and grads take `bytes_per_param` each; the optimizer state takes 8 (adamw), 4 (momentum) or 0 (sgd) float32 bytes per parameter, plus a float32 master copy (4) when `bytes_per_param < 4`. stage $s \ge 1$ shards the optimizer state, $s \ge 2$ also the gradients, $s = 3$ also the parameters: a sharded part counts $\lceil N / p \rceil$ parameters. `ValueError` for an unknown optimizer or a stage outside 0–3.

> **▶ build: stage 1.** implement `training_memory`.

### idea 2: activations, and trading them for compute

the backward pass needs values from the forward pass: a linear layer's input (to compute the weight gradient), a GELU's input, a softmax's output, a layernorm's normalized input and statistics. autograd saves them as the forward runs and frees them as the backward consumes them, so peak memory comes right at the end of the forward pass. per transformer block, for batch $B$ and context $T$, this is a few dozen numbers per token per channel, $c_1 BTd$, plus terms for the attention matrices, $c_2 BHT^2$, if the attention is computed naively. it scales with the batch and (quadratically) with the context; parameter memory doesn't. at the batch sizes that keep a GPU busy, activations usually dominate.

you can *measure* what autograd saves instead of estimating it. `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` calls `pack(t)` for every tensor saved for backward inside its context. counting the bytes of each distinct storage (two saves of the same tensor, or of a view of it, share memory) gives exactly the activation memory, and skipping the parameters' storages separates activations from weights.

**activation checkpointing** (Chen et al., 2016) saves memory by not saving: run a segment of the network (say, a transformer block) without recording its graph, keep only the segment's *input*, and when the backward pass reaches the segment, rerun its forward pass with recording on and backpropagate through that. the memory of the segment's internals is needed only briefly, one segment at a time; the price is one extra forward pass per segment, about a third more compute for a training step (forward is ~1/3 of the forward-plus-backward cost). checkpointing every one of $L$ blocks cuts activation memory from $O(L)$ blocks' worth to the block inputs plus one block's internals.

in PyTorch you build it as a `torch.autograd.Function`: its `forward` runs the segment under `torch.no_grad()` and saves the inputs; its `backward` recomputes the segment under `torch.enable_grad()` on detached copies of the inputs and calls `torch.autograd.backward` on the recomputed output with the incoming gradient. one trap: random operations. if the segment contains dropout, the recomputation must draw *the same* random numbers as the original forward, or the gradients are for a different network. save the RNG state in `forward` and restore it, temporarily, for the recomputation.

> **spec.** `saved_tensor_bytes(fn, *args, exclude=())` returns `(fn(*args), bytes)`, where `bytes` is the total size of the distinct storages (by `untyped_storage().data_ptr()`, sized by `untyped_storage().nbytes()`) saved for backward during the call, excluding storages shared with any tensor in `exclude`. `checkpoint(fn, *args)` returns `fn(*args)` (a single tensor; at least one input requires grad) such that only the inputs are saved for backward; during backward it calls `fn` once more, with the CPU RNG state the forward saw (restored inside `torch.random.fork_rng`, so the global random stream ends where an ordinary forward and backward would leave it), and produces the same gradients for the inputs and for the parameters `fn` uses as an ordinary call.

> **🔮 predict:** checkpointing every block of chapter 18's model M: by what factor does the saved-activation memory fall, and by what factor does the step time grow?

> **▶ build: stage 2.** implement `saved_tensor_bytes` and `checkpoint`.

### idea 3: tensor parallelism

when one layer's weights or activations are too big for a device, split the layer. Megatron-LM's insight (Shoeybi et al., 2019) is that a transformer MLP can be split across $p$ devices with a single all-reduce. write the MLP as $Y = \text{gelu}(XW_1^\top + b_1)W_2^\top + b_2$ with hidden width $h$.

- split $W_1$ **by output (column of the product)**: device $r$ gets the rows $W_1^{(r)}$ for hidden units $[rh/p, (r+1)h/p)$ and the matching slice of $b_1$, and computes $H^{(r)} = \text{gelu}(XW_1^{(r)\top} + b_1^{(r)})$, its slice of the hidden layer. GELU is elementwise, so no communication is needed: each device's slice of the hidden layer is complete.
- split $W_2$ **by input (row of the product)**: device $r$ gets the columns $W_2^{(r)}$ that read those same hidden units, and computes a partial output $Y^{(r)} = H^{(r)}W_2^{(r)\top}$. the true output is the sum $Y = \sum_r Y^{(r)} + b_2$: one **all-reduce** of a $B \times T \times d$ tensor (and $b_2$ added once, after it).

had you split $W_1$ the other way, you'd need to sum before the GELU, which isn't linear, adding a second all-reduce. attention splits the same way, by heads: each device computes a subset of heads (column-parallel $q, k, v$ projections, row-parallel output projection). so a transformer block costs two all-reduces of the activations in the forward pass and two in the backward. that's a lot of traffic per layer, which is why tensor parallelism is used within a node, where GPUs are connected by very fast links, and data or pipeline parallelism between nodes.

> **spec.** `split_mlp(fc1, fc2, world_size)`: for `nn.Linear` layers `fc1` ($d \to h$) and `fc2` ($h \to d$), returns a list of `world_size` tuples `(W1_r, b1_r, W2_r)`: cloned slices `fc1.weight[rk:(r+1)k]` $(k, d)$, `fc1.bias[rk:(r+1)k]` $(k,)$ and `fc2.weight[:, rk:(r+1)k]` $(d, k)$ with $k = h/p$; `ValueError` if $p$ doesn't divide $h$. `tp_mlp_forward(x, shards, b2)` returns `(out, partials)`: `partials[r]` $= \text{gelu}(x W_{1,r}^\top + b_{1,r}) W_{2,r}^\top$ (without $b_2$), and `out` = their sum plus `b2`, the simulated all-reduce.

> **▶ build: stage 3.** implement `split_mlp` and `tp_mlp_forward`.

### idea 4: pipeline parallelism

split the *layers* instead: device $s$ holds a contiguous **stage** of the network and passes its output activations to device $s + 1$. communication is only activations at stage boundaries, which is cheap. the problem is idleness: with one batch, device 1 waits while device 0 computes, device 0 waits while the rest compute, and in the backward pass the same happens in reverse.

**GPipe** (Huang et al., 2019) splits each batch into $m$ **micro-batches** and pipelines them. in the forward phase, micro-batch $j$ runs on stage $s$ at tick $s + j$: a diagonal wavefront. when every forward is done, the backward phase runs the same wavefront in reverse, starting from the last stage: stage $s$ processes micro-batch $j$'s backward at tick $(m + p - 1) + (p - 1 - s) + j$. the whole schedule takes $2(m + p - 1)$ ticks (counting a backward as one tick, a simplification), in which each stage is busy for $2m$. the idle fraction, the **pipeline bubble**, is

$$\frac{p - 1}{m + p - 1},$$

so you want $m \gg p$. but GPipe holds the activations of all $m$ micro-batches until their backward passes, so memory grows with $m$; the **1F1B** schedule (PipeDream, Narayanan et al., 2019) interleaves one forward with one backward once the pipeline is full, keeping the same bubble with at most $p$ micro-batches in flight.

> **spec.** `gpipe_schedule(n_stages, n_micro)` returns a grid of $2(m + p - 1)$ rows, one per tick, each a list of `n_stages` entries: `('F', j)` or `('B', j)` if stage $s$ processes micro-batch $j$'s forward or backward at that tick, else `None`, following the tick formulas above. `bubble_fraction(n_stages, n_micro)` $= (p - 1)/(m + p - 1)$.

> **🔮 predict:** with 4 stages, how many micro-batches does it take to bring the bubble below 10%? and what happens to the activation memory of each stage as you add them?

> **▶ build: stage 4.** implement `gpipe_schedule` and `bubble_fraction`.

### idea 5: ZeRO-1, built

now the sharded optimizer itself, with real processes. each of $p$ data-parallel ranks holds the full model (16-bit weights and gradients in a real system; float32 here, for simplicity) but AdamW state for only its shard. flatten all parameters into one vector of length $N$, pad it to $pS$ with $S = \lceil N/p \rceil$, and give rank $r$ the slice $[rS, (r + 1)S)$. a step:

1. flatten the gradients and **all-reduce** them (sum, then divide by $p$): every rank now has the averaged gradient. (ZeRO-2 would reduce-scatter instead, since each rank needs only its slice.)
2. update **only your slice** of the flattened weights with AdamW, using your slice's $m$ and $v$: decoupled weight decay $w \leftarrow w(1 - \eta\lambda)$, then $m \leftarrow \beta_1 m + (1 - \beta_1)g$, $v \leftarrow \beta_2 v + (1 - \beta_2)g^2$, $w \leftarrow w - \eta\, \hat m / (\sqrt{\hat v} + \epsilon)$ with the bias corrections $\hat m = m/(1 - \beta_1^t)$, $\hat v = v/(1 - \beta_2^t)$.
3. **all-gather** every rank's updated slice into the full weight vector, and copy it back into the parameters.

since every element is updated exactly once, by exactly one rank, with the same averaged gradient and the same AdamW arithmetic, the result is the same as one AdamW on the whole model. but each rank stores $2S$ moments instead of $2N$.

> **spec.** `ZeroAdamW(params, *, rank, world_size, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)`: attributes `shard = (start, end)`, rank's slice of the flattened parameters clipped to $[0, N]$, and float32 `exp_avg`, `exp_avg_sq` of length $S = \lceil N/p \rceil$. `step()`: steps 1–3 above (with `world_size == 1` no communication, and no process group needed); weight decay applies to every parameter, as `torch.optim.AdamW` does with a single parameter group; a parameter whose `.grad` is `None` is skipped entirely, as AdamW skips it (no decay, no moment update, and its step count for the bias correction doesn't advance, so keep step counts per element); matches `torch.optim.AdamW` on the averaged gradient. `zero_grad()` sets every `.grad` to `None`.

> **▶ build: stage 5.** implement `ZeroAdamW`. its two-process check runs from a terminal: `pytest -k stage5`.

## worked example

**a 7.5B model** (the ZeRO paper's example) in mixed precision with AdamW: $16 \times 7.5 \times 10^9 = 120$ GB per rank without sharding. over 64 ranks: stage 1, $2 + 2 + 12/64 = 4.19$ bytes per parameter, 31.4 GB; stage 2, $2 + 14/64 = 2.22$, 16.6 GB; stage 3, $16/64 = 0.25$, 1.9 GB.

**chapter 18's model M** (1.8M parameters, float32 AdamW): $16 \times 1.8\text{M} = 29$ MB for weights, gradients and moments. its saved activations at batch 64, context 128, measured in the lab with your `saved_tensor_bytes`, are hundreds of megabytes: the naive attention alone saves the $64 \times 6 \times 128 \times 128$ attention weights (25 MB in float32) and more per layer. for small models at real batch sizes, activations are the memory problem.

**the bubble.** 4 stages, 4 micro-batches: $3/7 = 43\%$ idle. 16 micro-batches: $3/19 = 16\%$. your `training_memory` and `bubble_fraction` must reproduce these.

## on paper

1. **ZeRO's communication.** show that ZeRO stages 1 and 2 move the same number of bytes per rank as a plain data-parallel all-reduce, and that stage 3 moves about 1.5 times as many.

<details><summary>solution</summary>

a ring all-reduce of the gradients is a reduce-scatter ($\frac{p-1}{p}\Psi$ elements sent per rank) plus an all-gather (the same): $2\frac{p-1}{p}\Psi$. stage 2 does the reduce-scatter of gradients, updates its slice, and all-gathers the updated weights: the same two halves. stage 1 all-reduces the gradients and all-gathers the weights, which counts 3 halves as written here, but a real implementation reduce-scatters as in stage 2. stage 3 must also all-gather the weights in the forward pass (they're sharded) and again in the backward: reduce-scatter + 2 all-gathers, $3\frac{p-1}{p}\Psi$, 1.5×.

</details>

2. **the bubble.** derive the GPipe bubble fraction from the schedule: $p$ stages, $m$ micro-batches, forward and backward each one tick per micro-batch per stage.

<details><summary>solution</summary>

the forward wavefront ends when the last stage finishes the last micro-batch, at tick $(p - 1) + (m - 1)$, so it takes $m + p - 1$ ticks; the backward phase the same. total $2(m + p - 1)$ ticks, during which each stage does $2m$ units of work. idle fraction $1 - 2m/(2(m + p - 1)) = (p-1)/(m + p - 1)$.

</details>

3. **checkpointing's cost.** with checkpointing at every block, a step computes one forward, one recomputed forward and one backward. if the backward costs twice a forward, what's the overhead? with checkpoints every $k$ blocks in an $L$-block network, how does memory scale?

<details><summary>solution</summary>

$(1 + 1 + 2)/(1 + 2) = 4/3$: 33% more compute. with checkpoints every $k$ blocks you store $L/k$ segment inputs plus the internals of one $k$-block segment while recomputing it: $O(L/k + k)$ blocks' worth, minimized at $k = \sqrt{L}$, giving $O(\sqrt{L})$ memory for one extra forward (Chen et al., 2016).

</details>

## experiment

the lab predicts the memory of training chapter 18's model M with your `training_memory` and checks it against the bytes PyTorch actually allocates for weights, gradients and AdamW state, then measures the saved activations with your `saved_tensor_bytes` as batch and context grow. it checkpoints every block with your `checkpoint` and measures what that saves and what it costs. finally it runs your `ZeroAdamW` on 1, 2 and 4 gloo processes and compares the per-rank optimizer state and the final weights.

## check your understanding

1. why doesn't data parallelism reduce per-device memory, and which ZeRO stage gets closest to dividing everything by $p$?

<details><summary>answer</summary>

every rank holds a full replica of weights, gradients and optimizer state; only the batch is split. stage 3 shards all three, so each rank holds $1/p$ of everything plus whatever layer it's currently all-gathering.

</details>

2. why does mixed precision not reduce the memory of weights plus optimizer state?

<details><summary>answer</summary>

the float32 master copy and moments stay; the 16-bit weights and gradients are *added* (2 + 2 bytes) while removing the float32 gradients (4). the total stays 16 bytes per parameter. the savings are in activations and in bandwidth.

</details>

3. a checkpointed segment contains dropout and you forget to restore the RNG state. what happens?

<details><summary>answer</summary>

the recomputation draws different dropout masks, so the backward pass computes the gradient of a different network than the one whose loss you measured. training still runs and the loss may even fall, but the gradients are wrong; the check comparing with an un-checkpointed run catches it.

</details>

4. why is the MLP split column-parallel then row-parallel, not the other way round?

<details><summary>answer</summary>

column-parallel $W_1$ gives each device complete hidden units, so the elementwise GELU runs locally; row-parallel $W_2$ then produces partial sums that one all-reduce combines. the reverse order would produce partial sums before the GELU, which isn't linear, forcing an all-reduce in the middle too.

</details>

5. how do you shrink the pipeline bubble, and what does that cost?

<details><summary>answer</summary>

more micro-batches per batch ($m \gg p$). GPipe then holds the activations of all $m$ micro-batches until their backward passes, so memory grows with $m$; 1F1B schedules cap it at $p$ micro-batches in flight. smaller micro-batches also run less efficiently on each device.

</details>

6. why do large runs use tensor parallelism inside a node and pipeline or data parallelism across nodes?

<details><summary>answer</summary>

tensor parallelism all-reduces activations twice per block in each pass, a lot of traffic that needs the fastest links (NVLink within a node). pipeline parallelism sends only boundary activations, and data parallelism one gradient all-reduce per step that can overlap with compute: both tolerate the slower network between nodes.

</details>

## common bugs

- **counting saved tensors instead of storages.** a tensor saved twice (`x * x`), or a weight saved as a transposed view, would be counted twice.
- **checkpointing a segment whose inputs don't require grad.** the autograd function's output then doesn't require grad, and the parameters inside get no gradients. at least one input must require grad.
- **forgetting `b2` or adding it on every rank.** the row-parallel output is a sum of partials; the bias belongs after the all-reduce, once.
- **updating the whole parameter vector on every rank in ZeRO.** it works, and saves nothing: each rank must keep moments only for its slice.
- **ranks disagreeing about the shard boundaries.** every rank must flatten parameters in the same order and pad the same way, or the all-gather assembles garbage.
- **a pipeline schedule that runs a backward before its forward reached the last stage.** check the dependencies, not just the counts.

## going deeper

- Rajbhandari, Rasley, Ruwase & He (2020), "ZeRO: memory optimizations toward training trillion parameter models", *SC20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Chen, Xu, Zhang & Guestrin (2016), "Training deep nets with sublinear memory cost". [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Shoeybi, Patwary, Puri, LeGresley, Casper & Catanzaro (2019), "Megatron-LM: training multi-billion parameter language models using model parallelism". [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Huang et al. (2019), "GPipe: efficient training of giant neural networks using pipeline parallelism", *NeurIPS*. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Narayanan et al. (2021), "Efficient large-scale language model training on GPU clusters using Megatron-LM", *SC21*: 3D parallelism and the interleaved 1F1B schedule. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Korthikanti et al. (2022), "Reducing activation recomputation in large transformer models": activation memory per block, derived term by term. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
- Zhao et al. (2023), "PyTorch FSDP: experiences on scaling fully sharded data parallel", *VLDB*. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)

## connections

- ← builds on: autograd's saved values (ch 14), AdamW and mixed precision (ch 15, topic 03), collectives (topic 04).
- → used in: chapter 27 sizes its Llama and its fine-tuning with `training_memory` (LoRA is another way to shrink optimizer state); the capstone's fused attention is also an activation-memory saving (no saved $T \times T$ weights).
