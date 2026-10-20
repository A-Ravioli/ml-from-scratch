# data parallelism

## why this matters

GPT-3's training took about $3 \times 10^{23}$ FLOPs. one GPU running flat out at $10^{14}$ FLOP/s would need about a century. the only way to train such a model is to use thousands of processors at once, and the simplest way to divide the work is **data parallelism**: every processor (a *rank*) holds a full copy of the model, processes a different slice of each batch, and the ranks average their gradients before every update, so all copies take the same step and stay identical.

the arithmetic splits perfectly; the catch is communication. every step, every rank must send and receive a gradient as large as the model. done naively, by sending everything to one machine, that machine's network link becomes the bottleneck and adding ranks makes things *slower*. done with a **ring all-reduce**, each rank sends about twice the model's size per step no matter how many ranks there are. this topic builds the ring (simulated in NumPy), the cost model that predicts when communication dominates, gradient accumulation and data sharding, and finally real data-parallel training across processes with `torch.distributed`, the same API PyTorch's `DistributedDataParallel` uses on GPU clusters.

## warm-up: predict first

1. 8 ranks each hold a 1 GB gradient. one rank collects all of them, sums them and sends the sum back. how many bytes cross that rank's network link? what's the minimum any scheme could get away with per rank?
2. data-parallel training on 4 ranks with a local batch of 32: is it equivalent to single-process training with batch 32 or batch 128?
3. you want a batch of 512 but only 64 examples fit in memory. can you get *exactly* the gradient of the 512-example batch?
4. 4 processes on one 8-core laptop, each training on a quarter of the data. how much faster than one process?

## prerequisites check

- [ ] you can write a training loop with `zero_grad`, `backward`, `step` and know what `.grad` accumulates (→ ch 15 · 07)
- [ ] you know why the gradient of a mean loss over a batch is the mean of per-example gradients (→ ch 06)
- [ ] you know bandwidth, latency and the roofline's "bytes over bandwidth" (→ topic 01)
- [ ] you can reshuffle data per epoch with a seeded generator (→ ch 00 · 05)

## core ideas

### idea 1: the ring all-reduce

an **all-reduce** takes one array per rank and leaves every rank holding their elementwise sum. the obvious implementation is a parameter server: everyone sends to rank 0, rank 0 sums and sends the result back. rank 0 receives $(p - 1)n$ bytes and sends as many: its link carries traffic that grows with $p$, and everyone else waits for it.

the **ring all-reduce** arranges the ranks in a circle, each sending only to its right neighbour, and splits each array into $p$ chunks. it runs in two phases of $p - 1$ steps each.

**reduce-scatter.** at step $s = 0, \dots, p-2$, rank $i$ sends its current copy of chunk $(i - s) \bmod p$ to rank $i + 1$, which adds it to its own copy of that chunk. follow chunk $c$: it starts at rank $c$, goes to $c + 1$ (which adds its piece), then $c + 2$, and so on; after $p - 1$ steps it has collected every rank's contribution and sits, complete, at rank $c - 1$. so after the phase, rank $i$ holds the full sum of chunk $(i + 1) \bmod p$, and every rank holds exactly one finished chunk.

**all-gather.** at step $s$, rank $i$ sends its *finished* chunk $(i + 1 - s) \bmod p$ to rank $i + 1$, which keeps it (no addition). after $p - 1$ steps every finished chunk has gone round the ring and every rank has the whole sum.

all ranks send at the same time in every step, each message is $n/p$ bytes, and there are $2(p - 1)$ steps, so each rank sends

$$2 \frac{p-1}{p}\, n \text{ bytes},$$

less than $2n$ however large $p$ gets. that's optimal: every rank must receive the other ranks' contribution to every element it doesn't reduce itself. (Patarasuk & Yuan, 2009, proved the bound; NVIDIA's NCCL and the gloo library implement rings and trees of rings.)

one detail of the simulation matters for the checks: the order of additions. chunk $c$ is summed as $x_c + x_{c+1} + \dots + x_{c-1}$, left to right, around the ring. floating-point addition isn't associative, so a ring's result differs in its last bits from a direct `sum(arrays)`, and differently for each chunk. the checks use that to tell a ring from a shortcut.

> **spec.** `ring_allreduce(arrays)`: `arrays` is a list of $p$ equal-shape arrays (one per rank). split each flattened array into $p$ chunks with `np.array_split`; run the reduce-scatter and all-gather steps above, each rank holding its own copies of the chunks; return $p$ separate arrays of the input shape, all equal to the sum. inputs are not modified.

> **🔮 predict:** with $p = 3$ ranks holding $[1, 2, 3]$, $[10, 20, 30]$ and $[100, 200, 300]$ (one element per chunk), which rank holds which finished chunk after the reduce-scatter?

> **▶ build: stage 1.** implement `ring_allreduce`.

### idea 2: what communication costs

a message of $m$ bytes over a link costs a fixed **latency** $\alpha$ (software, network hops) plus $m / \beta$ at **bandwidth** $\beta$. the ring sends $2(p-1)$ messages of $n/p$ bytes in sequence, so

$$T_\text{ring} = 2(p-1)\left(\alpha + \frac{n}{p \beta}\right).$$

for large $n$ the second term wins and the time is nearly independent of $p$: $\approx 2n/\beta$. for small $n$ the first term wins and grows linearly with $p$. that's why frameworks never all-reduce each parameter tensor separately: a GPT has a few dozen tensors, many of them tiny (biases, layer-norm weights), and paying $2(p-1)\alpha$ for each adds up. instead they flatten gradients into **buckets** of tens of megabytes and reduce each bucket with one call.

the other big trick is **overlap**. the backward pass produces gradients from the last layer to the first, and a bucket can be all-reduced as soon as all its gradients exist, while the backward pass continues on earlier layers. PyTorch's `DistributedDataParallel` does exactly this (Li et al., 2020). if communication takes $t_\text{comm}$ and a fraction $o$ of the compute time $t_\text{comp}$ is available to hide it under, a step takes $t_\text{comp} + \max(0, t_\text{comm} - o\, t_\text{comp})$, and the **scaling efficiency**, the fraction of ideal speedup you get, is

$$E = \frac{t_\text{comp}}{t_\text{comp} + \max(0,\; t_\text{comm} - o\, t_\text{comp})}.$$

data parallelism scales well when compute per step is large compared with gradient size: big models (more FLOPs per parameter-byte, via larger batches per rank) and fast links. a 1-GB gradient over a 10 GB/s link takes about 0.2 s per step whatever the number of ranks; whether that's negligible depends on whether a step computes for 2 s or 20 ms.

> **spec.** `allreduce_bytes(n_bytes, world_size)` = $2(p-1)n/p$. `allreduce_time(n_bytes, world_size, *, bandwidth, latency=0.0)` = $2(p-1)(\alpha + n/(p\beta))$. `ddp_efficiency(t_compute, t_comm, *, overlap=0.0)` = $E$ above.

> **▶ build: stage 2.** implement `allreduce_bytes`, `allreduce_time` and `ddp_efficiency`.

### idea 3: gradient accumulation

data parallelism over $p$ ranks with local batch $b$ is equivalent to one process with batch $pb$: the gradient of the mean loss over $pb$ examples is the average of the $p$ local mean-loss gradients (with equal local batches). so scaling out changes the effective batch size, and hyperparameters tuned for one size (the learning rate, above all) may need adjusting. the converse also holds: you can get a large batch's gradient on *one* device by splitting it into **micro-batches**, running forward and backward on each, and letting `.grad` accumulate, since gradients add.

the subtlety is the weighting. with micro-batch losses that are means, the full batch's mean loss is

$$L = \frac{1}{N}\sum_{n=1}^{N} \ell_n = \sum_k \frac{N_k}{N}\, \bar\ell_k,$$

so each micro-batch's mean loss $\bar\ell_k$ must be multiplied by its share $N_k / N$ before `backward()`. dividing every micro-batch by the number of micro-batches is the same thing only when they're all the same size, which the last micro-batch of an epoch often isn't. accumulation trades time for memory: the same FLOPs, done sequentially. combined with DDP it also saves communication: only the last micro-batch's backward needs an all-reduce (DDP's `no_sync()` context skips it for the others).

> **spec.** `accumulate_gradients(model, loss_fn, micro_batches)`: sets every parameter's gradient to `None` first, then for each `(x, y)` in `micro_batches` backpropagates `loss_fn(model(x), y) * len(x) / N` ($N$ = total examples), leaving in `.grad` the gradient of the mean loss over all examples. returns that mean loss as a float. doesn't step.

> **▶ build: stage 3.** implement `accumulate_gradients`.

### idea 4: giving each rank its own data

each rank must see different examples, or the ranks compute the same gradient $p$ times. the standard recipe (PyTorch's `DistributedSampler`): every rank draws *the same* random permutation of the dataset, seeded identically, then rank $r$ takes elements $r, r + p, r + 2p, \dots$. the shards are disjoint and together cover everything, with no communication needed to agree on them. reseeding with `seed + epoch` reshuffles every epoch, the same way on every rank.

if $n$ isn't divisible by $p$, some ranks would get one example fewer, and one fewer batch. that's a deadlock, not a rounding issue: in the last step the other ranks call all-reduce and wait forever for a rank that has finished. so the permutation is **padded** to a multiple of $p$ by repeating it cyclically from the start, and a handful of examples are seen twice per epoch.

> **spec.** `shard_indices(n, rank, world_size, *, epoch=0, seed=0)`: `perm = np.random.default_rng(seed + epoch).permutation(n)`, repeated cyclically (`perm[0], perm[1], ..., perm[n-1], perm[0], ...`) to the smallest length divisible by `world_size`, as PyTorch's `DistributedSampler` does (this still works when the padding is longer than the dataset, e.g. 1 example over 3 ranks), then `perm[rank::world_size]`.

> **▶ build: stage 4.** implement `shard_indices`.

### idea 5: real processes with `torch.distributed`

`torch.distributed` runs collectives between processes. each process calls `dist.init_process_group(backend, init_method=..., rank=r, world_size=p)` to join a **process group**: NCCL is the backend for NVIDIA GPUs; **gloo** works on CPUs anywhere, and is what the checks and lab use (each rank is an operating-system process on your machine, and "the network" is shared memory, but the program is exactly what would run on a cluster). then:

- `dist.all_reduce(t, op=dist.ReduceOp.SUM)` replaces tensor `t`, on every rank, with the sum of everyone's `t`, in place;
- `dist.broadcast(t, src=0)` overwrites `t` on every rank with rank 0's.

every collective is a rendezvous: all ranks must call the same collectives in the same order with tensors of the same shapes, or they hang. data-parallel training is then a short list of rules: broadcast rank 0's initial weights so all replicas start identical; each step, compute the local loss on the local batch, backpropagate, all-reduce the gradients and divide by $p$, and step the optimizer. identical starting weights plus identical averaged gradients plus a deterministic optimizer give identical weights on every rank forever, with no further synchronization.

> **spec.** `average_gradients(model, world_size)`: concatenates every non-`None` gradient into one flat tensor, calls `dist.all_reduce` on it exactly once (sum), divides by `world_size`, and copies the result back into the `.grad`s. `ddp_train(model, X, y, loss_fn, *, rank, world_size, epochs=1, batch_size=8, lr=0.1, seed=0)`: in an initialized process group, broadcasts every parameter from rank 0, creates `torch.optim.SGD(model.parameters(), lr=lr)`, and for each epoch takes `shard_indices(len(X), rank, world_size, epoch=epoch, seed=seed)` and steps through it in consecutive batches of `batch_size`: zero grads, `loss_fn(model(X[b]), y[b])`, backward, `average_gradients`, step. returns the list of local loss values.

> **🔮 predict:** after training, the checks compare the weights of both ranks with `torch.equal`, which demands exact bitwise equality. will they be exactly equal, or only close?

> **▶ build: stage 5.** implement `average_gradients` and `ddp_train`. the checks for this stage start two processes that load your notebook from disk, so they run only from a terminal: `pytest -k stage5`.

## worked example

**a ring with three ranks.** ranks hold $[1, 2, 3]$, $[10, 20, 30]$, $[100, 200, 300]$; chunk $c$ is element $c$.

| | rank 0 | rank 1 | rank 2 |
|---|---|---|---|
| start | 1, 2, 3 | 10, 20, 30 | 100, 200, 300 |
| reduce-scatter step 0 (rank $i$ sends chunk $i$) | 1, 2, **303** | **11**, 20, 30 | 100, **220**, 300 |
| reduce-scatter step 1 (rank $i$ sends chunk $i - 1$) | 1, **222**, 303 | 11, 20, **333** | **111**, 220, 300 |
| after all-gather (2 more steps) | 111, 222, 333 | 111, 222, 333 | 111, 222, 333 |

after the reduce-scatter rank 0 holds the finished chunk 1 (222), rank 1 chunk 2 (333), rank 2 chunk 0 (111): rank $i$ holds chunk $i + 1$. each rank sent 4 messages of one element: $2 \cdot 2/3 \cdot 3 = 4$ elements, as `allreduce_bytes` says.

**cost of syncing a GPT.** a model with 2 million float32 parameters has an 8 MB gradient. over 4 ranks with links of 10 GB/s and 10 µs latency: each rank sends $2 \cdot \tfrac34 \cdot 8\,\text{MB} = 12$ MB, in $6\,(10\,\mu\text{s} + 2\,\text{MB}/10\,\text{GB/s}) = 6 \times 210\,\mu\text{s} = 1.26$ ms. a parameter server would push $3 \times 8 = 24$ MB *into* rank 0 alone. your `ring_allreduce` and `allreduce_time` must reproduce both.

## on paper

1. **the bandwidth lower bound.** each element of the result starts as $p$ separate values, one per rank, and a message can carry one partial sum per element. show that combining them needs at least $(p - 1)n$ bytes of messages in total, and that, if the work is spread evenly, every rank must then also receive the $(p - 1)n/p$ finished bytes it didn't reduce itself. conclude that no all-reduce can have each rank send fewer than about $2(p-1)n/p$ bytes, so the ring is optimal in bandwidth.

<details><summary>solution</summary>

every message merges two partial sums of an element into one; going from $p$ pieces to one takes at least $p - 1$ merges, so at least $p - 1$ transmitted copies of every element: $(p - 1)n$ bytes over all ranks, $(p - 1)n/p$ per rank if spread evenly. after that each element's sum exists at one rank only, and every other rank must receive it: a rank that reduced $n/p$ of the elements must receive the other $(p - 1)n/p$. total $2(p - 1)n/p$ per rank, exactly what the ring sends. (Patarasuk & Yuan, 2009, give the rigorous version.)

</details>

2. **latency versus bandwidth.** with $\alpha = 10\,\mu$s and $\beta = 10$ GB/s, for what message size $n$ does a ring all-reduce over $p = 64$ ranks spend as much time on latency as on bandwidth? what does that say about reducing a 4 KB layer-norm gradient on its own?

<details><summary>solution</summary>

equal when $\alpha = n/(p\beta)$, $n = p \alpha \beta = 64 \times 10^{-5} \times 10^{10} = 6.4$ MB. a 4 KB tensor is 1,600× smaller: its all-reduce costs $126 \times 10\,\mu\text{s} = 1.26$ ms, almost all latency, for 4 KB. bucket small tensors together.

</details>

3. **DDP equals a big batch.** show that averaging the gradients of $p$ local mean losses over equal-size local batches gives the gradient of the mean loss over the union of the batches. what changes if the local batches have different sizes?

<details><summary>solution</summary>

$\frac{1}{p}\sum_r \nabla \frac{1}{b}\sum_{n \in B_r} \ell_n = \nabla \frac{1}{pb}\sum_{n \in \cup B_r} \ell_n$ by linearity. with sizes $b_r$, the plain average weights each rank equally rather than each example; the correct weights are $b_r / \sum b_r$, which is why samplers pad shards to equal size (and why the last partial batch of an epoch needs care).

</details>

## experiment

the lab times real all-reduces between 2 and 4 gloo processes on your machine for message sizes from kilobytes to tens of megabytes, fits the latency and bandwidth of your `allreduce_time` model to them, and compares one all-reduce of a big bucket with many small ones. then it runs your `ddp_train` with 1, 2 and 4 processes and measures the scaling efficiency, first with one thread per process and then with the machine's cores split between them.

## check your understanding

1. why does each rank of a ring all-reduce send about $2n$ bytes regardless of $p$, while a parameter server's root handles $2(p-1)n$?

<details><summary>answer</summary>

the ring splits the data into $p$ chunks and pipelines them, so every link carries $1/p$ of the data per step for $2(p-1)$ steps: $2(p-1)/p \cdot n$. the parameter server sends everything through one node, whose traffic grows with the number of ranks.

</details>

2. why must every rank start from the same weights, and how does `ddp_train` ensure it?

<details><summary>answer</summary>

the ranks apply the same averaged gradient, so they stay identical only if they start identical. `ddp_train` broadcasts rank 0's parameters before training; without it, each rank's random initialization would persist and the replicas would be different models receiving the same updates.

</details>

3. you double the number of ranks without changing the local batch size. what happens to the effective batch size, and what might you need to change?

<details><summary>answer</summary>

it doubles. the gradient noise halves; the learning rate (and warmup) tuned for the old batch may now be too small or need the linear-scaling rule (Goyal et al., 2017), and beyond the critical batch size more data per step stops buying faster progress.

</details>

4. why do the shards need padding, and what goes wrong without it?

<details><summary>answer</summary>

unequal shards give some ranks an extra batch; in that last step those ranks call all-reduce while the others have stopped, and the collective waits forever: a hang, not an error. padding makes every rank do the same number of steps.

</details>

5. why are your two ranks' final weights *bitwise* equal, not just close?

<details><summary>answer</summary>

every rank receives exactly the same averaged gradient bits from the all-reduce (gloo computes the sum once per chunk and distributes it), and applies the same deterministic optimizer to the same starting weights. same inputs, same operations, same order: same bits.

</details>

6. what does overlapping communication with the backward pass require of the order in which gradients are bucketed?

<details><summary>answer</summary>

buckets should fill in the order gradients are produced, last layer first, so the first bucket is ready (and can start its all-reduce) while the backward pass is still computing the earlier layers' gradients. DDP builds buckets in reverse parameter order for that reason.

</details>

## common bugs

- **summing without dividing by the world size.** the effective learning rate is multiplied by $p$; training diverges as you add ranks.
- **forgetting to broadcast the initial weights.** each replica trains a different model; losses look plausible on every rank, and the results are meaningless.
- **different shuffles on different ranks.** without a shared seed, shards overlap and miss examples.
- **a collective inside a branch that not every rank takes** (logging on rank 0 only, an unequal number of batches). the program hangs with no error message.
- **dividing by the number of micro-batches with unequal micro-batch sizes.** the gradient is biased towards the smaller micro-batch's examples.
- **timing a single run on a shared machine.** scaling efficiency is a ratio of two noisy numbers; repeat and report the spread.

## going deeper

- Patarasuk & Yuan (2009), "Bandwidth optimal all-reduce algorithms for clusters of workstations", *Journal of Parallel and Distributed Computing* 69(2). [doi:10.1016/j.jpdc.2008.09.002](https://doi.org/10.1016/j.jpdc.2008.09.002)
- Sergeev & Del Balso (2018), "Horovod: fast and easy distributed deep learning in TensorFlow": ring all-reduce for deep learning. [arXiv:1802.05799](https://arxiv.org/abs/1802.05799)
- Li et al. (2020), "PyTorch distributed: experiences on accelerating data parallel training", *VLDB*: bucketing and overlap in DDP. [arXiv:2006.15704](https://arxiv.org/abs/2006.15704)
- Goyal et al. (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour": the linear learning-rate scaling rule and warmup. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- McCandlish, Kaplan, Amodei et al. (2018), "An empirical model of large-batch training": the critical batch size. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

## connections

- ← builds on: the training loop (ch 15), bandwidth and the roofline (01).
- → used in: topic 05 shards the optimizer state with a reduce-scatter and an all-gather, the two halves of your ring, and splits single layers across ranks; chapter 19's critical batch size explains when more data parallelism stops helping. the capstone uses gradient accumulation's idea of trading batch size for steps.
