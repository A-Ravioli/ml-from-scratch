# replay & architectural approaches

## why this matters

topic 13 ended with a wall. in the class-incremental scenario, where the model must pick among every class it has seen without being told the task, a penalty on weights barely helps: EWC and SI land within a few points of fine-tuning's 20%. the problem isn't that important weights move. it's that nothing in the new task's training ever says "and 0s are still 0s": every gradient step pushes old classes' outputs down, and a spring on the weights can't push them back up.

the direct fix is to *show* the network old examples while it learns new ones. that's **replay**, and it's the oldest and most reliable idea in continual learning. brains seem to do something like it: the hippocampus replays recent experience during sleep, which the theory of complementary learning systems (McClelland, McNaughton & O'Reilly, 1995) proposes as the way new memories get woven into cortex without overwriting old ones. in a network, replay means a small **memory** of past examples, mixed into every training batch. it raises two questions this topic answers: *which* examples should a fixed-size memory hold, and does it have to hold real examples at all? a model of the data, a **generator**, can stand in for the data itself.

the other family avoids sharing in the first place. **architectural** or **parameter-isolation** methods give each task its own part of the network, so learning one task can't disturb another's parameters. they store no data, but they pay elsewhere: in capacity, and in needing to know at test time which task an input belongs to.

the fair way to compare all of this, and the regularization methods of topic 13, is a **memory budget**: how many floats does a method keep beyond the network itself? the block's project, at the end of the lab, gives you a budget of 39,200 floats (the size of 50 MNIST images) and asks you to beat fine-tuning on class-incremental split MNIST.

## warm-up: predict first

1. a stream of 1,000,000 examples arrives one at a time and you can store 100. you want your 100 to be a uniform random sample of everything seen so far, at every moment. is that possible without knowing the stream's length in advance?
2. a buffer of 10 images (one per digit, on average) replayed during class-incremental split MNIST: does it help at all? by how much?
3. 90% of a stream is class 0. a uniform sample of the stream is 90% class 0 too. is that what you want in a replay memory?
4. a diagonal Gaussian per class generates "digits" that look like noise with a faint shape. could replaying those help?
5. you give each of five tasks a random 20% of the hidden units. how much do two tasks' units overlap?

## prerequisites check

- [ ] you've built topic 12's harness, in particular `train_task`'s `replay` hook and the two scenarios (→ ch 29 · topic 12)
- [ ] you know what EWC and SI store (→ ch 29 · topic 13)
- [ ] you can sample without replacement and reason about inclusion probabilities (→ ch 04 · topic 02)
- [ ] you know the maximum-likelihood mean and (biased) standard deviation of a Gaussian (→ ch 05 · topic 02)
- [ ] you can write an `nn.Module` with buffers (`register_buffer`) (→ ch 15 · topic 07)

## core ideas

### idea 1: experience replay and the reservoir

**experience replay** for continual learning (Rolnick et al., 2019; Chaudhry et al., 2019) is almost embarrassingly simple: keep a small buffer of past examples, and append a random minibatch from it to every training batch of the current task. the loss now includes old classes' examples with their labels, so their logits get pulled *up* again on every step. on paper 4 makes this precise: replay restores the balance at the output layer that fine-tuning destroys.

what should a buffer of $M$ slots hold? if the stream's length were known you'd pick $M$ examples uniformly at random. you don't know it (a continual learner never knows how many tasks are coming), but **reservoir sampling** (Vitter, 1985) maintains a uniform sample anyway, with one pass and $O(M)$ memory:

- the first $M$ examples fill the slots.
- when example number $n > M$ arrives, draw $j$ uniformly from $\{0, 1, \dots, n-1\}$. if $j < M$, the new example replaces slot $j$; otherwise it's dropped.

after $n$ examples, every one of them is in the buffer with probability exactly $M/n$ (on paper 1), however the stream was ordered. the count $n$ is per *example*: when a whole task's data is added as a batch, each row is one step of the algorithm. reservoir sampling is the default memory in continual learning, and a strong one: Chaudhry et al. found that even tiny reservoirs, a handful of examples per class, beat most regularization methods.

> **spec · stage 1.** `ReservoirBuffer(capacity, *, seed=0)` (`ValueError` if `capacity < 1`), with all randomness from `np.random.default_rng(seed)`. `add(x, y)`: streams the rows of `x` (float tensor `(b, d)`) and `y` (int64 `(b,)`) in order through reservoir sampling as above, the count of seen examples kept in `n_seen`. `sample(n)`: `min(n, len(self))` distinct stored examples chosen uniformly at random, returned as `(x, y)` tensors. `__len__`: the number stored. `n_floats()`: the floats used to store inputs, `capacity * d` (0 before the first `add`).

> **▶ build: stage 1.** implement `ReservoirBuffer` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** in class-incremental split MNIST, fine-tuning ends at ACC 0.20 and joint training reaches 0.96. where does a reservoir of 200 images (about 3 KB per image as float32, 0.3% of the data) land?

### idea 2: balancing the memory

a reservoir mirrors the stream. that's ideal when classes arrive in equal amounts, as in split MNIST, and wrong when they don't: if 90% of what you've seen is class 0, a uniform sample spends 90% of the memory on class 0, and the rare classes, often the ones you most need to remember, get a few slots or none.

a simplified version of **class-balanced reservoir sampling** (CBRS; Chrysakis & Moens, 2020; the original tracks which classes have *ever* been the largest) keeps the reservoir's one-pass, fixed-memory character but aims for an equal share per class:

- while the buffer isn't full, store every example.
- once it's full, an example whose class is *not* currently among the largest in the buffer always gets in: it replaces a randomly chosen stored example of a largest class (if several classes tie for largest, pick one of them at random).
- an example whose class *is* already among the largest goes through reservoir sampling within its own class: with $m_c$ of its class stored and $n_c$ of its class seen so far (this one included), keep it with probability $m_c / n_c$, replacing a random stored example of the same class.

rare classes grow until they're as large as the largest, then everyone shares the memory evenly; within a class, the stored examples stay a uniform sample of that class. on split MNIST, where classes are already balanced, it behaves almost exactly like a plain reservoir; the checks feed it a 300/60/30 stream, where the difference is the whole point.

> **spec · stage 2.** `ClassBalancedBuffer(capacity, *, seed=0)` (`ValueError` if `capacity < 1`), randomness from `np.random.default_rng(seed)`; `add(x, y)` streams rows through the three rules above. `counts()`: dict class → number stored (classes with none left out). `sample(n)`, `__len__`, `n_floats()` as for `ReservoirBuffer`.

> **▶ build: stage 2.** implement `ClassBalancedBuffer`.

### idea 3: generative replay

a buffer stores examples. a **generative model** stores a *description* of the data and can produce as many examples as you like. **generative replay** (Shin et al., 2017) trains a generator alongside the classifier and, while learning a new task, replays samples from the generator instead of stored data. it sidesteps privacy limits on storing data and, potentially, uses memory far more efficiently. the hard part in general is that the generator is itself a network learning continually, so it forgets too; full-scale versions replay the old generator's samples to train the new generator (van de Ven, Siegelmann & Tolias, 2020).

this topic builds the smallest generator that could work: one **diagonal Gaussian per class**. for class $c$, store the per-pixel mean $\mu_c$ and standard deviation $\sigma_c$ of its training images (the maximum-likelihood fit), and generate by drawing $x = \mu_c + \sigma_c \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$. a pixel that's constant within a class (MNIST's border is always black) has $\sigma = 0$; floor the standard deviation at a small `min_std` so the model stays a proper distribution. because each class is fitted from its own task's data only, the generator can't forget: refitting a class replaces its Gaussian, and other classes are untouched.

the samples are poor. pixels are independent, so a generated 3 is the average 3 covered in static. but the cost is $2 \times 784$ floats per class: for all ten digits, 15,680 floats, the memory of just 20 stored images. whether those static-covered averages are worth more or less than 20 real images is the lab's second experiment.

counting memory honestly takes care. the unit matters: MNIST pixels are bytes, so stored as uint8 an image costs 784 bytes, not 3,136, and a byte budget favours buffers four times more than a float budget does. `n_floats()` is self-reported, so a comparison is only as fair as each memory's accounting. and the budget ignores compute: replaying 64 examples per step doubles the work of every step.

> **spec · stage 3.** `GaussianGenerator(*, min_std=0.01, seed=0)`, sampling randomness from a `torch.Generator` seeded with `seed`. `fit(x, y)`: for each class $c$ in `y`, `means[c]` = the mean of its rows and `stds[c]` = their biased standard deviation (divide by $n$), clamped below at `min_std`; refitting a class replaces it. `means` and `stds` are dicts class → tensor `(d,)`. `sample(n)`: `n` labels drawn uniformly from the fitted classes and, for each, `x = means[y] + stds[y] * standard normal noise`; returns `(x, y)` with `y` int64; `RuntimeError` before any `fit`. `n_floats()`: total size of all stored means and stds.

> **▶ build: stage 3.** implement `GaussianGenerator`. the worked example below is stage 3's first check.

### idea 4: parameter isolation

replay and regularization both let tasks share every parameter and then manage the interference. **parameter isolation** removes the interference at the source: give each task its own parameters. the spectrum runs from a new column of layers per task with lateral connections to the old ones (progressive networks; Rusu et al., 2016), through pruning a shared network and freezing each task's surviving weights (PackNet; Mallya & Lazebnik, 2018), to learned attention masks over units (HAT; Serra et al., 2018).

the simplest version is **context-dependent gating** (Masse, Grant & Freedman, 2018): before training, draw for every task a fixed random binary mask over each hidden layer's units, keeping a fraction `keep` of them. the forward pass for task $t$ multiplies each hidden layer's activations by task $t$'s mask:

$$h^{(l)} = \operatorname{relu}\big(W^{(l)} h^{(l-1)} + b^{(l)}\big) \odot m^{(l)}_t.$$

a unit that's gated off for task $t$ contributes nothing to task $t$'s output, so task $t$'s loss sends it no gradient (on paper 3): its incoming weights are untouched by task $t$'s training. two tasks with independent masks share, on average, a fraction `keep` of each other's units, so interference shrinks roughly in proportion. Masse et al. kept 20% and combined gating with SI; on its own it already removes most of the forgetting in the lab.

the price is in the forward pass's signature: `forward(x, task)`. the model must be told which task an input comes from, so gating is a task-incremental method; it has nothing to offer class-incrementally, where the task is exactly what's unknown. capacity is the other price: each task trains only `keep` of the network, and with many tasks the masks overlap heavily again.

> **spec · stage 4.** `GatedMLP(in_features, hidden_sizes, out_features, n_tasks, *, keep=0.2, seed=0)`, an `nn.Module` with `hidden`, an `nn.ModuleList` of the hidden `nn.Linear` layers, and `head`, the output `nn.Linear`. for each hidden layer of width $h$ and each task, a fixed 0/1 float mask with exactly `round(keep * h)` ones at random positions (drawn from a `torch.Generator` seeded with `seed`), registered as buffers, not parameters. `masks()`: list over hidden layers of the `(n_tasks, h)` mask tensors. `set_task(t)`: selects the task `forward` uses by default (`ValueError` unless $0 \le t <$ `n_tasks`); the initial task is 0. `forward(x, task=None)`: each hidden layer computes `relu(linear(h)) * mask[task]`, then `head`; `task=None` uses the selected task.

> **▶ build: stage 4.** implement `GatedMLP`.

> **🔮 predict:** gating with `keep = 0.2` gives each split-MNIST task 40 of each layer's 200 units. better or worse than `keep = 0.5`?

## worked example

fit a `GaussianGenerator` (`min_std = 0.01`) to four 3-pixel examples: $(0, 2, 5)$, $(2, 2, 5)$, $(1, 8, 5)$ with label 1, and $(9, 9, 9)$ with label 4.

- class 1: mean $(1, 4, 5)$. biased variances: pixel 1, $\frac{1 + 1 + 0}{3} = \frac23$; pixel 2, $\frac{4 + 4 + 16}{3} = 8$; pixel 3, $0$. standard deviations $(\sqrt{2/3}, \sqrt 8, 0) = (0.816, 2.828, 0)$, and the constant pixel is floored: $\sigma = (0.816, 2.828, 0.01)$.
- class 4: one example, so its mean is $(9, 9, 9)$ and every standard deviation is $0$, floored to $0.01$: it will replay (almost exactly) that single example.
- memory: 2 classes × (mean + std) × 3 pixels = 12 floats. for MNIST, 10 × 2 × 784 = 15,680.

your generator must reproduce these numbers. and for the class-balanced buffer, a hand trace of the check's stream: 300 examples of class 0 fill all 30 slots; each of the first 15 class-1 examples evicts a class-0 example (class 1 isn't the largest), reaching 15/15; from then on class 1 ties for largest and is reservoir-sampled within itself, staying at 15; the class-2 examples evict from whichever of classes 0 and 1 is larger until all three have 10.

## on paper

1. prove that after $n \ge M$ examples, reservoir sampling holds each example with probability $M/n$.

<details><summary>solution</summary>

induction on $n$. at $n = M$ everything is stored: probability $1 = M/M$. suppose it holds after $n - 1$ examples. the new example $n$ is stored iff $j < M$ for $j$ uniform on $n$ values: probability $M/n$ ✓. an older example, stored with probability $M/(n-1)$, survives step $n$ unless the new one is stored *and* lands in its slot, which happens with probability $\frac{M}{n}\cdot\frac{1}{M} = \frac1n$. so it's stored after step $n$ with probability $\frac{M}{n-1}\big(1 - \frac1n\big) = \frac{M}{n-1}\cdot\frac{n-1}{n} = \frac Mn$ ✓.

</details>

2. count memory in floats for a 784–200–200–10 network: (a) online EWC, (b) a reservoir of $M$ images, (c) a diagonal Gaussian per class for 10 classes, (d) `GatedMLP` masks for 5 tasks. which fit a budget of 39,200?

<details><summary>solution</summary>

the network has $784 \cdot 200 + 200 + 200 \cdot 200 + 200 + 200 \cdot 10 + 10 = 199{,}210$ weights. (a) anchor + Fisher: $398{,}420$, ten times the budget. (b) $784M$: up to $M = 50$. (c) $10 \times 2 \times 784 = 15{,}680$. (d) $5 \times (200 + 200) = 2{,}000$ bits' worth (stored as floats here, still tiny), but it needs the task identity, which the class-incremental project doesn't give it. labels cost one integer per stored example and are usually left out of the count.

</details>

3. show that in a `GatedMLP`, a hidden unit gated off for task $t$ receives zero gradient from any loss computed with task $t$'s gate. with independent masks keeping $k$ of $h$ units, how many units do two tasks share on average?

<details><summary>solution</summary>

unit $u$'s gated activation is $a_u \cdot m_{t,u}$ with $m_{t,u} = 0$, so the output doesn't depend on $a_u$; by the chain rule, $\partial L/\partial a_u = (\partial L/\partial(a_u m_{t,u}))\, m_{t,u} = 0$, and so are the gradients of $W_{u,:}$ and $b_u$. (the unit's *outgoing* weights also get zero gradient, since their input is zero.) each of task $t'$'s $k$ units is also in task $t$'s set with probability $k/h$, so they share $k^2/h$ on average: with $h = 200$ and `keep = 0.2` ($k = 40$), 8 units, a fifth of each task's units.

</details>

4. at the output layer, the gradient of the mean cross-entropy with respect to the bias of class $c$ is $\bar p_c - \hat\pi_c$, the batch's mean predicted probability of $c$ minus the fraction of the batch labelled $c$. use this to explain why fine-tuning suppresses old classes and how replay fixes it.

<details><summary>solution</summary>

from $\partial L/\partial z_c = p_c - y_c$ (topic 12, on paper 1), averaging over the batch gives $\bar p_c - \hat\pi_c$. gradient descent drives the bias towards $\bar p_c = \hat\pi_c$: the model's average prediction of $c$ matches $c$'s share of the training batches. fine-tuning on task $t$ has $\hat\pi_c = 0$ for every old class, so their predicted probabilities are driven to zero: the class-incremental collapse. replay makes $\hat\pi_c > 0$ again, roughly $c$'s share of the replayed half of the batch, so old classes are predicted at sensible rates. note this argument only needs replayed examples to *carry old labels in roughly the right region of input space*; it doesn't need them to be realistic, which is a hint about experiment 2.

</details>

## experiment

the lab plugs your buffers, generator and gated network into your topic-12 harness, mostly in the class-incremental scenario, where topics 12 and 13 left fine-tuning, EWC and SI near 20%. first a reservoir of 10 to 1,000 images. then the Gaussian generator against buffers of the same and of ten times its memory, with a look at what it generates. then gating, task-incrementally, at two levels of `keep`. finally **the block's project**: beat fine-tuning on class-incremental split MNIST with at most 39,200 floats of memory, over five seeds, with a confidence interval for your improvement. predict before each; experiment 2's result is the one to argue about.

## check your understanding

1. why must reservoir sampling count seen examples individually, even when a whole batch is added at once?

<details><summary>answer</summary>

the acceptance probability $M/n$ depends on each example's position $n$ in the stream. counting a batch as one step would give every row of the batch the same, too-large acceptance probability, and later batches would crowd out earlier ones. the checks run 2,000 small reservoirs (3 slots, 10 examples in batches of 4) and apply a chi-square test to how often each position is kept; counting batches, or drawing $j$ from a range one too large or too small, fails it decisively.

</details>

2. could the class-balanced buffer be worse than a reservoir?

<details><summary>answer</summary>

yes, if the class imbalance in the stream reflects the test distribution: then a uniform sample of the past matches what you'll be evaluated on, and balancing over-represents rare classes. it can also hurt when the class of interest is "the most common one". balancing is right when every class matters equally at test time, as in split MNIST's average accuracy.

</details>

3. the Gaussian generator can't forget, because each class is fitted once from its own data. what would break if a class appeared in two different tasks with different styles?

<details><summary>answer</summary>

the spec's `fit` replaces a refitted class's Gaussian, so the earlier style would be forgotten, exactly the problem it was meant to solve. a better generator would merge the statistics (pooling means and variances weighted by counts, which needs the counts stored), or keep a mixture component per (class, task). real generative replay faces the same issue in a harder form, since a neural generator shares all its parameters across classes.

</details>

4. gating needs the task identity at test time. is there any way to use it class-incrementally?

<details><summary>answer</summary>

only by inferring the task first: run the input through every task's gate and pick the most confident head, or train a separate task classifier (which itself must learn continually and forgets). both reintroduce the hard part of the class-incremental problem, which is why van de Ven & Tolias (2019) evaluate gating only in the task-incremental scenario.

</details>

5. EWC's memory doesn't grow with the data, and a replay buffer's does (if you want it to keep up). for a big model and a small stream, which is cheaper?

<details><summary>answer</summary>

for a 200,000-weight MLP, online EWC's $2P$ floats equal about 500 MNIST images. for a model with a billion parameters, $2P$ is 8 GB at float32, while a buffer of a few thousand examples is megabytes. the comparison flips with scale, which is part of why replay dominates continual learning for large models: storing a little data is usually far cheaper than storing a copy of the model.

</details>

## common bugs

- **counting batches instead of examples in the reservoir**: symptom: the uniformity check finds late examples over-represented. increment `n_seen` per row.
- **evicting from the incoming class in the balanced buffer**: symptom: rare classes never grow. the incoming example of a non-largest class evicts from a *largest* class.
- **sampling with replacement**: symptom: duplicated rows in a replay batch; the distinctness check fails. use `rng.choice(size, k, replace=False)`.
- **unbiased standard deviation or no floor**: symptom: the generator check is off by $\sqrt{n/(n-1)}$, or `nan`/zero-variance classes. use `unbiased=False` and clamp at `min_std`.
- **masks as parameters, or redrawn every forward pass**: symptom: the masks change during training or appear in `parameters()`. draw them once in `__init__` and `register_buffer` them.
- **gating before the nonlinearity with a bias**: `relu(linear(h) * mask)` is the same here, but `linear(h * mask)` gates the *inputs* instead of the units; the forward check catches it.

## going deeper

- Vitter (1985), "Random sampling with a reservoir", *ACM Transactions on Mathematical Software* 11(1). [doi:10.1145/3147.3165](https://doi.org/10.1145/3147.3165)
- Rolnick, Ahuja, Schwarz, Lillicrap & Wayne (2019), "Experience replay for continual learning", *NeurIPS*. [arXiv:1811.11682](https://arxiv.org/abs/1811.11682)
- Chaudhry, Rohrbach, Elhoseiny, Ajanthan, Dokania, Torr & Ranzato (2019), "On tiny episodic memories in continual learning". how far a few stored examples go. [arXiv:1902.10486](https://arxiv.org/abs/1902.10486)
- Chrysakis & Moens (2020), "Online continual learning from imbalanced data", *ICML*. class-balanced reservoir sampling. [PMLR 119](https://proceedings.mlr.press/v119/chrysakis20a.html)
- Shin, Lee, Kim & Kim (2017), "Continual learning with deep generative replay", *NeurIPS*. [arXiv:1705.08690](https://arxiv.org/abs/1705.08690)
- van de Ven, Siegelmann & Tolias (2020), "Brain-inspired replay for continual learning with artificial neural networks", *Nature Communications* 11. [doi:10.1038/s41467-020-17866-2](https://doi.org/10.1038/s41467-020-17866-2)
- Masse, Grant & Freedman (2018), "Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization", *PNAS* 115(44). [doi:10.1073/pnas.1803839115](https://doi.org/10.1073/pnas.1803839115)
- Rusu et al. (2016), "Progressive neural networks". [arXiv:1606.04671](https://arxiv.org/abs/1606.04671); Mallya & Lazebnik (2018), "PackNet: adding multiple tasks to a single network by iterative pruning", *CVPR*. [arXiv:1711.05769](https://arxiv.org/abs/1711.05769); Serra, Surís, Miron & Karatzoglou (2018), "Overcoming catastrophic forgetting with hard attention to the task", *ICML*. [arXiv:1801.01423](https://arxiv.org/abs/1801.01423)
- McClelland, McNaughton & O'Reilly (1995), "Why there are complementary learning systems in the hippocampus and neocortex", *Psychological Review* 102(3). [doi:10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419)

## connections

- ← builds on: topic 12's harness (`replay` hook, scenarios, metrics) and topic 13's methods (as the memory-budget comparison), sampling and inclusion probabilities (ch 04), Gaussian maximum likelihood (ch 05), `nn.Module` buffers and `mlfs.torchnn.MLP` (ch 15), `bootstrap_ci` (ch 05) for the project's interval.
- → used in: the elective-portfolio capstone reports this block's project against fine-tuning. replay buffers return in reinforcement learning (DQN's experience replay, ch 24), where they break the correlation of consecutive samples rather than fight forgetting; class-conditional generators are developed properly in chapters 21–22; and mixing a fraction of pretraining data into fine-tuning batches to keep a large model's general skills (ch 27) is replay at scale.
