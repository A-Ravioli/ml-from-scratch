# catastrophic forgetting

## why this matters

train a network to tell 0s from 1s until it's perfect. now keep training the same network, with the same optimizer, on 2s versus 3s, and never show it another 0 or 1. after a few hundred images of 2s and 3s, ask it about a 0. it has no idea. it hasn't merely got worse: in the most common setup it will *never* answer "0" again, for any input.

this is **catastrophic forgetting** (McCloskey & Cohen, 1989), and it's the default behaviour of every network trained by gradient descent. nothing in the update for the new data refers to the old data, so nothing protects what the old data taught. a deployed system that keeps learning from a stream (a robot in a new room, a recommender seeing new products) can't afford that. the field that studies this is **continual learning** (also lifelong or incremental learning).

before you can fix forgetting, you have to measure it, and that's harder than it sounds: the answer depends on how the tasks are cut, on what the model is told at test time, and on which numbers you report. so this topic builds the instrument: a small, careful **benchmark harness** with task sequences, an accuracy matrix, the standard transfer metrics, and the two evaluation scenarios that matter most. topics 13 (EWC, SI) and 14 (replay, parameter isolation) plug their methods into it, and the lab uses it to watch forgetting happen step by step.

## warm-up: predict first

1. a network is trained on five tasks in sequence (0/1, 2/3, 4/5, 6/7, 8/9), each to near-perfect accuracy, with one shared 10-way softmax. at the end, what's its average accuracy over the five tasks?
2. same network and data, but at test time you tell it which task an image comes from, so it only has to choose between that task's two digits. better, worse, or the same?
3. you measure accuracy on task 1 right after learning it (0.99) and after learning everything (0.80). is that the whole story of forgetting? what if task 1 had peaked at 0.995 after task 2?
4. can learning task 2 make the network better at task 1, which it isn't training on?
5. for permuted MNIST (the same digits, pixels shuffled by a fixed permutation per task), do you expect forgetting to be faster or slower than for split MNIST?

## prerequisites check

- [ ] you can write a PyTorch training loop: minibatches, `zero_grad`, `backward`, `step` (→ ch 15 · topic 07)
- [ ] you know softmax cross-entropy and its gradient with respect to the logits, $p - y$ (→ ch 08 · topic 05)
- [ ] you know what SGD and Adam do with a gradient (→ ch 06 · topic 04, ch 15 · topic 04)
- [ ] you know `model.train()` vs `model.eval()` and `torch.no_grad()` (→ ch 15 · topic 07)
- [ ] you can read a confusion matrix and an accuracy matrix (→ ch 07 · topic 03)

## core ideas

### idea 1: a stream of tasks

continual learning replaces one dataset with a **sequence of tasks** $\mathcal{T}_0, \mathcal{T}_1, \dots, \mathcal{T}_{T-1}$, seen one at a time. while training on $\mathcal{T}_t$, the learner gets only $\mathcal{T}_t$'s data; the old data is gone (or, in topic 14, partly remembered in a small memory). two constructions from MNIST are the standard toy benchmarks, and you'll build both.

**split MNIST** cuts the ten classes into groups and makes each group a task: $\{0,1\}, \{2,3\}, \{4,5\}, \{6,7\}, \{8,9\}$. every task is easy on its own (better than 99%), and the tasks are *different*: they share no classes, so later tasks bring outputs the network has never been asked to produce.

**permuted MNIST** (popularized by Goodfellow et al., 2013) keeps all ten classes in every task, and instead shuffles the 784 pixels with a fixed random permutation per task. task 0 is ordinary MNIST. to a fully connected network, each permutation is an unrelated new problem with the same label set: no spatial structure is shared, but the output layer's meaning is.

two details make a benchmark honest. first, the permutation is a property of the *task*, so a task's training set and test set must be permuted identically; the cleanest way is to derive the permutations from a seed alone, never from the data or a global random state. second, keep the data as given: flatten and convert, nothing more.

> **spec · stage 1.** a task is a dict `{"X": float32 tensor (n, d), "y": int64 tensor (n,), "classes": tuple of ints}`; `X` is the input flattened to `(n, d)` with values unchanged. `split_tasks(X, y, class_groups)`: one task per group, holding the examples whose label is in the group, in their original order, with original labels and `classes = tuple(group)`. `permuted_tasks(X, y, n_tasks, *, seed=0)`: `n_tasks` tasks over all the examples; task 0 is the flattened data as is, task $k > 0$ reorders the columns by a permutation that depends only on `(seed, k)` (so the same seed permutes a train and a test set identically); `classes` is the sorted tuple of distinct labels.

> **▶ build: stage 1.** implement `split_tasks` and `permuted_tasks` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the accuracy matrix and what to read off it

one number can't describe a continual learner, because "how good is it?" depends on *when* you ask and *about what*. the complete record is the **accuracy matrix** $R \in [0,1]^{T \times T}$:

$$R_{ij} = \text{test accuracy on task } j \text{ after finishing training on task } i.$$

row $i$ is a snapshot of the learner after task $i$. the diagonal $R_{jj}$ is how well each task was learned when it was current. below the diagonal is memory: what's left of task $j$ later. above the diagonal is anticipation: accuracy on tasks not yet seen. the standard summaries (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018):

- **average accuracy**, the headline: $\text{ACC} = \frac{1}{T}\sum_{j} R_{T-1,j}$, the mean of the last row.
- **backward transfer**: $\text{BWT} = \frac{1}{T-1}\sum_{j<T-1} (R_{T-1,j} - R_{jj})$, how much learning later tasks changed earlier ones. negative is forgetting; positive means later tasks *helped* old ones.
- **forward transfer**: $\text{FWT} = \frac{1}{T-1}\sum_{j>0} (R_{j-1,j} - b_j)$, where $b_j$ is the accuracy of the untrained model on task $j$. it asks whether earlier tasks prepared the network for task $j$ before it ever saw $j$'s data.
- **forgetting**: $f = \frac{1}{T-1}\sum_{j<T-1} \big(\max_{l<T-1} R_{lj} - R_{T-1,j}\big)$, the drop from the *best* accuracy task $j$ ever had to its final accuracy. it differs from $-\text{BWT}$ when a task kept improving after it was learned and only then decayed; on paper 3 shows $f \ge -\text{BWT}$ always.

BWT and forgetting need at least two tasks; define them only for $T \ge 2$ (raise an error otherwise). always report ACC *and* a forgetting measure, and ideally the learning accuracy (the mean of the diagonal): a method can show zero forgetting simply by never learning anything new.

> **spec · stage 2.** `average_accuracy(R)`, `backward_transfer(R)`, `forward_transfer(R, baseline)`, `forgetting(R)`: the four formulas above for an array-like `R` of shape `(T, T)` and `baseline` of shape `(T,)`, returning python floats. the last three raise `ValueError` when $T < 2$.

> **▶ build: stage 2.** implement the four metrics. the worked example below is stage 2's first check.

> **🔮 predict:** in a class-incremental split-MNIST run where every old task drops to exactly 0% and every task is learned perfectly, what are ACC, BWT and forgetting for $T = 5$?

### idea 3: what the model is told at test time

here's the distinction that decides whether forgetting is mild or total. van de Ven & Tolias (2019) separate three **scenarios** by what the learner must output at test time:

- **task-incremental**: the test input comes with its task identity, and the model only has to choose among that task's classes. for split MNIST: "this is from task 2; is it a 4 or a 5?"
- **domain-incremental**: no task identity, but the label set is the same in every task, so it isn't needed. permuted MNIST is the classic example: "which digit is this?", whatever the permutation.
- **class-incremental**: no task identity, and the model must choose among *all* classes seen so far. for split MNIST: "which of the ten digits is this?"

in code, the difference between the task- and class-incremental scenarios on split MNIST is one masking step. with a single 10-way output layer, task-incremental means restricting the softmax to the current task's classes: set every other logit to $-\infty$ before the softmax (or the argmax). that's equivalent to a separate two-way "head" per task sharing one body, the usual **multi-head** setup. class-incremental uses all ten logits. for permuted MNIST every task's classes are all ten digits, so the mask does nothing and the two coincide (both are really the domain-incremental scenario).

why does it matter so much? because in the class-incremental case, old classes compete with new ones *at the output*. when the network trains on 2s and 3s with a full softmax, the cross-entropy gradient on the logit of class 0 is $p_0 - 0 = p_0 > 0$ for every training image (on paper 1). gradient descent pushes class 0's logit down on every step, and nothing ever pushes it back up. with masking, the old classes' logits are out of the softmax and receive no gradient at all; the shared hidden layers still change, but the old task's decision only has to separate *its own* two classes, a far easier thing to preserve.

> **spec · stage 3.** `scenario_logits(logits, classes, scenario)`: for `scenario == "class"` returns `logits` unchanged; for `"task"` returns a new tensor where every column not in `classes` is $-\infty$ (differentiable, the input not modified); anything else raises `ValueError`. `evaluate(model, task, *, scenario="class", batch_size=1024)`: the fraction of `task`'s examples whose argmax of `scenario_logits(model(X), task["classes"], scenario)` equals the label, as a python float; evaluated in eval mode under `torch.no_grad()`, in batches (including the last partial one), restoring the model's previous train/eval mode.

> **▶ build: stage 3.** implement `scenario_logits` and `evaluate`.

### idea 4: training on one task, with hooks

the baseline every continual-learning method must beat is **fine-tuning**: train on each task as if it were the only one, starting from wherever the last task left the weights. you know how to write this loop; the harness version adds the two things a benchmark needs, determinism and extension points.

determinism means a training run is a function of its inputs: the shuffling of each epoch comes from a `torch.Generator` seeded by an explicit `seed`, never from the global state, and a fresh optimizer is built for each task (so no momentum or Adam statistics leak across task boundaries, which would be a hidden memory of the old task).

the extension points are three hooks, which topics 13 and 14 use:

- `penalty(model)`: a scalar added to the loss at every step. EWC and SI are exactly this: a quadratic pull towards old weights.
- `after_step(model)`: called after every optimizer step. SI uses it to integrate each weight's contribution to the loss along the training trajectory.
- `replay()`: returns a batch `(x, y)` of remembered examples to append to the current batch (or `None`). replay methods are this. in the task-incremental scenario each replayed example would need its own task's mask, so the harness supports replay only class-incrementally and raises an error otherwise.

why does fine-tuning forget at all, in the simplest terms? take one SGD step $\Delta\theta = -\eta \nabla L_B$ on the new task $B$. to first order, the old task's loss changes by $\Delta L_A \approx \nabla L_A^\top \Delta\theta = -\eta\, \nabla L_A^\top \nabla L_B$ (on paper 2). when the two gradients point in conflicting directions (negative inner product), every step on $B$ increases $L_A$; when they're orthogonal, $A$ is untouched to first order. continual-learning methods can be read as ways of controlling this inner product: by penalizing movement in the directions $A$ cares about (topic 13), by adding $A$'s gradient back in through replay (topic 14), or by giving $A$ and $B$ disjoint parameters so the inner product is zero by construction (topic 14).

> **spec · stage 4.** `train_task(model, task, *, scenario="class", epochs=1, lr=0.05, batch_size=64, seed=0, optimizer="sgd", penalty=None, after_step=None, replay=None)`: builds a fresh `torch.optim.SGD(model.parameters(), lr=lr)` (plain, no momentum) or, for `optimizer="adam"`, `torch.optim.Adam(model.parameters(), lr=lr)` (other values: `ValueError`); puts the model in train mode; creates `g = torch.Generator().manual_seed(seed)` once per call; at the start of each epoch draws `order = torch.randperm(n, generator=g)` and steps through minibatches `order[i:i + batch_size]` of `batch_size`, the last one possibly smaller. per step: if `replay` is given and `replay()` returns `(x_r, y_r)`, append them to the batch; the loss is the mean cross-entropy of `scenario_logits(model(x), task["classes"], scenario)` over the (combined) batch, plus `penalty(model)` if given; then `zero_grad`, `backward`, `step`, then `after_step(model)` if given. `replay` with `scenario="task"` raises `ValueError`. returns the list of per-step loss values as python floats.

> **▶ build: stage 4.** implement `train_task`.

> **🔮 predict:** fine-tuning class-incrementally, how many steps on task 1 does it take for task 0's accuracy to fall below 50%: closer to 20, 200, or 2,000?

### idea 5: running a sequence

the last piece is the protocol: train on the tasks in order, and after each one, evaluate on *every* task's test set, including the future ones (forward transfer needs them). before any training, evaluate once more to get the baseline $b$. the result is $R$ and $b$, from which every metric follows.

the training function is pluggable, `train_fn(model, task, t)`: a method is a closure that calls `train_task` with the right hooks and updates its own state between tasks. with no `train_fn`, the runner fine-tunes, seeding task $t$ with $t$.

protocol rules matter more than code: one fixed test set per task; the same network, optimizer and epochs for every method; no tuning on old tasks' test accuracy (a validation split of the current task is fair); and several seeds, because every task starts wherever the noisy previous one ended.

> **spec · stage 5.** `run_sequence(model, train_tasks, test_tasks, *, train_fn=None, scenario="class", **train_kwargs)`: `ValueError` unless the two lists have the same length $T$. first `baseline[j] = evaluate(model, test_tasks[j], scenario=scenario)`; then for $t = 0, \dots, T-1$: call `train_fn(model, train_tasks[t], t)`, or, if `train_fn` is `None`, `train_task(model, train_tasks[t], scenario=scenario, seed=t, **train_kwargs)` (a `seed` in `train_kwargs` replaces `t`); then fill row $t$ of $R$ with `evaluate` on every test task. returns `(R, baseline)` as NumPy arrays of shapes `(T, T)` and `(T,)`.

> **▶ build: stage 5.** implement `run_sequence`.

## worked example

three tasks, with this accuracy matrix and untrained baseline:

$$R = \begin{pmatrix} 0.90 & 0.20 & 0.20 \\ 0.95 & 0.80 & 0.30 \\ 0.50 & 0.70 & 0.95 \end{pmatrix}, \qquad b = (0.10,\ 0.15,\ 0.25).$$

- **ACC** $= (0.50 + 0.70 + 0.95)/3 = 0.7167$.
- **BWT** $= \big((0.50 - 0.90) + (0.70 - 0.80)\big)/2 = -0.25$.
- **FWT** $= \big((R_{01} - b_1) + (R_{12} - b_2)\big)/2 = \big((0.20 - 0.15) + (0.30 - 0.25)\big)/2 = 0.05$.
- **forgetting**: task 0 peaked at $0.95$ (after task 1, not at $0.90$ when it was learned), task 1 at $0.80$, so $f = \big((0.95 - 0.50) + (0.80 - 0.70)\big)/2 = 0.275$.

notice $f = 0.275 > -\text{BWT} = 0.25$: task 0 improved after it was learned (positive backward transfer from task 1), then lost more than BWT credits. your metrics must reproduce these four numbers.

## on paper

1. for softmax cross-entropy with logits $z$ and a one-hot label $y$ of class $c$, show $\partial L/\partial z_k = p_k - y_k$. deduce that if every training example's label is outside a set $S$ of classes, plain gradient descent never increases the bias of any class in $S$, and strictly decreases it.

<details><summary>solution</summary>

$L = -z_c + \log\sum_m e^{z_m}$, so $\partial L/\partial z_k = -[k = c] + e^{z_k}/\sum_m e^{z_m} = p_k - y_k$. for $k \in S$, $y_k = 0$ on every example, so the gradient with respect to the bias $b_k$ (which enters $z_k$ with coefficient 1) is the batch mean of $p_k > 0$. a gradient step changes $b_k$ by $-\eta \cdot \text{mean}(p_k) < 0$. the same pressure acts on the output weights of class $k$, pushing $z_k$ down along the directions the new data occupies. that's the class-incremental collapse: old classes are actively suppressed, not merely neglected.

</details>

2. show that to first order an SGD step on task $B$ changes task $A$'s loss by $-\eta\,\nabla L_A^\top \nabla L_B$. what's the change if $B$'s gradient is projected onto the orthogonal complement of $\nabla L_A$ before the step?

<details><summary>solution</summary>

Taylor: $L_A(\theta + \Delta\theta) = L_A(\theta) + \nabla L_A^\top \Delta\theta + O(\|\Delta\theta\|^2)$, and $\Delta\theta = -\eta\nabla L_B$. with the projected step $\Delta\theta = -\eta\big(\nabla L_B - \frac{\nabla L_A^\top\nabla L_B}{\|\nabla L_A\|^2}\nabla L_A\big)$, the first-order change is $-\eta(\nabla L_A^\top\nabla L_B - \nabla L_A^\top\nabla L_B) = 0$. projecting gradients away from old tasks' gradients is the idea behind GEM (Lopez-Paz & Ranzato, 2017), which only projects when the inner product is negative, and orthogonal gradient descent.

</details>

3. prove that forgetting is always at least minus the backward transfer: $f \ge -\text{BWT}$. when are they equal?

<details><summary>solution</summary>

for each $j < T-1$, the max in $f_j$ runs over $l = 0, \dots, T-2$, which includes $l = j$, so $\max_{l<T-1} R_{lj} \ge R_{jj}$ and $f_j = \max_l R_{lj} - R_{T-1,j} \ge R_{jj} - R_{T-1,j}$. averaging over $j$ gives $f \ge -\text{BWT}$. equality holds exactly when every task's best pre-final accuracy is the one right after learning it, i.e. no task ever improved after its own training (before the final row).

</details>

4. in class-incremental split MNIST with $T$ equal-sized tasks, suppose each task is learned perfectly and then completely forgotten when the next one starts. what are ACC and BWT?

<details><summary>solution</summary>

the last row is $(0, \dots, 0, 1)$, so $\text{ACC} = 1/T$, which is $0.2$ for five tasks. the diagonal is all ones, so $\text{BWT} = \frac{1}{T-1}\sum_{j<T-1}(0 - 1) = -1$, and forgetting is $1$. (MNIST's test tasks aren't exactly equal in size, but ACC averages per-task accuracies, so sizes don't matter here.)

</details>

## experiment

the lab runs your harness on real MNIST with a 784–200–200–10 network. first, you'll train on 0 vs 1, then on 2 vs 3, and record the accuracy on 0 vs 1 after *every* optimizer step, in both scenarios. then the full five-task split-MNIST sequence, both scenarios, three seeds, with the accuracy matrices, all four metrics and joint training as a reference. finally ten permuted-MNIST tasks, where you choose the learning rate and see what it does to learning and to forgetting. predict before each one; the first plot is worth guessing carefully.

## check your understanding

1. why must the permutations of permuted MNIST come from the seed alone, and not from, say, `np.random.permutation` at the time the task is built?

<details><summary>answer</summary>

the training and test versions of a task are built by separate calls. if the permutation depends on global random state (or on the data), the test set gets a different permutation from the training set, and you're measuring accuracy on an unrelated task, which will look like near-chance "forgetting" even right after training. deriving permutation $k$ from `(seed, k)` makes every call agree.

</details>

2. the task-incremental scenario restricts the softmax at training time too, not only at test time. what happens if you train with the full softmax and only mask at test time?

<details><summary>answer</summary>

old classes' logits are still pushed down during training (on paper 1), but at test time they only compete with their own task's other class, which was pushed down too, so some of the order inside each task survives. measured with this lab's network (seeds 0–2), masking a class-incrementally trained network at test time gives ACC 0.70–0.77, against 0.20 unmasked and 0.92 for training with the mask: telling the model the task helps a lot, and training it to *use* that knowledge helps more.

</details>

3. a method reports BWT ≈ 0. is it a good continual learner?

<details><summary>answer</summary>

not necessarily. BWT compares final accuracy with the accuracy right after each task was learned. a method that barely learns each task (low diagonal) has little to forget. check the learning accuracy (mean of the diagonal) and ACC too: the stability–plasticity tradeoff means you can always buy BWT ≈ 0 by refusing to learn.

</details>

4. why does `evaluate` restore the model's previous mode instead of simply switching back to train mode?

<details><summary>answer</summary>

it's called in the middle of other code: from an `after_step` hook during training, and on models the caller deliberately put in eval mode. restoring the mode it found keeps dropout and batch norm doing whatever the caller expects.

</details>

5. on permuted MNIST, why do the task- and class-incremental scenarios give identical results?

<details><summary>answer</summary>

every permuted task's `classes` is all ten digits, so the task mask hides nothing: `scenario_logits` returns the logits unchanged in both scenarios.

</details>

## common bugs

- **different permutations for train and test**: symptom: permuted tasks sit near 10% accuracy even right after training on them. derive permutations from the seed and the task index only.
- **dropping the last partial batch** (in `evaluate` or `train_task`): symptom: accuracies or step counts slightly off; the batch-size checks fail. use `range(0, n, batch_size)` and slice.
- **masking in place**: `logits[:, others] = -inf` on the model's output can break autograd or modify a tensor you need later. use `masked_fill`, which returns a new tensor.
- **evaluating in train mode, or forgetting to restore it**: symptom: results change with dropout or batch norm, or training continues in eval mode. save `model.training` and restore it.
- **filling only the lower triangle of $R$**: symptom: forward transfer is meaningless (or zero). evaluate on every task after every task.
- **reusing one optimizer across tasks**: symptom: the fresh-optimizer check fails: a second call's Adam step uses the first call's moment estimates.

## going deeper

- McCloskey & Cohen (1989), "Catastrophic interference in connectionist networks: the sequential learning problem", *Psychology of Learning and Motivation* 24. the original observation. [doi:10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421(08)60536-8)
- French (1999), "Catastrophic forgetting in connectionist networks", *Trends in Cognitive Sciences* 3(4). a classic review of why distributed representations forget. [doi:10.1016/S1364-6613(99)01294-2](https://doi.org/10.1016/S1364-6613(99)01294-2)
- Goodfellow, Mirza, Xiao, Courville & Bengio (2013), "An empirical investigation of catastrophic forgetting in gradient-based neural networks". permuted MNIST and the effect of activations and dropout. [arXiv:1312.6211](https://arxiv.org/abs/1312.6211)
- Lopez-Paz & Ranzato (2017), "Gradient episodic memory for continual learning", *NeurIPS*. the accuracy matrix, ACC, BWT and FWT. [arXiv:1706.08840](https://arxiv.org/abs/1706.08840)
- Chaudhry, Dokania, Ajanthan & Torr (2018), "Riemannian walk for incremental learning: understanding forgetting and intransigence", *ECCV*. the forgetting measure. [arXiv:1801.10112](https://arxiv.org/abs/1801.10112)
- van de Ven & Tolias (2019), "Three scenarios for continual learning". task-, domain- and class-incremental learning, and why results don't transfer between them. [arXiv:1904.07734](https://arxiv.org/abs/1904.07734); the extended journal version is van de Ven, Tuytelaars & Tolias (2022), "Three types of incremental learning", *Nature Machine Intelligence* 4. [doi:10.1038/s42256-022-00568-3](https://doi.org/10.1038/s42256-022-00568-3)
- Mirzadeh, Farajtabar, Pascanu & Ghasemzadeh (2020), "Understanding the role of training regimes in continual learning", *NeurIPS*. learning rate, batch size and dropout as forgetting knobs. [arXiv:2006.06958](https://arxiv.org/abs/2006.06958)
- Parisi, Kemker, Part, Kanan & Wermter (2019), "Continual lifelong learning with neural networks: a review", *Neural Networks* 113. [arXiv:1802.07569](https://arxiv.org/abs/1802.07569)

## connections

- ← builds on: PyTorch training loops and `mlfs.torchnn.MLP` (ch 15), softmax cross-entropy and its gradient (ch 08), SGD and Adam (ch 06, ch 15), honest evaluation with seeds and baselines (ch 07).
- → used in: topic 13 plugs EWC and SI into `train_task`'s `penalty` and `after_step` hooks; topic 14 plugs replay buffers into `replay` and evaluates parameter isolation with `evaluate`; the elective-portfolio capstone reports the block's project with this harness. the scenarios reappear wherever models must keep learning: fine-tuning a pretrained model without losing its general abilities (ch 27) is continual learning with a single, very large first task.
