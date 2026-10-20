# prototypical networks

## why this matters

a face-recognition system at a building entrance gets one photo of each new employee. tomorrow it must recognize them. retraining a classifier every time someone joins is out of the question, and a classifier trained on one photo per class would memorize the photo anyway. what you want is a model that, given a handful of labelled examples of classes it has never seen, can classify new examples of those classes immediately, with no gradient steps at all.

MAML (topic 09) answers with an initialization that adapts in a few gradient steps. **prototypical networks** (Snell, Swersky & Zemel, 2017) answer with something much simpler: learn an embedding in which examples of the same class cluster together, then represent each new class by the mean of its few embedded examples, its **prototype**, and classify a new example by its nearest prototype. adaptation is an average. the whole difficulty moves into learning the embedding, and the trick for that is to train exactly the way you'll be tested: on **episodes**, small few-shot problems built from training classes, never from test classes.

that last sentence hides most of what goes wrong in practice. the lab will show you that how the episodes are built (which classes, how many, how they're split, what distance turns into a logit) moves accuracy by as much as changing the network, and that the variability of the *class split* can be larger than either. in this topic you build the episode sampler with its guarantees against leakage, the prototypes and their logits, the episodic loss, and an evaluation that reports the conventional 95% interval and knows what that interval leaves out.

## warm-up: predict first

1. with raw pixels as the embedding and nearest-prototype classification, what accuracy do you expect on 5-way 1-shot recognition of faces you've never seen (chance is 20%)?
2. a model is trained on 28 people and tested on 12 others. a second model also trains on *other photos* of the 12 test people, never on the test photos themselves. how much higher will its test accuracy be?
3. you will test 5-way. is it better to train with 5-way episodes or 20-way ones?
4. negative squared Euclidean distance to the prototypes, used as logits, gives a linear classifier in the embedding. true or false?
5. cosine similarity throws away embedding lengths. for faces, better or worse than squared distance?

## prerequisites check

- [ ] you can write softmax cross-entropy and its gradient (→ ch 08 · topic 05)
- [ ] you know nearest-neighbour and nearest-centroid classification (→ ch 09 · topic 01)
- [ ] you know why a test set must be independent of everything you trained or tuned on, and how to report a mean with an interval (→ ch 07 · topics 02 and 03, ch 05 · topic 03)
- [ ] you can train an `nn.Module` with Adam and know what `train()`/`eval()` change for batch norm and dropout (→ ch 15)
- [ ] you've built convolutional networks (→ ch 16; the lab uses a small one)
- [ ] you've seen MAML's split into tasks, support sets and query sets (→ ch 29 · topic 09)

## core ideas

### idea 1: episodes, and the split that makes them honest

few-shot classification is evaluated on **episodes**. an $N$-way $K$-shot episode picks $N$ classes, gives you $K$ labelled examples of each (the **support set**, $NK$ examples) and asks you to classify $Q$ further examples of each (the **query set**). labels inside an episode are just positions $0, \dots, N-1$: "class 3" in one episode has nothing to do with "class 3" in the next. accuracy is averaged over many episodes.

the honesty of the whole exercise rests on one rule: **the classes are split, not the examples**. a set of meta-training classes is used to learn the embedding; a disjoint set of meta-test classes provides every test episode. if any test class leaks into training, even through photos that never appear in a test episode, the model has learned that class and the test no longer measures learning from $K$ examples. the lab measures how large that error is.

Vinyals et al. (2016) added the second principle: **train as you test**. if the test is a stream of $N$-way $K$-shot episodes, train on a stream of episodes drawn from the training classes, so that the training objective is the test metric on simulated tasks. this is meta-learning in the same sense as MAML: an outer loop over tasks, each task a small learning problem.

the sampler has to get several details exactly right. classes are drawn uniformly *without replacement*, so an episode never contains the same class twice; their order is random, so label 0 doesn't systematically mean the lowest class id; within each chosen class, $K + Q$ distinct examples are drawn uniformly, the first $K$ for support and the rest for query, so support and query never share an example. returning *indices* rather than images keeps the sampler independent of the data and makes leakage easy to audit.

> **spec · stage 1.** `split_classes(classes, n_test, *, generator=None)`: from the distinct ids in `classes` (any iterable of integers, duplicates allowed), returns `(train, test)`, two sorted tuples of Python ints that partition them, with the `n_test` test classes chosen uniformly at random using `generator`; `ValueError` unless $0 <$ `n_test` $<$ number of classes. `sample_episode(y, classes, n_way, k_shot, n_query, *, generator=None)`: `y` is the label array (NumPy or tensor) of the whole dataset; draws `n_way` distinct classes uniformly from `classes`, in random order, and for each $K + Q$ distinct examples uniformly; returns a dict of int64 tensors `support (N*K,)` and `query (N*Q,)` (indices into `y`, class-major: the $j$-th block belongs to the $j$-th class), `support_labels` and `query_labels` (the class's position $j$), and `classes (N,)` (the original ids in episode order). `ValueError` if `n_way > len(classes)` or any class in `classes` has fewer than $K + Q$ examples.

> **▶ build: stage 1.** implement `split_classes` and `sample_episode` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** suppose your sampler returned the episode's classes sorted by id instead of in random order, so label 0 is always the smallest class id in the episode. could a model exploit that during training? would it change the test accuracy you report?

### idea 2: prototypes, and distances as logits

with an embedding $f_\phi$, the prototype of class $j$ is the mean of its embedded support examples,

$$c_j = \frac{1}{|S_j|} \sum_{x \in S_j} f_\phi(x),$$

and a query $x$ gets a distribution over the episode's classes from a softmax over negative squared distances:

$$p_\phi(y = j \mid x) = \frac{\exp(-d(f_\phi(x), c_j))}{\sum_{j'} \exp(-d(f_\phi(x), c_{j'}))}, \qquad d(z, c) = \lVert z - c \rVert^2.$$

why the mean, and why squared Euclidean distance? the mean is the point that minimizes the summed squared distance to the class's points (on paper 1), so prototype and distance are consistent; Snell et al. note that this holds for every Bregman divergence, of which the squared Euclidean distance is one, and that it makes the model equivalent to a mixture density estimate with one exponential-family component per class. expanding the square shows something more practical:

$$-\lVert z - c_j\rVert^2 = 2 c_j^\top z - \lVert c_j\rVert^2 - \lVert z\rVert^2.$$

the last term is the same for every class, so the softmax ignores it: a prototypical network is a **linear classifier** on the embedding, with weights $2c_j$ and biases $-\lVert c_j\rVert^2$ computed from the support set instead of learned (warm-up 4). adaptation to a new episode is "fit a linear head" in closed form.

a natural alternative is **cosine similarity**, $\cos(z, c_j)$, which ignores the embedding's length. it has a trap: cosines live in $[-1, 1]$, and so do the logits, which caps how confident the softmax can be (on paper 4). multiplying by a **scale** (a temperature) removes the cap; Oreshkin et al. (2018) showed the scale matters as much as the choice of metric. the spec has both.

> **spec · stage 2.** `prototypes(z, labels, n_way)`: `z (S, D)` support embeddings, `labels (S,)` in $0..N-1$ in any order and with any counts; returns `(N, D)`, row $j$ the mean of the rows with label $j$, differentiable; `ValueError` if a class has no example. `proto_logits(zq, protos, *, metric="sqeuclidean", scale=1.0)`: `(Q, N)` logits, $-\text{scale}\cdot\lVert z_q - c_j\rVert^2$ for `"sqeuclidean"` and $\text{scale}\cdot\cos(z_q, c_j)$ for `"cosine"`; other metrics: `ValueError`.

> **▶ build: stage 2.** implement `prototypes` and `proto_logits`.

### idea 3: the episodic loss

one episode gives one loss: embed support and query, form prototypes from the support embeddings, compute the query logits, and take the cross-entropy against the query labels. the gradient does two things at once. it pulls each query embedding towards its own prototype and pushes it away from the others, weighted by how much probability they took (on paper 2); and because prototypes are means of support embeddings, it moves the support embeddings too, shaping the space so that *any* class's examples cluster, not just these $N$.

one implementation detail matters with batch normalization: embed support and query **in one call**, so that both are normalized with the same batch statistics in training mode. two separate calls normalize the support set by its own statistics and the query set by its own, which makes their embeddings subtly incomparable. (at test time the network is in eval mode and batch norm uses its running statistics, so this no longer matters; forgetting `eval()` makes test predictions depend on which other examples share the batch.)

> **spec · stage 3.** `episode_loss(embed, X, episode, *, metric="sqeuclidean", scale=1.0)`: embeds `X[support]` followed by `X[query]` in **one** call to `embed`, computes prototypes from the support embeddings (`n_way = len(episode["classes"])`) and query logits with `proto_logits(..., metric, scale)`, and returns `(loss, accuracy)`: the mean cross-entropy against `query_labels` as a scalar tensor, and the fraction of queries whose highest logit is their label, as a Python float.

> **▶ build: stage 3.** implement `episode_loss`. the worked example below is one of its checks.

> **🔮 predict:** in the worked example, both queries are classified correctly. which of the two contributes more to the loss, and by roughly what factor?

### idea 4: episodic training, and what an interval covers

training is a loop over freshly sampled episodes from the meta-training classes, with an optimizer step on each episode loss. evaluation is a loop over episodes from the meta-test classes, in eval mode and without gradients, recording each episode's query accuracy. the field's convention, from Vinyals et al. onwards, is to report the mean over 600 (or more) test episodes with a 95% interval, $\bar a \pm 1.96\, s / \sqrt{n}$, where $s$ is the standard deviation of the per-episode accuracies.

know what that interval covers: the randomness of **which episodes** were drawn, nothing else. it says nothing about the training seed, and nothing about **which classes** ended up in the test split, and with a few dozen classes the split can dominate. in the lab, the same raw-pixel baseline scores 0.71 on one split of the Olivetti faces and 0.82 on another, a gap ten times wider than the per-run interval. the remedy is the one from chapter 07: repeat over splits and seeds, and compare methods **paired on the same split**, so that a split's difficulty cancels out of the difference.

the training recipe has its own design choices, all of them about episodes: the number of ways and shots used in training (Snell et al. found that more ways than at test time helped them, and that matching the test's shot count matters), the number of distinct training classes, and the metric and scale. chapter 07's lesson applies: change one thing at a time and keep everything else, including the test episodes, fixed.

> **spec · stage 4.** `train_protonet(embed, X, y, classes, *, n_way, k_shot, n_query, n_episodes, lr=1e-3, metric="sqeuclidean", scale=1.0, seed=0)`: puts `embed` in train mode, creates `torch.optim.Adam(embed.parameters(), lr=lr)` and one `torch.Generator` seeded with `seed`, and for each of `n_episodes` episodes draws `sample_episode(y, classes, n_way, k_shot, n_query, generator=...)` and takes one step on `episode_loss` (with `metric`, `scale`); returns the list of losses (floats). `evaluate_protonet(embed, X, y, classes, *, n_way, k_shot, n_query=15, n_episodes=600, metric="sqeuclidean", scale=1.0, seed=0)`: eval mode, no gradients, episodes from one generator seeded with `seed`; returns `(mean accuracy, 1.96 * sd / sqrt(n_episodes))` with the sample standard deviation (`ddof=1`), both floats.

> **▶ build: stage 4.** implement `train_protonet` and `evaluate_protonet`. the last check trains a linear embedding that must learn to ignore ten noisy dimensions, on 16 classes, and tests it on 8 others.

## worked example

the identity embedding in 2-D, a 2-way 2-shot episode. support: class 0 at $(0, 0)$ and $(2, 0)$, class 1 at $(0, 2)$ and $(0, 4)$. queries: $(1, 1)$ with label 0, and $(1, 3)$ with label 1.

- **prototypes.** $c_0 = (1, 0)$, $c_1 = (0, 3)$.
- **query $(1, 1)$.** squared distances $0^2 + 1^2 = 1$ to $c_0$ and $1^2 + 2^2 = 5$ to $c_1$; logits $(-1, -5)$; $p(0) = 1/(1 + e^{-4}) = 0.98201$; loss $\log(1 + e^{-4}) = 0.018150$.
- **query $(1, 3)$.** squared distances $(1-1)^2 + (3-0)^2 = 9$ to $c_0$ and $(1-0)^2 + (3-3)^2 = 1$ to $c_1$; logits $(-9, -1)$; loss $\log(1 + e^{-8}) = 0.000335$.
- **episode.** loss $(0.018150 + 0.000335)/2 = 0.009243$, accuracy 1.0.

the first query contributes 54 times as much as the second: its margin, the difference of squared distances, is 4, against 8 for the second, and the loss falls exponentially in the margin. that's the softmax's way of spending gradient where the classification is least certain. your `episode_loss` with `nn.Identity()` must give $0.009243$ and accuracy $1.0$.

## on paper

1. show that the mean minimizes $\sum_{x \in S} \lVert z_x - c\rVert^2$ over $c$, and that the minimum is $|S|$ times the average squared distance to the mean. why does this make the class mean the natural prototype for squared-distance logits?

<details><summary>solution</summary>

the gradient with respect to $c$ is $-2\sum_x (z_x - c)$, zero at $c = \frac{1}{|S|}\sum_x z_x$, and the objective is convex. plugging in gives $\sum_x \lVert z_x - \bar z\rVert^2$. the logit $-\lVert z - c_j\rVert^2$ is the log-density, up to constants, of a unit-variance Gaussian centred at $c_j$; the maximum-likelihood centre of that Gaussian from the support points is exactly their mean, so the classifier is a Gaussian class-conditional model fitted to the support set, with a shared, fixed covariance.

</details>

2. differentiate the query loss $\ell = -\log p(y \mid z)$ with squared-distance logits with respect to the query embedding $z$. interpret the result.

<details><summary>solution</summary>

with $\ell = \lVert z - c_y\rVert^2 + \log\sum_j \exp(-\lVert z - c_j\rVert^2)$, $\nabla_z \ell = 2(z - c_y) - 2\sum_j p_j (z - c_j) = 2\big(\sum_j p_j c_j - c_y\big)$. the gradient step $-\nabla_z\ell$ moves $z$ towards $c_y$ and away from the probability-weighted average of all prototypes. when $p_y \to 1$ the two cancel and the gradient vanishes: confidently correct queries stop contributing, which is the margin effect of the worked example.

</details>

3. a learned linear embedding $f(x) = Ax$. show that the logits depend on $A$ only through $M = A^\top A$, and say what that means for which input directions the embedding can ignore.

<details><summary>solution</summary>

$\lVert Ax - Ac_j^x\rVert^2 = (x - c^x_j)^\top A^\top A (x - c^x_j)$ with $c^x_j$ the class mean in input space (the mean commutes with a linear map). so training learns a Mahalanobis metric $M \succeq 0$. a direction $v$ with $Av = 0$ contributes nothing to any distance: if it carries only noise, zeroing it removes the noise from every comparison. that's what the last stage-4 check rewards, and why it transfers to held-out classes, which share the noise directions.

</details>

4. with cosine logits $s\cos(\cdot)$ in $[-s, s]$ and $N$ classes, what's the smallest possible cross-entropy of one query? evaluate it for $N = 20$ and $s = 1$, and for $s = 10$.

<details><summary>solution</summary>

the best case puts the true class's logit at $s$ and all others at $-s$: $\ell_{\min} = -\log\frac{e^{s}}{e^{s} + (N-1)e^{-s}} = \log(1 + (N-1)e^{-2s})$. for $N = 20$, $s = 1$: $\log(1 + 19 e^{-2}) = 1.27$, a large loss even for a perfect embedding, so every query keeps a large gradient forever, correct or not, and training never focuses on the hard ones. for $s = 10$: $\log(1 + 19e^{-20}) \approx 4 \times 10^{-8}$. squared distances are unbounded, so they have no such floor.

</details>

## experiment

the lab trains your prototypical networks on the Olivetti faces (40 people, 10 photos each), split by person, on three splits. first, raw-pixel prototypes against a learned embedding, and against a model that cheated by seeing other photos of the test people. second, three changes to the recipe, against the same baseline on the same split: fewer ways, fewer training people, a convnet instead of an MLP. third, cosine logits with and without a scale. predict before each.

## check your understanding

1. why does `sample_episode` return indices rather than images?

<details><summary>answer</summary>

the same episode can be replayed through different embeddings (raw pixels, an MLP, a convnet) for a paired comparison, the sampler doesn't care what the data looks like, and leakage is easy to audit: every index can be checked against the allowed classes.

</details>

2. what would go wrong if episode labels were the original class ids instead of positions $0..N-1$?

<details><summary>answer</summary>

the logits are over the episode's $N$ prototypes, so labels must index them. with original ids the cross-entropy would index out of range (or the wrong prototype). more subtly, a fixed id-to-output mapping would let the model learn class identities across episodes, which is ordinary classification, not few-shot learning; random relabelling per episode forces it to use the support set.

</details>

3. the conventional 95% interval over 600 test episodes is $\pm 0.01$ in the lab. two methods differ by 0.03 on one split. is the difference real?

<details><summary>answer</summary>

the interval covers episode sampling only. a different training seed or a different class split can move each method by more than 0.03, so a single-split comparison can't settle it. evaluate both on the same few splits and seeds and look at the paired differences: if they agree in sign and exceed their spread, the difference is credible (chapter 05's sign-flip test makes this exact).

</details>

4. why is a prototypical network with squared distance equivalent to a linear classifier, and where do its weights come from?

<details><summary>answer</summary>

$-\lVert z - c_j\rVert^2 = 2c_j^\top z - \lVert c_j\rVert^2 - \lVert z\rVert^2$, and the last term is shared by all classes, so it drops out of the softmax. the weights $2c_j$ and biases $-\lVert c_j\rVert^2$ are computed from the support set: the "training" of the head on a new episode is an average.

</details>

5. a colleague evaluates with the network in train mode because "batch norm works better with batch statistics". what's wrong?

<details><summary>answer</summary>

in train mode each query's embedding depends on the other examples in the batch (and the running statistics get updated by test data). predictions become a function of the whole query set, which is a transductive setting with information shared across queries; results aren't comparable with the standard, inductive protocol, and they change with the batch composition.

</details>

6. MAML adapts with gradient steps, prototypical networks with an average. when would you expect the average to be the worse choice?

<details><summary>answer</summary>

when new tasks need the representation itself to change, not just a new linear head on a fixed embedding: test classes from a different domain than the training classes (Chen et al., 2019, find that fine-tuning helps under domain shift), or regression tasks like topic 09's sines, where there are no classes to average. when test classes resemble training classes, a good embedding plus a mean is very hard to beat (Tian et al., 2020).

</details>

## common bugs

- **reshaping instead of grouping**: `z.reshape(N, K, D).mean(1)` assumes class-major support with equal shots. symptom: the shuffled-labels check fails. group by label.
- **distance, not squared distance**: `-torch.cdist(zq, protos)`. symptom: the logit check is off; training still works, a little worse. square it.
- **test classes in training**: sampling training episodes from all classes, or splitting photos instead of classes. symptom: inflated test accuracy. the lab's leaky model never sees a photo its test episodes use, only 4 *other* photos of each test person, and still scores 0.93 against the honest model's 0.78.
- **support and query sharing an example**: drawing support and query separately from the same class. symptom: the disjointness check fails; accuracy is inflated because some queries are their own prototype.
- **two embedding calls with batch norm**: symptom: the one-call check fails; slightly worse training with BN models.
- **cosine logits without a scale**: symptom: training loss plateaus above 1 with 20-way episodes and accuracy is worse than raw pixels.

## going deeper

- Snell, Swersky & Zemel (2017), "Prototypical networks for few-shot learning", *NeurIPS*. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
- Vinyals, Blundell, Lillicrap, Kavukcuoglu & Wierstra (2016), "Matching networks for one shot learning", *NeurIPS*. episodic training, and the Conv-4 embedding. [arXiv:1606.04080](https://arxiv.org/abs/1606.04080)
- Oreshkin, Rodríguez López & Lacoste (2018), "TADAM: task dependent adaptive metric for improved few-shot learning", *NeurIPS*. metric scaling. [arXiv:1805.10123](https://arxiv.org/abs/1805.10123)
- Chen, Liu, Kira, Wang & Huang (2019), "A closer look at few-shot classification", *ICLR*. simple baselines, and how much evaluation details matter. [arXiv:1904.04232](https://arxiv.org/abs/1904.04232)
- Tian, Wang, Krishnan, Tenenbaum & Isola (2020), "Rethinking few-shot image classification: a good embedding is all you need?", *ECCV*. [arXiv:2003.11539](https://arxiv.org/abs/2003.11539)
- Triantafillou et al. (2020), "Meta-Dataset: a dataset of datasets for learning to learn from few examples", *ICLR*. evaluation across domains. [arXiv:1903.03096](https://arxiv.org/abs/1903.03096)
- Banerjee, Merugu, Dhillon & Ghosh (2005), "Clustering with Bregman divergences", *JMLR* 6. why the mean is the right centre. [jmlr.org/papers/v6/banerjee05b](https://www.jmlr.org/papers/v6/banerjee05b.html)
- Samaria & Harter (1994), "Parameterisation of a stochastic model for human face identification", *IEEE Workshop on Applications of Computer Vision*. the Olivetti (ORL) faces. [doi:10.1109/ACV.1994.341300](https://doi.org/10.1109/ACV.1994.341300)

## connections

- ← builds on: softmax cross-entropy (ch 08), nearest-centroid classification (ch 09), honest splits and intervals (ch 05, ch 07), PyTorch training and batch norm (ch 15), convnets (ch 16), topic 09's tasks with support and query sets.
- → used in: topic 11 takes "adapt without gradient steps" further, to a transformer that reads the support set as its input; the elective-portfolio capstone's paired, per-split comparisons are the habit experiment 2 builds.
