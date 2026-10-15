# evaluating representations

## why this matters

this chapter trains networks *without labels*. a SimCLR model learns to say "these two crops come from the same photo"; an MAE learns to fill in hidden patches. neither objective is the thing you care about. the loss going down tells you the network got better at its pretext task, and nothing about whether it learned anything useful for classifying, retrieving or segmenting images. a network can drive a self-supervised loss to zero by learning something useless (topic 03 shows one that does, by mapping every image to the same vector).

so before building any self-supervised method you need a way to *score a representation*: a fixed, cheap, honest procedure that takes an encoder $f$ and returns a number that tracks how useful $f(x)$ is for downstream tasks. the field has settled on three: train a **linear classifier** on frozen features, run a **k-nearest-neighbour** classifier on them, or **fine-tune** the whole network with a fresh head. they measure different things, they disagree with each other, and the choice changes which method "wins" a comparison. you'll build all three in this topic, and every later topic (and the chapter capstone) uses them as its yardstick.

## warm-up: predict first

1. you train a linear classifier on the 512 output features of a network that has never been trained (random weights). on a 5-class digit problem with 1,000 labels, do you expect chance (20%), something mediocre, or something good?
2. two encoders produce features $f(x)$ and $10\,f(x)$. does a linear probe score them the same? does kNN?
3. kNN on features from a supervised network and a linear probe on the same features: which do you expect to score higher, and why might they differ at all?
4. you have 25 labelled examples and a pretrained network. is it better to train only a new linear head, or to fine-tune everything?
5. why might "accuracy of a linear probe" reward a different kind of representation than "accuracy of kNN"?

## prerequisites check

- [ ] you can fit and use `SoftmaxRegression`, and know why features are standardised before gradient descent (→ ch 08 · topic 05, ch 06)
- [ ] you know k-nearest neighbours and why distance choices matter (→ ch 09 · topic 01)
- [ ] you can write a PyTorch training loop with `train()`/`eval()` and `torch.no_grad()` (→ ch 15 · topic 07)
- [ ] you know what BatchNorm's running statistics are and when they update (→ ch 15 · topic 02)
- [ ] you've used `replace_head`/`freeze_backbone` for transfer learning (→ ch 16 · topic 05)

## core ideas

### idea 1: a representation is what the encoder outputs, in eval mode

an **encoder** is a network $f_\theta: \mathcal X \to \mathbb R^d$; its output $h = f_\theta(x)$ is the **representation** (or embedding, or feature vector) of $x$. for a ResNet it's the pooled vector before the classification head (`forward_features`); for a ViT, the [CLS] token or the mean of the output tokens. self-supervised methods usually bolt a small **projection head** on top during training and throw it away afterwards; you evaluate the encoder's output, not the head's (topic 02 explains why).

evaluating a representation starts by computing it for a whole dataset, once. three details are easy to get wrong:

- **eval mode.** BatchNorm in train mode normalises each batch with its own statistics, so a sample's features depend on which other samples share its batch, and every forward pass also *updates* the running statistics. dropout zeroes random units. neither should happen while you measure. `encoder.eval()` fixes both.
- **no autograd graph.** you'll never backpropagate through these features, so run the encoder under `torch.no_grad()`: it's faster and keeps memory flat over tens of thousands of images.
- **restore what you changed.** the caller may be in the middle of training, or may have deliberately frozen some BatchNorm layers inside a model that is otherwise training. calling `encoder.train()` afterwards would un-freeze them. record every submodule's mode and put each one back.

> **spec.** `extract_features(encoder, loader, *, device='cpu')`: `loader` yields `(x, y)` batches. for each batch, move `x` to `device`, run `encoder(x)` in eval mode under `torch.no_grad()`, and flatten the output to `(B, d)`. return `(feats, labels)`: `feats` a float32 ndarray `(n, d)` and `labels` an ndarray `(n,)`, both in loader order, on the CPU. afterwards every submodule's `training` flag must be exactly what it was before. the encoder must already be on `device`.

> **▶ build: stage 1.** implement `extract_features`.

### idea 2: the linear probe

the most common score: freeze the encoder, compute features for a labelled training set, fit a **linear classifier** (multinomial logistic regression) on them, and report its accuracy on a test set. it asks a precise question: *is the information needed for this task present in the features in a linearly accessible form?* a linear model can only draw hyperplanes, so if the classes are separable by a hyperplane in feature space, the encoder has done the work. (Alain & Bengio, 2016, introduced the name "probe" for this.)

two implementation details matter more than they look:

**standardise, with training statistics.** subtract each feature's training mean and divide by its training standard deviation, then apply the *same* transform to the test features. without it, a feature on a scale of $10^4$ next to one on a scale of $10^{-3}$ makes gradient descent with a fixed learning rate either diverge or crawl (ch 06), and the L2 penalty $\frac\alpha2\|W\|^2$ punishes features unequally. computing the test set's own mean and std is a leak, and it also breaks the classifier when the test distribution differs from the training one (the checks build such a test set). a feature with zero training variance (a dead ReLU unit, common in real encoders) gets divided by 1, not 0.

**the probe is a model with hyperparameters.** the regularisation $\alpha$, the number of iterations, the optimiser: all of them change the number you report. the convention in papers is to tune them on a validation split; for comparisons within this chapter we fix them (`alpha=1e-4`, `max_iters=1000`) and use the same probe for every encoder, so differences come from the features.

> **spec.** `linear_probe(train_feats, train_labels, test_feats, test_labels, *, alpha=1e-4, max_iters=1000)`: standardise both feature sets with the training mean and std (a zero std becomes 1), fit `mlfs.linear.SoftmaxRegression(alpha=alpha, max_iters=max_iters)` on the training set, and return the test accuracy as a Python `float`. labels may be any sortable values.

> **🔮 predict:** you probe the features of a ResNet that has never been trained. you double every weight of its final layer, which multiplies the features by 2. does the probe's accuracy change? does kNN's?

> **▶ build: stage 2.** implement `linear_probe`.

### idea 3: kNN evaluation

the probe has to be trained, which takes time and introduces hyperparameters. **kNN evaluation** has no training at all: label each test point by a vote among its $k$ most similar training points. it asks a different question from the probe: *are images of the same class close together in feature space?* that's a statement about the geometry of the whole space, not just about the existence of a separating hyperplane, and it's what retrieval ("find images like this one") depends on.

self-supervised learning has a standard recipe, from Wu et al. (2018): L2-normalise the features, measure similarity by the **cosine** $s(u, v) = u^\top v / (\|u\|\|v\|)$, take the $k$ most similar training points (DINO uses $k = 20$), and weight each neighbour's vote by $\exp(s/T)$ with a temperature $T$ (0.07 is common):

$$\text{score}(c) = \sum_{j \in \mathcal N_k(x)} \mathbb 1[y_j = c]\, \exp\!\big(s(x, x_j)/T\big), \qquad \hat y = \arg\max_c \text{score}(c).$$

why cosine: contrastive methods (topics 02 and 05) train *on the unit sphere*, so only directions carry information; a feature's norm often encodes nuisance like image contrast. why the exponential weights: a plain majority vote ignores how close each neighbour is. with $T \to 0$ the closest neighbour dominates (1-NN); with $T \to \infty$ every weight is about 1 and you get a majority vote over $k$ (on paper 2). $T = 0.07$ sits much nearer the 1-NN end: a similarity gap of 0.07 is a factor $e$ in weight.

two practical points. $\exp(1/0.001)$ overflows float64, so subtract each query's largest similarity before exponentiating; the votes all scale by the same factor and the argmax doesn't change. and on CIFAR-10 the full similarity matrix is $10{,}000 \times 50{,}000$ floats, 2 GB in float32: process the test points in chunks.

> **spec.** `knn_eval(train_feats, train_labels, test_feats, test_labels, *, k=20, temperature=0.07)`: L2-normalise both sets; for each test point, take the $\min(k, n_\text{train})$ training points of highest cosine similarity and add $\exp(s/T)$ to each neighbour's class; predict the class with the largest total (ties: the smallest label). return the test accuracy as a `float`. stable for temperatures down to $10^{-3}$; memory must stay bounded (never build the full `(n_test, n_train)` matrix at once). labels may be any sortable values.

> **▶ build: stage 3.** implement `knn_eval`.

### idea 4: transfer by fine-tuning

the third protocol uses the encoder the way you would in practice: put a fresh linear head on top and train **the whole network** on the labelled downstream data, starting from the pretrained weights. this is **fine-tuning**, and it's how a pretrained model is usually deployed (ch 16 · topic 05). it scores a representation by how good a *starting point* it is, rather than how good it is as is. a representation that needs a small nonlinear adjustment to separate the classes scores badly with a probe and well with fine-tuning.

fine-tuning has a cost the frozen protocols don't: with few labels, updating millions of weights can overfit, or wander away from the pretrained solution (catastrophic forgetting, if the downstream data is narrow). freezing the encoder and training only the head is the safe extreme; it's a linear probe trained with SGD instead of a convex solver. one detail matters there: a frozen encoder must also stay in **eval mode**, or its BatchNorm statistics drift towards the new data even though no weight is trained.

the protocols answer different questions, so serious papers report several. the SimCLR paper (Chen et al., 2020) reports a linear probe on ImageNet, fine-tuning with 1% and 10% of the labels, and transfer to 12 other datasets; DINO (Caron et al., 2021) adds kNN.

> **spec.** `fine_tune(encoder, feature_dim, n_classes, loader, *, epochs=1, lr=1e-3, freeze=False, device='cpu', seed=None)` returns `nn.Sequential(encoder, head)` with `head = nn.Linear(feature_dim, n_classes)`, created after `torch.manual_seed(seed)` if `seed` is not `None` (don't disturb the caller's global RNG state). move the model to `device`; train with Adam (`lr`) on cross-entropy for `epochs` passes over `loader`'s `(x, y)` batches. `freeze=False`: every parameter trains, the whole model in train mode (even if the encoder arrived in eval mode). `freeze=True`: only the head trains; the encoder stays in eval mode and its parameters and buffers are bit-for-bit unchanged. return the model in eval mode.

> **🔮 predict:** with 5 labelled examples per class, does fine-tuning a pretrained network beat a linear probe on its frozen features? and with 500 per class?

> **▶ build: stage 4.** implement `fine_tune`.

## worked example

**kNN by hand.** four training points on the unit circle, $a = (1, 0)$ and $d = (-1, 0)$ of class 0, $b = (0.8, 0.6)$ and $c = (0.6, 0.8)$ of class 1, and a query $q = (0.96, 0.28)$ (also unit length). the cosine similarities are $q^\top a = 0.96$, $q^\top b = 0.936$, $q^\top c = 0.8$, $q^\top d = -0.96$.

| k | T | class 0 score | class 1 score | prediction |
|---|---|---|---|---|
| 1 | 0.1 | $e^{9.6} = 14{,}765$ | 0 | 0 |
| 3 | 0.1 | $e^{9.6} = 14{,}765$ | $e^{9.36} + e^{8} = 11{,}614 + 2{,}981 = 14{,}595$ | 0 (barely) |
| 3 | 1 | $e^{0.96} = 2.61$ | $e^{0.936} + e^{0.8} = 2.55 + 2.23 = 4.78$ | 1 |
| 4 | 10 | $e^{0.096} + e^{-0.096} = 2.009$ | $e^{0.0936} + e^{0.08} = 2.181$ | 1 |

at $T = 0.1$ the single closest point almost outvotes two slightly farther ones; at $T = 1$ the vote is close to a count. adding the far-away $d$ at $k = 4$ barely matters at $T = 10$, because the weights are nearly uniform and $d$'s vote is $e^{-0.096} \approx 0.91$, almost a full vote for class 0, but class 1's two votes still win.

**standardisation.** a training set with one feature $x \in \{0, 2, 4\}$ has mean 2 and std $\sqrt{8/3} = 1.633$, so it becomes $\{-1.22, 0, 1.22\}$. a test point $x = 6$ becomes $(6 - 2)/1.633 = 2.45$, with the *training* statistics. a test set made of the single point $6$ would, with its own statistics, become $0$: in the middle of the training range, wherever the decision boundary is.

your code must reproduce these numbers (the checks use the kNN table).

## on paper

1. **invariances of the probe.** let $h' = Ah + b$ with $A$ invertible. show that an *unregularised* linear classifier on $h'$ can represent exactly the same decision functions as one on $h$. why does this fail with an L2 penalty, and which family of $A$ does standardisation make harmless?

<details><summary>solution</summary>

a linear classifier computes logits $W^\top h + c$. on $h'$, $W'^\top(Ah + b) + c' = (A^\top W')^\top h + (W'^\top b + c')$, and any $(W, c)$ is reached by $W' = A^{-\top}W$, $c' = c - W'^\top b$. so the set of decision functions is the same, and a converged unregularised fit finds an equally good one. with the penalty $\frac\alpha2\|W'\|^2$ the optimum depends on $A$: a feature that $A$ shrinks needs a larger weight, which costs more. standardisation maps $h \mapsto (h - \mu)/\sigma$ elementwise, so any per-feature scaling and shift ($A$ diagonal with positive entries) is undone before the probe sees the data: the regularised probe is invariant to it. a rotation is not undone; for kNN with cosine similarity a rotation is harmless, but a per-feature scaling is not.

</details>

2. **the two limits of the temperature.** fix $k$ and a query whose neighbours have distinct similarities $s_1 > s_2 > \dots > s_k$. show that as $T \to 0$ the weighted vote predicts $y_1$, and as $T \to \infty$ it predicts the majority label among the $k$ (assuming no tie in the counts).

<details><summary>solution</summary>

divide every weight by $e^{s_1/T}$ (it doesn't change the argmax): the weights become $e^{(s_j - s_1)/T}$, which is 1 for $j = 1$ and $\to 0$ for $j > 1$ as $T \to 0$, so class $y_1$ scores about 1 and every other class about 0. as $T \to \infty$, $e^{s_j/T} = 1 + s_j/T + O(T^{-2}) \to 1$, so each class scores its count plus $O(1/T)$: for large enough $T$ the count decides. the cases in the checks sit between the limits, where both closeness and numbers matter.

</details>

3. **cosine versus Euclidean.** for unit vectors $u, v$, show $\|u - v\|^2 = 2 - 2u^\top v$. conclude that after L2 normalisation, "the $k$ nearest by Euclidean distance" and "the $k$ most similar by cosine" are the same set. what goes wrong if you normalise the training features but forget the test features?

<details><summary>solution</summary>

$\|u - v\|^2 = \|u\|^2 + \|v\|^2 - 2u^\top v = 2 - 2u^\top v$, a decreasing function of $u^\top v$, so the rankings agree. if only the training features are normalised, $q^\top x_j = \|q\|\cos(q, x_j)$: the ranking of neighbours is still by cosine (the factor $\|q\|$ is shared), but the weights become $\exp(\|q\|\cos/T)$, so a query with a large norm acts as if it had a much smaller temperature. the effective temperature then varies from query to query.

</details>

## experiment

`lab.py` builds a small transfer problem from MNIST: a CNN trained with labels on the digits 0–4 is evaluated on the unseen digits 5–9, with your four functions.

1. how good are the features of a network that has never been trained, compared with raw pixels and with the pretrained network?
2. linear probe, kNN, fine-tuning and training from scratch, with 5 to 500 labels per class.

## check your understanding

1. why must the probe standardise the test features with the training statistics, even when train and test come from the same distribution?

<details><summary>answer</summary>

the test set's statistics are information the classifier wouldn't have at deployment, where examples arrive one at a time. even from the same distribution they differ by sampling noise, so the decision boundary shifts slightly from what was fitted; with small or skewed test sets (one class only, in the checks) the shift is large.

</details>

2. a self-supervised paper reports 92% kNN accuracy and 85% linear-probe accuracy for the same features. is that a contradiction?

<details><summary>answer</summary>

no. they measure different things: kNN rewards local neighbourhoods being class-pure, which can hold even when classes form many small clusters that no single hyperplane separates (kNN is a nonlinear classifier). the probe can also be held back by its optimiser and regularisation. the reverse ordering is just as possible: in the lab, raw MNIST pixels score higher with kNN than with the probe.

</details>

3. why does BatchNorm make "eval mode during extraction" essential, while for a network with only LayerNorm it would merely be good hygiene?

<details><summary>answer</summary>

BatchNorm in train mode normalises with the current batch's statistics, so a sample's features depend on its batch mates (and the batch size), and every forward pass updates the running statistics. LayerNorm normalises each sample on its own, so its output doesn't depend on mode; only dropout (if any) would differ.

</details>

4. a frozen-encoder fine-tune leaves the encoder in train mode. no weight changes, yet test accuracy is different from a run with the encoder in eval mode. why?

<details><summary>answer</summary>

in train mode BatchNorm updates its running mean and variance with every batch of the downstream data. at test time, in eval mode, the encoder then normalises with statistics of the downstream training set instead of the pretraining data, so the features the head was trained on (computed with per-batch statistics) and the features it sees at test time differ, and the encoder has silently changed.

</details>

5. in the kNN vote, what does $T = 0.07$ mean in words?

<details><summary>answer</summary>

a neighbour whose cosine similarity is 0.07 higher counts $e \approx 2.7$ times as much; 0.21 higher, 20 times as much. neighbours that are only a little less similar still contribute, but the closest few dominate.

</details>

6. why fix the probe's hyperparameters when comparing encoders, instead of tuning them for each one?

<details><summary>answer</summary>

tuning per encoder is legitimate if it's done on a validation split, but it costs compute and adds noise; with a fixed protocol every encoder gets the same evaluation, so differences are attributable to the features. what you must never do is tune on the test set, which inflates the score of whichever encoder you tried hardest on.

</details>

7. why might a linear probe *under*estimate how useful a representation is in practice?

<details><summary>answer</summary>

in practice people fine-tune. a representation that holds the right information in a nonlinear form, or needs small adjustments to its early layers for the new domain, can fine-tune very well while scoring modestly with a probe. MAE (topic 04) is the famous example: weak linear probes, excellent fine-tuning.

</details>

## common bugs

- **features computed in train mode.** accuracy changes with the batch size of the extraction loader, and running statistics silently move. symptom: evaluating twice gives two different numbers.
- **standardising with test statistics.** looks fine on balanced, i.i.d. test sets; collapses on a shifted or single-class test set.
- **dividing by a zero std.** a dead unit produces `nan` features and a `nan` (or 0) accuracy.
- **kNN without normalisation, or on one side only.** the "nearest" neighbours become the training points with the largest norms; the effective temperature changes per query.
- **`exp(sim / T)` overflowing.** with $T = 10^{-3}$, every weight is `inf` and the vote is `nan`.
- **a frozen encoder left in train mode.** no weight changes, but the BatchNorm statistics do, and accuracy shifts.

## going deeper

- Alain & Bengio (2016), *Understanding intermediate layers using linear classifier probes*. [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
- Wu, Xiong, Yu & Lin (2018), *Unsupervised feature learning via non-parametric instance discrimination*: the weighted kNN protocol. [arXiv:1805.01978](https://arxiv.org/abs/1805.01978)
- Kolesnikov, Zhai & Beyer (2019), *Revisiting self-supervised visual representation learning*: how the evaluation protocol changes the ranking of methods. [arXiv:1901.09005](https://arxiv.org/abs/1901.09005)
- Kornblith, Shlens & Le (2019), *Do better ImageNet models transfer better?*: linear probes versus fine-tuning across 12 datasets. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
- Caron et al. (2021), *Emerging properties in self-supervised vision transformers* (DINO): kNN with $k = 20$, $T = 0.07$. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- on random features: Saxe et al. (2011), *On random weights and unsupervised feature learning*. [ICML 2011](https://icml.cc/2011/papers/551_icmlpaper.pdf)

## connections

- ← builds on: softmax regression (ch 08 · topic 05), kNN (ch 09 · topic 01), BatchNorm and train/eval mode (ch 15), transfer learning helpers (ch 16 · topic 05).
- → used in: every later topic of this chapter scores its encoder with `knn_eval` and `linear_probe`; the capstone's label-efficiency study is built on all four protocols; chapter 28 reuses the harness when it probes what networks represent (ch 28 · topic 01).
