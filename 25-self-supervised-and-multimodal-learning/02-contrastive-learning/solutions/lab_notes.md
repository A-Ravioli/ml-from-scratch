# contrastive learning · lab notes

numbers are from the reference solution with the lab's defaults (real MNIST, laptop CPU, 4 threads); the lab took about two minutes. yours will differ in detail.

## experiment 1: what does a loss value mean?

**what people usually predict:** random embeddings give "a high loss", maybe around 5 or 10, the same at every temperature; perfect positives give a loss near 0 regardless of the temperature.

**what happens:** (loss of random embeddings / loss with perfect positives, $d = 128$)

| B | log(2B−1) | T = 0.05 | T = 0.1 | T = 0.5 | T = 1.0 |
|---|---|---|---|---|---|
| 16 | 3.43 | 4.47 / 0.00 | 3.61 / 0.00 | 3.41 / 1.63 | 3.42 / 2.49 |
| 64 | 4.84 | 6.60 / 0.00 | 5.34 / 0.01 | 4.88 / 2.91 | 4.86 / 3.86 |
| 256 | 6.24 | 7.74 / 0.00 | 6.61 / 0.03 | 6.25 / 4.26 | 6.24 / 5.24 |
| 1024 | 7.62 | 9.11 / 0.00 | 7.99 / 0.13 | 7.63 / 5.64 | 7.63 / 6.63 |

random embeddings land on $\log(2B - 1)$ at $T \ge 0.5$, and well *above* it at small temperatures (7.7 instead of 6.2 at $B = 256$, $T = 0.05$). perfect positives score essentially 0 at $T \le 0.1$, but 5.2 at $T = 1$ with $B = 256$: not far below the random encoder's 6.2.

**why:** with all logits equal, InfoNCE is a uniform guess among $2B - 1$ candidates, $\log(2B - 1)$. random unit vectors in 128 dimensions have cosines spread by about $1/\sqrt{128} \approx 0.09$; at $T = 0.05$ that becomes a spread of 1.8 in the logits, and by Jensen's inequality (on paper 1) a spread among the negatives' logits *raises* the log-sum-exp, so the loss rises above chance even though nothing has been learned. with perfect positives the best possible logit gap between the positive (cosine 1) and a typical negative (cosine about 0) is $1/T$: 20 at $T = 0.05$, which crushes 510 negatives, but only 1 at $T = 1$, which can't. a high-temperature run therefore keeps a large loss, and a large gradient pulling the positives together, long after its positives are aligned, while the push on individual negatives is weak and spread evenly.

so a loss value means nothing without its batch size and temperature: report the chance level $\log(2B - 1)$ alongside it, or the gap between the two (the InfoNCE mutual-information bound).

## experiment 2: the temperature in training

**what people usually predict:** the lowest temperature gives the lowest loss and the best features; or "the textbook value 0.1–0.2 is best by a lot".

**what happens:** batch size 128, 2 epochs (156 steps), mean over 2 seeds with the range in brackets:

| encoder | final loss | kNN | linear probe | alignment ↓ | uniformity ↓ |
|---|---|---|---|---|---|
| random init (no training) | – | 0.769 | 0.774 | – | – |
| T = 0.05 | 3.26 | 0.857 (0.857–0.858) | 0.850 (0.848–0.853) | 0.288 (0.276–0.301) | −1.55 (−1.58 to −1.51) |
| T = 0.2 | 3.54 | 0.868 (0.867–0.870) | 0.864 (0.864–0.865) | 0.581 (0.571–0.591) | −2.81 (−2.84 to −2.79) |
| T = 1.0 | 4.89 | 0.856 (0.854–0.858) | 0.844 (0.839–0.850) | 0.441 (0.425–0.458) | −2.34 (−2.37 to −2.30) |

all three temperatures improve on the random network by 8–10 points of kNN accuracy. $T = 0.2$ is best by about a point, which is small but larger than the spread between the two seeds (a few tenths of a point). $T = 0.05$ ends with the lowest loss and the best alignment but the *least* uniform embeddings; $T = 0.2$ the most uniform.

**why:** the loss curves can't be compared across temperatures (experiment 1), but alignment, uniformity and downstream accuracy can. the gradient on a negative is proportional to its softmax weight (on paper 2). at $T = 0.05$ nearly all of the push goes to a few hard negatives per anchor, which after two epochs leaves most of the sphere unused: views agree tightly (good alignment) but different images are crowded together (poor uniformity), and kNN suffers. at $T = 1$ the push is spread over all negatives but weak, and the loss can't distinguish a good partner from a decent negative decisively. $T = 0.2$ balances the two. these differences are modest at this tiny scale and could shift with longer training (Wang & Liu, 2021, find small temperatures become more uniform with enough training); two seeds is enough to see that the 0.2 advantage is real here, not to estimate its size precisely.

the bigger lesson is the baseline: a random CNN already gets 77% with kNN on MNIST, so "SimCLR reaches 86%" sounds less impressive than it is out of context. always report the random-init number of the same architecture. changing the batch size with the dropdown also changes the number of steps (2 epochs of 10,000 images is 312 steps at 32 but 39 at 512), so at a fixed number of epochs smaller batches can win here, which says nothing about the large-batch results of the SimCLR paper (fixed and much longer training, much harder data).
