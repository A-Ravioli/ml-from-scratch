# evaluating representations · lab notes

numbers are from the reference solution with the lab's defaults (real MNIST, laptop CPU, 4 threads); the whole lab took about 2 minutes. the pretraining is a single run with seed 0, so treat differences of a point or two between representations as within run-to-run noise; experiment 2 shows the spread over label subsets explicitly. yours will differ in detail.

## experiment 1: how good is a network that has never been trained?

**what people usually predict:** the random CNN near chance (20–30%), raw pixels mediocre, the pretrained CNN clearly best with both evaluations.

**what happens:** 1,000 labels (200 per class) of the digits 5–9; 2,000 test images, so the binomial standard error is about ±0.007.

| representation | dim | linear probe | kNN (k = 20, T = 0.07) |
|---|---|---|---|
| raw pixels | 784 | 0.886 | **0.937** |
| random CNN | 64 | 0.858 | 0.779 |
| pretrained CNN (digits 0–4) | 64 | 0.884 | 0.773 |

three surprises. the never-trained network is far above chance with the probe (0.86). raw pixels win kNN by a wide margin. and the network trained on 0–4 is no better than the random one under kNN, and only matches raw pixels under the probe.

**why:** a random conv net is a bank of random local filters followed by ReLUs and pooling. that computes a nonlinear, somewhat translation-tolerant summary of local stroke patterns (a random-features model, ch 09 · topic 05), and 64 such features are enough for a linear classifier to separate five digit classes reasonably well. MNIST is also unusually friendly to raw-pixel kNN: digits are centred and size-normalised, so two 7s really are close pixel by pixel. kNN asks whether neighbourhoods are class-pure, and pixel space already has that property; the random CNN scrambles distances (its 64 features are not a distance-preserving map), so kNN gets worse even though linear separability is fine.

the pretrained network was trained (one epoch, cross-entropy) to map the digits 0–4 to *five* well-separated regions. a plausible explanation for its poor kNN score on 5–9 is that it squeezes new digits towards those five regions too: a 5 and an 8 can land near the same "3-like" region, which hurts nearest neighbours more than it hurts a linear probe that can still use the residual directions. the lesson to take away is less about this particular network than about the benchmark: when raw pixels score 94% with kNN, the task can't discriminate between good and bad representations. the capstone uses CIFAR-10, where raw-pixel kNN (cosine, k = 20, all 50,000 training images) scores 0.364, measured with the reference `knn_eval`.

## experiment 2: probe or fine-tune?

**what people usually predict:** with very few labels the linear probe wins (fine-tuning millions of weights on 25 images should overfit); with many labels fine-tuning wins; from scratch is far behind everywhere.

**what happens:** mean ± std over 3 label subsets (120 Adam steps of batch 32 for both fine-tuning and from scratch, at every budget):

| labels per class | linear probe | kNN | fine-tune | from scratch |
|---|---|---|---|---|
| 5 | 0.662 ± 0.040 | 0.647 ± 0.008 | 0.691 ± 0.044 | 0.596 ± 0.072 |
| 20 | 0.790 ± 0.034 | 0.692 ± 0.012 | 0.841 ± 0.020 | 0.746 ± 0.066 |
| 100 | 0.860 ± 0.003 | 0.732 ± 0.002 | 0.903 ± 0.014 | 0.885 ± 0.024 |
| 500 | 0.880 ± 0.003 | 0.795 ± 0.009 | 0.910 ± 0.026 | 0.913 ± 0.012 |

fine-tuning is at or above the probe at every budget here, including 5 labels per class (though at that budget the difference, 0.03, is smaller than the spread over subsets). the benefit of pretraining, fine-tune minus from scratch, is about 0.10 at 5 and 20 labels per class, 0.02 at 100, and zero at 500. kNN trails everywhere on these features.

**why:** fine-tuning with few labels is less dangerous than it sounds, for two reasons. it starts from weights that already produce useful features, and it runs only 120 small steps: early stopping is built in, so the network can't move far from the pretrained solution. the probe, meanwhile, is limited to what's linearly available in 64 features that were shaped for a different task; fine-tuning can adjust them. the spread over label subsets is largest at small budgets because with 5 examples per class *which* five you get matters a lot (one badly written 9 is 20% of the class). from scratch catches up once there are enough labels to learn the features directly: with 500 per class, the digits 5–9 alone teach the network everything the 0–4 pretraining did. that's the general shape of label-efficiency curves, and the capstone measures it for self-supervised pretraining on CIFAR-10: pretraining matters most when labels are scarce.

with a wider network (a first draft of this lab used 128 features and 150 steps), the probe was ahead of fine-tuning at 20 labels per class (0.84 ± 0.04 versus 0.80 ± 0.11) and level at 5. which protocol "wins" at low budgets depends on how good the frozen features are relative to what a few steps of fine-tuning can fix: report both.
