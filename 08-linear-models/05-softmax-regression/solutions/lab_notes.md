# softmax regression · lab notes

experiments 1 and 3 use MNIST. offline (with `MLFS_OFFLINE=1`, as in `verify.py`) the lab runs on a synthetic stand-in made of 10 smooth blob templates plus noise, which a linear model separates perfectly: every accuracy there is 1.000 and the confusion list is empty. that's a smoke test, not a result. the numbers below are from the reference solution on real MNIST (the lab's default, about 12 s on a laptop).

## experiment 1: how far does a linear model get on MNIST?

**what people usually predict:** 50–80%: "it's just a linear model on pixels, it can't know what a 7 looks like".

**what happens:** test accuracy **0.914** (95% bootstrap CI 0.908–0.919) with 20,000 training images and 5 epochs; training accuracy 0.929; the majority-class baseline is 0.114. trained on all 60,000 images the same model gets 0.922 after 5 epochs and 0.926 after 20 (training accuracy 0.931): more data and more epochs buy less than a point. that's in line with the classic figures for linear classifiers on MNIST (LeCun et al., 1998, list a single-layer linear classifier at 8–12% error depending on preprocessing). the most frequent confusions are 2 → 8 (49 times), 9 → 4 (41) and 7 → 9 (41): digits that share strokes. the weight images look like smudged digits: each class's weights are positive (red) where that digit has ink and the others usually don't, and negative (blue) where other digits have ink, so the 0 has a red ring around a blue center and the 1 a red vertical bar.

**why:** the score $w_c^\top x$ is a correlation between the image and a template. MNIST digits are centered and size-normalized, so "how much ink is where" is very informative: a 0 has ink in a ring and none in the middle, a 1 has ink in a central column. one template per class captures the *average* shape, and most digits are close enough to their average. the weights aren't the average images themselves, but the directions that best *discriminate* each class from the others, which is why blue regions matter as much as red ones. what a linear model can't do is represent several distinct shapes per class (a 4 with an open or a closed top, a 7 with or without a crossbar) or invariance to shifts: move a 1 three pixels sideways and it no longer matches the template. the 1–2 point train–test gap says the model isn't overfitting: it's limited by its bias. learned features (chapter 14's MLP gets about 98%) and convolutions (chapter 16, over 99%) remove exactly those limits.

## experiment 2: one model for all classes, or one per class?

**what people usually predict:** both linear, so about the same; maybe one-vs-rest is slightly worse overall.

**what happens** (5 classes in a row, 100 points each):

| class | softmax | one-vs-rest |
|---|---|---|
| 0 (left end) | 0.95 | 0.98 |
| 1 | 0.91 | 0.86 |
| 2 (middle) | 0.90 | **0.67** |
| 3 | 0.87 | **0.76** |
| 4 (right end) | 0.97 | 0.97 |
| all | **0.920** | **0.848** |

the end classes are fine either way. the middle classes lose up to 23 points under one-vs-rest. on the probability plot, softmax gives every class a clean bump centered on its cluster. one-vs-rest's curves for classes 1, 2 and 3 are low and flat (none rises above about 0.45), and class 2's never clearly dominates anywhere.

**why:** the one-vs-rest classifier for class 2 has to separate class 2 from classes 0, 1, 3 and 4, which lie on *both* sides of it along $x_1$. a logistic model's probability is monotone along any line, so the best it can do is a nearly flat probability (roughly the class's share of the data), and its neighbours' classifiers, which do have a direction to lean in near class 2, win the argmax there. softmax never asks one score to separate a class from everything: it compares five linear scores, and where class 2's score is the largest of the five (an interval in the middle) it gets the most probability. a maximum of linear functions is piecewise linear and convex, and the softmax of linear scores can carve the line into intervals, which one monotone sigmoid per class cannot. one-vs-rest is still a fine choice with flexible binary models (Rifkin and Klautau, 2004), or when classes overlap (multi-label problems).

## experiment 3: how big should a batch be?

**what people usually predict:** the same accuracy after each epoch whatever the batch size, since every row sees every image equally often; maybe the full batch is best because its gradient is exact.

**what happens** (test accuracy after 1, 2 and 5 epochs, $\eta = 0.5$):

| batch size | 1 epoch | 2 epochs | 5 epochs | steps in 5 epochs |
|---|---|---|---|---|
| 10 | 0.900 | 0.899 | 0.900 | 10,000 |
| 100 | 0.904 | 0.910 | **0.914** | 1,000 |
| 1,000 | 0.862 | 0.882 | 0.897 | 100 |
| full (20,000) | 0.735 | 0.752 | 0.799 | 5 |

after one epoch, batches of 10 and 100 are already at about 0.90; batches of 1,000 (20 steps per epoch) are behind, and full-batch gradient descent, after 1, 2 and 5 steps, is far behind. but the smallest batch doesn't win: batch 10 plateaus at 0.900 from the first epoch on, *below* batch 100's 0.914. at this step size, its gradient noise keeps the weights hovering in a wider region around the optimum, and averaging over 10 images per step isn't enough to settle it.

**why:** progress comes from steps, and every step uses an estimate of the gradient. early on, a gradient averaged over 10 random images points in nearly the same direction as one averaged over 20,000, so a step with the cheap estimate is almost as useful as one with the exact gradient, and you get 2,000 of them per epoch instead of one. that's why epochs are a misleading unit: what matters is (number of steps) × (usefulness of each step). the price of small batches is noise: near the optimum, the gradient noise keeps SGD hovering instead of converging, and the hovering is worse for smaller batches, which is exactly batch 10's plateau below batch 100 (a smaller step, or one that decays, would let it settle). and on real hardware, a batch of 1,000 takes about as long as a batch of 10 (vectorized arithmetic is nearly free until the hardware is saturated), so the sweet spot in wall-clock time is usually a batch in the tens to thousands. nobody trains large models with the full batch because it wastes most of the data's information per step; chapter 15 covers how batch size, step size and noise interact.
