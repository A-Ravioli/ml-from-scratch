# modules, losses & training loops · lab notes

numbers are for real MNIST. offline (the synthetic stand-in that `verify.py` uses) the qualitative picture is the same; the exact values differ, and are noted where the difference matters.

## experiment 1: overfit one batch

**what people usually predict:** random labels can't be learned ("there's nothing to learn"), and every bug in the list will make the test fail.

**what happens:** the correct loop drives the loss on the batch to **0.002** (true labels) and **0.004** (random labels), with 100% accuracy on both. the bugs split three ways:

| loop | true labels | random labels |
|---|---|---|
| correct | 0.002 / 100% | 0.004 / 100% |
| no zero_grad | 0.000 / 100% | 0.224 / 97% |
| no activation | 0.002 / 100% | 0.003 / 100% |
| lr too small | 2.248 / 38% | 2.271 / 28% |
| last layer only | 1.244 / 94% | 1.432 / 69% |

(final loss / accuracy on the batch.) the missing activation passes perfectly; the missing `zero_grad` passes with the true labels and nearly passes with random ones, but its loss curve oscillates wildly instead of falling smoothly; the tiny learning rate and the frozen first layer fail clearly. (offline, the missing `zero_grad` fails more visibly: 0.28 and 2.03.)

**why:** with about 100,000 parameters and only 32 examples, the network has vastly more freedom than constraints, so it can fit *any* labels, including random ones (Zhang et al., 2017, show the same for whole datasets). that's exactly what makes the test useful: a correct pipeline *must* reach near-zero loss, so failure points at plumbing. a learning rate 1,000 times too small barely moves the weights in 200 steps; training only the last layer means fitting 32 labels with a linear readout of 128 fixed random features, which Adam at this learning rate can't finish in 200 steps. the two survivors survive for different reasons. without the ReLU the model is linear, $784 \to 10$, and a linear map with 7,850 parameters can still memorize 32 points in 784 dimensions: the test checks that learning happens, not that the architecture is right (a check on test accuracy, or simply printing the model, catches it). without `zero_grad`, every step uses the sum of all gradients so far, a strange momentum that overshoots back and forth; on an easy batch it still gets there, and the only symptom is the jagged curve. the lesson: the one-batch test is necessary, not sufficient. look at the curve, not just the final number.

## experiment 2: the loss before training

**what people usually predict:** "random" or "large", rarely a specific number; and that scaling the weights by 10 makes the loss about 10 times larger.

**what happens:** with the default init the loss is **2.293**, within 0.01 of $\log 10 = 2.303$, and the network's average confidence is 0.11, barely above $1/10$. scaling down changes nothing (2.302 at $s = 0.1$). scaling up hurts fast: **2.321** at $s = 3$, **7.995** at $s = 10$, where the average top probability is 0.82: the untrained network is confidently wrong about most images. (offline: 2.311 at $s = 1$ and 18.3 at $s = 10$.)

**why:** with small logits, softmax is nearly uniform, and the cross-entropy of a uniform guess over 10 classes is $\log 10$. both layers are multiplied by $s$, so (ignoring biases, and since ReLU is positively homogeneous) the logits scale by $s^2$: a factor of 100 at $s = 10$, which turns tiny random logits into large random ones (the std of the logits goes from 0.07 to 5.7). softmax of large random logits is nearly one-hot on a random class, so most examples are confidently wrong and pay a loss proportional to the logit gaps. starting there is bad because the first many steps go into undoing the confident nonsense, and saturated softmax outputs give small, uninformative gradients for the examples the model is confidently right about. "the initial loss should be $\log k$" is a check worth running on every new classifier.

## experiment 3: too large a learning rate

**what people usually predict:** that past some learning rate the loss explodes to `inf` or `nan`.

**what happens:** after 3 epochs of SGD, the loss falls from **2.277** ($\eta = 10^{-3}$) to a best of **0.421** ($\eta = 0.3$), then rises again: **0.874** at $\eta = 1$ and **2.307** at $\eta = 3$, which is $\log 10$ almost exactly. no `nan` appears. the dead-unit column explains it: 0% of the hidden units are dead up to $\eta = 0.3$, **59%** at $\eta = 1$, **96%** at $\eta = 3$. (offline the synthetic data breaks earlier: the best is $\eta = 0.1$ with 0.046, and from $\eta = 0.3$ on every hidden unit is dead and the loss sits at 2.3.)

**why:** one huge early update pushes many hidden units' biases (and weights) so far negative that their pre-activation is negative for every image. a ReLU with negative input outputs 0 *and has gradient 0*, so nothing can ever push it back: it's dead for good. with most units dead, the hidden layer outputs (nearly) the same zero vector for every image, the network can only learn the class frequencies through its output bias, and the loss settles at the entropy of the label distribution, about $\log 10$. it's a quiet failure: nothing overflows, the loss just stops improving at exactly the value of a uniform guess, which is the signature to recognise. note also where the best learning rate sits: within a factor of about 3 of the one where training breaks. that's typical, and it's why chapter 15's learning-rate range test sweeps upwards until the loss blows up and then backs off.
