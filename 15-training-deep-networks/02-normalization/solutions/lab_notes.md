# normalization · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution: 8-layer ReLU networks, 128 wide, kaiming init, trained on 4,000 images and scored on 1,000 held-out ones (experiment 2: 2,000 test images). offline (the synthetic stand-in that `verify.py` uses) the task is much easier and most accuracies are near 1; the differences that remain are noted.

## experiment 1: how large can the learning rate be?

**what people usually predict:** that batch norm helps a little at every learning rate, and that layer norm, being "the same idea", helps about as much.

**what happens:** validation accuracy after 2 epochs of plain SGD (mean of 2 seeds):

| normalization | lr 0.01 | lr 0.03 | lr 0.1 | lr 0.3 | lr 1 | lr 3 |
|---|---|---|---|---|---|---|
| none | 0.82 | 0.89 | 0.90 | 0.69 | 0.10 | 0.10 |
| batch | 0.56 | 0.83 | 0.93 | 0.93 | 0.91 | 0.68 |
| layer | 0.72 | 0.87 | 0.87 | 0.88 | 0.10 | 0.10 |

the plain network is best at 0.1, already damaged at 0.3 and dead at 1. the batch-norm network is still at 0.91 at lr 1, ten times past the plain network's best, and still learning at 3; its best accuracy (0.93) is also the highest in the table. but at lr 0.01 it's the *worst* of the three. layer norm sits in between at small learning rates and dies at lr 1 exactly like the plain network. (offline: the same shape. the plain network fails from 0.3, batch norm reaches 1.00 up to lr 1, layer norm survives 0.3 and fails from 1.)

**why:** without normalization, a step that grows a weight matrix grows everything downstream, and at large learning rates that feedback blows up (or kills the ReLUs, as in ch 14 · topic 04). a batch-norm layer after each `Linear` removes that feedback: the next layer sees standardized inputs whatever the scale of the weights (on paper 3). better still, the gradient with respect to a weight in front of a batch norm scales like $1/\|W\|$, so a step that inflates $W$ automatically shrinks the following steps: an effective learning rate $\eta/\|W\|^2$ that falls when $\eta$ is too large. the poor result at lr 0.01 is the other side of the same coin (this lab doesn't isolate the cause, but two contributions are clear): the effective step on the directions that matter is $\eta/\|W\|^2$, which is small when $\eta$ is, and batch norm's projection removes the components of the gradient along "shift" and "scale", so in 2 epochs at a small learning rate the network makes less progress than the plain one. batch norm's gift is the wide window of usable learning rates, not speed at a learning rate that's too small anyway. layer norm normalizes across the 128 units of each example instead. that removes a layer-wide scale, but not the scale of an individual unit: if one step makes a few units huge, they dominate their example's mean and variance and the rest are squashed towards a constant, so large-learning-rate instability gets through. batch norm controls every unit's scale separately, across the data, which is the scale the instability lives in.

## experiment 2: forgetting `model.eval()`

**what people usually predict:** a small drop, a few points, when evaluating in training mode; most don't predict any effect of the order of the test set.

**what happens:**

| how it's scored | test batch size | accuracy |
|---|---|---|
| eval mode (running statistics) | any | 0.861 |
| training mode, shuffled batches | 1000 | 0.857 |
| training mode, shuffled batches | 64 | 0.841 |
| training mode, shuffled batches | 8 | 0.675 |
| training mode, shuffled batches | 2 | 0.235 |
| training mode, batches sorted by class | 1000 | 0.662 |
| training mode, batches sorted by class | 64 | 0.158 |

with large shuffled batches the mistake is almost invisible (0.857 versus 0.861). with batches of 2 the network is barely above chance, and with class-sorted batches of 64, a perfectly ordinary test-loader configuration, accuracy collapses to 16%. (offline: 1.000 in eval mode, 0.880 with batches of 8, 0.280 with batches of 2, 0.273 sorted in 64s.)

**why:** in training mode each feature is standardized with the statistics of the current batch. a large shuffled batch has statistics close to the population's (and close to the running averages), so nothing changes. a small batch has noisy statistics, and a batch of 2 has degenerate ones: every feature becomes exactly $\pm 1$ (the lesson's worked example), so all the network knows about each image is, feature by feature, whether it's above or below its batchmate. a class-sorted batch is worse in a different way: every image in it is, say, a 3, so the features that distinguish 3s from other digits have a large mean across this batch, and batch norm subtracts exactly that mean. the information the classifier needs is the part that gets normalized away. it's a silent bug because nothing errors and the training-mode outputs look like ordinary logits; the fix is the single line `model.eval()` (which `nn.predict`, `nn.evaluate` and `torchnn.evaluate` all do for you), and the capstone plants exactly this bug.

## experiment 3: tiny batches

**what people usually predict:** batch norm still helps (it did in experiment 1), perhaps a bit less.

**what happens:** after 1 epoch (mean of 2 seeds):

| batch size | normalization | validation accuracy | final training loss |
|---|---|---|---|
| 4 (lr 0.01) | none | 0.903 | 0.809 |
| 4 (lr 0.01) | batch | 0.563 | 1.882 |
| 4 (lr 0.01) | layer | 0.883 | 0.860 |
| 64 (lr 0.1) | none | 0.722 | 1.681 |
| 64 (lr 0.1) | batch | 0.891 | 1.272 |
| 64 (lr 0.1) | layer | 0.601 | 1.292 |

with batches of 4, batch norm is by far the worst, and even its *training* loss is high, while layer norm is unaffected. with batches of 64 the order flips and batch norm is best. (the two batch sizes aren't directly comparable: batch size 4 takes 16 times more steps per epoch. compare the rows within a batch size. offline, the easy synthetic task hides most of it: 1.000, 0.835 and 1.000 at batch size 4.)

**why:** two separate problems. during training, the mean and variance of 4 numbers are very noisy estimates, so every example's normalized features jitter from step to step depending on its three batchmates: noise that the network has to fit through, which is why even the training loss suffers (some of this noise acts as a regularizer for moderate batch sizes; at 4 it's too much). at evaluation, the running averages of those noisy batch statistics don't match what the network saw in training (the running variance averages *within-batch* variances, which for small batches are much noisier than the population's), so the eval-mode network is not the network that was trained. layer norm computes its statistics over the 128 features of one example, so neither problem exists: training and evaluation are the same function, whatever the batch. this is why sequence models, which often train with small batches and generate one token at a time, use layer norm or RMSNorm (ch 18, ch 27), and why small-batch vision models use group norm or "frozen" batch norm.
