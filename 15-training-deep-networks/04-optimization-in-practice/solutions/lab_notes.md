# optimization in practice · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution, scored on 1,000 held-out images. offline (the synthetic stand-in that `verify.py` uses) the task is easier; the differences are noted.

## experiment 1: the learning-rate range test

**what people usually predict:** that the suggestion lands near Adam's default of $10^{-3}$, and that 10× more is either much better (faster) or much worse (diverges).

**what happens:** the sweep stops after 101 of 150 steps, at lr $1.06 \times 10^{-2}$, where the loss jumps past 4× its best. the smoothed loss is lowest at $6.7 \times 10^{-3}$, so `suggest_lr` returns $6.7 \times 10^{-4}$, a little below Adam's default. training for 2 epochs (cosine schedule, 2 seeds) confirms it:

| learning rate | value | validation accuracy |
|---|---|---|
| suggestion / 10 | 6.7e-05 | 0.504 |
| suggestion | 6.7e-04 | 0.922 |
| suggestion × 10 | 6.7e-03 | 0.908 |

the suggestion is the best of the three; 10× more still trains but slightly worse; 10× less hasn't got anywhere in 2 epochs. (offline the curve is shifted and the rule is too conservative: suggestion $1.8 \times 10^{-4}$ gives 0.836, 10× more gives 1.000, 10× less 0.152.)

**why:** for the first few orders of magnitude, $\eta$ is so small that 30 steps barely move the weights, so the loss is flat (just minibatch noise; it starts at 2.7, above $\log 10$, because an 8-layer kaiming network starts with large logits). then each step makes measurable progress and the loss falls; but the learning rate keeps growing by about 10% per step, until a step overshoots, the loss jumps, and the next (larger) step makes it worse. the minimum of the curve is where the loss *has just been pushed down* by steps that are a few steps away from blowing up. training for thousands of steps there would sit at the edge of stability, which is why the rule backs off by 10×. on real MNIST the rule lands on the best of the three; offline, on an easier problem, a larger learning rate would have been better, a reminder that it's a heuristic for *where to start*, read in a fraction of a second instead of the 6 training runs this experiment took to check it.

## experiment 2: when warmup matters

**what people usually predict:** warmup is a ritual that helps a bit everywhere, or (since Adam is adaptive) that it doesn't matter at all.

**what happens:** validation accuracy after 2 epochs (cosine schedule, mean of 2 seeds), without warmup and with warmup over the first 50 of 126 steps:

| network, schedule | lr 1e-3 | lr 3e-3 | lr 1e-2 | lr 3e-2 |
|---|---|---|---|---|
| plain, no warmup | 0.93 | 0.93 | 0.90 | 0.62 |
| plain, 50 warmup steps | 0.93 | 0.94 | 0.93 | 0.83 |
| layer norm, no warmup | 0.93 | 0.94 | 0.71 | 0.10 |
| layer norm, 50 warmup steps | 0.93 | 0.93 | 0.92 | 0.89 |

at the default learning rate warmup changes nothing. at large peak learning rates it changes everything, and most of all for the layer-norm network: from chance (0.10) to 0.89 at lr 0.03. without warmup, the layer-norm network is the *least* robust of the four; with warmup it's the *most* robust. (offline: the same pattern at lr 0.03, 0.95 → 1.00 plain and 0.14 → 1.00 with layer norm; everything else is at 1.00.)

**why:** Adam's first update moves every weight by about $\pm\eta$ regardless of its gradient's size (on paper 3). at $\eta = 0.03$, with kaiming weights of typical size 0.05–0.13, that's a 25–60% change to every weight in the first step, in directions estimated from one minibatch, repeated for the next few steps while $\hat v$ is still a poor estimate. the network lands somewhere bad (saturated or dead units, a scale that the schedule later can't undo) and the rest of training can't get out. warmup makes those first steps small, so by the time the learning rate is large, the moments are good estimates and the weights have moved to a region where large steps are safe. the layer-norm network suffers more without warmup plausibly because normalization hides the scale of the weights from the loss: the gradient tells Adam nothing about how large the weights may safely get, so the early sign-like steps can inflate or distort them freely, and after normalization that shows up as large changes in what each unit computes. once the scale has settled, though, normalization makes the network tolerant of large steps (topic 02), which is why it's the most robust with warmup. this is the mechanism Liu et al. (2020) analyse for transformers, which use layer norm and Adam and essentially always warm up.

## experiment 3: batch size and the learning rate

**what people usually predict:** larger batches are "more accurate", so they should be at least as good; or, for those who know the linear scaling rule, that it holds everywhere.

**what happens:** validation accuracy after 2 epochs of SGD with momentum (mean of 2 seeds):

| learning rate | B = 8 | B = 32 | B = 128 | B = 512 | B = 2048 |
|---|---|---|---|---|---|
| fixed lr 0.01 | 0.944 | 0.950 | 0.936 | 0.899 | 0.631 |
| linear scaling: 0.01 × B / 32 | 0.951 | 0.950 | 0.954 | 0.946 | 0.099 |

with a fixed learning rate, accuracy falls steadily as the batch grows: 0.95 at 32, 0.90 at 512, 0.63 at 2,048. scaling the learning rate with the batch keeps accuracy flat up to $B = 512$ (lr 0.16), and then fails completely at 2,048 (lr 0.64): the run diverges. (offline: fixed lr 0.953 at 512 and 0.339 at 2,048; linear scaling 1.000 at 512 and 0.879 at 2,048, where only 2 optimizer steps are taken.)

**why:** with a fixed number of epochs, the number of optimizer steps is $2n/B$: 2,000 steps at $B = 8$, 8 at $B = 2048$. at a fixed learning rate each step moves the weights about as far whatever $B$ is (the gradient's mean doesn't depend on $B$), so a larger batch simply means fewer, slightly cleaner steps, and less progress. the linear scaling rule compensates by making each step $k$ times larger; to first order, one step with $kB$ examples and $k\eta$ equals $k$ steps with $B$ and $\eta$ (on paper 4), so progress per epoch is preserved. the approximation assumes the gradient doesn't change much over the big step. at $\eta = 0.64$ with momentum 0.9 (an effective step of $\eta/(1 - \mu) = 6.4$), it does, and the first steps overshoot catastrophically. Goyal et al. (2017) pushed the rule to batches of 8,192 on ImageNet with a ResNet by adding 5 epochs of gradual warmup; even then there's a critical batch size beyond which nothing helps (McCandlish et al., 2018). in this small problem it's somewhere between 512 and 2,048.
