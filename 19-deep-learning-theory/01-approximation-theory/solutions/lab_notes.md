# approximation theory · lab notes

numbers below are from the reference solution on a laptop CPU, with the seeds in the notebook. the lab uses only synthetic data, so online and offline runs are identical; it takes about 80 seconds on a busy laptop.

## experiment 1: a function depth makes cheap

**what people predict.** after reading about depth separation, most people predict that the deep network, whose shape can represent the sawtooth exactly, will learn it, or at least do much better than the shallow network of equal size.

**what happens.** the representation side matches the theory exactly:

| k | pieces | `sawtooth_net`: weights | pieces counted | shallow interpolant: weights | pieces counted |
|---|---|---|---|---|---|
| 1 | 2 | 7 | 2 | 7 | 2 |
| 4 | 16 | 25 | 16 | 49 | 16 |
| 6 | 64 | 37 | 64 | 193 | 64 |
| 8 | 256 | 49 | 256 | 769 | 256 |

the deep network's weights grow by 6 per doubling of the pieces; the shallow network's double. the training side doesn't follow. trained from random starts on 1,024 evenly spaced samples (2,000 Adam steps, cosine schedule, batch 256), measured as MSE divided by the target's variance (1.0 = predicting the mean), three seeds:

| k | network | weights | relative error |
|---|---|---|---|
| 4 | deep, width 2 (the sawtooth's shape) | 25 | 1.00, 1.00, 1.00 |
| 4 | deep, width 8 | 241 | 1.00, 1.00, 1.00 |
| 4 | shallow, 80 units | 241 | 0.99, 0.98, 0.99 |
| 6 | deep, width 2 | 37 | 1.00, 1.00, 1.00 |
| 6 | deep, width 8 | 385 | 1.00, 1.00, 1.00 |
| 6 | shallow, 128 units | 385 | 1.00, 1.00, 1.00 |

no network learns anything beyond the mean. the deep networks don't beat the shallow one; at $k = 4$ the shallow one is marginally better.

**why.** depth separation is a statement about the existence of weights, and the sawtooth's weights are a needle in a haystack. a randomly initialized deep ReLU network of width 2 or 8 computes a nearly constant function on $[0, 1]$ (many units are dead, the rest compose into something with a handful of pieces, experiment 3), so the gradient at initialization says only "move towards the mean". and the sawtooth gives no partial credit: its mean over any interval of a few teeth is $\frac12$, so a function that has learned some folds but not all scores no better than the mean, and there is no gradual path. the shallow network has the opposite problem: 128 units can't make 64 pieces in the right places without precise coordination either, and Adam from a random start doesn't find it in 2,000 steps. (with $k = 3$, eight pieces, a wide shallow network does fit it; the reference's exploration found relative errors near 0 for 166 units.) part of the failure is the training budget: with a learning rate of $10^{-2}$ and 8,000 steps (four times longer), the $k = 4$ sawtooth is partly learned (shallow 0.77–0.85, deep width-8 0.79, 0.90 and 1.00 over three seeds; a reviewer's run reached 0.53 for a deep net), while at $k = 6$ every network stays at 0.97–1.00 and the width-2 networks never leave 1.00 at either $k$. so "unlearnable" is about how fast the difficulty grows with $k$, not an absolute wall. the lesson: an expressivity result tells you a network *could* represent something, not that training will get there.

## experiment 2: a function depth makes learnable

**what people predict.** a modest benefit for depth, maybe 10–30%, or none ("a wide shallow network is a universal approximator, and 500 units is a lot").

**what happens.** (radial ripple in $d = 4$, 3,000 parameters, 20,000 training samples, 3,000 steps, three seeds; relative test error)

| hidden layers | width | ω = 4 (two ripples) | ω = 2 (one ripple) |
|---|---|---|---|
| 1 | 500 | 0.894, 0.898, 0.894 | 0.449, 0.451, 0.460 |
| 2 | 51 | 0.491, 0.737, 0.470 | 0.034, 0.183, 0.123 |
| 3 | 37 | 0.154, 0.103, 0.120 | 0.042, 0.014, 0.016 |
| 4 | 30 | 0.098, 0.060, 0.146 | 0.019, 0.011, 0.015 |

at the same budget, the 4-layer network's error is 6–15 times smaller than the 1-layer network's (ω = 4) and 20–40 times smaller at ω = 2. the one-hidden-layer network barely learns the two-ripple target at all (0.89). the 2-layer networks vary a lot between seeds; 3 and 4 layers are consistently good.

**why.** each unit of a one-hidden-layer network is a ridge function, constant along a hyperplane, and a radial ripple is far from any small sum of ridge functions: to bend a ring around the origin out of straight creases you need many units in many directions, more as the dimension grows (Eldan & Shamir, 2016, proved an exponential lower bound in $d$ for a construction of this kind). a deeper network can compute an approximation of $\lVert x\rVert^2 = \sum_i x_i^2$ in its first layers (each $x_i^2$ is a 1-D function, a few units each) and then a 1-D function of it, an easy job. unlike the sawtooth, every stage gives the gradient something to fit on its own (the radius is a smooth function, and the ripple has only one or two oscillations), so the composition is learnable as well as representable. the 2-layer network sits in between: it has one hidden layer to compute a radius-like quantity and one to apply the ripple, which works on some seeds and not others.

## experiment 3: how many pieces does a random network have?

**what people predict.** given the $(2w)^L$ bound, most people expect deeper networks to have many more pieces, possibly hundreds for depth 8.

**what happens.** (48 ReLUs in total, PyTorch's default initialization, on $[-3, 3]$, five networks each)

| layers × width | pieces |
|---|---|
| 1 × 48 | 42, 37, 39, 42, 44 |
| 2 × 24 | 33, 35, 38, 49, 37 |
| 4 × 12 | 23, 32, 28, 31, 34 |
| 8 × 6 | 11, 11, 9, 12, 10 |

fewer pieces than units at every depth, and *fewer* the deeper the network: the 8-layer networks have about 10 pieces from 48 units. `sawtooth_net(6)` has 64 pieces from 12 units.

**why.** Hanin & Rolnick (2019) proved that for a random network the expected number of pieces along a line is bounded by (about) the total number of neurons times a constant that depends on the weights' distribution, not exponentially in depth: a unit only creates a kink where its pre-activation crosses zero *inside* the interval, and in a deep network most units' pre-activations either never cross zero there or cross where nothing downstream amplifies them. narrow deep networks are also prone to dead units at initialization, which add no kinks at all. depth multiplies pieces only when each layer folds the interval onto itself just right, as the sawtooth's weights do. (the counter misses kinks whose slope change is below `tol` = $10^{-4}$ of the steepest slope, so these are lower bounds; the trend is robust to the tolerance.)
