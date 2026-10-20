# loss landscapes · lab notes

numbers below are from the reference solution on a laptop CPU, on the real MNIST (2,000 training and 2,000 test digits, pooled to 14 × 14), with the seeds in the notebook. the whole lab takes about 25 seconds. the offline stand-in is much easier (every run reaches zero test error, so the barriers in experiment 3 vanish), but the sharpness pattern of experiments 1 and 2 is the same.

## experiment 1: sharpness against batch size

**what people predict.** most people have heard "small batches find flat minima" and predict that ordering, with modest differences. few expect a number to appear: $2/\eta = 10$ looks like a coincidence until you see the trajectories.

**what happens.** (3 seeds per batch size, 3,000 steps, $\eta = 0.2$, squared loss)

| batch size | final train loss | sharpness $\lambda_{\max}$ | test error |
|---|---|---|---|
| 16 | 0.025–0.030 | 3.97, 4.03, 4.15 | 0.104–0.115 |
| 64 | 0.018–0.022 | 7.36, 8.52, 8.87 | 0.106–0.113 |
| 256 | 0.016–0.020 | 9.73, 9.83, 10.03 | 0.095–0.110 |
| 2,000 (full batch) | 0.017–0.018 | 9.75, 9.79, 9.83 | 0.105–0.113 |

every run starts at a sharpness of 28.7 (the same initialization for seed 0), almost three times $2/\eta = 10$, and every run is below the line by step 300: the first steps throw the network out of the sharp region it started in. after that, the full-batch and batch-256 runs settle at the line and stay there for the rest of training; batch 64 sits around 7.5–9; batch 16 stays near 4 the whole time. the seed spread is small next to the gap between batch 16 and full batch. the test errors overlap across all four batch sizes: a factor of 2.4 in sharpness comes with no measurable difference in test error here.

the effect depends on the loss and the preprocessing. with cross-entropy the curvature shrinks as the loss goes to zero, so whichever run reached the lowest loss looks flattest; with per-pixel standardized inputs and cross-entropy, the reference measured no batch-size effect at all (sharpness 0.39–0.43 for every batch size). that's why the lab uses the squared loss, as Cohen et al. did, and why the table gives the training losses: compare sharpness only between solutions at comparable loss.

**why.** the stability argument of on-paper 1: gradient descent with step $\eta$ can't stay in a region whose curvature exceeds $2/\eta$, because the component along the top eigenvector gets multiplied by $1 - \eta\lambda < -1$ each step. starting above the edge, the iterate is catapulted out to a flatter region within a few steps (Lewkowycz et al., 2020, call this the catapult phase); then *progressive sharpening* (Cohen et al., 2021) takes over: tracing the first steps of the full-batch run, the loss jumps 0.71 → 1.68 and the sharpness collapses 28.8 → 2.4 within about 3 steps, after which it climbs steadily back up (3.3, 4.9, 6.6, 8.7, 9.2 at steps 5, 8, 12, 20, 30) to sit at the edge, 2/η = 10, by about step 30. so both phases happen; the figure, which starts at step 300, only shows the end state. what it does show is their edge of stability: full-batch gradient descent settles at the edge and stays there, self-stabilizing, while the loss keeps going down. minibatch noise adds random kicks whose effect on the loss grows with the curvature (a kick of size $\epsilon$ along a direction of curvature $\lambda$ raises the loss by $\tfrac12\lambda\epsilon^2$), so a small-batch run can't settle in a region as sharp as $2/\eta$ even though gradient descent could: the effective edge is lower, and smaller batches mean bigger kicks. the generalization side of the story (Keskar et al., 2017) is much weaker in this experiment: the flattest solutions are not the best on test data.

## experiment 2: what a picture of a minimum can and can't tell you

**what people predict.** that the rescaled network, being the same function, has the same sharpness; or, if they've read the explain prompt of experiment 1, that sharpness is a property of the minimum.

**what happens.**

| solution | train loss | test error | sharpness | loss at the four edge midpoints of the plot |
|---|---|---|---|---|
| batch 16 | 0.0274 | 0.115 | 4.15 | 0.80 |
| full batch | 0.0169 | 0.105 | 9.75 | 1.42 |
| batch 16, first layer ×5, second ÷5 | 0.0274 | 0.115 | **23.2** | 0.80 |

the rescaled network has identical loss and predictions and a top eigenvalue 5.6 times larger. its filter-normalized surface is *identical* to the original's, contour for contour (the lab draws it with the same random seed). the full-batch solution's filter-normalized surface does rise faster away from the center (1.42 against 0.80 at the edges), consistent with its larger sharpness, but its test error is no worse.

**why.** the Hessian is a second derivative with respect to the *parameters*, so it depends on how the function is parameterized. scaling the first layer by $c$ and the second by $1/c$ leaves the function alone and scales the curvature of the second layer's weights by $c^2$ and the first layer's by $1/c^2$ (on-paper 2). measured on this network with $c = 5$: restricted to the first layer, the top eigenvalue falls from 3.0 to 0.12 (25-fold); restricted to the second layer (weights and bias), it rises from 1.6 to 23.2, and that block now sets the sharpness of the whole network. Dinh et al. (2017) used exactly this to show that any minimum of a ReLU network can be made arbitrarily sharp. filter normalization measures every perturbation in units of the filter it perturbs, which undoes the rescaling: the surface is a property of the function, not of the coordinates. the lesson: claims that "flat minima generalize" need a scale-invariant notion of flatness; a raw $\lambda_{\max}$ isn't one.

## experiment 3: is there a hill between two solutions?

**what people predict.** usually that the midpoint of two good networks is a good network ("both are near a minimum, the average should be too"), or, from those who know about permutation symmetry, a barrier for every pair.

**what happens.** with $k = 0$ (no shared training, only a shared initialization), three trials:

| pair | end test errors | midpoint test error | barrier (test error) |
|---|---|---|---|
| same init | 0.097–0.104 | 0.103, 0.112, 0.106 | 0.006, 0.013, 0.007 (about 0.01) |
| different init | 0.095–0.109 | 0.416, 0.500, 0.646 | 0.31, 0.40, 0.54 |

two networks that only share their initialization are linearly connected in test error: the error along the segment rises about one point above the line between the ends at most (barriers 0.006–0.013 here, about 0.01; a reviewer's machine gave 0.015). the training loss is less forgiving: at the midpoint it doubles (0.024 → 0.049, 0.026 → 0.052, 0.025 → 0.048), a real but small bump next to the different-initialization pairs, whose midpoint training loss is 12–14 times the ends' (0.026 → 0.31). two networks from different initializations have a large barrier: the midpoint misclassifies 40–65% of the test digits, with the ends at 10%. the training loss shows the same thing (right panel). moving the slider to $k = 100$ or $500$ shared steps changes nothing (barriers of 0.000–0.001 in all six pairs the reference ran), because there was no barrier to remove.

**why.** the hidden units of a layer have no fixed order, so two independently initialized networks learn their features in different slots; averaging slot 3 of one with slot 3 of the other mixes two unrelated features into a third that neither network uses, and the second layer's weights no longer match it. from the same initialization, each slot starts with the same random feature and the early gradient (dominated by the data, not by the minibatch order) pushes both copies the same way, so they stay in one basin. Frankle et al. (2020) found the same for small MNIST networks; for ResNets on CIFAR-10 and ImageNet they needed a few percent of training before the copies became stable. Ainsworth et al. (2023) align the units of one network to the other by solving a matching problem, and after that permutation the barrier between independent networks largely disappears: the "different basins" are mostly the same basin, relabeled.
