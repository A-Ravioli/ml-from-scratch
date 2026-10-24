# attribution & probing · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU), with MNIST from the cache: a LeNet trained for two epochs on the first 12,000 training digits (Adam, lr $2 \times 10^{-3}$), test accuracy **0.943** on the first 2,000 test digits. the notebook runs in about 25 seconds. where the notes quote a spread, it comes from two more training seeds (1 and 2, test accuracy 0.934 and 0.936) run with the same code outside the notebook. offline the notebook runs on a synthetic stand-in that LeNet classifies perfectly, so its numbers mean nothing.

## experiment 1: do the attributions add up?

**what people predict.** that a handful of steps (5 or 10) is enough, since IG is "just an average gradient"; and that saliency and IG highlight the same pixels.

**what happens.** the mean relative completeness error over 32 digits:

| n_steps | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| error | 0.504 | 0.235 | 0.076 | 0.013 | 0.008 | 0.004 | 0.002 | 0.001 |

it takes about 16 steps to get under 1%, and beyond 8 steps the error halves each time $n$ doubles: $O(1/n)$, not the $O(1/n^2)$ the trapezoid rule promises for smooth integrands. the maps differ sharply: saliency is speckled over the whole image, background included; gradient × input and IG are exactly zero off the stroke and look like the digit with red (positive) and occasional blue (negative) segments.

**why.** LeNet is piecewise linear (ReLU and max pooling), so along the straight path from black to the digit its gradient is piecewise constant, jumping every time a unit switches on or off or a pooling winner changes. the trapezoid rule's error for a jump is up to half a step times the jump, so the error is $O(1/n)$ (on paper 3). the first few doublings fall faster because a few big jumps dominate until the grid resolves them. on the maps: the gradient at a black background pixel is generally not zero (brightening it would change the logit), but its *input* is zero, and both gradient × input and IG with a black baseline multiply by $x - x' = 0$ there. which to trust depends on the question: saliency answers "what would change the output", IG "what made the output differ from a black image's". the knobs under the curve redraw the IG maps for any number of steps and two other baselines: with the mean digit or a blurred copy as the baseline, IG attributes the *difference* from an average or a smudged digit, so background pixels where the baseline is grey get nonzero attributions and the maps stop looking like a stroke-shaped mask (64 steps, black baseline: completeness error 0.0017 on the four digits shown).

## experiment 2: sanity checks for saliency maps

**what people predict.** that all three maps turn into noise once every layer is random, or (having read the lesson) that IG fails the test and saliency passes.

**what happens.** Spearman correlation of $|$map$|$ with the trained model's, 16 digits, cumulative randomization from the top:

| method | +fc3 | +fc2 | +fc1 | +conv2 | +conv1 (all random) | all random, stroke pixels only |
|---|---|---|---|---|---|---|
| saliency | 0.64 | 0.63 | 0.50 | 0.41 | 0.42 | 0.07 |
| gradient × input | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.52 |
| IG (16 steps) | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.51 |

with seeds 1 and 2 the fully randomized saliency correlation was 0.29 and 0.28 (stroke only 0.03 and 0.08), gradient × input and IG 0.99 in both (stroke only 0.46–0.51). the pictures agree: the random network's saliency map is uniform speckle, while its gradient × input and IG maps are a clean outline of the digit, almost indistinguishable from the trained model's.

**why.** two effects stack up. first, 81% of MNIST pixels are exactly 0 (83% in these 16 digits), so any method that multiplies by $x$ produces maps with the same huge block of tied zeros, whatever the model; ranks of tied zeros agree perfectly, and that alone pushes the whole-image correlation to 0.99. second, even on the stroke, $|x_i g_i|$ inherits the brightness pattern of $x$: correlation 0.5 with a *random* model. the saliency map survives partially (0.4 overall) only because a random conv net's gradients are still spatially structured around the input; on the stroke alone it falls to about 0. none of this makes IG violate its axioms (completeness holds for the random model too), but it shows that "the map looks like the digit" is evidence about the input, not about the model. a fairer comparison restricts to the pixels where maps can differ, or compares the *decisions* an attribution supports (deletion curves, for instance).

## experiment 3: what is a probe finding?

**what people predict.** that a probe for parity on a late layer is very accurate and the random grouping is much worse, near chance, because the network was never trained on "random groupings".

**what happens.** test accuracy of linear probes (1,500 training and 1,000 test digits):

| layer | features | parity | random grouping, mean of 5 | worst of 5 | selectivity |
|---|---|---|---|---|---|
| pixels (4×4 pooled) | 49 | 0.846 | 0.847 | 0.783 | −0.001 |
| conv1 | 294 | 0.872 | 0.858 | 0.796 | +0.014 |
| conv2 | 64 | 0.876 | 0.871 | 0.827 | +0.005 |
| fc1 | 120 | 0.931 | 0.924 | 0.893 | +0.007 |
| fc2 | 84 | 0.951 | 0.943 | 0.934 | +0.008 |
| fc3 (logits) | 10 | 0.909 | 0.908 | 0.884 | +0.001 |

parity is decodable, increasingly so with depth, and an arbitrary grouping of the digits is decodable just as well at every layer: selectivity is within ±0.015 of zero everywhere. at fc2, seeds 1 and 2 gave parity 0.928 and 0.951 against random groupings 0.930 and 0.939.

**why.** the network learned to separate the ten classes; any linear readout that separates the classes can also be thresholded into any union of classes, so *every* binary function of the class is about equally decodable. the probe has found "which digit is this", and parity is just one of the $2^{10}$ functions of it. the spread across groupings is larger in early layers (worst 0.78 vs mean 0.85 on pixels) because there some pairs of digits are much harder to separate linearly than others (a 4 and a 9), and a grouping that splits such a pair suffers. the logits layer is a little worse than fc2 because it has only 10 dimensions and was trained to rank, not to be linearly regrouped. to claim "fc2 represents parity" you'd need parity to be *more* decodable than control groupings, or better, an intervention showing the model uses a parity direction, which it doesn't, since it was never trained to.
