# prototypical networks · lab notes

numbers measured with the reference solution on the cached Olivetti faces (40 people × 10 photos, 32×32, standardized). people are split 28/12 with `split_classes` seeds 0, 1, 2 ("splits"); split $s$ also uses training seed $s$. each test person's photos are split once into 6 that test episodes may use and 4 held back (only the leaky model of experiment 1 ever sees those 4). every configuration trains for 400 1-shot episodes (Adam, $10^{-3}$, 3 queries per class) and is tested on 300 seeded 5-way episodes of the split's 12 held-out people: 1-shot with 5 queries per person, or 5-shot with 1 query (the 6-photo pool allows no more), so a 1-shot score's 95% interval is about ±0.01 and a 5-shot score's about ±0.02. "mean ± sd" is over the three splits. the MLP has 295,296 weights. the lab took 39 seconds on an idle laptop CPU. `verify.py` runs it offline on a synthetic stand-in (fewer people), which only tests the code path.

## experiment 1: what does an embedding learned on other people buy?

**what people usually predict:** raw pixels near chance (30–40%), the trained MLP far better (80–90%), and the leaky model a few points above it.

**what happens** (5-way, per split and over splits):

| embedding | trained on | 1-shot, splits 0 / 1 / 2 | 1-shot | 5-shot |
|---|---|---|---|---|
| raw pixels | nothing | 0.714, 0.819, 0.746 | 0.760 ± 0.054 | 0.938 ± 0.035 |
| MLP (20-way) | 28 training people | 0.722, 0.788, 0.825 | 0.778 ± 0.052 | 0.889 ± 0.042 |
| MLP (20-way) | all 40 people, 4 other photos per test person (leak) | 0.921, 0.948, 0.915 | 0.928 ± 0.018 | 0.977 ± 0.015 |

three surprises. raw pixels are already at 0.76 with one photo per person: Olivetti photos of one person share lighting, pose and background, so pixel distance goes a long way. the MLP's gain over pixels is small and depends on the split: +0.008, −0.031, +0.080 at 1-shot, so on one split the learned embedding is *worse* than doing nothing. at 5-shot the MLP is worse than pixels on average (0.89 vs 0.94), having been trained only with 1-shot episodes. and the leaky model is 15 points above the honest one on every split, although it never saw a single photo that a test episode uses: only 4 *other* photos of each test person, mixed into its training episodes.

**why:** 28 people with 10 photos each is very little data for a 295,000-weight MLP; it learns a little about which pixel directions separate people (lighting-invariant ones) and overfits the rest. the 1-shot-trained embedding is tuned to compare single photos; with five support photos the pixel mean is already a good prototype and the MLP's distortions cost more than they help (Snell et al. recommend matching the training shot to the test shot). the leak works because the model has learned the test people *as classes*: four photos per person are enough for its embedding to place that person's face in a well-separated cluster, and any new photo of them lands there. that is what splitting photos instead of people does to a few-shot benchmark, even with no photo shared between training and test, and it is why few-shot benchmarks fix their class splits (Vinyals et al., 2016; Triantafillou et al., 2020). in the episodes we stepped through, the errors were between people with similar glasses or lighting, and between photos of one person at very different angles.

## experiment 2: episode design or architecture?

**what people usually predict:** the convnet is the largest change, then 5-way training (a few points worse), then fewer training people.

**what happens** (differences from the 20-way MLP on the same split):

| training recipe | 1-shot | difference, splits 0 / 1 / 2 | 5-shot |
|---|---|---|---|
| MLP, 20-way (exp. 1) | 0.778 ± 0.052 | 0 | 0.889 ± 0.042 |
| MLP, 5-way | 0.733 ± 0.013 | −0.003, −0.045, −0.087 | 0.846 ± 0.024 |
| MLP, 5-way, 7 people | 0.714 ± 0.040 | −0.054, −0.045, −0.093 | 0.804 ± 0.063 |
| convnet, 5-way | 0.859 ± 0.043 | +0.088, +0.095, +0.060 | 0.968 ± 0.025 |

20-way training beats 5-way on every split, by 0.003 to 0.087 (mean 0.045). the right comparison for the 7-people row is the 5-way MLP, which differs from it only in the number of training people: −0.051, 0.000, −0.006, a mean of −0.019, small and driven by one split. the convnet gains 0.08 on average, on every split, and is the only recipe that clearly beats raw pixels at 5-shot (0.97 vs 0.94). so "episode design matters more than architecture" is half true: the number of ways is worth about half of the architecture change here, the number of training people (at 400 episodes) barely registers, and experiment 1's leak (+0.15, spurious) and experiment 3's metric scale (−0.09 without it) are as large as the architecture change or larger.

an earlier version of this lab, which trained with 5 queries per class and let test episodes use all 10 photos per test person, saw no consistent 20-way effect (+0.018, −0.018, +0.016 for 5-way over 20-way) and a larger cost for 7 people (−0.136, −0.046, −0.064 against the 5-way row). effects of a few points flip with details like these; three paired splits establish the convnet's gain and the leak, not the smaller effects.

**why:** what transfers to new people is knowledge about faces in general: which pixel directions vary between people and which within a person. a convnet builds in translation invariance and local features, so it gets more from the same 280 photos. more ways per episode make each training episode a harder discrimination with more negatives per query, so the gradient carries more information about what separates faces; with 400 episodes, 20-way training also shows the network each training person about 4× as often as 5-way training. the same exposure effect may explain why 7 people barely hurt here: 400 5-way episodes over 7 people revisit each one many times, and the MLP overfits either way. the per-split dots show why comparisons must be paired: the split-to-split range of a single method (0.72 to 0.83 for the MLP) is wider than any effect except the leak and the convnet, so unpaired means over a few splits would blur everything.

## experiment 3: which distance?

**what people usually predict:** cosine about as good as squared distance, or better because it ignores brightness-like scale; the scale factor makes little difference since it doesn't change which prototype is nearest.

**what happens:**

| logits | 1-shot, splits 0 / 1 / 2 | 1-shot | 5-shot |
|---|---|---|---|
| $-\lVert z - c\rVert^2$ (exp. 1) | 0.722, 0.788, 0.825 | 0.778 ± 0.052 | 0.889 ± 0.042 |
| cosine | 0.593, 0.742, 0.733 | 0.689 ± 0.084 | 0.811 ± 0.086 |
| 10 × cosine | 0.805, 0.810, 0.844 | 0.820 ± 0.021 | 0.936 ± 0.012 |

plain cosine is 9 points worse than squared distance on average (worse on every split) and 7 points worse than raw pixels; the same cosine times 10 is the best MLP in the lab, 4 points above squared distance (ahead on 2 of 3 splits).

**why:** on paper 4. with logits in $[-1, 1]$ and 20 classes the loss can't go below $\log(1 + 19e^{-2}) = 1.27$, so even a perfectly classified query keeps a large gradient, and the training signal is dominated by pushing easy queries towards an unreachable target instead of fixing hard ones. the scale doesn't change the nearest prototype at test time, but it changes the loss surface during training: at $s = 10$ the floor is gone and the softmax can concentrate on the errors. squared distances are unbounded, so they never had the problem, but their scale is set by the embedding's norm, which the network must learn; with cosine and a good fixed scale, the network only learns directions. Oreshkin et al. (2018) make the scale a learned parameter.
