# gradient boosting · lab notes

experiments 1 and 2 were run on the real (cached) California housing data: 5,000 training and 5,000 test districts from a fixed permutation (seed 0). the booster is deterministic with `subsample=1`, so your numbers should match these closely; offline (`MLFS_OFFLINE=1`, as `verify.py` runs it) you get a 1,000-row synthetic linear problem instead, where the numbers are different and the bin experiment says little. experiment 3 is synthetic in both modes. the whole lab takes about 35 seconds on a laptop CPU.

## experiment 1: learning rate versus number of trees

**what people usually predict:** that the learning rate only changes the speed: every rate eventually reaches the same test error, small rates just need more trees. and that learning rate 1 ("full Newton steps") should be best, since it's what the derivation optimizes.

**what happens:**

| learning rate | best test RMSE | at tree | at the last tree |
|---|---|---|---|
| 1.0 | 0.5802 | 23 | 0.6160 (200 trees) |
| 0.3 | 0.5149 | 186 | 0.5183 (600 trees) |
| 0.1 | 0.4985 | 792 | 0.4998 (2,000 trees) |
| 0.03 | 0.4961 | 1,946 | 0.4962 (2,000 trees) |

learning rate 1 bottoms out after 23 trees at 0.580 and then gets steadily worse; it never comes close to the others. each tenfold reduction of the rate buys a lower minimum (0.580 → 0.499 → 0.496), with diminishing returns, at the price of roughly proportionally more trees (23, 186, 792, ~2,000).

**why:** with $\eta = 1$ each tree fully fits the current residuals with a depth-4 tree. the first trees capture the big structure, but every tree also fits the noise in its leaves, and with full steps that noise is committed immediately and never undone: later trees fit residuals that are increasingly noise. with a small $\eta$, each tree moves the model only a little, so the noise fitted by one tree is mostly averaged out by the next ones (they see slightly different residuals), and many small steps can follow the true function more closely than a few big ones. shrinkage is a regularizer, not just a step size: Friedman (2001) found the same, and it's why everyone uses 0.01–0.1 plus early stopping. in practice, fix the learning rate as small as your time budget allows and choose the number of trees on a validation set; the product $\eta\times M$ at the optimum is roughly constant (about 25–80 here). note that at 0.03 the minimum is still at the very end: it would improve slightly with more trees.

## experiment 2: how many bins?

**what people usually predict:** after reading about histogram boosting, that 16 or 32 bins are as good as 256 ("boosting doesn't care about precise thresholds").

**what happens:** on California the bins matter more than that: 2 bins give 0.933, 4 give 0.744, 8 give 0.647, 16 give **0.564**, 32 give 0.524, 64 give 0.511, and only **128** bins (0.5024) are within 1% of 256 (0.5027).

**why:** two things set how many thresholds you need. a smooth, monotone effect (like median income on house value) is well approximated by a staircase with a few dozen steps, and boosting adds up hundreds of such staircases at different offsets, so it needs few bins. but California's strongest structure is *location*: prices jump at the coast and around the Bay Area and Los Angeles, which are narrow ranges of longitude and latitude. quantile bins are narrow where districts are dense (0.06–0.1° of latitude around Los Angeles) but wide elsewhere: with 16 bins, two latitude bins between Los Angeles and the Bay Area are 1.1 and 1.5 degrees wide, and longitude bins reach 1.1 degrees, so many jumps can't be placed where they are. so the folklore ("bins barely matter") holds for smooth features, not for sharp ones; 256 bins is the default in XGBoost, LightGBM and scikit-learn because it's essentially lossless and the cost of more bins is only in the split scan. two bins means one threshold per feature (the median) for the whole model, a very crude additive model, hence 0.93. offline the data is linear, every feature is smooth, and the curve is noisy at a tenth of the sample size: don't read anything into it.

## experiment 3: what does the second derivative buy?

**what people usually predict:** that the second derivative is a refinement worth a few percent, or that gradient leaves catch up after a few more trees.

**what happens:** with only 7.7% positives, the Newton version reaches a test log loss within 1% of the best either version achieves (0.1870) after **16** trees, and its minimum is **0.1852** at tree 54. the gradient version needs **322** trees to reach its best, **0.1877**, and never gets within 1% of the Newton minimum in 400 trees. after its minimum the Newton version overfits badly (0.2409 at 400 trees), while the gradient version is still slowly improving.

**why:** most rows have $p \approx 0.08$, so $h = p(1 - p) \approx 0.07$. a gradient leaf moves by $\overline{t - p}$, the mean residual, which in a leaf of mostly negatives is a tiny number like $-0.05$ in log-odds: the model creeps. a Newton leaf divides the same sum by $\sum h$ instead of the row count, i.e. by about 0.07 per row, so its step is about 14 times larger, exactly the step that minimizes the local quadratic approximation of the log loss. in leaves that collect the rare positives the difference is larger still. so the gradient version isn't wrong, just badly scaled: the hessian is the automatic, per-leaf step size. the flip side is the overfitting. we set `min_child_weight=1e-3` for both versions so that only the leaf formula differs; then Newton leaves happily form in tiny regions where $\sum h$ is minute and $-G/(H + \lambda)$ is huge, and with learning rate 0.3 those big steps fit noise fast. with the default `min_child_weight=1` (a leaf needs about 14 rows' worth of hessian at $p = 0.08$) and $\lambda$, that's held in check; and this is why XGBoost has `min_child_weight` at all.
