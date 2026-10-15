# hyperparameter search · lab notes

## experiment 1: grid versus random, 9 trials each

**what people usually predict:** the grid is at least as good, being systematic; or random wins "sometimes", by luck.

**what happens:** the grid chooses $k = 5$, $p = 3$, with test MSE **0.374**. random search beats it in **95%** of 20 runs (19 of 20), with a median test MSE of 0.348, essentially the best possible (0.346, at $k = 12$). its chosen $k$ values cluster around 7–18, right in the valley of the error curve; the one run that lost picked $k = 44$.

**why:** the error depends strongly on $k$ (too small: high variance; too large: the average blurs the function) and hardly on $p$. the grid tried only three values of $k$ (5, 50, 95), and the valley at 8–20 falls between them: with 9 trials it spent 6 of them re-testing the same three $k$ values with different $p$'s. random search tried 9 different values of $k$, so it almost always lands a couple in the valley. a finer grid would fix this example, but grids scale as $m^h$: a grid with 9 values of each of 5 hyperparameters is 59,049 settings, while random search keeps its coverage of each single hyperparameter at any budget. this is Bergstra and Bengio's argument: since a few hyperparameters usually dominate (and you don't know in advance which), random sampling wastes less.

## experiment 2: the search's own score is optimistic

**what people usually predict:** about 0.5 for both, since the labels are coin flips; or that cross-validation already protects against optimism.

**what happens:** the mean `best_score_` is **0.565** (ranging up to about 0.61), while nested CV gives **0.483**, close to the truth, 0.5, with a wide spread (about 0.38 to 0.59).

**why:** each of the 20 candidates' CV accuracies is an honest but noisy estimate of 0.5; with 80 points, a CV accuracy has a standard error of several points. the search reports the *maximum* of 20 such numbers, which sits systematically above 0.5: the winner's curse (on paper 3). cross-validation protects each candidate's estimate, not the act of choosing among them. nested CV moves the choice inside the outer folds: each outer test fold is scored by a winner that was picked without it, so the outer scores estimate the whole procedure honestly. their spread is large because each outer fold has only 16 points, and the tuning itself adds variance (different folds pick different $k$): an honest estimate on a small dataset is an *uncertain* one, which is the point.

## experiment 3: how many random trials do you need?

**what people usually predict:** it depends on the number of hyperparameters (more dimensions, more trials), maybe hundreds.

**what happens:** the measured probability follows $1 - 0.95^n$ almost exactly: 0.23 at 5 trials, 0.39 at 10, 0.64 at 20, 0.78 at 30, 0.90 at 45, 0.94 at 60.

**why:** each random trial lands in the top 5% with probability 0.05, independently, so all $n$ miss with probability $0.95^n$. nothing in that argument mentions the dimension: the "top 5%" is a fraction of the sampling distribution, however many hyperparameters define it. about 60 trials give 95% confidence, which is why "60 random trials" is a common default. the guarantee is only as good as the space: if the ranges or scales are wrong, the top 5% of that space can still be poor, and the scores themselves are noisy CV estimates (experiment 2), so "top 5% by CV" isn't exactly "top 5% on new data". a dense grid of 500 settings, used here as the reference, costs 2,500 fits; 60 random trials cost 300.
