# support vector machines · lab notes

numbers from the reference solution with the lab's seeds and the default slider value (noise 0.25); experiment 3 uses the real MNIST digits in `~/.cache/mlfs`.

## experiment 1: sweeping C and γ

**what people usually predict:** that large $C$ and large $\gamma$ overfit, small ones underfit, and that the number of support vectors grows with model complexity (more flexible model, more support vectors).

**what happens:**

| | $\gamma = 0.1$ | $\gamma = 1$ | $\gamma = 30$ |
|---|---|---|---|
| $C = 0.1$ | train 82%, test 77.7%, 147 SV | 90%, 87.6%, 115 SV | 57%, 53.6%, 189 SV |
| $C = 1$ | 86%, 83.3%, 88 SV | 97%, 92.4%, 53 SV | 98%, 91.9%, 139 SV |
| $C = 100$ | 92%, 90.0%, 55 SV | 98%, **92.7%**, 20 SV | 100%, 90.5%, 120 SV |

$\gamma = 0.1$ gives a nearly straight boundary (the kernel's reach is longer than the moons), $\gamma = 30$ wraps the boundary around individual points, $\gamma = 1$ follows the moons. the best test accuracy, 92.7%, is at $C = 100$, $\gamma = 1$, with only **20** support vectors. the worst setting is $C = 0.1$, $\gamma = 30$: 54% test accuracy, essentially predicting one class everywhere, with 189 of the 200 points as support vectors.

**why:** $\gamma$ controls the shape: each support vector's influence decays like $e^{-\gamma r^2}$, so a large $\gamma$ makes $f$ a sum of narrow bumps. $C$ controls the margin: small $C$ makes margin violations cheap, so the margin is wide and many points sit inside it, at $\alpha_i = C$, as bound support vectors. that's why the number of support vectors *falls* as $C$ grows along a row (147, 88, 55 at $\gamma = 0.1$): a narrow margin contains fewer points. it grows with $\gamma$ because narrow bumps can only "see" points near the boundary if there are many of them. at $C = 0.1$, $\gamma = 30$ both effects combine: each bump is tiny and capped at $\alpha = 0.1$, so $f$ is the bias $b$ almost everywhere, and the classifier predicts the majority side except right on top of minority points. $C = 100$, $\gamma = 30$ is the classic overfit: 100% training accuracy with a boundary that encircles small groups of points, and 90.5% on test data.

## experiment 2: only the support vectors matter

**what people usually predict:** throwing away 70% of the training data must change the model noticeably; removing one point can't matter much.

**what happens:** of 200 points, 60 are support vectors. refitting on those 60 alone changes the decision function by at most **0.012** anywhere on the grid, and flips **0.10%** of grid predictions. removing the single free support vector with the largest $\alpha$ changes it by up to **0.227**, 19 times more, and flips **0.50%**. the 15-NN classifier does the opposite: training on the support vectors only flips **11.7%** of its grid predictions, removing the one point **0.2%**.

**why:** the KKT conditions say points with $\alpha_i = 0$ contribute nothing to $f = \sum_i\alpha_i y_i k(x_i, \cdot) + b$, and the optimal $\alpha$ for the full data is still optimal for the support vectors alone (on paper 3). so the true optimum doesn't change at all. the 0.012 that remains is solver precision: SMO stops when KKT holds to `tol` and no pair can make progress, and the two runs stop at slightly different points. removing a support vector removes a constraint that was active at the optimum, so the solution has to move. kNN has no such sparsity: every training point is part of the model, and deleting the 140 points away from the boundary removes most of the "votes" in the interior, so large regions change. (deleting a single point barely matters to kNN, since each prediction pools 15 votes.) this sparsity is also what bounds the SVM's leave-one-out error by the fraction of support vectors.

## experiment 3: 4s versus 9s

**what people usually predict:** 10% error or more, since 4 and 9 look alike; and that the support-vector bound is a reasonable estimate of the error.

**what happens:** with 1,000 training digits and all 1,991 test 4s and 9s, the test error is **4.9%** (± 0.5% standard error) at $C = 0.1$ and **2.6%** (± 0.4%) at $C = 1$, 10 and 100. the fraction of support vectors falls from **74%** to **38%** and then **36%**, and the training error from 2.9% to 0.2% and then 0.0%. the bound $\#SV/n$ is valid (36% is above 2.6%) but about 14 times larger than the real error.

**why:** small $C$ underfits: it accepts training errors to keep a wide margin, and most points end up inside it (74% support vectors). the data are separable in the RBF feature space: at $C = 1$, 105 support vectors still sit at the bound $\alpha = C$, but from $C = 10$ on none do (the largest $\alpha$ is 6.06), so the solution is the hard-margin one and identical for $C = 10$ and $C = 100$. that's why the curves go flat. the bound is loose because it counts every support vector as a potential leave-one-out mistake, while in fact most of them, left out, are still classified correctly; it's a guarantee, not an estimate. high-dimensional data like images typically have many support vectors (a third of the training set here), which is also what makes kernel SVMs slow at prediction time for large data.
