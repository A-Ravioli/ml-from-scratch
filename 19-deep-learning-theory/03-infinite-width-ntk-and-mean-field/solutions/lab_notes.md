# infinite width: NTK & mean field · lab notes

numbers below are from the reference solution on a laptop CPU, with the seeds in the notebook (three random networks per setting). the lab uses no dataset, so online and offline runs are identical; it takes about 20 seconds.

## experiment 1: the kernel of a random network, as the width grows

**what people predict.** a common guess is that the error falls in proportion to the width (4× wider, 4× smaller), or that it reaches a floor because a finite network is "a different kind of object" from the limit.

**what happens.**

| width | relative error of the empirical NTK (3 networks) |
|---|---|
| 16 | 0.74, 0.84, 0.30 |
| 64 | 0.20, 0.16, 0.26 |
| 256 | 0.19, 0.30, 0.22 |
| 1,024 | 0.040, 0.076, 0.036 |
| 4,096 | 0.014, 0.041, 0.026 |
| 16,384 | 0.011, 0.025, 0.024 |

the fitted log-log slope is $-0.51$: the error halves for every 4× in width, with a lot of scatter between networks of the same width (a factor of up to three at 16, 256 and 4,096). the right panel shows why the smallest network is not just noisy but wrong in shape: at width 16 the kernel, plotted against the angle between the inputs, is a staircase with visible jumps and is far too high near zero angle (about 1.8 against 1.0) for that network; at width 1,024 it lies close to the infinite-width curve (slightly low at small angles).

**why.** each kernel entry is an average of `width` independent, identically distributed terms (one per hidden unit), so by the central limit theorem its fluctuation around $\Theta_\infty$ has standard deviation $\propto 1/\sqrt{\text{width}}$. the slope of $-1/2$ is that, and nothing more. the jumps come from the hidden-layer term $x\cdot x'\,\frac1m\sum_j a_j^2\,\mathbb 1[w_j\cdot x > 0]\,\mathbb 1[w_j\cdot x' > 0]$: as $x'$ rotates, each unit's indicator switches off at one angle, dropping that unit's $a_j^2$ contribution at once, so 16 units give a staircase with at most 16 steps (one per unit) on top of the continuous output-layer term, while the limit is smooth. a single unit with a large $a_j^2$ makes a large step, which is the drop near 0.6 rad in the figure. finite-width corrections to the NTK are a real subject (Hanin & Nica, 2020; Roberts, Yaida & Hanin, 2022): for deep networks they grow with depth/width, so "wide" means wide relative to depth.

## experiment 2: does a trained wide network do kernel regression?

**what people predict.** most people expect training to change a network's weights substantially (it's "learning"), and many expect the narrow and the wide networks to learn different functions.

**what happens.** learning rate $1/\lambda_{\max}(\Theta_\infty) \approx 0.12$, 6,000 full-batch steps on 16 points:

| width | final train loss | trained vs linearized (relative distance on 200 test angles) | $\lVert\theta - \theta_0\rVert / \lVert\theta_0\rVert$ |
|---|---|---|---|
| 64 | 9e-5 to 5e-3 | 0.25, 0.27, 0.35 | 0.86, 0.70, 0.75 |
| 1,024 | 3e-4 to 5e-4 | 0.16, 0.12, 0.12 | 0.19, 0.17, 0.18 |
| 16,384 | 1e-6 to 1e-5 | 0.044, 0.037, 0.039 | 0.041, 0.040, 0.038 |

the widest network moves its weights by about 4% and its trained predictions are within 4% of what the linearized model predicted before training. the distance to the linearization and the weight movement both shrink by about a factor of 4 for 16× the width, the $1/\sqrt m$ of on-paper 3. at width 64 the weights change by 70–86% of their initial norm and the trained function differs visibly from its linearization (dotted line), most of all in the gaps between training points. all networks interpolate the training points, but the predictions between them carry the random initial function $f_0$: they don't equal the infinite-width kernel regression from zero.

**why.** in the NTK parameterization every weight's influence on the output is $O(1/\sqrt m)$, so fitting $O(1)$ residuals needs only $O(1/\sqrt m)$ changes per weight, which leave the activation patterns and the kernel essentially fixed; the first-order Taylor expansion around $\theta_0$ is then accurate along the whole trajectory, and gradient descent on the network is gradient descent on a linear model with features $\nabla_\theta f(x;\theta_0)$. that is Jacot et al.'s (2018) result, and Lee et al. (2019) checked it on real architectures the same way the lab does.

## experiment 3: lazy versus rich, at the same width

**what people predict.** since both networks have 1,024 units and fit the same data, people expect similar behaviour, or that the mean-field network, with its tiny output scale, barely trains.

**what happens.** (width 1,024; learning rates 0.12 for NTK and $0.12 \times 1024/3 \approx 41$ for mean field)

| parameterization | final train loss | kernel change $\lVert K_T - K_0\rVert / \lVert K_0\rVert$ | units rotated by more than 10° |
|---|---|---|---|
| NTK | 3e-4 to 5e-4 | 0.11, 0.13, 0.12 | 26–27% |
| mean field | 8e-4 to 2e-3 | 6.7, 7.0, 6.7 | 75–77% |

both fit the data. the NTK network's tangent kernel changes by about 12%; the mean-field network's kernel grows to almost eight times its initial size ($\lVert K_T - K_0\rVert \approx 7\lVert K_0\rVert$) and three quarters of its hidden units turn by more than 10°. the learned functions look similar on this easy task (the plot), but they got there differently: one by fitting a fixed random feature map, the other by building its features. (a quarter of the NTK network's units rotate by more than 10° too: a unit whose input weight vector happens to be short turns a long way for a small change. the lazy-training argument predicts the NTK network's kernel change shrinks like $1/\sqrt m$; the lab doesn't measure that, try it.) with the full $m$-fold learning rate the mean-field network doesn't converge; the reference stalled at a training loss of 1.3.

**why.** in the mean-field scaling the output is an average over units, so no single unit's change is negligible: fitting the data requires every unit to move by $O(1)$, and the second-order terms of the Taylor expansion (the change of the gradients, i.e. of the kernel) are as large as the first. the kernel grows because the units align with the target's directions and their output weights grow, which is exactly feature learning. it is also why the mean-field limit is not a kernel method and why μP (topic 06) exists: it is the choice of per-layer scalings that keeps every layer in this "rich" regime at every width, with learning rates that transfer.
