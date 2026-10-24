# in-context learning as meta-learning · lab notes

numbers measured with the reference solutions (this topic, and topic 09's MAML), $D = 5$, sequences of 15 pairs, 2,000 test tasks per test set (fixed seeds 100 and 300–302), shared by every predictor. the transformer (3 blocks, $d_\text{model} = 32$, 4 heads, 39,393 parameters) trains for 2,500 steps of 64 fresh noiseless tasks with Adam at $2\times10^{-3}$; by default the lab trains it once (seed 0) in 50–100 s depending on machine load, and the whole lab took 64 s on a laptop CPU. ticking the lab's checkbox trains seeds 1–4 as well (285 s in total on the same machine) and adds the "mean ± sd over seeds" rows quoted below. "± se" is the standard error over test tasks. everything is synthetic, so `verify.py` runs exactly this lab (with the box unticked).

## experiment 1: what does one layer of attention learn?

**what people usually predict:** that the trained weights will be an unstructured mess that happens to work, or that a single attention layer will approximate least squares.

**what happens:** the effective preconditioner $cA + ub^\top$ (on paper 2) is a multiple of the identity: diagonal 0.0577–0.0604, largest off-diagonal entry 0.0017, against the best single step size $\eta^* = 1/16 = 0.0625$. the rows $u$ and $b$ that would add a rank-1 part have norms 0.009 and 0.010. the trained layer's test error with 10 examples is 1.869, one GD step with $\eta^*$ gives 1.847 (on paper 3 predicts 1.875), and the two predictors differ by a mean squared 0.023, about 1% of their error. least squares gets 0.000.

**why:** the prediction of one linear-attention layer is $\sum_j (u^\top x_j + c\,y_j)(x_j^\top A x_q + y_j\, b^\top x_q)$. the two terms linear in $y_j$ combine into $\sum_j y_j\, x_j^\top (cA + ub^\top)\, x_q$: one step of GD preconditioned by $cA + ub^\top$, so a cross term $ub^\top$ could contribute a rank-1 preconditioner and parity alone does not remove $u$ or $b$. what does remove them is the other two terms: $u^\top x_j\, x_j^\top A x_q$ contains no $y$, and $c\,y_j^2\, b^\top x_q$ is even in $w$, so neither correlates with the target $w^\top x_q$, and both add pure variance that grows with $\lVert u\rVert$ and $\lVert b\rVert$. training shrinks them, leaving $cA$; by rotational symmetry of the task distribution the best $cA$ is a multiple of the identity, $\eta^* I$. the slight shrinkage of the diagonal below $1/16$ is the finite-training version of that optimum. one step is a weak algorithm (warm-up 2): the error stays near 1.9 because a single step can't undo the correlations in $X^\top X$, which least squares inverts exactly.

## experiment 2: a transformer learns to regress in context

**what people usually predict:** a smooth decline that reaches least squares' 0 past $D$ examples, or a model that just predicts the mean.

**what happens** (seed 0, the lab's run, test seed 100):

| predictor | 0 ex. | 1 ex. | 2 ex. | 3 ex. | 4 ex. | 5 ex. | 6 ex. | 10 ex. | 14 ex. |
|---|---|---|---|---|---|---|---|---|---|
| predict 0 | 5.322 | 5.200 | 5.168 | 5.167 | 5.071 | 4.816 | 4.875 | 5.113 | 4.890 |
| one GD step (best η) | 5.322 | 4.291 | 3.712 | 3.349 | 3.107 | 2.701 | 2.471 | 1.850 | 1.541 |
| least squares | 5.322 | 3.983 | 2.980 | 2.047 | 1.099 | 0.000 | 0.000 | 0.000 | 0.000 |
| transformer | 5.325 | 4.126 | 3.238 | 2.712 | 2.246 | 1.666 | 1.390 | 0.804 | 0.717 |

with 1 and 2 examples the transformer is close to least squares (4.13 vs 3.98, 3.24 vs 2.98); then it falls behind as the context approaches $D$: 2.71 vs 2.05 at 3 examples, 2.25 vs 1.10 at 4, 1.67 vs 0 at 5. it is better than one GD step at every length. over five training seeds (test seed 300) its error at 2, 5, 10 and 14 examples is 3.19 ± 0.07, 1.40 ± 0.25, 0.51 ± 0.20 and 0.42 ± 0.20: seed 0 is the worst of the five (0.72 at 14 examples; the best, seed 3, reaches 0.22).

**why:** below $D$ examples nothing can beat $D - i$ (on paper 1: 4, 3, 2, 1 for $i$ = 1 to 4, which least squares hits). with one or two examples the optimal prediction is easy to approximate: project the query onto the span of the few context inputs, which a weighted sum over the context, like attention, does well. as $i$ approaches $D$, least squares exploits the exact constraints by inverting $X^\top X$, which is badly conditioned when $X$ is nearly square: the small-eigenvalue directions carry the information, and both one GD step and a briefly trained network handle them poorly. past $D$ least squares becomes exact, and the transformer's error keeps falling but slowly, because the network hasn't finished training: the training curve is still going down at step 2,500 (Garg et al. train for 500,000 steps), and the run-to-run spread, a factor of three at 14 examples, shows the learning is far from converged. four runs with a short curriculum (8 pairs, then 15; seeds 0–3) and three with 4,000 steps did not change that picture: errors at 14 examples of 0.18–0.68 with the curriculum (three of four runs good, one stuck, as without it) and 0.35–0.56 with 4,000 steps. it learned *a* regression procedure from random tasks alone; whether it is least squares is the project's question.

## project: in-context learning vs least squares vs MAML

**what people usually predict:** that the transformer, a far more flexible model, matches or beats MAML; that noise hurts everyone a little; that larger inputs hurt the learned methods.

**what happens** (MAML meta-trained with seed 0 and tuned per context length on 2,000 validation tasks; the transformer row is seed 0, the extra row seeds 0–4; ± standard error over test tasks, or sd over seeds):

| test set | predictor | 2 examples | 5 examples | 10 examples | 14 examples |
|---|---|---|---|---|---|
| noiseless | least squares | 2.949 ± 0.124 | 0.000 | 0.000 | 0.000 |
| | MAML (tuned) | 3.316 ± 0.135 | 1.160 ± 0.061 | 0.036 ± 0.004 | 0.004 ± 0.001 |
| | transformer, seed 0 | 3.258 ± 0.134 | 1.723 ± 0.085 | 0.769 ± 0.040 | 0.719 ± 0.042 |
| | transformer, seeds 0–4 (sd) | 3.188 ± 0.072 | 1.403 ± 0.248 | 0.511 ± 0.195 | 0.423 ± 0.198 |
| noise σ = 0.5 | least squares | 3.593 ± 0.159 | 1203.9 ± 636.9 | 0.552 ± 0.022 | 0.442 ± 0.014 |
| | ridge λ = σ² (oracle) | 3.552 ± 0.155 | 1.398 ± 0.056 | 0.504 ± 0.018 | 0.431 ± 0.014 |
| | MAML (tuned) | 3.988 ± 0.165 | 1.850 ± 0.070 | 0.598 ± 0.022 | 0.482 ± 0.015 |
| | transformer, seed 0 | 3.819 ± 0.166 | 2.115 ± 0.086 | 1.130 ± 0.045 | 1.154 ± 0.052 |
| | transformer, seeds 0–4 (sd) | 3.757 ± 0.054 | 1.965 ± 0.107 | 0.931 ± 0.132 | 0.886 ± 0.183 |
| inputs × 2 | least squares | 11.593 ± 0.501 | 0.000 | 0.000 | 0.000 |
| | MAML (tuned) | diverges | diverges | diverges | diverges |
| | transformer, seed 0 | 13.814 ± 0.583 | 7.722 ± 0.401 | 4.461 ± 0.246 | 4.492 ± 0.228 |
| | transformer, seeds 0–4 (sd) | 13.290 ± 0.392 | 6.787 ± 0.841 | 3.696 ± 0.869 | 3.427 ± 0.846 |

because every predictor sees the same 2,000 tasks, differences are better judged per task. paired differences from the seed-0 transformer (± standard error of the mean difference):

| test set | difference | 2 examples | 5 examples | 10 examples | 14 examples |
|---|---|---|---|---|---|
| noiseless | least squares − transformer | −0.309 ± 0.055 | −1.723 ± 0.085 | −0.769 ± 0.040 | −0.719 ± 0.042 |
| | MAML − transformer | +0.058 ± 0.056 | −0.563 ± 0.064 | −0.734 ± 0.039 | −0.715 ± 0.042 |
| noise σ = 0.5 | ridge (oracle) − transformer | −0.266 ± 0.043 | −0.716 ± 0.061 | −0.626 ± 0.041 | −0.723 ± 0.049 |
| | MAML − transformer | +0.169 ± 0.060 | −0.265 ± 0.061 | −0.532 ± 0.043 | −0.672 ± 0.049 |

compute: the transformer saw 160,000 training tasks in about a minute; MAML's 8 meta-training runs plus tuning took 3–4 s; least squares needs none.

**the findings:**

1. **on the training distribution, least squares wins everywhere, and MAML beats the transformer from $D$ examples on.** tuning chose 50 test-time steps at almost every length, with the inner learning rate growing with the context (0.03 at 1–2 examples, 0.2 at 10–14): MAML is running gradient descent from an init near 0 towards the minimum-norm solution (topic 09, on paper 2), and with 50 steps it gets to within 0.004 of it at 14 examples. the paired differences are many standard errors from 0 at 5, 10 and 14 examples (e.g. −0.73 ± 0.04 at 10), and the five-seed row shows no seed comes close (0.31–0.77 at 10 examples against MAML's 0.036). at 2 examples the paired difference MAML − transformer is +0.058 ± 0.056: about one standard error, so seed 0 and MAML are indistinguishable there; the other four seeds average 0.09 lower than seed 0 there (3.17 vs 3.26), which suggests a small real edge for the transformer with very short contexts, not established by this lab. least squares is ahead of the transformer by 0.31 ± 0.06 even at 2 examples.
2. **least squares' interpolation peak is real and huge**: at exactly $D = 5$ noisy examples its mean error is 1,204, with a standard error of 637, because the expected error is infinite there (on paper 4) and a handful of nearly singular tasks dominate the mean. ridge with the true $\sigma^2$ is best everywhere. MAML, tuned on *noiseless* validation tasks, still lands close to ridge (1.85 vs 1.40 at 5 examples, 0.48 vs 0.43 at 14), because 50 steps of GD from 0 with a moderate step size is early-stopped least squares, a form of ridge. the transformer **doesn't** have the peak either (1.83–2.12 at 5 examples over seeds), so it isn't computing least squares; but past $D$ it is the worst of the learned methods, roughly twice the oracle's error (paired differences of −0.6 to −0.7, about 15 standard errors).
3. **inputs twice as large break MAML completely and the transformer badly; least squares doesn't notice.** MAML's step sizes were tuned for unit-scale inputs; the loss's curvature grows by 4, so $\alpha \lambda_{\max} > 2$ and the inner loop diverges. this is a tuning failure, not a limit of MAML: for the bias-free part of the model, GD on the scaled problem with step size $\alpha/4$ follows the original iterates, so a re-tuned MAML (with its init near 0, bias aside) would score approximately 4 times its in-distribution error. the transformer's error grows by 6–15× across seeds at 14 examples, more than the 4× that a scale-equivariant algorithm would show. it learned a procedure that works on the inputs it was trained on.

**what this supports, and what it doesn't:** the lab supports "a transformer trained on random linear tasks learns a regression procedure in context that extracts most of the information with very few examples". it does not support "the transformer implements least squares": it falls behind as the context approaches $D$, misses least squares past $D$ by an order of magnitude at this budget, lacks the interpolation peak, and fails the scale test. with 200× the training (Garg et al.) the in-distribution gap closes, but Garg et al. also report degradation under input scaling, consistent with finding 3. the comparison with MAML is fair in MAML's favour (per-length tuning, a model class that contains the truth, 50 test-time steps); a transformer gets one model for every length and no test-time optimization. the default lab run uses one transformer seed, and it is the worst of five; the checkbox and the seed rows above correct that. the MAML side was measured with one meta-training seed, which matters little because its learned init is close to 0 for every hyperparameter setting.
