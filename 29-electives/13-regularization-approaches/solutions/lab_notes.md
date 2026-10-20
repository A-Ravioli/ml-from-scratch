# regularization approaches · lab notes

numbers measured with the reference solution on the cached real MNIST, the block's 784–200–200–10 MLP, Adam at $10^{-3}$, batch 64, one epoch per task, EWC Fishers from 200 examples (exact over classes). the lab takes about two minutes on a laptop CPU (two and a half on a busy one). uncertainties are standard deviations over seeds. `verify.py` runs the lab offline on synthetic stand-in data, which only tests the code path.

## experiment 1: the stability–plasticity tradeoff

**what people usually predict:** that a stronger penalty is better until "enough", then flat; or that ACC rises monotonically with $\lambda$.

**what happens** (five permuted-MNIST tasks, seed 0):

| method | strength | ACC | BWT | learning accuracy |
|---|---|---|---|---|
| fine-tuning | 0 | 0.748 | −0.166 | 0.881 |
| EWC | 10 | 0.801 | −0.093 | 0.876 |
| EWC | 100 | **0.820** | −0.042 | 0.853 |
| EWC | 1,000 | 0.789 | −0.015 | 0.801 |
| EWC | 10,000 | 0.717 | −0.004 | 0.720 |
| EWC | 100,000 | 0.652 | −0.004 | 0.655 |
| SI | 1 | 0.765 | −0.142 | 0.879 |
| SI | 10 | **0.807** | −0.075 | 0.867 |
| SI | 100 | 0.802 | −0.023 | 0.820 |
| SI | 1,000 | 0.677 | −0.015 | 0.689 |

over seeds 0–2: fine-tuning 0.760 ± 0.010, EWC $\lambda = 100$ 0.819 ± 0.002, EWC $\lambda = 10^4$ 0.726 ± 0.010, SI $c = 10$ 0.807 ± 0.005, SI $c = 100$ 0.805 ± 0.003. the curves cross: as the strength grows, BWT climbs from −0.17 to essentially 0 while the learning accuracy falls from 0.88 to 0.66. ACC is the product of the two pressures and peaks in between, about 6 points above fine-tuning. past $\lambda = 10^4$, EWC forgets nothing (BWT −0.004) and ACC is *worse than fine-tuning*: the network can barely learn tasks 3–5.

**why:** the penalty makes the important weights stiff; the stiffer they are, the less of the network is free for the new task. on the left of the peak, ACC is limited by forgetting; on the right, by what the network is still allowed to learn. on permuted MNIST every task wants the hidden layers to do something different, so they're in direct competition and the tradeoff is sharp. the useful ranges differ by about an order of magnitude ($\lambda \approx 100$ vs $c \approx 10$–$100$) because the numbers are in different units: EWC's Fisher is per example at the end of a task; SI's $\Omega$ is loss decrease per squared distance travelled, accumulated along the path. neither is "stronger"; only the product of strength and importance matters, and it has to be tuned per method and per benchmark. (with plain SGD at $\eta = 0.05$ the right half of this plot doesn't exist: runs with $\lambda \ge 100$ diverge to NaN, the step-size limit of the lesson's on paper 3.)

## experiment 2: the same methods, a different scenario

**what people usually predict:** EWC and SI help in both scenarios, perhaps less class-incrementally (0.4–0.6).

**what happens** (split MNIST, seeds 0–2):

| method | task-incremental ACC | class-incremental ACC |
|---|---|---|
| fine-tuning | 0.917 ± 0.024 | 0.196 ± 0.000 |
| EWC $\lambda = 10^4$ | 0.983 ± 0.005 | 0.227 ± 0.015 |
| SI $c = 100$ | 0.986 ± 0.002 | 0.208 ± 0.021 |

task-incrementally both methods remove most of the forgetting: 0.98–0.99, at the level of joint training. class-incrementally they're within 3 points of fine-tuning, and in a sweep over $\lambda \in \{10^2, 10^4, 10^6\}$ and $c \in \{1, \ldots, 1000\}$ (seed 0) the best was EWC at $\lambda = 10^6$ with 0.31, bought by learning the last task worse (0.84 instead of 0.98).

**why:** in the class-incremental scenario the damage isn't done by moving task 0's important weights; it's done at the output, by raising the logits of classes 2–9 on *every* input until they beat 0 and 1. when task 0 is consolidated, the output units of classes 2–9 have never been trained; task 0's Fisher for their weights is tiny (the model gives those classes probability near zero on 0s and 1s, and a log-probability's gradient with respect to an output weight of class $k$ is proportional to $[y = k] - p_k$, which is almost zero for every $y$ the model would sample), so the penalty leaves them free, and they're exactly what task 1 trains. a penalty on weights protects "the function task 0 computed, restricted to task 0's two outputs", not "task 0's outputs keep winning against outputs that don't exist yet". the task-incremental head never faces that competition, so protecting the shared features is enough. van de Ven & Tolias (2019) reported the same split on the same benchmark; it's the reason the field reports scenarios explicitly.

## experiment 3: which weights matter?

**what people usually predict:** loose agreement between the Fisher and SI (correlation 0.3–0.6), and that large weights are the important ones.

**what happens** (one network trained on permuted task 0; seed 0 in the lab, seeds 0–2 here):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Spearman ρ, Fisher vs SI Ω, all weights | 0.955 | 0.952 | 0.953 |
| … over weights where both are positive | 0.921 | 0.916 | 0.918 |
| Spearman ρ, Fisher vs \|weight\| | 0.198 | 0.199 | 0.193 |
| weights with Fisher exactly 0 (of 199,210) | 55,582 | 58,750 | 57,194 |
| task-0 accuracy, no noise | 0.879 | 0.884 | 0.884 |
| noise on Fisher top 1% | 0.762 | 0.526 | 0.762 |
| noise on SI top 1% | 0.759 | 0.530 | 0.754 |
| noise on magnitude top 1% | 0.864 | 0.854 | 0.879 |
| noise on random 1% | 0.880 | 0.875 | 0.883 |
| noise on Fisher bottom 1% (of F > 0) | 0.879 | 0.884 | 0.885 |

the two importance measures agree almost perfectly in rank (0.95 overall, 0.92 even after removing the weights both call zero), although one is a curvature at the end of training and the other a sum along the path. perturbing the top 1% by either measure costs 12–36 points of accuracy; the same noise on the largest weights costs 0.5–2.5 points, on random weights nothing measurable. about 85% of the top-1% Fisher weights are output-layer weights (1,623–1,680 of 1,992); the 55,000–59,000 zero-Fisher weights are mostly first-layer weights of pixels that are black in all 500 Fisher examples (205 such pixels for seed 0, so 41,000 weights), plus the weights of a few dozen hidden units that never fire on those examples (9 in the first layer and 22 in the second, for seed 0).

**why:** for gradient descent near a minimum, a weight's contribution to the loss decrease is roughly its curvature times its squared displacement (the lesson's on paper 4), so SI's $\Omega = \omega / \Delta^2$ estimates curvature, and so does the Fisher; two routes to one quantity. magnitude is a different property altogether: a large first-layer weight on a pixel that rarely matters for the decision is large and irrelevant, while a modest output-layer weight is read by every prediction. the bottom-1% row shows how much of the network task 0 doesn't care about at all, which is the room EWC and SI leave for later tasks. the seed-to-seed spread in the top-1% rows (0.53 vs 0.76) comes from which output weights the fixed noise vector happens to hit hardest; the ordering of the rows is the same in every seed.
