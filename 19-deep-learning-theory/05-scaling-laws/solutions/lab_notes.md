# scaling laws · lab notes

numbers below are from the reference solution on a laptop CPU, on the real tiny Shakespeare, with the seeds in the notebook. the lab takes about two minutes on an idle laptop (the reference measured 113 s on a heavily shared one). the offline stand-in (random common words) gives lower losses; the qualitative patterns are the same.

## experiment 1: predict the biggest model before you train it

**what people predict.** most people expect a power law fitted over two orders of magnitude to predict a model only 2.4× larger very accurately, and many expect the simpler no-offset fit to be safer ("fewer parameters, less overfitting").

**what happens.** (150 steps of 32 windows of 32 characters, learning rate $3 \times 10^{-3}$, two seeds)

| model | non-embedding N | final validation loss (nats/char, 2 seeds) |
|---|---|---|
| XS | 3,312 | 2.930, 2.971 |
| S | 12,768 | 2.781, 2.687 |
| M | 56,640 | 2.552, 2.558 |
| L | 100,096 | 2.501, 2.511 |
| XL | 335,712 | 2.458, 2.450 |
| XXL | 793,344 | 2.407, 2.409 |

fitted to the mean losses of the five smallest models, the offset fit predicts **2.411** for XXL, which got **2.408** (error +0.003); the no-offset fit predicts 2.326 (error −0.082). the local exponents between neighbours are 0.056, 0.045, 0.034, 0.017, 0.022: the curve bends, and a straight line on log-log axes overshoots the improvement. the offset fit is not as reliable as its +0.003 suggests: fitted to each seed's five runs separately it predicts 2.390 (seed 0) and 2.435 (seed 1), a spread of 0.045 nats, ten times its error on the averaged curve; the no-offset fit is stable across seeds (2.329, 2.322) and stably wrong.

**why.** at a fixed number of steps every model is undertrained, and the larger ones more so (each step improves a large model more, but none of them has converged), so the loss of the largest models is limited by the step budget as well as by their size. that limit acts like a floor that rises relative to what the model could reach, and an offset term absorbs it; a pure power law can't bend and extrapolates the early, steeper slope. the offset's value here ($c$ about 2.3 nats) is not the entropy of Shakespeare (the capstone of chapter 18 reached 1.43 nats per character with longer training): it's the floor *of this recipe*. the per-seed spread is the second lesson: the offset is the least determined parameter of the fit, and two noisy small models (S differs by 0.09 between seeds) move it a lot.

## experiment 2: one compute budget, five ways to spend it

**what people predict.** having heard of Chinchilla, most people predict a U-shaped profile with its bottom near 20 tokens per parameter, which here would be about 23,000 parameters, between S and M.

**what happens.** budget $C = 6 \times 100{,}096 \times 100 \times 1{,}024 = 6.15 \times 10^{10}$ FLOPs (model L for 100 steps):

| model | N | steps | tokens per parameter | validation loss (2 seeds) |
|---|---|---|---|---|
| XS | 3,312 | 3,022 | 934 | 2.201, 2.206 |
| S | 12,768 | 784 | 63 | 2.308, 2.291 |
| M | 56,640 | 177 | 3.2 | 2.524, 2.530 |
| L | 100,096 | 100 | 1.0 | 2.560, 2.591 |
| XL | 335,712 | 30 | 0.1 | 2.882, 2.856 |

there's no U: the smallest model wins clearly, with 934 tokens per parameter, and every step up in size is worse (the seed spread is at most 0.03). `compute_optimal` would have chosen 22,638 parameters trained on 453,000 tokens, which sits between S and M, both well behind XS. the bottom of this profile is at or below 3,000 parameters; the lab doesn't find it.

**why.** two effects, which the lab can't separate. first, the Chinchilla ratio isn't universal: it was measured on subword tokens (several characters, and much more information, each) with models from 70M to 16B parameters, and a character carries far less to learn from, so a tiny model can keep improving over many more tokens per parameter. second, the recipe favours small models at this budget: a fixed learning rate and 10% warmup are fine for 3,000 steps and poor for 30 (XL spends three of its thirty steps warming up and never gets far from its initial loss), so the large models are handicapped beyond the compute trade-off. Hoffmann et al. tuned the schedule per run and fitted the profiles' minima at many budgets; finding the bottom here needs smaller models (or a larger budget), and per-size learning rates (topic 06's μP is designed to make one learning rate work at every size).

## experiment 3: chapter 18's capstone, revisited

**what people predict.** that a 30M-parameter model trained like chapter 18's L would be clearly better, by roughly the same step as from M to L or more.

**what happens.** through the three points (S 2.567, M 2.190, L 2.062 bits per character), the offset fit is exact (three parameters, three numbers) and gives $b = 0.61$, $c = 1.97$ bits; it predicts **2.008** bits/char at 30M parameters, 0.05 below L. the no-offset fit gives $b = 0.073$ and predicts **1.837**, 0.23 below L. the local exponents are 0.097 and 0.043. with each model's compute ($D = 16.4$M tokens):

| model | N | compute $6ND$ | `compute_optimal` N | `compute_optimal` D / D used |
|---|---|---|---|---|
| S | 354,240 | $3.5 \times 10^{13}$ | 538,696 | 0.7× |
| M | 1,816,896 | $1.8 \times 10^{14}$ | 1,220,000 | 1.5× |
| L | 7,460,160 | $7.3 \times 10^{14}$ | 2,472,117 | 3.0× |

**why.** both fits pass through (or near) the data and disagree by 0.17 bits at 30M: three points can't distinguish a power law that bends towards a floor from one that keeps going, and the offset fit's huge exponent and its floor just under the last point are artifacts of forcing three bending points through a three-parameter curve (lesson, on paper 4). Chinchilla's rule says S was trained on slightly more data than optimal for its compute and L on a third of what it should have had: L was too big for its 2,000 steps. that is the same handicap as in experiment 1, and it explains part of the bend: at a fixed step budget the larger models are further from their own limits. the experiment that separates the fits is one more point: a ~15M-parameter model trained with the same recipe (or, better, all sizes trained to compute-optimal lengths). the rule's 20 tokens per parameter is uncalibrated for characters (experiment 2), so read the last column as a direction, not a prescription.
