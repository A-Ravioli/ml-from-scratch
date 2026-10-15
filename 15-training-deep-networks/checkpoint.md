# 15 · training deep networks · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. a linear layer with fan_in $n$ receives inputs with $E[h^2] = 1$. what weight variance keeps the pre-activations' variance at 1, and what changes if a ReLU follows every layer?

<details><summary>answer</summary>

$\operatorname{Var}(z) = n\operatorname{Var}(W)E[h^2]$, so $\operatorname{Var}(W) = 1/n$ (LeCun). a ReLU halves the second moment of a symmetric input, so to keep it constant through ReLU layers you need $\operatorname{Var}(W) = 2/n$: kaiming (He) init. with $1/n$ (or xavier on square layers) the signal halves at every layer, $2^{-L}$ after $L$ layers.

</details>

2. kaiming init keeps the scale of a 50-layer ReLU network's activations constant. why can the network still be nearly useless at initialization?

<details><summary>answer</summary>

every ReLU layer makes the representations of different inputs more aligned (zeroing negative coordinates removes the sign differences), so after many layers all inputs map to nearly the same direction (cosine similarity ~0.99 in the topic 01 lab). the output then barely depends on the input. variance preservation is necessary, not sufficient; signal propagation also needs correlations preserved (residual connections, normalization, careful activations).

</details>

3. why does batch norm behave differently in training and evaluation, and what does it keep for evaluation?

<details><summary>answer</summary>

in training it standardizes each feature with the current batch's mean and (biased) variance; at evaluation there may be a single example, and predictions shouldn't depend on batchmates, so it uses running averages of the batch statistics collected during training (exponential moving averages with `momentum`, the variance unbiased). the running statistics are buffers, not trained parameters.

</details>

4. write the backward pass of $\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$ along an axis, given $g = \partial L/\partial\hat x$, and explain its two correction terms.

<details><summary>answer</summary>

$\partial L/\partial x = s(g - \bar g - \hat x\,\overline{g\hat x})$ with $s = (\sigma^2 + \epsilon)^{-1/2}$ and bars meaning means over the axis. $-\bar g$ removes the component along "all ones" (shifting the inputs doesn't change $\hat x$); $-\hat x\,\overline{g\hat x}$ removes the component along $\hat x$ (scaling the inputs doesn't change it).

</details>

5. why does layer norm, not batch norm, appear in transformers and language models?

<details><summary>answer</summary>

layer norm computes its statistics over the features of each example (each token), so it doesn't depend on the batch: it works with tiny batches, variable sequence lengths and one-token-at-a-time generation, and behaves identically in training and evaluation.

</details>

6. Adam with an L2 penalty versus AdamW: what's different, and which weights does the L2 penalty actually shrink under Adam?

<details><summary>answer</summary>

Adam divides the whole gradient, penalty included, by $\sqrt{\hat v}$; AdamW applies the decay $w \leftarrow w - \eta\lambda w$ outside that normalization. under Adam + L2, weights with large loss gradients are barely regularized ($\lambda w/\sqrt{\hat v}$ is tiny), while weights with no loss gradient are pushed towards zero at a fixed rate $\eta$ per step regardless of $\lambda$. AdamW shrinks every weight by the same fraction.

</details>

7. what does "inverted" mean in inverted dropout, and why is it the standard?

<details><summary>answer</summary>

the kept activations are divided by $1 - p$ during training, so the expected output equals the input and evaluation is simply the identity: nothing to rescale at test time, and the same model code serves both modes.

</details>

8. with label smoothing $s$ over $k$ classes, what is the most confident prediction the loss rewards, and what does that do to calibration?

<details><summary>answer</summary>

the loss is minimized when the predicted distribution equals the target, $1 - s + s/k$ on the true class (0.91 for $s = 0.1$, $k = 10$). the model stops pushing its logits apart, so it is far less overconfident on its errors, but it can become *under*confident overall (higher ECE when accuracy exceeds $1 - s + s/k$).

</details>

9. why warm up the learning rate, especially with Adam?

<details><summary>answer</summary>

early steps are the least reliable: Adam's first update moves every weight by about $\pm\eta$ regardless of its gradient's size (bias-corrected $\hat m/\sqrt{\hat v} = \operatorname{sign}(g)$), in a direction estimated from one minibatch, while the network's scale and the optimizer's statistics haven't settled. with a large peak learning rate this can push the network into a region it never recovers from. a linear ramp keeps those first steps small.

</details>

10. what does a learning-rate range test measure, and why not train at the minimum of its curve?

<details><summary>answer</summary>

it trains briefly with an exponentially increasing learning rate (one minibatch per step) and records the loss: flat while the rate is too small, falling, then exploding. the minimum is where steps are just short of instability, after only ~100 steps; training there for thousands of steps would be at the edge of divergence, so the usual rule backs off by about 10×.

</details>

11. where must gradient clipping go in the training step, and what does clipping by global norm preserve?

<details><summary>answer</summary>

after `loss.backward()` (so `.grad` holds this step's gradient) and before `optimizer.step()`. scaling every gradient by the same factor $c/\|g\|$ preserves the direction of the update and caps its norm at $\eta c$ for SGD.

</details>

12. a training run's logger reports dead ReLUs (100%) and an update ratio of exactly 0 for the hidden layers, with the loss flat at $\log 10$. name two possible causes.

<details><summary>answer</summary>

large negative hidden biases (or a bad initialization) that make every pre-activation negative from the start; or an instability early in training (a learning rate far too large, unnormalized inputs) whose first huge steps pushed the pre-activations negative. in both cases no gradient flows through the dead units, so nothing changes, and the output layer can only learn the class frequencies.

</details>

13. which bugs can a gradient/weight/activation logger not see, and what check catches them?

<details><summary>answer</summary>

bugs in the data or the evaluation: labels misaligned with inputs, different preprocessing for training and validation, leakage, an accuracy computed on the wrong split. optimization looks perfectly healthy (the network memorizes whatever it's given). comparing training and validation loss (and looking at a few examples with their labels) catches them.

</details>

14. why does random search beat grid search, and why sample the learning rate log-uniformly?

<details><summary>answer</summary>

usually only a few hyperparameters matter; a $k \times k$ grid tests only $k$ distinct values of each, while $k^2$ random trials test $k^2$. the learning rate acts multiplicatively, so equal ratios matter equally; uniform sampling would put most trials within a factor of 10 of the maximum.

</details>

15. you train the same configuration with 10 seeds and the validation accuracy ranges over 2 percentage points. a colleague reports that a new trick improves accuracy by 0.3 points, from one run each. what do you ask for?

<details><summary>answer</summary>

several seeds of both (ideally paired, the same seeds for both), the mean difference with a standard error or confidence interval, and the evaluation set's size (the binomial standard error of an accuracy on a few thousand images is itself several tenths of a point). with a seed spread of 2 points, one run each can't distinguish 0.3 points from zero.

</details>

## 2. by hand (no computer)

1. **variance through depth.** a 10-layer ReLU network of width 512 (zero biases) receives inputs with $E[x^2] = 1$. compute $E[h_{10}^2]$ (after the 10th ReLU) for (a) kaiming init, (b) xavier init, (c) PyTorch's default $U(-1/\sqrt n, 1/\sqrt n)$ without biases.

<details><summary>solution</summary>

$E[h_{l}^2] = n\operatorname{Var}(W)\cdot\frac12E[h_{l-1}^2]$ (the ReLU halves the second moment of the symmetric pre-activation). (a) $\operatorname{Var}(W) = 2/n$: factor 1, so $E[h_{10}^2] = 1$. (b) $\operatorname{Var}(W) = 2/(n + n) = 1/n$: factor $\frac12$, so $2^{-10} \approx 10^{-3}$. (c) $\operatorname{Var}(W) = 1/(3n)$: factor $\frac16$, so $6^{-10} \approx 1.7 \times 10^{-8}$. (the first layer has fan_in equal to the input dimension, not 512, but with the same scheme the factor is the same.)

</details>

2. **batch norm by hand.** one feature, a batch of four: $x = (1, 2, 3, 6)$, $\epsilon = 0$, $\gamma = 2$, $\beta = 1$. (a) compute the outputs. (b) with $g = \partial L/\partial y = (1, 0, 0, 0)$, compute $\partial L/\partial x$, $\partial L/\partial\gamma$ and $\partial L/\partial\beta$. (c) the running mean and variance after this batch, from 0 and 1, with momentum 0.1.

<details><summary>solution</summary>

(a) $\mu = 3$, $\sigma^2 = (4 + 1 + 0 + 9)/4 = 3.5$, $s = 1/\sqrt{3.5} = 0.53452$. $\hat x = (-2, -1, 0, 3)s = (-1.06904, -0.53452, 0, 1.60357)$; $y = 2\hat x + 1 = (-1.13809, -0.06904, 1, 4.20713)$.
(b) $\partial L/\partial\beta = \sum g = 1$; $\partial L/\partial\gamma = \sum g\hat x = -1.06904$. the gradient into $\hat x$ is $\gamma g = (2, 0, 0, 0)$, with mean $0.5$ and $\overline{(\gamma g)\hat x} = 2 \cdot (-1.06904)/4 = -0.53452$. so $\partial L/\partial x = s\big((2, 0, 0, 0) - 0.5 + 0.53452\,\hat x\big) = s\,(1.5 - 0.57143,\ -0.5 - 0.28571,\ -0.5,\ -0.5 + 0.85714) = s\,(0.92857, -0.78571, -0.5, 0.35714) = (0.49634, -0.41998, -0.26726, 0.19090)$. check: the entries sum to 0, and $\sum x_i\,\partial L/\partial x_i = 0.49634 - 0.83996 - 0.80178 + 1.14540 = 0$.
(c) running mean $0.9 \cdot 0 + 0.1 \cdot 3 = 0.3$; the unbiased variance is $14/3 = 4.6667$, so running variance $0.9 + 0.46667 = 1.36667$.

</details>

3. **hyperband's arithmetic.** for `max_budget=27`, `eta=3`: list the brackets (configurations and rungs), the total number of evaluations and the total budget. how many full-budget random-search trials would the same budget buy?

<details><summary>solution</summary>

$s_\text{max} = 3$. bracket $s$ starts $n = \lceil 4 \cdot 3^s/(s + 1)\rceil$ configurations at budget $27/3^s$. $s = 3$: 27 configs, rungs 27@1, 9@3, 3@9, 1@27 (cost 108). $s = 2$: $\lceil 36/3\rceil = 12$ configs, rungs 12@3, 4@9, 1@27 (cost $36 + 36 + 27 = 99$). $s = 1$: $\lceil 12/2\rceil = 6$ configs, rungs 6@9, 2@27 (cost $54 + 54 = 108$). $s = 0$: 4 configs at 27 (cost 108). total: $40 + 17 + 8 + 4 = 69$ evaluations, $108 + 99 + 108 + 108 = 423$ budget units, $49$ distinct configurations; the same budget buys $\lfloor 423/27\rfloor = 15$ full random-search trials.

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **normalization** (topic 02, stages 1–3): `standardize`, `standardize_backward`, `BatchNorm1d` (training and evaluation, running statistics, one `Tensor.from_op` node) and `LayerNorm`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=15-training-deep-networks/02-normalization
mkdir -p /tmp/rebuild-15
git show HEAD:$T/exercise.py > /tmp/rebuild-15/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-15/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-15/solutions/` and run `python scripts/stubify.py /tmp/rebuild-15`.) open `/tmp/rebuild-15/exercise.py` with `marimo edit`, implement stages 1–3 from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-15/test_exercise.py -k "stage1 or stage2 or stage3"
```

you pass if every check is green within the hour.

**harder variant** (for testing out of the chapter with confidence): within 100 minutes in total, also rebuild topic 04's schedulers and range test (stages 2 and 4) and topic 06's successive halving and hyperband (stages 2 and 3), each in its own scratch directory the same way:

```bash
pytest /tmp/rebuild-15-opt/test_exercise.py -k "stage2 or stage4"
pytest /tmp/rebuild-15-hp/test_exercise.py -k "stage2 or stage3"
```

(run them from the repo root: the notebooks import earlier `mlfs` modules, which the tests resolve to the reference solutions, so only the code you rebuilt is being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each part of a problem separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the idea, and the topic lessons are organized by those ideas.
