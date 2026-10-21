# 28 · checkpoint: interpretability, robustness & alignment

pass this to finish the chapter, or to skip it if you already know the material. no notes, no code from the chapter, no internet. a pen, paper and a calculator are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. what does integrated gradients' completeness axiom promise, and why does the plain gradient fail the sensitivity axiom?

<details><summary>answer</summary>

the attributions of one input add up to $f(x) - f(x')$, the change in output between the baseline and the input. the gradient is local: a feature whose effect has saturated (a ReLU that's off, a sigmoid at 1) has zero gradient at $x$ even though moving it from the baseline changed the output, so it gets no attribution.

</details>

2. what is a control task in probing, and what does low selectivity tell you?

<details><summary>answer</summary>

a task that gives every type (word, digit class) a fixed random label, so it can only be solved by memorising types. selectivity = task accuracy − control accuracy. low selectivity means the probe does about as well on meaningless labels, so its accuracy reflects what the layer lets it memorise (type identity), not the property.

</details>

3. why do some saliency methods pass "looking like the object" while failing the model-randomisation test?

<details><summary>answer</summary>

methods that multiply the gradient by the input (gradient × input, IG with a black baseline) inherit the input's support and brightness, so their maps look like the object whatever the weights are. a map that survives replacing the weights by random ones can't be explaining what the network learned.

</details>

4. in activation patching, why patch the *clean* activation into the *corrupted* run, and what does a full restoration at (layer $\ell$, position $t$) show?

<details><summary>answer</summary>

it asks whether that activation is sufficient to carry the information that distinguishes the inputs (a denoising experiment). full restoration means everything the metric depends on about the clean/corrupt difference is present in the residual stream at that layer and position.

</details>

5. how does an induction head find the token to copy, and why does it need a head in an earlier layer?

<details><summary>answer</summary>

at position $t$ holding token $u$, its query matches keys that say "my previous token was $u$", which sit one position after every earlier occurrence of $u$; it copies the token found there. that "previous token" information is written into the stream by a previous-token head in an earlier layer (K-composition); one layer of attention can't match a key's predecessor.

</details>

6. why can a ReLU model store more features than dimensions when features are sparse, but not when they're dense?

<details><summary>answer</summary>

superposed features interfere only when they're active together, with probability $(1-S)^2$, and negative interference is removed by the ReLU. with sparse features the expected cost of sharing a direction is small compared with the benefit of representing one more feature; with dense features every input pays the interference.

</details>

7. why are a sparse autoencoder's decoder columns kept at unit norm?

<details><summary>answer</summary>

otherwise the L1 penalty can be dodged: shrink a feature's activation by $c$ and grow its decoder column by $c$, and the reconstruction is unchanged while the penalty falls by $c$. normalising makes the penalty measure sparsity, not scale.

</details>

8. why is FGSM's step $\epsilon\,\mathrm{sign}(\nabla_x \ell)$, and why is PGD stronger?

<details><summary>answer</summary>

it maximises the first-order change $\nabla_x\ell^\top\delta$ over the $\ell_\infty$ ball, gaining $\epsilon\|\nabla\ell\|_1$. the network isn't linear over the whole ball; PGD takes small steps, re-linearises after each, and projects back into the ball, so it follows the actual loss surface to higher losses.

</details>

9. what does adversarial training cost, and what doesn't it buy?

<details><summary>answer</summary>

$n_\text{steps} + 1$ forward-backward passes per update and some clean accuracy (robust models give up weakly predictive, easily flipped features). it doesn't buy robustness beyond the training budget, or to perturbations outside the threat model (in the lab, lower contrast got worse).

</details>

10. why can temperature scaling never change accuracy, and what kind of miscalibration can't it fix?

<details><summary>answer</summary>

dividing all logits by the same $T > 0$ preserves their order, so the argmax doesn't change. a single global $T$ can't fix miscalibration that differs between classes or between easy and hard inputs.

</details>

11. state split conformal prediction's guarantee precisely. what does it not promise?

<details><summary>answer</summary>

if calibration and test points are exchangeable, the test point's true label is in the set with probability at least $1-\alpha$ (exactly $\lceil (n+1)(1-\alpha)\rceil/(n+1)$ for continuous scores), averaged over calibration sets and test points. it doesn't promise coverage for any particular calibration split, class or input, or under distribution shift.

</details>

12. why does DP-SGD clip each example's gradient and not the batch gradient?

<details><summary>answer</summary>

the Gaussian mechanism needs a bound on how much one example can change the released sum. clipping each example's gradient to norm $C$ bounds its contribution by $C$; clipping the mean bounds nothing about any individual (one example can dominate the mean before clipping).

</details>

13. a loss-threshold membership attack reaches AUROC 0.95 on one model and 0.54 on a DP model with $\epsilon = 1.4$. what's the DP bound on the attack's TPR at 1% FPR?

<details><summary>answer</summary>

$\text{TPR} \le e^{1.4} \cdot 0.01 + \delta \approx 0.04$. the bound holds for any attack; the loss threshold is far below it.

</details>

14. why can't a classifier satisfy demographic parity and equalized odds at once when base rates differ?

<details><summary>answer</summary>

equalized odds with an informative score approves more people from the group with more positives (equal TPR and FPR applied to different base rates give different approval rates); forcing equal approval rates then requires approving more negatives in one group or rejecting more positives in the other, which makes the error rates unequal. even a perfect classifier doesn't escape (it has equal error rates, TPR 1 and FPR 0, but approves each group at its base rate); only a classifier whose decisions are independent of $y$ (TPR = FPR in every group, at the same rate) satisfies both. (the "perfect classifier" exception belongs to a different theorem: calibration within groups plus balanced error rates, Kleinberg et al., 2017.)

</details>

15. why is potential-based shaping immune to the loop that games the corridor bonus?

<details><summary>answer</summary>

its rewards are differences $\gamma\Phi(s') - \Phi(s)$; around any loop they telescope to (nearly) zero, so every policy's value shifts by the same $-\Phi(s_0)$ and the optimal policy is unchanged. a respawning bonus pays for an event that a loop can repeat.

</details>

## 2. by hand

1. **integrated gradients and FGSM.** for $f(x) = x_1 x_2 + 2x_2$ (one logit) at $x = (1, 3)$ with a zero baseline: (a) compute IG, and check completeness; (b) a two-class model has logits $(0, f(x))$ and label 1: in which direction does FGSM move each coordinate?

<details><summary>solution</summary>

(a) $\nabla f = (x_2,\; x_1 + 2)$; along $\alpha x$: $(3\alpha,\; \alpha + 2)$; averages over $\alpha$: $(1.5, 2.5)$; times $x$: $\mathrm{IG} = (1.5, 7.5)$, sum 9 $= f(1, 3) - f(0) = 3 + 6$. (b) the loss is $-\log\sigma(f)$, whose gradient is $-(1-\sigma(f))\nabla f$, with sign $-\mathrm{sign}(3, 3) = (-1, -1)$: both coordinates decrease.

</details>

2. **conformal and temperature.** (a) 19 calibration scores $0.05, 0.10, \dots, 0.95$ and $\alpha = 0.1$: what is $\hat q$, and what's the exact expected coverage? (b) one example with logits $(2, 0)$ and label 0: write the NLL as a function of $\beta = 1/T$ and its derivative at $\beta = 1$.

<details><summary>solution</summary>

(a) rank $\lceil 20 \cdot 0.9\rceil = 18$, so $\hat q = 0.90$; coverage $18/20 = 0.90$. (b) $\ell(\beta) = \log(e^{2\beta} + 1) - 2\beta$; $\ell'(\beta) = 2\sigma(2\beta) - 2 = -2\sigma(-2\beta)$, at $\beta = 1$: $-2 \cdot 0.119 = -0.238$. negative: on this (correctly classified) example, sharper predictions (larger $\beta$, $T < 1$) lower the NLL.

</details>

3. **privacy accounting and best-of-n.** (a) one step of the Gaussian mechanism on the full data with $\sigma = 1$: RDP is $\alpha/2$; with $\delta = 10^{-5}$, find $\epsilon$ (search integer $\alpha$). (b) best-of-4 over three responses with reference probabilities $(0.5, 0.25, 0.25)$ and proxy scores $(0, 1, 2)$: compute the distribution.

<details><summary>solution</summary>

(a) $\epsilon(\alpha) = \alpha/2 + 11.51/(\alpha - 1)$: $\alpha = 5$: $2.5 + 2.88 = 5.38$; $\alpha = 6$: $3 + 2.30 = 5.30$; $\alpha = 7$: $3.5 + 1.92 = 5.42$. $\epsilon \approx 5.30$ at $\alpha = 6$. (b) top item: $1 - 0.75^4 = 0.684$; middle: $0.75^4 - 0.5^4 = 0.254$; bottom: $0.5^4 = 0.0625$.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild in a fresh copy of the topic notebooks:

- `run_with_cache`, `patch_activation`, `activation_patching` and `position_patching` (topic 02, stages 1–3), and
- `per_sample_gradients` and `dp_sgd_step` (topic 06, stages 1–2).

check them with the topics' own tests:

```bash
pytest 28-interpretability-robustness-and-alignment/02-mechanistic-interpretability/test_exercise.py -k "stage1 or stage2 or stage3"
pytest 28-interpretability-robustness-and-alignment/06-privacy-and-fairness/test_exercise.py -k "stage1 or stage2"
```

## 4. readiness

move on when you score at least 80% on parts 1 and 2 and the blind rebuild passes every check. the same checkpoint works as a placement test: if you pass it cold, skim the labs (each has a result worth seeing: IG maps of a randomly initialised LeNet that still look like the digits, induction heads that arrive in a sudden phase change, toy features that settle into antipodal pairs and pentagons, adversarial training that makes contrast changes worse, conformal sets that cover below 90% on half of all calibration splits while averaging exactly 90%, a membership attack at AUROC 0.95, and a corridor agent that circles a bonus forever) and go on to chapter 29.
