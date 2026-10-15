# 29 · electives · checkpoint

use this to check that you've finished the chapter, or to test out of it. it tests the skills every elective exercises, whichever blocks you took: choosing a baseline, designing a fair comparison, reading the evidence honestly, and finding your way into a new subfield. work without notes, then compare with the answers. section 3 is checked by the capstone's own tests.

## 1. concept check

### cross-cutting (answer all)

1. what makes a baseline "classical" and worth beating? name the baseline you'd use for two of: a causal effect estimate, a continual-learning method, a forecaster, a recommender, a quantum kernel.

<details><summary>answer</summary>

the strongest simple method a skeptic would try first, tuned with the same care as yours. beating a weak or untuned baseline says nothing. examples: the naive regression *and* "regress on everything" for an effect estimate (one is confounded, the other conditions on mediators and colliders, and you want to show you beat both); fine-tuning plus a small replay buffer for continual learning (fine-tuning alone is a floor, not a competitor); the seasonal-naive forecast (the same value one season ago) for forecasting, which is notoriously hard to beat; popularity and ALS matrix factorization for recommenders; an RBF-kernel SVM with a tuned bandwidth for a quantum kernel. (capstone; ch 07)

</details>

2. method and baseline were each run on 5 seeds. why pair the runs by seed, and how can you tell from the data whether pairing helped?

<details><summary>answer</summary>

a seed fixes the initialisation, data order and sampled data, which affect both methods in the same way. the paired difference $d_i$ cancels that shared part, so its variance is $\mathrm{Var}(m) + \mathrm{Var}(b) - 2\,\mathrm{Cov}(m, b)$, smaller than the unpaired $\mathrm{Var}(m) + \mathrm{Var}(b)$ whenever the two methods' scores are positively correlated across seeds. check the correlation of the per-seed scores, or compare $s_d$ with $\sqrt{s_m^2 + s_b^2}$. pairing only works if the pairs line up: pair by seed key, never by list position. (capstone, stage 1)

</details>

3. your method beats the baseline on all 5 paired seeds by a wide margin. the exact two-sided sign-flip test gives p = 0.0625. is the method not significantly better?

<details><summary>answer</summary>

with 5 seeds there are only $2^5 = 32$ sign patterns, and the observed pattern and its mirror image are always the two most extreme, so the smallest possible two-sided p is $2/32 = 0.0625$. no effect, however large, can reach 0.05 with five seeds. that's a property of the test at this $n$, not evidence against the method. report the interval for the mean difference (here it will exclude 0 by a wide margin) and say that the p-value is at its floor; or run more seeds (with 6 the floor is 0.031). (capstone, stage 3)

</details>

4. why use a $t$ quantile rather than 1.96 for an interval over 5 seeds, and why is the percentile bootstrap a poor choice there?

<details><summary>answer</summary>

with 5 values, $s_d$ is itself very noisy and often too small; the $t$ distribution with 4 degrees of freedom accounts for that (quantile 2.776 instead of 1.96). using 1.96 gives about 88% coverage instead of 95%. the percentile bootstrap resamples the 5 observed values, which understate the spread of the distribution they came from (the plug-in variance divides by $n$, and resamples often repeat values), so its intervals are too short: about 83% coverage in the capstone's demo. at a few dozen seeds the two agree. (capstone, stage 2; ch 05 · topic 03)

</details>

5. the capstone's verdict reads the interval, not the p-value. what's the difference between "inconclusive" and "no effect"?

<details><summary>answer</summary>

"inconclusive" means the interval contains 0 *and* effects that would matter: the experiment can't tell. "no effect" (more precisely, "no effect larger than δ") needs an interval that sits entirely inside $(-\delta, \delta)$ for a δ you'd consider negligible. a wide interval around 0 is the first case, and reporting it as the second is one of the commonest errors in ML papers. say what effect size your experiment could have detected. (capstone, stage 6)

</details>

6. a method stores 15,680 float32 values; a baseline stores 200 uint8 images of 784 pixels. which uses less memory?

<details><summary>answer</summary>

it depends on the unit. in stored numbers: 15,680 against 156,800, the method is 10× smaller. in bytes: 62,720 against 156,800, the method is 2.5× smaller. if the budget is 39,200 numbers, the images don't fit; if it's 156,800 bytes (the same 39,200 float32s), they fit exactly, and in the continual-learning project they then beat the method. state the unit, justify it (what will the deployed system actually store?), and say what the budget leaves out: replay doubles the compute per step, and a generator has to be trained. (capstone, stage 5; ch 29 · topic 14)

</details>

7. you tried 40 hyperparameter configurations of your method and ran the baseline at its defaults. what's wrong, and what are two fixes?

<details><summary>answer</summary>

the comparison mixes "better method" with "more tuning", and selecting the best of 40 configurations on the same seeds you report also overfits the evaluation (the winner's score is biased upwards). fixes: give the baseline the same tuning budget (same number of configurations, same search); select hyperparameters on separate validation seeds or a validation split and report fresh seeds; or report performance as a function of the tuning budget. (ch 07)

</details>

8. what does "the simulator favours the method" mean? give an example from a block you took, and a way to test it.

<details><summary>answer</summary>

the data-generating world was built so that the method's assumptions hold exactly, or the baseline's fail, in ways real data might not. examples: topic 04's causal world is linear-Gaussian and faithful by construction (with the covariates' real dependence kept, PC orients a false v-structure in half the draws and IDA drops from 0.96 to 0.82); split MNIST's classes are blobs around their means, which is why a per-class Gaussian generator works there and wouldn't on CIFAR-10; a long-range copy task is made for models with long memory. test it by breaking the assumption on purpose: real covariate dependence, a harder dataset, a task built for the baseline, and see whether the verdict survives. (capstone, write-up prompt 4)

</details>

9. why use semi-synthetic data (real inputs, simulated mechanism) for a project, and what can't it tell you?

<details><summary>answer</summary>

it gives you ground truth (the true graph, the true effect, the true forecast distribution), which real observational data never does, while keeping realistic input distributions (skew, dependence, outliers) that a fully synthetic world would idealize away. it can't tell you whether the simulated mechanism resembles the real one: if the true mechanism is nonlinear, confounded by something unrecorded, or non-stationary, success on the semi-synthetic version doesn't transfer. treat it as a necessary test, not a sufficient one. (ch 29 · topic 04)

</details>

10. a portfolio makes 6 claims, each at α = 0.05. what's the chance of at least one false claim if none of the methods actually helps (independent tests)? what do Bonferroni, Holm and Benjamini–Hochberg control?

<details><summary>answer</summary>

$1 - 0.95^6 \approx 0.26$. Bonferroni and Holm control the family-wise error rate (the chance of *any* false rejection) at α; Holm is uniformly more powerful, rejecting everything Bonferroni does and sometimes more. Benjamini–Hochberg controls the false discovery rate (the expected *fraction* of rejections that are false), which is weaker and suits screening many hypotheses; for a handful of headline claims, FWER control (Holm) is the usual choice. (capstone, stage 4; ch 05 · topic 04)

</details>

11. you have 1,000 seeds and the method is better by 0.002 accuracy with p < 0.001. what should the report say?

<details><summary>answer</summary>

that the difference is real but tiny: with enough seeds any nonzero difference becomes significant. report the effect size with its interval (e.g. +0.002, [+0.001, +0.003]) and judge it against what matters: the seed-to-seed spread, the cost of the method, differences that would change a decision. significance answers "is it zero?", not "does it matter?". (ch 05; ch 07)

</details>

12. what should a "seed" control in a fair comparison, and why might "same seed" not give two methods the same data order?

<details><summary>answer</summary>

everything random that isn't the method: initialisation, data order, minibatch sampling, data splits or simulated draws. but if both methods share one generator and consume random numbers differently (one draws dropout masks or replay samples, the other doesn't), the same seed yields different data orders after the first divergence. use separate generators for data and for the method's own randomness (or derive them from the seed independently) so that "seed 3" means the same data for both. (ch 00 · topic 05; ch 29 · topic 12)

</details>

13. you open a paper in a subfield you've never studied. what do you read first, and what are the three things you check before believing its headline table?

<details><summary>answer</summary>

read the problem setup and evaluation protocol before the method: what exactly is measured, under what scenario (continual learning's task- vs class-incremental results don't transfer; a causal paper's identification assumptions; a recommender's offline metric and negative sampling). then check: (a) the baselines (are they the strong classical ones, equally tuned?), (b) the budget (compute, memory, data, tuning) and whether it's matched, and (c) the uncertainty (number of seeds, intervals, whether differences exceed the seed noise). then read the method, and look for the setting where it should fail. (ch 29 · every block)

</details>

14. your method lost to the baseline in one elective. is that a failed project?

<details><summary>answer</summary>

no, if the comparison was fair. a well-run negative result tells the reader where the method's assumptions break or what the baseline already captures, and it's worth more than a weakly supported positive one. report it with the same care (interval, seeds, budget), explain it with the block's ideas, and say what would have to change for the method to win. the curriculum asks "when (if ever) do quantum kernels beat classical ones?" precisely because "never, here, for this reason" is a legitimate answer. (capstone rubric)

</details>

### per block (answer the ones for the blocks you took)

15. **causal inference.** in topic 04's project, "regress $Y$ on $T$ and every other variable" estimates 0.41 when the truth is 0.96. name the two mistakes it makes.

<details><summary>answer</summary>

it conditions on the mediator $M$, which blocks the indirect path $T \to M \to Y$ and leaves only the direct effect (0.6 instead of 0.96); and it conditions on $C$, a common effect of $T$ and $Y$ (a collider), which induces a spurious negative association between them and pulls the estimate down further. "control for everything" is not an adjustment strategy: the adjustment set must block backdoor paths without blocking causal paths or opening collider paths. (topics 01, 04)

</details>

16. **causal inference.** from observational data alone, what can the PC algorithm recover, and why can't it tell a chain $A \to B \to C$ from a fork $A \leftarrow B \to C$?

<details><summary>answer</summary>

the Markov equivalence class, represented by a CPDAG: the skeleton plus the directions forced by v-structures and Meek's rules. a chain, a reversed chain and a fork all imply the same single independence $A \perp C \mid B$, so no conditional-independence test can separate them; a collider implies the opposite pattern and is fully oriented. telling them apart needs interventions, time order or extra assumptions (e.g. non-Gaussian noise, as in LiNGAM). (topic 04)

</details>

17. **time series & state-space models.** why must a forecaster be evaluated by rolling-origin backtesting rather than random K-fold cross-validation?

<details><summary>answer</summary>

random folds put future observations in the training set for past test points, so the model can use information that wouldn't exist at forecast time (and autocorrelation makes neighbouring points nearly duplicates), which inflates accuracy. rolling-origin evaluation trains on everything up to time $t$, forecasts $t + 1, \dots, t + h$, then moves the origin forward, exactly as the forecaster would be used. (topic 05)

</details>

18. **time series & state-space models.** a Kalman filter's measurement-noise variance is set 100× too small. what do its estimates look like?

<details><summary>answer</summary>

the filter trusts each measurement far too much: the Kalman gain is near its maximum, so the state estimate follows the noisy measurements closely and jitters, and the reported uncertainty is far smaller than the actual error (overconfident). set too large, the opposite: the filter ignores measurements, lags behind real changes and is underconfident. (topic 06)

</details>

19. **meta-learning.** what term does first-order MAML drop from the meta-gradient, and why does it often work anyway?

<details><summary>answer</summary>

the meta-gradient differentiates through the inner update $\theta' = \theta - \alpha\nabla L_{\text{train}}(\theta)$, which brings in the Jacobian $I - \alpha\nabla^2 L_{\text{train}}(\theta)$ (a Hessian term). first-order MAML replaces it with the identity, using $\nabla L_{\text{test}}(\theta')$ directly. with a small inner step size and few inner steps the Hessian term is a small correction, and the first-order direction still moves $\theta$ towards parameters from which one step adapts well. (topic 09)

</details>

20. **meta-learning.** why should training episodes match the test-time setting (N-way, K-shot)?

<details><summary>answer</summary>

meta-learning learns an adaptation procedure for a distribution of tasks; the episodes *are* that distribution. a model trained on 5-way 5-shot episodes learns embeddings and decision rules tuned to that difficulty and support-set size; tested on 20-way 1-shot it faces a different task distribution, and accuracy typically drops (prototypes from one example are noisier, more classes crowd the space). matching, or deliberately training a bit harder (more ways), is part of the method. (topic 10)

</details>

21. **continual learning.** why does EWC barely help class-incremental split MNIST while a buffer of 200 images lifts accuracy from 0.20 to 0.81?

<details><summary>answer</summary>

in class-incremental learning, every gradient step on a new task pushes the old classes' logits down (no old labels appear in the batch), so the output layer forgets old classes even if the hidden weights barely move. a penalty on weight changes can't push those logits back up. replay puts old labels back into every batch, which restores the balance at the output layer; even a few examples per class do most of the work. (topics 13, 14)

</details>

22. **continual learning.** a method reports zero backward transfer. why isn't that enough to call it good?

<details><summary>answer</summary>

zero forgetting is achievable by never learning anything new (e.g. a learning rate of 0 after task 1). report average accuracy and the learning accuracy (the diagonal of $R$) alongside BWT or forgetting, so stability and plasticity are both visible. (topic 12)

</details>

23. **optimal transport.** two distributions with disjoint supports are moved closer together. why does the Wasserstein distance give a useful training signal when KL and Jensen–Shannon don't?

<details><summary>answer</summary>

with disjoint supports KL is infinite and JS is constant ($\log 2$) whatever the gap, so their gradients carry no information about how far apart the distributions are. the Wasserstein distance is the cost of moving mass, which shrinks smoothly as the supports approach (for two point masses it's simply their distance), so its gradient points towards the other distribution. (topic 15)

</details>

24. **optimal transport.** what happens to Sinkhorn's coupling and running time as the entropic regularization ε shrinks, and why do you need the log-domain version?

<details><summary>answer</summary>

as ε → 0 the coupling approaches the exact (sparse) OT plan and the cost approaches the Wasserstein cost, but convergence slows (more iterations) and the kernel $e^{-C/\varepsilon}$ underflows to zero for moderate costs, giving divisions by zero. the log-domain version works with potentials and log-sum-exp, which stays finite. large ε blurs the coupling towards the independent product of the marginals and converges fast. (topic 16)

</details>

25. **information geometry.** why is the natural gradient invariant to how you parameterize the model, while the ordinary gradient isn't?

<details><summary>answer</summary>

under a reparameterization $\theta = g(\phi)$ with Jacobian $J$, the gradient transforms as $J^\top\nabla_\theta L$ and the Fisher as $J^\top F J$, so the natural-gradient step $F^{-1}\nabla L$ in $\phi$ maps (to first order) to the same step in $\theta$, i.e. the same change of the distribution. it measures step size by KL divergence between distributions, not by Euclidean distance between parameter vectors. the ordinary gradient depends on the coordinates. (topics 18, 19)

</details>

26. **information geometry.** what does K-FAC approximate, and what makes it cheap?

<details><summary>answer</summary>

it approximates the Fisher as block-diagonal over layers, and each layer's block as a Kronecker product $A \otimes G$ of the second moment of the layer's inputs (activations) and of the gradients with respect to its outputs, which assumes the two are independent. a Kronecker product inverts factor by factor, $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$, so a layer with $n_{\text{in}} \times n_{\text{out}}$ weights needs two small inverses instead of one of size $n_{\text{in}}n_{\text{out}}$. (topic 20)

</details>

27. **recommender systems.** why can a model with lower rating RMSE give worse top-10 recommendations?

<details><summary>answer</summary>

RMSE weighs every observed rating equally, mostly mid-range ratings of items users already chose, while a top-10 list depends only on getting the order right at the top across all items, including the many the user never rated. a model can predict observed ratings well and still rank unwatched items poorly (e.g. by shrinking everything to item means). evaluate with ranking metrics such as recall@k or NDCG on held-out interactions. (topics 21, 22)

</details>

28. **recommender systems.** in implicit feedback, why isn't a missing interaction a negative label, and how does BPR handle it?

<details><summary>answer</summary>

a user not clicking an item may mean they dislike it or simply never saw it, so missing entries are a mix of negatives and unobserved positives. BPR doesn't label them: it trains on pairs, asking only that an item the user interacted with is ranked above a sampled item they didn't, which matches the ranking goal and tolerates some sampled "negatives" being positives. (topic 22)

</details>

29. **audio & speech.** what does lengthening the STFT window buy and cost?

<details><summary>answer</summary>

finer frequency resolution (bins $f_s/N$ apart) at the cost of coarser time resolution: a long window averages over everything that happens inside it, smearing onsets and fast changes. you can't have both; this is the time–frequency uncertainty tradeoff. speech typically uses 20–30 ms windows. (topic 24)

</details>

30. **audio & speech.** why does CTC need a blank token?

<details><summary>answer</summary>

CTC collapses repeated labels and then removes blanks to map a frame-level path to an output string. without a blank, "hello" would be impossible (the two l's would collapse into one), and every frame would have to emit a character even during silence or transitions. the blank separates genuine repeats ("l, blank, l") and lets frames emit nothing. (topic 25)

</details>

31. **federated learning.** why do many local steps on non-IID clients hurt FedAvg?

<details><summary>answer</summary>

each client's local optimum differs from the global one when its data distribution differs. with many local steps every client drifts towards its own optimum, and averaging the drifted models doesn't give the global optimum (client drift); the error grows with the number of local steps and the heterogeneity. FedProx adds a proximal term that keeps local models near the global one; fewer local steps trade drift against communication. (topics 27, 28)

</details>

32. **federated learning.** secure aggregation hides each client's update from the server. why is differential privacy still needed?

<details><summary>answer</summary>

secure aggregation protects individual updates in transit and from the server, but the aggregate itself (and the final model) can still leak information about individual clients' data, e.g. through membership inference or when few clients participate. differential privacy (clipping each update and adding calibrated noise) bounds what the output reveals about any one client. they protect against different threats and are used together. (topic 29)

</details>

33. **quantum ML.** what's a barren plateau, and why does it make large variational circuits hard to train?

<details><summary>answer</summary>

for sufficiently random, deep parameterized circuits, the variance of the cost gradient decays exponentially with the number of qubits, so almost everywhere the landscape is flat to within shot noise. estimating a gradient then needs exponentially many measurements, and training stalls. shallow, structured circuits and local cost functions mitigate it. (topic 31)

</details>

34. **quantum ML.** a paper reports a quantum kernel beating a classical SVM. what would you check?

<details><summary>answer</summary>

that the classical baseline was tuned (an RBF kernel's bandwidth and $C$ searched as carefully as the quantum feature map's settings), that the dataset wasn't constructed to suit the quantum feature map, that the kernel values were estimated with realistic shot noise (or say they were exact simulations), that the comparison holds at the dataset sizes and dimensions reported with intervals over splits, and whether the quantum kernel can be simulated classically at that size (in which case it's a classical kernel with an unusual feature map). (topic 32)

</details>

## 2. by hand

1. **a small paired comparison.** four paired seeds give differences $d = (0.04, 0.06, -0.01, 0.05)$. compute (a) the exact two-sided sign-flip p-value, (b) the smallest p four seeds allow, (c) the 95% $t$ interval for the mean difference ($t_{0.975, 3} = 3.182$), and (d) the verdict.

<details><summary>solution</summary>

(a) the observed sum is 0.14; the magnitudes sum to 0.16. sums with $|S| \ge 0.14$: all signs positive (0.16), only the 0.01 flipped (0.14, the observed pattern), and their two mirror images: 4 of 16 patterns, so $p = 0.25$. (b) $2/2^4 = 0.125$. (c) $\bar d = 0.035$; deviations $(0.005, 0.025, -0.045, 0.015)$, squared sum $0.0029$, $s_d = \sqrt{0.0029/3} = 0.0311$, standard error $0.0155$, half-width $3.182 \times 0.0155 = 0.0495$: $[-0.014, 0.084]$. (d) inconclusive: the interval contains 0, and also effects as large as 0.08. more seeds needed.

</details>

2. **Holm by hand.** three electives give primary p-values $(0.012, 0.030, 0.004)$. which claims survive Bonferroni at 0.05, and which survive Holm? give Holm's adjusted p-values.

<details><summary>solution</summary>

Bonferroni compares each with $0.05/3 = 0.0167$: 0.012 and 0.004 survive, 0.030 doesn't. Holm sorts them: $0.004 \le 0.05/3$ ✓, $0.012 \le 0.05/2 = 0.025$ ✓, $0.030 \le 0.05/1$ ✓, so all three survive. adjusted: $3 \times 0.004 = 0.012$, $\max(0.012, 2 \times 0.012) = 0.024$, $\max(0.024, 1 \times 0.030) = 0.030$; in input order $(0.024, 0.030, 0.012)$.

</details>

3. **budgets in two units.** a memory budget is 100,000 bytes. (a) how many CIFAR-10 images ($3 \times 32 \times 32$) fit as uint8? as float32? (b) does a per-class diagonal Gaussian (a mean and a variance per pixel for 10 classes) fit in float16? (c) the same budget restated as 25,000 "floats" (stored values): how many images fit now, and does the Gaussian?

<details><summary>solution</summary>

(a) an image is 3,072 values: uint8, 3,072 bytes, so 32 images (98,304 bytes); float32, 12,288 bytes, so 8 images. (b) $10 \times 2 \times 3{,}072 = 61{,}440$ values × 2 bytes = 122,880 bytes: no (the means alone, 61,440 bytes, would). (c) 8 images (24,576 values), whatever their dtype; the Gaussian's 61,440 values don't fit. counting values penalizes compact dtypes; counting bytes rewards them. say which your budget uses.

</details>

## 3. blind rebuild (60 minutes)

with no notes, rebuild the capstone toolkit's statistical core: `paired_differences`, `paired_ci` (the $t$ path; call your `mlfs.stats.bootstrap_ci` for the other), `sign_flip_test` with its exact enumeration and `min_p_value`, and `holm`, in a fresh copy of `capstone/capstone.py`. check them with

```bash
python -m pytest 29-electives/capstone/test_capstone.py -k "stage1 or stage2 or stage3 or stage4"
```

then pick the core algorithm of one block you took (for example the PC algorithm, EWC's penalty, the Kalman filter, Sinkhorn, BPR, the CTC loss, FedAvg or a state-vector simulator) and rebuild it in 30 minutes against that topic's `test_exercise.py`.

## 4. readiness

move on when you get at least 80% of section 1 (all the cross-cutting questions plus those for your blocks) and section 2, and the capstone's stage 1–4 tests and one block's tests pass in section 3. to test out of an elective without doing it, pass its per-block questions, its last topic's tests, and the blind rebuild of its core algorithm.
