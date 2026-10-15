# 19 · deep learning theory

> what we know, and don't, about why deep learning works.

by now you've trained MLPs, CNNs, RNNs and transformers, and they worked better than classical learning theory says they should: millions of parameters fitted to thousands of examples, non-convex losses minimized from random starts, models that keep improving as they get bigger. this chapter takes the main theoretical ideas people have built to explain that and puts each one to an experiment you run yourself. you'll build a ReLU network that represents a 256-piece sawtooth with 49 weights and watch gradient descent fail to find it; draw loss surfaces that mean something, and find a "sharp minimum" you can make five times sharper without changing a single prediction; compute the neural tangent kernel of any PyTorch model and see a 16,384-unit network land on the prediction a kernel method made before training started; memorize random labels and reproduce double descent; fit scaling laws to your own GPTs and predict a bigger model's loss; and derive the edge of chaos and μP, then watch a 20-layer network train well only near the edge and a learning rate tuned on a 64-wide network transfer to a 1,024-wide one. every theory comes with an honest account of where it stops working.

**you will build:** `relu_interpolant`, `sawtooth_net`, `count_linear_pieces`, `width_for_budget`, `fit_regressor` (topic 01); `filter_normalized_direction`, `loss_surface_2d`, `linear_interpolation`, `loss_barrier`, `hessian_vector_product`, `top_hessian_eigenvalue` (02); `empirical_ntk`, `kernel_regression_predict`, `NTKNet`, `relu_ntk`, `ntk_gd_predictions`, `gd_train`, `relative_change` (03); `corrupt_labels`, `RandomReLUFeatures`, `min_norm_least_squares`, `double_descent_curve`, `gd_least_squares` (04); `power_law`, `local_exponents`, `fit_power_law`, `compute_optimal`, `chinchilla_loss`, `optimal_allocation`, `non_embedding_params` (05); `gaussian_expectation`, `variance_map`, `q_fixed_point`, `chi1`, `propagate`, `MuMLP`, `param_groups`, `activation_deltas` (06). the landscape, NTK and scaling-law tools make up **`mlfs.dltheory`** (`filter_normalized_direction`, `loss_surface_2d`, `linear_interpolation`, `empirical_ntk`, `kernel_regression_predict`, `fit_power_law`, `power_law`, `compute_optimal`), which later chapters use to look at their own models.
**prerequisites:** chapter 13's generalization bounds (`mlfs.theory`), chapter 09's kernels and kernel ridge regression (`mlfs.kernels`), chapter 15's initialization, Adam and PyTorch toolkit (`mlfs.train`, `mlfs.torchnn`), chapter 18's GPT and `train_lm` (`mlfs.transformer`) for topic 05, and the math of chapters 02–06 (eigendecompositions, power iteration, Hessians, Gaussian expectations, gradient descent on quadratics).
**time · scaffolding:** about 75 hours · L4 (spec-driven): each stage is specified in a **spec** box in the lesson and checked live; the notebooks give signatures and one-line docstrings, the hints have one level. the capstone gives the brief, the experiment cells and the checks; the five functions it needs are yours.
**newly unlocked library components:** from here on you may use `torch.autograd.functional.hessian`/`hvp` and Hessian-spectrum tools such as PyHessian (topic 02), loss-landscape plotting libraries (02), `torch.func.jacrev`/`vmap` Jacobians and NTK libraries such as `neural-tangents` (03), `scipy.optimize.curve_fit` for scaling-law fits (05), and the `mup` package (06).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [approximation theory](01-approximation-theory/) | `relu_interpolant`, `sawtooth_net`, `count_linear_pieces`, `width_for_budget`, `fit_regressor` | depth represents a 64-piece sawtooth with 37 weights (a shallow net needs 193), yet no network trained from random starts finds it (relative error 1.00); on a radial ripple at an equal 3,000-parameter budget, 4 layers beat 1 layer by 6–15× (0.06–0.15 vs 0.89); random 8-layer networks have about 10 linear pieces from 48 units |
| 02 | [loss landscapes](02-loss-landscapes/) | `filter_normalized_direction`, `loss_surface_2d`, `linear_interpolation`, `loss_barrier`, `top_hessian_eigenvalue` → `mlfs.dltheory` | the sharpness starts at 29, catapults to about 2 within three steps, then sharpens progressively back up to $2/\eta = 10$, where full-batch GD sits while batch 16 stays near 4, with the same test error; rescaling two layers makes a minimum 5.6× sharper with identical predictions; two nets from one initialization are linearly connected (test-error barrier about 0.01, though the midpoint's training loss doubles), from two initializations the midpoint misclassifies 42–65% |
| 03 | [infinite width: NTK & mean field](03-infinite-width-ntk-and-mean-field/) | `empirical_ntk`, `kernel_regression_predict`, `NTKNet`, `relu_ntk`, `ntk_gd_predictions`, `gd_train` → `mlfs.dltheory` | the empirical NTK converges at slope −0.51 in width; a 16,384-unit network moves its weights 4% and lands within 4% of its linearization's pre-training prediction; at width 1,024 the mean-field network's kernel grows about 7× while the NTK one's changes 12% |
| 04 | [generalization in deep learning](04-generalization-in-deep-learning/) | `corrupt_labels`, `RandomReLUFeatures`, `min_norm_least_squares`, `double_descent_curve`, `gd_least_squares` | an MLP memorizes fully random labels in 83 epochs (true labels: 23); random-features test error goes 16% → 86% at $p = n$ → 10.5% at $p = 8n$, a ridge of 0.001 removes the peak; GD from zero lands exactly on the minimum-norm interpolant, from a random start 4 points worse |
| 05 | [scaling laws](05-scaling-laws/) | `power_law`, `local_exponents`, `fit_power_law`, `compute_optimal`, `optimal_allocation`, `non_embedding_params` → `mlfs.dltheory` | a power law with an offset, fitted to five of your GPTs, predicts the sixth within 0.003 nats (without the offset: off by 0.08), but per-seed fits disagree by 0.045; at a fixed compute budget the smallest model wins with 900 tokens per parameter, far from Chinchilla's 20 |
| 06 | [parameterization & signal propagation](06-parameterization-and-signal-propagation/) | `variance_map`, `q_fixed_point`, `chi1`, `propagate`, `MuMLP`, `param_groups`, `activation_deltas` | a 20-layer tanh net trains well only in a window around the edge of chaos, $\chi_1 \approx 0.9$–$1.1$ (loss 0.75–0.95, best at the edge, $\sigma_w = 1.40$), and stays at or near chance far from it; SP outputs move 27× more per step at width 2,048 than at 64, μP's are flat; SP's best learning rate falls 16× over a 16× width increase, μP's stays put |

## capstone

*reproduce a theory figure*. the required figure is **model-wise double descent with trained networks**: two-layer networks from 4 to 1,000 hidden units (3 thousand to 800 thousand parameters) trained to interpolate 4,000 MNIST digits, with 0% and 20% of the labels wrong, three seeds per width, test error and loss against parameters with t intervals. the reference run (14.6 minutes on a laptop GPU) finds the textbook shape with 20% label noise: test error 34.4% at width 32, a peak of 41.8% at width 80 (63,610 parameters, where the training error first reaches zero), and 20.2% at width 1,000, while the clean-label control falls monotonically to 6.0%. the options are the NTK-convergence figure (Jacot et al., Lee et al.) and the μTransfer figure (Yang et al., 2022), built from topics 03 and 06. see [`capstone/brief.md`](capstone/brief.md). small CPU config: under a minute on the offline synthetic stand-in.

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a ReLU interpolant, an infinite-width kernel value, a compute-optimal allocation), and a 60-minute blind rebuild of the empirical NTK, kernel regression, the infinite-width ReLU kernel and the closed-form linearized dynamics, checked by topic 03's tests.

## notation

| symbol | meaning |
|---|---|
| $\theta$, $\theta_0$, $P$ | all parameters, their initial values, their number |
| $n$, $d$, $K$ | training-set size, input dimension, number of classes |
| $m$ (or $n$ for a layer), $L$ | width (hidden units), number of hidden layers |
| $t(x)$, $t^k$ | the tent map $1 - \lvert 2x - 1\rvert$ and its $k$-fold composition |
| $d_1, d_2$; $\alpha, \beta$ | landscape directions; coordinates along them |
| $H$, $\lambda_{\max}$, $\eta$ | Hessian of the training loss, its largest eigenvalue (sharpness), learning rate |
| $B$ | loss barrier along a path |
| $J$, $\Theta(x, x')$, $K$ | Jacobian of the outputs, (empirical) neural tangent kernel, its Gram matrix on the training set |
| $\Theta_\infty$, $\vartheta$ | infinite-width ReLU NTK, angle between two inputs |
| $f_0$, $f_t$ | the network's function at initialization and after $t$ steps |
| $\phi(x)$, $\Phi$, $p$ | random features of an input, the training feature matrix, number of features |
| $X^+$, $\lambda$ (ridge) | pseudoinverse, ridge penalty |
| $N$, $D$, $C$ | (non-embedding) parameters, training tokens, training FLOPs $\approx 6ND$ |
| $a, b, c$ | power-law scale, exponent and offset, $L = aN^{-b} + c$ |
| $E, A, B, \alpha, \beta$ | the parametric loss $E + A/N^\alpha + B/D^\beta$ |
| $\sigma_w, \sigma_b$, $q^l$, $q^*$ | initialization scales, pre-activation variance at layer $l$, its fixed point |
| $\mathcal V$, $\mathcal C$, $\chi_1$ | variance map, correlation map, slope of $\mathcal C$ at $c = 1$ |
| SP, μP, $m$ (width multiplier) | standard and maximal-update parameterizations, width / base width |

## how this connects

- builds on: PAC learning, VC and Rademacher bounds, PAC-Bayes (ch 13); kernels, kernel ridge and random features (ch 09); bias–variance (ch 07); initialization, Adam, schedules and the PyTorch toolkit (ch 15); your GPT and its capstone's scaling curve (ch 18); eigenvalues and power iteration (ch 02), Hessians (ch 03), Gaussian expectations (ch 04), gradient descent on quadratics (ch 06).
- used in: chapter 20's expressivity questions for graph networks; scaling and compute budgets in chapters 26–27 (`compute_optimal`, `fit_power_law`); the NTK and landscape tools wherever later chapters ask whether a model is lazy or sharp; and the research milestones, where a scaling curve or a theory figure is often the most convincing small experiment.

## core references

- Telgarsky (2016), "Benefits of depth in neural networks", *COLT*. [arXiv:1602.04485](https://arxiv.org/abs/1602.04485)
- Li, Xu, Taylor, Studer & Goldstein (2018), "Visualizing the loss landscape of neural nets", *NeurIPS*. [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- Jacot, Gabriel & Hongler (2018), "Neural tangent kernel: convergence and generalization in neural networks", *NeurIPS*. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS* 116(32). [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
- Hoffmann et al. (2022), "Training compute-optimal large language models", *NeurIPS*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen & Gao (2022), "Tensor programs V: tuning large neural networks via zero-shot hyperparameter transfer". [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- Roberts, Yaida & Hanin (2022), *The Principles of Deep Learning Theory*, Cambridge University Press. [arXiv:2106.10165](https://arxiv.org/abs/2106.10165)
