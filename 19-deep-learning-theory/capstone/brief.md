# 19 · capstone: reproduce a theory figure

## goal

pick one figure from the theory of deep learning, reproduce it carefully on a laptop, and write up how closely your measurement matches the theory and where it doesn't. the **required core** is the figure that started the modern generalization debate: **model-wise double descent with trained networks** (Belkin, Hsu, Ma & Mandal, 2019, figure 3; Nakkiran et al., 2020). two-layer networks of increasing width are trained to fit 4,000 MNIST digits, a fifth of them with the wrong label, until they interpolate. classical intuition says the test error should rise once the model can memorize. it does, until the networks have just enough parameters to fit every label, and then it falls again, far below its first minimum.

the figure uses every topic of the chapter. the random-features version of topic 04 is the exactly solvable model of the same curve, and the write-up compares the two; the networks at the right of the figure are overparameterized interpolators of the kind topics 03 (lazy training) and 04 (implicit bias) try to explain; the threshold is a parameter count, as in the approximation results of topic 01; the comparison of seeds uses intervals the way topic 05's fits should. two **options** reproduce the other theory figures of the chapter (NTK convergence and μTransfer) and can replace or extend the write-up's analysis; they're described at the end.

## data

**MNIST** from `mlfs.datasets.mnist()`, cached under `~/.cache/mlfs/`: a random subset of 4,000 training images (drawn with your `noisy_subset`, seed 0), pixels scaled to $[0, 1]$, flattened to 784 numbers and centred by the training mean, and the first 5,000 test images with their true labels. the training labels come in two versions: 20% of them (exactly 800) moved to a uniformly random *other* class, and clean, as the control. offline (`MLFS_OFFLINE=1`) you get the synthetic stand-in of 2,000 smooth class templates, so the notebook always runs. the tests use neither: they use small synthetic problems.

## configurations

| config | sweep | training | runtime |
|---|---|---|---|
| **small** (tests, `verify.py`, offline, or `MLFS_CAPSTONE=small`) | 300 images; widths 2–128, 2 seeds; control at 3 widths, 1 seed | 400 full-batch Adam steps at $3 \times 10^{-3}$, CPU | under a minute |
| **full** (your write-up) | 4,000 images, 20% label noise: widths 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 100, 150, 250, 500, 1,000 (3,190 to 795,010 parameters), seeds 0, 1, 2; clean-label control: widths 8, 32, 64, 100, 250, 1,000, seeds 0, 1 | each network: `Linear(784, w) → ReLU → Linear(w, 10)`, PyTorch's default initialization, 4,000 full-batch Adam steps at $10^{-3}$ on the squared error to one-hot labels (no weight decay, no early stopping), `get_device()` | **14.6 minutes** for the whole notebook (14.5 of them the 57 training runs), measured on an Apple-silicon laptop GPU (`mps`) with no other GPU jobs. an earlier, larger version of the config (the control at all 15 widths and 3 seeds, 90 runs) took 87 minutes while two other jobs shared the GPU: on a busy machine, expect several times longer. on a laptop CPU (4 threads) the measured per-step times (2.5 ms at width 4 to 37 ms at width 1,000) add up to about 35 minutes of training when the machine is idle; a reviewer measured 76 ms per step at width 1,000 on a loaded machine, which puts the whole sweep at 1–1.5 hours |

full-batch training on the squared loss follows Belkin et al.; a fixed, long step budget without early stopping is what lets the networks interpolate the noisy labels. the threshold to watch is $nK = 4{,}000 \times 10 = 40{,}000$ parameters, a width of about 50 (the number of parameters needed to fit 10 outputs at 4,000 points).

## deliverables

1. `capstone.py` with the five stages implemented and their checks green:
   - `noisy_subset(X, y, n, *, noise=0.0, n_classes=10, rng=None)`: `n` images without replacement (`ValueError` if there aren't enough), flattened float32 in $[0, 1]$, their true labels, and a copy with exactly `round(noise * n)` labels moved to a uniformly random different class
   - `train_interpolator(model, X, y, *, steps=4000, lr=1e-3, loss="mse", n_classes=10, device="cpu")` and `evaluate(model, X, y, *, loss="mse", n_classes=10, device="cpu")`: full-batch Adam in train mode, recording the loss before each step; test error and loss without gradients in eval mode, restoring the mode (`task_loss` is given: squared error to one-hot summed over classes and averaged over examples, or cross-entropy)
   - `model_wise_curve(X_train, y_train, X_test, y_test, widths, *, seeds=(0,), steps=4000, lr=1e-3, loss="mse", n_classes=10, device="cpu")`: for every width and seed, `torch.manual_seed(seed)`, the given `two_layer` model, your trainer and your evaluation; arrays `(len(widths), len(seeds))` of training error, test error and test loss, and the parameter counts
   - `locate_threshold(widths, train_error, test_metric, *, tol=0.01)`: the first width whose seed-mean training error is at most `tol`, the width where the seed-mean test metric (error or loss) peaks at or beyond that width, and the width of its lowest point below the peak; the notebook runs it on both test error and test loss
   - `mean_ci(values, *, alpha=0.05)`: seed means with $t$ intervals (no interval for a single seed)
2. the full-config run, its figure and table, and the write-up section at the end of the notebook, answering its six prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`pytest test_capstone.py`) |
| the noisy-label curve | a test-*error* peak at or just past the width where the training error reaches zero, above the first minimum of the test error (compare their seed intervals, and say how much they separate), then a descent below the first minimum at the largest widths; the test loss as a second view |
| the control | the clean-label curve, on the same axes, without a peak |
| reporting | the figure (training error, test error, test loss against parameters, threshold marked), the table with seed intervals, the located widths, the runtime on your device, and the comparison with topic 04's random-features curve |

what the reference gets (full config, `mps`):

noisy labels (20% wrong), means over seeds 0–2 with 95% $t$ intervals:

| width | parameters | train error | test error | test loss |
|---|---|---|---|---|
| 4 | 3,190 | 0.529 | 0.577 [0.499, 0.654] | 0.790 [0.528, 1.052] |
| 16 | 12,730 | 0.104 | 0.392 [0.259, 0.524] | 1.145 [0.481, 1.809] |
| 32 | 25,450 | 0.038 | **0.344** [0.306, 0.382] | 1.083 [0.950, 1.215] |
| 48 | 38,170 | 0.009 | 0.373 [0.360, 0.386] | 1.338 [1.230, 1.445] |
| 64 | 50,890 | 0.002 | 0.407 [0.361, 0.452] | 1.560 [1.481, 1.639] |
| 80 | 63,610 | 0.000 | **0.418** [0.398, 0.437] | **1.616** [1.561, 1.670] |
| 100 | 79,510 | 0.000 | 0.410 [0.384, 0.435] | 1.515 [1.465, 1.566] |
| 150 | 119,260 | 0.000 | 0.367 [0.337, 0.398] | 1.230 [1.225, 1.235] |
| 250 | 198,760 | 0.000 | 0.303 [0.291, 0.314] | 0.916 [0.887, 0.946] |
| 500 | 397,510 | 0.000 | 0.236 [0.226, 0.245] | 0.669 [0.659, 0.679] |
| 1,000 | 795,010 | 0.000 | **0.202** [0.191, 0.214] | 0.544 [0.530, 0.558] |

(the full table in the notebook has all 15 widths.) the test error falls to 34.4% at width 32, rises to a peak of 41.8% at width 80, where the training error first reaches (almost) zero (0.0005 in seed 2: two of the 4,000 images unfitted; 0 in seeds 0 and 1), and then falls steadily to 20.2% at width 1,000, fourteen points below the first minimum. on the test error the peak only just clears the first minimum: the intervals [0.398, 0.437] and [0.306, 0.382] are 0.016 apart, with three seeds (2 degrees of freedom) and both widths chosen after looking at the curve, so treat it as suggestive rather than significant. the test loss separates them clearly: 1.08 [0.95, 1.22] at width 32 against 1.62 [1.56, 1.67] at width 80, then 0.54 at width 1,000.

`locate_threshold` on the noisy curve: interpolation width 48 (mean training error 0.009 ≤ 0.01) for both metrics; on the **test error**, peak at width 80 (63,610 parameters, 1.6 times $nK$) and first minimum at width 32; on the **test loss**, peak at width 80 and first minimum at width 4. the loss's "first minimum" is an artifact: the narrowest networks underfit so badly that their outputs stay near zero, which the squared loss rewards, so the classical U is read off the test error. on the clean control both metrics give interpolation width 32, "peak" 32 and first minimum 8, but there the metric at the "peak" is *below* the first minimum (0.179 against 0.297 error): no peak, as the function's definition makes visible only if you compare the two values.

clean labels (control), seeds 0–1:

| width | 8 | 32 | 64 | 100 | 250 | 1,000 |
|---|---|---|---|---|---|---|
| test error | 0.297 | 0.179 | 0.138 | 0.113 | 0.070 | 0.060 |
| test loss | 0.586 | 0.549 | 0.508 | 0.429 | 0.260 | 0.195 |

no peak: with clean labels the test error falls monotonically with width (an earlier run of the control at all 15 widths and 3 seeds agreed: monotone within the intervals, 5.9% at width 1,000). the noise is what the interpolating networks have to memorize, and the peak is the price of memorizing it with the fewest possible parameters, as in topic 04's random-features curve, though much gentler: topic 04's peak was 86% error and 800 times the minimum MSE exactly at $p = n$; here the peak is 1.2 times the first minimum's error and sits at 1.6 $nK$, where the networks first fit every label.

## stretch goals

- **cross-entropy.** rerun the noisy sweep with `loss="ce"`. Nakkiran et al. used cross-entropy; does the peak survive, and where is it?
- **a ridge.** add weight decay to the trainer and sweep its strength at the peak width and at width 1,000. topic 04 found that a small ridge removed the random-features peak entirely.
- **epoch-wise double descent.** record the test error during training for widths 64 and 1,000 (every 250 steps). Nakkiran et al. found the test error of large models falls, rises and falls again with training time.
- **sample-wise.** fix the width at 64 and vary the number of training images around 5,000: near the threshold, more data can make the test error worse.
- **more seeds at the peak.** three seeds give wide intervals at small widths (the $t$ quantile with 2 degrees of freedom is 4.3); add seeds where your conclusions depend on them.

## options: the other two theory figures

each option can replace prompts 3–5 of the write-up (keep prompts 1, 2 and 6 for the core figure) or be added as an appendix. both reuse your own topic code: load it with `notebooks.load_topic("19-deep-learning-theory/03-infinite-width-ntk-and-mean-field", key="ntk")` (or `06-parameterization-and-signal-propagation`) as AUTHORING.md §5.6 describes.

- **NTK convergence** (Jacot et al., 2018, figure 1; Lee et al., 2019). with `NTKNet`, `empirical_ntk`, `relu_ntk`, `ntk_gd_predictions` and `gd_train`: for widths from 64 to 65,536, (a) the relative distance between the empirical NTK at initialization and $\Theta_\infty$ on 100 MNIST digits (normalized to unit norm, with a constant coordinate as in topic 03's lab), and (b) the relative distance between a trained network's test predictions and its linearization's, and the change of its kernel during training, each with 5 seeds. the figure is three lines on log-log axes with fitted slopes. the theory says $-\frac12$ for all three; report what you get and where finite width shows. topic 03's lab measured slope $-0.51$ for (a) on 16 points of a circle. budget: a few minutes on a CPU up to width 16,384; the largest widths need a GPU or patience.
- **μTransfer** (Yang et al., 2022, figure 1). with `MuMLP` and `param_groups`: train MLPs of widths 64 to 4,096 (or a small transformer if you add μP to your GPT) on CIFAR-10 or MNIST, sweeping the Adam learning rate over 8–10 values for each width in both parameterizations, 3 seeds, and plot training loss against learning rate. report the optimum per width with its uncertainty (fit a parabola in log learning rate around the minimum), and whether the μP optimum is constant within it. topic 06's lab found the SP optimum moving 16-fold over a 16-fold width increase and the μP optimum constant at the resolution of a factor-4 grid; it also found SP at its own tuned rate *beating* μP at width 1,024 after 200 steps, which a longer run should test. budget: about 30–60 minutes on `mps` for widths up to 2,048 and 1,000 steps.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS* 116(32). [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
- Nakkiran, Kaplun, Bansal, Yang, Barak & Sutskever (2020), "Deep double descent: where bigger models and more data hurt", *ICLR*. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
- Advani & Saxe (2017), "High-dimensional dynamics of generalization error in neural networks". [arXiv:1710.03667](https://arxiv.org/abs/1710.03667)
- Spigler, Geiger, d'Ascoli, Sagun, Biroli & Wyart (2019), "A jamming transition from under- to over-parametrization affects generalization in deep learning", *Journal of Physics A* 52(47). [arXiv:1810.09665](https://arxiv.org/abs/1810.09665)
- Jacot, Gabriel & Hongler (2018), "Neural tangent kernel: convergence and generalization in neural networks", *NeurIPS*. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- Lee, Xiao, Schoenholz, Bahri, Novak, Sohl-Dickstein & Pennington (2019), "Wide neural networks of any depth evolve as linear models under gradient descent", *NeurIPS*. [arXiv:1902.06720](https://arxiv.org/abs/1902.06720)
- Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen & Gao (2022), "Tensor programs V: tuning large neural networks via zero-shot hyperparameter transfer". [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
