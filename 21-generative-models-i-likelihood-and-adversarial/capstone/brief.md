# 21 · capstone: four families, one dataset

## goal

train one model from each family of this chapter on the **same data**, and compare them the way a careful paper would: likelihood, sample quality, diversity and cost, each with its uncertainty and its caveats. the deliverable is a table, a figure of samples and a written report that explains *why* the rankings come out as they do, and where they disagree.

| family | model (yours) | likelihood | sampling |
|---|---|---|---|
| autoregressive | `PixelCNN` (topic 02), 2 levels | exact | one forward pass per pixel |
| latent variable | `VAE` (topic 03), Bernoulli decoder | upper bound: `iwae_nll` | one pass |
| adversarial | `DCGANGenerator` / `DCGANDiscriminator` (topic 05) | none | one pass |
| flow | `RealNVP` (topic 07) on logit-transformed, dequantized pixels | upper bound: `flow_bits_per_dim` | one pass |

## data

**binarized MNIST.** the 60,000 training digits of `mlfs.datasets.mnist()` (cached under `~/.cache/mlfs/`), with each pixel drawn as a coin flip with its intensity as the probability (`binarize(X, seed=0)`; the 10,000 test digits with `seed=1`). binary pixels make the likelihoods of all three likelihood families exact or bounded *discrete* probabilities on the same scale: a model that knows nothing scores exactly 1 bit/dim, and published results on (statically) binarized MNIST are around 80–95 nats per image (0.15–0.17 bits/dim) for strong models (a PixelCNN reported 81.3 nats), though with a different binarization than ours, so treat them as a rough scale, not a benchmark.

**the ruler for samples.** FID, precision/recall and label entropy are computed in the feature space of your topic 01 `FeatureNet`, trained once on the *binarized* training digits (`cached_feature_net(X_train, y_train, name="mnist-binary", epochs=2, seed=0)`) so that its features describe binary digits. these numbers are comparable within the table and with nothing published (no Inception network here; see topic 01). offline (tests and `verify.py`), synthetic stand-ins with the same shapes are used; your report must use the real data.

## configurations

the constants `FULL` and `SMALL` in `capstone.py`:

| family | full config | small config (offline, tests) |
|---|---|---|
| autoregressive | 64 channels, 7 layers, 8 epochs, batch 128, Adam $10^{-3}$ | 16 channels, 3 layers, 2 epochs |
| VAE | latent 32, hidden (512, 256), 40 epochs, batch 128, Adam $10^{-3}$, IWAE $k = 500$ | latent 8, hidden (128,), 4 epochs, $k = 20$ |
| GAN | latent 64, base width 64, 15 epochs, batch 128, Adam $2 \times 10^{-4}$ ($\beta_1 = 0.5$) | latent 16, width 16, 3 epochs |
| flow | 8 couplings, hidden 1,024, 40 epochs, batch 256, Adam $5 \times 10^{-4}$, $\alpha = 0.05$ | 4 couplings, hidden 128, 4 epochs |

the full config runs on the best device `get_device()` finds (about an hour on an Apple M2); the small one on the CPU with 2,000 training and 400 test images. set `MLFS_CAPSTONE_FULL=0` to run the small config on real data while you develop.

## what you build

the docstrings in `capstone.py` are the specification; the checks in `test_capstone.py` enforce it.

| stage | functions | what they do |
|---|---|---|
| 1 | `binarize(X, *, seed=0)` | the shared dataset |
| 2 | `build_family(family, config, *, img_size=28, seed=0)`, `train_family(family, model, X, config, *, device, seed)` | one model per family; one training function with seeded shuffling, returning time and per-epoch losses |
| 3 | `nll_bits_per_dim(family, model, X, config, *, device, generator, batch_size)` | per-image test bits/dim: exact (PixelCNN), importance-weighted bound (VAE), dequantized bound (flow), `None` (GAN) |
| 4 | `sample_binary(family, model, n, config, *, img_size, device, generator)` | binary samples from every family, in the data's own space |
| 5 | `evaluate_samples(samples, real_feats, featnet, *, k=3)`, `bpd_ci(bpd, *, n_boot, alpha, seed)` | FID, precision, recall and label entropy; bits/dim with a bootstrap interval |

## deliverables

1. `capstone.py` with every check green (`python -m pytest test_capstone.py`).
2. the experiment on the real data, in your own cells: for every family, build, train (timed), score the test set, draw 2,000 samples twice (timed) and evaluate them; a reference row for 2,000 real training images; 16 samples per family.
3. the table: test bits/dim with its 95% bootstrap interval (and which values are bounds), FID for both sample draws, precision, recall, label entropy, parameters, training minutes and sampling milliseconds per image, plus the hardware.
4. the write-up at the end of the notebook (prompts there; grade it with [`rubric.md`](rubric.md)).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass |
| likelihood families, full config, real data | test bits/dim well below the uniform 1 bit/dim, with intervals |
| table | every family, every column, the real-data row and the hardware |

## reference results

measured once with the reference solution on an Apple M2 laptop (MPS, 16 GB) while other jobs shared the machine: the whole full config took **62.5 minutes** (training 48 minutes, PixelCNN sampling and the evaluations the rest). the feature network, trained on binarized digits, classifies binarized test digits with 96.8% accuracy.

| family | test bits/dim (95% CI) | nats / image | FID (draw 1 / 2) | precision | recall | label entropy | parameters | training | sampling |
|---|---|---|---|---|---|---|---|---|---|
| autoregressive (PixelCNN) | **0.1649** [0.1642, 0.1656], exact | 89.6 | 121.7 / 130.4 | 0.585 | 0.886 | 1.964 | 229,058 | 8.4 min | **198 ms / image** |
| VAE | 0.1695 [0.1686, 0.1704], bound ($k = 500$) | 92.1 | **46.2 / 49.2** | **0.789** | 0.906 | 2.232 | 1,091,920 | 1.1 min | 0.01 ms |
| GAN (DCGAN) | — | — | 51.9 / 49.9 | 0.667 | 0.906 | 2.232 | 1,622,914 | 16.0 min | 0.31 ms |
| flow (RealNVP) | 0.3752 [0.3744, 0.3759], bound | 203.9 | 316.7 / 322.7 | 0.622 | 0.210 | 1.419 | 27,691,392 | 22.8 min | 0.18 ms |
| real training images | — | — | 1.8 | 0.909 | 0.894 | 2.300 | | | |

(FID and precision/recall from 2,000 samples against the 10,000 test digits; label entropy for a uniform spread over 10 classes is 2.303.) what to take from it, and what not to:

- **the rankings disagree.** on likelihood: PixelCNN < VAE ≪ flow (the GAN can't be ranked). on FID: VAE ≈ GAN ≪ PixelCNN ≪ flow. the PixelCNN has the best likelihood and the second-worst FID; its samples have locally convincing strokes that often don't add up to a digit (precision 0.585), exactly the "good likelihood, poor samples" case of Theis et al. (2016).
- **bounds versus exact values.** the VAE's 92.1 nats is an *upper bound* on its NLL, 2.5 nats from the PixelCNN's exact 89.6: the VAE may well be as good a density model; the comparison can't tell. the flow's bound is loose for a structural reason: uniform dequantization of *binary* pixels asks the flow to spread each pixel's mass evenly over half of $[0, 1)$, and its samples show the cost as salt-and-pepper noise (and recall 0.21).
- **noise.** the bootstrap intervals cover only the test set's sampling noise (about ±0.0008 bits/dim); the two FID draws differ by up to 9 points, so FID differences below ~10 aren't meaningful here (VAE vs GAN is a tie); training-seed noise wasn't measured (one seed per family; the first stretch goal).
- **cost.** the PixelCNN needs 784 sequential network evaluations per image: in this run it sampled about 650 times slower than the GAN and about 20,000 times slower than the VAE, while the VAE trained in a minute. sampling times per image are indicative only (they depend on batch size, warm-up and what else the GPU is doing: a reviewer's rerun measured 44 ms per image for the GAN, not 0.31; the notebook now makes a warm-up call before timing), but the PixelCNN's orders-of-magnitude gap is structural. the flow has 120 times the PixelCNN's parameters for the worst numbers in the table.
- **the scale.** these binarized-MNIST likelihoods are in the range of published values (80–95 nats; ours use a different, dynamically drawn binarization), and the FIDs are in the course's own feature space: comparable within this table only.

the small config (offline, CPU, synthetic stand-in) runs the same notebook end to end in under a minute; its numbers are a smoke test, not results.

## stretch goals

- **seeds.** retrain the two cheapest families with a second seed: which differences in the table are larger than the seed-to-seed spread?
- **a tighter VAE bound.** report `iwae_nll` with $k = 1, 10, 100, 1000$: how much of the gap to the PixelCNN is the bound, and how much the model?
- **the GAN's stabilizer.** swap the DCGAN for a WGAN-GP with your `train_wgan` (a convolutional critic without batch norm): does sample quality or diversity change?
- **an autoregressive prior.** train a small PixelCNN over the code grids of a VQ-VAE (topic 04): sampling cost, sample quality, and the likelihood of the codes.
- **temperature.** sample the PixelCNN with logits divided by $T \in \{0.8, 1.2\}$: how do precision and recall trade off?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- LeCun, Cortes & Burges (1998), the MNIST database. [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/)
- Salakhutdinov & Murray (2008), "On the quantitative analysis of deep belief networks", *ICML*. binarized MNIST as a likelihood benchmark. [doi:10.1145/1390156.1390266](https://doi.org/10.1145/1390156.1390266)
- van den Oord, Kalchbrenner & Kavukcuoglu (2016), "Pixel recurrent neural networks", *ICML*. binarized MNIST likelihoods for autoregressive models. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
- Burda, Grosse & Salakhutdinov (2016), "Importance weighted autoencoders", *ICLR*. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
- Theis, van den Oord & Bethge (2016), "A note on the evaluation of generative models", *ICLR*. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
- Lucic, Kurach, Michalski, Gelly & Bousquet (2018), "Are GANs created equal? a large-scale study", *NeurIPS*. seeds, budgets and fair comparisons. [arXiv:1711.10337](https://arxiv.org/abs/1711.10337)
