# 22 · capstone: a class-conditional generator for CIFAR-10

## goal

build a class-conditional image generator for CIFAR-10 with classifier-free guidance, twice: once as a **DDPM** noise predictor and once as a **flow-matching** velocity field, with the same network, the same data, the same number of optimizer steps and the same label dropout, and sample both with the same number of network calls. then answer, with numbers and their noise: which objective gives better samples at a small sampling budget, which at a larger one, how guidance trades class fidelity against distribution quality, and which "details" of the pipeline (the weight average, the sampler's clipping) decide the answer.

| piece | yours, from |
|---|---|
| `TimeUNet`, `DiffusionSchedule`, `ddpm_loss` | topic 03 |
| `ddim_sample`, `alpha_bar_at` | topic 04 |
| `flow_matching_loss`, `sample_flow` | topic 05 |
| `drop_labels`, `cfg_model` | topic 06 |
| `cached_feature_net`, `fid_from_features` | chapter 21 · topic 01 |
| `ema_update`, `get_device`, `count_parameters` | chapter 15 · topic 07 |

## data

**CIFAR-10**: the 50,000 training images of `mlfs.datasets.cifar10()` (cached under `~/.cache/mlfs/`), scaled to $[-1, 1]$, 10 classes; the 10,000 test images are the reference set for FID. offline (tests and `verify.py`), a synthetic stand-in with the same shapes is used; your report must use the real data.

**the ruler.** FID and class accuracy are computed with chapter 21's `FeatureNet`, trained once on the training images with `cached_feature_net(..., name="cifar10", epochs=5, seed=0)`. it classifies the test images with 62.6% accuracy (a small CNN trained for 5 epochs), and its 128-dimensional penultimate layer is the feature space. **these FIDs are course-internal**: comparable within this table and with nothing published (published CIFAR-10 FIDs use an Inception network, other sample sizes and other reference sets; the numbers below are not on that scale at all). class accuracy is the feature network's agreement with the requested label; real training images score 63.1%, so that's roughly the ceiling of an honest sampler.

## configurations

the constants `FULL` and `SMALL` in `capstone.py`:

| | full config | small config (offline, tests) |
|---|---|---|
| network | `TimeUNet`, base width 32, `channel_mults=(1, 2, 2)`, dropout 0.1, 10 classes (1,554,179 parameters) | base width 8, `channel_mults=(1, 2)`, no dropout |
| training (per objective) | 2,500 steps of batch 128 (6.4 epochs), Adam $5 \times 10^{-4}$ with 100 warmup steps, EMA 0.999 (warmed up), $p_\text{uncond} = 0.1$, horizontal flips | 40 steps of batch 32 on 2,000 images |
| DDPM | 1,000-step linear schedule ($\beta$ from $10^{-4}$ to 0.02); sampled with DDIM ($\eta = 0$) | 100-step schedule (betas rescaled) |
| flow matching | straight paths, sampled with Euler | the same |
| sampling | the predicted clean image clipped to $[-1, 1]$ at every step (`ClipPrediction`); 1,000 samples per cell (100 per class) | 60 samples per cell |

both objectives start from the same initialization (`build_model(..., seed=0)`) and see the same minibatches, flips and dropped labels (one generator seeded with `seed`). the full config runs on the best device `get_device()` finds; set `MLFS_CAPSTONE_FULL=0` to run the small config on real data while you develop.

## what you build

the docstrings in `capstone.py` are the specification; the checks in `test_capstone.py` enforce it.

| stage | functions | what they do |
|---|---|---|
| 1 | `prepare_cifar(X, y)`, `to_unit(x)`, `random_hflip(x, *, generator)` | the data in model space, back to $[0, 1]$, and the augmentation |
| 2 | `build_model(config, *, seed)`, `make_schedule(config)` | one network for both objectives; the DDPM schedule |
| 3 | `train_generator(objective, model, X, y, config, *, device, seed)` | label dropout, warmup, a warmed-up EMA; identical budgets for `'ddpm'` and `'flow'` |
| 4 | `ClipPrediction(model, objective, schedule)`, `generate(objective, model, labels, config, *, guidance_scale, n_steps, device, generator, batch_size)` | classifier-free guidance in one batched call per step, the same number of network calls for both objectives, optional clipping of the implied clean image |
| 5 | `evaluate(images, labels, featnet, real_feats)` | course FID and class accuracy |

## deliverables

1. `capstone.py` with every check green (`python -m pytest test_capstone.py`).
2. the experiment on the real data, in your own cells: train both objectives (timed); sample each EMA model at $w = 2$ with 5, 10 and 25 network calls and at 25 calls with $w = 1$, 2 and 4; the main setting ($w = 2$, 25 calls) a second time with another noise seed, once with the raw (non-EMA) weights, and once without clipping; the real-data reference row; a sample grid per objective at the smallest and largest $w$.
3. the table: FID and class accuracy for every cell, the sampling noise, the reference row, parameters, training minutes and sampling seconds, and the hardware.
4. the write-up at the end of the notebook (prompts there; grade it with [`rubric.md`](rubric.md)).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass |
| full config, real data | both objectives produce recognizable class-conditional samples at $w = 2$; every cell of the table filled |
| table | both objectives, every column, the second draw, the ablations, the real-data row and the hardware |

## reference results

measured once with the reference solution on an Apple M2 laptop (16 GB, `mps`), training seed 0, while other jobs shared the machine (load average about 4): the whole full config took **43 minutes** (training 17.8 and 17.9 minutes, about 0.43 s per step; each table cell 6–32 s of sampling; the feature network was already cached, otherwise add about 2 minutes). the samples are recognizable at $w = 2$–4 (cars, ships, horses, frogs) but far from a converged CIFAR-10 model: this is 6 epochs of a 1.5M-parameter network.

| objective | guidance $w$ | network calls | FID (course) | class accuracy | sampling (1,000 images) |
|---|---|---|---|---|---|
| DDPM | 2 | 5 | 21.93 | 0.362 | 6 s |
| DDPM | 2 | 10 | 10.21 | 0.455 | 13 s |
| DDPM | 1 | 25 | 12.69 | 0.272 | 32 s |
| DDPM | 2 | 25 | **5.42** (second draw 6.53) | 0.469 (0.471) | 30 s |
| DDPM | 4 | 25 | 10.85 | 0.730 | 32 s |
| DDPM, raw weights | 2 | 25 | 12.53 | 0.430 | |
| DDPM, no clipping | 2 | 25 | 19.76 | 0.246 | |
| flow matching | 2 | 5 | **16.05** | 0.700 | 6 s |
| flow matching | 2 | 10 | **8.26** | 0.740 | 12 s |
| flow matching | 1 | 25 | 7.19 | 0.459 | 30 s |
| flow matching | 2 | 25 | 5.84 (second draw 5.44) | 0.732 (0.772) | 31 s |
| flow matching | 4 | 25 | 29.02 | 0.901 | 30 s |
| flow matching, raw weights | 2 | 25 | 5.72 | 0.656 | |
| flow matching, no clipping | 2 | 25 | 5.84 | 0.732 | |
| 1,000 real training images | | | 0.35 | 0.631 | |

what to take from it, and what not to:

- **the noise.** two sample draws of the same setting differ by 1.1 FID points for DDPM and 0.4 for flow matching (and by up to 0.04 in accuracy): differences below about 1.5 points are ties at 1,000 samples. only one training seed was run, so the seed-to-seed spread is unknown (the first stretch goal); treat single-cell differences of a few points with suspicion.
- **few calls may favour flow matching; 25 calls is a tie.** at 5 calls flow matching is well ahead (16.1 vs 21.9), as its straighter ODE predicts (topic 05; DDIM is Euler on a more curved path, topic 04). at 10 calls its lead (8.3 vs 10.2) is only about twice the draw-to-draw spread, from one draw and one training seed: suggestive, not established. at 25 calls the two are within the sampling noise (5.4/6.5 vs 5.8/5.4).
- **guidance has a sweet spot.** both objectives have their best FID at $w = 2$ while class accuracy keeps rising with $w$ (DDPM 0.27 → 0.47 → 0.73, flow 0.46 → 0.73 → 0.90, above the 0.63 of real images): at large $w$ the samples become stereotyped class prototypes (in the grid, the $w = 4$ flow samples of a class look alike), which is what topic 06's lab showed on 2-D classes. flow matching's FID degrades much faster with $w$ (29.0 at $w = 4$) than DDPM's (10.9): at equal $w$ the two objectives are not equally guided.
- **the pipeline decides the comparison.** without clipping the predicted clean image, DDPM's FID goes from 5.4 to 19.8 while flow matching doesn't change at all (5.84 both ways: its implied $\hat x_1 = x + (1 - t)v$ involves no division, and at $w = 2$ only 0.6% of its pixels fall outside $[-1, 1]$ on an average step, 1.5% on the worst, measured on a separate training run with the same seed and config; at $w = 4$ it is 5%, and these few, slightly out-of-range pixels are clamped again at the end anyway); the raw weights cost DDPM 7 points and flow matching nothing. both effects come from the same fact: $\epsilon$-prediction recovers the image as $(x - \sqrt{1 - \bar\alpha}\,\epsilon)/\sqrt{\bar\alpha}$, so at high noise any error in $\epsilon$ is amplified by up to $1/\sqrt{\bar\alpha_{T-1}} \approx 160$. an earlier run of this capstone used an EMA *without* warmup (decay 0.999 from the first step): after 1,500 steps 22% of the averaged weights were still the initialization, the averaged network's $\epsilon$ was systematically too small, and DDPM's FID was **248** (raw weights: 19.0; flow matching barely affected). a comparison of objectives that doesn't control these details is a comparison of pipelines.
- **the scale.** FIDs here are in the course's own feature space and at 1,000 samples, against a reference set the feature network has seen classes of; they are useful for ranking these runs and nothing else.

the small config (offline, CPU, synthetic stand-in) runs the same notebook end to end in under a minute; its numbers are a smoke test, not results.

## stretch goals

- **seeds.** retrain both objectives with a second seed (45 more minutes) and state which differences in the table survive.
- **a fairer sampler.** sample both models with the same second-order solver (midpoint or Heun on each model's ODE, topic 04's `probability_flow_sample` for DDPM) at equal network calls: does flow matching's few-step advantage shrink?
- **guidance interval.** apply guidance only in the middle of the sampling trajectory (Kynkäänniemi et al., 2024): can you keep the accuracy of $w = 4$ with the FID of $w = 2$?
- **latent diffusion.** train a `ConvVAE` on CIFAR-10 and a conditional `DiT` on its latents (topic 07) with the same 35-minute budget: cost per step, FID, and the reconstruction floor.
- **reflow.** one round of rectified flow (topic 05) on the flow-matching model: what does 1–5-call sampling look like?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Krizhevsky (2009), "Learning multiple layers of features from tiny images", technical report. CIFAR-10. [cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Ho, Jain & Abbeel (2020), "Denoising diffusion probabilistic models", *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, Meng & Ermon (2021), "Denoising diffusion implicit models", *ICLR*. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow matching for generative modeling", *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Ho & Salimans (2021), "Classifier-free diffusion guidance". [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Karras, Aittala, Aila & Laine (2022), "Elucidating the design space of diffusion-based generative models", *NeurIPS*. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Kynkäänniemi, Aittala, Karras, Laine, Aila & Lehtinen (2024), "Applying guidance in a limited interval improves sample and distribution quality in diffusion models", *NeurIPS*. [arXiv:2404.07724](https://arxiv.org/abs/2404.07724)
