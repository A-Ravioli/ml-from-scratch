# 22 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | a model doesn't train or sample | all checks pass | all checks pass, and the full run produces every cell of the table on real CIFAR-10 with both objectives trained from the same initialization and budget | all of that, with the comparison provably fair (same network, data order, steps, batch, label dropout and EMA; samplers at equal network calls; the same labels and noise seeds for both), and a pipeline another person could rerun from the notebook alone |
| **results** | a table without units or uncertainty | numbers without the real-data row or the sampling noise | FID and class accuracy for both objectives over the guidance scales and call budgets, the second sample draw, the real-data reference row, parameters, training and sampling time, the hardware | all of that plus a second training seed for at least the main setting, and a statement of which differences exceed the seed-to-seed spread |
| **analysis** | the numbers are restated | each objective's numbers described | DDPM vs flow matching explained at few and at many calls (ODE curvature, DDIM as Euler, straight paths), the guidance tradeoff explained ($p(y \mid x)^w$, FID vs accuracy), with the noise level stated | all of that plus an ablation that tests one explanation (e.g. sampling both models with the same higher-order solver, or turning off the $\hat x$ clipping) and a costed plan for ten times the compute |
| **clarity** | hard to follow | readable | one table, one sample figure per objective, numbers with sensible precision, every metric defined | a reader could reproduce any number (config, seed, device, minutes) and knows which numbers are course-internal and not comparable with published FIDs |

## what graders look for

- **equal compute, stated precisely.** same network, same number of optimizer steps and batch size, same label dropout and EMA; at sampling time the same number of network calls (DDIM steps = Euler steps), each call on $2B$ rows because of guidance.
- **course-internal FID.** the feature network is chapter 21's `FeatureNet` trained on CIFAR-10, not Inception: say so next to the table, and don't compare with published CIFAR-10 FIDs.
- **noise before conclusions.** the two sample draws show how much FID moves from sampling alone; a training seed shows how much moves from training; differences smaller than these are ties.
- **guidance read correctly.** class accuracy keeps rising with $w$ while FID has a minimum; the write-up says which one tracks what, and shows the large-$w$ samples.
- **the sampler is part of the model.** a write-up that notices the $\hat x_0$ clipping (or its absence) changes DDPM's few-step results, and tests it, is doing the reasoning this chapter is about.
