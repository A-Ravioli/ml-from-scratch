# 14 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs on real MNIST with your own `mlfs.tensor`, `mlfs.nn` and `mlfs.optim` (no PyTorch, no `MLFS_REFERENCE`) | all pass, plus your own checks: e.g. `gradcheck` on a small version of the network, or your trained model's predictions matching a NumPy-only forward pass with the same weights |
| **results** | no test number | test accuracy without uncertainty or baselines | $\ge$ 0.97 test accuracy with a 95% interval, against softmax regression and the majority class, with the epoch chosen on the validation set | all of that plus the paired interval for the MLP's advantage, and seed-to-seed variation (several seeds) compared with the bootstrap interval |
| **analysis** | none | describes the curves | explains the learning curves (where validation loss and accuracy part ways, and why) and one ablation measured with an interval | several ablations with intervals and stated predictions, a failure analysis of the misclassified digits, and a profile of where your framework spends its time |
| **clarity** | hard to follow | readable | labelled figures and a narrative answering the five prompts | a reader could reproduce every number (config, seeds, epochs, runtime, machine) from the write-up alone |

## what graders look for

- **the test set used once.** choosing the architecture, the learning rate or the epoch by test accuracy turns the test set into a validation set, and the reported number becomes optimistic. choose on the validation set; report test once.
- **intervals that mean something.** with 10,000 test images and accuracy near 0.977, the 95% interval is about $\pm 1.96\sqrt{0.977 \cdot 0.023/10{,}000} \approx \pm 0.003$. two models 0.002 apart are not distinguishable on this test set; say so. for comparing two models on the *same* test set, use the paired interval, which is narrower because both models get the same easy and hard images (here about ±0.005, against about ±0.006 from combining the two separate intervals as if they were independent).
- **the right baseline.** "97.7%" means little without "softmax regression, the same pipeline without a hidden layer, gets 92.7%": that difference is what the hidden layers buy.
- **validation loss versus accuracy.** the best write-ups notice that validation *loss* starts rising (the network becomes overconfident on the examples it gets wrong) while validation *accuracy* is still flat or improving, and explain why early stopping on one or the other gives different epochs.
- **your framework, not just your model.** a short profile (how long forward, backward and the optimizer step take, and which operation dominates) shows you understand what you built.
