# 21 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | some family doesn't train or sample | all checks pass | all checks pass, and the full run produces every number in the table on real binarized MNIST, with the likelihood families clearly below 1 bit/dim | all of that, with the bounds and exact likelihoods computed on the same test images and the same binarization, no test-set tuning, and a pipeline another person could rerun from the notebook alone |
| **results** | a table without units or uncertainty | numbers without intervals or reference rows | bits/dim with bootstrap intervals, FID from two sample draws, precision, recall and label entropy for every family, the real-data reference row, parameters, training and sampling time | all of that plus a second training seed for at least one family, and a statement of which differences exceed the seed-to-seed noise |
| **analysis** | the numbers are restated | each family's numbers described | rankings on likelihood, quality, diversity and cost, with the mechanism behind each disagreement (mode-covering likelihood vs adversarial training, bounds vs exact values, sequential vs one-pass sampling) | all of that plus a failure for each family that a single metric hides (seen in the samples), and a costed plan for what ten times the compute would change |
| **clarity** | hard to follow | readable | one table, one sample figure, numbers with sensible precision, every metric defined | a reader could reproduce any number (config, seed, device, minutes) and knows exactly which numbers are bounds and which aren't comparable outside the course |

## what graders look for

- **likelihoods on one scale, and honest about bounds.** the PixelCNN's bits/dim is exact; the VAE's (importance-weighted) and the flow's (dequantized) are upper bounds, so a family that "loses" on bits/dim by less than its bound's looseness may not really lose. say which is which.
- **a GAN row with a dash, not a zero.** no likelihood is not a bad likelihood.
- **FID with its noise and its ruler.** two sample draws show the sampling noise; the feature network is the course's own, trained on binarized digits, so these FIDs are comparable within the table and nowhere else.
- **disagreeing rankings explained.** a model can win on likelihood and lose on FID (or the reverse) for principled reasons; the best write-ups predict the disagreement from the chapter's ideas before showing it.
- **cost per sample, not just training time.** sampling the PixelCNN takes one pass per pixel; that's a property of the family, not an implementation detail.
- **no tuning on the test set.** the configs are fixed in advance; if you change one, say why and hold out a validation split from the training digits.
