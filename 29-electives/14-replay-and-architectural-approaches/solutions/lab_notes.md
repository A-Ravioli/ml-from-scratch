# replay & architectural approaches · lab notes

numbers measured with the reference solution on the cached real MNIST, the block's 784–200–200–10 MLP, Adam at $10^{-3}$, batch 64, one epoch per task, 64 replayed examples appended to each batch. the lab takes about two minutes on a laptop CPU. uncertainties are standard deviations over seeds. `verify.py` runs the lab offline on synthetic stand-in data, which only tests the code path.

## experiment 1: how much memory does replay need?

**what people usually predict:** that 10 images do nothing, 200 help somewhat (0.3–0.5), and 1,000 approach joint training.

**what happens** (class-incremental split MNIST, seeds 0–2):

| memory | floats | ACC |
|---|---|---|
| none (fine-tuning) | 0 | 0.196 ± 0.000 |
| reservoir, M = 10 | 7,840 | 0.296 ± 0.004 |
| reservoir, M = 50 | 39,200 | 0.569 ± 0.012 |
| reservoir, M = 200 | 156,800 | 0.811 ± 0.017 |
| reservoir, M = 1,000 | 784,000 | 0.898 ± 0.009 |

ten stored images, about one per digit, add ten points; fifty almost triple fine-tuning's accuracy; 200 images (0.3% of the training set) reach 0.81, four times fine-tuning and far beyond anything EWC or SI managed in this scenario. the curve then flattens: 1,000 images still leave 6 points to joint training (0.96). a class-balanced buffer gives the same numbers on this benchmark (at $M = 200$: 0.809 and 0.793 for seeds 0 and 1, against the reservoir's 0.794 and 0.811), because split MNIST's classes arrive in equal amounts and a reservoir is already balanced in expectation.

**why:** the class-incremental failure is at the output layer: without old labels in the batch, old classes' logits are driven down on every step (the lesson's on paper 4). replay puts old labels back into every batch, so their logits are pulled up again, and even a few examples per class are enough to keep the output biases in balance. the network fits ten stored images in a handful of steps, so the information they carry is mostly "these classes still exist, and live roughly here"; more images add the variety that keeps the old classes' *decision boundaries* accurate, which is why returns diminish. the remaining gap to joint training comes from overfitting the small memory (each stored image is replayed hundreds of times) and from the imbalance between a whole new task's data and a few hundred old examples.

## experiment 2: a model of the data instead of the data

**what people usually predict:** that the static-covered samples are useless or worse than 20 real images; a few predict they match 20 images.

**what happens** (seeds 0–2):

| memory | floats | ACC |
|---|---|---|
| reservoir, 20 images | 15,680 | 0.412 ± 0.026 |
| Gaussian per class | 15,680 | **0.786 ± 0.019** |
| reservoir, 200 images | 156,800 | 0.811 ± 0.017 |

the Gaussian generator, with the memory of 20 images, nearly matches a buffer of 200 real images (0.79 vs 0.81) and beats a buffer of the same size by 37 points. its samples (the image grid) are the class average with independent per-pixel noise: recognizable shapes in static, nothing like real digits.

**why:** two reasons. first, replay's main job in this scenario is the output-layer balance, and for that a replayed example only needs to carry an old label and land in roughly the right region of input space: a class mean plus noise does that. second, the generator never repeats itself: every replayed batch is a fresh draw, so the network can't memorize 20 particular images and overfit them; it sees an endless supply of variations around each class mean, which acts like data augmentation centred on the class. 20 real images carry more realistic detail but are replayed hundreds of times each. the cheap generator would break down where a class isn't a blob around its mean: natural images (a CIFAR-10 "cat" averaged over poses and backgrounds is brown fog), classes with several modes, or convolutional networks, which can easily tell static from real texture and may learn to classify the noise instead of the shape. that's what full generative replay, with VAEs or GANs trained continually, is for.

## experiment 3: isolating parameters instead of remembering data

**what people usually predict:** that gating helps, more with smaller `keep`, or that keeping only 20% of the units hurts each task's accuracy.

**what happens** (task-incremental split MNIST, seeds 0–2):

| keep | hidden units per task | ACC | BWT |
|---|---|---|---|
| 1 (fine-tuning) | 200 of 200 | 0.917 ± 0.024 | −0.096 ± 0.030 |
| 0.5 | 100 of 200 | 0.977 ± 0.011 | −0.018 ± 0.013 |
| 0.2 | 40 of 200 | 0.974 ± 0.012 | −0.020 ± 0.015 |

gating removes most of the forgetting (BWT from −0.10 to −0.02), with no stored data and no extra loss term, and 40 units per layer are as good as 100.

**why:** units gated off for a task get no gradient from it, so each task overwrites only its own units' incoming weights. two tasks still share units: on average $k^2/h$ of each task's $k$ (on paper 3), 50 of 100 at `keep = 0.5` and 8 of 40 at `keep = 0.2`. even sharing half its units, a task's damage to another is spread over far fewer of the other's weights, and the output layer is protected by the multi-head loss, so most of the forgetting goes. each split-MNIST task is easy enough for 40 units per layer, so the capacity cost of `keep = 0.2` doesn't show; with harder tasks or more of them, small `keep` runs out of room and large `keep` overlaps more, which is why Masse et al. pair gating with SI. none of this applies class-incrementally: `forward` needs the task.

## project: beat fine-tuning with a fixed memory budget

**what people usually predict:** that the starting strategy, a 50-image reservoir filling the whole budget, is hard to beat, at 0.5–0.6.

**what happens.** the notebook's default `my_strategy` (reservoir, 50 images, 39,200 floats), seeds 0–4:

| strategy | floats | ACC per seed | mean ± sd |
|---|---|---|---|
| fine-tuning | 0 | 0.196, 0.195, 0.196, 0.196, 0.196 | 0.196 ± 0.000 |
| reservoir, 50 images (default) | 39,200 | 0.561, 0.563, 0.583, 0.556, 0.564 | 0.565 ± 0.010 |
| Gaussian generator, 64 per step | 15,680 | 0.791, 0.802, 0.765, 0.780, 0.760 | 0.780 ± 0.018 |

the default improves on fine-tuning by +0.370 (95% paired bootstrap CI [+0.363, +0.379]); the Gaussian generator, the strategy experiment 2 points to, by +0.584 ([+0.570, +0.598]) with 40% of the budget. other strategies within the budget, seeds 0–2: the generator plus a 30-image reservoir (15,680 + 23,520 = 39,200 floats, 32 replayed from each) 0.795 ± 0.007; the generator with 128 samples per step 0.796 ± 0.006. EWC and SI can't enter: online EWC stores 398,420 floats for this network.

**why:** the budget rewards memories that generalize per float, and a per-class Gaussian is ten times cheaper in floats than the 200-image buffer it nearly matches (experiment 2). spending the leftover budget on real images adds about a point, within the seed noise; so does replaying more samples. caveats a good write-up states:

- **the unit decides the winner.** MNIST pixels are uint8. a budget of 156,800 *bytes* (the same 39,200 float32s) holds 200 uint8 images, which reach 0.811 ± 0.017 (experiment 1), a little above the generator's 0.78–0.79. in bytes the generator (62,720) is about 2.5× cheaper than that buffer, not 10×.
- **`n_floats()` is self-reported**: the budget check trusts each memory class's own accounting.
- **replay doubles the compute** of every step (64 replayed examples per 64 new ones), which the memory budget doesn't count.
- **five paired seeds** make the percentile bootstrap interval optimistic (it under-covers at this sample size). seed-to-seed standard deviations are 0.01–0.02, so differences under about 0.02–0.03 between good strategies aren't resolvable; "replay beats fine-tuning by 0.4–0.6" is solid, the fine ranking of good strategies needs more seeds.

a learner who wants to go further can try a memory of class means only (7,840 floats), a low-rank Gaussian per class, uint8 storage, or hidden-layer activations instead of pixels.
