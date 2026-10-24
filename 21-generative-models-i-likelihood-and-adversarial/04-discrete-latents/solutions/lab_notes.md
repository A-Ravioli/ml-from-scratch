# discrete latents · lab notes

numbers are for real MNIST (10,000 training and 1,000 test digits), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 20; the lab took 180 s there, and 129 s at a load of about 5; about 23–28 s per model; experiment 3 reuses experiment 2's $K = 256$ model). offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end in about 40 s, but its numbers say nothing about MNIST.

## experiment 1: how much of the codebook gets used?

**what people usually predict:** most of the 256 codes, or at least half.

**what happens:** before training, only **5 and 1** of the 256 codes win any argmin on 1,000 training digits (seeds 0 and 1). after 3 epochs, the two seeds use **25 and 45 of the 256 codes** on the test set, with test perplexities of **12.5 and 16.2**: the model behaves like a codebook of about 15 equally used codes. the batch perplexity starts near 3 and climbs slowly through training (it hasn't levelled off after 3 epochs). test MSE: 0.0136 and 0.0103.

**why:** the codebook is initialized in a tiny ball ($\pm 1/256$ per coordinate) and the encoder's outputs start far from it, so the winner for an encoder output $z$ is the code most aligned with $z$'s direction. in 16 dimensions almost every code is extreme in *some* direction (on paper 4: the 2-D convex-hull argument doesn't carry over), but a freshly initialized encoder maps all digits into a narrow cone of directions, so only the one to five codes aligned with that cone win (the count above), and rich get richer: the winners move towards the data and keep winning. only winning codes get codebook-loss gradient and move towards the data; a code that never wins never moves, and stays dead. as the encoder's outputs spread out during training, a few more codes become nearest to something and wake up, which is why the perplexity creeps up, but most of the codebook is never touched. the effective rate is about $49 \log_2 15 \approx 190$ bits per image, not the nominal 392.

## experiment 2: restarting dead codes

**what people usually predict:** with restarts, nearly all codes are used and the reconstruction error drops substantially, since the model has 15 times more effective codes.

**what happens** (test set, mean of 2 seeds, ± half their difference):

| codebook | test MSE | perplexity | codes used |
|---|---|---|---|
| plain | **0.0120** ± 0.0016 | 14.4 ± 1.9 | 35 |
| restarts every 50 steps | 0.0173 ± 0.0006 | **201.1** ± 6.9 | 250 |

restarts do exactly what they're designed to do, raising the perplexity from 14 to 201 and the codes in use from 35 to 250 of 256, and the reconstructions get **worse** (0.0173 vs 0.0120, outside both seed spreads). in follow-up runs, the gap persisted with twice the training (6 epochs: 0.0102 and 0.0087 with restarts vs 0.0080 and 0.0063 without) and when restarts were applied only during the first 150 steps (3 epochs: 0.0181 and 0.0167).

**why:** a restart brings a code back to life, but a live code is only useful once the decoder has learned what to draw for it, and the encoder has learned when to choose it. with 200 codes sharing 10,000 images × 49 positions, each code (and the decoder's response to it) gets a fifteenth of the training signal that each of ~15 codes gets without restarts, so within this budget the many-code model is less well trained everywhere. and the bottleneck of this small VQ-VAE isn't the number of symbols: experiment 3 shows that 16 codes already reconstruct better than 256. revived codes pay off when the model is starved for symbols (bigger, more varied images, a longer budget, a larger decoder), which is the setting where the technique was introduced (Dhariwal et al., 2020). the lesson isn't "restarts are bad" but that **codebook usage is a diagnostic, not a goal**: measure the quantity you care about.

## experiment 3: how many bits is a digit?

**what people usually predict:** $K = 2$ (49 bits) gives unrecognizable blobs; quality improves steadily with $K$.

**what happens** (one seed each, with restarts):

| K | bits per image | test MSE | perplexity |
|---|---|---|---|
| 2 | 49 | 0.0307 | 1.9 |
| 16 | 196 | **0.0149** | 11.1 |
| 256 | 392 | 0.0178 | 208.0 |

**49 bits are enough to recognize every one of the eight test digits**: blurry, with noisy speckle around the strokes, but the 7, 2, 1, 0 and 4s are clear (the 9 is the most damaged). going from 2 to 16 codes halves the error; going from 16 to 256 makes it slightly worse in this budget, consistent with experiment 2.

**why:** the decoder is a shared, learned function that knows what digits look like: the code only has to say *which* stroke goes through each $4 \times 4$ patch, and the decoder fills in a smooth stroke from its knowledge of the data. a $7 \times 7$ binary sketch says a lot about a digit. that's the core idea behind learned compression: the decoder carries most of the information about the *distribution*, the code carries only what distinguishes this image. the returns per bit shrink quickly (the first bits choose the digit's shape, later ones refine it), and a larger codebook only helps if the model and budget can train all of its codes. a VQ-VAE alone can't generate: its index grids need a prior, and an autoregressive model over the 49 indices (topic 02) gives a sampler with an exact likelihood for the codes.

## references

- Dhariwal et al. (2020), "Jukebox: a generative model for music". random restarts of dead codes. [arXiv:2005.00341](https://arxiv.org/abs/2005.00341)
- van den Oord, Vinyals & Kavukcuoglu (2017), "Neural discrete representation learning", *NeurIPS*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
