# latent diffusion & DiT · lab notes

numbers measured with the reference solution on real MNIST on an Apple M2 laptop's GPU (`mps`), shared with other jobs. the default run (experiments 1 and 2 without the baseline) took 1.7 minutes; with both buttons pressed the lab took 5.9 minutes (the U-Net baseline about 2.7 more, experiment 3 about 1.5). offline (tests, CI) the lab runs tiny budgets on the synthetic stand-in as a smoke test, and its numbers mean nothing. FIDs are course-internal (chapter 21's `FeatureNet`, 1,000 samples against 2,000 test digits); 1,000 real training digits score 8.85, the floor at this sample size. one training seed per model.

## experiment 1: what an autoencoder's latents look like

**what people usually predict:** a VAE's latents look like Gaussian noise, with standard deviation about 1.

**what happens** (`ConvVAE`, 4 latent channels, $\beta = 10^{-3}$, 1,500 steps in 33 s): each latent channel is recognizably a blurry $7 \times 7$ version of the digit (the ring of the 0, the strokes of the 4), each with its own offset (one channel mostly negative, another mostly positive) and contrast. the overall latent standard deviation is 0.88 (per channel 0.70–1.00), while the encoder's posterior standard deviation averages 0.27; the KL term is 353 nats per digit, and the reconstructions are nearly perfect (MSE 0.0016 per pixel, reconstruction FID 11.2 against the 8.85 floor).

**why:** with a tiny KL weight the KL term barely constrains the encoder, so the latent space is shaped by reconstruction: a compressed image, not a Gaussian. a large $\beta$ pulls every posterior towards $\mathcal{N}(0, I)$, which makes the latents smoother and more Gaussian but throws away the detail the decoder needs (experiment 3). the posterior noise (0.27) is small next to the spread of the means across digits (0.88), so the means carry almost all the information: encoding them, rather than samples, loses little and spares the latent model from learning the encoder's noise. note that one global standard deviation doesn't make every channel zero-mean and unit-scale; Stable Diffusion's single scale factor has the same limitation.

## experiment 2: latent DiT against pixel U-Net

**what people usually predict:** the latent model is cheaper and roughly as good, close to the reconstruction floor.

**what happens** (800 flow-matching steps of batch 128 each, 20 Euler steps for sampling; the U-Net row comes from the button):

| | parameters | training | ms per step | sampling 1,000 | FID |
|---|---|---|---|---|---|
| latent DiT (+ VAE decoder) | 1,254,532 | 54 s | 67 | 4.2 s | 191.5 |
| pixel U-Net | 390,273 | 156 s | 195 | 4.4 s | **59.9** |
| VAE reconstructions | | | | | 11.2 |
| real training digits | | | | | 8.85 |

the latent DiT trains three times faster per step, with three times the parameters, but its samples are far worse: thin, broken strokes that are mostly not digits, FID 192 against the U-Net's 60 (whose digits are recognizable if rough). the reconstruction floor (11.2) is nowhere near binding: almost all of the latent model's gap is the DiT's own. an earlier run of this lab at 2,000 steps per model gave 95 against 29: both models improve a lot with training, and the U-Net stays far ahead.

**why:** a step of the latent model processes 196 numbers in 49 tokens instead of 784 pixels through a convolutional network, so it is cheap; but at this scale and budget the comparison is not about cost. a U-Net builds in locality and translation equivariance, which digits reward; a small transformer must learn spatial structure from its positional embeddings, and DiTs are known to need long training (Peebles & Xie train for hundreds of thousands of steps). the reconstruction floor only binds once the latent model is good: here it would take a much longer (or larger) DiT to get there. at the scale of a 512 × 512 photograph the balance flips: a pixel-space U-Net would process 786,432 numbers per step against 16,384 latents, the attention cost of a DiT on 1,024 latent tokens is affordable while it would be impossible on pixels, and DiTs scale more predictably with compute. this lab shows the cost structure of latent diffusion, not its quality ceiling.

## experiment 3: the autoencoder's KL weight

**what people usually predict:** a larger KL weight makes smoother latents that are easier to model, so the latent model gets better even if reconstructions get a bit blurrier.

**what happens:**

| VAE β | reconstruction MSE | reconstruction FID | latent std | latent DiT FID |
|---|---|---|---|---|
| 0.001 | 0.0016 | 11.2 | 0.900 | 191.5 |
| 0.1 | 0.0052 | 22.2 | 0.497 | 228.6 |

both the floor and the latent model get worse: reconstruction error triples, the reconstruction FID doubles, and the latent DiT's FID rises from 192 to 229 (at 2,000 DiT steps, an earlier run: 95 to 145).

**why:** the KL weight trades the two error terms: the autoencoder's floor and the latent model's own error. here the latent model's error dominates at both settings, and a blurrier, lower-information latent space didn't make it easier to model enough to compensate for the worse decoder. this is why Rombach et al. keep the KL weight tiny (around $10^{-6}$) and recover sharpness with perceptual and adversarial losses on the autoencoder, rather than using a $\beta = 1$ VAE: the decoder's quality caps everything, and the latent model is better spent on detail-rich latents. one seed per setting, so treat the size of the gap (not just its sign) with some caution.

## references

- Rombach, Blattmann, Lorenz, Esser & Ommer (2022), "High-resolution image synthesis with latent diffusion models", *CVPR*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Peebles & Xie (2023), "Scalable diffusion models with transformers", *ICCV*. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
