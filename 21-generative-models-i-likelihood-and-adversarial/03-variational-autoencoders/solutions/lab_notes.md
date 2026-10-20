# variational autoencoders · lab notes

numbers are for real MNIST (10,000 training and 1,000 test digits, binarized with a fixed seed), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 20; the lab took 153 s there, and 61 s at a load of about 5; each VAE trained for 10 epochs in about 15 s). FIDs use topic 01's cached `FeatureNet`, so they are comparable only within this chapter. offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end, but its numbers say nothing about MNIST.

## experiment 1: walking through latent space

**what people usually predict:** both walks produce a blend of the two digits; or the latent walk produces garbage in the middle, as an autoencoder would.

**what happens:** the pixel walk is a cross-fade: a 1 superimposed on a fading-in 0, a 3 on an 8, two ghosts at every step. the latent walk shows single, if blurry, digits that morph: the 1 thickens and bends into something like a 5 or 6 before closing into a 0; the 3 passes through a 2-like shape on its way to the 8. two side observations: the VAE's reconstructions are blurrier than the inputs (10 epochs, a 16-dimensional code), and its reconstruction of this particular 4 already looks like a 9, so the 4 → 9 walk barely changes.

**why:** the reconstruction term with a *noisy* code forces nearby codes to decode to similar images (the decoder sees $\mu + \sigma\epsilon$, not $\mu$, so it must be smooth on the scale of $\sigma$), and the KL term packs the codes of all digits into one region around the origin. so the straight line between two codes passes through codes the decoder was trained on, and each decodes to a plausible digit rather than a superposition. the morph goes through *other* digit classes because the line passes through their regions of the code space. the blur is the Bernoulli decoder's mean (a probability map) averaging over the variation the code doesn't capture.

## experiment 2: sweeping β

**what people usually predict:** at $\beta = 1$ the VAE uses most of its 32 dimensions; at $\beta = 16$ almost none. the best reconstructions give the best likelihood.

**what happens** (mean of 2 seeds, ± half their difference):

| β | recon (nats) | KL (nats) | −ELBO | active units | IWAE NLL, k = 100 | bits/dim | FID samples | FID recon |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 94.3 | 35.7 | 129.9 | 32.0 ± 0.0 | 123.3 ± 0.1 | 0.227 | 441 ± 17 | 68 ± 14 |
| 1 | 112.6 | 14.5 | 127.1 | 31.5 ± 0.5 | **122.0** ± 0.9 | **0.224** | **335** ± 0 | 128 ± 21 |
| 4 | 141.0 | 5.4 | 146.4 | 20.0 ± 4.0 | 135.5 ± 0.3 | 0.249 | 373 ± 16 | 261 ± 18 |
| 16 | 197.1 | 0.2 | 197.4 | 12.0 ± 4.0 | 187.0 ± 0.4 | 0.344 | 1171 ± 13 | 1139 ± 1 |

(the "−ELBO" column is recon + KL, the $\beta = 1$ bound, whatever β was used for training.) three surprises. first, the active-unit count says almost every coordinate is in use at $\beta = 1$ (31.5 of 32), but the **KL-per-dimension plot tells a different story**: 7 coordinates carry more than a nat each, and the other 25 carry between 0.01 and 0.25 nats, on their way to collapse but still varying enough to pass the 0.01 variance threshold after 10 epochs. at $\beta = 16$ the total KL is 0.2 nats: the latent space has essentially collapsed, and both FIDs explode. second, the likelihood at $\beta = 1$ (122.0 ± 0.9 nats, 0.224 bits/dim) is **slightly better** than at $\beta = 0.25$ (123.3 ± 0.1), a difference of about 1.5 times the seed spread with only two seeds, even though $\beta = 0.25$ reconstructs far better (94 vs 113 nats) and pays 36 nats of KL for it. third, the samples are much worse than the reconstructions at every $\beta$, and the gap is largest for small $\beta$: at $\beta = 0.25$ reconstructions score the best FID (68) while samples score worse (441) than at $\beta = 1$ (335).

**why:** each coordinate costs its KL in nats per example and pays only through better reconstruction. the first few coordinates (the digit's class, slant, thickness) buy many nats of reconstruction each; later ones buy less than they cost, so the optimum switches them off ($q(z_i \mid x) = \mathcal{N}(0, 1)$, KL 0), and raising $\beta$ raises the price of every nat, so fewer survive. gradient descent approaches that optimum slowly, which is why the variance-threshold count lags behind the KL profile. the likelihood is the sum recon + KL with *equal* weights, so training with $\beta = 1$ optimizes exactly that trade; a smaller β overspends on KL, and its likelihood comes out no better (here slightly worse). samples come from the prior, reconstructions from the encoder's posteriors: when β is small, the codes of real digits spread away from $\mathcal{N}(0, I)$, so prior samples land in regions the decoder never trained on, and sample quality falls while reconstructions improve. likelihood and sample FID agree on $\beta = 1$ here, but that's not guaranteed: they measure different things (topic 01). the seed spread of the FIDs (up to ±21) is a reminder that one run per setting can't separate small differences.

## experiment 3: the shape of the aggregate posterior

**what people usually predict:** the codes fill the prior's disk evenly, because the KL term pulls each code towards $\mathcal{N}(0, I)$.

**what happens:** the codes form class clusters of very different sizes and densities: 1s stretch in a thin arc out to and beyond the $3\sigma$ circle, 0s sit at the right edge, 2s, 4s, 5s, 8s and 9s crowd the centre. 30% of the test codes lie outside the $2\sigma$ circle, where the prior puts only 14%, and the aggregate posterior's covariance is $\begin{pmatrix} 1.40 & -0.56 \\ -0.56 & 1.96 \end{pmatrix}$ instead of the identity. the decoder grid shows the consequences: smooth morphs along cluster boundaries, but also blurry hybrids (in the region between the 1s and the 3s, and in the crowded centre) that look like no digit at all.

**why:** the KL term is applied to each $q(z \mid x)$ separately; it penalizes each posterior for being far from the prior, but nothing asks their *mixture* over the data, the aggregate posterior $q(z) = \frac1N\sum_n q(z \mid x_n)$, to equal $p(z)$. the reconstruction term prefers well-separated codes, so the mixture ends up wider than the prior and lumpy, with low-density gaps between clusters. sampling from the prior then lands in the gaps and in over-represented regions, where the decoder outputs blends, which is exactly why experiment 2's samples score so much worse than reconstructions. fixes include learned priors (a VampPrior, or an autoregressive prior over the codes, as VQ-VAE does in topic 04) and two-stage models that fit a second generative model to the aggregate posterior (latent diffusion in chapter 22 is one).

## references

- Burda, Grosse & Salakhutdinov (2016), "Importance weighted autoencoders", *ICLR*. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
- Hoffman & Johnson (2016), "ELBO surgery: yet another way to carve up the variational evidence lower bound", *NeurIPS workshop on advances in approximate Bayesian inference*. the aggregate posterior and its mismatch with the prior. [approximateinference.org/2016/accepted/HoffmanJohnson2016.pdf](http://approximateinference.org/2016/accepted/HoffmanJohnson2016.pdf)
- Tomczak & Welling (2018), "VAE with a VampPrior", *AISTATS*. [arXiv:1705.07120](https://arxiv.org/abs/1705.07120)
