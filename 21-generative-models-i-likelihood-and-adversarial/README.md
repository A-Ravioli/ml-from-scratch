# 21 · generative models I: likelihood & adversarial

> the four classic families of deep generative models, and how to evaluate them.

a classifier answers "which digit is this?"; a generative model has to know what *all* digits look like, well enough to draw new ones or to say how probable an image is. this chapter builds the four classic families that do it, each with a different answer to the same hard problem (a probability distribution over millions of pixel configurations): **autoregressive** models write the joint as a chain of conditionals and get exact likelihoods but sample one pixel at a time; **variational autoencoders** introduce a compressed latent code and optimize a bound; **GANs** give up likelihood entirely and train a generator against a learned critic; **normalizing flows** build invertible networks with exact likelihoods and one-pass sampling. first, though, you'll build the instruments that grade them, because "the samples look good" is not a measurement: bits per dimension with dequantization, the Fréchet distance and its sample-size bias, precision and recall, and a feature network of your own. on the way you'll find that real data scored against real data gets an FID of 123 at 50 samples, that a fully connected MADE learns digits *better* in a random pixel order than in raster order, that a PixelCNN can never see the pixels up and to its right, that a VQ-VAE's revived codebook gets used and makes reconstructions worse, that the "broken" saturating GAN loss covers more modes than the fixed one, that three "1-Lipschitz" critics disagree about a distance by a factor of 86, and that a likelihood-trained flow never drops a mode but leaks a third of its samples into the gaps.

**you will build:** `bits_per_dim`, `dequantize`, `discrete_nll`, `sqrtm_psd`, `frechet_distance`, `fid_from_features`, `fid_infinity`, `precision_recall_knn`, `FeatureNet`, `train_feature_net`, `cached_feature_net` (topic 01); `MaskedLinear`, `MADE`, `made_nll`, `made_sample`, `MaskedConv2d`, `PixelCNN`, `pixelcnn_nll` (02); `reparameterize`, `kl_standard_normal`, `reconstruction_nll`, `VAE`, `iwae_nll`, `ConvVAE`, `active_units`, `kl_per_dim` (03); `nearest_code`, `VectorQuantizer`, `codebook_usage`, `restart_dead_codes`, `VQVAE` (04); `discriminator_loss`, `generator_loss`, `gan_step`, `DCGANGenerator`, `DCGANDiscriminator`, `mode_coverage` (05); `critic_loss`, `wgan_generator_loss`, `clip_weights_`, `gradient_penalty`, `spectral_normalize`, `SNLinear`, `train_wgan` (06); `standard_normal_log_prob`, `AffineCoupling`, `RealNVP`, `fit_flow`, `logit_transform`, `flow_bits_per_dim` (07). together they make up **`mlfs.generative`**. chapter 22 reuses the evaluation code (FID, bits/dim) and trains latent diffusion on your `ConvVAE`; chapter 25 uses the VAE for representation learning; the audio elective builds a residual quantizer from your `VectorQuantizer`.
**prerequisites:** chapter 16 (convolutions, transposed convolutions, CNN classifiers, ResNets), chapter 15 (`mlfs.torchnn`'s `MLP`, `get_device`, `count_parameters`; Adam, batch norm, training loops), chapter 12 · topic 07 (the ELBO and the reparameterization trick), chapter 11 (EM, probabilistic PCA, k-means), chapter 5 (entropy, KL, maximum likelihood, the bootstrap), chapter 3 · topic 06 (change of variables) and chapter 2 (eigendecompositions, singular values).
**time · scaffolding:** about 100 hours · L4 (spec-driven): each stage's exact specification is in its lesson's **spec** box and the checks enforce it; the exercise notebooks give you the contract signatures and one-line docstrings, and the internal structure (which layers, which attributes, how to vectorize) is yours. one hint per stage. the capstone gives you a brief, the configurations and the checks; the code and the experiment are yours.
**newly unlocked library components:** from chapter 22 on, and inside this chapter from the topic after you build them, you may use library versions of what you've built: `torch.nn.utils.parametrizations.spectral_norm` and `torch.nn.utils.spectral_norm` (06); `torch.distributions` and its `kl_divergence` for Gaussians and Bernoullis (03); `torchmetrics`/`pytorch-fid`-style FID, precision/recall and Inception-score code (01), *with the course's own feature network* (no downloaded Inception weights in this course); masked and coupling flow layers from libraries such as `nflows` or `normflows` (02, 07); and vector-quantization layers such as `vector-quantize-pytorch` (04). inside `mlfs`, later chapters use your code through `mlfs.generative` per [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [evaluating generative models](01-evaluating-generative-models/) | `bits_per_dim`, `dequantize`, `discrete_nll`, `sqrtm_psd`, `frechet_distance`, `fid_from_features`, `fid_infinity`, `precision_recall_knn`, `FeatureNet` + cache → `mlfs.generative` | two halves of the MNIST test set score an FID of 123 at $n = 50$ and still 1.0 at $n = 5{,}000$, and FID∞ still keeps the reference set's own bias (0.54 ± 0.25); a "model" missing three digit classes and a noisy one get nearly the same precision and recall while their FIDs differ fivefold |
| 02 | [autoregressive models](02-autoregressive-models/) | `MaskedLinear`, `MADE`, `made_nll`, `made_sample`, `MaskedConv2d`, `PixelCNN`, `pixelcnn_nll` → `mlfs.generative` | sampling a 28 × 28 digit costs 740 forward passes' worth of time; a MADE learns binarized digits 1.9 nats *better* in a random pixel order than in raster order; 10 pixels above-right of the centre are invisible to a 6-layer PixelCNN |
| 03 | [variational autoencoders](03-variational-autoencoders/) | `reparameterize`, `kl_standard_normal`, `reconstruction_nll`, `VAE`, `iwae_nll`, `ConvVAE`, `active_units`, `kl_per_dim` → `mlfs.generative` | the active-unit count says 31.5 of 32 latents are used while 7 carry most of the KL; β = 1 has the best likelihood (0.224 bits/dim) though β = 0.25 reconstructs far better; samples score an FID 2.6× worse than reconstructions |
| 04 | [discrete latents](04-discrete-latents/) | `nearest_code`, `VectorQuantizer`, `codebook_usage`, `restart_dead_codes`, `VQVAE` → `mlfs.generative` | a 256-code VQ-VAE uses about 35 codes; dead-code restarts revive 250 of them and make reconstructions *worse*; 49 bits per digit are enough to recognize it |
| 05 | [GANs](05-gans/) | `discriminator_loss`, `generator_loss`, `gan_step`, `DCGANGenerator`, `DCGANDiscriminator`, `mode_coverage` → `mlfs.generative` | the GAN reaches all 8 gaussians and loses them again, or collapses onto 2 with the best sample quality; the saturating loss starts with a 70× smaller gradient yet ends covering 7–8 modes on every seed |
| 06 | [stabilizing GANs](06-stabilizing-gans/) | `critic_loss`, `wgan_generator_loss`, `clip_weights_`, `gradient_penalty`, `spectral_normalize`, `SNLinear`, `train_wgan` → `mlfs.generative` | three "1-Lipschitz" critics estimate a distance of 1 as 0.013, 0.42 and 1.12; the non-saturating gradient is largest *next to* the data; WGAN-GP covers every mode on two seeds and fails completely on the third |
| 07 | [normalizing flows](07-normalizing-flows/) | `standard_normal_log_prob`, `AffineCoupling`, `RealNVP`, `fit_flow`, `logit_transform`, `flow_bits_per_dim` → `mlfs.generative` | the flow integrates to 1 at any weights, covers all 8 modes and puts 26–51% of its samples on thin bridges between them; 12 layers beat 8 by less than the seed-to-seed spread |

## capstone

*four families, one dataset*: a PixelCNN, a VAE, a DCGAN and a RealNVP trained on the same binarized MNIST and compared on likelihood (bits/dim with bootstrap intervals; exact, importance-weighted bound, dequantized bound, none), sample quality and diversity (FID from two sample draws, precision and recall, label entropy, all in the feature space of your own `FeatureNet` trained on binarized digits) and cost (parameters, training time, sampling time per image). the reference run (one hour on an Apple M2 GPU) shows the rankings disagreeing: the PixelCNN has the best likelihood (0.165 bits/dim, exact) but the second-worst FID (122), the VAE (0.170, a bound) and the DCGAN tie for the best FID (46–52), the flow trails on everything (0.375 bits/dim, recall 0.21), and the PixelCNN samples hundreds to thousands of times slower than the rest (784 sequential passes per image). small CPU config: under a minute on the offline synthetic stand-in.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (likelihood bookkeeping with dequantization and the logit transform, a VAE step with its importance-weighted bound, and the GAN and Wasserstein objectives), and a 60-minute blind rebuild of the VAE.

## notation

| symbol | meaning |
|---|---|
| $p_\text{data}$, $p_\theta$, $p_g$ | the data distribution, a model's distribution, a generator's distribution |
| $D$ | number of dimensions of an example (pixels × channels); also a discriminator, by context |
| $L$ | number of discrete levels per dimension (256 for 8-bit images, 2 for binary) |
| $u$, $y$ | dequantized data in $[0, 1)^D$; logit-transformed data |
| NLL, bits/dim | $-\log p(x)$ in nats; $\text{NLL}/(D\ln 2)$ |
| $\mu$, $C$; $\hat\mu$, $\hat C$ | mean and covariance of features; their sample estimates |
| FID, $\text{FID}_\infty$ | Fréchet distance between Gaussian fits of features; its $n \to \infty$ extrapolation |
| $m(\cdot)$ | a MADE degree; a binary coupling mask $m \in \{0, 1\}^D$ |
| $z$, $p(z)$, $q_\phi(z \mid x)$ | latent code, prior, approximate posterior (encoder) |
| $\mu$, $\log\sigma^2$, $\epsilon$ | encoder mean and log-variance; standard normal noise of the reparameterization |
| $\beta$ | KL weight ($\beta$-VAE); commitment weight (VQ-VAE) |
| $k$ | number of importance samples; number of nearest neighbours in precision/recall |
| $e_k$, $K$, $\operatorname{sg}$ | codebook vectors, codebook size, stop-gradient |
| $G$, $D$, $\ell$ | generator, discriminator, discriminator logit; $D = \sigma(\ell)$ |
| $f$, $W_1$, $\lambda$ | Wasserstein critic, earth mover's distance, gradient-penalty weight |
| $\sigma(W)$, $u$, $v$ | spectral norm and power-iteration vectors |
| $s(\cdot)$, $t(\cdot)$ | log-scale and shift networks of an affine coupling layer |

## how this connects

- builds on: change of variables (ch 03), eigendecompositions and singular values (ch 02), entropy, KL and maximum likelihood (ch 05), k-means, EM and probabilistic PCA (ch 11), variational inference (ch 12), training deep networks (ch 15), convolutional networks (ch 16), autoregressive sequence models (ch 17).
- used later in: energy-based models, score matching, diffusion and flow matching, all graded with this chapter's FID and bits/dim, and latent diffusion on your `ConvVAE` (ch 22); transformer language models as autoregressive models (ch 18); representation learning with VAEs and discrete tokens (ch 25); optimal transport (electives 15–17); neural audio codecs with residual quantization (audio elective).

## core references

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 20 (deep generative models). [deeplearningbook.org](https://www.deeplearningbook.org)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapters 14–17 (unsupervised learning, GANs, normalizing flows, VAEs). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
- Murphy (2023), *Probabilistic Machine Learning: Advanced Topics*, MIT Press, part IV (generation). [probml.github.io/pml-book/book2](https://probml.github.io/pml-book/book2.html)
- Theis, van den Oord & Bethge (2016), "A note on the evaluation of generative models", *ICLR*. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
- Kingma & Welling (2019), "An introduction to variational autoencoders", *Foundations and Trends in Machine Learning*. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
- Papamakarios et al. (2021), "Normalizing flows for probabilistic modeling and inference", *JMLR* 22. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
