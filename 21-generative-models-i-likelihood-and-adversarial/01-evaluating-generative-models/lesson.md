# evaluating generative models

## why this matters

a classifier is easy to grade: hold out labelled data and count mistakes. a generative model has no labels to count. it hands you a pile of new images, or a number called a likelihood, and you have to decide whether it's any good. both obvious answers fail. "the samples look great" is what a model that memorized its training set would earn, and eyes miss a model that draws only three of the ten digits. Theis, van den Oord & Bethge (2016) showed that a model can have an excellent likelihood and terrible samples, or excellent samples and a likelihood of $-\infty$. which number you report changes which model wins.

so before building any generative model, this topic builds the instruments that grade them: **likelihoods** in bits per dimension, made comparable by dequantization; the **Fréchet distance** between feature statistics of real and generated images (FID), and its habit of depending on how many samples you draw; **precision and recall**, which split "are the samples good?" from "are all the modes there?"; and the **feature network** every sample score silently depends on. the standard one (Inception-v3) isn't available offline, so you'll train your own on MNIST and see exactly what that does and doesn't let you claim.

## warm-up: predict first

1. a model assigns every 8-bit grayscale image the same probability. how many bits per pixel does it cost?
2. you fit a continuous density to integer pixel values. what stops it from putting a spike of infinite height on each integer?
3. you compute FID between two disjoint halves of the MNIST test set, 100 images each. is the answer 0?
4. a GAN draws perfect 1s and nothing else. which of precision and recall notices?
5. FID numbers in papers are computed with Inception features. could you compare your MNIST FIDs with them?

## prerequisites check

- [ ] entropy, cross-entropy and KL divergence, in nats and in bits (→ ch 05 · topic 05)
- [ ] the multivariate Gaussian, its mean and covariance, and fitting them to data (→ ch 04 · topic 04)
- [ ] symmetric eigendecompositions $A = V \Lambda V^\top$ (→ ch 02 · topic 05)
- [ ] the change-of-variables rule for densities (→ ch 03 · topic 06)
- [ ] training a small CNN classifier (→ ch 16 · topic 03) and the bootstrap (→ ch 05 · topic 03)

## core ideas

### idea 1: likelihood, in bits per dimension

the cleanest grade a generative model can get is its **likelihood** on held-out data: how much probability $p_\theta$ puts on test images it never trained on. averaged as a negative log, it's the cross-entropy between the data distribution and the model:

$$\mathbb{E}_{x \sim p_\text{data}}[-\log p_\theta(x)] = H(p_\text{data}) + \operatorname{KL}(p_\text{data} \,\|\, p_\theta).$$

the entropy $H(p_\text{data})$ is fixed by the data, so lower test NLL means smaller KL, closer to the truth. it's also a compression rate: a model with NLL $\ell$ nats can code test images in about $\ell / \ln 2$ bits each.

raw NLLs depend on the image size, so the field reports **bits per dimension**: the NLL in nats per example, divided by the number of dimensions $D$ (pixels × channels) and by $\ln 2$:

$$\text{bits/dim} = \frac{\text{NLL}}{D \ln 2}.$$

a model that knows nothing (uniform over 256 levels per pixel) scores exactly 8 bits/dim. a good PixelCNN scores about 3 on CIFAR-10; on 8-bit MNIST, good models get about 1.

that only works for models that assign probabilities to **discrete** values. a model with a continuous density (a normalizing flow, topic 07, or a VAE with a Gaussian decoder) assigns a density $p(y)$ to real-valued $y$, and a density on integer data can cheat: put a narrow spike at every integer and the density at the data goes to infinity while every image stays possible. the fix is **uniform dequantization** (Uria, Murray & Larochelle, 2013; Theis et al., 2016). add independent uniform noise $u \in [0, 1)^D$ to the integer pixels $x$ and scale by the number of levels $L$:

$$y = \frac{x + u}{L} \in [0, 1)^D.$$

each discrete image $x$ now owns a little box $B_x$ of volume $L^{-D}$, the boxes tile $[0, 1)^D$, and the model's density defines a proper discrete distribution, $P(x) = \int_{B_x} p(y)\,dy$. by Jensen's inequality, the dequantized NLL upper-bounds the discrete one:

$$-\log P(x) \le \mathbb{E}_{u}\big[-\log p(y)\big] + D \log L.$$

the $D \log L$ term is the log of the box volume, put back because you rescaled to $[0, 1)$: a uniform density on $[0, 1)^D$ has $\log p = 0$ everywhere, and the bound gives it $D \log 256$ nats, 8 bits/dim, exactly what the discrete uniform model costs. with this correction, continuous and discrete models are graded on the same scale, and a continuous model can only look worse than it is, never better. a spike can no longer help: the box around each integer has a fixed width.

> **spec · stage 1.**
> - `bits_per_dim(nll, n_dims)`: `nll / (n_dims ln 2)`, elementwise, for a float, an ndarray or a tensor (returning the same type). this is `mlfs.generative.bits_per_dim`.
> - `dequantize(x, *, n_levels=256, generator=None)`: `x` a tensor of integer levels in `0..n_levels-1` (any dtype) → float32 `(x + u) / n_levels` with `u ~ U[0, 1)` drawn from `generator`, same shape.
> - `discrete_nll(log_density, n_dims, *, n_levels=256)`: the bound $-\log p(y) + D \log L$ in nats, for the log-density of dequantized data scaled to $[0, 1)^D$ (float or tensor). (a model of data scaled to $[-1, 1)^D$ has a density $2^{-D}$ times smaller per unit volume, so add $D \ln 2$ to its log-density first (if $y' = 2y - 1$, $\log p_{[0,1)}(y) = \log p_{[-1,1)}(y') + D \ln 2$).)

> **▶ build: stage 1.** implement `bits_per_dim`, `dequantize` and `discrete_nll` in `exercise.py` (`marimo edit exercise.py`).

> **🔮 predict:** a flow trained on MNIST reports an average log-density of $+3{,}100$ nats per image on dequantized images in $[0, 1)^{784}$. a *positive* log-likelihood: is this model better or worse than one that assigns every image the same probability?

### idea 2: the Fréchet distance between two Gaussians

likelihood needs a model that can compute it, and GANs can't (topic 05). a sample-based score needs a distance between two *sets of images*. comparing pixels directly is hopeless (a digit shifted by one pixel is "far"), so you compare them in the **feature space** of a trained network, where distances mean something. then a cheap summary: fit a Gaussian to each set's features and measure the distance between the two Gaussians.

the distance of choice is the **Fréchet** (or 2-Wasserstein) distance: the smallest expected squared distance $\mathbb{E}\|a - b\|^2$ over all ways of pairing samples $a$ from one distribution with samples $b$ from the other. between Gaussians $\mathcal{N}(\mu_1, C_1)$ and $\mathcal{N}(\mu_2, C_2)$ it has a closed form (Dowson & Landau, 1982):

$$d^2 = \|\mu_1 - \mu_2\|^2 + \operatorname{Tr}\Big(C_1 + C_2 - 2\,(C_1 C_2)^{1/2}\Big).$$

the first term compares centres; the trace term compares shapes. in one dimension it's $(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$, a distance between (mean, standard deviation) pairs.

the awkward part is $(C_1 C_2)^{1/2}$. the product of two symmetric matrices isn't symmetric, and a general matrix square root (`scipy.linalg.sqrtm`, a Schur decomposition) can return complex numbers from round-off. you only need its **trace**, and there's a symmetric route: $C_1 C_2$ is similar to $M = C_1^{1/2} C_2 C_1^{1/2}$ (on paper 2), which is symmetric positive semidefinite, so

$$\operatorname{Tr}(C_1 C_2)^{1/2} = \operatorname{Tr} M^{1/2} = \sum_i \sqrt{\lambda_i(M)}.$$

two symmetric eigendecompositions and no complex arithmetic. round-off can still make tiny eigenvalues negative (clip them to 0) and the final answer slightly negative (clip it to 0 too).

> **spec · stage 2.**
> - `sqrtm_psd(A)`: the symmetric PSD square root of a symmetric PSD matrix, from `np.linalg.eigh`, with negative eigenvalues clipped to 0.
> - `frechet_distance(mu1, cov1, mu2, cov2)`: NumPy inputs → a python `float` $\ge 0$, computed with symmetric square roots as above (it must agree with the `scipy.linalg.sqrtm` formula on well-conditioned inputs, and be symmetric in its arguments). this is `mlfs.generative.frechet_distance`.

> **▶ build: stage 2.** implement `sqrtm_psd` and `frechet_distance`.

### idea 3: FID, and why it depends on how many samples you draw

the **Fréchet Inception Distance** (Heusel et al., 2017) is idea 2 applied to features: embed $n_1$ real and $n_2$ generated images with a fixed network, estimate each set's mean and covariance (the unbiased covariance, `np.cov`, as the reference implementation does), and compute the Fréchet distance. lower is better, and 0 means the two feature distributions have the same mean and covariance.

FID is computed from *estimates*, and that makes it **biased**. take two independent samples of size $n$ from the *same* distribution. their sample means differ by noise with $\mathbb{E}\|\hat\mu_1 - \hat\mu_2\|^2 = 2\operatorname{Tr}(C)/n$ (on paper 3), and the covariance estimates differ too. so the expected FID between two samples of the same distribution isn't 0, it's positive and shrinks roughly like $1/n$. with 128-dimensional features and $n = 100$ the covariance estimate isn't even full rank. the bias also depends on the generator, so two models compared at a small $n$ can come out in a different order than at a large one (Chong & Forsyth, 2020).

two habits follow. always report $n$ and compare models only at the same $n$. and if you want a number that doesn't depend on $n$, extrapolate: compute FID on generated subsets of several sizes, fit $\text{FID}(n) = a + b/n$ by least squares, and report the intercept $a$, an estimate of FID at infinitely many *generated* samples, called $\text{FID}_\infty$ (Chong & Forsyth, 2020). it keeps the bias of the fixed reference set (its statistics are estimated too), and the $1/n$ law only holds once $n$ is well above the feature dimension, so fit it on large $n$.

> **spec · stage 3.**
> - `fid_from_features(feats1, feats2)`: features `(n1, d)` and `(n2, d)` (NumPy) → the Fréchet distance of their Gaussian fits (mean, and `np.cov(..., rowvar=False)` with `ddof=1`). this is `mlfs.generative.fid_from_features`.
> - `fid_infinity(real_feats, fake_feats, ns, *, rng=None)`: with the real statistics from the full real set, for each `n` in `ns` draw `n` fake rows **without replacement** (`np.random.default_rng(rng).choice`, one draw per `n`, in order) and compute the FID; fit `fids ≈ a + b / n` by least squares; return `(a, b, fids)`, `fids` an ndarray of shape `(len(ns),)`.

> **▶ build: stage 3.** implement `fid_from_features` and `fid_infinity`.

> **🔮 predict:** you embed the 10,000 MNIST test images and compute FID between two disjoint random halves of size $n$. what is it at $n = 50$, and at $n = 5{,}000$? the two halves come from exactly the same distribution.

### idea 4: precision and recall

one number can't tell you *how* a model fails. a model that draws flawless 1s and nothing else, and a model that draws every digit but smudged, can have the same FID. Sajjadi et al. (2018) proposed to measure two things separately, by analogy with classification:

- **precision**: what fraction of generated samples look like real data? (fidelity)
- **recall**: what fraction of the real data could the model have produced? (coverage, diversity)

Kynkäänniemi et al. (2019) made this concrete with $k$-nearest-neighbour balls in feature space. around every real feature vector, draw a ball whose radius is the distance to its $k$-th nearest *other* real point. the union of these balls is an estimate of the "real manifold". **precision** is the fraction of generated points that fall inside at least one real ball. **recall** swaps the roles: build balls around the generated points (radius: distance to the $k$-th nearest other generated point) and count the fraction of real points inside at least one. a collapsed model scores high precision and low recall; a blurry, over-dispersed one scores the reverse.

the radii adapt to the local density. that's also the weakness: one outlier among the real points gets a huge ball that accepts nearly anything nearby, so precision is sensitive to outliers and to $k$ ($k = 3$ is usual).

> **spec · stage 4.** `precision_recall_knn(real_feats, fake_feats, *, k=3)`: NumPy features → `(precision, recall)`, two floats. each point's radius is the Euclidean distance to its $k$-th nearest neighbour **in its own set, excluding itself**; a point is inside a ball if its distance to the centre is $\le$ the radius (inclusive). this is `mlfs.generative.precision_recall_knn`. vectorize the distances (an `(n, m)` matrix is fine for a few thousand points).

> **▶ build: stage 4.** implement `precision_recall_knn`.

### idea 5: the feature network, and what your numbers mean

every sample-based score above lives in someone's feature space. the papers' FID uses the 2,048-dimensional pool features of Inception-v3 trained on ImageNet: a published network, so FIDs from different papers are (roughly) comparable. this course runs offline and can't download those weights. so you'll train your own feature network, a small CNN classifier on MNIST, and use its penultimate layer.

that's a legitimate choice, but be precise about what it buys you. a network trained to classify digits is sensitive to what makes a digit a digit and may be blind to stroke texture, background noise or faint artifacts. and the scale of the numbers is set by *your* network: an "FID of 5" in this course means nothing next to an Inception FID of 5 in a paper. **course FIDs are comparable with each other, computed with the same trained feature network, and with nothing else.** the same goes for precision and recall. what does transfer is everything about the *method*: the bias, the extrapolation, the separation of fidelity and coverage.

to keep course numbers comparable across notebooks, the network must be the same network every time. `cached_feature_net` trains it once (deterministically, from a seed) and saves its weights under the course data cache, with a file name that hashes the data, the settings and the code, so any change retrains rather than silently reusing stale weights.

> **spec · stage 5.**
> - `FeatureNet(in_channels=1, n_classes=10, *, feat_dim=128)`, an `nn.Module`: convolutions with downsampling, then **global average pooling** (so any image size works), a `feat_dim`-unit layer with a ReLU, and a linear head `self.fc`. `forward_features(x)`: `(B, C, H, W)` → `(B, feat_dim)`; `forward(x) = fc(forward_features(x))`; `embed(x, *, batch_size=500)`: features of a whole tensor or ndarray (cast to float32) as a float64 ndarray `(n, feat_dim)`, in eval mode under `no_grad`, in batches, restoring the train/eval mode it found.
> - `train_feature_net(X, y, *, epochs=2, batch_size=128, lr=1e-3, seed=0, device="cpu")`: Adam on cross-entropy, `torch.manual_seed(seed)` before building the network and a `torch.Generator` seeded with `seed` for shuffling; the number of classes is `y.max() + 1`; returns the network in eval mode.
> - `cached_feature_net(X, y, *, name="mnist", epochs=2, seed=0, device="cpu", cache=None)`: load or train-and-save a `state_dict` in `cache` (default `mlfs.datasets.cache_dir()`), in a file named `ch21-featnet-<name>-<hash>.pt` where the hash covers the data (`X` and `y`), `epochs`, `seed`, `device` and the source code of `FeatureNet` and `train_feature_net`; a new file is exactly `train_feature_net(X, y, epochs=epochs, seed=seed, device=device)`; the net comes back in eval mode either way.

> **▶ build: stage 5.** implement `FeatureNet`, `train_feature_net` and `cached_feature_net`.

## worked example

**bits per dimension.** a binarized-MNIST model with a test NLL of 85 nats per image ($D = 784$): $85 / (784 \ln 2) = 0.156$ bits/dim. a flow on 8-bit MNIST reports a mean log-density of $\log p(y) = 3{,}100$ nats on dequantized images in $[0, 1)^{784}$: the discrete bound is $-3{,}100 + 784 \ln 256 = 1{,}247.4$ nats, so $1{,}247.4 / (784 \ln 2) = 2.30$ bits/dim. a uniform density ($\log p = 0$) gives $784 \ln 256$ nats, exactly 8 bits/dim.

**the dequantization bound.** a 1-D density $p(y) = 2y$ on $[0, 1)$ models data with $L = 4$ levels. level 0 owns $[0, 0.25)$ and has exact probability $\int_0^{1/4} 2y\,dy = 1/16$: an NLL of $\ln 16 = 2.773$ nats. the dequantized bound averages $-\log(2(0 + u)/4) + \log 4$ over $u \in [0, 1)$: $-\ln 2 + 2 \ln 4 - \int_0^1 \ln t\,dt = -0.693 + 2.773 + 1 = 3.079$ nats, which is larger, as Jensen promised. for level 3 (exact probability $7/16$) the gap is much smaller: $0.831$ vs $0.827$ nats: the bound is tight where the density is nearly flat across the box.

**the Fréchet distance in 1-D.** $\mathcal{N}(1, 4)$ against $\mathcal{N}(-1, 1)$: $(1 - (-1))^2 + (2 - 1)^2 = 5$. with diagonal covariances $\operatorname{diag}(1, 4, 0)$ and $\operatorname{diag}(9, 0, 0)$, the matrices commute, $(C_1C_2)^{1/2} = \operatorname{diag}(3, 0, 0)$, and $d^2 = (1 - 3)^2 + (2 - 0)^2 + 0 = 8$.

**precision and recall by hand** ($k = 1$, 1-D features). real points $\{0, 1, 2, 3, 10\}$ have radii $1, 1, 1, 1, 7$. generated points $\{0.5, -1, 6, 20, 21\}$: $0.5$ is inside 0's ball, $-1$ lies exactly on 0's ball (distance 1, inclusive), $6$ is inside 10's ball, and 20 and 21 are in none: precision $3/5$. the generated radii are $1.5, 1.5, 5.5, 1, 1$; the real points 0, 1 and 2 (exactly on the boundary) lie in $0.5$'s ball, and 3 and 10 in 6's large ball: recall $1$, even though no generated point is anywhere near 3 or 10. one isolated sample with a big radius covers a lot. the checks reproduce all four examples.

## on paper

1. **the dequantization bound.** show that $\int_{B_x} p(y)\,dy$ defines a probability distribution over discrete images, and prove $-\log P(x) \le \mathbb{E}_u[-\log p((x+u)/L)] + D \log L$.

<details><summary>solution</summary>

the boxes $B_x = \prod_d [x_d/L, (x_d + 1)/L)$ for $x \in \{0, \dots, L-1\}^D$ are disjoint and cover $[0, 1)^D$, so $\sum_x P(x) = \int_{[0,1)^D} p = 1$, with each $P(x) \ge 0$. the change of variables $y = (x + u)/L$ has Jacobian $L^{-D}$, so $P(x) = \int_{[0,1)^D} p((x + u)/L)\, L^{-D}\, du = L^{-D}\,\mathbb{E}_u[p((x+u)/L)]$ with $u$ uniform. then $-\log P(x) = D \log L - \log \mathbb{E}_u[p] \le D \log L + \mathbb{E}_u[-\log p]$, by Jensen's inequality for the convex function $-\log$. equality holds when $p$ is constant on the box.

</details>

2. **why the symmetric route gives the same trace.** show that $C_1 C_2$ and $M = C_1^{1/2} C_2 C_1^{1/2}$ have the same eigenvalues when $C_1$ is positive definite, and conclude $\operatorname{Tr}(C_1C_2)^{1/2} = \sum_i \sqrt{\lambda_i(M)}$.

<details><summary>solution</summary>

$C_1 C_2 = C_1^{1/2} (C_1^{1/2} C_2 C_1^{1/2}) C_1^{-1/2} = C_1^{1/2} M C_1^{-1/2}$: similar matrices, same eigenvalues. $M$ is symmetric PSD, so its eigenvalues are real and $\ge 0$, and $C_1 C_2$ is diagonalizable with the same nonnegative eigenvalues. its principal square root has eigenvalues $\sqrt{\lambda_i}$, and the trace is their sum. (for singular $C_1$, take a limit $C_1 + \epsilon I$; the eigenvalues are continuous.)

</details>

3. **the bias of the mean term.** two independent samples of size $n$ from a distribution with covariance $C$ have means $\hat\mu_1$, $\hat\mu_2$. show $\mathbb{E}\|\hat\mu_1 - \hat\mu_2\|^2 = 2\operatorname{Tr}(C)/n$. with $C = I$ in 128 dimensions, what does the mean term alone contribute to FID at $n = 100$ and $n = 10{,}000$?

<details><summary>solution</summary>

$\hat\mu_1 - \hat\mu_2$ has mean 0 and covariance $C/n + C/n = 2C/n$, and $\mathbb{E}\|z\|^2 = \operatorname{Tr}\operatorname{Cov}(z)$ for a zero-mean $z$. with $C = I_{128}$: $256/100 = 2.56$ at $n = 100$ and $0.0256$ at $n = 10{,}000$. the covariance term adds a bias of its own, which grows like $d^2/n$ and dominates in high dimensions.

</details>

## experiment

`lab.py` computes FID and precision/recall in the feature space of your `FeatureNet` trained on MNIST:

1. FID between two halves of the *real* test set, as the number of samples grows, and what $\text{FID}_\infty$ makes of it.
2. two fake "models" built from real images, one with dropped digits and one with added noise: which one does FID prefer, and does the answer depend on $n$?
3. precision and recall for the same two models.

## check your understanding

1. why is test NLL a fair comparison between two models with different architectures, as long as both compute exact (or upper-bounded) likelihoods of the same discrete data?

<details><summary>answer</summary>

$\mathbb{E}[-\log p_\theta] = H(p_\text{data}) + \operatorname{KL}(p_\text{data} \| p_\theta)$, and the entropy is the same for both models, so the difference in NLL is a difference in KL to the true distribution. an upper bound (dequantized flows, a VAE's ELBO) can only make a model look worse, so if a bounded model wins, it really wins.

</details>

2. what goes wrong if you report a continuous model's log-density on dequantized data scaled to $[0, 1)$ without the $D \log 256$ term?

<details><summary>answer</summary>

the density on $[0, 1)^D$ is $256^D$ times larger than the density on $[0, 256)^D$, so the "NLL" is $D \ln 256$ nats too small: 8 bits/dim too good. the model would appear to beat every discrete model, and could even report negative bits/dim.

</details>

3. why compute $\operatorname{Tr}(C_1C_2)^{1/2}$ through $C_1^{1/2}C_2C_1^{1/2}$ instead of `scipy.linalg.sqrtm(C1 @ C2)`?

<details><summary>answer</summary>

$C_1C_2$ isn't symmetric, so a general square root (Schur-based) is needed, which is slower and can return small imaginary parts from round-off, or fail on singular matrices. $C_1^{1/2}C_2C_1^{1/2}$ is symmetric PSD with the same eigenvalues: two `eigh` calls, real arithmetic, and clipping handles the round-off.

</details>

4. you compare two GANs: one evaluated on 1,000 samples (FID 12.1), the other on 10,000 (FID 11.4). which is better?

<details><summary>answer</summary>

you can't tell. FID is biased upward by an amount that shrinks with $n$, and the difference here is within what the sample size alone could cause. recompute both at the same $n$, or report $\text{FID}_\infty$.

</details>

5. a model's samples have precision 0.98 and recall 0.15. what does that suggest, and would FID alone have told you?

<details><summary>answer</summary>

almost every sample looks real, but the model covers only a small part of the real distribution: mode collapse (topic 05). FID would be high, but the same FID could come from blurry samples covering everything; precision and recall separate the two failure modes.

</details>

## common bugs

- **forgetting $\ln 2$ or $D$.** symptom: bits/dim around 5.5 instead of 8 for a uniform model (you used nats), or a number in the hundreds (you didn't divide by $D$).
- **dequantizing with a constant offset.** symptom: $x + 0.5$ instead of $x + u$; a continuous model learns spikes at the bin centres and its "likelihood" improves without bound.
- **`sqrtm(C1) @ sqrtm(C2)` for $(C_1C_2)^{1/2}$.** symptom: correct only when the covariances commute (diagonal tests pass, random ones fail).
- **FIDs at different $n$.** symptom: a "better" model that is only better because it was evaluated on more samples.
- **counting a point as its own nearest neighbour.** symptom: every radius is 0 at $k = 1$, so precision and recall collapse to the fraction of exact duplicates.

## going deeper

- Theis, van den Oord & Bethge (2016), "A note on the evaluation of generative models", *ICLR*. likelihood vs samples, dequantization. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
- Heusel, Ramsauer, Unterthiner, Nessler & Hochreiter (2017), "GANs trained by a two time-scale update rule converge to a local Nash equilibrium", *NeurIPS*. introduces FID. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
- Chong & Forsyth (2020), "Effectively unbiased FID and Inception Score and where to find them", *CVPR*. the bias and $\text{FID}_\infty$. [arXiv:1911.07023](https://arxiv.org/abs/1911.07023)
- Sajjadi, Bachem, Lucic, Bousquet & Gelly (2018), "Assessing generative models via precision and recall", *NeurIPS*. [arXiv:1806.00035](https://arxiv.org/abs/1806.00035)
- Kynkäänniemi, Karras, Laine, Lehtinen & Aila (2019), "Improved precision and recall metric for assessing generative models", *NeurIPS*. the $k$-NN version built here. [arXiv:1904.06991](https://arxiv.org/abs/1904.06991)
- Dowson & Landau (1982), "The Fréchet distance between multivariate normal distributions", *Journal of Multivariate Analysis* 12(3). [doi:10.1016/0047-259X(82)90077-X](https://doi.org/10.1016/0047-259X(82)90077-X)

## connections

- ← builds on: entropy and KL (ch 05 · topic 05), the multivariate Gaussian (ch 04 · topic 04), eigendecompositions (ch 02 · topic 05), change of variables (ch 03 · topic 06), CNN classifiers (ch 16 · topic 03).
- → used in: the exact likelihoods of autoregressive models (topic 02) and flows (topic 07), the VAE's bound (topic 03), mode collapse measured on GANs (topics 05–06), the chapter capstone (four families graded with all of these), diffusion and flow-matching models (ch 22), and every later generative experiment in the course.
