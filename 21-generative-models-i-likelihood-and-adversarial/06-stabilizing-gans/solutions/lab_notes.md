# stabilizing GANs · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (the lab took 68 s at a load average of about 5, and 4.5–7 minutes when the machine was shared at a load of 15–25). all data come from `mlfs.datasets.eight_gaussians`, which is synthetic, so offline runs are the same experiments, only shorter (a smoke test, not the numbers below).

## experiment 1: what does the critic measure?

**what people usually predict:** all three are 1-Lipschitz, so all three estimate $W_1 \approx 1$; maybe clipping is a bit off.

**what happens** (the 8 gaussians against the same mixture on a circle of half the radius, true $W_1 \approx 1$; 1,000 critic steps each):

| critic | $W_1$ estimate (last 100 steps) | median $\|\nabla f\|$ at interpolates | 10th – 90th percentile |
|---|---|---|---|
| weight clipping ($c = 0.01$) | **0.013** | 0.010 | 0.000 – 0.016 |
| gradient penalty ($\lambda = 10$) | **1.117** | 1.046 | 0.968 – 1.114 |
| spectral norm | **0.420** | 0.377 | 0.300 – 0.501 |

the three "1-Lipschitz" critics disagree by a factor of 86. (an earlier version with 1,500 critic steps gave 0.013, 1.120 and 0.446: the estimates have converged.) only the gradient-penalty critic estimates the distance (12% high); the spectrally normalized critic gets less than half of it, and the clipped critic about 1%.

**why:** the dual estimate is $\int$ (critic slope) along the transport paths, so a critic whose gradient norm is $s$ along them reports about $s \cdot W_1$ (on paper 2), and the measured gradient norms match the estimates almost exactly. clipping bounds every weight by 0.01, and the product of the three layers' norms bounds the slope by about 0.09 (on paper 3); the trained critic doesn't even reach that bound, because clipped weights can't align to maximize the gain. spectral normalization makes each *layer* have norm 1, but a composition's Lipschitz constant is at most the product, and the direction each layer amplifies most isn't the one the next layer passes on, so the network's slope sits around 0.4. the gradient penalty targets the slope directly, at exactly the points that matter; it's a soft constraint, so the slope overshoots 1 slightly (median 1.05) and the estimate lands a bit high. a clipped or SN critic still gives a gradient in a useful direction, scaled by an unknown factor (which Adam's normalization largely absorbs), so both can train GANs; only the GP critic's value can be read as a distance.

## experiment 2: the gradient far from the data

**what people usually predict:** both GAN losses lose their gradient as the samples move away (topic 05's JS argument); the Wasserstein gradient stays at 1.

**what happens** (mean gradient norm per sample, after 200 discriminator/critic steps against samples on a circle of radius $2s$):

| $s$ | distance | mean $D(x)$ on the samples | saturating | non-saturating | Wasserstein |
|---|---|---|---|---|---|
| 1.25 | 0.5 | $5.5 \times 10^{-3}$ | 0.114 | 20.6 | 1.06 |
| 1.5 | 1.0 | $1.1 \times 10^{-3}$ | 0.0146 | 13.0 | 1.11 |
| 2 | 2.0 | $3.3 \times 10^{-4}$ | 0.0025 | 7.9 | 1.17 |
| 3 | 4.0 | $1.4 \times 10^{-4}$ | 0.00055 | 4.2 | 1.30 |
| 4 | 6.0 | $1.4 \times 10^{-4}$ | 0.00036 | 2.7 | 1.40 |

the saturating gradient vanishes, falling 300-fold as the samples move from 0.5 to 6 away. the non-saturating gradient doesn't vanish, but it's **huge near the data (21) and shrinks with distance** (2.7 at distance 6), the opposite of what a generator needs. the Wasserstein critic's gradient stays near 1 everywhere (1.06–1.40).

**why:** the discriminator separates the two sets easily, so $D(x) \approx 10^{-3}$–$10^{-4}$ on the samples. the saturating gradient is $D(x)$ times the logit's slope, and $D(x)$ is tiny. the non-saturating gradient is $(1 - D(x)) \approx 1$ times the logit's slope, and nothing controls that slope: the discriminator only learns to separate, and it makes the logit steepest just outside the data, where the decision is made, and flatter further away. so the non-saturating loss gives a generator *direction* from far away, but a magnitude that depends on the discriminator's arbitrary shape. the Lipschitz critic's slope is pinned near 1 by the penalty: the gradient points towards the data with the same strength at any distance, which is the earth mover's distance's promise. (the slight growth with distance is the penalty's softness: it's enforced on interpolates, and these samples lie further out than most of them.)

## experiment 3: training with the stabilizers

**what people usually predict:** the WGAN-GP covers all modes with good quality and is more reliable across seeds.

**what happens** (WGAN-GP: 1,500 generator steps × 5 critic steps, lr $10^{-3}$; the lab runs seeds 0 and 1. the GAN rows and WGAN-GP seed 2 come from an earlier version of this lab that also trained topic 05's GAN for 5,000 steps, with the same code and seeds):

| method | seed | modes (3 std) | quality (3 std) | modes (10 std) | quality (10 std) |
|---|---|---|---|---|---|
| GAN | 0 | 5 | 0.41 | 8 | 0.93 |
| GAN | 1 | 2 | 0.89 | 2 | 0.91 |
| GAN | 2 | 6 | 0.72 | 6 | 0.96 |
| WGAN-GP | 0 | 5 | 0.11 | **8** | 0.48 |
| WGAN-GP | 1 | **0** | 0.00 | **0** | 0.00 |
| WGAN-GP | 2 (earlier run) | 5 | 0.12 | **8** | 0.61 |

two of three WGAN-GP seeds behave as the theory promises: they cover all 8 modes at the loose radius, with a ring of samples around the circle that is still blurry at the sharp radius (11–12% within 3 std), and their critic's Wasserstein estimate falls steadily from about 1 to 0.2 while the fraction of good samples rises from 0 to 0.5–0.6. the GAN is sharp but collapsed (2 and 6 modes for two seeds). seed 1 of the WGAN-GP **fails completely**: its critic's estimate swings wildly between about +1 and −1.1, and the generator never finds the data.

**why:** the Wasserstein critic gives every sample a gradient towards *all* the data (experiment 2), so the generator spreads out and covers the modes rather than chasing one; the price, at this budget, is precision: a 1-Lipschitz critic has a gentle slope, so pulling samples into a narrow mode (std 0.02) takes many more steps than for a GAN, whose discriminator can be arbitrarily sharp. the falling estimate is the meaningful loss curve topic 05 lacked. but WGAN-GP's guarantees assume the critic is near its optimum; with a learning rate of $10^{-3}$ (ten times the paper's) and only 5 critic steps per generator step, one seed's critic lost track of the moving generator, its estimate went negative (the critic briefly scoring fakes above real data), and the generator was pushed away instead of towards the data. lowering the learning rate to the recommended $10^{-4}$ would make this rarer but needs far more steps than a CPU lab allows. "stabilized" means better behaved, not unbreakable: report seeds, not one run.

## references

- Gulrajani, Ahmed, Arjovsky, Dumoulin & Courville (2017), "Improved training of Wasserstein GANs", *NeurIPS*. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- Miyato, Kataoka, Koyama & Yoshida (2018), "Spectral normalization for generative adversarial networks", *ICLR*. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
- Arjovsky & Bottou (2017), "Towards principled methods for training generative adversarial networks", *ICLR*. [arXiv:1701.04862](https://arxiv.org/abs/1701.04862)
