# diffusion (DDPM) · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU (one thread, as in CI) while the machine was heavily shared; the whole lab took about 110 s there. experiments 2 and 3 use synthetic data, so offline runs are the same experiments; experiment 1's schedule numbers don't depend on the digit (offline it shows a synthetic blob instead of a real digit). one training seed per configuration; the remarks on noise say where that matters.

## experiment 1: what the schedule does to an image

**what people usually predict:** with 1,000 steps and a gently increasing schedule, the image fades out gradually and is gone near the end.

**what happens:**

| schedule | first step with $\bar\alpha < 0.5$ (SNR < 1) | first step with $\bar\alpha < 0.01$ | $\bar\alpha$ at the last step |
|---|---|---|---|
| linear | 259 | 673 | 4.04e-05 |
| cosine | 496 | 935 | 2.43e-09 |

on a real digit (MNIST training image 0, a "5"), the linear schedule leaves the 5 faint at step 300 and unrecognizable by step 500; the cosine schedule keeps it visible at step 500 and faint at 600, and it's gone by about 700. the last third of the linear chain (steps 673–999) works on images whose signal amplitude is below a tenth of the noise's. in log-SNR terms ($\log_{10}$): the linear schedule falls from 4.0 to 0 in the first 250 steps and then steadily to −4.4; the cosine schedule crosses 0 only at step 500, is still at −1.6 at step 900, and plunges to −8.6 in the last few steps.

**why:** $\log\bar\alpha_t \approx -\sum_{s \le t}\beta_s$, and a linear $\beta$ makes that sum quadratic in $t$: the signal decays slowly at first and then very fast. sampling cost is spread evenly over steps, but the information content isn't: steps at extreme low SNR do little (the network can only predict "noise" there), while the steps around SNR ≈ 1, where coarse structure is decided, are comparatively few. Nichol & Dhariwal's cosine schedule was designed to fix exactly this, and it improved likelihoods, especially on small images where the linear schedule destroys information fastest.

## experiment 2: how hard is it to guess the noise?

**what people usually predict:** a well-trained network predicts the noise best when there's little of it (small $t$) and worst at large $t$, where everything is noise.

**what happens** (MSE per coordinate at single steps, after 4,000 training steps; final training loss 0.170):

| step | 0 | 2 | 5 | 10 | 20 | 50 | 100 | 150 | 200 | 300 | 400 | 500 | 600 | 800 | 999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 0.811 | 0.665 | 0.590 | 0.540 | 0.522 | 0.506 | 0.569 | 0.542 | 0.469 | 0.246 | 0.109 | 0.042 | 0.015 | 0.002 | 0.002 |

exactly the reverse: the loss is highest at the smallest steps, sits on a **plateau near 0.5** from step 5 to step 200 (noise std from 0.03 to 0.6), and falls to almost zero once the noise dominates.

**why:** split the noise at a point near the circle into a component across the circle and one along it. once the noise is larger than the data's thickness (0.002), the component *across* is readable from $x_t$ (it's the distance from the circle), but the component *along* the circle is not: any nearby point on the circle could have been $x_0$. one of two coordinates is unpredictable, so the MSE per coordinate is about 1/2. at step 0 the noise (std 0.01) is only five times the thickness, so even the radial part is partly hidden and the loss is higher (0.81). in general the plateau sits near $d/D$ for data of intrinsic dimension $d$ in $D$ dimensions, because the unpredictable components are the $d$ directions *along* the manifold (here $1/2$; for a surface in 3-D it would be $2/3$). at large $t$, $x_t$ is almost pure noise and "predict the input" is nearly perfect, so the loss goes to zero, but those steps aren't *easy for sampling*: they're where the network decides nothing, because the score there barely depends on the data (the bump at step 100 is training noise from one seed and one evaluation batch, not a feature).

## experiment 3: how many steps?

**what people usually predict:** a 200× cheaper sampler (T = 5) should be clearly worse, but not dramatically so, since each model was trained for its own T.

**what happens** (cosine schedule, same training budget, 5,000 samples):

| T | network calls | modes covered | within 3 std | samples per mode | training | sampling 5,000 |
|---|---|---|---|---|---|---|
| 5 | 5 | 8 | 0.692 | 268 – 591 | 14 s | 0.10 s |
| 20 | 20 | 8 | 0.875 | 347 – 740 | 14 s | 0.32 s |
| 100 | 100 | 8 | 0.888 | 457 – 655 | 14 s | 2.2 s |
| 1000 | 1000 | 8 | 0.965 | 461 – 815 | 18 s | 24 s |

every model finds all 8 modes, but with $T = 5$ almost a third of the samples sit between modes, and the plot shows where: on the straight segments joining neighbouring modes, so the samples trace an octagon. with 20 steps the bridges thin to faint arcs and most of the quality is back; 1,000 steps are best, at 240 times the cost of 5. the per-mode counts are uneven for every $T$ (one seed; the counts' spread is mostly model error, not sampling noise, which is about ±25 per mode).

**why:** each reverse step models $q(x_{t-1} \mid x_t)$ as a Gaussian with a fixed variance. that's accurate when a step changes the noise level only slightly (the true reverse transition is then nearly Gaussian), but when one step must go from heavy noise halfway to the data, the true transition is multimodal: from a point between two modes, the clean point could be in either. a single Gaussian step averages them and lands between the modes. more steps make each transition closer to Gaussian; beyond a few hundred, the error is dominated by the network rather than by the discretization. with a fixed budget of about 20 calls, you'd rather keep the 1,000-step model and sample it with a deterministic solver that takes large steps well (topic 04).

## references

- Ho, Jain & Abbeel (2020), "Denoising diffusion probabilistic models", *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021), "Improved denoising diffusion probabilistic models", *ICML*. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Pope, Zhu, Abdelkader, Goldblum & Goldstein (2021), "The intrinsic dimension of images and its impact on learning", *ICLR*. [arXiv:2104.08894](https://arxiv.org/abs/2104.08894)
