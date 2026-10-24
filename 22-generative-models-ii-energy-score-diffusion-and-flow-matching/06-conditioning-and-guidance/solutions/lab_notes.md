# conditioning & guidance · lab notes

numbers measured with the reference solution on an Apple M2 laptop CPU (one thread, as in CI) on a shared machine; the whole lab took about 110–170 s. the data are synthetic, so offline runs are the same experiments. one training seed per configuration; with 2,000 samples per class, sampling noise is about ±0.01 on the accuracies and stds and about ±0.003 on the Fréchet distances near $w = 1$.

## experiment 1: the guidance scale

**what people usually predict:** stronger guidance makes samples "better" across the board, maybe with a bit less variety.

**what happens** (averages over the 3 classes):

| samples | Bayes accuracy | posterior of the intended class | std | distance of the class mean from 0 | Fréchet distance to the true class |
|---|---|---|---|---|---|
| real data | 0.935 | 0.898 | 0.498 | 1.010 | 0.0008 |
| w = 0 | 0.327 | 0.326 | 0.840 | 0.098 | 1.2685 |
| w = 1 | 0.930 | 0.894 | 0.492 | 0.992 | **0.0031** |
| w = 2 | 1.000 | 0.996 | 0.385 | 1.423 | 0.2195 |
| w = 4 | 1.000 | 1.000 | 0.374 | 1.898 | 0.9592 |
| w = 8 | 1.000 | 1.000 | 0.413 | 2.584 | 3.3513 |

$w = 1$ reproduces the classes almost exactly (Fréchet distance 0.003, accuracy equal to the real data's 0.93). from $w = 2$ on, every sample is classified correctly, *better* than real data, while the Fréchet distance grows a hundredfold: the samples are 25% narrower than the classes and their centres drift outward, to 1.9 at $w = 4$ and 2.6 at $w = 8$ (the true centres are at distance 1). the scatter plot at $w = 8$ shows each class pushed far outside the data, the orange one stretched into a thin arc. $w = 0$ ignores the label: one blob, a third correct (chance).

**why:** the guided target weights each point by $p(y \mid x)^w$. for overlapping classes, $p(y \mid x)$ keeps increasing as you move *away* from the other classes, beyond the class centre, so raising it to a power moves mass outward (the overshoot of on paper 2) and away from the overlap regions (higher accuracy, smaller spread). the classifier's view (accuracy, the posterior) keeps improving with $w$, which is what a person asked "is this a clear example of class $k$?" would also reward; the distribution's view (Fréchet distance, like FID) is best at $w = 1$, because only there is the target the class distribution itself. that is the fidelity–diversity tradeoff in one table. (the std's slight rise at $w = 8$ comes from the stretched arcs, not from more diversity.)

## experiment 2: how often to drop the label

**what people usually predict:** $p_\text{uncond}$ is a training detail with little effect.

**what happens:**

| p_uncond | w | Bayes accuracy | std | distance of the class mean from 0 | Fréchet distance |
|---|---|---|---|---|---|
| 0.0 | 1 | 0.920 | 0.508 | 0.990 | 0.0035 |
| 0.0 | 4 | 1.000 | 0.813 | **5.364** | **19.5** |
| 0.2 | 1 | 0.930 | 0.492 | 0.992 | 0.0031 |
| 0.2 | 4 | 1.000 | 0.374 | 1.898 | 0.959 |
| 0.5 | 1 | 0.935 | 0.486 | 1.012 | 0.0045 |
| 0.5 | 4 | 1.000 | 0.320 | 1.844 | 0.811 |

without guidance, all three are the same model to within noise. with $w = 4$, the model that never saw the null label sends its samples five units from the origin, twenty times the Fréchet distance of the others. dropping half the labels costs nothing visible at $w = 1$ here and gives slightly sharper guided samples.

**why:** at $w = 1$ the null label is never queried, so $p_\text{uncond}$ only matters through the data it takes from the conditional task (here, 20,000 points for a simple problem, so even half is plenty). with $p_\text{uncond} = 0$ the null row of the embedding table keeps its random initialization, the "unconditional" prediction is an arbitrary field, and guidance extrapolates away from garbage: $(1 - w)$ times an arbitrary vector field is added to every step. the usual 0.1–0.2 trains a good enough unconditional model while keeping most of the data for the conditional one; on hard, data-hungry problems a large $p_\text{uncond}$ does cost conditional quality.

## experiment 3: classifier guidance against classifier-free guidance

**what people usually predict:** classifier guidance, with its dedicated classifier, gives sharper class separation; classifier-free gives better distributions.

**what happens** (noise prediction, 200-step linear schedule, 50 DDIM steps):

| guidance | w | Bayes accuracy | std | distance of the class mean from 0 | Fréchet distance |
|---|---|---|---|---|---|
| classifier guidance | 1 | 0.932 | 0.516 | 0.941 | 0.0127 |
| classifier-free | 1 | 0.886 | 0.524 | 0.945 | 0.0154 |
| classifier guidance | 4 | 1.000 | 0.419 | 1.880 | 0.793 |
| classifier-free | 4 | 1.000 | 0.421 | 1.778 | 0.770 |

on this toy the two are nearly interchangeable: at $w = 1$ the unconditional model plus the noisy classifier reproduces the classes about as well as the conditional model (accuracy 0.93 vs 0.89, Fréchet distance 0.013 vs 0.015, both a little worse than the flow of experiment 1), and at $w = 4$ both overshoot by about the same amount.

**why:** with an exact classifier, $w = 1$ classifier guidance *is* the conditional model (on paper 1, and the stage 3 check), so the gap at $w = 1$ measures how well each learned piece approximates its target; a 2-D, 3-class posterior is easy to learn, so the classifier is good. at $w > 1$ each method amplifies different errors: classifier guidance amplifies the classifier's gradient errors (and, in high dimension, its adversarial directions: gradients that raise $p_\phi(y \mid x)$ without making $x$ look like $y$), classifier-free guidance amplifies the difference between two outputs of one network, whose errors partly cancel. in high dimension that, plus not needing a second network trained on noisy images, is why classifier-free guidance won. a classifier trained only on clean data would see the noisy $x_t$ as out-of-distribution, and its gradients at high noise would be meaningless. (an earlier version of this experiment used the cosine schedule with DDIM and every sample exploded: the cosine schedule's last $\bar\alpha$ is about $10^{-9}$, and DDIM divides by $\sqrt{\bar\alpha}$; see topic 04's common bugs.)

## references

- Ho & Salimans (2021), "Classifier-free diffusion guidance". [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Dhariwal & Nichol (2021), "Diffusion models beat GANs on image synthesis", *NeurIPS*. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
