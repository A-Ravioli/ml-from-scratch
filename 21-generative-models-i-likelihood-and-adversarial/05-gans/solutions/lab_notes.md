# GANs · lab notes

numbers measured with the reference solution on an Apple M2 laptop while other jobs were running (load average 15–20): the toy GANs on the CPU, the DCGAN on the GPU (MPS); the lab took 176 s wall clock under load (about 29 s per toy run) and 119 s at a load of about 5. the DCGAN result reproduced exactly on a second run. the toy experiments use `mlfs.datasets.eight_gaussians`, which is synthetic either way, so experiments 1–2 give the same numbers offline when run for the same number of steps (offline the lab uses 1,500 instead of 3,000 steps); experiment 3 needs real MNIST.

## experiment 1: chasing modes

**what people usually predict:** the number of covered modes rises steadily and ends at 8, perhaps after some early wobbling.

**what happens:** modes covered (of 8) every 500 steps, with the final high-quality fraction:

| seed | modes covered, steps 500 … 3,000 | final high-quality fraction |
|---|---|---|
| 0 | 0 0 3 **8** 3 **5** | 0.12 |
| 1 | 0 0 1 2 1 **2** | 0.55 |
| 2 | 0 1 4 6 6 **6** | 0.28 |

no seed ends with all 8 modes. seed 0 reaches all 8 at step 2,000 and loses five of them 500 steps later. seed 1 settles on 2 modes, and has the *highest* sample quality of the three. its snapshots show the other face of the problem: at step 1,500 the samples form a ring through all 8 modes (coverage but low quality), later they condense onto some modes and abandon others. (an earlier version of this lab ran 5,000 steps with the same seeds: seed 0 went on to 7, 7, 8 and 5 modes, seeds 1 and 2 stayed at 2 and 6, with final quality 0.41, 0.89 and 0.72: the collapse persists with more training while quality keeps improving.)

**why:** each generator step moves samples towards wherever the *current* discriminator scores them as real. once the generator over-produces some modes, the discriminator learns that those regions are mostly fake and lowers its scores there; the generator's samples then flow towards modes the discriminator currently likes, and the cycle repeats. nothing in the per-step objective rewards *spreading* the samples over all modes; that would only follow if the discriminator were optimal at every step, which alternating gradient steps never achieve (on paper 4 shows the rotational dynamics in a two-parameter game). a collapsed generator with high precision is a stable outcome for a while (seed 1), because a discriminator that can't tell the covered modes from real data gives no gradient towards the others. practical lesson: a GAN's last checkpoint can be its worst, and one seed says little.

## experiment 2: the saturating loss

**what people usually predict:** the saturating generator barely learns, so it ends worse than the non-saturating one.

**what happens:** after 50 discriminator steps against the untrained generator, $D(G(z))$ averages 0.012, and the generator's gradient norm is **2.33 with the non-saturating loss and 0.034 with the saturating loss**, 70 times smaller, exactly the saturation the lesson derives. but over the full run:

| generator loss | final modes covered (3 seeds) | final high-quality fraction |
|---|---|---|
| non-saturating | 5, 2, 6 | 0.12, 0.55, 0.28 |
| saturating | **8, 8, 7** | 0.30, 0.26, 0.44 |

the saturating GAN learns anyway (its gradients are small but not zero, and Adam normalizes their scale), and in this toy it ends **covering 7–8 modes on every seed**, while the non-saturating one collapses, at a similar sample quality. (with 5,000 steps in the earlier version: 8, 8, 8 modes for the saturating loss, 5, 2, 6 for the non-saturating one.)

**why:** the two losses send the same direction of gradient to every sample but weight samples differently. per sample, the non-saturating gradient on the logit is $-(1 - D)$, largest for samples the discriminator *rejects*; the saturating gradient is $-D$, largest for samples it already *accepts*. so the non-saturating loss throws its whole weight behind moving the rejected samples, all towards the discriminator's current favourite regions, which is exactly the herd behaviour that drives mode hopping; the saturating loss mostly refines samples that are already plausible and moves the others gently. this explanation is a hypothesis, but one control rules out the simplest alternative ("it's just a smaller step"): the non-saturating GAN with the generator's learning rate halved collapsed *more* (2, 3 and 2 modes for the three seeds, 5,000 steps). with only three seeds per setting and one toy problem, don't over-generalize: the non-saturating loss remains the default because it avoids the stall at the start of training on real data (Fedus et al., 2018, found both losses can work in practice, with outcomes depending on the setting). what this does show is that "the gradient vanishes" is a statement about the start of training, not a prediction of the final result.

## experiment 3: a DCGAN on MNIST

**what people usually predict:** all ten digits roughly equally, or a strong collapse onto one or two easy digits like 1.

**what happens:** after 2 epochs on 20,000 digits (624 steps, 18 s on the GPU), the samples are digit-like with many broken strokes. FID of 2,000 samples against 2,000 test digits: **171** (two runs of the final lab agreed exactly; an intermediate run that scored the same kind of samples with a separately trained, CPU-cached feature network gave 154: a reminder that FID is only comparable under one fixed feature network). the classifier's labels cover all ten classes (entropy 2.16 nats of a possible 2.30) but unevenly: 3s, 5s, 7s and 9s each make up about 15%, while 1s (1.3%), 4s and 6s (about 4% each) are rare, although 1s are the most common digit in MNIST (11%). a longer run in an earlier version (3 epochs on the CPU, 938 steps) reached an FID of 95 with 4.4% 1s.

**why:** this is partial mode collapse, the image-space version of experiment 1: the generator over-produces some modes and neglects others. 1s are an interesting casualty: a thin vertical stroke is easy to draw but also easy for the discriminator to check, so the generator gains little by producing them. the classifier check has a blind spot, though: it must assign every sample to *some* class, so a malformed blob still counts as a digit, which inflates the apparent diversity of a bad generator. the FIDs here use $n = 2{,}000$ against 2,000 test digits, while topic 01's lab compared its flawed "models" against all 10,000 test digits at $n$ from 100 to 3,000, so the numbers are only roughly comparable (topic 01: at $n = 1{,}000$ against 10,000 test digits, the noisy model scored 22 and the model without 7, 8 and 9 scored 102). even allowing for that, this short-trained generator is worse than both flawed models in feature space: its problem is mostly sample quality, plus imbalance, not wholesale missing classes.

## references

- Fedus, Rosca, Lakshminarayanan, Dai, Mohamed & Goodfellow (2018), "Many paths to equilibrium: GANs do not need to decrease a divergence at every step", *ICLR*. [arXiv:1710.08446](https://arxiv.org/abs/1710.08446)
- Metz, Poole, Pfau & Sohl-Dickstein (2017), "Unrolled generative adversarial networks", *ICLR*. [arXiv:1611.02163](https://arxiv.org/abs/1611.02163)
