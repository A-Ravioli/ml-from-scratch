# 28 · interpretability, robustness & alignment

> understand what models compute, where they fail, and how to make them safe.

by now you can train models that work. this chapter is about the questions that come after "does it work?": *why* does it work, *when* does it stop working, and does it do what we actually want? you'll ask a classifier which pixels it used and then test whether its answer means anything; open a transformer, find the two attention heads that let it copy in context, and prove they're the cause by patching activations; watch a toy model pack twenty features into five dimensions and recover them with a sparse autoencoder; break an MNIST model with perturbations you can barely see and harden it, at a price; make a classifier's confidence honest and wrap it in prediction sets with guaranteed coverage; train with differential privacy and attack the result to see what leaks; measure fairness and run into the theorem that you can't have it all; and watch optimisers exploit every flaw in a proxy reward. the capstone puts the interpretability tools to their hardest test: a transformer learns modular addition, generalises thousands of steps after memorising, and you read the algorithm it found, Fourier features and trigonometric identities, out of its weights.

**you will build:** saliency and integrated gradients, control tasks for probes, the model-randomisation sanity check; activation caching and patching with hooks, patching sweeps over layers and positions, attention-head scores, head ablation; the toy model of superposition, feature dimensionality, a sparse autoencoder and its evaluation (MMCS, L0, FVU); FGSM, PGD in $\ell_\infty$ and $\ell_2$, adversarial training, natural corruptions; temperature scaling, split conformal prediction for classification and regression, the ensemble uncertainty decomposition and OOD AUROC; per-sample gradients with `torch.func`, DP-SGD, a Rényi-DP accountant, a membership-inference attack scored at low false-positive rates, group-fairness metrics and equal-opportunity thresholds; KL-regularised and best-of-$n$ optimisation in closed form, a specification-gaming MDP, and a behavioural eval with intervals. topics 01–06 make up `mlfs.safety`; topic 07 is a workbench of its own.
**prerequisites:** chapter 27 (the RLHF objective, best-of-$n$, the eval harness and its intervals, your `Llama`), chapter 18 (your `GPT` and attention modules with `save_weights`), chapter 25 (`linear_probe`), chapter 23 (MDPs and value iteration), chapter 16 (`LeNet`), chapter 15 (`MLP`, `fit`), chapter 07 (calibration and ROC metrics), chapter 05 (entropy, KL, the bootstrap), chapter 04 (Gaussians and order statistics).
**time · scaffolding:** L4 · ~95 h. the lessons' **spec** boxes and the checks are the contract; internal structure is yours.
**newly unlocked library components:** Captum's `Saliency` and `IntegratedGradients` (after 01), TransformerLens-style hook libraries (after 02), SAELens-style SAE libraries (after 03), torchattacks / ART-style attack libraries (after 04), MAPIE-style conformal libraries (after 05), Opacus and Fairlearn (after 06). (build-before-you-use, see AUTHORING.md §9)

**data.** MNIST and CIFAR-10 from the course cache for the labs of topics 01 and 04–06; everything else is synthetic on purpose, because a known ground truth (planted features, a known true reward, a known algorithm) is what makes interpretability and alignment claims checkable: repeated random tokens (02), sparse features (03), a synthetic lending dataset (06, no real demographic data is used anywhere), proxy rewards and corridors (07), and modular addition (capstone). **hardware.** every lab runs on a laptop CPU in under a minute; the capstone's full configuration took about 12 minutes on an Apple-silicon GPU through `mps`, and its small configuration (which also groks) runs anywhere in about half a minute.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build |
|---|---|---|
| 01 | [attribution & probing](01-attribution-and-probing/) | saliency, integrated gradients (trapezoid rule), control tasks and probe selectivity, cascading-randomisation sanity checks |
| 02 | [mechanistic interpretability](02-mechanistic-interpretability/) | `run_with_cache`, `patch_activation`, activation and position patching, attention patterns and induction / previous-token scores, head ablation |
| 03 | [superposition & dictionary learning](03-superposition-and-dictionary-learning/) | sparse feature data, the toy model of superposition, feature dimensionality, a sparse autoencoder, MMCS / L0 / FVU |
| 04 | [adversarial robustness](04-adversarial-robustness/) | FGSM, PGD ($\ell_\infty$, $\ell_2$, random start), adversarial training, robust accuracy, corruptions |
| 05 | [uncertainty & OOD](05-uncertainty-and-ood/) | temperature scaling, the conformal quantile, conformal sets and intervals, ensemble uncertainty decomposition, OOD AUROC |
| 06 | [privacy & fairness](06-privacy-and-fairness/) | per-sample gradients, DP-SGD, an RDP accountant, loss-threshold membership inference and TPR at low FPR, demographic parity, equalized odds, equal-opportunity thresholds |
| 07 | [alignment](07-alignment/) | KL-regularised and best-of-$n$ optimisation in closed form, a specification-gaming corridor MDP, a behavioural eval with paired intervals |

## capstone

*reverse-engineer a circuit*: train a one-layer transformer (your GPT) on addition modulo a prime, watch it memorise and then grok, find the handful of Fourier frequencies its embedding uses, show that its logits are a sum of $\cos(w_k(a + b - c))$ terms and its neurons compute products of rotations, and track the circuit forming long before the test loss moves. see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it.

## notation

| symbol | meaning |
|---|---|
| $f_c(x)$ | the logit of class $c$ for input $x$ |
| $x'$ | the baseline of integrated gradients |
| $\mathrm{IG}_i(x)$ | integrated-gradients attribution of feature $i$ |
| $n$ (IG) | number of trapezoid intervals (`n_steps`) |
| $W_Q, W_K, W_V, W_O$ | an attention head's query, key, value and output maps (QK and OV circuits) |
| $P$ (topic 02) | the period of a repeated sequence |
| $S$, $I_i$ | feature sparsity, feature importance (toy model) |
| $W \in \mathbb{R}^{m \times n}$, $W_i$ | the toy model's embedding, feature $i$'s direction |
| $D_i$ | feature dimensionality |
| $f$, $D$, $\lambda$ | SAE features, dictionary (unit-norm columns), L1 coefficient |
| $\epsilon$ (topic 04), $\alpha$ (PGD) | perturbation budget, PGD step size |
| $T$, $\beta = 1/T$ | temperature, inverse temperature |
| $\alpha$ (topic 05), $\hat q$ | miscoverage level, conformal quantile |
| $C$, $\sigma$, $q$ | DP-SGD clip norm, noise multiplier, sampling rate |
| $(\epsilon, \delta)$, $\epsilon(\alpha)$ | differential privacy parameters, Rényi DP at order $\alpha$ |
| $g$, $\hat y$ | group, decision |
| $\pi_\text{ref}$, $\pi_\beta$, $\beta$ (topic 07) | reference policy, KL-regularised policy, KL coefficient |
| $r$, $u$ | proxy reward, true value |
| $B$, $G$, $\gamma$ | corridor bonus, goal reward, discount |
| $p$, $w_k = 2\pi k/p$ | the modulus, the $k$-th Fourier frequency (capstone) |

## how this connects

- builds on: your GPT and attention internals (ch 18), Llama and the RLHF / DPO / eval machinery (ch 27), linear probes (ch 25), LeNet (ch 16), training loops and `MLP` (ch 15), MDPs and value iteration (ch 23), calibration and ROC metrics (ch 07), information theory and the bootstrap (ch 05), projected gradient methods (ch 06).
- used later in: chapter 29's federated-learning elective applies the clip-and-noise recipe to whole client updates; chapter 30's thesis projects can start from `mlfs.safety` (attributions, patching, SAEs, attacks, conformal sets, DP-SGD and its audit) and from topic 07's exact optimisation experiments.

## core references

- Sundararajan, Taly & Yan (2017), "Axiomatic attribution for deep networks", *ICML*. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- Elhage et al. (2021), "A mathematical framework for transformer circuits", Anthropic. [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)
- Elhage et al. (2022), "Toy models of superposition", Anthropic. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Madry et al. (2018), "Towards deep learning models resistant to adversarial attacks", *ICLR*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
- Angelopoulos & Bates (2021), "A gentle introduction to conformal prediction and distribution-free uncertainty quantification". [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- Abadi et al. (2016), "Deep learning with differential privacy", *CCS*. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- Gao, Schulman & Hilton (2023), "Scaling laws for reward model overoptimization", *ICML*. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability", *ICLR*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
