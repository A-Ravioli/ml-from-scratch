# 22 · generative models II: energy, score, diffusion & flow matching

> the modern generative stack.

chapter 21's models each bought their likelihood or their one-pass sampling with an architectural constraint. this chapter's models drop the constraints and learn one thing instead: the direction towards more probable data, the **score** $\nabla_x\log p(x)$, or its close relative, a velocity that carries noise to data. you'll start with energy-based models, which need only the gradient of a network to sample (Langevin dynamics) and a tug of war to train (contrastive divergence); learn the score directly with denoising and sliced score matching, and sample it with annealed Langevin dynamics; turn that into DDPM, the noise-prediction model behind most image generators; reinterpret DDPM as a continuous-time SDE with a deterministic twin, the probability-flow ODE, that your chapter 20 solver integrates in 20 steps instead of 1,000; replace the noising process with straight paths and plain regression (flow matching) and straighten them further (rectified flow); make conditional models obey their labels with classifier-free guidance; and finally move the whole thing into the latent space of your chapter 21 autoencoder with a diffusion transformer. along the way you'll find that Langevin samples of $\mathcal{N}(0, 1)$ have variance 40 at a step size of 3.9, that a perfect score still puts half of the samples in the mode that should get a fifth, that a diffusion model's loss is *highest* where the noise is smallest, that a perfect network sampled with 10 DDIM steps shrinks a Gaussian by 28%, that one Euler step of a flow-matching model sends every sample to the data mean, and that more guidance keeps improving class accuracy while making the distribution worse.

**you will build:** `score_from_energy`, `langevin_dynamics`, `mala`, `EnergyMLP`, `ebm_loss`, `train_ebm` (topic 01); `dsm_loss`, `sliced_score_matching_loss`, `geometric_sigmas`, `NoiseConditionalScore`, `fit_dsm`, `annealed_langevin` (02); `DiffusionSchedule`, `sinusoidal_features`, `TimeEmbedding`, `TimeMLP`, `ddpm_loss`, `fit_ddpm`, `ddpm_sample`, `TimeUNet` (03); `alpha_bar_at`, `beta_at`, `eps_to_score`, `pf_ode_drift`, `ddim_timesteps`, `ddim_sample`, `probability_flow_sample`, `reverse_sde_sample` (04); `flow_matching_loss`, `sample_flow`, `fit_flow_matching`, `make_reflow_pairs`, `straightness` (05); `drop_labels`, `cfg_model`, `classifier_guided_eps`, `fit_conditional` (06); `patchify`, `unpatchify`, `pos_embed_2d`, `DiTBlock`, `DiT`, `encode_latents`, `decode_latents` (07). together they make up **`mlfs.diffusion`**. the optimal-transport elective builds minibatch-OT flow matching on your `flow_matching_loss(x0=...)`, `sample_flow` and `TimeMLP`; the R2 milestone reproduces a paper with them.

**prerequisites:** chapter 21 (FID with your `FeatureNet`, `ConvVAE`, `mode_coverage`, flows and likelihood bookkeeping), chapter 20 · topic 06 (`odeint`: Euler, midpoint, RK4), chapter 18 (attention, `PatchEmbedding`, sinusoidal positions), chapter 16 (residual blocks and the U-Net), chapter 15 (training loops, `ema_update`, `get_device`), chapter 12 · topics 05–07 (MCMC, HMC, the ELBO), chapter 5 · topic 06 (exponential families) and chapter 4 · topics 04 and 07 (Gaussians, Markov chains, Brownian motion).

**time · scaffolding:** about 100 hours · L4 (spec-driven): each stage's exact specification is in its lesson's **spec** box and the checks enforce it; the exercise notebooks give you the signatures and one-line docstrings, and the internal structure is yours (topic 03 hands you two U-Net building blocks). one hint per stage. the capstone gives you a brief, the configurations and the checks; the code and the experiment are yours. image experiments benefit from a GPU (an Apple M-series `mps` device is enough); every lab and the capstone's small config run on a CPU.

**newly unlocked library components:** from chapter 23 on (and inside this chapter from the topic after you build them): diffusion libraries such as `diffusers` (schedulers like `DDPMScheduler` and `DDIMScheduler`, `UNet2DModel`, `DiTTransformer2DModel`) and `torchdiffeq`-style ODE samplers (03–04); flow-matching libraries such as `torchcfm` and `flow_matching` (05); classifier-free guidance pipelines (06); `torch.func`-based Jacobian-trace estimators for score matching (02). inside `mlfs`, later chapters use your code through `mlfs.diffusion` per [`CONTRACTS.md`](../CONTRACTS.md).

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [energy-based models](01-energy-based-models/) | `score_from_energy`, `langevin_dynamics`, `mala`, `EnergyMLP`, `ebm_loss`, `train_ebm` → `mlfs.diffusion` | Langevin "samples" of $\mathcal{N}(0, 1)$ have variance 2 at step 2 and 40 at step 3.9, exactly as the formula says, while MALA stays at 1; running a trained EBM's sampler 100× longer than it was trained with sharpens every mode but leaves the mode weights unbalanced (up to 4:1 between modes for persistent training, 1.4–1.8:1 for short-run, over two seeds) |
| 02 | [score matching](02-score-matching/) | `dsm_loss`, `sliced_score_matching_loss`, `geometric_sigmas`, `NoiseConditionalScore`, `fit_dsm`, `annealed_langevin` → `mlfs.diffusion` | Langevin with the *exact* score puts 49% of the samples in a component of weight 0.2 (annealing: 26%); sliced score matching gets 4× closer to the clean score than DSM at a small noise level |
| 03 | [diffusion (DDPM)](03-diffusion-ddpm/) | `DiffusionSchedule`, `sinusoidal_features`, `TimeEmbedding`, `TimeMLP`, `ddpm_loss`, `fit_ddpm`, `ddpm_sample`, `TimeUNet` → `mlfs.diffusion` | the noise-prediction loss is highest at the smallest noise and sits on a plateau of 0.5 for data on a curve (half the coordinates are unpredictable); a 5-step DDPM draws an octagon between the 8 modes |
| 04 | [the SDE view & fast sampling](04-the-sde-view-and-fast-sampling/) | `alpha_bar_at`, `beta_at`, `eps_to_score`, `pf_ode_drift`, `ddim_timesteps`, `ddim_sample`, `probability_flow_sample`, `reverse_sde_sample` → `mlfs.diffusion` | with a perfect network, RK4 *loses* to Euler and DDIM at 20 calls; with a 10%-wrong network, deterministic samplers stay at 75% quality however many steps they take, while stochastic ones recover to 98% |
| 05 | [flow matching & rectified flow](05-flow-matching-and-rectified-flow/) | `flow_matching_loss`, `sample_flow`, `fit_flow_matching`, `make_reflow_pairs`, `straightness` → `mlfs.diffusion` | one Euler step of a flow-matching model lands every sample on the data mean; after one reflow, one step covers all 8 modes; noise-to-noise flow matching can't go below a loss of $\pi/2$ |
| 06 | [conditioning & guidance](06-conditioning-and-guidance/) | `drop_labels`, `cfg_model`, `classifier_guided_eps`, `fit_conditional` → `mlfs.diffusion` | guidance makes 100% of the samples correctly classified from $w = 2$ on while the Fréchet distance to the true classes grows 70× at $w = 2$ and about 1,000× at $w = 8$, with classes pushed far past their centres; a model never trained without labels sends guided samples 5 units away |
| 07 | [latent diffusion & DiT](07-latent-diffusion-and-dit/) | `patchify`, `unpatchify`, `pos_embed_2d`, `DiTBlock`, `DiT`, `encode_latents`, `decode_latents` → `mlfs.diffusion` | a VAE's latents look like small blurry digits, not noise; the latent DiT trains 3× faster per step than a pixel U-Net yet scores FID 192 against 60 at equal steps (800), nowhere near its reconstruction floor of 11; a smoother ($\beta = 0.1$) latent space makes both worse |

## capstone

*a class-conditional generator for CIFAR-10*: the same `TimeUNet` trained twice with the same budget, data order and label dropout, once as a DDPM noise predictor and once as a flow-matching velocity field, then sampled with classifier-free guidance at equal numbers of network calls, and scored with course-internal FID and class accuracy (chapter 21's `FeatureNet`, not Inception). the reference run (43 minutes on a shared Apple M2 GPU, 2,500 steps per objective) finds flow matching well ahead at 5 network calls (FID 16.1 vs 21.9), probably ahead at 10 (8.3 vs 10.2, one draw and one seed) and a tie at 25 (5.4–6.5 vs 5.4–5.8), a guidance sweet spot at $w = 2$ for both while class accuracy keeps rising with $w$, and that the comparison hinges on pipeline details that only matter for noise prediction: without clipping the predicted image DDPM's FID goes from 5.4 to 19.8, and an EMA without warmup made it 248. small CPU config: about a minute on the offline synthetic stand-in.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a diffusion step with its posterior and DDIM jump, flow matching on a Gaussian, guidance in closed form), and a 60-minute blind rebuild of DDPM.

## notation

| symbol | meaning |
|---|---|
| $E_\theta(x)$, $Z_\theta$ | energy; partition function, $p_\theta = e^{-E_\theta}/Z_\theta$ |
| $s(x)$, $s_\theta(x, \sigma)$ | the score $\nabla_x\log p(x)$; a (noise-conditional) score network |
| $h$, $\epsilon$ (topics 01–02) | Langevin step size; the annealed sampler's step at the smallest noise level |
| $\sigma$, $q_\sigma$ | a noise level; the data blurred by $\mathcal{N}(0, \sigma^2 I)$ |
| $T$, $t$ | number of diffusion steps; time ($t = \text{step}/T \in [0, 1)$ for diffusion, $t \in [0, 1]$ for flows) |
| $\beta_t$, $\alpha_t$, $\bar\alpha_t$ | noise schedule, $1 - \beta_t$, $\prod_{s \le t}\alpha_s$; $\bar\alpha(t)$, $\beta(t)$ in continuous time |
| $\epsilon$, $\epsilon_\theta(x_t, t, y)$ | standard normal noise; the noise predictor |
| $\tilde\mu_t$, $\tilde\beta_t$ | mean and variance of the forward posterior $q(x_{t-1} \mid x_t, x_0)$ |
| $\hat x_0$ | the predicted clean sample |
| $\eta$ | DDIM's stochasticity ($0$ = deterministic) |
| $x_0$, $x_1$, $v_\theta$ (topic 05) | flow matching: noise at $t = 0$, data at $t = 1$; the velocity network |
| $S$ | straightness of a flow |
| $y$, $\varnothing$, $K$ | class label; the null label (index $K$ for $K$ classes) |
| $w$, $p_\text{uncond}$ | guidance scale; label dropout probability |
| $p$, $N$, $d$ | patch size; number of tokens; model width (topic 07) |
| $\gamma, \beta, \alpha$ (topic 07) | adaLN scale, shift and gate |
| NFE | network function evaluations per sample |

## how this connects

- builds on: likelihood, FID and the autoencoders of ch 21; ODE solvers and neural ODEs (ch 20); transformers and ViTs (ch 18); U-Nets (ch 16); training practice and EMA (ch 15); MCMC, HMC and variational inference (ch 12); exponential families (ch 05); Gaussians, Markov chains and Brownian motion (ch 04).
- used later in: self-supervised and multimodal learning, where diffusion models are conditioned on text and images (ch 25); training at scale (ch 26); optimal transport (electives 15–17), whose topic 17 project straightens flow-matching paths with minibatch OT couplings on top of this chapter's code; audio generation (audio elective); the R2 research milestone ([`milestones/R2.md`](../milestones/R2.md)), which reproduces a generative modeling paper with these modules.

## core references

- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021), "Score-based generative modeling through stochastic differential equations", *ICLR*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Ho, Jain & Abbeel (2020), "Denoising diffusion probabilistic models", *NeurIPS*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow matching for generative modeling", *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Karras, Aittala, Aila & Laine (2022), "Elucidating the design space of diffusion-based generative models", *NeurIPS*. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Prince (2023), *Understanding Deep Learning*, MIT Press, chapter 18 (diffusion models). [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)
- Murphy (2023), *Probabilistic Machine Learning: Advanced Topics*, MIT Press, chapters 24–25 (energy-based and diffusion models). [probml.github.io/pml-book/book2](https://probml.github.io/pml-book/book2.html)
